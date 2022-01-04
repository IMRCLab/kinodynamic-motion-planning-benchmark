import numpy as np
import yaml
import argparse
import tempfile
from pathlib import Path
import subprocess
import shutil
import time

import sys
import os
sys.path.append(os.getcwd())
from motionplanningutils import CollisionChecker

from scp import SCP
import robots

def run_scp(filename_env, filename_initial_guess, filename_result='result_scp.yaml', iterations=5):

	with open(filename_env) as f:
		env = yaml.safe_load(f)

	robot_node = env["robots"][0]
	robot = robots.create_robot(robot_node["type"])

	x0 = np.array(robot_node["start"])
	xf = np.array(robot_node["goal"])

	cc = CollisionChecker()
	cc.load(filename_env)
	# r = cc.distance(np.array(robot_node["start"]))
	# print(r)
	# exit()

	if filename_initial_guess == "gen_random_rollout":
		# initialize with random rollout
		T = 100
		states = np.empty((T, len(robot.state_desc)))
		states[0] = x0
		actions = np.empty((T-1, len(robot.action_desc)))
		for k in range(T-1):
			actions[k] = np.random.uniform(robot.min_u, robot.max_u)
			states[k+1] = robot.step(states[k], actions[k])
	elif filename_initial_guess == "gen_straight":
		# initialize with linear interpolation x0 -> xf
		T = 80
		states = np.empty((T, len(robot.state_desc)))
		actions = np.zeros((T-1, len(robot.action_desc)))
		for dim in range(x0.shape[0]):
			states[:, dim] = np.interp(np.linspace(0, 1, T), [0, 1], [x0[dim], xf[dim]])

		# add some noise
		states[1:] += np.random.normal(0, 0.001, states.shape)[1:]
		actions += np.random.normal(0, 0.001, actions.shape)

		print(states)
		print(actions)
	else:

		with open(filename_initial_guess) as f:
			initial_guess = yaml.safe_load(f)

		states = np.array(initial_guess["result"][0]["states"])
		actions = np.array(initial_guess["result"][0]["actions"])

	eps = 0.1
	trust_x_est = np.max(np.abs(np.diff(states, axis=0)), axis=0) + eps
	trust_u_est = np.max(np.abs(np.diff(actions, axis=0)), axis=0) + eps
	print(trust_x_est, trust_u_est)
	# exit()

	# scp = SCP(robot)
	scp = SCP(robot, cc)
	print(xf)
	X, U, val = scp.min_u(states, actions, x0, xf, iterations, trust_x=2*trust_x_est, trust_u=2*trust_u_est, verbose=True)
	# X, U, val = scp.min_u(states, actions, x0, xf, 3, trust_x=None, trust_u=None, verbose=True)

	result = dict()
	result["result"] = [{'states': X[-1].tolist(), 'actions': U[-1].tolist()}]

	if len(X) == iterations + 1:
		with open(filename_result, 'w') as f:
			yaml.dump(result, f)
		return True
	return False

def run_scp_standalone(filename_env, folder, timelimit, cfg):

	with open(filename_env) as f:
		env = yaml.safe_load(f)

	robot_node = env["robots"][0]
	robot = robots.create_robot(robot_node["type"])

	x0 = np.array(robot_node["start"])
	xf = np.array(robot_node["goal"])

	cc = CollisionChecker()
	cc.load(filename_env)
	scp = SCP(robot, cc)
	# scp = SCP(robot, None)

	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)

		start = time.time()

		# compute initial guess via OMPL
		filename_initial_guess = "{}/result_ompl.yaml".format(folder)
		result = subprocess.run(["./main_ompl_geometric", 
			"-i", filename_env,
			"-o", filename_initial_guess,
			"--timelimit", str(1),
			"-p", "rrt*"
			])

		with open(filename_initial_guess) as f:
			guess = yaml.safe_load(f)

		states = np.array(guess['result'][0]['states'])
		length = guess['result'][0]['pathlength']
		max_speed = 0.5
		min_T = max(int(length / max_speed), 3)
		max_T = None

		# prepare stats
		filename_stats = "{}/stats.yaml".format(folder)
		filename_result = "{}/result_scp.yaml".format(folder)

		with open(filename_stats, 'w') as stats:
			stats.write("stats:\n")

			# modified binary search
			T = None
			best_T = None
			while time.time() - start < timelimit:
				if max_T is not None:
					if min_T >= max_T:
						break
					T = int((min_T + max_T) / 2)
				else:
					if T is None:
						T = min_T
					else:
						T = T * 2

				print("TRYING ", T, min_T, max_T)

				states_interp = np.empty((T, states.shape[1]))
				for k in range(states.shape[1]):
					states_interp[:,k] = np.interp(np.linspace(0,1,T), np.linspace(0, 1, states.shape[0]), states[:,k])

				# filename_modified_guess = p / "guess.yaml"
				# with open(filename_modified_guess, 'w') as f:
				# 	guess['result'][0]['states'] = states_interp.tolist()
				# 	guess['result'][0]['actions'] = np.zeros((T-1, 2)).tolist()
				# 	yaml.dump(guess, f)

				# Run SCP
				filename_temp_result = p / "T_{}.yaml".format(T)
				# success = run_scp(filename_env, filename_modified_guess, filename_temp_result)

				states_guess = states_interp
				# actions_guess = np.zeros((T-1, 2))
				actions_guess = np.random.normal(0, 0.01, (T-1, 2))
				iterations = 5
				trust_x = 0.1
				trust_u = 0.5
				X, U, val = scp.min_u(states_guess, actions_guess, x0, xf, iterations,
				                      trust_x=trust_x, trust_u=trust_u, verbose=True, soft_xf=True)
				max_error_to_goal = np.linalg.norm(X[-1][-1] - xf, np.inf)

				success = (len(X) == iterations + 1) and max_error_to_goal < 1e-3
				if not success:
					print("SCP failed with T", T, val, len(X))
					min_T = T

					# return False
				else:
					print("SCP SUCCESS with T", T)
					now = time.time()
					t = now - start
					stats.write("  - t: {}\n    cost: {}\n".format(t, T / 10))

					max_T = T - 1
					if best_T is None or T < best_T:
						best_T = T

					# write output to file
					result = dict()
					result["result"] = [{'states': X[-1].tolist(), 'actions': U[-1].tolist()}]
					with open(filename_temp_result, 'w') as f:
						yaml.dump(result, f)
			
		if best_T is not None:
			shutil.copyfile(p / "T_{}.yaml".format(best_T), filename_result)
			return True
		return False

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("initial_guess", help="file containing the initial_guess (e.g., from db-A*) (YAML)")
	args = parser.parse_args()

	run_scp(args.env, args.initial_guess)


if __name__ == '__main__':
	main()
