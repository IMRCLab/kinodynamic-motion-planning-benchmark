import numpy as np
import yaml
import argparse

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

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("initial_guess", help="file containing the initial_guess (e.g., from db-A*) (YAML)")
	args = parser.parse_args()

	run_scp(args.env, args.initial_guess)


if __name__ == '__main__':
	main()
