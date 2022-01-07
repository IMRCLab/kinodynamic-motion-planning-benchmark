import numpy as np
import yaml
import argparse
import tempfile
from pathlib import Path
import subprocess
import shutil
import time

import robots
import translate_g

def run_komo(filename_env, filename_initial_guess, filename_result):

	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)

		with open(filename_env) as f:
			env = yaml.safe_load(f)
		robot_type = env["robots"][0]["type"]
		if "first_order" in robot_type:
			order = 1
		elif "second_order" in robot_type:
			order = 2

		# convert environment YAML -> g
		filename_g = p / "env.g"
		translate_g.write(filename_env, str(filename_g))

		# Run KOMO
		result = subprocess.run(["./rai_dubins",
				"-model", "\""+str(filename_g)+"\"",
				"-waypoints", "\""+str(filename_initial_guess)+"\"",
				"-one_every", "1",
				"-display", str(0),
				"-animate", str(0),
				"-order", str(order),
				"-out", "\""+str(filename_result)+"\""])
		if result.returncode != 0:
			print("KOMO failed")
			return False
		else:
			return True


def run_komo_standalone(filename_env, folder, timelimit, cfg):

	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)

		start = time.time()

		# convert environment YAML -> g
		filename_g = p / "env.g"
		translate_g.write(filename_env, str(filename_g))

		# compute initial guess via OMPL
		filename_initial_guess = "{}/result_ompl.yaml".format(folder)
		result = subprocess.run(["./main_ompl_geometric", 
			"-i", filename_env,
			"-o", filename_initial_guess,
			"--timelimit", str(10),
			"-p", "rrt*"
			])

		with open(filename_initial_guess) as f:
			guess = yaml.safe_load(f)

		states = np.array(guess['result'][0]['states'])
		length = guess['result'][0]['pathlength']
		max_speed = 0.5
		min_T = int(length / max_speed)
		max_T = None

		# prepare stats
		filename_stats = "{}/stats.yaml".format(folder)
		filename_result = "{}/result_komo.yaml".format(folder)

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

				filename_modified_guess = p / "guess.yaml"
				with open(filename_modified_guess, 'w') as f:
					guess['result'][0]['states'] = states_interp.tolist()
					yaml.dump(guess, f)

				# Run KOMO
				filename_temp_result = p / "T_{}.yaml".format(T)
				while True:
					result = subprocess.run(["./rai_dubins",
									"-model", "\""+str(filename_g)+"\"",
									"-waypoints", "\""+str(filename_modified_guess)+"\"",
									"-one_every", "1",
									"-display", str(0),
									"-animate", str(0),
									"-out", "\""+str(filename_temp_result)+"\""],
									stdout=subprocess.DEVNULL)
					# a negative returncode indicates an internal error -> repeat
					if result.returncode >= 0:
						break
				if result.returncode != 0:
					print("KOMO failed with T", T, result.returncode)
					min_T = T

					# return False
				else:
					print("KOMO SUCCESS with T", T)
					now = time.time()
					t = now - start
					stats.write("  - t: {}\n    cost: {}\n".format(t, T / 10))

					max_T = T - 1
					if best_T is None or T < best_T:
						best_T = T
					# return True
			
		if best_T is not None:
			shutil.copyfile(p / "T_{}.yaml".format(best_T), filename_result)
			return True
		return False
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("initial_guess", help="file containing the initial_guess (e.g., from db-A*) (YAML)")
	parser.add_argument("result", help="file containing the optimization result (YAML)")
	args = parser.parse_args()

	run_komo(args.env, args.initial_guess, args.result)


if __name__ == '__main__':
	main()
