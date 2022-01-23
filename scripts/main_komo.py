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
from utils_optimization import UtilsSolutionFile

def run_komo(filename_env, filename_initial_guess, filename_result, cfg = ""):

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

		# write config file
		filename_cfg = p / "rai.cfg"
		with open(filename_cfg, 'w') as f:
			f.write(cfg)

		# hack
		utils_sol_file = UtilsSolutionFile()
		utils_sol_file.load(filename_initial_guess)
		filename_modified_guess = p / "guess.yaml"

		for factor in [0.9, 1.0, 1.1]:
			T = int(utils_sol_file.T() * factor)
			print("Trying T ", T)
			# utils_sol_file.save_rescaled(filename_modified_guess, int(utils_sol_file.T() * 1.1))
			utils_sol_file.save_rescaled(filename_modified_guess, T)

			while True:
				# Run KOMO
				result = subprocess.run(["./main_rai",
						"-model", "\""+str(filename_g)+"\"",
						# "-waypoints", "\""+str(filename_initial_guess)+"\"",
						"-waypoints", "\""+str(filename_modified_guess)+"\"",
						"-one_every", "1",
						"-display", str(0),
						"-animate", str(0),
						"-order", str(order),
						"-cfg", "\""+str(filename_cfg)+"\"",
						"-out", "\""+str(filename_result)+"\""])
				# a negative returncode indicates an internal error -> repeat
				if result.returncode >= 0:
					break
			if result.returncode != 0:
				print("KOMO failed")
				# return False
			else:
				return True


def run_komo_standalone(filename_env, folder, timelimit, cfg = "",
		search = "binarySearch",
		initialguess = "ompl"):

	# search = "linear"
	# search = "binarySearch"

	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)

		start = time.time()

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

		# write config file
		filename_cfg = p / "rai.cfg"
		with open(filename_cfg, 'w') as f:
			f.write(cfg)

		# compute initial guess via OMPL
		filename_initial_guess = "{}/result_ompl.yaml".format(folder)
		result = subprocess.run(["./main_ompl_geometric", 
			"-i", filename_env,
			"-o", filename_initial_guess,
			"--timelimit", str(10),
			"-p", "rrt*"
			])

		utils_sol_file = UtilsSolutionFile()
		utils_sol_file.load(filename_initial_guess)

		length = utils_sol_file.file['result'][0]['pathlength']
		max_speed = 0.5
		min_T = int(length / max_speed)
		max_T = None

		# prepare stats
		filename_stats = "{}/stats.yaml".format(folder)
		filename_result = "{}/result_komo.yaml".format(folder)

		with open(filename_stats, 'w') as stats:
			stats.write("stats:\n")

			best_T = None
			if search == "linear":
				T = min_T - 1
			else:
				# modified binary search
				T = None
			while time.time() - start < timelimit:
				if search == "linear":
					T = T + 1
					print("TRYING ", T)
				else:
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

				filename_modified_guess = p / "guess.yaml"
				utils_sol_file.save_rescaled(filename_modified_guess, T)

				# Run KOMO
				filename_temp_result = p / "T_{}.yaml".format(T)
				while True:
					result = subprocess.run(["./main_rai",
									"-model", "\""+str(filename_g)+"\"",
									"-waypoints", "\""+str(filename_modified_guess)+"\"",
									"-one_every", "1",
									"-display", str(0),
									"-animate", str(0),
									"-order", str(order),
									"-cfg", "\"" + str(filename_cfg)+"\"",
									"-out", "\""+str(filename_temp_result)+"\""])#,
									# stdout=subprocess.DEVNULL)
					# a negative returncode indicates an internal error -> repeat
					if result.returncode >= 0:
						break
				if result.returncode != 0:
					print("KOMO failed with T", T, result.returncode)
					min_T = T + 1

					# return False
				else:
					print("KOMO SUCCESS with T", T)
					now = time.time()
					t = now - start
					stats.write("  - t: {}\n    cost: {}\n".format(t, T / 10))

					if search == "linear":
						best_T = T
						break
					else:
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
