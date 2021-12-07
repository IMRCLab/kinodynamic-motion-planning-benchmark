import numpy as np
from scp import SCP
import robots
import yaml
import argparse
import subprocess

import sys
import os
sys.path.append(os.getcwd())

import main_scp
import gen_motion_primitive
from motionplanningutils import RobotHelper

# ./dbastar -i ../benchmark/dubins/kink_0.yaml -m motions.yaml -o output.yaml --delta 0.3


def run_dbastar(filename_env):

	with open(filename_env) as f:
		env = yaml.safe_load(f)

	robot_node = env["robots"][0]
	if robot_node["type"] == "car_first_order_0":
		robot = robots.RobotCarFirstOrder(0.5, 0.5)
	else:
		raise Exception("Unknown robot type!")

	rh = RobotHelper(robot_node["type"])
	# initialize delta
	x0 = np.array(robot_node["start"])
	xf = np.array(robot_node["goal"])

	delta = rh.distance(x0, xf) * 0.9
	maxCost = 1e6


	# load existing motions
	with open('motions.yaml') as f:
		motions = yaml.safe_load(f)
	# motions = []

	median = np.median([m['distance'] for m in motions])
	if delta > median:
		print("Adjusting delta!", delta, median)
		delta = median

	initialDelta = delta

	while True:
		print("delta", delta, "maxCost", maxCost)

		result = subprocess.run(["./dbastar", "-i", filename_env, "-m", "motions.yaml", "-o", "result_dbastar.yaml", "--delta", str(delta), "--maxCost", str(maxCost)])
		if result.returncode != 0:
			print("dbA* failed; Generating more primitives")

			for _ in range(10):
				print("gen motion", len(motions))
				motion = gen_motion_primitive.gen_random_motion(robot)
				motion['distance'] = rh.distance(motion['x0'], motion['xf'])
				motions.append(motion)
			with open('motions.yaml', 'w') as file:
				yaml.dump(motions, file)

			median = np.median([m['distance'] for m in motions])
			if delta > median:
				print("Adjusting delta!", delta, median)
				delta = median

		else:
			success = main_scp.run_scp(filename_env, "result_dbastar.yaml")
			if not success:
				print("Optimization failed; Reducing delta")
				delta = delta * 0.9
			else:
				with open("result_dbastar.yaml") as f:
					result = yaml.safe_load(f)
					cost = len(result["result"][0]["states"])
				print("success!", cost)
				maxCost = cost * 0.99
				# delta = initialDelta
				# break

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	args = parser.parse_args()

	run_dbastar(args.env)


if __name__ == '__main__':
	main()
