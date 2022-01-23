import numpy as np
# from scp import SCP
from main_komo import run_komo_standalone
import robots
import yaml
import multiprocessing as mp
import tqdm
import itertools
import argparse
import subprocess
import tempfile
from pathlib import Path
import psutil


import sys, os
sys.path.append(os.getcwd())
from motionplanningutils import RobotHelper


def gen_random_motion(robot_type):
	# robot = robots.create_robot(robot_type)
	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)
		rh = RobotHelper(robot_type)
		env = {
			"environment":{
				"dimensions": [4, 4],
				"obstacles": []
			},
			"robots": [{
				"type": "unicycle_first_order_0",
				"start": [2, 2, rh.sampleUniform()[2]],
				"goal": (np.array(rh.sampleUniform()) + np.array([2,2,0])).tolist(),
			}]
		}

		filename_env = str(p / "env.yaml")
		with open(filename_env, 'w') as f:
			yaml.dump(env, f, Dumper=yaml.CSafeDumper)

		run_komo_standalone(filename_env, str(p), 60, search="linear")

		# subprocess.run(["python3",
		# 			"../benchmark/unicycleFirstOrder/visualize.py",
		# 			"env.yaml",
		# 			"--result", "../results/test/result_komo.yaml",
		# 			"--video", "../results/test/result_komo.mp4"])

		# read the result
		with open(p / "result_komo.yaml") as f:
			result = yaml.load(f, Loader=yaml.CSafeLoader)

		states = np.array(result["result"][0]["states"])
		actions = np.array(result["result"][0]["actions"])

		eucledian_distance = 0
		start_k = 0
		motions = []
		for k in range(1, len(states)):
			eucledian_distance += np.linalg.norm(states[k-1][0:2] - states[k][0:2])
			if eucledian_distance >= 0.5:
				# shift states
				# print(states[start_k:k+1, 0:2])
				states[start_k:, 0:2] -= states[start_k, 0:2]
				# print(states[start_k:k+1, 0:2])
				# create motion
				motion = dict()
				motion['x0'] = states[start_k].tolist()
				motion['xf'] = states[k].tolist()
				motion['states'] = states[start_k:k+1].tolist()
				motion['actions'] = actions[start_k:k].tolist()
				motion['T'] = k-start_k
				eucledian_distance = 0
				start_k = k
				motions.append(motion)
		return motions


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("robot_type", help="name of robot type to generate motions for")
	parser.add_argument("output", help="output file (YAML)")
	parser.add_argument("--N", help="number of motions", default=100, type=int)
	args = parser.parse_args()

	rh = RobotHelper(args.robot_type)

	motions = []
	tasks = itertools.repeat(args.robot_type, args.N)

	# if args.N <= 10:
	if False:
		while len(motions) < args.N:
			multiple_motions = gen_random_motion(args.robot_type)
			for motion in multiple_motions:
				motion['distance'] = rh.distance(motion['x0'], motion['xf'])
				motion['name'] = 'm{}'.format(len(motions))
				motions.append(motion)
	else:
		# mp.set_start_method('spawn')
		use_cpus = psutil.cpu_count(logical=False)
		async_results = []
		with mp.Pool(use_cpus) as p:
			while len(motions) < args.N:
				# clean up async_results
				async_results = [x for x in async_results if not x.ready()]
				# run some more workers
				while len(async_results) < use_cpus:
					ar = p.apply_async(gen_random_motion, (args.robot_type,), callback=lambda r: motions.extend(r))
					async_results.append(ar)
			p.terminate()
			# r.get()
			# p.close()
			# p.join()

			# for multiple_motions in tqdm.tqdm(p.imap_unordered(gen_random_motion, tasks)):
			# 	for motion in multiple_motions:
			# 		motion['distance'] = rh.distance(motion['x0'], motion['xf'])
			# 		motion['name'] = 'm{}'.format(len(motions))
			# 		motions.append(motion)
			# 		if len(motions) >= args.N:
			# 			break
			# 	if len(motions) >= args.N:
			# 		break

	for k, motion in enumerate(motions):
		motion['distance'] = rh.distance(motion['x0'], motion['xf'])
		motion['name'] = 'm{}'.format(k)

	print("Generated {}".format(len(motions)))

	with open(args.output, 'w') as file:
		yaml.dump(motions, file)


if __name__ == '__main__':
	main()
