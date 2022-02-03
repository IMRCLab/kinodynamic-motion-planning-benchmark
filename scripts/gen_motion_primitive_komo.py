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
import checker


import sys, os
sys.path.append(os.getcwd())
from motionplanningutils import RobotHelper

def gen_motion(robot_type, start, goal):
	dbg = False
	with tempfile.TemporaryDirectory() as tmpdirname:
		if dbg:
			p = Path("../results/test")
		else:
			p = Path(tmpdirname)
		env = {
			"environment":{
				"dimensions": [4, 4],
				"obstacles": []
			},
			"robots": [{
				"type": robot_type,
				"start": list(start),
				"goal": list(goal),
			}]
		}

		filename_env = str(p / "env.yaml")
		with open(filename_env, 'w') as f:
			yaml.dump(env, f, Dumper=yaml.CSafeDumper)

		run_komo_standalone(filename_env, str(p), 60, "action_factor: 1.0", search="linear")

		filename_result = p / "result_komo.yaml"
		checker.check(str(filename_env), str(filename_result))

		if dbg:
			subprocess.run(["python3",
						"../benchmark/unicycleFirstOrder/visualize.py",
						str(filename_env),
						"--result", str(filename_result),
						"--video", str(filename_result.with_suffix(".mp4"))])

		# read the result
		with open(filename_result) as f:
			result = yaml.load(f, Loader=yaml.CSafeLoader)

		states = np.array(result["result"][0]["states"])
		actions = np.array(result["result"][0]["actions"])

		eucledian_distance = 0
		split = [0]
		for k in range(1, len(states)):
			eucledian_distance += np.linalg.norm(states[k-1][0:2] - states[k][0:2])
			if eucledian_distance >= 0.5:
				split.append(k)
				eucledian_distance = 0
		
		# include last segment, if it not very short
		if len(states) - split[-1] > 5:
			split.append(len(states)-1)

		# create motions
		motions = []
		for idx in range(1, len(split)):
			start_k = split[idx-1]
			k = split[idx]
			# shift states
			states[start_k:, 0:2] -= states[start_k, 0:2]
			# create motion
			motion = dict()
			motion['x0'] = states[start_k].tolist()
			motion['xf'] = states[k].tolist()
			motion['states'] = states[start_k:k+1].tolist()
			motion['actions'] = actions[start_k:k].tolist()
			motion['T'] = k-start_k
			motions.append(motion)
		return motions


def gen_random_motion(robot_type):
	rh = RobotHelper(robot_type)
	start = rh.sampleUniform()
	goal = rh.sampleUniform()
	# shift to center (at 2,2)
	start[0] = 2
	start[1] = 2
	goal[0] += 2
	goal[1] += 2
	return gen_motion(robot_type, start, goal)


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
