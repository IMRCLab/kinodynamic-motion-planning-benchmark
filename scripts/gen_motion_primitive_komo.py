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

def gen_motion(robot_type, start, goal, is2D):
	dbg = False
	with tempfile.TemporaryDirectory() as tmpdirname:
		if dbg:
			p = Path("../results/test")
		else:
			p = Path(tmpdirname)
		env = {
			"environment":{
				"min": [-2, -2],
				"max": [2, 2],
				"obstacles": []
			},
			"robots": [{
				"type": robot_type,
				"start": list(start),
				"goal": list(goal),
			}]
		}
		if not is2D:
			env["environment"]["min"].append(-2)
			env["environment"]["max"].append(2)

		filename_env = str(p / "env.yaml")
		with open(filename_env, 'w') as f:
			yaml.dump(env, f, Dumper=yaml.CSafeDumper)

		success = run_komo_standalone(filename_env, str(p), 5 * 60, "action_factor: 1.0", search="linear", initialguess="none")
		if not success:
			return []

		filename_result = p / "result_komo.yaml"
		# checker.check(str(filename_env), str(filename_result))

		if dbg:
			subprocess.run(["python3",
						# "../benchmark/unicycleFirstOrder/visualize.py",
						# "../benchmark/unicycleSecondOrder/visualize.py",
						# "../benchmark/carFirstOrderWithTrailers/visualize.py",
						"../benchmark/quadrotor/visualize.py",
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
			# use the following break to only create the first motion
			# this will create a nicer (uniform) distribution, but take
			# much longer
			# break
		return motions


def gen_random_motion(robot_type):
	# NOTE: It is *very* important to keep this as a local import, otherwise
	#       random numbers may repeat, when using multiprocessing
	from motionplanningutils import RobotHelper

	rh = RobotHelper(robot_type)
	start = rh.sampleUniform()
	goal = rh.sampleUniform()
	# shift to position = zeros
	start[0] = 0
	start[1] = 0
	if not rh.is2D():
		start[2] = 0
	#TODO:
	goal[0:3] = np.random.uniform(-0.25, 0.25, 3).tolist()
	motions =  gen_motion(robot_type, start, goal, rh.is2D())
	for motion in motions:
		motion['distance'] = rh.distance(motion['x0'], motion['xf'])
	return motions


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("robot_type", help="name of robot type to generate motions for")
	parser.add_argument("--N", help="number of motions", default=100, type=int)
	args = parser.parse_args()

	# rh = RobotHelper(args.robot_type)

	motions = []
	tasks = itertools.repeat(args.robot_type, args.N)

	# if args.N <= 10:
	if False:
		while len(motions) < args.N:
			multiple_motions = gen_random_motion(args.robot_type)
			motions.extend(multiple_motions)
	else:
		# mp.set_start_method('spawn')
		use_cpus = psutil.cpu_count(logical=False)
		async_results = []
		with mp.Pool(use_cpus) as p:
			len_motions_last_printed = 0
			while len(motions) < args.N:
				# clean up async_results
				async_results = [x for x in async_results if not x.ready()]
				# run some more workers
				while len(async_results) < use_cpus:
					ar = p.apply_async(gen_random_motion, (args.robot_type,), callback=lambda r: motions.extend(r))
					async_results.append(ar)
				if len(motions) > len_motions_last_printed:
					print("Generated {} motions".format(len(motions)))
					len_motions_last_printed = len(motions)
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
		motion['name'] = 'm{}'.format(k)

	print("Generated {}".format(len(motions)))

	with open("motions_{}.yaml".format(args.robot_type), 'w') as file:
		yaml.dump(motions, file)


if __name__ == '__main__':
	main()
