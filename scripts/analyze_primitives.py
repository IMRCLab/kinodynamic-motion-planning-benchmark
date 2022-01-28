import argparse
import yaml
import sys, os
import numpy as np
import tempfile
from pathlib import Path

sys.path.append(os.getcwd())
from motionplanningutils import RobotHelper

def sort_primitives(motions: list, robot_type: str, top_k=None) -> list:
	rh = RobotHelper(robot_type)

	# use as first/seed motion the one that moves furthest
	best_motion = motions[0]
	largest_d = 0
	for m in motions:
		d = rh.distance(m["x0"], m["xf"])
		if d > largest_d:
			largest_d = d
			best_motion = m

	used_motions = [best_motion]
	unused_motions = list(motions)
	unused_motions.remove(best_motion)

	if top_k is None:
		top_k = len(motions) - 1

	for k in range(top_k):
		best_d = -1
		best_motion = None
		for m1 in unused_motions:
			# find smallest distance to existing neighbors
			smallest_d_x0 = np.inf
			for m2 in used_motions:
				d = rh.distance(m1["x0"], m2["x0"])
				if d < smallest_d_x0:
					smallest_d_x0 = d
			smallest_d_xf = np.inf
			for m2 in used_motions:
				d = rh.distance(m1["xf"], m2["xf"])
				if d < smallest_d_xf:
					smallest_d_xf = d
			# find largest among the smallest
			smallest_d = smallest_d_x0 + smallest_d_xf
			if smallest_d > best_d:
				best_motion = m1
				best_d = smallest_d
		assert(best_motion is not None)
		used_motions.append(best_motion)
		unused_motions.remove(best_motion)
		print("sorting ", k)
	return used_motions

def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("motions", help="file containing the motions (YAML)")
	parser.add_argument("robot_type", help="name of robot type to generate motions for")
	args = parser.parse_args()

	with open(args.motions) as f:
		motions = yaml.load(f, Loader=yaml.CSafeLoader)

	filename_motions_sorted = Path(args.motions).stem + "_sorted.yaml"
	used_motions = sort_primitives(motions, args.robot_type)
	with open(filename_motions_sorted, 'w') as file:
		yaml.dump(used_motions, file, Dumper=yaml.CSafeDumper)

	exit()


	robot_type = "unicycle_second_order_0"



	used_motions = sort_primitives(motions, robot_type, 10)

	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)

		for k, m in enumerate(used_motions):
			for s in m["states"]:
				s[0] += 2
				s[1] += 2

			# convert to result file format
			result = {
				"result": [{
						"states": m["states"],
						"actions": m["actions"],
				}]}
			filename_result = str(p / "result.yaml")
			with open(filename_result, 'w') as f:
				yaml.dump(result, f, Dumper=yaml.CSafeDumper)

			# generate environment file
			env = {
				"environment":{
					"dimensions": [4, 4],
					"obstacles": []
				},
				"robots": [{
					"type": robot_type,
					"start": list(m["states"][0]),
					"goal": list(m["states"][-1]),
				}]
			}
			filename_env = str(p / "env.yaml")
			with open(filename_env, 'w') as f:
				yaml.dump(env, f, Dumper=yaml.CSafeDumper)

			import subprocess
			subprocess.run(["python3",
				"../benchmark/unicycleSecondOrder/visualize.py",
				str(filename_env),
				"--result", str(filename_result),
				"--video", "../results/test/{}.mp4".format(k)])

	for m in used_motions:
		print(m["x0"], m["xf"])

if __name__ == "__main__":
	main()