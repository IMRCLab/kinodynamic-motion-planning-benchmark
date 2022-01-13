import numpy as np
from scp import SCP
import robots
import yaml
import multiprocessing as mp
import tqdm
import itertools
import argparse
import re
import gen_motion_primitive

import sys, os
sys.path.append(os.getcwd())

# normalize angle between -pi and pi
def norm_angle(angle):
	# eps = 0.01
	# if angle < np.pi + eps and angle > -np.pi - eps:
	# 	return angle 
	return (angle + np.pi) % (2 * np.pi) - np.pi

def diff_angle(angle1, angle2):
	return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def gen(description):
	prim, robot, x0, xf = description
	# for _ in range(3):
	r = gen_motion_primitive.gen_motion(robot, x0, xf)
	if r is None:
		print("Warning: couldn't solve ", x0, xf, prim)
	return prim, r


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("mprim", help="SBPL MPRIM File (to copy x0->xf pairs from)")
	parser.add_argument("output_mprim", help="SBPL MPRIM File")
	parser.add_argument("output_yaml", help="SBPL MPRIM File")
	args = parser.parse_args()

	# parse the input file
	prims = []
	with open(args.mprim) as f:
		for line in f:
			key = "resolution_m:"
			if line.startswith(key):
				resolution = float(line[len(key):])
			key = "numberofangles:"
			if line.startswith(key):
				num_angles = int(line[len(key):])

			key = "primID:"
			if line.startswith(key):
				primId = int(line[len(key):])
				prim = {'primID': primId}
			key = "startangle_c:"
			if line.startswith(key):
				start_c = int(line[len(key):])
				prim['start_c'] = start_c
			key = "endpose_c:"
			if line.startswith(key):
				data = [int(num) for num in line[len(key):].split()]
				prim['endpose_c'] = data
				prims.append(prim)

	# rh = RobotHelper("car_first_order_0")
	robot = robots.create_robot("unicycle_first_order_0")

	# angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
	d_angle = 2*np.pi / num_angles
	tasks = []

	for prim in prims:
		x0 = np.array([0, 0, norm_angle(prim['start_c'] * d_angle)])
		xf = np.array([prim['endpose_c'][0] * resolution,
						prim['endpose_c'][1] * resolution,
						norm_angle(prim['endpose_c'][2] * d_angle)])
		# normalize angles
		xf[2] = x0[2] + diff_angle(xf[2], x0[2])
		if xf[2] < -np.pi or x0[2] < -np.pi:
			xf[2] += 2*np.pi
			x0[2] += 2*np.pi
		# print(x0[2], xf[2], diff_angle(xf[2], x0[2]), x0[2] + diff_angle(xf[2], x0[2]))

		# if np.isclose(x0[2], np.pi) and xf[2] < 0:
			# x0

		tasks.append((prim, robot, x0, xf))
		# r = gen_motion_primitive.gen_motion(robot, x0, xf)
		# if r is None:
		# 	print("WARNING: COULDN't FIND ", x0, xf)
		# else:
		# 	print(r)
	# exit()



	# for goal_discrete_x in range(-8, 8):
	# 	goal_x = goal_discrete_x * resolution
	# 	for goal_discrete_y in range(-8, 8):
	# 		goal_y = goal_discrete_y * resolution
	# 		for discrete_start_theta, start_theta in enumerate(angles):
	# 			for discrete_goal_theta, goal_theta in enumerate(angles):
	# 				x0 = np.array([0, 0, start_theta])
	# 				xf = np.array([goal_x, goal_y, goal_theta])
	# 				tasks.append((x0, xf))
	# 				# r = gen_motion(robot, x0, xf)
	# 				# if r is not None:
	# 					# print(r)

	# print(len(tasks))


	# # tasks = list(itertools.repeat(robot, N))
	# # print(tasks)
	# # for k in range(N):
	# # 	print(k)
	# # 	motion = gen_random_motion(robot)
	# # 	motions.append(motion)

	motions = []
	mp.set_start_method('spawn')
	with mp.Pool() as p:
		for prim, motion in tqdm.tqdm(p.imap_unordered(gen, tasks)):
			if motion:
				# motion['distance'] = rh.distance(motion['x0'], motion['xf'])
				motion['name'] = 'm{}'.format(len(motions))
				motion['prim'] = prim
				motions.append(motion)

	print(len(motions))

	# # for x in [-0.25, 0, 0.25]:
	# # 	for y in [-0.25, 0, 0.25]:
	# # 		for yaw in np.linspace(-np.pi, np.pi, 8):
	# # 			motion = gen_motion(robot, 
	# # 				np.array([0, 0, 0], dtype=np.float32),
	# # 				np.array([x, y, yaw], dtype=np.float32))
	# # 			if motion is not None:
	# # 				print(x,y, yaw)
	# # 				motions.append(motion)

	with open(args.output_mprim, 'w') as f:
		f.write("resolution_m: {}\n".format(resolution))
		f.write("numberofangles: {}\n".format(num_angles))
		f.write("totalnumberofprimitives: {}\n".format(len(motions)))
		for m in motions:
			f.write("primID: {}\n".format(m['prim']['primID']))
			f.write("startangle_c: {}\n".format(m['prim']['start_c']))
			f.write("endpose_c: {} {} {}\n".format(
				m['prim']['endpose_c'][0],
				m['prim']['endpose_c'][1],
				m['prim']['endpose_c'][2]))
			f.write("additionalactioncostmult: {}\n".format(1))
			f.write("intermediateposes: {}\n".format(len(m['states'])))
			for s in m['states']:
				f.write("{} {} {}\n".format(s[0], s[1], s[2]))

	with open(args.output_yaml, 'w') as f:
		yaml.dump(motions, f)


if __name__ == '__main__':
	main()
