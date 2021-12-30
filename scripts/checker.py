import argparse
import yaml
import numpy as np

import sys
import os
sys.path.append(os.getcwd())
from motionplanningutils import CollisionChecker, RobotHelper
import robots


def check(filename_env: str, filename_result: str) -> bool:

	with open(filename_env) as f:
		env = yaml.safe_load(f)

	robot_node = env["robots"][0]
	robot = robots.create_robot(robot_node["type"])

	x0 = np.array(robot_node["start"])
	xf = np.array(robot_node["goal"])

	cc = CollisionChecker()
	cc.load(filename_env)

	with open(filename_result) as f:
		result = yaml.safe_load(f)

	states = np.array(result["result"][0]["states"])
	actions = np.array(result["result"][0]["actions"])

	def check_array(a, b, msg):
		success = np.allclose(a, b, rtol=0.01, atol=1e-3)
		if not success:
			print(msg, a, b)
		return success

	success = True
	success &= check_array(states[0], x0, "start state")
	success &= check_array(states[-1], xf, "end state")
	# dynamics
	T = states.shape[0]
	for t in range(T-1):
		state_desired = robot.step(states[t], actions[t])
		success &= check_array(states[t+1], state_desired, "dynamics at t={}".format(t))
	# state limits
	for t in range(T):
		if (states[t] > robot.max_x).any() or (states[t] < robot.min_x).any():
			print("State outside bounds at t={} ({})".format(t, states[t]))
			success = False
	# action limits
	for t in range(T-1):
		if (actions[t] > robot.max_u).any() or (actions[t] < robot.min_u).any():
			print("Action outside bounds at t={} ({})".format(t, actions[t]))
			success = False
	# collisions
	for t in range(T):
		dist, _, _ = cc.distance(states[t])
		if dist < -1e-2:
			print("Collision at t={} ({})".format(t, dist))
			success = False

	return success


def compute_delta(filename_env: str, filename_result: str) -> float:

	with open(filename_env) as f:
		env = yaml.safe_load(f)

	robot_node = env["robots"][0]
	robot = robots.create_robot(robot_node["type"])

	x0 = np.array(robot_node["start"])
	xf = np.array(robot_node["goal"])

	rh = RobotHelper(robot_node["type"])

	with open(filename_result) as f:
		result = yaml.safe_load(f)

	states = np.array(result["result"][0]["states"])
	actions = np.array(result["result"][0]["actions"])

	deltas = []
	deltas.append(rh.distance(x0, states[0]))
	# dynamics
	T = states.shape[0]
	for t in range(T-1):
		state_desired = robot.step(states[t], actions[t])
		delta = rh.distance(state_desired, states[t+1])
		# print(t, delta, states[t], actions[t], states[t+1], state_desired)
		deltas.append(delta)
	deltas.append(rh.distance(xf, states[-1]))
	# idx = np.argmax(deltas) + 1
	# print(idx, states[idx], actions[idx], states[idx+1])
	return np.max(deltas)


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("result", help="file containing the result (YAML)")
	args = parser.parse_args()

	print(check(args.env, args.result))
	print(compute_delta(args.env, args.result))


if __name__ == "__main__":
	main()
