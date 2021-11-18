import numpy as np
from scp import SCP
import robots
import yaml
import argparse

import sys
import os
sys.path.append(os.getcwd())
from motionplanningutils import CollisionChecker


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("initial_guess", help="file containing the initial_guess (e.g., from db-A*) (YAML)")
	args = parser.parse_args()


	with open(args.env) as f:
		env = yaml.safe_load(f)

	robot_node = env["robots"][0]
	if robot_node["type"] == "car_first_order_0":
		robot = robots.RobotCarFirstOrder(0.5, 0.5)
	else:
		raise Exception("Unknown robot type!")

	x0 = np.array(robot_node["start"])
	xf = np.array(robot_node["goal"])

	cc = CollisionChecker()
	cc.load(args.env)
	# r = cc.distance(np.array(robot_node["start"]))
	# print(r)
	# exit()

	with open(args.initial_guess) as f:
		initial_guess = yaml.safe_load(f)

	states = np.array(initial_guess["result"][0]["states"])
	actions = np.array(initial_guess["result"][0]["actions"])

	trust_x_est = np.max(np.abs(np.diff(states, axis=0)), axis=0)
	trust_u_est = np.max(np.abs(np.diff(actions, axis=0)), axis=0)
	# print(trust_x_est, trust_u_est)
	# exit()
	
	# scp = SCP(robot)
	scp = SCP(robot, cc)
	X, U, val = scp.min_u(states, actions, x0, xf, 3, trust_x=2*trust_x_est, trust_u=2*trust_u_est, verbose=True)

	result = dict()
	result["result"] = [{'states': X[-1].tolist(), 'actions': U[-1].tolist()}]
	with open('result_scp.yaml', 'w') as f:
		yaml.dump(result, f)


