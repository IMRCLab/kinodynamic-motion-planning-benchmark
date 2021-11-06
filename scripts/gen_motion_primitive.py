import numpy as np
from scp import SCP
import robots
import yaml

# two point boundary value problem
def TPBVP_fixed_time(robot, x0, xf, T):
	scp = SCP(robot)

	# initialize with random rollout
	states = np.empty((T,len(robot.state_desc)))
	states[0] = x0
	actions = np.empty((T-1,len(robot.action_desc)))
	for k in range(T-1):
		actions[k] = np.random.uniform(robot.min_u, robot.max_u)
		states[k+1] = robot.step(states[k], actions[k])

	# states = np.tile(x0, (T, 1))
	# actions = np.zeros((T-1, 2))
	# states[1:] += np.random.normal(0, 0.001, states.shape)[1:]
	# actions += np.random.normal(0, 0.001, actions.shape)

	X, U, val = scp.min_xf(states, actions, xf, 10, trust_x=0.1, trust_u=0.1)

	if len(X) > 1:
		return X[-1], U[-1], val

	return None, None, None

def gen_random_motion(robot):
	while True:
		yaw0 = np.random.uniform(-np.pi, np.pi)
		x0 = np.array([0, 0, yaw0], dtype=np.float32)

		xf = np.random.uniform(-2, 2)
		yf = np.random.uniform(-2, 2)
		yawf = np.random.uniform(-np.pi, np.pi)
		xf = np.array([xf, yf, yawf], dtype=np.float32)
		T = np.random.choice([8, 16, 32])

		X, U, _ = TPBVP_fixed_time(robot, x0, xf, T)
		if X is not None:
			r = dict()
			r['x0'] = x0.tolist()
			r['xf'] = X[-1].tolist()
			r['states'] = X.tolist()
			r['actions'] = U.tolist()
			r['T'] = int(T)
			return r


def gen_motion(robot, x0, xf):
	print("Try ", xf)
	for T in range(2, 32):
	# for T in [32]:
		X, U, val = TPBVP_fixed_time(robot, x0, xf, T)
		if X is not None and val < 1e6:
			r = dict()
			r['x0'] = x0.tolist()
			r['xf'] = X[-1].tolist()
			r['states'] = X.tolist()
			r['actions'] = U.tolist()
			r['T'] = int(T)
			return r


if __name__ == '__main__':
	robot = robots.RobotCarFirstOrder(0.5, 0.5)

	motions = []
	# motion = gen_random_motion(robot)
	for x in [-0.25, 0, 0.25]:
		for y in [-0.25, 0, 0.25]:
			for yaw in np.linspace(-np.pi, np.pi, 8):
				motion = gen_motion(robot, 
					np.array([0, 0, 0], dtype=np.float32),
					np.array([x, y, yaw], dtype=np.float32))
				if motion is not None:
					print(x,y, yaw)
					motions.append(motion)

	with open('result.yaml', 'w') as file:
		yaml.dump(motions, file)
