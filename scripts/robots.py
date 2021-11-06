import jax.numpy as np

class RobotCarFirstOrder:

	def __init__(self, v_limit, w_limit):
		self.action_desc = ["v [m/s]", "w [rad/s]"]
		self.min_u = np.array([-v_limit, -w_limit])
		self.max_u = np.array([v_limit, w_limit])

		self.state_desc = ["x [m]", "y [m]", "yaw [rad]"]
		self.min_x = np.array([-np.inf, -np.inf, -np.pi])
		self.max_x = np.array([np.inf, np.inf, np.pi])

	def step(self, state, action):
		dt = 0.1
		x, y, yaw = state
		v, w = action

		x_next = x + v * np.cos(yaw) * dt
		y_next = y + v * np.sin(yaw) * dt
		yaw_next = yaw + w * dt
		# normalize yaw between -pi and pi
		yaw_next_norm = (yaw_next + np.pi) % (2 * np.pi) - np.pi

		state_next = np.array([x_next, y_next, yaw_next_norm])
		return state_next


if __name__ == '__main__':
	pass
	# import matplotlib.pyplot as plt

	# position = np.arange(-1.2, 0.6, 0.05)
	# f = - 0.0025 * np.cos(3 * position)
	# print(np.min(f), np.max(f))
	# fig, ax = plt.subplots()
	# ax.plot(position, f)
	# plt.show()

	# x = np.array([-0.5, 0.0])

	# robot = ContinuousMountainCarAutograd()
	# u = [robot.max_u]

	# states = [x]
	# for _ in range(100):
	# 	x = robot.step(x, u)
	# 	states.append(x)

	# states = np.array(states)


	# fig, axs = plt.subplots(3, 1, sharex='all')
	# axs[0].plot(states[:,0])
	# axs[0].set_ylabel("position [m]")
	# axs[1].plot(states[:,1])
	# axs[1].set_ylabel("velocity [m/s]")

	# ext_force = 0.0025 * np.cos(3 * states[:,0])
	# axs[2].plot(ext_force)
	# axs[2].set_ylabel("external force [N]")
	# plt.show()
