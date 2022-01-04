import jax.numpy as np

def normalize_angle(angle):
	return (angle + np.pi) % (2 * np.pi) - np.pi

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


class RobotCarSecondOrder:

	def __init__(self, v_limit, w_limit, a_limit, w_dot_limit):
		self.action_desc = ["a [m^2/s]", "w_dot [rad^2/s]"]
		self.min_u = np.array([-a_limit, -w_dot_limit])
		self.max_u = np.array([a_limit, w_dot_limit])

		self.state_desc = ["x [m]", "y [m]", "yaw [rad]", "v [m/s]", "w [rad/s]"]
		self.min_x = np.array([-np.inf, -np.inf, -np.pi, -v_limit, -w_limit])
		self.max_x = np.array([np.inf, np.inf, np.pi, v_limit, w_limit])

	def step(self, state, action):
		dt = 0.1
		x, y, yaw, v, w = state
		a, w_dot = action

		x_next = x + v * np.cos(yaw) * dt
		y_next = y + v * np.sin(yaw) * dt
		yaw_next = yaw + w * dt
		# normalize yaw between -pi and pi
		yaw_next_norm = (yaw_next + np.pi) % (2 * np.pi) - np.pi
		v_next = v + a * dt
		w_dot_next = w + w_dot * dt

		state_next = np.array([x_next, y_next, yaw_next_norm, v_next, w_dot_next])
		return state_next


# LaValle book, Equation 13.19
class RobotCarFirstOrderWithTrailers:

	def __init__(self, v_limit, phi_limit, L, hitch_lengths):
		self.action_desc = ["v [m/s]", "steering angle [rad]"]
		self.min_u = np.array([-v_limit, -phi_limit])
		self.max_u = np.array([v_limit, phi_limit])

		self.state_desc = ["x [m]", "y [m]", "yaw [rad]"]
		min_x = [-np.inf, -np.inf, -np.pi]
		max_x = [np.inf, np.inf, np.pi]
		for k in range(len(hitch_lengths)):
			self.state_desc.append("trailer{} yaw [rad]".format(k+1))
			min_x.append(-np.pi)
			max_x.append(np.pi)
		self.min_x = np.array(min_x)
		self.max_x = np.array(max_x)

		self.L = L
		self.hitch_lengths = hitch_lengths

	def step(self, state, action):
		""""
		x_dot = v * cos (theta_0)
		y_dot = v * sin (theta_0)
		theta_0_dot = v / L * tan(phi)
		theta_1_dot = v / hitch_lengths[0] * sin(theta_0 - theta_1)
		...
		"""
		dt = 0.1
		x, y, yaw = state[0], state[1], state[2]
		v, phi = action

		x_next = x + v * np.cos(yaw) * dt
		y_next = y + v * np.sin(yaw) * dt
		yaw_next = yaw + v / self.L * np.tan(phi) * dt
		# normalize yaw between -pi and pi
		yaw_next_norm = normalize_angle(yaw_next)

		state_next_list = [x_next, y_next, yaw_next_norm]

		for i, d in enumerate(self.hitch_lengths):
			theta_dot = v / d
			for j in range(0, i):
				theta_dot *= np.cos(state[2+j] - state[3+j])
			theta_dot *= np.sin(state[2+i] - state[3+i])

			theta = state[3+i]
			theta_next = theta + theta_dot * dt
			theta_next_norm = normalize_angle(theta_next)
			state_next_list.append(theta_next_norm)

		state_next = np.array(state_next_list)
		return state_next

def create_robot(robot_type):
	if robot_type == "car_first_order_0":
		return RobotCarFirstOrder(0.5, 0.5)
	elif robot_type == "car_second_order_0":
		return RobotCarSecondOrder(0.5, 0.5, 2, 2)
	elif robot_type == "car_first_order_with_0_trailers_0":
		return RobotCarFirstOrderWithTrailers(0.5, np.pi/3, 0.4, [])
	elif robot_type == "car_first_order_with_1_trailers_0":
		return RobotCarFirstOrderWithTrailers(0.5, np.pi/3, 0.4, [0.5])
	else:
		raise Exception("Unknown robot type {}!".format(robot_type))


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
