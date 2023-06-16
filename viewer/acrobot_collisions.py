import acrobot_viewer
import matplotlib.pyplot as plt
import numpy as np

viewer = acrobot_viewer.AcrobotViewer()
env = "../benchmark/acrobot/swing_up_obs.yaml"

fig, ax = plt.subplots()

viewer.view_problem(ax, env)
#
p1 = [1.15, 1.15, 0.5]
p2 = [0.05, 0, 0.5]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')
p1 = [1.15, 1.5, 0.5]
p2 = [0.050004, 1.5, 0.5]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')

p1 = [1.15, 1.15, 0.5]
p2 = [1.05, 1, 0.5]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


x = [np.pi / 2., np.pi / 2., 0, 0]


viewer.view_state(ax, x)
plt.show()
