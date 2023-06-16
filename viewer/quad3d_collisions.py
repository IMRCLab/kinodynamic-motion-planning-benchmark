import quad3d_viewer
import viewer_utils
import matplotlib.pyplot as plt

viewer = quad3d_viewer.Quad3dViewer()
env = "../benchmark/quadrotor_0/quad_one_obs.yaml"


fig = plt.figure()
ax = plt.axes(projection='3d')


viewer.view_problem(ax, env)


p1 = [1.5, 1.5, 1.00001]
p2 = [1.28482, 1.28085, 1.00001]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '-o')

p1 = [4.5, 4.5, 1.0002]
p2 = [4.71305, 4.72133, 1.00017]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '-o')


x1 = [1., 1., 1., 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

viewer.view_state(ax, x1)

viewer_utils.plt_sphere(ax, [(1., 1., 1.), (5., 5., 1.)], [.4, .4])


x2 = [1.2, 1.5, 2., 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]


# p1 = [1.5, 1.5, 2]
# p2 = [1.52646, 1.57948, 2]


p1 = [1.5, 1.5, 1.99987]
p2 = [1.6, 1.5002, 1.99987]


ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], '-o')


# x = [  1.8,1.7,2., 0,0,0,1, 0,0,0, 0,0,0 ]

x3 = [5., 5., 1., 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

viewer.view_state(ax, x3)

viewer_utils.plt_sphere(
    ax, [x1[:3], x2[:3], x3[:3]], [.4, .4, .4])


plt.show()
