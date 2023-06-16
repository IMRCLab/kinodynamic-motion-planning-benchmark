import unicycle1_viewer
import matplotlib.pyplot as plt


viewer = unicycle1_viewer.Unicycle1Viewer()


env = "../benchmark/unicycle_first_order_0/parallelpark_0.yaml"


fig, ax = plt.subplots()

viewer.view_problem(ax, env)


p1 = [0.5, 0.425, 0.5]
p2 = [0.5, 0.675, 0.5]
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


p1 = [1.35, 0.425, 0.5]
p2 = [1.65, 0.425, 0.5]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


viewer.view_state(ax, [1.5, .3, .1], facecolor='none', edgecolor='blue')

p1 = [1.35, 0.399417, 0.448834]
p2 = [1.23877, 0.399417, 0.448834]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')

plt.show()


# I have P

# P in global coordinates
# P in local is (P - trans)


# Pg = R * Plocal + Center
# D(Pg) / drotation = R' * local
# D(Pg) / dCenter = 1


# Convex Approximation of the distance function is

# for distance Pose
# d is positive
# d(pRobot(q), pObs ) = || pRobot - PObs ||

# if d > 0 (not collision)
# d (pRobot(q), pObs ) = || pRobot - PObs || +  1 / d    ( pRobot - PObs )
# * dpRobot  / dq

# if d is negative
# d(pRobot(q), pObs ) =  - || pRobot - PObs ||
# d (pRobot(q), pObs ) =  - || pRobot - PObs ||  +  -1 / d    ( pRobot -
# PObs ) * dpRobot  / dq
