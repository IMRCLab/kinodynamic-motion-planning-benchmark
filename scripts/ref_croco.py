# workaround, to get a simple system working.
# I will just have some weigh vector, 
# and change the weights for the goal.

# It should be straight forward to integrated with Wolfgangs
# collisions

# Then, I can add for help with the other parts.
# What about the jacobians? # Is calc diff working fine?

# Lets starts with unicycle!

import matplotlib.pylab as plt
import crocoddyl
import numpy as np
from unicycle_utils import *
import sys
from croco_models import * 

solver = crocoddyl.SolverDDP
solver = crocoddyl.SolverFDDP
solver = crocoddyl.SolverBoxDDP

LB = np.array([-.5,-.5])
UB = np.array([.5,.5])

ORDER = 2
obs = [np.array([-.5 , -.6])]
obs_radius = .3

if ORDER == 1:
    ActionModel = ActionModelUnicycle2
elif ORDER == 2:
    ActionModel = ActionModelUnicycleSecondOrder


if ORDER == 1:
    goal = np.zeros(3) # first order
elif ORDER == 2:
    goal = np.zeros(5) # second ordeORDER == 2:



weight_goal =  10
weight_control=  1
weight_obstacles=  10


if ORDER == 1:
    x0 = np.array([-1,-1,1])
elif ORDER == 2:
    x0 = np.array([-1,-1,1, 0,0])

T = 50

featss = [] 
for i in range(T): 
    feats = FeatUniversal([AuglagFeat(Feat_terminal(goal, weight_goal),len(goal),0), CostFeat(Feat_control(weight_control),2,1.) , AuglagFeat(Feat_obstacles(obs, obs_radius , weight_obstacles),len(obs),1.)])
    featss.append(feats)

feats = FeatUniversal([AuglagFeat(Feat_terminal(goal, weight_goal),len(goal),1.0), CostFeat(Feat_control(weight_control),2,0.0) , AuglagFeat(Feat_obstacles(obs, obs_radius, weight_obstacles),len(obs),1.)])
featss.append(feats)

rM_basic = []
rM = []
# create models and features
for i in range(T):
    unicycle =  ActionModel(featss[i], featss[i].nr)
    unicycleIAM = crocoddyl.ActionModelNumDiff(unicycle, True)
    unicycleIAM.u_lb=LB
    unicycleIAM.u_ub=UB
    print(unicycleIAM.has_control_limits)
    rM_basic.append(unicycle)
    rM.append(unicycleIAM)

# terminal
rT_basic =  ActionModel(featss[T], featss[T].nr)
rT = crocoddyl.ActionModelNumDiff(rT_basic, True)

problem = crocoddyl.ShootingProblem(x0, rM, rT)


ddp = solver(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
done = ddp.solve()

log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2, show=False)

plt.show()

plt.clf()
for x in ddp.xs: 
    plotUnicycle(x)


plt.scatter( goal[0], goal[1], 
facecolors='none', edgecolors='b' )

for o in obs:
    circle = plt.Circle((o[0], o[1]), obs_radius, color='b', fill=False)
    ax = plt.gca()
    ax.add_patch(circle)

plt.axis([-2, 2, -2, 2])
plt.show()


# sys.exit()

# print("second solve")
# print(log.xs)
# print(log.xs.shape)

xs = log.xs
xsPlotIdx = 111
nx = xs[0].shape[0]
X = [0.] * nx
print(nx)
for i in range(nx):
    X[i] = [np.asscalar(x[i]) for x in xs]
print(X)

print("before")
for i in xs:
    print(i)

# init_xs = [x0] * (problem.T + 1)

init_xs = []
for i in range(len(xs)):
    init_xs.append( np.zeros(len(goal)))
    for ii in range(len(xs[i])):
        init_xs[i][ii] = xs[i][ii]
        init_xs[i][ii] += .1

print("init_xs" , init_xs)

print("after")
for i in xs:
    print(i)

# import sys
# sys.exit()

# for i in range(len(X)):
#     for ii in range(len(X[i])):
#         X[i][ii] += 0.1
        

# print("after")
# print(xs)
print("second solve")
ddp2 = solver(problem)
ddp2.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
done = ddp2.solve(init_xs=init_xs, init_us=log.us)

for i in ddp2.xs:
    print(i)


# evaluate the model!

xs = ddp.xs
us = ddp.us









# print(ddp2.xs)

# plotUnicycleSolution(log.xs, figIndex=3, show=True)

# assert done


# assert done


# In[25]:


plt.clf()
for x in ddp.xs: 
    plotUnicycle(x)


plt.scatter( goal[0], goal[1], 
facecolors='none', edgecolors='b' )

for o in obs:
    circle = plt.Circle((o[0], o[1]), obs_radius, color='b', fill=False)
    ax = plt.gca()
    ax.add_patch(circle)

plt.axis([-2, 2, -2, 2])
plt.show()


# get the  cost


# plt.clf()

# for obs in model.obs:
#     print("circle")
#     circle = plt.Circle((obs[0], obs[1]), model.obs_radius, color='b', fill=False)
#     circle = plt.Circle((obs[0], obs[1]), .1 , color='b')

# plt.axis([-2, 2, -2, 2])
# plt.show()



# and the final state is:

# In[17]:

print(ddp.xs[-1])
l = [1]
bb = 10 * l
print(bb)
bb[-1]=2
print(bb)

## SIMPLE PENALTY METHOD: Increase the penalty

penalty=False
visualize=True
# ddp = 

solver = crocoddyl.SolverDDP
solver = crocoddyl.SolverFDDP
solver = crocoddyl.SolverBoxDDP

ddp = solver(problem)

def extra_plot(plt):
    plt.scatter( goal[0], goal[1], facecolors='none', edgecolors='b' )

    for o in obs:
        circle = plt.Circle((o[0], o[1]), obs_radius, color='b', fill=False)
        ax = plt.gca()
        ax.add_patch(circle)

if penalty:
    penalty_solver(ddp,[],[], featss, visualize,extra_plot)

# Augmented Lagrangian 
# Number of iterations
num_it = 20

l = []
xs = []
us = []


# I just need better tools!


auglag = True
if auglag:
    # penalty_solver(ddp,featss, visualize,extra_plot)

    auglag_solver( ddp, [], [], featss, np.zeros(2),  visualize , extra_plot)

    












