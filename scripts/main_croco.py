import numpy as np
import yaml
import argparse
import tempfile
from pathlib import Path
import subprocess
import shutil
import time

import sys
import os
sys.path.append(os.getcwd())
import crocoddyl
from motionplanningutils import CollisionChecker
from croco_models import *

from scp import SCP
import robots
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import benchmark.unicycle_first_order_0.visualize as vis
obs_radius = .1


# def draw_box_patch(ax, center, size, angle = 0, **kwargs):
#   xy = np.asarray(center) - np.asarray(size) / 2
#   rect = Rectangle(xy, size[0], size[1], **kwargs)
#   t = matplotlib.transforms.Affine2D().rotate_around(
#       center[0], center[1], angle)
#   rect.set_transform(t + ax.transData)
#   ax.add_patch(rect)
#   return rect


## TODO: Option to use robot as a sphere

def croc_problem_from_env(env, T, cc):

    robot_node = env["robots"][0]
    x0 = np.array(robot_node["start"])
    goal = np.array(robot_node["goal"])

    LB = np.array([-.5,-.5])
    UB = np.array([.5,.5])

    obs = []
    for obstacle in env["environment"]["obstacles"]:
        obs.append(np.array(obstacle["center"]))

    print("obs are")
    print(obs)
    name = env["robots"][0]["type"] 
    if name == "unicycle_first_order_0":
        ActionModel = ActionModelUnicycle2
    elif name == "unicycle_second_order_0":
        ActionModel = ActionModelUnicycleSecondOrder

    weight_goal =  10
    weight_control=  1
    weight_bounds =  5
    weight_obstacles=  10
    
    featss = [] 
    rM_basic = []
    rM = []
    Eq_feat = AuglagFeat
    Feat_obstacles = FeatObstaclesFcl # 

    for i in range(T): 
        feats = FeatUniversal([Eq_feat(Feat_terminal(goal, weight_goal),len(goal),0), CostFeat(Feat_control(weight_control),2,1.) , Eq_feat(Feat_obstacles(obs,  weight_obstacles,cc),1,1.)])
        # In the second order car, i have to add limits on velocity?
        # TODO: Wolfgang check
        if name == "unicycle_second_order_0":
            ## Add feature
            x_lb = np.array([-np.inf, -np.inf, -np.inf, -.5 ,-.5])
            x_ub = np.array([np.inf, np.inf, np.inf, .5 ,.5])
            feats.append([Eq_feat(FeatBoundsX(x_lb, x_ub, weight_bounds),2*len(x_lb),1.)])
        featss.append(feats)
        # Model
        unicycle =  ActionModel(feats, feats.nr)
        unicycleIAM = crocoddyl.ActionModelNumDiff(unicycle, True)
        unicycleIAM.disturbance=1e-5 
        unicycleIAM.u_lb=LB
        unicycleIAM.u_ub=UB
        rM_basic.append(unicycle)
        rM.append(unicycleIAM)

    feats = FeatUniversal([Eq_feat(Feat_terminal(goal, weight_goal),len(goal),1.0), CostFeat(Feat_control(weight_control),2,0.0) , Eq_feat(Feat_obstacles(obs,  weight_obstacles, cc),1,1.)])
    featss.append(feats)
    rT_basic =  ActionModel(featss[T], featss[T].nr)
    rT = crocoddyl.ActionModelNumDiff(rT_basic, True)
    rT.disturbance=1e-5
    # create models and features
    # terminal
    # NOTE: study whether the disturbance is important or not!!
    problem = crocoddyl.ShootingProblem(x0, rM, rT)
    return problem , featss


def run_croco(filename_env, filename_initial_guess, filename_result='result_scp.yaml', iterations=5):


    with open(filename_env) as f:
        env = yaml.safe_load(f)

    with open(filename_initial_guess) as f:
        guess = yaml.safe_load(f)

    robot_node = env["robots"][0]

    goal =  np.array(robot_node["goal"])
    start =  np.array(robot_node["start"])

    cc = CollisionChecker()
    cc.load(filename_env)

    print("DONE")

    test_coll = True

    obs = []
    for obstacle in env["environment"]["obstacles"]:
        obs.append([obstacle["center"], obstacle["size"]])
    def extra_plot():
        plt.scatter( goal[0], goal[1], facecolors='none', edgecolors='b' )

        # for o in obs:
        #     circle = plt.Circle((o[0], o[1]), obs_radius, color='b', fill=False)
        #     ax = plt.gca()
        #     ax.add_patch(circle)

        for o in obs:
            ax = plt.gca()
            vis.draw_box_patch(ax, o[0], o[1], angle = 0, fill=False, color="b")
              # xy = np.asarray(center) - np.asarray(size) / 2
              # rect = Rectangle(xy, size[0], size[1], **kwargs)
              # t = matplotlib.transforms.Affine2D().rotate_around(
              #     center[0], center[1], angle)
              # rect.set_transform(t + ax.transData)
              # ax.add_patch(rect)
              # return rect


    if test_coll:
        ll = [ np.array([.3,.55+.25/2+.01]), np.zeros(2) , np.ones(2), np.array([.4,.4]), np.array([3.7,.3,np.pi/2.0]) ]
        for l in ll: 
            dist_tilde, p_obs, p_robot = cc.distance(l) # Float, Vector, Vector
            print("distance", l)
            print(dist_tilde)
            print(p_obs)
            print(p_robot)


        XX = np.linspace(0,3,50)
        YY = np.linspace(0,1,50)
        badx = []
        bady = []
        goodx = []
        goody = []
        for x in XX:
            for y in YY:
                l = np.array([x,y,0])
                dist_tilde, p_obs, p_robot = cc.distance(l) # Float, Vector, Vector
                if dist_tilde < 0 :
                    bady.append(y)
                    badx.append(x)
                if dist_tilde > 0 :
                    goody.append(y)
                    goodx.append(x)
        plt.scatter(badx,bady,c="r",marker=".")
        plt.scatter(goodx,goody,c="b",marker=".")
        extra_plot()
        plt.axis('equal')
        plt.show()



    # Seems fine! but only giving collision with respect to one objec!
    # Is this bad?


    # SOLVE THE PROBLEM
    aa = vis.Animation(filename_env)
    aa.show()

    ## GENERATE THE PROBLEM IN CROCODYLE

    T = len(guess["result"][0]["states"])
    problem, featss = croc_problem_from_env(env, T, cc)
    visualize = True

    #solve the problem

    solver = crocoddyl.SolverBoxFDDP

    
    ddp = solver(problem)

    # TODO: set the initial guess
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    # get xs and us
    X = guess["result"][0]["states"]
    U = guess["result"][0]["actions"]


    check_yaml  = False
    if check_yaml:
        d = {}
        d["result"] = []
        d["result"].append( { "states": X , "actions": U } )

        filename_out = "tmp_erase.yaml"
        with open(filename_out,'w') as f:
            yaml.dump(d, f, Dumper=yaml.CSafeDumper)

        with open(filename_out) as f:
            guess = yaml.safe_load(f)

        X = guess["result"][0]["states"]
        U = guess["result"][0]["actions"]


    Xl = [ np.array(x) for x in X ]
    Xl = [start] + Xl
    Ul = [ np.array(u) for u in U ]
    Ul = [Ul[0]] + Ul 

    print(len(Xl))
    print("xl")
    for i in Xl:
        print(i)
    print("ul")
    print(len(Ul))
    for i in Ul:
        print(i)
    crocoddyl.plotOCSolution(Xl, Ul, figIndex=1, show=False)
    plt.show()

    # done = ddp.solve(init_xs=Xl, init_us=Ul)
    # done = ddp.solve(init_xs=Xl, init_us=Ul)
    done = ddp.solve(init_xs=[], init_us=[])

    


    log = ddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2, show=False)

    plt.show()

    for x in ddp.xs: 
        plotUnicycle(x)
    plt.scatter(ddp.xs[-1][0],ddp.xs[-1][1],c="yellow",marker=".")

    goal =  np.array(robot_node["goal"])





    extra_plot()





    plt.show()

    penalty=False

    def plot_fun(xs):
        ax = plt.gca()
        for i in xs:
            plt.scatter(i[0],i[1],c="red",marker=".")
            vis.draw_box_patch(ax, [ i[0], i[1]] , [.5  ,.25] , angle=i[2], fill=False)
        plt.scatter(xs[-1][0],xs[-1][1],c="yellow",marker=".")
        extra_plot()
        plt.axis([0, 3, 0, 1.5])
        plt.axis('equal')
        plt.show()


    plot_fun(ddp.xs)



    if penalty:
        print("penalty")
        penalty_solver(ddp,Xl,Ul , featss, visualize,plot_fun)

    lagrangian = True
    if lagrangian:
        print("auglag")
        xs, us = auglag_solver(ddp,Xl,Ul, featss, np.zeros(2),visualize,plot_fun)
    # WORKING

    # NEXT: use the obstacles!!

    X = [ x.tolist() for x in ddp.xs]
    U = [ x.tolist() for x in ddp.us]

    d = {}
    d["result"] = [ { "states": X , "actions": U } ]

    filename_out = "tmp_erase.yaml"
    with open(filename_out,'w') as f:
        yaml.dump(d, f, Dumper=yaml.CSafeDumper)

    aa = vis.Animation(filename_env)
    aa = vis.Animation(filename_env, filename_out)
    aa.show()
    aa.save("vid.mp4",1)
    #     with open(filename_out) as f:
    #         guess = yaml.safe_load(f)



    # with open(filename_out,'w') as f:
    #     guess = yaml.safe_load(f)


    ## FIRST: add minimal TESTs for crocodyle

    ## DO it NOW :)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("initial_guess", help="file containing the initial_guess (e.g., from db-A*) (YAML)")
	args = parser.parse_args()

	run_croco(args.env, args.initial_guess)


if __name__ == '__main__':
	main()

