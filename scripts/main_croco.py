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

currentdir = os.path.dirname(
    os.path.abspath(
        inspect.getfile(
            inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import benchmark.unicycle_first_order_0.visualize as vis1
import benchmark.unicycle_second_order_0.visualize as vis2
import benchmark.car_first_order_with_1_trailers_0.visualize as vis3
import faulthandler
faulthandler.enable()
obs_radius = .1


# All the models are ready :)
# next: allow for different number of features!
# do the tests!
# integrate in the benchmark.

def croc_problem_from_env(env, T, cc):

    robot_node = env["robots"][0]
    x0 = np.array(robot_node["start"])
    goal = np.array(robot_node["goal"])
    name = env["robots"][0]["type"]

    D = {"goal": goal, "x0": x0, "cc": cc, "T": T}
    print(name)

    cro = CroAbstract()

    if name == "unicycle_first_order_0":
        D["min_u"] = np.array([-.5, -.5])
        D["max_u"] = np.array([.5, .5])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        cro = CroRobotUnicycleFirstOrder(**D)
    elif name == "unicycle_first_order_1":
        D["min_u"] = np.array([.25, -.5])
        D["max_u"] = np.array([.5, .5])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        cro = CroRobotUnicycleFirstOrder(**D)
    elif name == "unicycle_first_order_2":
        D["min_u"] = np.array([.25, -.25])
        D["max_u"] = np.array([.5, .5])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        cro = CroRobotUnicycleFirstOrder(**D)
    elif name == "unicycle_second_order_0":
        D["min_u"] = np.array([-.25, -.25])
        D["max_u"] = np.array([.25, .25])
        D["min_x"] = np.concatenate(
            (-np.inf * np.ones(3), np.array([-.5, -.5])))
        D["max_x"] = np.concatenate((np.inf * np.ones(3), np.array([.5, .5])))
        cro = CroRobotUnicycleSecondOrder(**D)
    elif name == "car_first_order_with_1_trailers_0":
        D["min_u"] = np.array([-.1, -np.pi / 3])
        D["max_u"] = np.array([.5, np.pi / 3])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        cro = CarFirstOrder1Trailers(**D)

    return cro


def run_croco(filename_env, filename_initial_guess,
              filename_result, visualize):

    with open(filename_env) as f:
        env = yaml.safe_load(f)

    with open(filename_initial_guess) as f:
        guess = yaml.safe_load(f)

    robot_node = env["robots"][0]

    goal = np.array(robot_node["goal"])
    start = np.array(robot_node["start"])

    vis = None
    name = env["robots"][0]["type"]
    if name == "unicycle_first_order_0":
        vis = vis1
    elif name == "unicycle_second_order_0":
        vis = vis2
    elif name == "car_first_order_with_1_trailers_0":
        vis = vis3

    cc = CollisionChecker()
    cc.load(filename_env)

    print("DONE")

    test_coll = False

    obs = []
    for obstacle in env["environment"]["obstacles"]:
        obs.append([obstacle["center"], obstacle["size"]])

    def extra_plot():
        plt.scatter(goal[0], goal[1], facecolors='none', edgecolors='b')

        # for o in obs:
        #     circle = plt.Circle((o[0], o[1]), obs_radius, color='b', fill=False)
        #     ax = plt.gca()
        #     ax.add_patch(circle)

        for o in obs:
            ax = plt.gca()
            vis.draw_box_patch(ax, o[0], o[1], angle=0, fill=False, color="b")
            # xy = np.asarray(center) - np.asarray(size) / 2
            # rect = Rectangle(xy, size[0], size[1], **kwargs)
            # t = matplotlib.transforms.Affine2D().rotate_around(
            #     center[0], center[1], angle)
            # rect.set_transform(t + ax.transData)
            # ax.add_patch(rect)
            # return rect

    if test_coll:
        ll = [np.array([.3, .55 + .25 / 2 + .01]), np.zeros(2), np.ones(2),
              np.array([.4, .4]), np.array([3.7, .3, np.pi / 2.0])]
        for l in ll:
            dist_tilde, p_obs, p_robot = cc.distance(
                l)  # Float, Vector, Vector
            print("distance", l)
            print(dist_tilde)
            print(p_obs)
            print(p_robot)

        XX = np.linspace(0, 3, 50)
        YY = np.linspace(0, 1, 50)
        badx = []
        bady = []
        goodx = []
        goody = []
        for x in XX:
            for y in YY:
                l = np.array([x, y, 0])
                dist_tilde, p_obs, p_robot = cc.distance(
                    l)  # Float, Vector, Vector
                if dist_tilde < 0:
                    bady.append(y)
                    badx.append(x)
                if dist_tilde > 0:
                    goody.append(y)
                    goodx.append(x)
        plt.scatter(badx, bady, c="r", marker=".")
        plt.scatter(goodx, goody, c="b", marker=".")
        extra_plot()
        plt.axis('equal')
        plt.show()

    # Seems fine! but only giving collision with respect to one objec!
    # Is this bad?

    # SOLVE THE PROBLEM

    print("show anime")
    if visualize:
        aa = vis.Animation(filename_env, filename_initial_guess)
        aa.show()
        print("end show anime")

    # GENERATE THE PROBLEM IN CROCODYLE

    T = len(guess["result"][0]["states"])
    # TODO: lets createa a initial guess by adding noise to solution
    # T = 60
    problem = croc_problem_from_env(env, T, cc)

    # solve the problem

    solver = crocoddyl.SolverBoxFDDP

    ddp = solver(problem.problem)

    # TODO: set the initial guess
    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    # get xs and us
    X = guess["result"][0]["states"]
    U = guess["result"][0]["actions"]

    check_yaml = False
    if check_yaml:
        d = {}
        d["result"] = []
        d["result"].append({"states": X, "actions": U})

        filename_out = "tmp_erase.yaml"
        with open(filename_out, 'w') as f:
            yaml.dump(d, f, Dumper=yaml.CSafeDumper)

        with open(filename_out) as f:
            guess = yaml.safe_load(f)

        X = guess["result"][0]["states"]
        U = guess["result"][0]["actions"]

    Xl = [np.array(x) for x in X]
    Xl = [start] + Xl
    Ul = [np.array(u) for u in U]
    Ul = [Ul[0]] + Ul

    print(len(Xl))
    print("xl")
    for i in Xl:
        print(i)
    print("ul")
    print(len(Ul))
    for i in Ul:
        print(i)

    # change Xl
    Xl, Ul = problem.recompute_init_guess(Xl, Ul)

    if visualize:
        crocoddyl.plotOCSolution(Xl, Ul, figIndex=1, show=False)
        plt.show()

    # done = ddp.solve(init_xs=Xl, init_us=Ul)
    # done = ddp.solve(init_xs=Xl, init_us=Ul)
    # done = ddp.solve(init_xs=[], init_us=[])

    # log = ddp.getCallbacks()[0]
    # crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2, show=False)

    # plt.show()

    # for x in ddp.xs:
    #     plotUnicycle(x)
    # plt.scatter(ddp.xs[-1][0],ddp.xs[-1][1],c="yellow",marker=".")

    # goal =  np.array(robot_node["goal"])

    # extra_plot()

    # plt.show()

    penalty = False

    def plot_fun(xs, us):
        ax = plt.gca()
        for i in xs:
            plt.scatter(i[0], i[1], c="red", marker=".")
            vis.draw_box_patch(
                ax, [i[0], i[1]], [.5, .25], angle=i[2], fill=False)
        plt.scatter(xs[-1][0], xs[-1][1], c="yellow", marker=".")
        extra_plot()
        plt.axis([0, 3, 0, 1.5])
        plt.axis('equal')
        plt.show()
        crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
        plt.show()

    # plot_fun(ddp.xs)
    if penalty:
        print("penalty")
        penalty_solver(ddp, Xl, Ul, problem, visualize, plot_fun)

    lagrangian = True
    if lagrangian:
        print("auglag")
        auglag_solver(ddp, Xl, Ul, problem, np.zeros(2), visualize, plot_fun)
    # WORKING

    # NEXT: use the obstacles!!

    X = [x.tolist() for x in ddp.xs]
    U = [x.tolist() for x in ddp.us]

    d = {}
    d["result"] = [{"states": X, "actions": U}]

    filename_out = "tmp_erase.yaml"
    with open(filename_out, 'w') as f:
        yaml.dump(d, f, Dumper=yaml.CSafeDumper)

    if visualize:
        aa = vis.Animation(filename_env, filename_out)
        aa.show()
        aa.save("vid.mp4", 1)
    #     with open(filename_out) as f:
    #         guess = yaml.safe_load(f)

    # with open(filename_out,'w') as f:
    #     guess = yaml.safe_load(f)

    # FIRST: add minimal TESTs for crocodyle

    # DO it NOW :)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="file containing the environment (YAML)")
    parser.add_argument(
        "initial_guess",
        help="file containing the initial_guess (e.g., from db-A*) (YAML)")
    parser.add_argument(
        "result",
        help="file containing the optimization result (YAML)")
    parser.add_argument(
        "--vis",
        help="visualize optimization variables and environment",
        default=False)
    args = parser.parse_args()
    run_croco(args.env, args.initial_guess, args.result, args.vis)


if __name__ == '__main__':
    main()
