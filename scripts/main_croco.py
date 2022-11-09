import numpy as np
import yaml
import argparse
import tempfile
from pathlib import Path
import subprocess
import shutil
import time

from jax.config import config
config.update("jax_enable_x64", True)

import crocoddyl
import sys
import os
sys.path.append(os.getcwd())
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
# import faulthandler
# faulthandler.enable()
import checker
from utils_optimization import UtilsSolutionFile
obs_radius = .1


def test_col(cc):
    """
    example: tests collision checking for
    unycicle with yaw angle = 0
    """
    ll = [np.array([.3, .55 + .25 / 2 + .01]), np.zeros(2), np.ones(2),
          np.array([.4, .4]), np.array([3.7, .3, np.pi / 2.0])]
    for l in ll:
        dist_tilde, p_obs, p_robot = cc.distance(l)

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
    plt.axis('equal')
    plt.show()


def croc_problem_from_env(env, T, cc, free_time=False):

    robot_node = env["robots"][0]
    x0 = np.array(robot_node["start"])
    goal = np.array(robot_node["goal"])

    name = env["robots"][0]["type"]

    D = {"goal": goal, "x0": x0, "cc": cc, "T": T}
    D["free_time"] = free_time
    print("free_time: " , D["free_time"])

    cro = OCP_abstract()

    if name == "unicycle_first_order_0" or name == "unicycle_first_order_0_time":
        D["min_u"] = np.array([-.5, -.5])
        D["max_u"] = np.array([.5, .5])
        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        cro = OCP_unicycle_order1(**D)

    elif name == "unicycle_first_order_1":
        D["min_u"] = np.array([.25, -.5])
        D["max_u"] = np.array([.5, .5])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        cro = OCP_unicycle_order1(**D)

    elif name == "unicycle_first_order_2":
        D["min_u"] = np.array([.25, -.25])
        D["max_u"] = np.array([.5, .5])
        D["min_x"] = -np.inf * np.ones(3)
        D["max_x"] = np.inf * np.ones(3)
        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        cro = OCP_unicycle_order1(**D)

    elif name == "unicycle_second_order_0":
        D["min_u"] = np.array([-.25, -.25])
        D["max_u"] = np.array([.25, .25])
        D["min_x"] = np.concatenate(
            (-np.inf * np.ones(3), np.array([-.5, -.5])))
        D["max_x"] = np.concatenate((np.inf * np.ones(3), np.array([.5, .5])))
        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        cro = OCP_unicycle_order2(**D)

    elif name == "car_first_order_with_1_trailers_0":
        D["min_u"] = np.array([-.1, -np.pi / 3])
        D["max_u"] = np.array([.5, np.pi / 3])
        D["min_x"] = -np.inf * np.ones(4)
        D["max_x"] = np.inf * np.ones(4)

        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        cro = OCP_trailer(**D)

    elif name == "quadrotor_0":
        D["min_u"] = np.zeros(4)
        D["max_u"] = np.ones(4) * 12.0 / 1000.0 * 9.81

        max_v = 2
        max_omega = 4

        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])

        D["min_x"] = [-np.inf, -np.inf, -np.inf,
                      -1.001, -1.001, -1.001, -1.001,
                      -max_v, -max_v, -max_v,
                      -max_omega, -max_omega, -max_omega]
        D["max_x"] = [np.inf, np.inf, np.inf,
                      1.001, 1.001, 1.001, 1.001,
                      max_v, max_v, max_v,
                      max_omega, max_omega, max_omega]

        cro = OCP_quadrotor(**D, use_jit=True)

    return cro


def run_croco(filename_env, filename_initial_guess,
              filename_result, visualize, free_time=False):

    # TODO: add option to use trivial inital guess (either
    # u = udefault (zeros or hover for quadrotor) and fixed x0
    # or u = default and linear interpolation between x0 and goal for x.
    with open(filename_env) as f:
        env = yaml.safe_load(f)

    with open(filename_initial_guess) as f:
        guess = yaml.safe_load(f)

    robot_node = env["robots"][0]

    goal = np.array(robot_node["goal"])
    x0 = np.array(robot_node["start"])

    vis = None
    name = env["robots"][0]["type"]
    if name == "unicycle_first_order_0" or name == "unicycle_first_order_0_time":
        vis = vis1
    elif name == "unicycle_second_order_0":
        vis = vis2
    elif name == "car_first_order_with_1_trailers_0":
        vis = vis3

    cc = CollisionChecker()
    cc.load(filename_env)

    obs = []
    for obstacle in env["environment"]["obstacles"]:
        obs.append([obstacle["center"], obstacle["size"]])

    def extra_plot():
        plt.scatter(goal[0], goal[1], facecolors='none', edgecolors='b')

        for o in obs:
            ax = plt.gca()
            vis.draw_box_patch(ax, o[0], o[1], angle=0, fill=False, color="b")

    check_free_space_unicycle = False
    if check_free_space_unicycle:
        test_col(cc)

    print("showing intial guess")
    if visualize and vis is not None:
        print(
            f"filename_env {filename_env} filename_initial_guess {filename_initial_guess}")
        aa = vis.Animation(filename_env, filename_initial_guess)
        aa.show()
        plt.clf()
    print("showing intial guess -- done")

    # GENERATE THE PROBLEM IN CROCODYLE
    T = len(guess["result"][0]["states"]) - \
        1  # number of steps = num of actions
    problem = croc_problem_from_env(env, T, cc, free_time)
    solver = crocoddyl.SolverBoxFDDP

    ddp = solver(problem.problem)
    ddp.th_acceptNegStep = .3
    ddp.th_stop = 1e-2

    ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    X = guess["result"][0]["states"]
    U = guess["result"][0]["actions"]

    assert len(X) == len(U)+1
    Xl = [np.array(x) for x in X]
    Xl[0] = np.array(x0) # make sure intial state is equal to x0
    Ul = [np.array(u) for u in U]

    print(f"*** XL *** n= { len(Xl)}")
    _ = [print(x) for x in Xl]
    print(f"*** UL *** n ={len(Ul)}")
    _ = [print(u) for u in Ul]

    problem.recompute_init_guess(Xl, Ul)

    reguralize_wrt_init_guess = True
    if reguralize_wrt_init_guess:
        aa = 0
        for i in problem.featss:
            for jj in i.list_feats:
                if jj.fn.name == "regx":
                    jj.fn.ref = np.copy(Xl[aa])
                    aa += 1

    if visualize:
        crocoddyl.plotOCSolution(Xl, Ul, figIndex=1, show=False)
        plt.show()

    def plot_fn(xs, us):
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

    if vis is None:
        def plot_fn(xs, us):
            crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
            plt.show()

    auglag_solver(ddp, Xl, Ul, problem, np.zeros(3), visualize, plot_fn,
                  max_it_ddp=60)

    X = [x.tolist() for x in ddp.xs]
    U = [x.tolist() for x in ddp.us]

    problem.normalize(X, U)

    d = {}
    d["result"] = [{"states": X, "actions": U}]
    print("**solution**")
    print("X is:")
    _ = [ print(x) for x in X]
    print("U is:")
    _ = [ print(u) for u in U]

    with open(filename_result, 'w') as f:
        yaml.dump(d, f, Dumper=yaml.CSafeDumper)

    if visualize and vis is not None:
        print("showing final solution")
        bb = vis.Animation(filename_env, filename_result)
        bb.show()
        print("showing final solution -- done")

    result = checker.check(filename_env, filename_result)  # PASSING :)
    print(f"result is {result}")
    return result


def run_croco_with_T_scaling(
        filename_env,
        filename_initial_guess,
        filename_result,
        visualize=False,
        max_T=None):
    """
    """

    with tempfile.TemporaryDirectory() as tmpdirname:
        # p = Path(tmpdirname)
        p = Path("results/dbg")

        with open(filename_env) as f:
            env = yaml.safe_load(f)

            # convert environment YAML -> g

        robot_type = env["robots"][0]["type"]

        # hack
        utils_sol_file = UtilsSolutionFile(robot_type)
        utils_sol_file.load(filename_initial_guess)
        if utils_sol_file.T() == 0:
            return False
        filename_modified_guess = p / "guess.yaml"
        # for factor in [1.0]:
        for factor in [0.8, 1.0, 1.2]:
            T = int(utils_sol_file.T() * factor)
            if max_T is not None and T > max_T:
                return False
            print("Trying T ", T)
            # utils_sol_file.save_rescaled(filename_modified_guess, int(utils_sol_file.T() * 1.1))
            if factor == 1.0:
                result = run_croco(
                    filename_env,
                    filename_initial_guess,
                    filename_result,
                    visualize)
            else:
                utils_sol_file.save_rescaled(filename_modified_guess, T)
                result = run_croco(
                    filename_env,
                    filename_modified_guess,
                    filename_result,
                    visualize)
            # shutil.copyfile(filename_modified_guess, filename_result)
            # return True
            if result:
                return True
        return False


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
        action="store_true")
    parser.add_argument(
        "--freetime",
        help="visualize optimization variables and environment",
        action="store_true")
    args = parser.parse_args()
    result = run_croco(
        args.env,
        args.initial_guess,
        args.result,
        args.vis,
        free_time=args.freetime)
    if not result:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
