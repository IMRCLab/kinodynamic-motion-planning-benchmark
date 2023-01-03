import numpy as np
import yaml
import argparse
import tempfile
from pathlib import Path
import subprocess
import shutil
import time

from datetime import datetime
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

# I want a jit version of collisions!


def croc_problem_from_env(D):

    name = D["name"]

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

    elif name == "robot_double_integrator_n":
        D["min_u"] = -np.inf * np.ones(2)
        D["max_u"] = +np.inf * np.ones(2)
        D["min_x"] = -np.inf * np.ones(4)
        D["max_x"] = +np.inf * np.ones(4)
        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        cro = OCP_double_integrator_n(**D)

    elif name == "robot_single_integrator_n":
        D["min_u"] = -np.inf * np.ones(2)
        D["max_u"] = +np.inf * np.ones(2)
        D["min_x"] = -np.inf * np.ones(2)
        D["max_x"] = +np.inf * np.ones(2)
        if D["free_time"]:
            D["min_u"] = np.append(D["min_u"], [.5])
            D["max_u"] = np.append(D["max_u"], [2])
        cro = OCP_Single_integrator_n(**D)

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


def sample_at_dt(X, U):

    # the third component of U is dt scaling

    dts = np.array([0] + [u[-1] for u in U])
    dts_accumulated = np.cumsum(dts)
    print("dts", dts)
    print("dts_accumulated", dts_accumulated)
    print("total time", dts_accumulated[-1])

    query_t = np.arange(np.ceil(dts_accumulated[-1]) + 1)

    Xnew = []
    Unew = []

    for i in range(len(X[0])):
        x_a = np.array([x[i] for x in X])
        x_a_new = np.interp(query_t, dts_accumulated, x_a)

        plt.plot(dts_accumulated, x_a, 'o-', label="dat")
        plt.plot(query_t, x_a_new, 'o-', label="query")
        plt.legend()
        plt.show()

        Xnew.append(x_a_new)

    for i in range(len(U[0]) - 1):
        u_a = np.array([u[i] for u in U])
        u_a_new = np.interp(query_t[:-1], dts_accumulated[:-1], u_a)

        plt.plot(dts_accumulated[:-1], u_a, 'o-', label="dat")
        plt.plot(query_t[:-1], u_a_new, 'o-', label="query")
        plt.legend()
        plt.show()

        Unew.append(u_a_new)

    _Xnew = np.stack(Xnew)  # num_coordinates x num_datapoints
    _Unew = np.stack(Unew)  # num_coordinates x num_datapoints

    print("plotting all")
    for i, x in enumerate(_Xnew):
        plt.plot(query_t, x, 'o-', label=f"x{i}")

    for i, u in enumerate(_Unew):
        plt.plot(query_t[:-1], u, 'o-', label=f"u{i}")

    plt.legend()
    plt.show()

    _Xnew = np.transpose(_Xnew)  # num_datapoints x num_coordinates
    _Unew = np.transpose(_Unew)  # num_datapoints x num_coordinates

    return _Xnew, _Unew


def run_croco(filename_env, filename_initial_guess,
              filename_result, visualize, free_time=False,
              horizon=False):

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

    X = guess["result"][0]["states"]
    U = guess["result"][0]["actions"]

    assert len(X) == len(U) + 1
    Xl = [np.array(x) for x in X]
    Xl[0] = np.array(x0)  # make sure intial state is equal to x0
    Ul = [np.array(u) for u in U]

    print(f"*** XL *** n= { len(Xl)}")
    _ = [print(x) for x in Xl]
    print(f"*** UL *** n ={len(Ul)}")
    _ = [print(u) for u in Ul]

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

    if visualize:
        crocoddyl.plotOCSolution(Xl, Ul, figIndex=1, show=False)
        plt.show()

    joint_optimization = not horizon
    T = len(guess["result"][0]["states"]) - \
        1  # number of steps = num of actions
    name = env["robots"][0]["type"]

    if joint_optimization:

        # GENERATE THE PROBLEM IN CROCODYLE

        print(len(Xl))
        print(len(Ul))

        robot_node = env["robots"][0]
        x0 = np.array(robot_node["start"])
        goal = np.array(robot_node["goal"])
        D = {"goal": goal, "x0": x0, "cc": cc, "T": T, "free_time": free_time,
             "name": name}
        if name == "quadrotor_0":
            D["col"] = False
        else:
            D["col"] = True

        problem = croc_problem_from_env(D)
        problem.recompute_init_guess(Xl, Ul)
        solver = crocoddyl.SolverBoxFDDP
        ddp = solver(problem.problem)
        ddp.th_acceptNegStep = .3
        ddp.th_stop = 1e-2
        ddp.setCallbacks([crocoddyl.CallbackLogger(),
                          crocoddyl.CallbackVerbose()])

        reguralize_wrt_init_guess = True
        if reguralize_wrt_init_guess:
            aa = 0
            for i in problem.featss:
                for jj in i.list_feats:
                    if jj.fn.name == "regx":
                        jj.fn.ref = np.copy(Xl[aa])
                        aa += 1

        auglag_solver(
            ddp,
            Xl,
            Ul,
            problem,
            np.zeros(3),
            visualize,
            plot_fn,
            max_it_ddp=20,
            max_it=5)

        X = [x.tolist() for x in ddp.xs]
        U = [x.tolist() for x in ddp.us]

        reoptimize_fixed_time = False
        if reoptimize_fixed_time:
            _Xnew, _Unew = sample_at_dt(X, U)

            xs_rollout = []

            xs_rollout.append(x0)
            x = np.copy(x0)
            model_ = problem.running_model[0]
            model_.robot.free_time = False
            for u in _Unew:
                xnew = model_.robot.step(x, u)
                x = np.copy(xnew)
                xs_rollout.append(x)

            plot_rollouts = False
            if plot_rollouts:

                print("plotting rollouts")
                for i in range(len(_Unew[0])):
                    us = [u[i] for u in _Unew]
                    plt.plot(us, 'o-', label=f"u{i}")

                for i in range(len(xs_rollout[0])):
                    xs = [x[i] for x in xs_rollout]
                    plt.plot(xs, 'o-', label=f"x{i}-rollout")

                for i in range(len(_Xnew[0])):
                    xs = [x[i] for x in _Xnew]
                    plt.plot(xs, 'o-', label=f"x{i}")

                plt.legend()
                plt.show()

            Xl = [np.array(x) for x in _Xnew.tolist()]
            Ul = [np.array(u) for u in _Unew.tolist()]

            D["free_time"] = False
            D["T"] = _Unew.shape[0]
            problem = croc_problem_from_env(D)
            problem.recompute_init_guess(Xl, Ul)
            solver = crocoddyl.SolverBoxFDDP
            ddp = solver(problem.problem)
            ddp.th_acceptNegStep = .3
            ddp.th_stop = 1e-2
            ddp.setCallbacks([crocoddyl.CallbackLogger(),
                              crocoddyl.CallbackVerbose()])

            reguralize_wrt_init_guess = True
            if reguralize_wrt_init_guess:
                aa = 0
                for i in problem.featss:
                    for jj in i.list_feats:
                        if jj.fn.name == "regx":
                            jj.fn.ref = np.copy(Xl[aa])
                            aa += 1

            auglag_solver(
                ddp,
                Xl,
                Ul,
                problem,
                np.zeros(3),
                visualize,
                plot_fn,
                max_it_ddp=20,
                max_it=5)
            X = [x.tolist() for x in ddp.xs]
            U = [x.tolist() for x in ddp.us]

        problem.normalize(X, U)

    else:

        # TODO: write easy loop
        # num_steps = 2 # add code here

        # Tnew = int(T/2)
        horizon_opti = 50
        fix_opti = 10
        assert fix_opti <= horizon_opti
        finished = False
        x0_ = np.copy(np.array(robot_node["start"]))
        finished = False
        last = False
        ii = 0

        X_ = []
        U_ = []

        # IDEA: OPTIMIZE N actions, TAKE only K actions.

        first = True
        Xdraw = []
        while (not finished):
            # first index

            first_index = ii * fix_opti
            last_index = first_index + horizon_opti

            reuse = True
            if not reuse or first:
                Xlnew = [x0_] + Xl[first_index + 1:last_index + 1]

                # Xlnew = [x0_] + Xl[ii*Tnew+1:(ii+1)*Tnew+1]
                # Ulnew = Ul[ii*Tnew:(ii+1)*Tnew]
                Ulnew = Ul[first_index:last_index]
                first = False
            else:
                print(x0_)
                offset = len(Xl_prev[fix_opti + 1:])
                Xlnew = [x0_] + Xl_prev[fix_opti + 1:] + \
                    Xl[first_index + 1 + offset: last_index + 1]

                # Xlnew = [x0_] + Xl[ii*Tnew+1:(ii+1)*Tnew+1]
                # Ulnew = Ul[ii*Tnew:(ii+1)*Tnew]

                Ulnew = Ul_prev[fix_opti:] + Ul[first_index +
                                                horizon_opti - fix_opti:last_index]

            last = last_index + 1 > len(Xl)
            print("x0_", x0_)
            print(f"first_index,last_index, {first_index}, {last_index}")
            print(f"len X, U,{len(Xlnew)}, {len(Ulnew)}")
            print(Xlnew)
            print(Ulnew)
            print("last", last)

            if last:
                goal = np.array(robot_node["goal"])
            else:
                goal = Xlnew[-1]

            D = {
                "goal": goal,
                "x0": x0_,
                "cc": cc,
                "T": len(Ulnew),
                "free_time": free_time,
                "name": name,
                "goal_constraint": last}

            problem = croc_problem_from_env(D)
            problem.recompute_init_guess(Xlnew, Ulnew)
            solver = crocoddyl.SolverBoxFDDP
            ddp = solver(problem.problem)
            ddp.th_acceptNegStep = .3
            ddp.th_stop = 1e-2
            ddp.setCallbacks([crocoddyl.CallbackLogger(),
                              crocoddyl.CallbackVerbose()])

            reguralize_wrt_init_guess = True
            if reguralize_wrt_init_guess:
                aa = 0
                for i in problem.featss:
                    for jj in i.list_feats:
                        if jj.fn.name == "regx":
                            jj.fn.ref = np.copy(Xlnew[aa])
                            aa += 1

            auglag_solver(ddp, Xlnew, Ulnew, problem, np.zeros(3), visualize,
                          plot_fn, max_it_ddp=60)

            finished = last_index > len(Ul)

            if finished:
                X_ += [x.tolist() for x in ddp.xs[1:]]
                U_ += [u.tolist() for u in ddp.us]

            else:
                if ii == 0:
                    print("fixing")
                    print([x.tolist() for x in ddp.xs[0:fix_opti + 1]])
                    X_ += [x.tolist() for x in ddp.xs[0:fix_opti + 1]]
                else:
                    print("fixing")
                    print([x.tolist() for x in ddp.xs[1:fix_opti + 1]])
                    X_ += [x.tolist() for x in ddp.xs[1:fix_opti + 1]]

                U_ += [u.tolist() for u in ddp.us[:fix_opti]]

            print("X_", X_)
            print("U_", U_)

            Xl_prev = [x for x in ddp.xs]
            if free_time:
                Ul_prev = [u[:-1] for u in ddp.us]
            else:
                Ul_prev = [u for u in ddp.us]

            x0_ = ddp.xs[fix_opti]
            ii += 1

            Xdraw.append([x.tolist() for x in ddp.xs])

            for xx in Xdraw:
                xs = [x[0] for x in xx]
                ys = [x[1] for x in xx]
                plt.plot(xs, ys, 'o-')
            plt.show()

        X = X_
        U = U_

    d = {}
    d["result"] = [{"states": X, "actions": U}]
    print("**solution**")
    print("X is:")
    _ = [print(x) for x in X]
    print("U is:")
    _ = [print(u) for u in U]

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
    parser.add_argument("--freetime", action="store_true")
    parser.add_argument("--horizon", action="store_true")

    args = parser.parse_args()
    result = run_croco(
        args.env,
        args.initial_guess,
        args.result,
        args.vis,
        free_time=args.freetime,
        horizon=args.horizon)
    if not result:
        sys.exit(1)
    else:
        sys.exit(0)


def data_generation(free_time=False):
    """
    Still under dev. Working for two hardcoded cases
    """

    T = 10

    name = "unicycle_second_order_0"

    name = "robot_double_integrator_n"

    num_data = 20
    data_out = []
    np.random.seed()
    for i in range(num_data):

        if name == "unicycle_second_order_0":
            def _plot_fn(xs, us):
                ax = plt.gca()
                for i in xs:
                    plt.scatter(i[0], i[1], c="red", marker=".")
                    vis2.draw_box_patch(
                        ax, [i[0], i[1]], [.5, .25], angle=i[2], fill=False)
                plt.scatter(xs[-1][0], xs[-1][1], c="yellow", marker=".")
                plt.axis([0, 3, 0, 1.5])
                plt.axis('equal')
                plt.show()
                crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
                plt.show()

            x0 = np.zeros(5)
            lb = np.array([-.25, -.25])
            ub = np.array([.25, .25])

            us_rollout = [lb + 1.0 *
                          np.random.randint(0, 2, 2) * (ub - lb) for _ in range(T)]
            xs_rollout = []

            x = x0.copy()
            xs_rollout.append(x0)
            robot = RobotUnicycleSecondOrder(semi_implicit=False)
            for u in us_rollout:
                xnew = robot.step(x, u)
                xs_rollout.append(xnew)
                x = np.copy(xnew)
            # _plot_fn(xs_rollout,us_rollout)

            # last state
            goal = np.array(xs_rollout[-1].tolist())

            Xl = [np.array(x.tolist()) for x in xs_rollout]
            Ul = [np.array(u.tolist()) for u in us_rollout]

        else:
            def _plot_fn(xs, us):
                crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
                plt.show()

            ndim = 2
            x0 = np.zeros(2 * ndim)
            lb = -np.inf
            ub = +np.inf
            goal = np.concatenate((np.random.rand(ndim), np.zeros(ndim)))
            goal = np.concatenate((np.ones(ndim), np.zeros(ndim)))

            Xl = [np.zeros(2 * ndim) for _ in range(T + 1)]
            Ul = [np.zeros(ndim) for _ in range(T)]

        # goal = np.zeros(5)
        print("x0:", x0)
        print("goal:", goal)

        D = {
            "goal": goal,
            "x0": x0,
            "cc": None,
            "T": T,
            "free_time": free_time,
            "name": name,
            "use_jit": False,
            "weight_regx": 0.}

        if name == "unicycle_first_order_0" or name == "unicycle_first_order_0_time":
            vis = vis1
        elif name == "unicycle_second_order_0":
            vis = vis2
        elif name == "car_first_order_with_1_trailers_0":
            vis = vis3

        problem = croc_problem_from_env(D)
        problem.recompute_init_guess(Xl, Ul)
        solver = crocoddyl.SolverBoxFDDP
        ddp = solver(problem.problem)
        ddp.th_acceptNegStep = .3
        ddp.th_stop = 1e-2
        ddp.setCallbacks([crocoddyl.CallbackLogger(),
                          crocoddyl.CallbackVerbose()])

        reguralize_wrt_init_guess = False
        if reguralize_wrt_init_guess:
            aa = 0
            for i in problem.featss:
                for jj in i.list_feats:
                    if jj.fn.name == "regx":
                        jj.fn.ref = np.copy(Xl[aa])
                        aa += 1

        xs, us = auglag_solver(ddp, Xl, Ul, problem, np.zeros(
            3), False, plot_fn=None, max_it_ddp=20, max_it=5)

        print("checking the solution")
        # _plot_fn(xs,us)
        c = 0
        obj = 0
        unone = None
        for i in range(T + 1):
            xx = xs[i]
            uu = us[i] if i < T else unone
            for f in problem.featss[i].list_feats:
                if f.tt == OT.auglag:
                    c += np.sum(np.abs(f.fn(xx, uu)))
                if f.tt == OT.cost:
                    r = f.fn(xx, uu)
                    obj += .5 * np.dot(r, r)
        print(f"c:{c} , obj:{obj}")

        X = [x.tolist() for x in ddp.xs]
        U = [x.tolist() for x in ddp.us]

        c_th = .01
        if c < c_th:
            data_out.append((X, U))

    print("num trajectories", len(data_out))

    now = datetime.now()  # current date and time

    date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
    print("date and time:", date_time)

    with open("data_rl_" + date_time + ".yaml", "w") as f:
        yaml.dump(data_out, f, Dumper=yaml.CSafeDumper)

    # TODO: Allow for free time and resampling


if __name__ == '__main__':
    main()
    # data_generation()
