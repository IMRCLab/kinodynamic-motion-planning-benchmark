import subprocess
import yaml
from typing import List
import numpy as np
import pathlib
from typing import Tuple

from multiprocessing import Pool
import multiprocessing
import uuid
from pathlib import Path
import csv
import pandas

import argparse


all_problems = [
    "unicycle_first_order_0/parallelpark_0",
    "unicycle_first_order_0/bugtrap_0",
    "unicycle_first_order_0/kink_0",
    "unicycle_first_order_1/kink_0",
    "unicycle_first_order_2/wall_0",
    "unicycle_second_order_0/parallelpark_0",
    "unicycle_second_order_0/bugtrap_0",
    "unicycle_second_order_0/kink_0",
    "car_first_order_with_1_trailers_0/bugtrap_0",
    "car_first_order_with_1_trailers_0/parallelpark_0",
    "car_first_order_with_1_trailers_0/kink_0",
    "quad2d/empty_0",
    "quad2d/quad_obs_column",
    "quad2d/quad2d_recovery_wo_obs",
    # CONTINUE QUADROTOR_0
    "quadrotor_0/empty_0_easy",
    "quadrotor_0/recovery",
    "quadrotor_0/quad_one_obs",
    "acrobot/swing_up_empty",
    "acrobot/swing_down_easy",
    "acrobot/swing_down",
    "car_first_order_with_1_trailers_0/easy_0"]


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode")
parser.add_argument("-bc", "--bench_cfg")
parser.add_argument("-d", "--dynamics")
parser.add_argument("-f", "--file_in")

args = parser.parse_args()

print(args.__dict__)

mode = args.mode
bench_cfg = args.bench_cfg
file_in = args.file_in
dynamics = args.dynamics

MAX_TIME_PLOTS = 40


do_compare = False
do_benchmark = False
do_debug = False
do_vis_primitives = False


if mode == "compare":
    do_compare = True

elif mode == "bench":
    do_benchmark = True

elif mode == "debug":
    do_debug = True

elif mode == "vis":
    do_vis_primitives = True


import sys
sys.path.append('..')

import viewer.viewer_cli as viewer_cli
import matplotlib.pyplot as plt

from datetime import datetime


def plot_stats(motions: list, robot_model: str, filename: str):
    all_actions = []
    all_states = []
    all_start_states = []
    all_end_states = []
    for m in motions:
        all_actions.extend(m["actions"])
        all_states.extend(m["states"])
        all_start_states.append(m["start"])
        all_end_states.append(m["goal"])

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    all_actions = np.array(all_actions)
    all_states = np.array(all_states)
    all_start_states = np.array(all_start_states)
    all_end_states = np.array(all_end_states)

    print(f"writing pdf to {filename}")
    pp = PdfPages(filename)

    # r = robots.create_robot(robot_type)
    # base_path = "../models/"
    # from motionplanningutils import robot_factory
    # r = robot_factory(base_path + robot_model + ".yaml")

    def get_desc(name: str) -> Tuple[List[str], List[str]]:
        if name.startswith("unicycle1"):
            return (["x", "y", "theta"], ["v", "w"])
        elif name.startswith("unicycle2"):
            return (["x", "y", "theta", "v", "w"], ["a", "aa"])
        else:
            raise NotImplementedError(f"unknown {name}")

    x_desc, u_desc = get_desc(robot_model)

    fig, ax = plt.subplots()
    ax.set_title("Num steps: ")
    ax.hist([len(m["actions"]) for m in motions])
    pp.savefig(fig)
    plt.close(fig)

    for k, a in enumerate(u_desc):
        fig, ax = plt.subplots()
        ax.set_title("Action: " + a)
        ax.hist(all_actions[:, k])
        pp.savefig(fig)
        plt.close(fig)

    for k, s in enumerate(x_desc):
        fig, ax = plt.subplots()
        ax.set_title("state: " + s)
        ax.hist(all_states[:, k])
        pp.savefig(fig)
        plt.close(fig)

    for k, s in enumerate(x_desc):
        fig, ax = plt.subplots()
        ax.set_title("start state: " + s)
        ax.hist(all_start_states[:, k])
        pp.savefig(fig)
        plt.close(fig)

    for k, s in enumerate(x_desc):
        fig, ax = plt.subplots()
        ax.set_title("end state: " + s)
        ax.hist(all_end_states[:, k])
        pp.savefig(fig)
        plt.close(fig)

    pp.close()


def create_empty_env(start: List, goal: List, robot_model: str):
    if robot_model.startswith("quadrotor_0"):
        env_max = [2., 2., 2.]
        env_min = [-2., -2., -2.]
    elif robot_model.startswith("acrobot"):
        env_min = [-2.5, -2.5]
        env_max = [2.5, 2.5]
    else:
        env_min = [-2, -2]
        env_max = [2, 2]

    env = {
        "environment": {
            "min": env_min,
            "max": env_max,
            "obstacles": []
        },
        "robots": [{
            "type": robot_model,
            "start": start,
            "goal": goal,
        }]
    }
    return env


color_map = {
    "sst_v0": "blue",
    "geo_v0": "green",
    "geo_v1": "red",
    "idbastar_v0": "cyan",
    "idbastar_v0_heu1": "yellow",
    "idbastar_tmp": "deeppink",
    "idbastar_v0_mpcc": "black",
    "idbastar_v0_search": "orange"
}


def get_config(alg: str, dynamics: str, problem: str):

    base_path_algs = "../algs/"
    file = base_path_algs + alg + ".yaml"

    print(f"loading {file}")
    with open(file, "r") as f:
        data = yaml.safe_load(f)
    print("data is:\n", data)

    cfg = {}
    if "reference" in data:
        # I have to load another configuration as default
        print("there is a reference!")
        reference = data.get("reference")
        print(f"reference {reference}")
        cfg = get_config(reference, dynamics, problem)
        print("reference cfg is: ", cfg)

    print(f"dynamics {dynamics}")
    print(f"problem is {problem}")

    cfg_default = data.get("default")
    cfg_dynamics = data.get(dynamics, {}).get("default", {})
    __problem = Path(problem).stem

    cfg_problem = data.get(dynamics, {}).get(__problem, {})

    print(f"cfg_default {cfg_default}")
    print(f"cfg_dynamics {cfg_dynamics}")
    print(f"cfg_problem {cfg_problem}")

    for k, v in cfg_default.items():
        cfg[k] = v

    for k, v in cfg_dynamics.items():
        cfg[k] = v

    for k, v in cfg_problem.items():
        cfg[k] = v

    return cfg


def solve_problem_with_alg(
        problem: str,
        alg: str,
        out: str,
        timelimit: float) -> List[str]:
    # load params

    # get the dynamics
    with open(problem, 'r') as f:
        data_problem = yaml.safe_load(f)

    print(f"data_problem {data_problem}")

    dynamics = data_problem["robots"][0]["type"]
    print(f"dynamics {dynamics}")

    cfg = get_config(alg, dynamics, problem)

    print("merged cfg\n", cfg)

    cfg_out = out + ".cfg.yaml"

    with open(cfg_out, "w") as f:
        yaml.dump(cfg, f)

    if alg.startswith("sst"):
        cmd = [
            "./main_ompl",
            "--env_file",
            problem,
            "--results_file",
            out, "--timelimit", str(timelimit), "--cfg", cfg_out]
    elif alg.startswith("geo"):
        cmd = [
            "./main_ompl_geo",
            "--env_file",
            problem,
            "--results_file",
            out, "--timelimit", str(timelimit), "--cfg", cfg_out]

    elif alg.startswith("idbastar"):
        # motions_file = get_motions_file(problem)
        cmd = [
            "./main_idbastar",
            "--env_file",
            problem,
            "--results_file",
            out,
            "--timelimit",
            str(timelimit),
            "--cfg",
            cfg_out]

    else:
        raise NotImplementedError()

    return cmd
    # print("**\n**\nRUNNING cpp\n**\n", *cmd, "\n", sep=" ")
    # subprocess.run(cmd)
    # print("**\n**\nDONE RUNNING cpp\n**\n")


class Experiment():
    def __init__(self, path, problem, alg):
        self.path = path
        self.problem = problem
        self.alg = alg

    def __str__(self):
        print(f"path:{self.path} problem:{self.problem} alg:{self.alg}")


redirect_output = True


def run_cmd(cmd: str):
    print("**\n**\nRUNNING cpp\n**\n", *cmd, "\n", sep=" ")

    if redirect_output:
        id = str(uuid.uuid4())[:7]
        stdout_name = f"stdout-{id}.log"
        stderr_name = f"stderr-{id}.log"
        print(
            *
            cmd,
            f"---- stderr_name {stderr_name}   stdout_name {stdout_name} ----")
        f_stdout = open(stdout_name, 'w')
        f_stderr = open(stderr_name, 'w')
        subprocess.run(cmd, stdout=f_stdout, stderr=f_stderr)
        f_stdout.close()
        f_stderr.close()
    else:
        subprocess.run(cmd)
    print("**\n**\nDONE RUNNING cpp\n**\n")


def benchmark(bench_cfg: str):

    with open(bench_cfg) as f:
        data = yaml.safe_load(f)

    print("bench cfg")
    print(data)
    problems = data["problems"]
    algs = data["algs"]
    trials = data["trials"]
    timelimit = data["timelimit"]
    n_cores = data["n_cores"]
    if n_cores == -1:
        n_cores = int(multiprocessing.cpu_count() / 2)

    print(f"problems {problems}")
    print(f"algs {algs}")

    base_path_problem = "../benchmark/"
    folder_results = "../results_new/"

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
    cmds = []
    paths = []

    experiments = []
    for problem in problems:
        for alg in algs:
            path = folder_results + problem + "/" + alg + "/" + date_time
            paths.append(path)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            experiments.append(Experiment(path=path, problem=problem, alg=alg))
            for i in range(trials):
                out = path + f"/run_{i}_out.yaml"
                cmd = solve_problem_with_alg(
                    base_path_problem + problem + ".yaml", alg, out, timelimit)
                cmds.append(cmd)
    print("commands are: ")
    for i, cmd in enumerate(cmds):
        print(i, cmd)

    print(f"Start a pool with {n_cores}:")
    with Pool(n_cores) as p:
        p.map(run_cmd, cmds)
    print("Pool is DONE")

    fileouts = []
    for experiment in experiments:
        print("experiment")

        fileout, _ = analyze_runs(
            experiment.path,
            experiment.problem,
            experiment.alg,
            visualize=False)
        fileouts.append(fileout)

    compare(fileouts, False)

    # basic analysisi of the results?


def compare(
        files: List[str],
        interactive: bool = False):

    print("calling compare:")
    print(f"files {files}")

    # load
    datas = []
    for file in files:
        print("loading ", file)
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
            datas.append(data)
    print("datas\n", datas)

    # print("artificially adding the problem...")
    # for data in datas:
    #     data["problem"] = "unicycle_first_order_0/bugtrap_0"
    # print(datas)

    # organize by problem

    D = {}
    fields = [
        "problem",
        "alg",
        "cost_at_01",
        "cost_at_05",
        "cost_at_10",
        "cost_at_20",
        "cost_first_solution",
        "time_first_solution",
        "last_cost"]
    fields.sort()

    reduced_data = []
    for d in datas:
        # take only some fields
        dd = {}
        for field in fields:
            dd[field] = d[field]
        reduced_data.append(dd)

    # save as csv file

    # id = str(uuid.uuid4())[:7]
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
    filename_csv = f"../results_new/summary/summary_{date_time}.csv"

    # log file

    filename_csv_log = filename_csv + ".log"

    with open(filename_csv_log, "w") as f:
        dd = {"input": files, "output": filename_csv}
        yaml.dump(dd, f)

        # input

        # f.writelines(files)
        # f.write("---")
        # f.write(filename_csv)

    print("saving reduced data to", filename_csv)
    with open(filename_csv, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for dictionary in reduced_data:
            writer.writerow(dictionary.values())

    import shutil
    shutil.copy(filename_csv, '/tmp/tmp_reduced_data.csv')

    create_latex_table(filename_csv)

    # check
    print("checking data")
    with open(filename_csv, 'r') as myFile:
        print(myFile.read())

    for problem in all_problems:
        print(f"problem {problem}")
        # check if data belongs to this problem
        _datas = []
        for data in datas:
            print("**")
            print(data["problem"])
            print(problem)
            print("**")
            print("**")
            if data["problem"] == problem:
                print("match!")
                _datas.append(data)
        D[problem] = _datas
    print("D", D)

    # now print the data!

    from matplotlib.backends.backend_pdf import PdfPages

    filename_pdf = f"../results_new/plots/plot_{date_time}.pdf"

    print(f"writing pdf to {filename_pdf}")

    filename_log_pdf = filename_pdf + ".log"

    with open(filename_log_pdf, "w") as f:
        dd = {"input": files, "output": filename_pdf}
        yaml.dump(dd, f)

    pp = PdfPages(filename_pdf)

    for problem in all_problems:

        if D[problem]:
            fig, ax = plt.subplots(2, 1, sharex=True)
            fig.suptitle(problem)
            for d in D[problem]:
                print("d", d)
                alg = d["alg"]
                times = d["times"]
                success = d["success"]
                cost_mean = d["cost_mean"]
                cost_std = d["cost_std"]
                color = color_map[alg]

                ax[0].plot(times, cost_mean, color=color, label=alg)
                ax[0].fill_between(times,
                                   np.array(cost_mean) + np.array(cost_std),
                                   np.array(cost_mean) - np.array(cost_std),
                                   facecolor=color,
                                   alpha=0.5)
                ax[1].plot(times, success, color=color, label=alg)

                ax[0].set_xscale('log')
                ax[1].set_xscale('log')

            ax[1].legend()
            ax[0].set_ylabel("cost")
            ax[1].set_xlabel("time [s]")
            ax[1].set_ylabel("success %")
            ax[1].set_ylim(-10, 110)
            ax[1].set_xlim(.1, MAX_TIME_PLOTS)

            if (interactive):
                plt.show()

            pp.savefig(fig)
            plt.close(fig)
    pp.close()

    import shutil
    shutil.copy(filename_pdf, '/tmp/tmp_compare.pdf')


def make_videos(robot: str, problem: str, file: str):

    with open(file, "r") as f:
        data = yaml.load(f, yaml.SafeLoader)

    trajs_opts = data["trajs_opt"]
    trajs_raw = data["trajs_raw"]
    base_name = "vid"
    robot = "unicycle1"
    viewer = viewer_cli.get_robot_viewer(robot)

    for i, traj in enumerate(trajs_opts):
        # create the videos
        filename = f"{base_name}_traj_opt_{i}.mp4"
        viewer.make_video(problem, traj, filename)

    for i, traj in enumerate(trajs_raw):
        filename = f"{base_name}_traj_raw_{i}.mp4"
        viewer.make_video(problem, traj, filename)


def get_cost_evolution(ax, file: str, **kwargs):

    print(f"loading file {file}")
    with open(file, "r") as f:
        data = yaml.load(f, yaml.SafeLoader)

    trajs_opt = data["trajs_opt"]

    time_cost_pairs = []

    best_cost = 1e10

    if trajs_opt is not None:
        for traj in trajs_opt:
            ts = float(traj["time_stamp"]) / 1000
            cs = traj["cost"]
            feas = traj["feasible"]
            if feas:
                print(cs)
                print(best_cost)
                if cs < best_cost:
                    time_cost_pairs.append([ts, cs])
                    best_cost = cs

    if len(time_cost_pairs) == 0:
        print("there is not solution...")
        unsolved_num = 100
        time_cost_pairs.append([unsolved_num, unsolved_num])

    # plot

    ts = [x[0] for x in time_cost_pairs]
    cs = [x[1] for x in time_cost_pairs]

    ax.plot(ts, cs, color=kwargs.get("color", "blue"), alpha=0.3)
    return time_cost_pairs


def get_av_cost(all_costs_np) -> Tuple[np.ndarray, np.ndarray]:
    # easy
    mean = all_costs_np.mean(axis=0)
    std = all_costs_np.std(axis=0)

    # more complex.
    # wait until half of them are solved

    print("all_costs_np")
    print(all_costs_np)

    num_instances = all_costs_np.shape[0]
    mean = []
    std = []
    for j in range(all_costs_np.shape[1]):
        column = all_costs_np[:, j]
        # how many nan?
        non_nan = np.count_nonzero(~np.isnan(column))
        print(column)
        print(non_nan)
        if non_nan > int(num_instances / 2):
            print("success rate is enough")
            cc = column[~np.isnan(column)]
            print("cc", cc)
            mean.append(cc.mean())
            std.append(cc.std())
        else:
            print("success rate is very slow")
            mean.append(np.nan)
            std.append(np.nan)

    return np.array(mean), np.array(std)


def first(iterable, condition):
    """
    Returns the first item in the `iterable` that
    satisfies the `condition`.

    If the condition is not given, returns the first item of
    the iterable.

    Raises `StopIteration` if no item satysfing the condition is found.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( () )
    Traceback (most recent call last):
    ...
    StopIteration
    """

    return next(x for x in iterable if condition(x))


def analyze_runs(path_to_dir: str,
                 problem: str,
                 alg: str,
                 visualize: bool, **kwargs) -> Tuple[str, str]:

    print(
        f"path_to_dir:{path_to_dir}\nproblem:{problem}\nalg:{alg}\nvisualize:{visualize}")

    __files = [f for f in pathlib.Path(
        path_to_dir).iterdir() if f.is_file()]

    # filter some files out.

    files = [f for f in __files if "cfg" not in f.name and "debug" not in f.name and f.suffix ==
             ".yaml" and "report" not in f.name and "traj" not in f.name]

    print(f"files ", [f.name for f in files])

    fig, ax = plt.subplots(2, 1, sharex=True)
    fig.suptitle(problem)

    T = MAX_TIME_PLOTS  # max time
    dt = 0.1
    times = np.linspace(0, T, int(T / dt) + 1)

    all_costs = []

    first_solution = []
    raw_data = []

    for file in [str(f) for f in files]:
        time_cost_pairs = get_cost_evolution(ax[0], file, **kwargs)
        raw_data.append(time_cost_pairs)
        first_solution.append(time_cost_pairs[0][0])
        t = [x[0] for x in time_cost_pairs]
        c = [x[1] for x in time_cost_pairs]

        t.insert(0, 0)
        c.insert(0, np.nan)

        t.append(T)
        c.append(c[-1])

        cost_times = np.interp(times, t, c)
        print(cost_times)
        all_costs.append(cost_times)

    all_costs_np = np.array(all_costs)

    mean, std = get_av_cost(all_costs_np)

    # // nan nan 1

    __where = np.argwhere(np.isnan(mean)).flatten()
    time_first_solution = dt

    if __where.size > 0:
        time_first_solution = float(dt * (__where[-1] + 1))

    try:
        cost_first_solution = first(mean.tolist(), lambda x: not np.isnan(x))
        last_cost = float(mean[-1])
    except StopIteration:
        # TODO : define a way to deal whit this numbers
        cost_first_solution = -1
        last_cost = -1

    # 10 seconds

    cost_at_1 = mean.tolist()[int(1 / dt)]
    cost_at_5 = mean.tolist()[int(5 / dt)]
    cost_at_10 = mean.tolist()[int(10 / dt)]
    cost_at_20 = mean.tolist()[int(20 / dt)]

    # cost of first solution

    # get the first not nan

    success = np.count_nonzero(
        ~np.isnan(all_costs_np),
        axis=0) / len(files) * 100

    ax[0].plot(times, mean, color='blue')
    ax[0].fill_between(
        times,
        mean + std,
        mean - std,
        facecolor='blue',
        alpha=0.2)
    ax[0].set_ylabel("cost")

    ax[1].plot(times, success)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("success %")

    # ax[0].set_ylim([0, 30])
    ax[1].set_xlim([0, 20])
    ax[0].set_xlim([0, 20])

    data_out = {}
    data_out["times"] = times.tolist()
    data_out["success"] = success.tolist()

    print("time_first_solution", time_first_solution)
    print("type(time_first_solution)")
    print(type(time_first_solution))
    data_out["time_first_solution"] = time_first_solution
    data_out["cost_first_solution"] = cost_first_solution

    data_out["cost_at_01"] = cost_at_1
    data_out["cost_at_05"] = cost_at_5
    data_out["cost_at_10"] = cost_at_10
    data_out["cost_at_20"] = cost_at_20
    data_out["last_cost"] = last_cost

    data_out["cost_mean"] = mean.tolist()

    data_out["cost_std"] = std.tolist()
    data_out["raw_data"] = raw_data

    data_out["problem"] = problem
    data_out["alg"] = alg

    fileout = path_to_dir + "/report.yaml"
    print(f"fileout {fileout}")
    with open(fileout, "w") as f:
        yaml.dump(data_out, f)

    figureout = path_to_dir + "/report.pdf"
    print(f"figureout {figureout}")
    plt.savefig(figureout)
    if (visualize):
        plt.show()

    return fileout, figureout


def visualize_motion_together(
        motions: list,
        robot_model: str,
        output_file: str):

    num_motions = len(motions)
    print(f"num_motions {num_motions}")

    # ny = 2
    # nx = max(num_motions // ny, 1)

    ny = 2
    nx = 3

    # G = ny * x + y
    # G -- > y = G % ny
    # G --> x  = G // ny

    viewer = get_viewer(robot_model)
    draw = viewer.view_static

    print(nx, ny)
    fig = plt.figure()
    axs = []

    for i in range(nx * ny):
        y = i % ny
        if robot_model.startswith("quad3d"):
            ax = fig.add_subplot(ny, nx, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(ny, nx, i + 1)
        x = i // ny
        axs.append(ax)

    for i in range(nx * ny):
        y = i % ny
        x = i // ny
        if i >= len(motions):
            continue
        motion = motions[i]

        env = create_empty_env(motion["x0"], motion["xf"], robot_model)
        result = {"states": motion["states"], "actions": motion["actions"]}
        draw(axs[i], env, result)

    print(f"saving drawing of primitives to {output_file}")
    fig.tight_layout()
    plt.savefig(output_file)
    plt.show()


def create_latex_table(csv_file: str) -> None:
    # lets create the latex table

    # some replacements

    df = pandas.read_csv(csv_file)
    str_raw = df.to_latex(index=False, float_format="{:.1f}".format)

    str_ = format_latex_str(str_raw)
    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d--%H-%M-%S")

    with open(f"../results_new/tex/data_{date_time}.tex", "w") as f:
        f.write(str_)

    problems = df["problem"].unique()
    print("problems", problems)
    for problem in problems:
        df_i = df[df["problem"] == problem].drop("problem", axis=1)

        df_i_str_raw = df_i.to_latex(index=False, float_format="{:.1f}".format)
        str_ = format_latex_str(df_i_str_raw)

        problem_ = problem.replace("/", "--")

        with open(f"../results_new/tex/data_{problem_}_{date_time}.tex", "w") as f:
            f.write(str_)


def visualize_primitives(motions: list, robot: str,
                         interactive: bool = True, output_file: str = ""):

    viewer = viewer_cli.get_robot_viewer(robot)

    num_motions = len(motions)
    print(f"num_motions {num_motions}")

    ny = 2
    # nx = max(num_motions // ny, 1)

    nx = 3

    # G = ny * x + y
    # G -- > y = G % ny
    # G --> x  = G // ny

    draw = viewer.view_static

    print(nx, ny)
    fig = plt.figure()
    axs = []
    for i in range(nx * ny):
        y = i % ny
        if robot.startswith("quad3d"):
            ax = fig.add_subplot(ny, nx, i + 1, projection='3d')
        else:
            ax = fig.add_subplot(ny, nx, i + 1)
        x = i // ny
        axs.append(ax)

    for i in range(nx * ny):
        y = i % ny
        x = i // ny
        if i >= len(motions):
            continue
        motion = motions[i]

        env = create_empty_env(motion["start"], motion["goal"], robot)

        result = {"states": motion["states"], "actions": motion["actions"]}
        draw(axs[i], env, result)

    print(f"saving drawing of primitives to {output_file}")
    plt.tight_layout()
    if len(output_file):
        plt.savefig(output_file)
    if interactive:
        plt.show()


def fancy_table(filenames: List[str]):

    df = pandas.DataFrame()

    for file in filenames:
        __df = pandas.read_csv(file)
        df = pandas.concat([df, __df], axis=0)

    def get_data(frame, alg: str, problem: str, field: str, **kwargs):
        print(alg, problem, field)
        __f = frame.loc[(frame["alg"] == alg) & (
            frame["problem"] == problem)].reset_index()
        print(__f)
        print("**" + field + "**")
        return __f[field]

    # buu = get_data(
    #     df,
    #     "idbastar_v0",
    #     "unicycle_first_order_0/parallelpark_0",
    #     "time_first_solution")

    data = []

    algs = ["idbastar_v0", "sst_v0", "geo_v1"]
    fields = ["cost_first_solution", "last_cost", "time_first_solution"]

    def get_column_name(alg: str, field: str):
        token1 = ""
        if alg == "idbastar_v0":
            token1 = "i"
        elif alg == "sst_v0":
            token1 = "s"
        elif alg == "geo_v1":
            token1 = "g"
        else:
            raise KeyError(alg)
        token2 = ""
        if field == "cost_first_solution":
            token2 = "cf"
        elif field == "time_first_solution":
            token2 = "tf"
        elif field == "last_cost":
            token2 = "lc"
        else:
            raise KeyError()
        return token1 + token2

    # problems = [
    #     "unicycle_first_order_0/bugtrap_0",
    #     "unicycle_first_order_0/kink_0",
    #     "unicycle_first_order_0/parallelpark_0",
    #     "unicycle_second_order_0/bugtrap_0",
    #     "unicycle_second_order_0/kink_0",
    #     "unicycle_second_order_0/parallelpark_0"]

    problems =  df.problem.unique()

    all_df = pandas.DataFrame()

    for problem in problems:
        row = []
        headers = []
        for alg in algs:
            for field in fields:
                # D = {"alg": alg,
                #      "problem": problem,
                #      "field": field}
                header = get_column_name(alg, field)
                headers.append(header)
                _data = get_data(df, alg, problem, field)
                row.append(_data[0])
        print("final")
        print(row)
        print(headers)

        new_df = pandas.DataFrame([row], columns=headers, index=[problem])
        print(new_df)
        all_df = pandas.concat([all_df, new_df], axis=0)
    print(all_df)
    all_df.to_csv("tmp.csv")
    # now i could just export as table!!
    # CONTINUE HERE!!!

    str_raw = all_df.to_latex(index=True, float_format="{:.1f}".format)
    str_ = format_latex_str(str_raw)

    lines = str_.splitlines()

    algs = r"&  \multicolumn{3}{c}{idba*} & \multicolumn{3}{c}{sst} & \multicolumn{3}{c}{geo}\\"
    mid_rules = r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}"
    lines.insert(2, algs)
    lines.insert(3, mid_rules)

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y-%m-%d--%H-%M-%S")
    fileout = f"../results_new/tex/merged_{date_time}.tex"

    print("saving", fileout)

    with open(fileout, "w") as f:
        f.write('\n'.join(lines))

    fileout_log = fileout + ".log"
    print("saving", fileout_log)

    with open(fileout_log, "w") as f:
        D = {"input": filenames, "output": fileout, "problems": problems}
        yaml.dump(D, f)


def format_latex_str(str_in: str) -> str:

    str_ = str_in[:]

    D = {"unicycle_first_order_0": "uni1_0",
         "unicycle_first_order_1": "uni1_1",
         "unicycle_first_order_2": "uni1_2",
         "unicycle_second_order_0": "uni2_0",
         "parallelpark_0": "park",
         "cost_at": "c",
         "cost_first_solution": "cs",
         "time_first_solution": "ts",
         "last_cost": "cf"}

    for k, v in D.items():
        str_ = str_.replace(k, v)

    str_ = str_.replace("_", "\\_")
    print("final string is:")
    print(str_)
    return str_


    # group by problem
    # should I use the latex output of pandas?
if __name__ == "__main__":


    # path_to_dir= "../results_new/quadrotor_0/recovery/geo_v1/2023-05-11--15-54-45"
    # problem = "quadrotor_0/recovery"
    # alg = "geo_v1"
    # visualize = False
    # analyze_runs(path_to_dir, problem, alg, visualize)
    # sys.exit(0)

    files = [
        "../results_new/car_first_order_with_1_trailers_0/kink_0/idbastar_v0/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/kink_0/sst_v0/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/kink_0/geo_v1/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/bugtrap_0/idbastar_v0/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/bugtrap_0/sst_v0/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/bugtrap_0/geo_v1/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/parallelpark_0/idbastar_v0/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/parallelpark_0/sst_v0/2023-05-10--16-26-38/report.yaml",
        "../results_new/car_first_order_with_1_trailers_0/parallelpark_0/geo_v1/2023-05-10--16-26-38/report.yaml"]

    # files = ['../results_new/unicycle_second_order_0/bugtrap_0/idbastar_v0/2023-05-10--12-51-39/report.yaml', '../results_new/unicycle_second_order_0/bugtrap_0/idbastar_tmp/2023-05-10--12-51-39/report.yaml']
    #
    # compare(files )
    # raise Exception('DONE')

    # fileout, _ = analyze_runs(path_to_dir="/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/unicycle_first_order_0/kink_0/sst_v0/2023-04-27--17-34-46/",
    # problem="unicycle_first_order_0/kink_0", alg="sst_v0", visualize=True)

    # csv_file = "../results_new/summary/summary_2023-04-27--13-03-13.csv"
    # create_latex_table(csv_file)
    # sys.exit(1)

    do_fancy_table = True
    if do_fancy_table:

        files = [
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-10--15-18-08.csv",
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-10--15-41-42.csv",
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-10--15-56-06.csv",
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-11--07-50-08.csv",
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-11--12-02-25.csv",
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-11--15-06-40.csv", 
            "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/summary/summary_2023-05-11--17-34-33.csv", 
        ]




        fancy_table(files)
        sys.exit(0)
    if do_compare:
        # folders = ["geo_v0/04-04-2023--15-59-51",
        #            "idbastar_v0/04-04-2023--15-59-51",
        #            "idbastar_v1/04-04-2023--15-59-51",
        #            "sst_v0/04-04-2023--15-59-51"]
        #
        # base = "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/unicycle_first_order_0/bugtrap_0/"
        # files = [base + folder + "/report.yaml" for folder in folders]
        files = [
            "../results_new/unicycle_second_order_0/parallelpark_0/idbastar_v0/04-24-2023--18-07-26/report.yaml",
            "../results_new/car_first_order_with_1_trailers_0/parallelpark_0/idbastar_v0/04-24-2023--18-07-26/report.yaml"]

        compare(files, interactive=True)
    if do_benchmark:
        benchmark(bench_cfg)
    if do_debug:
        file = "../results_new/unicycle_second_order_0/parallelpark_0/idbastar_v0/04-06-2023--14-52-41/run_1_out.yaml"
        with open(file, "r") as f:
            data = yaml.safe_load(f)
        print(data)
    if do_vis_primitives:
        # file = "../cloud/motionsV2/tmp_car1.bin.yaml"
        # robot = "car"
        # with open(file, "r") as f:
        #     motions = yaml.safe_load(f)
        # visualize_primitives(motions, robot, True, "out.pdf");

        # file = "../cloud/motionsV2/tmp_acrobot.bin.yaml"
        # robot = "acrobot"
        # with open(file, "r") as f:
        #     motions = yaml.safe_load(f)
        # visualize_primitives(motions, robot, True, "out.pdf");

        file = file_in
        robot = dynamics
        with open(file, "r") as f:
            motions = yaml.safe_load(f)
        visualize_primitives(motions, robot, True, "out.pdf")

    # path = "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/unicycle_first_order_0/bugtrap_0/sst_v0/04-04-2023--11-08-40"
    # path = "../results_new/unicycle_first_order_0/bugtrap_0/geo_v0/04-04-2023--15-23-12/"
    # analyze_runs(path, visualize=False)

    # folder = '../results_new/unicycle_first_order_0/parallelpark_0/idbastar_v0/04-01-2023--11-50-36/'
    # analyze_runs(folder, visualize=True)
    # file = "results_sst.yaml"
    # make_videos(robot, problem, file)

    # file = "tmp_trajectories.yaml"
    # visualize_primitives(file,robot, "primitives.pdf")

    # file = "motions__s__unicycle1_v0__03-04-2023--08-48-54.yaml"
    # robot = "unicycle2"
    # file = "../cloud/motionsV2/unicycle2_v0__2023_04_03__15_01_24.yaml"
    # with open(file, "r") as f:
    #     motions = yaml.safe_load(f)
    #
    # visualize_primitives(
    #     motions,
    #     robot,
    #     interactive=True,
    #     output_file="primitives.pdf")
    #
    # visualize_primitives(motions[10:],
    #                      robot,
    #                      interactive=True,
    #                      output_file="primitives2.pdf")
    #
    # with open(file, "r") as f:
    #     motions = yaml.safe_load(f)
    # print(f"len(motions) {len(motions)}")
    # plot_stats(motions, robot, "new_stats.pdf")
    #

    # stats of primitives
    # Get the stats...
    # Continue here!!
    # visualize_primitives(file,robot, "primitives.pdf")
