import subprocess
import yaml
from typing import List
import numpy as np
import pathlib


import sys
sys.path.append('..')


import viewer.viewer_utils as viewer_utils
import viewer.quad2d_viewer as quad2d_viewer
import viewer.quad3d_viewer as quad3d_viewer
import viewer.acrobot_viewer as acrobot_viewer
import viewer.unicycle1_viewer as unicycle1_viewer
import viewer.unicycle2_viewer as unicycle2_viewer
import viewer.car_with_trailer_viewer as car_with_trailer_viewer
import viewer.robot_viewer as robot_viewer
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
            return ( [ "x" , "y" , "theta"] , ["v", "w"])
        elif name.startswith("unicycle2") : 
            return ( [ "x" , "y" , "theta", "v" , "w"] , ["a", "aa"])
        else :
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


problems = [
    "unicycle_first_order_0/parallelpark_0",
]

algs = [
    # "sst_v0",
    # "geo_v0",
    "idbastar_v0"
]


color_map = {
    "idbastar_v0": "red",
    "geo_v0": "green",
    "sst_v0": "blue",
}


trials = 10
timelimit = 6


def get_motions_file(problem: str) -> str:
    return "../cloud/motions/unicycle_first_order_0_sorted.msgpack"


def solve_problem_with_alg(problem: str, alg: str, out: str):
    # load params

    # get the dynamics
    with open(problem, 'r') as f:
        data_problem = yaml.safe_load(f)

    dynamics = data_problem["robots"][0]["type"]
    print(f"dynamics {dynamics}")

    base_path_algs = "../algs/"
    file = base_path_algs + alg + ".yaml"
    print(f"loading {file}")
    with open(file, "r") as f:
        cfg = yaml.safe_load(f)
    print("cfg is:\n", cfg)

    cfg_default = cfg.get("default")
    cfg_dynamics = cfg.get(dynamics, {}).get("default", {})
    cfg_problem = cfg.get(dynamics, {}).get(problem, {})

    print(f"cfg_default {cfg_default}")
    print(f"cfg_dynamics {cfg_dynamics}")
    print(f"cfg_problem {cfg_problem}")

    cfg = cfg_default

    for k, v in cfg_dynamics.items():
        cfg[k] = v

    for k, v in cfg_problem.items():
        cfg[k] = v

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
        motions_file = get_motions_file(problem)
        cmd = [
            "./main_idbastar",
            "--env_file",
            problem,
            "--results_file",
            out,
            "--motionsFile",
            motions_file,
            "--timelimit",
            str(timelimit),
            "--cfg",
            cfg_out]

    else:
        raise NotImplementedError()

    print("**\n**\nRUNNING cpp\n**\n", *cmd, "\n", sep=" ")
    subprocess.run(cmd)
    print("**\n**\nDONE RUNNING cpp\n**\n")


def benchmark():

    base_path_problem = "../benchmark/"
    folder_results = "../results_new/"

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m-%d-%Y--%H-%M-%S")
    for problem in problems:
        for alg in algs:
            path = folder_results + problem + "/" + alg + "/" + date_time
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
            for i in range(trials):
                out = path + f"/run_{i}_out.yaml"
                solve_problem_with_alg(
                    base_path_problem + problem + ".yaml", alg, out)
            analyze_runs(path, visualize=False)

            # basic analysisi of the results?


def compare(interactive: bool = False, filename : str = "/tmp/tmp_compare.pdf"):
    files = ["/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/unicycle_first_order_0/parallelpark_0/idbastar_v0/04-01-2023--11-50-36/report.yaml",
             "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/unicycle_first_order_0/parallelpark_0/idbastar_v0/04-01-2023--11-50-36/report.yaml"]

    # load
    datas = []
    for file in files:
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
            datas.append(data)

    # organize by problem

    D = {}

    for problem in problems:
        # check if data belongs to this problem
        datas = []
        for data in datas:
            if data["problem"] == problem:
                datas.append(data)
        D["problem"] = datas

    # now print the data!

    from matplotlib.backends.backend_pdf import PdfPages


    print(f"writing pdf to {filename}")
    pp = PdfPages(filename)


    for problem in problems: 

        fig, ax = plt.subplots(2,1)
        for d in D[problem]:
            alg = d["alg"]
            times = d["times"]
            success = d["success"]
            cost_mean = d["cost_mean"]
            cost_std = d["cost_std"]
            color = color_map["alg"]

            ax[0].plot(times, cost_std, color=color, label=alg)
            ax[0].fill_between( times,
                cost_mean + cost_std,
                cost_mean - cost_std,
                facecolor=color,
                alpha=0.5)
            ax[1].plot(times, success)

        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylabel("cost")
        ax[1].set_xlabel("time [s]")
        ax[1].set_ylabel("success %")

        if (interactive):
            plt.show()

        pp.savefig(fig)
        plt.close(fig)







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


def get_cost_evolution(ax, file: str):

    with open(file, "r") as f:
        data = yaml.load(f, yaml.SafeLoader)

    trajs_opt = data["trajs_opt"]

    time_cost_pairs = []

    best_cost = 1e10
    for traj in trajs_opt:
        ts = float(traj["time_stamp"]) / 1000
        cs = traj["cost"]
        feas = traj["cost"]
        if feas:
            if cs < best_cost:
                time_cost_pairs.append([ts, cs])
                best_cost = cs

    # plot

    ts = [x[0] for x in time_cost_pairs]
    cs = [x[1] for x in time_cost_pairs]

    ax.plot(ts, cs)
    return time_cost_pairs


def analyze_runs(path_to_dir: str, visualize: bool):

    __files = [f for f in pathlib.Path(
        path_to_dir).iterdir() if f.is_file()]

    # filter some files out.

    files = [f for f in __files if "cfg" not in f.name and "debug" not in f.name]

    print(f"files ", [f.name for f in files])

    fig, ax = plt.subplots(2, 1)

    T = 10  # max time
    dt = 0.1
    times = np.linspace(0, T, int(T / dt) + 1)

    all_costs = []

    first_solution = []
    raw_data = []

    for file in [str(f) for f in files]:
        time_cost_pairs = get_cost_evolution(ax[0], file)
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
    mean = all_costs_np.mean(axis=0)
    std = all_costs_np.std(axis=0)

    success = np.count_nonzero(
        ~np.isnan(all_costs_np),
        axis=0) / len(files) * 100

    ax[0].plot(times, mean, color='blue')
    ax[0].fill_between(
        times,
        mean + std,
        mean - std,
        facecolor='blue',
        alpha=0.5)
    ax[0].set_ylabel("cost")

    ax[1].plot(times, success)
    ax[1].set_xlabel("time [s]")
    ax[1].set_ylabel("success %")

    data_out = {}
    data_out["times"] = times.tolist()
    data_out["success"] = success.tolist()
    data_out["cost_mean"] = mean.tolist()
    data_out["cost_std"] = std.tolist()
    data_out["raw_data"] = raw_data

    with open(path_to_dir + "/report.yaml", "w") as f:
        yaml.dump(data_out, f)

    plt.savefig(path_to_dir + "/report.pdf")
    if (visualize):
        plt.show()


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


if __name__ == "__main__":
    path1 = "results_sst.yaml"

    problem = "../benchmark/unicycle_first_order_0/parallelpark_0.yaml"
    # benchmark()

    # folder = "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/results_new/unicycle_first_order_0/parallelpark_0/geo/03-31-2023--17-27-58"

    # folder = '../results_new/unicycle_first_order_0/parallelpark_0/idbastar_v0/04-01-2023--11-50-36/'
    # analyze_runs(folder, visualize=True)

    # file = "results_sst.yaml"
    # make_videos(robot, problem, file)

    # file = "tmp_trajectories.yaml"
    # visualize_primitives(file,robot, "primitives.pdf")

    # file = "motions__i__unicycle1_v0__02-04-2023--17-22-04.yaml";

    file = "motions__s__unicycle1_v0__03-04-2023--08-48-54.yaml"

    robot = "unicycle2"
    file = "../cloud/motionsV2/unicycle2_v0__2023_04_03__15_01_24.yaml"

    with open(file, "r") as f:
        motions = yaml.safe_load(f)

    visualize_primitives(
        motions,
        robot,
        interactive=True,
        output_file="primitives.pdf")

    visualize_primitives(motions[10:],
                         robot,
                         interactive=True,
                         output_file="primitives2.pdf")

    with open(file, "r") as f:
        motions = yaml.safe_load(f)
    print(f"len(motions) {len(motions)}")
    plot_stats(motions, robot, "new_stats.pdf")

    # stats of primitives
    # Get the stats...
    # Continue here!!
    # visualize_primitives(file,robot, "primitives.pdf")
