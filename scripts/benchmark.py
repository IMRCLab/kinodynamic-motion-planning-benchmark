# import numpy as np
import yaml
# import argparse
# TODO: I need the primitives!!
from main_ompl import run_ompl
from main_sbpl import run_sbpl
from main_dbastar import run_dbastar
from main_geo_croco import run_main_geo
# from main_komo import run_komo_standalone
# from main_scp import run_scp_standalone
from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass
import multiprocessing as mp
import tqdm
import psutil
import checker
import argparse


@dataclass
class ExecutionTask:
    """Class for keeping track of an item in inventory."""
    # env: Path
    # cfg: Path
    # result_folder: Path
    instance: str
    alg: str
    trial: int
    timelimit: float


def run_visualize(script, filename_env, filename_result):

    subprocess.run(["python3",
                    script,
                    filename_env,
                    "--result", filename_result,
                    "--video", filename_result.with_suffix(".mp4")])


def execute_task(task: ExecutionTask):

    benchmark_path = Path("../benchmark")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--id", default="0")
    args = parser.parse_args()
    id = args.id

    results_path = Path(f"../results/{id}")
    tuning_path = Path("../tuning")

    env = (benchmark_path / task.instance).with_suffix(".yaml")
    assert (env.is_file())

    _cfg = (tuning_path / task.instance).parent / "algorithms.yaml"
    assert (_cfg.is_file())

    print(f"loading _cfg {_cfg}")
    with open(_cfg) as f:
        cfg = yaml.safe_load(f)
    print("cfg")
    print(cfg)

    result_folder = results_path / task.instance / \
        task.alg / "{:03d}".format(task.trial)
    if result_folder.exists():
        print("Warning! {} exists already. Deleting...".format(result_folder))
        shutil.rmtree(result_folder)
    result_folder.mkdir(parents=True, exist_ok=False)

    # find cfg
    mycfg = cfg[task.alg]
    mycfg = mycfg['default']
    if Path(task.instance).name in cfg[task.alg]:
        mycfg_instance = cfg[task.alg][Path(task.instance).name]
        mycfg = {**mycfg, **mycfg_instance}  # merge two dictionaries

    print("Using configurations ", mycfg)
    task.timelimit = 10

    if task.alg == "sst":
        run_ompl(str(env), str(result_folder), task.timelimit, mycfg)

    elif task.alg == "geo-croco":
        run_main_geo(str(env), str(result_folder), task.timelimit, mycfg)

    # if task.alg == "sst":
    #     run_ompl(str(env), str(result_folder), task.timelimit, mycfg)
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_*')]
    # elif task.alg == "sbpl":
    #     run_sbpl(str(env), str(result_folder))
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_*')]
    # elif task.alg == "dbAstar-komo":
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "komo")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "dbAstar-q": # new
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "komo")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "dbAstar-crococpp":
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "crococpp")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "dbAstar-crococpp-v2":
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "crococpp-v2")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "dbAstar-croco":
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "croco")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "dbAstar-crococpp":
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "crococpp")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "dbAstar-scp":
    #     run_dbastar(str(env), str(result_folder), task.timelimit, mycfg, "scp")
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_opt*')]
    # elif task.alg == "komo":
    #     run_komo_standalone(str(env), str(result_folder), task.timelimit, mycfg["rai_cfg"])
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_komo*')]
    # elif task.alg == "geo-croco":
    #     run_main_geo(str(env), str(result_folder), task.timelimit, mycfg)
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_komo*')]
    # elif task.alg == "scp":
    #     run_scp_standalone(str(env), str(result_folder), task.timelimit, mycfg)
    #     visualize_files = [p.name for p in result_folder.glob('result_*')]
    #     check_files = [p.name for p in result_folder.glob('result_*')]
    # else:
    #     raise Exception("Unknown algorithms {}".format(task.alg))
    #
    # for in_f in check_files:
    #     with open((result_folder / in_f).with_suffix(".txt"), 'w') as out_f:
    #         print("CHECK: ", checker.check(str(env), str(result_folder / in_f), out_f))
    #
    # vis_script = (benchmark_path / task.instance).parent / "visualize.py"
    # for file in visualize_files:
    #     run_visualize(vis_script, env, result_folder / file)


def main():
    parallel = False
    instances = [
        "unicycle_first_order_0/parallelpark_0",
        # "unicycle_first_order_0/kink_0",
        # "unicycle_first_order_0/bugtrap_0",
        # "unicycle_first_order_1/kink_0",
        # "unicycle_first_order_2/wall_0",
        # "unicycle_second_order_0/parallelpark_0",
        # "unicycle_second_order_0/kink_0",
        # "unicycle_second_order_0/bugtrap_0",
        # "car_first_order_with_1_trailers_0/parallelpark_0",
        # "car_first_order_with_1_trailers_0/kink_0",
        # "car_first_order_with_1_trailers_0/bugtrap_0",
        # "quadrotor_0/empty_0",
    ]
    algs = [
        # "sst",
        # "sbpl",
        "geo-croco",
        # "dbAstar-komo",
        # "dbAstar-crococpp",
        # "dbAstar-crococpp-v2",
        # "dbAstar-q",
        # "crococpp",
        # "dbAstar-croco",
        # "dbAstar-scp",
    ]
    # trials = 10
    # timelimit = 5 * 60
    trials = 1
    timelimit = 30

    tasks = []
    for instance in instances:
        for alg in algs:
            # sbpl only supports unicycleFirstOrder
            if alg == "sbpl" and "unicycle_first_order_0" not in instance:
                continue
            for trial in range(trials):
                tasks.append(ExecutionTask(instance, alg, trial, timelimit))

    if parallel and len(tasks) > 1:
        use_cpus = psutil.cpu_count(logical=False) - 1
        print("Using {} CPUs".format(use_cpus))
        with mp.Pool(use_cpus) as p:
            for _ in tqdm.tqdm(p.imap_unordered(execute_task, tasks)):
                pass
    else:
        for task in tasks:
            execute_task(task)


if __name__ == '__main__':
    main()
