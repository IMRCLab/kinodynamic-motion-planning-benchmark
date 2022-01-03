# import numpy as np
import yaml
# import argparse
from main_ompl import run_ompl
from main_sbpl import run_sbpl
from main_dbastar import run_dbastar
from pathlib import Path
import shutil
import subprocess
from dataclasses import dataclass
import multiprocessing as mp
import tqdm
import psutil


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
	results_path = Path("../results")
	tuning_path = Path("../tuning")

	env = (benchmark_path / task.instance).with_suffix(".yaml")
	assert(env.is_file())

	cfg = (tuning_path / task.instance).parent / "algorithms.yaml"
	assert(cfg.is_file())

	with open(cfg) as f:
		cfg = yaml.safe_load(f)


	result_folder = results_path / task.instance / task.alg / "{:03d}".format(task.trial)
	if result_folder.exists():
			print("Warning! {} exists already. Deleting...".format(result_folder))
			shutil.rmtree(result_folder)
	result_folder.mkdir(parents=True, exist_ok=False)

	# find cfg
	mycfg = cfg[task.alg]
	if Path(task.instance).name in mycfg:
		mycfg = mycfg[Path(task.instance).name]
	else:
		mycfg = mycfg['default']

	print("Using configurations ", mycfg)

	if task.alg == "sst":
		run_ompl(str(env), str(result_folder), task.timelimit, mycfg)
		visualize_files = ["result_ompl.yaml", "result_scp.yaml"]
	elif task.alg == "sbpl":
		run_sbpl(str(env), str(result_folder))
		visualize_files = ["result_sbpl.yaml", "result_scp.yaml"]
	elif task.alg == "dbAstar-komo":
		run_dbastar(str(env), str(result_folder), task.timelimit, "komo")
		visualize_files = [p.name for p in result_folder.glob('result_*')]
	elif task.alg == "dbAstar-scp":
		run_dbastar(str(env), str(result_folder), task.timelimit, "scp")
		visualize_files = [p.name for p in result_folder.glob('result_*')]
	else:
		raise Exception("Unknown algorithms {}".format(task.alg))

	vis_script = (benchmark_path / task.instance).parent / "visualize.py"
	for file in visualize_files:
		run_visualize(vis_script, env, result_folder / file)



def main():
	parallel = True
	# instances = ["carFirstOrder/bugtrap_0", "carFirstOrder/kink_0", "carFirstOrder/parallelpark_0"]
	# algs = ["sst", "sbpl",  "dbAstar-komo", "dbAstar-scp"]
	# trials = 5
	# timelimit = 5 * 60

	# # instances = ["carFirstOrder/bugtrap_0"]
	# instances = ["carFirstOrder/bugtrap_0", "carFirstOrder/kink_0", "carFirstOrder/parallelpark_0"]
	# algs = ["dbAstar-komo", "dbAstar-scp"]
	# trials = 5
	# timelimit = 5 * 60

	# instances = ["carFirstOrder/bugtrap_0"]
	instances = ["carSecondOrder/parallelpark_0", "carSecondOrder/kink_0", "carSecondOrder/bugtrap_0"]
	algs = ["sst", "dbAstar-scp"]
	trials = 5
	timelimit = 5 * 60

	tasks = []
	for instance in instances:
		for alg in algs:
			for trial in range(trials):
				tasks.append(ExecutionTask(instance, alg, trial, timelimit))

	if parallel:
		use_cpus = psutil.cpu_count(logical=False)-1
		print("Using {} CPUs".format(use_cpus))
		with mp.Pool(use_cpus) as p:
			for _ in tqdm.tqdm(p.imap_unordered(execute_task, tasks)):
				pass
	else:
		for task in tasks:
			execute_task(task)

if __name__ == '__main__':
	main()
