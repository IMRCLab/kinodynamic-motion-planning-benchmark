# import numpy as np
import yaml
# import argparse
from main_ompl import run_ompl
from main_sbpl import run_sbpl
from main_dbastar import run_dbastar
from pathlib import Path
import shutil
import subprocess


def run_visualize(script, filename_env, filename_result):

	subprocess.run(["python3",
				script,
				filename_env,
				"--result", filename_result,
				"--video", filename_result.with_suffix(".mp4")])

def main():
	benchmark_path = Path("../benchmark")
	results_path = Path("../results")
	tuning_path = Path("../tuning")

	instances = ["carFirstOrder/bugtrap_0", "carFirstOrder/kink_0", "carFirstOrder/parallelpark_0"]
	# instances = ["carFirstOrder/parallelpark_0"]
	# algs = ["sst", "sbpl", "dbAstar"]
	algs = ["sbpl"]
	timelimit = 5 * 60

	for instance in instances:
		env = (benchmark_path / instance).with_suffix(".yaml")
		assert(env.is_file())

		cfg = (tuning_path / instance).parent / "algorithms.yaml"
		assert(cfg.is_file())
		with open(cfg) as f:
			cfg = yaml.safe_load(f)

		for alg in algs:
			result_folder = results_path / instance / alg
			if result_folder.exists():
				print("Warning! {} exists already. Deleting...".format(result_folder))
				shutil.rmtree(result_folder)
			result_folder.mkdir(parents=True, exist_ok=False)

			# find cfg
			mycfg = cfg[alg]
			if Path(instance).name in mycfg:
				mycfg = mycfg[Path(instance).name]
			else:
				mycfg = mycfg['default']

			print("Using configurations ", mycfg)

			if alg == "sst":
				run_ompl(str(env), str(result_folder), timelimit, mycfg)
				visualize_files = ["result_ompl.yaml", "result_scp.yaml"]
			elif alg == "sbpl":
				run_sbpl(str(env), str(result_folder))
				visualize_files = ["result_sbpl.yaml", "result_scp.yaml"]
			elif alg == "dbAstar":
				run_dbastar(str(env), str(result_folder), timelimit)
				visualize_files = [p.name for p in result_folder.glob('result_*')]
			else:
				raise Exception("Unknown algorithms {}".format(alg))

			vis_script = (benchmark_path / instance).parent / "visualize.py"
			for file in visualize_files:
				run_visualize(vis_script, env, result_folder / file)

if __name__ == '__main__':
	main()
