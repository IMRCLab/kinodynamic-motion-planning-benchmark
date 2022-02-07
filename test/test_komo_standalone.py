import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
from main_komo import run_komo_standalone
import checker
import tempfile
from pathlib import Path
import yaml


def _run_check(robot_type: str, start: list, goal: list, expected_T: int):
	dbg = False
	with tempfile.TemporaryDirectory() as tmpdirname:
		if dbg:
			p = Path("../results/test")
		else:
			p = Path(tmpdirname)

		# generate environment file
		env = {
			"environment":{
				"min": [0, 0],
				"max": [4, 4],
				"obstacles": []
			},
			"robots": [{
				"type": robot_type,
				"start": list(start),
				"goal": list(goal),
			}]
		}

		filename_env = str(p / "env.yaml")
		with open(filename_env, 'w') as f:
			yaml.dump(env, f, Dumper=yaml.CSafeDumper)

		result = run_komo_standalone(filename_env, str(p), 60, search="linear")
		assert result == True

		filename_result = p / "result_komo.yaml"

		if dbg:
			import subprocess
			subprocess.run(["python3",
				"../benchmark/unicycleSecondOrder/visualize.py",
				str(filename_env),
				"--result", str(filename_result),
				"--video", str(filename_result.with_suffix(".mp4"))])

		result = checker.check(filename_env, filename_result, expected_T=expected_T)
		assert result == True

def test_unicycle_first_order():
	# move in a straight line for 1m => 2s => 20 timesteps
	_run_check("unicycle_first_order_0",
			 [2,2,0],
			 [3,2,0],
			 [20,21])

def test_unicycle_second_0_order_initial_velocity():
	# move in a straight line for 1m => 2s => 20 timesteps
	_run_check("unicycle_second_order_0",
			 [2,2,0,0.5,0],
			 [3,2,0,0.5,0],
			 [20,21,22,23])

def test_unicycle_second_0_order_start_stop():
	# move in a straight line for 1m => 2s => 20 timesteps
	# Additional 3 timesteps for accelerating/breaking
	_run_check("unicycle_second_order_0",
			 [2,2,0,0.0,0],
			 [3,2,0,0.0,0],
			 [42,43])
