import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
from main_komo import run_komo
from gen_motion_primitive_komo import gen_motion
import checker
import tempfile
from pathlib import Path
import yaml


def _run_check(robot_type: str, start: list, goal: list):
	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)

		# generate motion
		motions = gen_motion(robot_type, start, goal)
		assert len(motions) > 0

		for motion in motions:
			# convert to result file format
			result = {
				"result": [{
						"states": motion["states"],
						"actions": motion["actions"],
				}]}
			filename_result = str(p / "result.yaml")
			with open(filename_result, 'w') as f:
				yaml.dump(result, f, Dumper=yaml.CSafeDumper)

			# generate environment file (to be used with the checker)
			env = {
				"environment":{
					"dimensions": [4, 4],
					"obstacles": []
				},
				"robots": [{
					"type": robot_type,
					"start": list(motion["states"][0]),
					"goal": list(motion["states"][-1]),
				}]
			}

			filename_env = str(p / "env.yaml")
			with open(filename_env, 'w') as f:
				yaml.dump(env, f, Dumper=yaml.CSafeDumper)

			result = checker.check(filename_env, filename_result)

def test_unicycle_first_order():
	_run_check("unicycle_first_order_0",
			 [2,2,0],
			 [3,2,0])

def test_unicycle_second_order():
	_run_check("unicycle_second_order_0",
			 [2,2,0,0.5,0],
			 [3,2,0,0.5,0])


