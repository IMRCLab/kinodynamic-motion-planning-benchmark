import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
from main_komo import run_komo
from gen_motion_primitive_komo import gen_motion
import checker
import tempfile
from pathlib import Path
import yaml
import numpy as np
import rowan


def _run_check(robot_type: str, start: list, goal: list, is2D: bool = False):
	with tempfile.TemporaryDirectory() as tmpdirname:
		p = Path(tmpdirname)
		# p = Path("../results/test")


		# generate motion
		motions = gen_motion(robot_type, start, goal, is2D)
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
					"min": [-2, -2],
					"max": [2, 2],
					"obstacles": []
				},
				"robots": [{
					"type": robot_type,
					"start": list(motion["states"][0]),
					"goal": list(motion["states"][-1]),
				}]
			}
			if not is2D:
				env["environment"]["min"].append(-2)
				env["environment"]["max"].append(2)

			filename_env = str(p / "env.yaml")
			with open(filename_env, 'w') as f:
				yaml.dump(env, f, Dumper=yaml.CSafeDumper)

			result = checker.check(filename_env, filename_result)
			assert(result)

def test_unicycle_first_order():
	_run_check("unicycle_first_order_0",
			 [2,2,0],
			 [3,2,0])

def test_unicycle_second_order():
	_run_check("unicycle_second_order_0",
			 [2,2,0,0.5,0],
			 [3,2,0,0.5,0])

def test_car_first_order_with_1_trailers():
	_run_check("car_first_order_with_1_trailers_0",
			 [2,2,0,0],
			 [3,2,0,0])

# def test_car_first_order_with_1_trailers():
# 	_run_check("car_first_order_with_1_trailers_0",
# 			 [1,1,np.pi/4,np.pi/4],
# 			 [2,2,np.pi/4,np.pi/4])

def test_quadrotor():
	# fly 0.5 upwards
	_run_check("quadrotor_0",
			 [1,1,1, 0,0,0,1, 0,0,0, 0,0,0], # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz,
			 [1,1,1.5, 0,0,0,1, 0,0,0, 0,0,0], # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
			 is2D=False
	)

	# # initial & final yaw
	# qs = rowan.from_euler(np.radians(0),np.radians(0),np.radians(45), "xyz").tolist()
	# qf = rowan.from_euler(np.radians(0),np.radians(0),np.radians(-45), "xyz").tolist()
	# _run_check("quadrotor_0",
	# 		 [1,1,1, qs[1], qs[2], qs[3], qs[0], 0,0,0, 0,0,0], # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz,
	# 		 [1,1,1, qf[1], qf[2], qf[3], qf[0], 0,0,0, 0,0,0] # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
	# )

	# # initial & final roll
	# qs = rowan.from_euler(np.radians(30),np.radians(0),np.radians(0), "xyz").tolist()
	# qf = rowan.from_euler(np.radians(-30),np.radians(0),np.radians(0), "xyz").tolist()
	# _run_check("quadrotor_0",
	# 		 [1,1,1, qs[1], qs[2], qs[3], qs[0], 0,0,0, 0,0,0], # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz,
	# 		 [1,1,1, qf[1], qf[2], qf[3], qf[0], 0,0,0, 0,0,0] # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
	# )

	# # initial & final roll
	# qs = rowan.from_euler(np.radians(0),np.radians(0),np.radians(0), "xyz").tolist()
	# qf = rowan.from_euler(np.radians(0),np.radians(0),np.radians(0), "xyz").tolist()
	# _run_check("quadrotor_0",
	# 		 [1,1,1, qs[1], qs[2], qs[3], qs[0], 0.5,0,0, 0,0,0], # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz,
	# 		 [1,1,1, qf[1], qf[2], qf[3], qf[0], 0,0,0, 0,0,0] # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
	# )