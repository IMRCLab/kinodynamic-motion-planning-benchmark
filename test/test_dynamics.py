import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/../scripts")
from motionplanningutils import RobotHelper
import robots
import numpy as np

def _test_dynamics(robot_type, dt):
    robot_cpp = RobotHelper(robot_type)
    robot_py = robots.create_robot(robot_type)

    for _ in range(100):
        state = robot_cpp.sampleUniform()
        control = robot_cpp.sampleControlUniform()

        next_state_cpp = robot_cpp.step(state, control, dt)
        next_state_py = robot_py.step(state, control)

        assert np.allclose(next_state_cpp, next_state_py)


def test_dynamics_car_first_order_0():
    _test_dynamics('car_first_order_0', 0.1)


def test_dynamics_car_second_order_0():
    _test_dynamics('car_second_order_0', 0.1)


def test_dynamics_car_first_order_with_0_trailers_0():
    _test_dynamics('car_first_order_with_0_trailers_0', 0.1)


def test_dynamics_car_first_order_with_1_trailers_0():
    _test_dynamics('car_first_order_with_1_trailers_0', 0.1)


def test_dynamics_quadrotor_0():
    _test_dynamics('quadrotor_0', 0.01)



