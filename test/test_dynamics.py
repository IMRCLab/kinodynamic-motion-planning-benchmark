import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/../scripts")
from motionplanningutils import RobotHelper
import robots
import numpy as np

def _test_dynamics_cpp_py(robot_type):
    robot_cpp = RobotHelper(robot_type)
    robot_py = robots.create_robot(robot_type)

    for _ in range(100):
        state = robot_cpp.sampleUniform()
        control = robot_cpp.sampleControlUniform()

        next_state_cpp = robot_cpp.step(state, control, robot_py.dt)
        next_state_py = robot_py.step(state, control)

        assert np.allclose(next_state_cpp, next_state_py, rtol=1.e-4, atol=1.e-7)


def test_dynamics_cpp_py_unicycle_first_order_0():
    _test_dynamics_cpp_py('unicycle_first_order_0')


def test_dynamics_cpp_py_unicycle_first_order_1():
    _test_dynamics_cpp_py('unicycle_first_order_1')


def test_dynamics_cpp_py_unicycle_first_order_2():
    _test_dynamics_cpp_py('unicycle_first_order_2')


def test_dynamics_cpp_py_unicycle_second_order_0():
    _test_dynamics_cpp_py('unicycle_second_order_0')


# def test_dynamics_cpp_py_car_first_order():
#     _test_dynamics_cpp_py('car_first_order_0')


def test_dynamics_cpp_py_car_first_order_with_1_trailers_0():
    _test_dynamics_cpp_py('car_first_order_with_1_trailers_0')


def test_dynamics_cpp_py_quadrotor_0():
    _test_dynamics_cpp_py('quadrotor_0')



