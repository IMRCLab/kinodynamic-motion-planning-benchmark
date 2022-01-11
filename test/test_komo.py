import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
from main_komo import run_komo
import checker


def _run_check(filename_env: str, filename_guess: str, filename_result: str):
    result = run_komo(filename_env,
                        filename_guess,
                        filename_result)
    assert result == True
    result = checker.check(filename_env, filename_result)
    assert result == True

def test_car_first_order_parallelpark_0():
    _run_check("../benchmark/carFirstOrder/parallelpark_0.yaml",
             "../test/carFirstOrder/guess_parallelpark_0_sol0.yaml",
             "tmp.yaml")


def test_car_first_order_0_kink_0():
    _run_check("../benchmark/carFirstOrder/kink_0.yaml",
               "../test/carFirstOrder/guess_kink_0_sol0.yaml",
               "tmp.yaml")


def test_car_first_order_0_bugtrap_0():
    _run_check("../benchmark/carFirstOrder/bugtrap_0.yaml",
               "../test/carFirstOrder/guess_bugtrap_0_sol0.yaml",
               "tmp.yaml")


def test_car_second_order_0_parallelpark_0():
    _run_check("../benchmark/carSecondOrder/parallelpark_0.yaml",
               "../test/carSecondOrder/guess_parallelpark_0_sol0.yaml",
               "tmp.yaml")


def test_car_second_order_0_kink_0():
    _run_check("../benchmark/carSecondOrder/kink_0.yaml",
               "../test/carSecondOrder/guess_kink_0_sol0.yaml",
               "tmp.yaml")


def test_car_second_order_0_bugtrap_0():
    _run_check("../benchmark/carSecondOrder/bugtrap_0.yaml",
               "../test/carSecondOrder/guess_bugtrap_0_sol0.yaml",
               "tmp.yaml")
