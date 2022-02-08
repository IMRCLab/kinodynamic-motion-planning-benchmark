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

def test_unicycle_first_order_parallelpark_0():
    _run_check("../benchmark/unicycle_first_order_0/parallelpark_0.yaml",
             "../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml",
             "tmp.yaml")


def test_unicycle_first_order_0_kink_0():
    _run_check("../benchmark/unicycle_first_order_0/kink_0.yaml",
               "../test/unicycle_first_order_0/guess_kink_0_sol0.yaml",
               "tmp.yaml")


def test_unicycle_first_order_0_bugtrap_0():
    _run_check("../benchmark/unicycle_first_order_0/bugtrap_0.yaml",
               "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml",
               "tmp.yaml")


def test_unicycle_second_order_0_parallelpark_0():
    _run_check("../benchmark/unicycle_second_order_0/parallelpark_0.yaml",
               "../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml",
               "tmp.yaml")


def test_unicycle_second_order_0_kink_0():
    _run_check("../benchmark/unicycle_second_order_0/kink_0.yaml",
               "../test/unicycle_second_order_0/guess_kink_0_sol0.yaml",
               "tmp.yaml")


def test_car_first_order_with_trailers_parallelpark_0():
    _run_check("../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml",
             "../test/car_first_order_with_1_trailers_0/guess_bugtrap_0_sol0.yaml",
             "tmp.yaml")
