import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
from main_komo import run_komo
import checker


def _run_check(filename_env: str, filename_guess: str, filename_result: str):
    run_komo(filename_env,
             filename_guess,
             filename_result)
    result = checker.check(filename_env, filename_result)
    assert result == True

def test_car_first_order_0():
    _run_check("../benchmark/carFirstOrder/parallelpark_0.yaml",
             "../benchmark/carFirstOrder/initGuess/result_dbastar_parallelpark.yaml",
             "tmp.yaml")



