import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
import subprocess as sp

# quick integration tests


def test_croco1():
    """
    Run basic test of komo trailer to check that it solve and easy optimization problem.
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_croco2():
    """
    """
    cmd = "python3  ../scripts/main_croco.py parallel_circle.yaml  ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_croco3():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml ../test/car_first_order_with_1_trailers_0/guess_bugtrap_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_croco4():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_second_order_0/parallelpark_0.yaml   ../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_croco5():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/bugtrap_0.yaml ../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_croco6():
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/bugtrap_0.yaml ../test/unicycle_first_order_0/guess_bugtrap_0_sol1.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0
