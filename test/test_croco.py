import sys
import os
sys.path.append(os.getcwd() + "/../scripts")
import subprocess as sp

# quick integration tests

def test_unicycle1_park():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0

def test_unicycle1_park_time():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml --freetime out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0

def test_unicycle1_park_time_noguess():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  ../test/unicycle_first_order_0/guess_trivial.yaml out.yaml --vis --freetime"                         
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_unicycle1_park_noguess():
    """
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  ../test/unicycle_first_order_0/guess_trivial.yaml out.yaml --vis"                         
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0

def test_unicycle2_park():
    """
    TODO: check issue with collisions. It is not in collision, but the Lag multiplier is very big. why? Maybe is unstable because of the inequalities?
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_second_order_0/parallelpark_0.yaml   ../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_unicycle_circles():
    """
    """
    cmd = "python3  ../scripts/main_croco.py parallel_circle.yaml  ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_croco3():
    """
    note: good optimization but check afterwards are failing. why?
    """

    cmd = "python3  ../scripts/main_croco.py ../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml ../test/car_first_order_with_1_trailers_0/guess_bugtrap_0_sol0.yaml out.yaml"
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
    """
    note: good optimization check complains about 2*pi difference
    """
    cmd = "python3  ../scripts/main_croco.py ../benchmark/unicycle_first_order_0/bugtrap_0.yaml ../test/unicycle_first_order_0/guess_bugtrap_0_sol1.yaml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0

def test_coco_quad_hover():
    cmd = f"python3 ../scripts/main_croco.py ../benchmark/quadrotor_0/empty_test_easy.yaml ../benchmark/quadrotor_0/empty_easy_guess.yml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_coco_quad():
    cmd = f"python3  ../scripts/main_croco.py ../benchmark/quadrotor_0/empty_test_easy2.yaml ../benchmark/quadrotor_0/empty_easy_guess.yml out.yaml"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0




