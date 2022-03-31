import sys
import os
import pytest
import subprocess as sp

# skip all tests in this module
pytestmark = pytest.mark.skip(reason="Test needs to be updated")


def test_trailer_komo():
    """
    Run basic test of komo trailer to check that it solve and easy optimization problem.
    """

    cmd = "./main_rai -robot trailer -model \"../src/car_with_trailer.g\" -N 50"

    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0


def test_trailer_komo2():

    cmd = "./main_rai -robot trailer -model \"../src/g_trailer_test.g\" -N 50"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0


def test_yaml_to_komo():
    # generate from yaml file
    cmd = "python3 ../scripts/translate_g.py  --fin  ../benchmark/car_first_order_with_1_trailers_0/parallelpark_0.yaml --fout tmp_trailer.g"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0

    # generate from yaml file
    cmd = "./main_rai -robot trailer -model tmp_trailer.g -N 50"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


def test_bugtrap():
    cmd = "./main_rai -model \"../test/env_debug_bugtrap_trailer.g\"  -waypoints \"../test/car_first_order_with_1_trailers_0/guess_bugtrap_0_sol0.yaml\" -N -1 -display 0 -animate 0 -order 1 -robot car_first_order_with_1_trailers_0 -cfg rai.cfg -env \"../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml\" -out \"tmp.yaml\""
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0
