import sys
import os
import subprocess as sp


def test_trailer_komo():
    """
    Run basic test of komo trailer to check that it solve and easy optimization problem.
    """

    cmd = "./trailer -model \"../src/car_with_trailer.g\" -N 50"

    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0


def test_trailer_komo2():

    cmd = "./trailer -model \"../src/g_trailer_test.g\" -N 50"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0


def test_yaml_to_komo():
    # generate from yaml file
    cmd = "python3 ../scripts/translate_g.py  --fin  ../benchmark/carFirstOrderWithTrailers/parallelpark_0.yaml --fout tmp_trailer.g"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0

    # generate from yaml file
    cmd = "./trailer -model tmp_trailer.g -N 50"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0
