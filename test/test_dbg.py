import sys
import os
import subprocess as sp


def test_dbg1():
    case = "dbg/1"
    cmd = "./main_rai -model \"../test/{0}/env.g\" -waypoints \"../test/car_first_order_with_1_trailers_0/parallelpark_0_sst.yaml\" -N -1 -display 1 -animate 1 -order 1 -robot car_first_order_with_1_trailers_0 -cfg \"../test/{0}/rai.cfg\" -env \"../benchmark/car_first_order_with_1_trailers_0/parallelpark_0.yaml\" -out \"tmp.yaml\"".format(case)
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0
