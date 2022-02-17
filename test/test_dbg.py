import sys
import os
import subprocess as sp


def test_dbg1():
    case = "dbg/1"
    cmd = "./main_rai -model \"../test/{0}/env.g\" -waypoints \"../test/{0}/result_ompl.yaml\" -N -1 -display 1 -animate 1 -order 1 -robot unicycle_first_order_2 -cfg \"../test/{0}/rai.cfg\" -env \"../benchmark/unicycle_first_order_2/wall_0.yaml\" -out \"../test/{0}/result_opt.yaml\"".format(case)
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0
