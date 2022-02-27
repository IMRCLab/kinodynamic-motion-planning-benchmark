import sys
import os
import subprocess as sp


def main():
    cmd = "./main_rai -model \"../paper/env.g\" -waypoints \"../paper/result_dbastar_sol1.yaml\" -N -1 -display 1 -animate 2 -order 1 -robot unicycle_first_order_2 -env \"../benchmark/unicycle_first_order_2/wall_0.yaml\" -out \"tmp.yaml\""
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 0


if __name__ == '__main__':
	main()
