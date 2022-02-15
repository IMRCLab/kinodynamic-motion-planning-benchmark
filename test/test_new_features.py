import sys
import os
import subprocess as sp

def test_parallel_park_horizon():
    """
    """
    cmd = "./main_rai -model \"../test/car_first_order_with_1_trailers_0/env_parallel.g\" -waypoints \"../test/car_first_order_with_1_trailers_0/guess_parallel.yaml\" -display false -order 1 -robot car_first_order_with_1_trailers_0 -mode horizon -num_h 5"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 1


def test_parallel_park_time_trick():
    """
    """
    # CONTINUE HERE!!
    # somehow not working now? why?
    # quim@fourier ~/s/w/k/build (time_sos)> make && ./main_rai -model \"../test/car_first_order_with_1_trailers_0/env_parallel.g\" -waypoints \"../test/car_first_order_with_1_trailers_0/guess_parallel.yaml\" -display true -order 1 -robot car_first_order_with_1_trailers_0 -mode time_trick


def test_parallel_binary_search():
    """
    """
    cmd = "./main_rai -model \"../test/car_first_order_with_1_trailers_0/env_parallel.g\" -waypoints \"../test/car_first_order_with_1_trailers_0/guess_parallel.yaml\" -display false -order 1 -robot car_first_order_with_1_trailers_0 -mode search_time"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 1





def test_parallel_park_horizon_with_time():
    """
    """
    cmd = "./main_rai -model \"../test/car_first_order_with_1_trailers_0/env_parallel.g\" -waypoints \"../test/car_first_order_with_1_trailers_0/guess_parallel.yaml\" -display false -order 1 -robot car_first_order_with_1_trailers_0 -mode horizon_binary_search -num_h 10"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)
    assert out.returncode == 1




def test_bugtrap_horizon_hard():
    """
    Run basic test of komo trailer to check that it solve and easy optimization problem.
    """
    cmd = "./main_rai -model \"../test/env_debug_bugtrap_trailer.g\"  -waypoints \"../test/car_first_order_with_1_trailers_0/guess_bugtrap_0_sol0.yaml\" -N -1 -display 1 -animate 0 -order 1 -robot car_first_order_with_1_trailers_0 -cfg rai.cfg -env \"../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml\" -out \"tmp.yaml\" -display true -mode  horizon -num_h 100"
    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0


def test_bugtrap_horizon_smooth():
    """
    Run basic test of komo trailer to check that it solve and easy optimization problem.
    """
    cmd = "./main_rai -model \"../test/env_debug_bugtrap_trailer.g\"  -waypoints \"../test/car_first_order_with_1_trailers_0/smooth.yaml\" -N -1 -display 1 -animate 0 -order 1 -robot car_first_order_with_1_trailers_0 -cfg rai.cfg -env \"../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml\" -out \"tmp.yaml\" -display true -mode  horizon -num_h 100"

    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0

def test_bugtrap_binary_search():
    """
    Run basic test of komo trailer to check that it solve and easy optimization problem.
    """
    cmd = "./main_rai -model \"../test/env_debug_bugtrap_trailer.g\"  -waypoints \"../test/car_first_order_with_1_trailers_0/smooth.yaml\" -N -1 -display 1 -animate 0 -order 1 -robot car_first_order_with_1_trailers_0 -cfg rai.cfg -env \"../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml\" -out \"tmp.yaml\" -display false -mode  search_time"

    print("Running", cmd)
    out = sp.run(cmd.split())
    print("Running DONE", cmd)

    assert out.returncode == 0


# def test_bugtrap_horizon_and_binary_search():
#     """
#     Run basic test of komo trailer to check that it solve and easy optimization problem.
#     """
#     cmd = "./main_rai -model \"../test/env_debug_bugtrap_trailer.g\"  -waypoints \"../test/car_first_order_with_1_trailers_0/smooth.yaml\" -N -1 -display 1 -animate 0 -order 1 -robot car_first_order_with_1_trailers_0 -cfg rai.cfg -env \"../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml\" -out \"tmp.yaml\" -display true -mode  horizon -num_h 100"

#     print("Running", cmd)
#     out = sp.run(cmd.split())
#     print("Running DONE", cmd)

#     assert out.returncode == 0




