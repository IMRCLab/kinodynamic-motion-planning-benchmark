
#include "dbastar.hpp"
#include "idbastar.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

// #include <boost/test/unit_test_suite.hpp>
// #define BOOST_TEST_DYN_LINK
// #include <boost/test/unit_test.hpp>

// see
// https://www.boost.org/doc/libs/1_81_0/libs/test/doc/html/boost_test/usage_variants.html
// #define BOOST_TEST_MODULE test module name

#define BOOST_TEST_MODULE test idbastar
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(parallel_park_1) {
  Problem problem("../benchmark/unicycle_first_order_0/parallelpark_0.yaml");
  Options_idbAStar options_idbas;
  Options_dbastar options_dbastar;
  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  Options_trajopt options_trajopt;
  Trajectory traj_out;

  options_trajopt.solver_id =
      static_cast<int>(SOLVER::traj_opt_free_time_linear);

  Info_out_idbastar info_out_idbastar;

  idbA(problem, options_idbas, options_dbastar, options_trajopt, traj_out,
       info_out_idbastar);

  BOOST_TEST(info_out_idbastar.solved == true);
  BOOST_TEST(info_out_idbastar.cost < 5.);
}


BOOST_AUTO_TEST_CASE(bugtrap_uni1_heu) {


  Problem problem( "../benchmark/unicycle_first_order_0/bugtrap_0.yaml" );
  Options_dbastar options_dbastar;
options_dbastar.heuristic = 1;

    options_dbastar.motionsFile = "../cloud/motionsV2/good/unicycle1_v0/unicycle1_v0__ispso__2023_04_03__14_56_57.bin" ;

    options_dbastar.max_size_heu_map = 500;








}







BOOST_AUTO_TEST_CASE(idbastar_cli) {

  std::string cmd =
      "make main_idbastar && ./main_idbastar --env_file "
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml "
      "--motionsFile ../cloud/motions/unicycle_first_order_0_sorted.msgpack "
      "--results_file results_idbastar.yaml --timelimit 10 --solver_id 1";

  std::cout << "Running cpp command: \n" << cmd << std::endl;
  int out = std::system(cmd.c_str());
  BOOST_TEST(out == 0);
}
