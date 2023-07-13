
// #include <boost/test/unit_test_suite.hpp>
// #define BOOST_TEST_DYN_LINK
// #include <boost/test/unit_test.hpp>

// see
// https://www.boost.org/doc/libs/1_81_0/libs/test/doc/html/boost_test/usage_variants.html
// #define BOOST_TEST_MODULE test module name
#define BOOST_TEST_MODULE test module name
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>


#include "ompl_sst.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>



BOOST_AUTO_TEST_CASE(parallel_park_1) {

  Options_sst options_ompl_sst;
  Options_trajopt options_trajopt;
  Trajectory traj_out;
  Info_out info_out_omplgeo;

  options_ompl_sst.timelimit = 10;
  Problem problem("../benchmark/unicycle_first_order_0/parallelpark_0.yaml");

  solve_sst(problem, options_ompl_sst, options_trajopt, traj_out,
            info_out_omplgeo);

  BOOST_TEST(info_out_omplgeo.solved == true);
  BOOST_TEST(info_out_omplgeo.cost <= 4);
}

BOOST_AUTO_TEST_CASE(sst_cli) {

  std::string cmd = "make main_ompl && ./main_ompl --env_file "
                    "../benchmark/unicycle_first_order_0/parallelpark_0.yaml "
                    "--results_file results_sst.yaml --timelimit 10";

  std::cout << "Running cpp command: \n" << cmd << std::endl;
  int out = std::system(cmd.c_str());
  BOOST_TEST(out == 0);
}
