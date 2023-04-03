#include "ompl_geo.hpp"
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

#define BOOST_TEST_MODULE test module name
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(parallel_park_1) {

  Options_geo options_geo;
  Options_trajopt options_trajopt;

  // TODO: this will not work (not ready for new way of getting init guess).
  // options_trajopt.solver_id = static_cast<int>(SOLVER::time_search_traj_opt);
  options_trajopt.solver_id =
      static_cast<int>(SOLVER::traj_opt_free_time_linear);

  Trajectory traj_out;
  Info_out info_out_omplgeo;

  Problem problem("../benchmark/unicycle_first_order_0/parallelpark_0.yaml");

  solve_ompl_geometric(problem, options_geo, options_trajopt, traj_out,
                       info_out_omplgeo);

  BOOST_TEST(info_out_omplgeo.solved == true);
  BOOST_TEST(info_out_omplgeo.cost < 5.);
}

BOOST_AUTO_TEST_CASE(ompl_geo_cli) {

  std::string cmd =
      "make main_ompl_geometric && ./main_ompl_geometric --env_file "
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml "
      "--results_file results_geo.yaml --timelimit 10";

  std::cout << "Running cpp command: \n" << cmd << std::endl;
  int out = std::system(cmd.c_str());
  BOOST_TEST(out == 0);
}
