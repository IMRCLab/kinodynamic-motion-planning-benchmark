#include "dbastar.hpp"
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

  Options_dbastar options_dbastar;

  Problem problem("../benchmark/unicycle_first_order_0/parallelpark_0.yaml");

  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";
  options_dbastar.max_motions = 2000;
  options_dbastar.delta = 0.2;
  options_dbastar.epsilon = 1.;
  options_dbastar.alpha = 0.3;
  options_dbastar.filterDuplicates = false;
  options_dbastar.maxCost = 100;
  options_dbastar.resolution = 0.1;
  options_dbastar.cost_delta_factor = 1.;
  options_dbastar.rebuild_every = 5000;
  options_dbastar.cut_actions = false;
  options_dbastar.use_landmarks = false;
  options_dbastar.num_sample_trials = 1000;

  {

    options_dbastar.heuristic = 0;
    Trajectory traj;
    Out_info_db out_db;
    dbastar(problem, options_dbastar, traj, out_db);
    std::cout << "***" << std::endl;
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 3.5);
    BOOST_TEST(out_db.cost_with_delta_time < 3.5);
  }
  // TODO: FIX this!!!

  // very slow in debug mode!
  {
    options_dbastar.heuristic = 1;
    Trajectory traj;
    Out_info_db out_db;
    dbastar(problem, options_dbastar, traj, out_db);
    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 3.5);
    BOOST_TEST(out_db.cost_with_delta_time < 3.5);
  }
}

BOOST_AUTO_TEST_CASE(bugtrap_1) {

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 2000;

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");
  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  options_dbastar.delta = 0.3;
  options_dbastar.epsilon = 1.;
  options_dbastar.alpha = 0.3;
  options_dbastar.filterDuplicates = false;
  options_dbastar.maxCost = 100;
  options_dbastar.resolution = 0.1;
  options_dbastar.cost_delta_factor = 1.;
  options_dbastar.rebuild_every = 5000;
  options_dbastar.cut_actions = false;
  options_dbastar.use_landmarks = false;

  {

    options_dbastar.heuristic = 0;

    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 35);
    BOOST_TEST(out_db.cost_with_delta_time < 35);
  }

  {
    options_dbastar.heuristic = 1;

    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 35);
    BOOST_TEST(out_db.cost_with_delta_time < 35);
  }
}

BOOST_AUTO_TEST_CASE(t_kink) {

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 200;

  Problem problem("../benchmark/unicycle_first_order_0/kink_0.yaml");

  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";
  options_dbastar.delta = 0.4;
  options_dbastar.epsilon = 1.;
  options_dbastar.alpha = 0.3;
  options_dbastar.filterDuplicates = false;
  options_dbastar.maxCost = 100;
  options_dbastar.resolution = 0.1;
  options_dbastar.cost_delta_factor = 1.;
  options_dbastar.rebuild_every = 5000;
  options_dbastar.cut_actions = false;
  options_dbastar.use_landmarks = false;
  options_dbastar.num_sample_trials = 5000;

  {
    options_dbastar.heuristic = 0;
    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 25);
    BOOST_TEST(out_db.cost_with_delta_time < 25);
  }

  {
    options_dbastar.heuristic = 1;
    Out_info_db out_db;

    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 25);
    BOOST_TEST(out_db.cost_with_delta_time < 25);
  }
}

BOOST_AUTO_TEST_CASE(t_parallel2) {

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 5000;

  Problem problem("../benchmark/unicycle_second_order_0/parallelpark_0.yaml");
  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_second_order_0_sorted.msgpack";

  options_dbastar.delta = 0.4;
  options_dbastar.epsilon = 1.;
  options_dbastar.alpha = 0.3;
  options_dbastar.filterDuplicates = false;
  options_dbastar.maxCost = 100;
  options_dbastar.resolution = 0.1;
  options_dbastar.cost_delta_factor = 1.;
  options_dbastar.rebuild_every = 5000;
  options_dbastar.cut_actions = false;
  options_dbastar.use_landmarks = false;
  options_dbastar.num_sample_trials = 5000;

  {
    options_dbastar.heuristic = 0;
    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);
    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 7.5);
    BOOST_TEST(out_db.cost_with_delta_time < 7.5);
  }

  {
    options_dbastar.heuristic = 1;

    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 7.5);
    BOOST_TEST(out_db.cost_with_delta_time < 7.5);
  }
}

BOOST_AUTO_TEST_CASE(t_new_modes) {

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 200;

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");

  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  options_dbastar.delta = 0.4;
  options_dbastar.epsilon = 1.;
  options_dbastar.alpha = 0.3;
  options_dbastar.filterDuplicates = false;
  options_dbastar.cost_delta_factor = 1.;
  options_dbastar.rebuild_every = 5000;
  options_dbastar.num_sample_trials = 2000;
  options_dbastar.heuristic = 1;

  {
    options_dbastar.add_node_if_better = true;

    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;

    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 30);
    BOOST_TEST(out_db.cost_with_delta_time < 30);
  }
  {
    options_dbastar.add_node_if_better = false;
    options_dbastar.add_after_expand = true;

    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);

    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;

    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 30);
    BOOST_TEST(out_db.cost_with_delta_time < 30);
  }
}

BOOST_AUTO_TEST_CASE(t_bug2) {
  // (opti) ⋊> ~/s/w/k/build on dev ⨯ make -j8 &&      ./main_dbastar
  //
  //

  Options_dbastar options_dbastar;

  Problem problem("../benchmark/unicycle_second_order_0/bugtrap_0.yaml");
  options_dbastar.motionsFile =
      "../cloud/motions/unicycle_second_order_0_sorted.msgpack";

  options_dbastar.max_motions = 1000;
  options_dbastar.delta = .4;
  options_dbastar.epsilon = 1.;
  options_dbastar.alpha = .3;
  options_dbastar.heuristic = 1;
  options_dbastar.cost_delta_factor = 1.;
  options_dbastar.rebuild_every = 5000;
  options_dbastar.num_sample_trials = 5000;

  {
    Out_info_db out_db;
    Trajectory traj;
    dbastar(problem, options_dbastar, traj, out_db);
    std::cout << "***" << std::endl;
    out_db.print(std::cout);
    std::cout << "***" << std::endl;

    BOOST_TEST(out_db.solved);
    BOOST_TEST(out_db.cost < 70);
    BOOST_TEST(out_db.cost_with_delta_time < 70);
  }
}

// CONTINUE HERE -- visualization and primitives for 3d case!!

// this works
// (croco) ⋊> ~/s/w/k/build on dev ⨯ make && ./croco_main --yaml_solver_file
// ../solvers_timeopt/mpcc_v0.yaml --env_file
// ../benchmark/quad2d/quad_obs_column.yaml --out out.yaml --init_guess
// ../test/quad2d/quad2d_obs_column_init_guess .yaml > quim.txt

// (croco) ⋊> ~/s/w/k/build on dev ⨯ make &&   ./croco_main --yaml_solver_file
// ../solvers_timeopt/mpcc_v0.yaml --env_file
// ../benchmark/quad2d/quad_obs_column.yaml --out out.yaml --init_guess
// ../test/quad2d/quad2d_obs_column_init_gue ss.yaml --new_format 1

// (croco) ⋊> ~/s/w/k/build on dev ⨯ make &&  ./main_dbastar --inputFile
// ../benchmark/quad2d/quad_obs_recovery.yaml  --motionsFile
// ../cloud/motionsX/quad2d_sorted_2023-02-26--13-40-CONCAT.yaml.split_v0.msgpack
// --outFile out.yaml  - -delta .8  --max_motions 1000 --maxCost 1100000.0
// --cfg_file ../params_v0.yaml

// quad 2d
//  (croco) ⋊> ~/s/w/k/build on dev ⨯ make &&  ./main_dbastar --inputFile
//  ../benchmark/quad2d/quad_obs_column.yaml  --motionsFile
//  ../cloud/motionsX/quad2d_sorted_2023-02-26--13-40-CONCAT.yaml.split_v0.msgpack
//  --outFile out.yaml  --d elta .8  --max_motions 1000 --maxCost 1100000.0
//  --cfg_file ../params_v0.yaml

BOOST_AUTO_TEST_CASE(t_kink2) {

  // (opti) ⋊> ~/s/w/k/build on dev ⨯ make -j8 &&      ./main_dbastar
  // --inputFile ../benchmark/unicycle_second_order_0/kink_0.yaml --motionsFile
  // ../cloud/motions/unicycle_second_order_0_sorted.msgpack --max_motions 1000
  // --outputFile
  //  qdbg/result_dbastar.yaml --delta 0.4  --epsilon 1.0 --alpha 0.3
  //  --filterDuplicates False --maxCost 1000000.0 --heuristic 1 --resolution
  //  0.1
  //  --cost_delta_factor 1.0  --rebuild_every 5000 --num_sample 5000
  //  --cut_actions false --max
  // _expands 100000 --use_landmarks 0
}

// IDEAS: the connect radius in the roadmap can depend on the dimension of the
// problem. -- is always have the check step at some resolution. Work on the
// quadrotor example! -- I need to generate motions primitives, and good
// visualization tools.
