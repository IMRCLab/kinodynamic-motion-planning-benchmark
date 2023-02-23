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

  Options_db options_db;
  Inout_db inout_db;
  options_db.max_motions = 2000;

  inout_db.inputFile =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  inout_db.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  options_db.delta = 0.2;
  options_db.epsilon = 1.;
  options_db.alpha = 0.3;
  options_db.filterDuplicates = false;
  options_db.maxCost = 100;
  options_db.resolution = 0.1;
  options_db.cost_delta_factor = 1.;
  options_db.rebuild_every = 5000;
  options_db.cut_actions = false;
  options_db.use_landmarks = false;

  {

    options_db.new_heu = 0;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 3.5);
    BOOST_TEST(inout_db.cost_with_delta_time < 3.5);
  }

  {
    options_db.new_heu = 1;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 3.5);
    BOOST_TEST(inout_db.cost_with_delta_time < 3.5);
  }
}

BOOST_AUTO_TEST_CASE(bugtrap_1) {

  Options_db options_db;
  Inout_db inout_db;
  options_db.max_motions = 2000;

  inout_db.inputFile = "../benchmark/unicycle_first_order_0/bugtrap_0.yaml";
  inout_db.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  options_db.delta = 0.3;
  options_db.epsilon = 1.;
  options_db.alpha = 0.3;
  options_db.filterDuplicates = false;
  options_db.maxCost = 100;
  options_db.resolution = 0.1;
  options_db.cost_delta_factor = 1.;
  options_db.rebuild_every = 5000;
  options_db.cut_actions = false;
  options_db.use_landmarks = false;

  {

    options_db.new_heu = 0;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 35);
    BOOST_TEST(inout_db.cost_with_delta_time < 35);
  }

  {
    options_db.new_heu = 1;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 35);
    BOOST_TEST(inout_db.cost_with_delta_time < 35);
  }
}

BOOST_AUTO_TEST_CASE(t_kink) {

  Options_db options_db;
  Inout_db inout_db;
  options_db.max_motions = 200;

  inout_db.inputFile = "../benchmark/unicycle_first_order_0/kink_0.yaml";
  inout_db.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  options_db.delta = 0.4;
  options_db.epsilon = 1.;
  options_db.alpha = 0.3;
  options_db.filterDuplicates = false;
  options_db.maxCost = 100;
  options_db.resolution = 0.1;
  options_db.cost_delta_factor = 1.;
  options_db.rebuild_every = 5000;
  options_db.cut_actions = false;
  options_db.use_landmarks = false;
  options_db.num_sample_trials = 5000;

  {
    options_db.new_heu = 0;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 25);
    BOOST_TEST(inout_db.cost_with_delta_time < 25);
  }

  {
    options_db.new_heu = 1;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 25);
    BOOST_TEST(inout_db.cost_with_delta_time < 25);
  }
}

BOOST_AUTO_TEST_CASE(t_parallel2) {

  Options_db options_db;
  Inout_db inout_db;
  options_db.max_motions = 5000;

  inout_db.inputFile =
      "../benchmark/unicycle_second_order_0/parallelpark_0.yaml";
  inout_db.motionsFile =
      "../cloud/motions/unicycle_second_order_0_sorted.msgpack";

  options_db.delta = 0.4;
  options_db.epsilon = 1.;
  options_db.alpha = 0.3;
  options_db.filterDuplicates = false;
  options_db.maxCost = 100;
  options_db.resolution = 0.1;
  options_db.cost_delta_factor = 1.;
  options_db.rebuild_every = 5000;
  options_db.cut_actions = false;
  options_db.use_landmarks = false;
  options_db.num_sample_trials = 5000;

  {
    options_db.new_heu = 0;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 7.5);
    BOOST_TEST(inout_db.cost_with_delta_time < 7.5);
  }

  {
    options_db.new_heu = 1;
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;
    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 7.5);
    BOOST_TEST(inout_db.cost_with_delta_time < 7.5);
  }
}

BOOST_AUTO_TEST_CASE(t_new_modes) {

  Options_db options_db;
  Inout_db inout_db;
  options_db.max_motions = 200;

  inout_db.inputFile = "../benchmark/unicycle_first_order_0/bugtrap_0.yaml";
  inout_db.motionsFile =
      "../cloud/motions/unicycle_first_order_0_sorted.msgpack";

  options_db.delta = 0.4;
  options_db.epsilon = 1.;
  options_db.alpha = 0.3;
  options_db.filterDuplicates = false;
  options_db.cost_delta_factor = 1.;
  options_db.rebuild_every = 5000;
  options_db.num_sample_trials = 2000;
  options_db.new_heu = 1;

  {
    options_db.add_node_if_better = true;

    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;

    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 30);
    BOOST_TEST(inout_db.cost_with_delta_time < 30);
  }
  {
    options_db.add_node_if_better = false;
    options_db.add_after_expand = true;

    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;

    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 30);
    BOOST_TEST(inout_db.cost_with_delta_time < 30);
  }
}

BOOST_AUTO_TEST_CASE(t_bug2) {
  // (opti) ⋊> ~/s/w/k/build on dev ⨯ make -j8 &&      ./main_dbastar
  //
  //

  Options_db options_db;
  Inout_db inout_db;

  inout_db.inputFile = "../benchmark/unicycle_second_order_0/bugtrap_0.yaml";
  inout_db.motionsFile =
      "../cloud/motions/unicycle_second_order_0_sorted.msgpack";
  inout_db.outFile = "out.yaml";

  options_db.max_motions = 1000;
  options_db.delta = .4;
  options_db.epsilon = 1.;
  options_db.alpha = .3;
  options_db.new_heu=1;
  options_db.cost_delta_factor = 1.;
  options_db.rebuild_every = 5000;
  options_db.num_sample_trials = 5000;

  {
    solve(options_db, inout_db);
    std::cout << "***" << std::endl;
    inout_db.print(std::cout);
    std::cout << "***" << std::endl;

    BOOST_TEST(inout_db.solved);
    BOOST_TEST(inout_db.cost < 70);
    BOOST_TEST(inout_db.cost_with_delta_time < 70);
  }




}

BOOST_AUTO_TEST_CASE(t_kink2) {

  // (opti) ⋊> ~/s/w/k/build on dev ⨯ make -j8 &&      ./main_dbastar
  // --inputFile ../benchmark/unicycle_second_order_0/kink_0.yaml --motionsFile
  // ../cloud/motions/unicycle_second_order_0_sorted.msgpack --max_motions 1000
  // --outputFile
  //  qdbg/result_dbastar.yaml --delta 0.4  --epsilon 1.0 --alpha 0.3
  //  --filterDuplicates False --maxCost 1000000.0 --new_heu 1 --resolution 0.1
  //  --cost_delta_factor 1.0  --rebuild_every 5000 --num_sample 5000
  //  --cut_actions false --max
  // _expands 100000 --use_landmarks 0
}

// IDEAS: the connect radius in the roadmap can depend on the dimension of the
// problem. -- is always have the check step at some resolution. Work on the
// quadrotor example! -- I need to generate motions primitives, and good
// visualization tools.
