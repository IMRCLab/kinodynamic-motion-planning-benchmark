#include "dbrrt.hpp"
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

#include "../kdtree.hpp"
// #include "nigh/impl/kdtree_median/strategy.hpp"
#include "spdlog/spdlog.h"
#include <nigh/kdtree_batch.hpp>
#include <nigh/kdtree_median.hpp>
#include <nigh/se3_space.hpp>
#include <nigh/so2_space.hpp>
#include <nigh/so3_space.hpp>

BOOST_AUTO_TEST_CASE(hello_world) {

  int a = 0;
  int b = 1;
  int c = 2;

#define X(a) std::cout << a << std::endl;

  APPLYXn(a, b, c);

  APPLYXn(a, b);
  APPLYXn(a, b, c, a, b, c);
  APPLYXn(a, b, c, a, b, c, a, a, a, a, a, a, a, a, a, a);

#undef X
}

BOOST_AUTO_TEST_CASE(test_uni1_bugtrap_with_opt) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char **argv = boost::unit_test::framework::master_test_suite().argv;

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");

  Options_dbrrt options_dbrrt;
  options_dbrrt.choose_first_motion_valid = true;
  options_dbrrt.search_timelimit = 1e5; // in ms
  options_dbrrt.max_motions = 30;
  options_dbrrt.debug = false;
  options_dbrrt.max_expands = 30000;
  options_dbrrt.cost_bound = 1e6;
  options_dbrrt.delta = .3;
  options_dbrrt.goal_bias = .1;
  options_dbrrt.use_nigh_nn = true;
  options_dbrrt.fix_seed = true;
  options_dbrrt.do_optimization = 1;

  options_dbrrt.motionsFile = "../cloud/motionsV2/good/unicycle1_v0/"
                              "unicycle1_v0__ispso__2023_04_03__14_56_57.bin";

  po::options_description desc("Allowed options");
  options_dbrrt.add_options(desc);

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
  }

  std::vector<Motion> motions;

  load_motion_primitives_new(
      options_dbrrt.motionsFile, *robot_factory_ompl(problem), motions,
      options_dbrrt.max_motions * 2, options_dbrrt.cut_actions, false,
      options_dbrrt.check_cols);

  options_dbrrt.motions_ptr = &motions;
  Trajectory traj_out;
  Info_out out_info;

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 0; // experimental solver
  dbrrt(problem, options_dbrrt, options_trajopt, traj_out, out_info);

  BOOST_TEST(out_info.cost < 60);
  BOOST_TEST(out_info.solved);
  BOOST_TEST(out_info.cost_raw);
  BOOST_TEST(out_info.trajs_raw.size() == 1);
  BOOST_TEST(out_info.trajs_opt.size() == 1);

  // now with aorrt
  options_dbrrt.ao_rrt = true;

  Info_out out_info2;
  dbrrt(problem, options_dbrrt, options_trajopt, traj_out, out_info2);
  BOOST_TEST(out_info2.cost < 50);
  BOOST_TEST(out_info2.solved);
  BOOST_TEST(out_info2.solved_raw);
  BOOST_TEST(out_info2.cost_raw);
  BOOST_TEST(out_info2.trajs_raw.size() > 1);
  BOOST_TEST(out_info2.trajs_opt.size() > 1);
}

BOOST_AUTO_TEST_CASE(test_uni1_bugtrap_aorrt) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char **argv = boost::unit_test::framework::master_test_suite().argv;

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");

  Options_dbrrt options_dbrrt;
  options_dbrrt.ao_rrt_rebuild_tree = true;
  options_dbrrt.choose_first_motion_valid = true;
  options_dbrrt.cost_weight = 0.1;
  options_dbrrt.search_timelimit = 1e5; // in ms
  options_dbrrt.max_motions = 30;
  options_dbrrt.debug = false;
  options_dbrrt.max_expands = 30000;
  options_dbrrt.ao_rrt = true;
  options_dbrrt.cost_bound = 100;
  options_dbrrt.delta = .3;
  options_dbrrt.goal_bias = .1;
  options_dbrrt.use_nigh_nn = true;
  options_dbrrt.fix_seed = false;

  options_dbrrt.motionsFile = "../cloud/motionsV2/good/unicycle1_v0/"
                              "unicycle1_v0__ispso__2023_04_03__14_56_57.bin";

  po::options_description desc("Allowed options");
  options_dbrrt.add_options(desc);

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
  }

  std::vector<Motion> motions;

  load_motion_primitives_new(
      options_dbrrt.motionsFile, *robot_factory_ompl(problem), motions,
      options_dbrrt.max_motions * 2, options_dbrrt.cut_actions, false,
      options_dbrrt.check_cols);

  options_dbrrt.motions_ptr = &motions;

  size_t num_of_runs = 10;

  spdlog::info("number of runs: {}", num_of_runs);

  std::vector<double> costs_aorrt;
  std::vector<bool> solved_aorrt;
  Stopwatch s1;
  for (size_t i = 0; i < num_of_runs; i++) {
    Info_out out_info;
    Trajectory traj_out;
    dbrrt(problem, options_dbrrt, Options_trajopt(), traj_out, out_info);
    costs_aorrt.push_back(out_info.cost_raw);
    solved_aorrt.push_back(out_info.solved_raw);
  }
  double time_ao = s1.elapsed_ms() / num_of_runs;

  std::vector<double> costs_rrt;
  std::vector<bool> solved_rrt;
  Stopwatch s2;
  for (size_t i = 0; i < num_of_runs; i++) {
    Info_out out_info;
    Trajectory traj_out;
    options_dbrrt.ao_rrt = false;
    dbrrt(problem, options_dbrrt, Options_trajopt(), traj_out, out_info);
    costs_rrt.push_back(out_info.cost_raw);
    solved_rrt.push_back(out_info.solved_raw);
  }
  double time_rrt = s2.elapsed_ms() / num_of_runs;

  double cost_aorrt =
      std::accumulate(costs_aorrt.begin(), costs_aorrt.end(), 0.0) /
      num_of_runs;
  double cost_rrt =
      std::accumulate(costs_rrt.begin(), costs_rrt.end(), 0.0) / num_of_runs;

  bool all_solved_rrt = std::all_of(solved_rrt.begin(), solved_rrt.end(),
                                    [](bool v) { return v; });

  bool all_solved_ao = std::all_of(solved_aorrt.begin(), solved_aorrt.end(),
                                   [](bool v) { return v; });

  spdlog::info("all_solved_rrt: {}", all_solved_rrt);
  spdlog::info("all_solved_aorrt: {}", all_solved_ao);

  spdlog::info("average cost aorrt: {}", cost_aorrt);
  spdlog::info("average cost rrt: {}", cost_rrt);
  spdlog::info("average time_ao: {}", time_ao);
  spdlog::info("average time_rrt: {}", time_rrt);

  BOOST_TEST(all_solved_rrt);
  BOOST_TEST(all_solved_ao);

  BOOST_TEST(cost_aorrt < 45);
  BOOST_TEST(cost_rrt < 60);

  spdlog::info("Test in time in Release Mode");

  BOOST_WARN(time_rrt < 50.);
  BOOST_WARN(time_ao < 300.);
}

BOOST_AUTO_TEST_CASE(test_uni1_bugtrap) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char **argv = boost::unit_test::framework::master_test_suite().argv;

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");
  Options_dbrrt options_dbrrt;
  options_dbrrt.choose_first_motion_valid = true;
  options_dbrrt.search_timelimit = 1e5; // in ms
  options_dbrrt.max_motions = 30;
  options_dbrrt.debug = true;
  options_dbrrt.max_expands = 10000;
  options_dbrrt.delta = .3;
  options_dbrrt.goal_bias = .1;
  options_dbrrt.use_nigh_nn = true;
  options_dbrrt.motionsFile = "../cloud/motionsV2/good/unicycle1_v0/"
                              "unicycle1_v0__ispso__2023_04_03__14_56_57.bin";

  po::options_description desc("Allowed options");
  options_dbrrt.add_options(desc);

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
  }

  Info_out out_info;
  Trajectory traj_out;
  dbrrt(problem, options_dbrrt, Options_trajopt(), traj_out, out_info);
  BOOST_TEST(out_info.solved_raw);
  BOOST_TEST(out_info.cost_raw < 100.);
}

BOOST_AUTO_TEST_CASE(test_uni2_bugtrap) {

  int argc = boost::unit_test::framework::master_test_suite().argc;
  char **argv = boost::unit_test::framework::master_test_suite().argv;

  Problem problem("../benchmark/unicycle_second_order_0/bugtrap_0.yaml");
  Options_dbrrt options_dbrrt;
  options_dbrrt.choose_first_motion_valid = true;
  options_dbrrt.search_timelimit = 1e5; // in ms
  options_dbrrt.max_motions = 200;
  options_dbrrt.max_expands = 20000;
  // options_dbrrt.ao_rrt = true;
  options_dbrrt.delta = .25;
  options_dbrrt.goal_region = .5;
  options_dbrrt.goal_bias = .1;
  options_dbrrt.use_nigh_nn = true;
  options_dbrrt.motionsFile =
      "../cloud/motionsV2/good/unicycle2_v0/"
      "unicycle2_v0__ispso__2023_04_03__15_36_01.bin.im.bin.im.bin";

  po::options_description desc("Allowed options");
  options_dbrrt.add_options(desc);

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
  }

  Info_out out_info;
  Trajectory traj_out;
  dbrrt(problem, options_dbrrt, Options_trajopt(), traj_out, out_info);
  BOOST_TEST(out_info.solved_raw);
  BOOST_TEST(out_info.cost_raw < 100.);
}

BOOST_AUTO_TEST_CASE(test_quad2d_bugtrap) {

  Problem problem("../benchmark/quad2d/quad_bugtrap.yaml");
  Options_dbrrt options_dbrrt;
  options_dbrrt.choose_first_motion_valid = true;
  options_dbrrt.search_timelimit = 1e5; // in ms
  options_dbrrt.max_motions = 400;
  options_dbrrt.max_expands = 50000;
  options_dbrrt.delta = .3;
  options_dbrrt.debug = false;
  options_dbrrt.goal_region = .5;
  options_dbrrt.goal_bias = .1;
  options_dbrrt.use_nigh_nn = true;
  options_dbrrt.motionsFile =
      "../cloud/motionsV2/good/quad2d_v0/quad2d_v0_all_im.bin.sp.bin.ca.bin";
  // "../cloud/motionsV2/good/unicycle2_v0/"
  // "unicycle2_v0__ispso__2023_04_03__15_36_01.bin.im.bin.im.bin";
  Info_out out_info;
  Trajectory traj_out;
  dbrrt(problem, options_dbrrt, Options_trajopt(), traj_out, out_info);
  BOOST_TEST(out_info.solved_raw);
  CSTR_(out_info.cost_raw);
  BOOST_TEST(out_info.cost_raw < 100.);
}
