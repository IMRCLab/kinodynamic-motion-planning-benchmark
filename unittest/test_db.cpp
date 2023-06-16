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

#include "../kdtree.hpp"
// #include "nigh/impl/kdtree_median/strategy.hpp"
#include <nigh/kdtree_batch.hpp>
#include <nigh/kdtree_median.hpp>
#include <nigh/se3_space.hpp>
#include <nigh/so2_space.hpp>
#include <nigh/so3_space.hpp>

// quad2d is working!

BOOST_AUTO_TEST_CASE(t_check_cols) {

  Trajectory traj;
  traj.read_from_yaml("quad2dpole_window_col2.yaml");
  // continue here
  Problem problem("../benchmark/quad2dpole/window.yaml");


  std::shared_ptr<Model_robot> model_robot =
      robot_factory(robot_type_to_path(problem.robotType).c_str());
  load_env_quim(*model_robot, problem);

  traj.check(model_robot, true);


}

BOOST_AUTO_TEST_CASE(test_distance) {

  Problem problem("../benchmark/quad2dpole/up.yaml");

  // Compound state [
  // RealVectorState [-0.323241 -0.116652]
  // SO2State [-0.344584]
  // SO2State [2.78135]
  // RealVectorState [0.159636 -0.767126]
  // RealVectorState [-0.0622687]
  // RealVectorState [-0.00392124]
  // ]
  auto robot = robot_factory_ompl(problem);

  Eigen::VectorXd x(8), g(8);
  x << 0.323241, -0.116652, -0.344584, 2.78135, 0.159636, -0.767126, -0.0622687,
      -0.00392124;

  g = problem.goal;
  //
  // x(6) = 0;
  // g(6) = 0;
  // x(7) = 0;
  // g(7) = 0;
  //
  //
  // x(2) = 0;
  // g(2) = 0;
  //
  //
  // x(3) = 0;
  // g(3) = 0;

  auto xompl = robot->si_->allocState();
  auto gompl = robot->si_->allocState();

  CSTR_(robot->diff_model->distance(x, g));

  robot->fromEigen(xompl, x);
  robot->fromEigen(gompl, g);

  robot->si_->printState(xompl);
  robot->si_->printState(gompl);

  CSTR_(robot->si_->distance(xompl, gompl));
}

BOOST_AUTO_TEST_CASE(tt_quad2dpole) {

  if (false) {
    // Problem problem("../benchmark/quad2dpole/move_with_down.yaml");
    Problem problem("../benchmark/quad2dpole/move_with_up.yaml");
    Options_dbastar options_dbastar;
    options_dbastar.max_motions = 10000;
    options_dbastar.search_timelimit = 60000;
    options_dbastar.delta = .9;
    options_dbastar.cost_delta_factor = 1;
    options_dbastar.delta_factor_goal = 1;
    options_dbastar.use_nigh_nn = 1;
    options_dbastar.motionsFile = "../build/quad2dpole_all.bin.sp1.bin.ca.bin";
    options_dbastar.new_invariance = true;
    options_dbastar.use_collision_shape = false;
    options_dbastar.limit_branching_factor = 30;

    Out_info_db out_info_db;
    Trajectory traj_out;
    dbastar(problem, options_dbastar, traj_out, out_info_db);
    CSTR_(out_info_db.cost);
    BOOST_TEST(out_info_db.solved);
  }

  if (true) {
    // Problem problem("../benchmark/quad2dpole/move_with_down.yaml");
    // Problem problem("../benchmark/quad2dpole/move_with_up.yaml");
    Problem problem("../benchmark/quad2dpole/up.yaml");
    // Problem problem("../benchmark/quad2dpole/up_obs.yaml");
    Options_dbastar options_dbastar;
    options_dbastar.max_motions = 4000;
    // options_dbastar.max_motions = 2000;
    options_dbastar.search_timelimit = 60000;
    options_dbastar.delta = .8;
    options_dbastar.cost_delta_factor = 1;
    options_dbastar.delta_factor_goal = 1;
    options_dbastar.use_nigh_nn = 1;
    options_dbastar.motionsFile = "../build/quad2dpole_all.bin.sp1.bin.ca.bin";
    // options_dbastar.motionsFile = "../build/quad2dpole_all.bin.im.bin.sp1.bin.ca.bin";
    options_dbastar.new_invariance = true;
    options_dbastar.use_collision_shape = false;
    options_dbastar.limit_branching_factor = 15;

    Out_info_db out_info_db;
    Trajectory traj_out;
    dbastar(problem, options_dbastar, traj_out, out_info_db);
    CSTR_(out_info_db.cost);
    BOOST_TEST(out_info_db.solved);
  }
  if (false) {
    // Problem problem("../benchmark/quad2dpole/window_easy.yaml");
    // Problem problem("../benchmark/quad2dpole/window.yaml");
    Problem problem("../benchmark/quad2dpole/window_hard.yaml");
    Options_dbastar options_dbastar;
    options_dbastar.read_from_yaml("../tmp.yaml");
    // options_dbastar.max_motions = 10000;
    // options_dbastar.search_timelimit = 60000;
    // options_dbastar.delta = .8;
    // options_dbastar.cost_delta_factor = 1;
    // options_dbastar.delta_factor_goal = 1;
    // options_dbastar.use_nigh_nn = 1;
    // options_dbastar.motionsFile =
    // "../build/quad2dpole_all.bin.sp1.bin.ca.bin";
    // options_dbastar.new_invariance = true;
    // options_dbastar.use_collision_shape = false;
    // options_dbastar.limit_branching_factor = 30;

    Out_info_db out_info_db;
    Trajectory traj_out;
    dbastar(problem, options_dbastar, traj_out, out_info_db);
    CSTR_(out_info_db.cost);
    BOOST_TEST(out_info_db.solved);
  }
  if (false){
    // Problem problem("../benchmark/quad2dpole/window_easy.yaml");
    Problem problem("../benchmark/quad2dpole/down.yaml");
    Options_dbastar options_dbastar;
    options_dbastar.read_from_yaml("../tmp.yaml");
    // options_dbastar.max_motions = 10000;
    // options_dbastar.search_timelimit = 60000;
    // options_dbastar.delta = .8;
    // options_dbastar.cost_delta_factor = 1;
    // options_dbastar.delta_factor_goal = 1;
    // options_dbastar.use_nigh_nn = 1;
    // options_dbastar.motionsFile =
    // "../build/quad2dpole_all.bin.sp1.bin.ca.bin";
    // options_dbastar.new_invariance = true;
    // options_dbastar.use_collision_shape = false;
    // options_dbastar.limit_branching_factor = 30;

    Out_info_db out_info_db;
    Trajectory traj_out;
    dbastar(problem, options_dbastar, traj_out, out_info_db);
    CSTR_(out_info_db.cost);
    BOOST_TEST(out_info_db.solved);

  }
}

BOOST_AUTO_TEST_CASE(tt_quad3_recovery_with_obs) {

  // Problem problem("../benchmark/quadrotor_0/window.yaml");
  Problem problem("../benchmark/quadrotor_0/recovery_with_obs.yaml");

  Options_dbastar options_dbastar;
  // options_dbastar.max_motions = 1e5;
  // options_dbastar.max_motions = 200000;
  options_dbastar.max_motions = 5000;
  // options_dbastar.max_motions = 50000;
  // options_dbastar.max_motions = 5000;
  options_dbastar.search_timelimit = 60000;
  // options_dbastar.delta = .8;
  options_dbastar.delta = .7;
  options_dbastar.cost_delta_factor = 1;
  options_dbastar.delta_factor_goal = 1;
  options_dbastar.use_nigh_nn = 1;
  // options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp.bin";
  // options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp2.bin.ca.bin";
  // The improvement seems to help!!!
  // options_dbastar.motionsFile =
  //     "../build/quad3d_v0_all3.bin.im.bin.sp.bin.ca.bin";
  options_dbastar.motionsFile =
      "../build/quad3d_v0_all3.bin.im.bin.sp1.bin.ca.bin";
  options_dbastar.new_invariance = true;

  options_dbastar.use_collision_shape = false;
  options_dbastar.limit_branching_factor = 30;

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(tt_quad3_ompl) {

  // Problem problem("../benchmark/quadrotor_ompl/empty_easy_2_ompl.yaml");
  // Problem problem("../benchmark/quadrotor_ompl/recovery_ompl.yaml");
  Problem problem("../benchmark/quadrotor_ompl/one_obs.yaml");
  // TODO: Continue here!!

  Options_dbastar options_dbastar;
  // options_dbastar.max_motions = 1e6;
  options_dbastar.max_motions = 1e4;
  options_dbastar.search_timelimit = 60000 * 10;
  // options_dbastar.delta = .8;
  options_dbastar.delta = 1.;
  options_dbastar.cost_delta_factor = 1;
  options_dbastar.delta_factor_goal = 1;
  options_dbastar.use_nigh_nn = 1;
  // options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp.bin";
  options_dbastar.motionsFile = "../build/quad3dompl_all.bin.sp.bin.ca.bin";
  options_dbastar.new_invariance = true;
  options_dbastar.use_collision_shape = false;
  options_dbastar.limit_branching_factor = 15;

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(tt_quad3_one_obs_new) {

  // Problem problem("../benchmark/quadrotor_0/quad_one_obs.yaml");
  // Problem problem("../benchmark/quadrotor_0/recovery.yaml");
  // quad_one_obs.yaml");

  Problem problem("../benchmark/quadrotor_0/window.yaml");

  Options_dbastar options_dbastar;
  // options_dbastar.max_motions = 1e5;
  // options_dbastar.max_motions = 200000;
  // options_dbastar.max_motions = 1e4;
  // options_dbastar.max_motions = 50000;
  options_dbastar.max_motions = 4000;
  options_dbastar.search_timelimit = 60000;
  // options_dbastar.delta = .8;
  options_dbastar.delta = .7;
  options_dbastar.cost_delta_factor = 1;
  options_dbastar.delta_factor_goal = 1;
  options_dbastar.use_nigh_nn = 1;
  // options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp.bin";
  // options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp2.bin.ca.bin";
  // The improvement seems to help!!!
  // options_dbastar.motionsFile =
  //     "../build/quad3d_v0_all3.bin.im.bin.sp.bin.ca.bin";
  options_dbastar.motionsFile =
      "../build/quad3d_v0_all3.bin.im.bin.sp1.bin.ca.bin";
  options_dbastar.new_invariance = true;

  options_dbastar.use_collision_shape = false;
  options_dbastar.limit_branching_factor = 15;

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(tt_quad3_one_obs_old) {

  Problem problem("../benchmark/quadrotor_0/quad_one_obs.yaml");

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 1e5;
  options_dbastar.search_timelimit = 60000;
  options_dbastar.delta = 1;
  options_dbastar.cost_delta_factor = 1;
  options_dbastar.delta_factor_goal = .7;
  options_dbastar.use_nigh_nn = 1;
  options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp.bin";

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(test_quad2d_invariance) {

  Problem problem1("../benchmark/quad2d/quad_obs_column.yaml");
  Problem problem2("../benchmark/quad2d/quad_bugtrap.yaml");
  Problem problem3("../benchmark/quad2d/quad2d_recovery_wo_obs.yaml");
  Problem problem4("../benchmark/quad2d/quad2d_recovery_obs.yaml");
  Problem problem5("../benchmark/quad2d/fall_through.yaml");

  std::vector<Problem *> problems;

  // problems.push_back(&problem1);
  // problems.push_back(&problem2);
  // problems.push_back(&problem3);
  problems.push_back(&problem4);
  // problems.push_back(&problem5);

  for (auto &ptr : problems) {
    auto &problem = *ptr;
    Options_dbastar options_dbastar;
    options_dbastar.max_motions = 400; // I need to start with this!!
    options_dbastar.delta = .5;
    options_dbastar.search_timelimit = 20 * 1000;
    options_dbastar.use_nigh_nn = 1;
    options_dbastar.limit_branching_factor = 50;

    // check what is happening with the collisions!!
    options_dbastar.check_cols = true;
    options_dbastar.motionsFile = "../build/quad2d_v0_all_im.bin.sp.bin.ca.bin";
    options_dbastar.new_invariance = true;
    options_dbastar.use_collision_shape = false;

    Out_info_db out_info_db;
    Trajectory traj_out;
    dbastar(problem, options_dbastar, traj_out, out_info_db);
    CSTR_(out_info_db.cost);
    BOOST_TEST(out_info_db.solved);
  }
}

BOOST_AUTO_TEST_CASE(t_quad2dpole_toy) {
  Problem problem("../benchmark/quad2dpole/up.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 400; // I need to start with this!!
  options_dbastar.delta = .1;
  options_dbastar.search_timelimit = 20 * 1000;
  options_dbastar.use_nigh_nn = 1;
  options_dbastar.limit_branching_factor = 50;

  // check what is happening with the collisions!!
  options_dbastar.check_cols = false;
  options_dbastar.motionsFile = "../build/quad2dpole_tmp.bin.ca.bin";
  // options_dbastar.motionsFile = "../build/quad2dpole_tmp.bin.ca.bin";
  options_dbastar.new_invariance = true;
  options_dbastar.use_collision_shape = false;

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}



