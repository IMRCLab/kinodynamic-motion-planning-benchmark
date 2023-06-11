#include "dbastar.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

// TODO: how to compute the heuristic map for systems with velocities in the
// state space?

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

namespace nigh = unc::robotics::nigh;

using SpaceQuad2d = nigh::CartesianSpace<
    nigh::L2Space<double, 2>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::L2Space<double, 2>, std::ratio<1, 5>>,
    nigh::ScaledSpace<nigh::L2Space<double, 1>, std::ratio<1, 5>>>;

using Space = SpaceQuad2d;

using Key = Space::Type;
struct Functor {
  std::vector<Key> *keys;
  Functor(std::vector<Key> *keys) : keys(keys) {}
  const Key &operator()(const std::size_t &i) const { return keys->at(i); }
};

// Compound state [
// RealVectorState [5.24416 5.22166 0.841158]
// SO3State [0.000928958 -0.0285151 0.128069 0.991355]
// RealVectorState [0.83485 0.602365 -0.0284322]
// RealVectorState [0.0255689 0.0105831 -0.264379]
// ]
// distance 0.690961

BOOST_AUTO_TEST_CASE(quad2_fallthrough) {

  Problem problem("../benchmark/quad2d/fall_through.yaml");

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 200000;
  options_dbastar.search_timelimit = 60000;
  options_dbastar.delta = 1.5;
  options_dbastar.use_nigh_nn = 1;
  options_dbastar.limit_branching_factor = 100;
  // options_dbastar.motionsFile =
  // "../cloud/motionsV2/good/quad2d_v0/quad2d_v0_all.bin.sp1.bin";
  options_dbastar.motionsFile =
      "../cloud/motionsV2/good/quad2d_v0/quad2d_v0_DEFand3.sp.bin";
  options_dbastar.motionsFile = "quad2d_v0_all.bin";

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(quad3_one_obs) {

  Problem problem("../benchmark/quadrotor_0/quad_one_obs.yaml");

  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 100000;
  options_dbastar.search_timelimit = 60000;
  options_dbastar.delta = 1;
  options_dbastar.use_nigh_nn = 0;
  options_dbastar.motionsFile = "../build/quad3d_v0_all.bin.sp.bin";

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(evaluate_nn_dynamic) {

  // using tree_t = jk::tree::KDTree<size_t, 6, 8, jk::tree::L1>;
  using tree_t = jk::tree::KDTree<size_t, 6>;
  using point_t = std::array<double, 6>;

  point_t weights = {1., 1., .5, .2, .2, .2};
  // point_t weights = {1., 1., 1., 1.,1.,1.};

  Stopwatch watch;
  const char *data_name = "data_Tn.txt";

  std::vector<std::vector<double>> data;

  {
    std::ifstream ifs(data_name);
    load_matrix(ifs, data);
  }
  CSTR_(data.size());

  std::vector<point_t> point_tree(data.size());
  std::transform(data.begin(), data.end(), point_tree.begin(), [&](auto &x) {
    point_t out;
    std::copy_n(x.data(), x.size(), out.begin());
    element_wise(out.data(), weights.data(), 6);
    return out;
  });

  tree_t tree;
  double radius = .6;
  double radius_tree = .6;
  std::vector<double> ds(data.size());

  // point is at distance 2
  // l2 is distance sqrt(2)
  // l2 squared is 2

  // --

  // d = n * ( d/n)

  // d^2 / n

  // d -> d/2 + d/2

  // d^2/4 + d^2/4 = d^2 / 2

  watch.reset();
  tree.addPoint(point_tree.at(0), 0);
  for (size_t i = 1; i < point_tree.size(); i++) {
    auto &p = point_tree.at(i);
    // auto out = tree.searchBall(p, radius * radius / 4); // metric is
    // SquaredL2
    auto out = tree.searchKnn(p, 30);
    tree.addPoint(p, i);
  }
  std::cout << "time tree: " << watch.elapsed_ms() << std::endl;
  CSTR_(tree.size());

  using Tree = nigh::Nigh<size_t, Space, Functor, nigh::NoThreadSafety,
                          nigh::KDTreeBatch<>>;

  std::vector<Key> data_nigh(data.size());

  std::transform(data.begin(), data.end(), data_nigh.begin(), [&](auto &__x) {
    return std::tuple(Eigen::Vector2d(__x.at(0), __x.at(1)), __x.at(2),
                      Eigen::Vector2d(__x.at(3), __x.at(4)), V1d(__x.at(5)));
  });

  Functor functor(&data_nigh);

  Tree tree_nigh(Space{}, functor);

  watch.reset();
  tree_nigh.insert(0);
  for (size_t i = 1; i < data_nigh.size(); i++) {
    Key &key = data_nigh.at(i);
    std::vector<std::pair<size_t, double>> __nbh;
    // tree_nigh.nearest(__nbh, key, 1e6, radius);
    tree_nigh.nearest(__nbh, key, 30);
    tree_nigh.insert(i);
  }
  std::cout << "time nigh: " << watch.elapsed_ms() << std::endl;
  CSTR_(tree.size());

  {
    point_t zero{0., 0., 0, 0, 0, 0};
    auto out =
        tree.searchBall(zero, radius * radius / 4); // metric is SquaredL2
    std::cout << "tree " << std::endl;
    CSTR_(out.size());
    for (auto &o : out) {
      std::cout << o.payload << " - " << o.distance << std::endl;
    }
    // tree_.nearest(__nbh, zero, 1e6, radius);
    // CSTR_(__nbh.size());
    // std::cout << "nigh " << std::endl;
    // for (auto &n : __nbh) {
    //   std::cout << n.first << " - " << n.second << std::endl;
    // }
  }

  {
    Key zero =
        std::tuple(Eigen::Vector2d(0, 0), 0., Eigen::Vector2d(0, 0), V1d(0.));
    std::vector<std::pair<size_t, double>> __nbh;
    tree_nigh.nearest(__nbh, zero, 1e6, radius);
    CSTR_(__nbh.size());
    std::cout << "nigh " << std::endl;
    for (auto &n : __nbh) {
      std::cout << n.first << " - " << n.second << std::endl;
    }
  }

  CSTR_(tree_nigh.size());
}

BOOST_AUTO_TEST_CASE(evaluate_nn) {

  Stopwatch watch;

  using tree_t = jk::tree::KDTree<size_t, 6>;
  using point_t = std::array<double, 6>;

  const char *data_query_name = "data_out_query_Tm.txt";
  const char *data_tree_name = "data_out_Tm.txt";

  std::vector<std::vector<double>> data_query;
  std::vector<std::vector<double>> data_tree;

  {
    std::ifstream ifs(data_query_name);
    load_matrix(ifs, data_query);
  }

  {
    std::ifstream ifs(data_tree_name);
    load_matrix(ifs, data_tree);
  }

  CSTR_(data_query.size());
  CSTR_(data_tree.size());

  point_t weights = {1., 1., .5, .2, .2, .2};
  std::vector<point_t> data_query_tree(data_query.size());
  std::vector<point_t> data_tree_tree(data_tree.size());
  std::transform(data_query.begin(), data_query.end(), data_query_tree.begin(),
                 [&](auto &x) {
                   point_t out;
                   std::copy_n(x.data(), x.size(), out.begin());
                   element_wise(out.data(), weights.data(), 6);
                   return out;
                 });

  std::transform(data_tree.begin(), data_tree.end(), data_tree_tree.begin(),
                 [&](auto &x) {
                   point_t out;
                   std::copy_n(x.data(), x.size(), out.begin());
                   element_wise(out.data(), weights.data(), 6);
                   return out;
                 });

  tree_t tree;

  for (size_t i = 0; i < data_tree_tree.size(); i++) {
    auto v = data_tree_tree.at(i);
    tree.addPoint(v, 0);
  }
  double radius = .6 * .6;
  watch.reset();
  std::vector<double> ds(data_query.size());
  for (size_t i = 0; i < data_query_tree.size(); i++) {
    // auto out = tree.searchBall(p, radius * radius); // metric is SquaredL2
    auto nn = tree.searchKnn(data_query_tree.at(i), 1);
    ds[i] = nn[0].distance;
  }
  std::cout << "KD time nn: " << watch.elapsed_ms() << std::endl;

  watch.reset();
  for (size_t i = 0; i < data_query_tree.size(); i++) {
    auto &v = data_query_tree.at(i);
    auto out = tree.searchBall(v, radius * radius); // metric is SquaredL2
    // auto nn = tree.searchKnn(p, 1);
    // ds[i] = nn[0].distance;
  }
  std::cout << "KD time R: " << watch.elapsed_ms() << std::endl;

  // Linear search with L2 squared

  watch.reset();
  for (size_t i = 0; i < data_query.size(); i++) {
    auto &v = data_query.at(i);
    double min_d = std::numeric_limits<double>::max();
    size_t min_index = 0;
    for (size_t j = 0; j < data_tree.size(); j++) {
      auto &w = data_tree.at(j);
      double d = l2_squared(v.data(), w.data(), 6);
      // double d = l2(v.data(), w.data(), 6);
      if (d < min_d) {
        min_d = d;
        min_index = j;
      }
    }
    ds[i] = min_d;
  }
  std::cout << "LINEAR time : " << watch.elapsed_ms() << std::endl;

  // nigh

  using SpaceQuad2d = nigh::CartesianSpace<
      nigh::L2Space<double, 2>,
      nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
      nigh::ScaledSpace<nigh::L2Space<double, 2>, std::ratio<1, 5>>,
      nigh::ScaledSpace<nigh::L2Space<double, 1>, std::ratio<1, 5>>>;

  using Space = SpaceQuad2d;

  using Key = Space::Type;
  struct Functor {
    std::vector<Key> *keys;
    Functor(std::vector<Key> *keys) : keys(keys) {}
    const Key &operator()(const std::size_t &i) const { return keys->at(i); }
  };

  using Tree = nigh::Nigh<size_t, Space, Functor, nigh::NoThreadSafety,
                          nigh::KDTreeBatch<>>;

  std::vector<Key> data_tree_nigh(data_tree.size());
  std::vector<Key> data_query_nigh(data_query.size());

  std::transform(data_tree.begin(), data_tree.end(), data_tree_nigh.begin(),
                 [&](auto &__x) {
                   return std::tuple(
                       Eigen::Vector2d(__x.at(0), __x.at(1)), __x.at(2),
                       Eigen::Vector2d(__x.at(3), __x.at(4)), V1d(__x.at(5)));
                 });

  std::transform(data_query.begin(), data_query.end(), data_query_nigh.begin(),
                 [&](auto &__x) {
                   return std::tuple(
                       Eigen::Vector2d(__x.at(0), __x.at(1)), __x.at(2),
                       Eigen::Vector2d(__x.at(3), __x.at(4)), V1d(__x.at(5)));
                 });

  Functor functor(&data_tree_nigh);

  Tree tree_nigh(Space{}, functor);

  for (size_t i = 0; i < data_tree_nigh.size(); i++) {
    tree_nigh.insert(i);
  }
  // euq

  watch.reset();
  for (size_t i = 0; i < data_query_nigh.size(); i++) {
    Key &key = data_query_nigh.at(i);
    std::optional<std::pair<size_t, double>> pt = tree_nigh.nearest(key);
    ds[i] = pt.value().second;
  }
  std::cout << "NIGH nn: " << watch.elapsed_ms() << std::endl;

  watch.reset();
  for (size_t i = 0; i < data_query_nigh.size(); i++) {
    Key &key = data_query_nigh.at(i);
    std::vector<std::pair<size_t, double>> __nbh;
    tree_nigh.nearest(__nbh, key, 1e6, radius);
  }
  std::cout << "NIGH R: " << watch.elapsed_ms() << std::endl;
}

BOOST_AUTO_TEST_CASE(test_quad2d_column) {

  // it is solving, continue here!!
  Problem problem("../benchmark/quad2d/quad_obs_column.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 10000; // I need to start with this!!
  options_dbastar.delta = .6;
  options_dbastar.use_nigh_nn = 1; // with 0 it works
  options_dbastar.search_timelimit = 30 * 1000;
  // OMPL
  // time_nearestMotion: 723.968
  // time_nearestNode: 1029.44

  // NIGH
  // time_nearestMotion: 256.891
  // time_nearestNode: 156.13

  // options_dbastar.motionsFile =
  //     "../cloud/motionsV2/good/quad2d_v0/quad2_all_split_2_sorted.bin";

  // options_dbastar.motionsFile = "../build/quad2d_v0_all.bin.sp.bin.so.bin";
  // nn_motions_average: 3.95727
  // nn_motions_min: 0
  // nn_motions_max: 357

  // T_m->size(): 10000
  // __min: 0
  // __max: 1.05039
  // __sum / min_distance_primitives.size(): 0.254638

  // NOTE: only one split seems much better!!
  options_dbastar.motionsFile = "../build/quad2d_v0_all.bin.sp.bin";
  // __min: 0.0210583
  // __max: 0.41995
  // __sum / min_distance_primitives.size(): 0.192483
  // nn_motions_average: 3.8615
  // nn_motions_min: 0
  // nn_motions_max: 13

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(test_quad3d_recovery) {

  Problem problem("../benchmark/quadrotor_0/recovery.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 120000;
  // seems impossible...  -- only solution is to fix the distance!
  options_dbastar.delta = .9;
  options_dbastar.use_nigh_nn = 1;
  options_dbastar.delta_factor_goal = .5;

  // options_dbastar.motionsFile = "../build/quad3d_v0_all_approx.bin.sp.bin";
  // // also working
  options_dbastar.motionsFile =
      "../build/quad3d_v0_all3.bin.sp.bin"; // yes, with nigh
  // options_dbastar.motionsFile =
  // "../cloud/motionsV2/good/quad3d_v0/quad3d_v0_all.bin.sp.bin"; // yes with
  // delta=.9
  options_dbastar.motionsFile =
      "../build/quad3d_v0_sp1_merged.bin"; // also good :)
  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(test_quad2d_recovery) {

  // it is solving, continue here!!
  Problem problem("../benchmark/quad2d/quad2d_recovery_wo_obs.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 50000; // I need to start with this!!
  options_dbastar.delta = .6;
  options_dbastar.use_nigh_nn = 1; // with 0 it works

  // OMPL
  // time_nearestMotion: 723.968
  // time_nearestNode: 1029.44

  // NIGH
  // time_nearestMotion: 256.891
  // time_nearestNode: 156.13

  // options_dbastar.motionsFile =
  //     "../cloud/motionsV2/good/quad2d_v0/quad2_all_split_2_sorted.bin";

  options_dbastar.motionsFile = "../build/quad2d_v0_all.bin.sp.bin";

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(test_quad2d_col) {

  // Problem problem("../benchmark/quad2d/quad_obs_column.yaml");
  Problem problem("../benchmark/quad2d/quad_bugtrap.yaml");
                  // quad_obs_column.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 200; // I need to start with this!!
  options_dbastar.delta = .5;
  options_dbastar.search_timelimit = 20 * 1000;
  options_dbastar.use_nigh_nn = 1;
  options_dbastar.limit_branching_factor = 100;

  // check what is happening with the collisions!!
  options_dbastar.check_cols = true;

  // too many primitives now? looking interesting!! -- double check the new
  // implementation!

  // options_dbastar.motionsFile =
  //     "../cloud/motionsV2/good/quad2d_v0/quad2_all_split_2_sorted.bin";

  options_dbastar.motionsFile = "../build/quad2d_v0_all_im.bin.sp.bin.ca.bin";

  // options_dbastar.motionsFile =
  //     "../build/quad2d_v0_all.bin.sp.bin.ca.bin"; // velocity is always zero
  //                                                 // here!!

  options_dbastar.new_invariance = true;
  options_dbastar.use_collision_shape = false;

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(test_quad2d_recovery_new) {

  // it is solving, continue here!!
  Problem problem("../benchmark/quad2d/quad2d_recovery_wo_obs.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 50000; // I need to start with this!!
  options_dbastar.delta = .6;
  options_dbastar.use_nigh_nn = 1; // with 0 it works
  options_dbastar.debug = true;

  // OMPL
  // time_nearestMotion: 723.968
  // time_nearestNode: 1029.44

  // NIGH
  // time_nearestMotion: 256.891
  // time_nearestNode: 156.13

  // options_dbastar.motionsFile =
  //     "../cloud/motionsV2/good/quad2d_v0/quad2_all_split_2_sorted.bin";

  options_dbastar.motionsFile = "../build/quad2d_v0_all.bin.sp.bin.ca.bin";

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

BOOST_AUTO_TEST_CASE(test_quad2d) {

  Problem problem("../benchmark/quad2d/empty_0.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.max_motions = 20000; // I need to start with this!!
  options_dbastar.delta = .5;
  // options_dbastar.motionsFile =
  //     "../cloud/motionsV2/good/quad2d_v0/quad2_all_split_2_sorted.bin";
  options_dbastar.motionsFile =
      "../build/quad2d_v0_all.bin.sp.bin.ca.bin"; // velocity is always zero
                                                  // here!!

  options_dbastar.new_invariance = true;
  options_dbastar.use_collision_shape = true;

  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.solved);
}

// test the second order!
BOOST_AUTO_TEST_CASE(test_bugtrap_second) {

  Problem problem("../benchmark/unicycle_second_order_0/bugtrap_0.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.search_timelimit = 1e5; // in ms
  options_dbastar.max_motions = 500;
  options_dbastar.heuristic = 1;
  options_dbastar.motionsFile = "../cloud/motionsV2/good/unicycle2_v0/"
                                "unicycle2_v0__ispso__2023_04_03__15_36_01.bin";
  options_dbastar.use_nigh_nn = 1; // both seem to work!!
  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  BOOST_TEST(out_info_db.solved);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.cost < 32.);
}

// GOOD
BOOST_AUTO_TEST_CASE(test_bugtrap_heu) {

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.search_timelimit = 1e5; // in ms
  options_dbastar.max_motions = 1000;
  options_dbastar.heuristic = 1;
  options_dbastar.motionsFile = "../cloud/motionsV2/good/unicycle1_v0/"
                                "unicycle1_v0__ispso__2023_04_03__14_56_57.bin";
  options_dbastar.max_size_heu_map = 500;
  options_dbastar.use_nigh_nn = 0;
  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  BOOST_TEST(out_info_db.solved);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.cost < 30.);
}

BOOST_AUTO_TEST_CASE(test_bugtrap) {

  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");
  Options_dbastar options_dbastar;
  options_dbastar.search_timelimit = 1e5; // in ms
  options_dbastar.max_motions = 1000;
  options_dbastar.heuristic = 0;
  options_dbastar.motionsFile = "../cloud/motionsV2/good/unicycle1_v0/"
                                "unicycle1_v0__ispso__2023_04_03__14_56_57.bin";
  options_dbastar.max_size_heu_map = 500;
  options_dbastar.use_nigh_nn = 1;
  options_dbastar.new_invariance = 0;
  options_dbastar.cost_delta_factor =
      1; // equivalent number of expands, maybe more clean!!
  options_dbastar.always_add = 0; // very, very slow!!
  options_dbastar.use_collision_shape = true;
  Out_info_db out_info_db;
  Trajectory traj_out;
  dbastar(problem, options_dbastar, traj_out, out_info_db);
  BOOST_TEST(out_info_db.solved);
  CSTR_(out_info_db.cost);
  BOOST_TEST(out_info_db.cost < 30.);
}

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
  options_dbastar.heu_resolution = 0.1;
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
    out_db.write_yaml(std::cout);
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
  options_dbastar.heu_resolution = 0.1;
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
    out_db.write_yaml(std::cout);
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
    out_db.write_yaml(std::cout);
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
  options_dbastar.heu_resolution = 0.1;
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
    out_db.write_yaml(std::cout);
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
    out_db.write_yaml(std::cout);
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
  options_dbastar.heu_resolution = 0.1;
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
    out_db.write_yaml(std::cout);
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
    out_db.write_yaml(std::cout);
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
    out_db.write_yaml(std::cout);
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
    out_db.write_yaml(std::cout);
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
    out_db.write_yaml(std::cout);
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
  // --inputFile ../benchmark/unicycle_second_order_0/kink_0.yaml
  // --motionsFile
  // ../cloud/motions/unicycle_second_order_0_sorted.msgpack --max_motions
  // 1000
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
