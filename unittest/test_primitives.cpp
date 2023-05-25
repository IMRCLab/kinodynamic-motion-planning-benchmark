#define BOOST_TEST_MODULE test module name
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "general_utils.hpp"
#include "generate_primitives.hpp"

BOOST_AUTO_TEST_CASE(t_generate) {

  const Options_trajopt options_trajopt;
  Options_primitives options_primitives;

  options_primitives.max_num_primitives = 10;

  std::cout << " *** options_primitives *** " << std::endl;
  options_primitives.print(std::cout);
  std::cout << " *** " << std::endl;
  std::cout << " *** options_trajopt *** " << std::endl;
  options_trajopt.print(std::cout);
  std::cout << " *** " << std::endl;

  Trajectories trajectories;

  generate_primitives(options_trajopt, options_primitives, trajectories);

  std::string time_stamp = get_time_stamp();
  trajectories.save_file_boost(("/tmp/motions__" + options_primitives.dynamics +
                                "__" + time_stamp + ".bin")
                                   .c_str());
  trajectories.save_file_yaml(("/tmp/motions__" + options_primitives.dynamics +
                               "__" + time_stamp + ".yaml")
                                  .c_str(),
                              10000);
}

BOOST_AUTO_TEST_CASE(t_improve) {

  std::string dynamics = "unicycle1_v0";
  Trajectories trajectories, trajectories_out;

  const char *filename = "motions__unicycle1_v0__02-04-2023--17-20-20.bin";

  trajectories.load_file_boost(filename);

  // "motions__unicycle1_v0__02-04-2023--15-29-05.bin");

  // improve_primitives(options_trajopt, trajectories);

  Options_trajopt options_trajopt;
  options_trajopt.solver_id =
      static_cast<int>(SOLVER::traj_opt_free_time_linear);

  Options_primitives options_primitives;
  improve_motion_primitives(options_trajopt, trajectories, dynamics,
                            trajectories_out, options_primitives);

  std::string time_stamp = get_time_stamp();
  trajectories_out.save_file_boost(
      ("motions__i__" + dynamics + "__" + time_stamp + ".bin").c_str());

  trajectories_out.save_file_yaml(
      ("motions__i__" + dynamics + "__" + time_stamp + ".yaml").c_str(), 10000);
}

BOOST_AUTO_TEST_CASE(t_split) {

  std::string dynamics = "unicycle1_v0";
  Trajectories trajectories, trajectories_out;
  trajectories.load_file_boost(
      "motions__unicycle1_v0__02-04-2023--17-20-20.bin");
  size_t num_translation = 2;
  Options_primitives options_primitives;
  split_motion_primitives(trajectories, dynamics, trajectories_out,
                          options_primitives);

  // check that they are valid...
  std::shared_ptr<Model_robot> robot_model =
      robot_factory(robot_type_to_path(dynamics).c_str());

  for (auto &traj : trajectories_out.data) {
    traj.check(robot_model);
    traj.update_feasibility();
    CHECK(traj.feasible, AT);
  }

  std::string time_stamp = get_time_stamp();
  trajectories_out.save_file_boost(
      ("motions__s__" + dynamics + "__" + time_stamp + ".bin").c_str());

  trajectories_out.save_file_yaml(
      ("motions__s__" + dynamics + "__" + time_stamp + ".yaml").c_str());
}

BOOST_AUTO_TEST_CASE(t_sort) {

  using V2d = Eigen::Vector2d;

  Trajectory t1;
  Trajectory t2;
  Trajectory t3;
  Trajectory t4;

  t1.states.push_back(V2d(0, 0));
  t1.states.push_back(V2d(10, 0));

  t1.start = t1.states.front();
  t1.goal = t1.states.back();

  t2.states.push_back(V2d(0, 0));
  t2.states.push_back(V2d(0, 0));

  t2.start = t2.states.front();
  t2.goal = t2.states.back();

  t3.states.push_back(V2d(1, 1));
  t3.states.push_back(V2d(100, 0));

  t3.start = t3.states.front();
  t3.goal = t3.states.back();

  t4.states.push_back(V2d(2, 1));
  t4.states.push_back(V2d(100, 0));

  t4.start = t4.states.front();
  t4.goal = t4.states.back();

  Trajectories trajectories, trajectories_out;
  trajectories.data = {t1, t2, t3, t4};

  auto fun = [](const auto &x, const auto &y) { return (x - y).norm(); };

  sort_motion_primitives(trajectories, trajectories_out, fun);

  // check.
  CHECK_LEQ(trajectories_out.data.at(0).distance(t3), 1e-8, AT);
  CHECK_LEQ(trajectories_out.data.at(1).distance(t2), 1e-8, AT);
  CHECK_LEQ(trajectories_out.data.at(2).distance(t1), 1e-8, AT);
  CHECK_LEQ(trajectories_out.data.at(3).distance(t4), 1e-8, AT);
}
