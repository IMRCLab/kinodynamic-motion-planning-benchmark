
#include "general_utils.hpp"
#include "motions.hpp"
#include "robot_models.hpp"
// #include "robots.h"

int main(int argc, char *argv[]) {

  std::string robot_file;
  std::string env_file;
  std::string result_file;
  double col_tol = 1e-2;
  double traj_tol = 1e-2;
  double goal_tol = 0.05;

  po::options_description desc("Allowed options");

  set_from_boostop(desc, VAR_WITH_NAME(robot_file));
  set_from_boostop(desc, VAR_WITH_NAME(env_file));
  set_from_boostop(desc, VAR_WITH_NAME(result_file));
  set_from_boostop(desc, VAR_WITH_NAME(col_tol));
  set_from_boostop(desc, VAR_WITH_NAME(traj_tol));
  set_from_boostop(desc, VAR_WITH_NAME(goal_tol));

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  Problem problem;
  problem.read_from_yaml(env_file.c_str());

  if (!robot_file.size()) {
    // read from env file
    std::string base_path = "../models/";
    std::string suffix = ".yaml";
    robot_file = base_path + problem.robotType + suffix;
  }

  std::shared_ptr<Model_robot> robot = robot_factory(robot_file.c_str());
  load_env_quim(*robot, problem);

  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> actions;

  YAML::Node init = load_yaml_safe(result_file);
  CHECK(init["result"], AT);
  CHECK(init["result"][0], AT);
  get_states_and_actions(init["result"][0], states, actions);

  CHECK(states.size(), AT);
  CHECK(actions.size(), AT);
  CHECK_EQ(states.size(), actions.size() + 1, AT);

  bool col_feas = check_cols(robot, states) < col_tol;

  size_t T = actions.size();

  Eigen::VectorXd dts(T);
  dts.setOnes();

  dts.array() *= robot->ref_dt;

  bool traj_feas = check_trajectory(states, actions, dts, robot) < 1e-2;

  double distance = robot->distance(states.back(), problem.goal);
  CSTR_(distance);

  bool goal_feas = distance < goal_tol;

  double distance_start = robot->distance(states.front(), problem.start);
  bool start_feas = distance_start < 1e-3;

  // TODO: I should also check the position bounds!!

  CSTR_(traj_feas);
  CSTR_(col_feas);
  CSTR_(goal_feas);
  CSTR_(start_feas);

  if (traj_feas && col_feas && goal_feas && start_feas) {
    return 0;
  } else {
    return 1;
  }
}
