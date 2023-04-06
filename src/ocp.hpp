#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>

#include "Eigen/Core"
#include <boost/program_options.hpp>

#include "croco_macros.hpp"
// #include "croco_models.hpp"
#include "general_utils.hpp"
#include "motions.hpp"
#include "robot_models.hpp"

#include "crocoddyl/core/action-base.hpp"

inline std::string robot_type_to_path(const std::string &robot_type) {
  std::string base_path = "../models/";
  std::string suffix = ".yaml";
  return base_path + robot_type + suffix;
}

struct Options_trajopt {

  bool CALLBACKS = true;
  std::string solver_name;
  bool use_finite_diff = false;
  bool use_warmstart = true;
  bool rollout_warmstart = false;
  bool repair_init_guess = true;
  bool control_bounds = true;
  bool states_reg = false;
  int solver_id = 0;
  double disturbance = 1e-4;

  double th_stop = 1e-2;
  double init_reg = 1e2;
  double th_acceptnegstep = .3;
  double noise_level = 1e-5; // factor on top of [-1., 1.]
  double k_linear = 10.;
  double k_contour = 10.;
  double u_bound_scale = 1;

  size_t max_iter = 50;
  size_t window_optimize = 20;
  size_t window_shift = 10;
  size_t max_mpc_iterations = 50;
  std::string debug_file_name = "debug_file.yaml";
  double weight_goal = 200.;
  double collision_weight = 100.;
  bool smooth_traj = false;
  bool shift_repeat = true;

  double tsearch_max_rate = 2;
  double tsearch_min_rate = .7;
  int tsearch_num_check = 20;
  bool ref_x0 = false;
  bool interp = false;
  bool welf_format = false;

  void add_options(po::options_description &desc);

  void __read_from_node(const YAML::Node &node);

  void print(std::ostream &out, const std::string &be = "",
             const std::string &af = ": ") const;

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
};

using namespace crocoddyl;

enum class SOLVER {
  traj_opt = 0,
  traj_opt_free_time = 1,
  traj_opt_smooth_then_free_time = 2,
  mpc = 3,
  mpcc = 4,
  mpcc2 = 5,
  traj_opt_mpcc = 6,
  mpc_nobound_mpcc = 7,
  mpcc_linear = 8,
  time_search_traj_opt = 9,
  mpc_adaptative = 10,
  traj_opt_free_time_proxi = 11,
  traj_opt_no_bound_bound = 12,
  traj_opt_free_time_proxi_linear = 13,
  traj_opt_free_time_linear = 14,
  none = 15
};

void PrintVariableMap(const boost::program_options::variables_map &vm,
                      std::ostream &out);

template <typename Derived>
boost::shared_ptr<crocoddyl::ActionModelAbstract>
to_am_base(boost::shared_ptr<Derived> am) {
  return boost::static_pointer_cast<crocoddyl::ActionModelAbstract>(am);
};

void print_data(boost::shared_ptr<ActionDataAbstractTpl<double>> data);

struct Generate_params {
  bool free_time = false;
  bool free_time_linear = false;
  std::string name;
  size_t N;
  Eigen::VectorXd goal;
  Eigen::VectorXd start;
  std::shared_ptr<Model_robot> model_robot;
  std::vector<Eigen::VectorXd> states = {};
  std::vector<Eigen::VectorXd> states_weights = {};
  std::vector<Eigen::VectorXd> actions = {};
  bool contour_control = false;
  ptr<Interpolator> interpolator = nullptr;
  double max_alpha = 100.;
  bool linear_contour = true;
  bool goal_cost = true;
  bool collisions = true;

  void print(std::ostream &out) const;
};

ptr<crocoddyl::ShootingProblem>
generate_problem(const Generate_params &gen_args,
                 const Options_trajopt &options_trajopt, size_t &nx,
                 size_t &nu);

struct File_parser_inout {
  std::string problem_name;
  std::string init_guess;
  std::string env_file;
  bool new_format = false;
  std::string name;
  std::string robot_model_file;
  double dt;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  Eigen::VectorXd ts; // times;
  Eigen::VectorXd start;
  Eigen::VectorXd goal;
  size_t T = 0; // number of time steps. If init_guess = "" , then initial guess
  // is T * [start]

  void add_options(po::options_description &desc);
  void print(std::ostream &out);
  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
};

Eigen::VectorXd enforce_bounds(const Eigen::VectorXd &us,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub);

void read_from_file(File_parser_inout &inout);

void convert_traj_with_variable_time(const std::vector<Eigen::VectorXd> &xs,
                                     const std::vector<Eigen::VectorXd> &us,
                                     std::vector<Eigen::VectorXd> &xs_out,
                                     std::vector<Eigen::VectorXd> &us_out,
                                     const double &dt);

struct Result_opti {
  // Note: success is not the same as feasible.
  // Feasible=1 means that the trajectory fulfil all the constraints.
  // Success=1 means that the trajectory fulfil the constraints imposed by
  // the current formulation (e.g. some formulations will solve
  // first without bound constraints).
  bool feasible = false;
  bool success = false;
  double cost = -1;
  std::string name;
  std::vector<Eigen::VectorXd> xs_out;
  std::vector<Eigen::VectorXd> us_out;

  void write_yaml(std::ostream &out);

  void write_yaml_db(std::ostream &out);
};

std::vector<Eigen::VectorXd>
smooth_traj(const std::vector<Eigen::VectorXd> &xs_init, const StateQ &state);

void __trajectory_optimization(const Problem &problem,
                               std::shared_ptr<Model_robot> &model_robot,
                               const Trajectory &init_guess,
                               const Options_trajopt &options_trajopt,
                               Trajectory &traj, Result_opti &opti_out);

// void trajectory_optimization(Problem &problem, File_parser_inout
// &file_inout,
//                              Result_opti &opti_out);

void trajectory_optimization(const Problem &problem,
                             const Trajectory &init_guess,
                             const Options_trajopt &opti_parms,
                             Trajectory &traj, Result_opti &opti_out);

bool check_problem(ptr<crocoddyl::ShootingProblem> problem,
                   ptr<crocoddyl::ShootingProblem> problem2,
                   const std::vector<Eigen::VectorXd> &xs,
                   const std::vector<Eigen::VectorXd> &us);
