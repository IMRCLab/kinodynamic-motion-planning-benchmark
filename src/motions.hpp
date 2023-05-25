#pragma once
#include "Eigen/Core"
#include "croco_macros.hpp"
#include "fcl/broadphase/broadphase_collision_manager.h"
#include "general_utils.hpp"
#include "math_utils.hpp"
#include "robot_models.hpp"
#include <algorithm>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>
#include <yaml-cpp/node/node.h>

double check_u_bounds(const std::vector<Eigen::VectorXd> &us_out,
                      std::shared_ptr<Model_robot> model, bool verbose);

double check_x_bounds(const std::vector<Eigen::VectorXd> &xs_out,
                      std::shared_ptr<Model_robot> model, bool verbose);

void get_states_and_actions(const YAML::Node &data,
                            std::vector<Eigen::VectorXd> &states,
                            std::vector<Eigen::VectorXd> &actions);

struct Problem {
  using Vxd = Eigen::VectorXd;

  Problem(const char *file) { read_from_yaml(file); }
  Problem() = default;

  std::string name; // name of the proble: E.g. bugtrap-car1

  Eigen::VectorXd goal;
  Eigen::VectorXd start;

  Eigen::VectorXd p_lb; // position bounds
  Eigen::VectorXd p_ub; // position bounds

  std::vector<Obstacle> obstacles;
  std::string robotType;

  void read_from_yaml(const YAML::Node &env);

  void read_from_yaml(const char *file);

  void write_to_yaml(const char *file);
};

// next: time optimal linear, use so2 space, generate motion primitives

double check_trajectory(const std::vector<Eigen::VectorXd> &xs_out,
                        const std::vector<Eigen::VectorXd> &us_out,
                        const Eigen::VectorXd &dt,
                        std::shared_ptr<Model_robot> model,
                        bool verbose = false);

double check_cols(std::shared_ptr<Model_robot> model_robot,
                  const std::vector<Eigen::VectorXd> &xs);

// namespace selection

BOOST_SERIALIZATION_SPLIT_FREE(Eigen::VectorXd)

template <class Archive>
void naive_eigen_vector_save(Archive &ar, const Eigen::VectorXd &v) {
  std::vector<double> vv(v.data(), v.data() + v.size());
  ar &vv;
}

template <class Archive>
void naive_eigen_vector_load(Archive &ar, Eigen::VectorXd &v) {
  std::vector<double> vv;
  ar &vv;
  v = Eigen::VectorXd::Map(vv.data(), vv.size());
}

namespace boost {
namespace serialization {
template <class Archive>
inline void load(Archive &ar, Eigen::VectorXd &v,
                 const unsigned int file_version) {

  naive_eigen_vector_load(ar, v);
}

template <class Archive>
inline void save(Archive &ar, const Eigen::VectorXd &v,
                 const unsigned int file_version) {

  naive_eigen_vector_save(ar, v);
}
} // namespace serialization
} // namespace boost

struct Trajectory {

  double time_stamp; // when it was generated?
  double cost = 1e8;
  bool feasible = 0;

  bool traj_feas = 0;
  bool goal_feas = 0;
  bool start_feas = 0;
  bool col_feas = 0;
  bool x_bounds_feas = 0;
  bool u_bounds_feas = 0;

  double max_jump = -1;
  double max_collision = -1.;
  double goal_distance = -1;
  double start_distance = -1.;
  double x_bound_distance = -1.;
  double u_bound_distance = -1.;
  std::string info = "";

  Trajectory() = default;
  Trajectory(const char *file) { read_from_yaml(file); }

  Eigen::VectorXd start;
  Eigen::VectorXd goal;
  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> actions;
  size_t num_time_steps = 0; // use this if we want default init guess.
  Eigen::VectorXd times;

  void to_yaml_format(std::ostream &out, const std::string &prefix = "") const;

  void read_from_yaml(const YAML::Node &node);

  void read_from_yaml(const char *file);

  void check(std::shared_ptr<Model_robot> &robot, bool verbose = false);

  void update_feasibility(double traj_tol = 1e-2, double goal_tol = 1e-2,
                          double col_tol = 1e-2, double x_bound_tol = 1e-2,
                          double u_bound_tol = 1e-2, bool verbose = false);

  // boost serialization

  template <class Archive>
  inline void serialize(Archive &ar, const unsigned int file_version) {
    ar &states;
    ar &actions;
    ar &cost;
    ar &feasible;
    ar &start;
    ar &goal;
    if (file_version > 0)
      ar &info;
  }

  double distance(const Trajectory &other) const;

  void save_file_boost(const char *file) const;

  void load_file_boost(const char *file);
};
BOOST_CLASS_VERSION(Trajectory, 1);

struct Trajectories {

  std::vector<Trajectory> data{};

  template <class Archive>
  inline void serialize(Archive &ar, const unsigned int file_version) {
    ar &data;
  }

  void save_file_boost(const char *file) const;
  void load_file_boost(const char *file);

  void save_file_yaml(const char *file, int num_motions = -1) const {
    // format is:
    // - TRAJ 1
    // - TRAJ 2

    std::ofstream out(file);
    std::string prefix = "  ";

    if (num_motions == -1) {
      num_motions = data.size();
    }
    num_motions = std::min(num_motions, static_cast<int>(data.size()));

    for (size_t i = 0; i < static_cast<size_t>(num_motions); i++) {
      auto &traj = data.at(i);
      out << "-" << std::endl;
      traj.to_yaml_format(out, prefix);
    }
  }

  void load_file_yaml(const YAML::Node &node) {
    CSTR_(node.size());
    for (const auto &nn : node) {
      Trajectory traj;
      traj.read_from_yaml(nn);
      data.push_back(traj);
    }
  }

  void load_file_yaml(const char *file) {
    std::cout << "Loading file: " << file << std::endl;
    load_file_yaml(load_yaml_safe(file));
  }

  void compute_stats(const char *filename_out) const;

  //
};

double max_rollout_error(std::shared_ptr<Model_robot> robot,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us);

void resample_trajectory(std::vector<Eigen::VectorXd> &xs_out,
                         std::vector<Eigen::VectorXd> &us_out,
                         Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us,
                         const Eigen::VectorXd &ts, double ref_dt,
                         const std::shared_ptr<StateQ> &state);

struct Info_out {
  bool solved = false;
  double cost = 1e8;
  std::vector<Trajectory> trajs_raw;
  std::vector<Trajectory> trajs_opt;
  std::vector<std::map<std::string, std::string>> infos_raw;
  std::vector<std::map<std::string, std::string>> infos_opt;

  Info_out() = default;
  ~Info_out() = default;

  void virtual print(std::ostream &out, const std::string &be = "",
                     const std::string &af = ": ");
  void virtual to_yaml(std::ostream &out, const std::string &be = "",
                       const std::string &af = ": ");
};

void load_env_quim(Model_robot &robot, const Problem &problem);

Trajectory from_welf_to_quim(const Trajectory &traj_raw, double u_nominal);

Trajectory from_quim_to_welf(const Trajectory &traj_raw, double u_nominal);
