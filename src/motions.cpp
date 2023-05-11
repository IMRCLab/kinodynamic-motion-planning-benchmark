#include "pinocchio/math/fwd.hpp"
#include "pinocchio/multibody/liegroup/liegroup.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>
#include <yaml-cpp/node/iterator.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "Eigen/Core"
#include "croco_macros.hpp"

#include <fcl/fcl.h>

#include "fcl/broadphase/broadphase_collision_manager.h"
#include "fcl/broadphase/broadphase_dynamic_AABB_tree.h"
#include "fcl/broadphase/default_broadphase_callbacks.h"
#include "fcl/geometry/shape/box.h"
#include "fcl/geometry/shape/sphere.h"
#include "general_utils.hpp"
#include "math_utils.hpp"
#include "robot_models.hpp"

#include "motions.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

using vstr = std::vector<std::string>;
using V2d = Eigen::Vector2d;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;
using Vxd = Eigen::VectorXd;
using V1d = Eigen::Matrix<double, 1, 1>;

using namespace pinocchio;
// using namespace crocoddyl;

void Trajectory::read_from_yaml(const YAML::Node &node) {

  auto get_all = [this](auto &node) {
    this->states = yaml_node_to_xs(node["states"]);
    this->actions = yaml_node_to_xs(node["actions"]);

    if (node["times"]) {
      std::vector<double> __times =
          node["times"].template as<std::vector<double>>();
      this->times = Eigen::VectorXd::Map(__times.data(), __times.size());
    }
  };

  if (node["states"] && node["actions"]) {
    get_all(node);
  } else if (node["result"] && node["result"]["states"] &&
             node["result"]["actions"]) {
    get_all(node["result"]);
  } else if (node["result"] && node["result"][0] &&
             node["result"][0]["states"] && node["result"][0]["actions"]) {
    get_all(node["result"][0]);
  } else {
    ERROR_WITH_INFO("this format is not supported!");
  }
}

void Trajectory::read_from_yaml(const char *file) {
  std::cout << "Loading file: " << file << std::endl;
  read_from_yaml(load_yaml_safe(file));
}

void Trajectory::to_yaml_format(std::ostream &out,
                                const std::string &prefix) const {

  out << prefix << STR_(time_stamp) << std::endl;
  out << prefix << STR_(cost) << std::endl;
  out << prefix << STR_(feasible) << std::endl;
  out << prefix << STR_(traj_feas) << std::endl;
  out << prefix << STR_(goal_feas) << std::endl;
  out << prefix << STR_(start_feas) << std::endl;
  out << prefix << STR_(col_feas) << std::endl;
  out << prefix << STR_(x_bounds_feas) << std::endl;
  out << prefix << STR_(u_bounds_feas) << std::endl;
  out << prefix << STR_V(start) << std::endl;
  out << prefix << STR_V(goal) << std::endl;

  out << prefix << STR_(max_jump) << std::endl;
  out << prefix << STR_(max_collision) << std::endl;
  out << prefix << STR_(goal_distance) << std::endl;
  out << prefix << STR_(start_distance) << std::endl;
  out << prefix << STR_(x_bound_distance) << std::endl;
  out << prefix << STR_(u_bound_distance) << std::endl;

  out << prefix << "num_states: " << states.size() << std::endl;
  out << prefix << "states:" << std::endl;
  for (auto &state : states) {
    out << prefix << "  - " << state.format(FMT) << std::endl;
  }
  out << prefix << "num_actions: " << actions.size() << std::endl;
  out << prefix << "actions:" << std::endl;
  for (auto &action : actions) {
    out << prefix << "  - " << action.format(FMT) << std::endl;
  }
  if (times.size()) {
    out << prefix << "num_times: " << times.size() << std::endl;
    out << prefix << "times:" << std::endl;
    for (size_t i = 0; i < static_cast<size_t>(times.size()); i++) {
      out << prefix << "  - " << times(i) << std::endl;
    }
  }
  if (info.size()) {
    out << prefix << "info: " << info << std::endl;
  }
};

void Trajectory::update_feasibility(double traj_tol, double goal_tol,
                                    double col_tol, double x_bound_tol,
                                    double u_bound_tol, bool verbose) {
  traj_feas = max_jump < traj_tol;
  goal_feas = goal_distance < goal_tol;
  start_feas = start_distance < goal_tol;
  col_feas = max_collision < col_tol;
  x_bounds_feas = x_bound_distance < x_bound_tol;
  u_bounds_feas = u_bound_distance < u_bound_tol;

  feasible = traj_feas && goal_feas && start_feas && col_feas &&
             x_bounds_feas && u_bounds_feas;

  if (verbose) {
    std::cout << "updating flags " << std::endl;
    CSTR_(feasible);
    CSTR_(traj_feas);
    CSTR_(goal_feas);
    CSTR_(start_feas);
    CSTR_(col_feas);
    CSTR_(x_bounds_feas);
    CSTR_(u_bounds_feas);
  }
}

void Trajectory::check(std::shared_ptr<Model_robot> &robot, bool verbose) {

  max_collision = check_cols(robot, states);
  Eigen::VectorXd dts;

  if (times.size()) {
    CHECK_EQ(static_cast<size_t>(times.size()), states.size(), AT);
    dts.resize(times.size() - 1);
    CHECK_GE(times.size(), 1, AT);
    for (size_t i = 0; i < static_cast<size_t>(times.size() - 1); i++) {
      dts(i) = times(i + 1) - times(i);
    }

  } else {
    size_t T = actions.size();
    dts.resize(T);
    dts.setOnes();
    dts.array() *= robot->ref_dt;
  }

  max_jump = check_trajectory(states, actions, dts, robot, verbose);
  x_bound_distance = check_x_bounds(states, robot, verbose);
  u_bound_distance = check_u_bounds(actions, robot, verbose);

  if (goal.size()) {
    if (verbose) {
      CSTR_V(states.back());
      CSTR_V(goal);
    }
    goal_distance = robot->distance(states.back(), goal);
  }
  if (start.size())
    start_distance = robot->distance(states.front(), start);

  if (verbose) {
    std::cout << " -- Checking trajectory -- " << std::endl;
    CSTR_(max_jump);
    CSTR_(x_bound_distance);
    CSTR_(u_bound_distance);
    CSTR_(goal_distance);
    CSTR_(start_distance);
    CSTR_(max_collision);
  }

  update_feasibility();

  // &&goal_feas &&start_feas &&col_feas &&x_bounds_feas &&u_bounds_feas;
}

void Problem::read_from_yaml(const YAML::Node &env) {

  std::vector<double> _start, _goal;
  YAML::Node tmp = env["robots"][0]["start"];

  if (auto nn = env["name"]; nn)
    name = nn.as<std::string>();

  for (const auto &e : env["robots"][0]["start"]) {
    _start.push_back(e.as<double>());
  }

  for (const auto &e : env["robots"][0]["goal"]) {
    _goal.push_back(e.as<double>());
  }

  start = Vxd::Map(_start.data(), _start.size());
  goal = Vxd::Map(_goal.data(), _goal.size());

  // const auto &env_min = env["environment"]["min"];
  // const auto &env_max = env["environment"]["max"];

  std::vector<double> min_ =
      env["environment"]["min"].as<std::vector<double>>();
  std::vector<double> max_ =
      env["environment"]["max"].as<std::vector<double>>();

  CHECK_EQ(min_.size(), max_.size(), AT);
  CHECK((min_.size() <= 3), AT);
  p_lb = Eigen::Map<Eigen::VectorXd>(&min_.at(0), min_.size());
  p_ub = Eigen::Map<Eigen::VectorXd>(&max_.at(0), max_.size());

  for (const auto &obs : env["environment"]["obstacles"]) {
    std::vector<double> size_ = obs["size"].as<std::vector<double>>();
    Vxd size = Vxd::Map(size_.data(), size_.size());

    auto obs_type = obs["type"].as<std::string>();

    std::vector<double> center_ = obs["center"].as<std::vector<double>>();
    Vxd center = Vxd::Map(center_.data(), center_.size());

    obstacles.push_back(
        Obstacle{.type = obs_type, .size = size, .center = center});
  }

  robotType = env["robots"][0]["type"].as<std::string>();
}

void Problem::read_from_yaml(const char *file) {
  std::cout << "Loading yaml file:" << file << std::endl;
  read_from_yaml(load_yaml_safe(file));
}

void Problem::write_to_yaml(const char *file) {
  (void)file;
  ERROR_WITH_INFO("not implemented");
}

void get_states_and_actions(const YAML::Node &data,
                            std::vector<Eigen::VectorXd> &states,
                            std::vector<Eigen::VectorXd> &actions) {

  using Vxd = Eigen::VectorXd;
  CHECK(data["states"], AT);
  CHECK(data["actions"], AT);

  std::vector<std::vector<double>> __states;
  std::vector<std::vector<double>> __actions;

  for (const auto &state : data["states"]) {
    std::vector<double> p;
    for (const auto &elem : state) {
      p.push_back(elem.as<double>());
    }
    __states.push_back(p);
  }

  for (const auto &state : data["actions"]) {
    std::vector<double> p;
    for (const auto &elem : state) {
      p.push_back(elem.as<double>());
    }
    __actions.push_back(p);
  }

  states.resize(__states.size());
  actions.resize(__actions.size());

  std::transform(__states.begin(), __states.end(), states.begin(),
                 [](const auto &s) { return Vxd::Map(s.data(), s.size()); });

  std::transform(__actions.begin(), __actions.end(), actions.begin(),
                 [](const auto &s) { return Vxd::Map(s.data(), s.size()); });
}

double check_u_bounds(const std::vector<Vxd> &us_out,
                      std::shared_ptr<Model_robot> model, bool verbose) {
  CHECK(us_out.size(), AT);
  CHECK(model, AT);

  double max_out = 0;
  for (const auto &u : us_out) {

    if (check_bounds_distance(u, model->get_u_lb(), model->get_u_ub()) > 1e-3) {
      CSTR_V(u);
      CSTR_V(model->get_u_lb());
      CSTR_V(model->get_u_ub());
    }

    max_out = std::max(max_out, check_bounds_distance(u, model->get_u_lb(),
                                                      model->get_u_ub()));
  }

  return max_out;
  ;
}

double check_x_bounds(const std::vector<Vxd> &xs_out,
                      std::shared_ptr<Model_robot> model, bool verbose) {
  CHECK(xs_out.size(), AT);
  CHECK(model, AT);

  double max_out = 0;
  // for (const auto &x : xs_out) {
  for (size_t i = 0; i < xs_out.size(); i++) {

    auto &x = xs_out.at(i);
    double d = check_bounds_distance(x, model->get_x_lb(), model->get_x_ub());

    if (d > .01 && verbose) {
      std::cout << "BOUND VIOLATION t=" << i << std::endl;
      CSTR_(d);
      CSTR_V(x);
      CSTR_V(model->get_x_lb());
      CSTR_V(model->get_x_ub());
    }
    max_out = std::max(max_out, d);
  }
  return max_out;
}

double check_trajectory(const std::vector<Vxd> &xs_out,
                        const std::vector<Vxd> &us_out, const Vxd &dt,
                        std::shared_ptr<Model_robot> model, bool verbose) {
  CHECK(xs_out.size(), AT);
  CHECK(us_out.size(), AT);
  CHECK(model, AT);
  CHECK_EQ(xs_out.size(), us_out.size() + 1, AT);
  CHECK_EQ(static_cast<size_t>(dt.size()), static_cast<size_t>(us_out.size()),
           AT);

  size_t N = us_out.size();

  double max_jump_distance = 0;

  for (size_t i = 0; i < N; i++) {
    Vxd xnext(model->nx);
    auto &x = xs_out.at(i);
    auto &u = us_out.at(i);

    model->step(xnext, x, u, dt(i));

    double jump = model->distance(xnext, xs_out.at(i + 1));
    if (jump > 1e-3 && verbose) {
      std::cout << "jump of " << jump << std::endl;
      CSTR_(i);
      CSTR_V(x);
      CSTR_V(u);
      CSTR_V(xnext);
      CSTR_V(xs_out.at(i + 1));

      max_jump_distance = jump;
    }
  }

  return max_jump_distance;
}

double check_cols(std::shared_ptr<Model_robot> model_robot,
                  const std::vector<Vxd> &xs) {
  double accumulated_c = 0;
  double max_c = 0;
  CollisionOut out;
  for (size_t i = 0; i < xs.size(); i++) {
    auto &x = xs.at(i);
    model_robot->collision_distance(x, out);
    if (out.distance < 0) {
      std::cout << "Warning -- col at: " << STR_V(x) << " time:" << i
                << " distance: " << out.distance << std::endl;
      accumulated_c += std::abs(out.distance);
      if (std::abs(out.distance) > max_c) {
        max_c = std::abs(out.distance);
      }
    }
  }

  return max_c;
}

double max_rollout_error(std::shared_ptr<Model_robot> robot,
                         const std::vector<Vxd> &xs,
                         const std::vector<Vxd> &us) {
  CHECK_EQ(xs.size(), us.size() + 1, AT);

  size_t N = us.size();

  size_t nx = xs.front().size();

  Vxd xnext(nx);
  double max_error = 0;

  for (size_t i = 0; i < N; i++) {

    robot->step(xnext, xs.at(i), us.at(i), robot->ref_dt);
    double d = robot->distance(xnext, xs.at(i + 1));

    if (d > max_error) {
      max_error = d;
    }
  }
  return max_error;
}

void resample_trajectory(std::vector<Eigen::VectorXd> &xs_out,
                         std::vector<Eigen::VectorXd> &us_out,
                         Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us,
                         const Eigen::VectorXd &ts, double ref_dt,
                         const std::shared_ptr<StateQ> &state)

{

  xs_out.clear();
  us_out.clear();

  double total_time = ts(ts.size() - 1);

  ptr<Interpolator> path_u = mk<Interpolator>(ts.head(ts.size() - 1), us);

  ptr<Interpolator> path_x = mk<Interpolator>(ts, xs, state);

  size_t num_time_steps = std::ceil(total_time / ref_dt);

  auto ts__ = Eigen::VectorXd::LinSpaced(num_time_steps + 1, 0,
                                         num_time_steps * ref_dt);

  times = ts__;
  std::cout << "taking samples at " << ts__.format(FMT) << std::endl;

  std::vector<Eigen::VectorXd> new_xs;
  std::vector<Eigen::VectorXd> new_us;

  CHECK(xs.size(), AT);
  CHECK(us.size(), AT);
  size_t nx = xs.front().size();
  size_t nu = us.front().size();

  Vxd xout(nx);
  Vxd Jout(nx);
  Vxd uout(nu);
  Vxd Juout(nu);
  for (size_t ti = 0; ti < num_time_steps + 1; ti++) {
    path_x->interpolate(ts__(ti), xout, Jout);
    new_xs.push_back(xout);
    if (ti < num_time_steps) {
      path_u->interpolate(ts__(ti), uout, Juout);
      new_us.push_back(uout);
    }
  }

  xs_out = new_xs;
  us_out = new_us;
}

void Info_out::print(std::ostream &out, const std::string &be,
                     const std::string &af) {
  STRY(solved, out, be, af);
  STRY(cost, out, be, af);
}

void Info_out::to_yaml(std::ostream &out, const std::string &be,
                       const std::string &af) {

  STRY(solved, out, be, af);
  STRY(cost, out, be, af);
  out << be << "trajs_opt" << af << std::endl;
  std::string prefix = "   ";
  for (size_t i = 0; i < trajs_opt.size(); i++) {
    out << be << "  -" << std::endl;
    trajs_opt.at(i).to_yaml_format(out, be + prefix);
  }

  out << "trajs_raw:" << std::endl;
  for (size_t i = 0; i < trajs_raw.size(); i++) {
    out << be << "  -" << std::endl;
    std::string prefix = "   ";
    trajs_raw.at(i).to_yaml_format(out, be + prefix);
  }
}

double Trajectory::distance(const Trajectory &other) const {

  CHECK_EQ(actions.size(), other.actions.size(), AT);
  CHECK_EQ(states.size(), other.states.size(), AT);

  double distance = 0;

  for (size_t i = 0; i < states.size(); i++) {
    auto &x = states.at(i);
    auto &y = other.states.at(i);
    CHECK_EQ(x.size(), y.size(), AT);
    distance += (x - y).norm();
  }
  for (size_t i = 0; i < actions.size(); i++) {
    auto &x = actions.at(i);
    auto &y = other.actions.at(i);
    CHECK_EQ(x.size(), y.size(), AT);
    distance += (x - y).norm();
  }
  return distance;
}

void Trajectory::save_file_boost(const char *file) const {
  std::cout << "Traj: save file boost to: " << file << std::endl;
  std::ofstream out(file, std::ios::binary);
  CHECK(out.is_open(), AT);
  boost::archive::binary_oarchive oa(out);
  oa << *this;
}

void Trajectory::load_file_boost(const char *file) {
  std::cout << "Traj: load file boost from: " << file << std::endl;
  std::ifstream in(file, std::ios::binary);
  CHECK(in.is_open(), AT);
  boost::archive::binary_iarchive oi(in);
  oi >> *this;
}

void Trajectories::save_file_boost(const char *file) const {

  std::cout << "Trajs: save file boost to: " << file << std::endl;
  std::ofstream out(file, std::ios::binary);
  CHECK(out.is_open(), AT);
  boost::archive::binary_oarchive oa(out);
  oa << *this;
}

void Trajectories::load_file_boost(const char *file) {
  std::cout << "Trajs: load file boost from: " << file << std::endl;

  std::ifstream in(file, std::ios::binary);
  CHECK(in.is_open(), AT);
  boost::archive::binary_iarchive oi(in);
  oi >> *this;
  std::cout << "Trajs: load file boost from: " << file << " -- DONE"
            << std::endl;
}

void load_env_quim(Model_robot &robot, const Problem &problem) {
  std::vector<fcl::CollisionObjectd *> obstacles;
  double ref_pos = 0;
  double ref_size = 1.;
  for (const auto &obs : problem.obstacles) {
    auto &obs_type = obs.type;
    auto &size = obs.size;
    auto &center = obs.center;

    if (obs_type == "box") {
      std::shared_ptr<fcl::CollisionGeometryd> geom;
      geom.reset(new fcl::Boxd(size(0), size(1),
                               size.size() == 3 ? size(2) : ref_size));
      auto co = new fcl::CollisionObjectd(geom);
      co->setTranslation(fcl::Vector3d(center(0), center(1),
                                       size.size() == 3 ? center(2) : ref_pos));
      co->computeAABB();
      obstacles.push_back(co);
    } else if (obs_type == "sphere") {
      std::shared_ptr<fcl::CollisionGeometryd> geom;
      geom.reset(new fcl::Sphered(size(0)));
      auto co = new fcl::CollisionObjectd(geom);
      co->setTranslation(fcl::Vector3d(
          center(0), center(1), center.size() == 3 ? center(2) : ref_pos));
      co->computeAABB();
      obstacles.push_back(co);
    } else {
      throw std::runtime_error("Unknown obstacle type! --" + obs_type);
    }
  }
  robot.env.reset(new fcl::DynamicAABBTreeCollisionManagerd());
  robot.env->registerObjects(obstacles);
  robot.env->setup();
}

Trajectory from_welf_to_quim(const Trajectory &traj_raw, double u_nominal) {

  Trajectory traj = traj_raw;
  traj.states.resize(traj_raw.states.size());
  traj.actions.resize(traj_raw.actions.size());

  Eigen::VectorXd tmp(traj_raw.states.front().size());

  std::transform(traj_raw.states.begin(), traj_raw.states.end(),
                 traj.states.begin(), [&](auto &x) {
                   from_welf_format(x, tmp);
                   tmp.segment(3, 4).normalize();
                   return tmp;
                 });

  Eigen::VectorXd u_tmp(traj_raw.actions.front().size());
  std::transform(traj_raw.actions.begin(), traj_raw.actions.end(),
                 traj.actions.begin(), [&](auto &x) {
                   u_tmp = x / u_nominal;
                   return u_tmp;
                 });

  return traj;
}

Trajectory from_quim_to_welf(const Trajectory &traj_raw, double u_nominal) {

  Trajectory traj = traj_raw;
  traj.states.resize(traj_raw.states.size());
  traj.actions.resize(traj_raw.actions.size());

  Eigen::VectorXd tmp(traj_raw.states.front().size());

  std::transform(traj_raw.states.begin(), traj_raw.states.end(),
                 traj.states.begin(), [&](auto &x) {
                   from_quim_format(x, tmp);
                   tmp.segment(6, 4).normalize();
                   return tmp;
                 });

  Eigen::VectorXd u_tmp(traj_raw.actions.front().size());
  std::transform(traj_raw.actions.begin(), traj_raw.actions.end(),
                 traj.actions.begin(), [&](auto &x) {
                   u_tmp = u_nominal * x;
                   return u_tmp;
                 });

  return traj;
}

#include <boost/histogram.hpp>

struct Bin {

  int index;
  double min;
  double max;
  double center;
  double x;

  void to_yaml(std::ostream &out, const char *prefix = "") const {
    out << prefix << "index: " << index << std::endl;
    out << prefix << "min: " << min << std::endl;
    out << prefix << "max: " << max << std::endl;
    out << prefix << "center: " << center << std::endl;
    out << prefix << "x: " << x << std::endl;
  }
};

std::vector<std::vector<Bin>>
make_componentwise_histogram(const std::vector<Eigen::VectorXd> &states) {
  CHECK(states.size(), AT);
  std::vector<std::vector<Bin>> out;

  for (size_t i = 0; i < states.front().size(); i++) {
    std::vector<double> xi;
    for (const auto &s : states) {
      xi.push_back(s(i));
    }
    size_t num_bins = 100;

    // print_vec(xi.data(), xi.size());

    double min_ = *std::min_element(xi.begin(), xi.end());
    double max_ = *std::max_element(xi.begin(), xi.end());

    // if (max_ - min_ < 1e-10) {

    double diff = max_ - min_;
    max_ += .1 * std::max(diff, .1);
    min_ -= .1 * std::max(diff, .1);

    auto h = boost::histogram::make_histogram(boost::histogram::axis::regular<>(
        num_bins, min_, max_, "x" + std::to_string(i)));

    // CSTR_(h.size());
    std::for_each(xi.begin(), xi.end(), std::ref(h));
    // CSTR_(h.size());

    std::vector<Bin> bins;
    for (auto &&x : indexed(h, boost::histogram::coverage::all)) {
      Bin bin{.index = x.index(),
              .min = x.bin().lower(),
              .max = x.bin().upper(),
              .center = (x.bin().upper() + x.bin().lower()) / 2,
              .x = *x};
      bins.push_back(bin);
    }

    out.push_back(bins);
  }
  return out;
}

void bins_to_yaml(std::ostream &out,
                  const std::vector<std::vector<Bin>> &bins_start) {
  for (auto &bs : bins_start) {
    out << " -" << std::endl;
    for (auto &b : bs) {
      out << "  -" << std::endl;
      b.to_yaml(out, "   ");
    }
  }
}
void Trajectories::compute_stats(const char *filename_out) const {

  std::vector<Eigen::VectorXd> starts;
  std::vector<Eigen::VectorXd> goals;

  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> actions;

  using V1d = Eigen::Matrix<double, 1, 1>;

  std::vector<Eigen::VectorXd> lengths;

  for (const auto &t : data) {
    CHECK(t.states.size(), AT);
    starts.push_back(t.states.front());
    goals.push_back(t.states.back());
    states.insert(states.end(), t.states.begin(), t.states.end());
    actions.insert(actions.end(), t.actions.begin(), t.actions.end());

    lengths.push_back(V1d(t.actions.size()));
  }

  std::vector<std::vector<Bin>> bins_lengths =
      make_componentwise_histogram(lengths);

  /// continue!!
  std::vector<std::vector<Bin>> bins_start =
      make_componentwise_histogram(starts);
  std::vector<std::vector<Bin>> bins_goals =
      make_componentwise_histogram(goals);
  std::vector<std::vector<Bin>> bins_states =
      make_componentwise_histogram(states);
  std::vector<std::vector<Bin>> bins_actions =
      make_componentwise_histogram(actions);

  std::ofstream out(filename_out);

  out << "bins_lengths:" << std::endl;
  bins_to_yaml(out, bins_lengths);

  out << "bins_start:" << std::endl;
  bins_to_yaml(out, bins_start);

  out << "bins_goals:" << std::endl;
  bins_to_yaml(out, bins_goals);

  out << "bins_states:" << std::endl;
  bins_to_yaml(out, bins_states);

  out << "bins_actions:" << std::endl;
  bins_to_yaml(out, bins_actions);
}
