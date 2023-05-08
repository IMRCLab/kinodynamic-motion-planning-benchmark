#pragma once
#include "croco_macros.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "ocp.hpp"

struct Options_primitives {

  double time_limit = 1000;      // in seconds
  int max_num_primitives = 1000; // use -1 to say MAX
  size_t max_attempts = 1e8;
  std::string dynamics = "unicycle1_v0";
  bool adapt_infeas_primitives = false;
  size_t ref_time_steps = 50;
  size_t min_length_cut = 5;
  size_t max_length_cut = 20;
  size_t max_splits = 10;

  void print(std::ostream &out, const std::string &be = "",
             const std::string &af = ": ") {
    STRY(time_limit, out, be, af);
    STRY(max_num_primitives, out, be, af);
    STRY(dynamics, out, be, af);
    STRY(adapt_infeas_primitives, out, be, af);
    STRY(max_attempts, out, be, af);
    STRY(min_length_cut, out, be, af);
    STRY(max_length_cut, out, be, af);
    STRY(max_splits, out, be, af);
    STRY(ref_time_steps, out, be, af);
  };

  void add_options(po::options_description &desc) {
    set_from_boostop(desc, VAR_WITH_NAME(time_limit));
    set_from_boostop(desc, VAR_WITH_NAME(max_num_primitives));
    set_from_boostop(desc, VAR_WITH_NAME(dynamics));
    set_from_boostop(desc, VAR_WITH_NAME(adapt_infeas_primitives));
    set_from_boostop(desc, VAR_WITH_NAME(max_attempts));
    set_from_boostop(desc, VAR_WITH_NAME(min_length_cut));
    set_from_boostop(desc, VAR_WITH_NAME(max_length_cut));
    set_from_boostop(desc, VAR_WITH_NAME(max_splits));
    set_from_boostop(desc, VAR_WITH_NAME(ref_time_steps));
  }
};

void sort_motion_primitives(
    const Trajectories &trajs, Trajectories &trajs_out,
    std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)>
        distance_fun,
    int top_k = -1);

void split_motion_primitives(const Trajectories &in,
                             const std::string &dynamics, Trajectories &out,
                             const Options_primitives &options_primitives);

void improve_motion_primitives(const Options_trajopt &options_trajopt,
                               const Trajectories &trajs_in,
                               const std::string &dynamics,
                               Trajectories &trajs_out);

void generate_primitives(const Options_trajopt &options_trajopt,
                         const Options_primitives &options_primitives,
                         Trajectories &trajectories);
