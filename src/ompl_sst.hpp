#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>

#include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>

// OMPL headers
#include <ompl/base/objectives/ControlDurationObjective.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include "general_utils.hpp"
#include "ocp.hpp"
#include "robots.h"

namespace po = boost::program_options;

struct Options_sst {

  std::string planner = "sst";
  double timelimit = 60; // TODO: which unit?
  double goal_epsilon = .1;
  double goal_bias = .05;
  double selection_radius = .2;
  double pruning_radius = .1;
  bool reach_goal_with_opt = true;

  double propagation_step_size = .1;
  int min_control_duration = 1;
  int max_control_duration = 10;

  void add_options(po::options_description &desc) {
    set_from_boostop(desc, VAR_WITH_NAME(planner));
    set_from_boostop(desc, VAR_WITH_NAME(timelimit));
    set_from_boostop(desc, VAR_WITH_NAME(goal_epsilon));
    set_from_boostop(desc, VAR_WITH_NAME(goal_bias));
    set_from_boostop(desc, VAR_WITH_NAME(selection_radius));
    set_from_boostop(desc, VAR_WITH_NAME(pruning_radius));

    set_from_boostop(desc, VAR_WITH_NAME(propagation_step_size));
    set_from_boostop(desc, VAR_WITH_NAME(min_control_duration));
    set_from_boostop(desc, VAR_WITH_NAME(max_control_duration));
    set_from_boostop(desc, VAR_WITH_NAME(reach_goal_with_opt));
  }

  void print(std::ostream &out, const std::string &be = "",
             const std::string &af = ": ") {
    STRY(planner, out, be, af);
    STRY(timelimit, out, be, af);
    STRY(goal_epsilon, out, be, af);
    STRY(goal_bias, out, be, af);
    STRY(selection_radius, out, be, af);
    STRY(pruning_radius, out, be, af);
    STRY(propagation_step_size, out, be, af);
    STRY(min_control_duration, out, be, af);
    STRY(max_control_duration, out, be, af);
    STRY(reach_goal_with_opt, out, be, af);
  }

  void __read_from_node(const YAML::Node &node) {
    set_from_yaml(node, VAR_WITH_NAME(planner));
    set_from_yaml(node, VAR_WITH_NAME(timelimit));
    set_from_yaml(node, VAR_WITH_NAME(goal_epsilon));
    set_from_yaml(node, VAR_WITH_NAME(goal_bias));
    set_from_yaml(node, VAR_WITH_NAME(selection_radius));
    set_from_yaml(node, VAR_WITH_NAME(pruning_radius));
    set_from_yaml(node, VAR_WITH_NAME(propagation_step_size));
    set_from_yaml(node, VAR_WITH_NAME(min_control_duration));
    set_from_yaml(node, VAR_WITH_NAME(max_control_duration));
  }

  void read_from_yaml(YAML::Node &node) {

    if (YAML::Node parameter = node["options_sst"]) {
      __read_from_node(parameter);
    } else {
      __read_from_node(node);
    }
  }

  void read_from_yaml(const char *file) {
    std::cout << "loading file: " << file << std::endl;
    YAML::Node node = YAML::LoadFile(file);
    read_from_yaml(node);
  }
};

struct Inout_sst {

  std::string inputFile;
  std::string outFile = "out.yaml"; // default
  std::string problem_name;
  std::string statsFile = "ompl_stats.yaml";
  double cost = -1;
  bool solved = 0;

  void print(std::ostream &out) {

    std::string be = "";
    std::string af = ": ";

    out << be << STR(inputFile, af) << std::endl;
    out << be << STR(outFile, af) << std::endl;
    out << be << STR(cost, af) << std::endl;
    out << be << STR(problem_name, af) << std::endl;
    out << be << STR(solved, af) << std::endl;
    out << be << STR(statsFile, af) << std::endl;
  }

  void add_options(po::options_description &desc) {

    set_from_boostop(desc, VAR_WITH_NAME(inputFile));
    set_from_boostop(desc, VAR_WITH_NAME(outFile));
    set_from_boostop(desc, VAR_WITH_NAME(problem_name));
    set_from_boostop(desc, VAR_WITH_NAME(statsFile));
  }

  void read_from_yaml(YAML::Node &node) {

    set_from_yaml(node, VAR_WITH_NAME(inputFile));
    set_from_yaml(node, VAR_WITH_NAME(outFile));
    set_from_yaml(node, VAR_WITH_NAME(problem_name));
    set_from_yaml(node, VAR_WITH_NAME(statsFile));
  }
};

void solve_sst(const Problem &problem, const Options_sst &options_ompl_sst,
               const Options_trajopt &options_trajopt, Trajectory &traj_out,
               Info_out &info_out_omplsst);
