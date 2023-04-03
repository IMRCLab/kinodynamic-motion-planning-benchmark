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
#include "ompl_sst.hpp"
#include "robots.h"

int main(int argc, char *argv[]) {

  Options_sst options_ompl_sst;
  Options_trajopt
      options_trajopt; // only use if we optimize to reach the goal exactly

  namespace po = boost::program_options;
  po::options_description desc("Allowed options");
  std::string cfg_file = "";
  std::string results_file = "";
  std::string env_file;
  options_ompl_sst.add_options(desc);
  set_from_boostop(desc, VAR_WITH_NAME(cfg_file));
  set_from_boostop(desc, VAR_WITH_NAME(env_file));
  set_from_boostop(desc, VAR_WITH_NAME(results_file));

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }

    if (cfg_file != "") {
      options_ompl_sst.read_from_yaml(cfg_file.c_str());
    }

  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  std::cout << "*** Options SST ***" << std::endl;
  options_ompl_sst.print(std::cout);
  std::cout << "***" << std::endl;

  std::cout << "*** Options TrajOpt ***" << std::endl;
  options_trajopt.print(std::cout);
  std::cout << "***" << std::endl;

  Problem problem(env_file.c_str());
  Trajectory traj_out;
  Info_out info_out_omplgeo;

  solve_sst(problem, options_ompl_sst, options_trajopt, traj_out,
            info_out_omplgeo);

  std::cout << "solve_sst done" << std::endl;
  info_out_omplgeo.print(std::cout);

  std::ofstream results(results_file);

  results << "alg: sst" << std::endl;
  results << "time_stamp: " << get_time_stamp() << std::endl;

  results << "options_sst:" << std::endl;
  options_ompl_sst.print(results, "  ");

  results << "options_trajopt:" << std::endl;
  options_trajopt.print(results, "  ");

  info_out_omplgeo.to_yaml(results);
}
