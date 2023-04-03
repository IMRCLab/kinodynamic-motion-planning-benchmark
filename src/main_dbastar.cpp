#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>

#include <flann/flann.hpp>
#include <msgpack.hpp>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include <boost/program_options.hpp>

// OMPL headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/datastructures/NearestNeighborsFLANN.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>
#include <ompl/datastructures/NearestNeighborsSqrtApprox.h>

#include "ompl/base/ScopedState.h"
#include "robots.h"

#include "dbastar.hpp"
#include "general_utils.hpp"

int main(int argc, char *argv[]) {

  Options_dbastar options_dbastar;
  Out_info_db out_db;
  po::options_description desc("Allowed options");
  options_dbastar.add_options(desc);
  // inout_db.add_options(desc);
  std::string cfg_file;
  std::string data_file;
  std::string inputFile;
  set_from_boostop(desc, VAR_WITH_NAME(cfg_file));
  set_from_boostop(desc, VAR_WITH_NAME(data_file));
  set_from_boostop(desc, VAR_WITH_NAME(inputFile));

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
  // print options

  if (cfg_file != "") {
    options_dbastar.read_from_yaml(cfg_file.c_str());
  }

  Problem problem;
  problem.read_from_yaml(inputFile.c_str());

  std::cout << "*** options_dbastar ***" << std::endl;
  options_dbastar.print(std::cout);
  std::cout << "***" << std::endl;

  std::cout << "*** out_db ***" << std::endl;
  out_db.print(std::cout);
  std::cout << "***" << std::endl;

  Trajectory traj;
  dbastar(problem, options_dbastar, traj, out_db);

  std::cout << "*** inout_db *** " << std::endl;
  out_db.print(std::cout);
  std::cout << "***" << std::endl;

  if (out_db.solved) {
    return 0;
  } else {
    return 1;
  }
}
