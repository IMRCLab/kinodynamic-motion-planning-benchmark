#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>

// OMPL headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/planners/rrt/RRT.h>
#include <ompl/control/planners/sst/SST.h>
#include <ompl/base/objectives/ControlDurationObjective.h>
#include "AO_RRT.h"

#include "robotCarFirstOrder.h"
#include "robotCarSecondOrder.h"
#include "robotStatePropagator.hpp"
#include "fclStateValidityChecker.hpp"

namespace ob = ompl::base;
namespace oc = ompl::control;

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string outputFile;
  std::string statsFile;
  std::string plannerDesc;
  int timelimit;
  float goalRegion;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
    ("output,o", po::value<std::string>(&outputFile)->required(), "output file (yaml)")
    ("stats", po::value<std::string>(&statsFile)->default_value("ompl_stats.yaml"), "output file (yaml)")
    ("planner,p", po::value<std::string>(&plannerDesc)->default_value("rrt"), "Planner")
    ("timelimit", po::value<int>(&timelimit)->default_value(60), "Time limit for planner")
    ("goalregion", po::value<float>(&goalRegion)->default_value(0.1), "radius around goal to count success");

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error& e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  // load problem description
  YAML::Node env = YAML::LoadFile(inputFile);

  std::vector<fcl::CollisionObjectf *> obstacles;
  for (const auto &obs : env["environment"]["obstacles"])
  {
    if (obs["type"].as<std::string>() == "box")
    {
      const auto &size = obs["size"];
      std::shared_ptr<fcl::CollisionGeometryf> geom;
      geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0));
      const auto &center = obs["center"];
      auto co = new fcl::CollisionObjectf(geom);
      co->setTranslation(fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
      obstacles.push_back(co);
    }
    else
    {
      throw std::runtime_error("Unknown obstacle type!");
    }
  }
  std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env(new fcl::DynamicAABBTreeCollisionManagerf());
  // std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env(new fcl::NaiveCollisionManagerf());
  bpcm_env->registerObjects(obstacles);
  bpcm_env->setup();

  const auto& robot_node = env["robots"][0];
  auto robotType = robot_node["type"].as<std::string>();
  std::shared_ptr<Robot> robot;
  if (robotType == "car_first_order_0") {
    ob::RealVectorBounds position_bounds(2);
    const auto& dims = env["environment"]["dimensions"];
    position_bounds.setLow(0);
    position_bounds.setHigh(0, dims[0].as<double>());
    position_bounds.setHigh(1, dims[1].as<double>());

    robot.reset(new RobotCarFirstOrder(
        position_bounds,
        /*w_limit*/ 0.5 /*rad/s*/,
        /*v_limit*/ 0.5 /* m/s*/));
  } else if (robotType == "car_second_order_0")
  {
    ob::RealVectorBounds position_bounds(2);
    const auto &dims = env["environment"]["dimensions"];
    position_bounds.setLow(0);
    position_bounds.setHigh(0, dims[0].as<double>());
    position_bounds.setHigh(1, dims[1].as<double>());

    robot.reset(new RobotCarSecondOrder(
        position_bounds,
        /*v_limit*/ 0.5 /*m/s*/,
        /*w_limit*/ 0.5 /*rad/s*/,
        /*a_limit*/ 2.0 /*m/s^2*/,
        /*w_dot_limit*/ 2.0 /*rad/s^2*/
      ));
  } else {
    throw std::runtime_error("Unknown robot type!");
  }

  auto si = robot->getSpaceInformation();

  // set number of control steps (use 0.1s as increment -> 0.1 to 1s per Steer function)
  si->setPropagationStepSize(0.1);
  si->setMinMaxControlDuration(1, 10);

  // set state validity checking for this space
  auto stateValidityChecker(std::make_shared<fclStateValidityChecker>(si, bpcm_env, robot));
  si->setStateValidityChecker(stateValidityChecker);

  // set the state propagator
  std::shared_ptr<oc::StatePropagator> statePropagator(new RobotStatePropagator(si, robot));
  si->setStatePropagator(statePropagator);

  si->setup();

  // create a problem instance
  auto pdef(std::make_shared<ob::ProblemDefinition>(si));

  // create and set a start state
  auto startState = si->allocState();
  std::vector<double> reals;
  for (const auto& v : robot_node["start"]) {
    reals.push_back(v.as<double>());
  }
  si->getStateSpace()->copyFromReals(startState, reals);
  pdef->addStartState(startState);
  si->freeState(startState);

  // set goal state
  auto goalState = si->allocState();
  reals.clear();
  for (const auto &v : robot_node["goal"]) {
    reals.push_back(v.as<double>());
  }
  si->getStateSpace()->copyFromReals(goalState, reals);
  pdef->setGoalState(goalState, goalRegion);
  si->freeState(goalState);

  // create a planner for the defined space
  std::shared_ptr<ob::Planner> planner;
  if (plannerDesc == "rrt")
  {
    planner.reset(new oc::RRT(si));
  } else if (plannerDesc == "aorrt") {
    planner.reset(new AO_RRT(si));
  } else if (plannerDesc == "sst") {
    planner.reset(new oc::SST(si));
  }
  // rrt->setGoalBias(params["goalBias"].as<float>());
  // auto planner(rrt);

  pdef->setOptimizationObjective(std::make_shared<ob::ControlDurationObjective>(si));

  // empty stats file
  std::ofstream stats(statsFile);
  stats << "stats:" << std::endl;

  auto start = std::chrono::steady_clock::now();

  pdef->setIntermediateSolutionCallback(
      [start, &stats](const ob::Planner *, const std::vector<const ob::State *> &, const ob::Cost cost)
      {
        auto now = std::chrono::steady_clock::now();
        double t = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        stats << "  - t: " << t/1000.0f << std::endl;
        stats << "    cost: " << cost.value() << std::endl;
        std::cout << "Intermediate solution! " << cost.value() << " " << t/1000.0f << std::endl;
      });

  // set the problem we are trying to solve for the planner
  planner->setProblemDefinition(pdef);



  // perform setup steps for the planner
  planner->setup();

  // print the settings for this space
  si->printSettings(std::cout);

  // print the problem settings
  pdef->print(std::cout);

  // attempt to solve the problem within timelimit
  ob::PlannerStatus solved;

  // for (int i = 0; i < 3; ++i) {
  solved = planner->ob::Planner::solve(timelimit);
  std::cout << solved << std::endl;
  // }

  if (solved)
  {
    // get the goal representation from the problem definition (not the same as the goal state)
    // and inquire about the found path
    // ob::PathPtr path = pdef->getSolutionPath();
    std::cout << "Found solution:" << std::endl;

    // print the path to screen
    // path->print(std::cout);
  }
  else {
    std::cout << "No solution found" << std::endl;
  }

  std::cout << solved << std::endl;
  auto path = pdef->getSolutionPath()->as<oc::PathControl>();

  // update propagation step size to get right interpolation resolution
  si->setPropagationStepSize(0.1);
  path->interpolate(); // normalize to a single control step

  std::cout << path->getStateCount() << "," << path->getControlCount() << std::endl;
  assert(path->getStateCount() == path->getControlCount() + 1);

  std::ofstream out(outputFile);
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (size_t i = 0; i < path->getStateCount(); ++i) {
    const auto state = path->getState(i);

    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, state);
    out << "      - [";
    for (size_t i = 0; i < reals.size(); ++i) {
      out << reals[i];
      if (i < reals.size() - 1) {
        out << ",";
      }
    }
    out << "]" << std::endl;
  }
  out << "    actions:" << std::endl;
  for (size_t i = 0; i < path->getControlCount(); ++i) {

    const size_t dim = si->getControlSpace()->getDimension();
    out << "      - [";
    for (size_t d = 0; d < dim; ++d)
    {
      const auto action = path->getControl(i);
      double *address = si->getControlSpace()->getValueAddressAtIndex(action, d);
      out << *address;
      if (d < dim - 1)
      {
        out << ",";
      }
    }
    out << "]" << std::endl;
  }

  return 0;
}
