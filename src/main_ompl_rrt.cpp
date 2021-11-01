#include <fstream>
#include <iostream>
#include <algorithm>

#include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>

// OMPL headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/planners/rrt/RRT.h>

#include "robotDubinsCar.h"
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
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
    ("output,o", po::value<std::string>(&outputFile)->required(), "output file (yaml)");

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
  if (robotType == "dubins_0") {
    ob::RealVectorBounds position_bounds(2);
    const auto& dims = env["environment"]["dimensions"];
    position_bounds.setLow(0);
    position_bounds.setHigh(0, dims[0].as<double>());
    position_bounds.setHigh(1, dims[1].as<double>());

    robot.reset(new RobotDubinsCar(
        position_bounds,
        /*w_limit*/ 0.5 /*rad/s*/,
        /*v*/ 0.1 /* m/s*/));
  } else if (robotType == "car_first_order_0") {
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

  // set number of control steps
  si->setPropagationStepSize(1);
  si->setMinMaxControlDuration(1, 1);

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
  pdef->setGoalState(goalState, 0.01);
  si->freeState(goalState);

  // create a planner for the defined space
  auto rrt(std::make_shared<oc::RRT>(si));
  // rrt->setGoalBias(params["goalBias"].as<float>());
  auto planner(rrt);

  // set the problem we are trying to solve for the planner
  planner->setProblemDefinition(pdef);

  // perform setup steps for the planner
  planner->setup();

  // print the settings for this space
  si->printSettings(std::cout);

  // print the problem settings
  pdef->print(std::cout);

  // attempt to solve the problem within timelimit
  float timelimit = 10; 
  ob::PlannerStatus solved = planner->ob::Planner::solve(timelimit);

  if (solved)
  {
    // get the goal representation from the problem definition (not the same as the goal state)
    // and inquire about the found path
    ob::PathPtr path = pdef->getSolutionPath();
    std::cout << "Found solution:" << std::endl;

    // print the path to screen
    path->print(std::cout);
  }
  else {
    std::cout << "No solution found" << std::endl;
  }

  std::cout << solved << std::endl;
  auto path = pdef->getSolutionPath()->as<oc::PathControl>();

  std::cout << path->getStateCount() << "," << path->getControlCount() << std::endl;
  assert(path->getStateCount() == path->getControlCount() + 1);

  std::ofstream out(outputFile);
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (size_t i = 0; i < path->getControlCount(); ++i) {
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

    // auto stateCS = state->as<ob::CompoundStateSpace::StateType>();
    // double pos = stateCS->as<ob::RealVectorStateSpace::StateType>(0)->values[0];
    // double vel = stateCS->as<ob::RealVectorStateSpace::StateType>(1)->values[0];

    // double a = path->getControl(i)->as<ompl::control::RealVectorControlSpace::ControlType>()->values[0];
    // out << pos << "," << vel << "," << a << std::endl;
  }

  return 0;
}
