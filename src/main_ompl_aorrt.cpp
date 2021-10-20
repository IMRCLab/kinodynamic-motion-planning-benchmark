#include <fstream>
#include <iostream>
#include <algorithm>

// #include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>

// OMPL headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/control/planners/rrt/RRT.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

void propagate(
    const ob::State *start,
    const oc::Control *control,
    const double /*duration*/,
    ob::State *result)
{
  const auto startCS = start->as<ob::CompoundStateSpace::StateType>();
  const double *pos = startCS->as<ob::RealVectorStateSpace::StateType>(0)->values;
  const double *vel = startCS->as<ob::RealVectorStateSpace::StateType>(1)->values;
  const double *cost = startCS->as<ob::RealVectorStateSpace::StateType>(2)->values;

  const double *ctrl = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values;

  auto resultCS = result->as<ob::CompoundStateSpace::StateType>();
  double *resPos = resultCS->as<ob::RealVectorStateSpace::StateType>(0)->values;
  double *resVel = resultCS->as<ob::RealVectorStateSpace::StateType>(1)->values;
  double *resCost = resultCS->as<ob::RealVectorStateSpace::StateType>(2)->values;

  // double force = ctrl[0]; // guaranteed to be within bounds
  // resVel[0] = vel[0] + force - 0.0025 * cos(3 * pos[0]);
  // resVel[0] = std::clamp(resVel[0], -0.07, 0.07);

  // resPos[0] = pos[0] + resVel[0];
  // resPos[0] = std::clamp(resPos[0], -1.2, 0.6);

  double force = ctrl[0]; // guaranteed to be within bounds
  resVel[0] = vel[0] + force - 0.0025 * cos(3 * pos[0]);
  // resVel[0] = std::clamp(resVel[0], -0.07, 0.07);

  resPos[0] = pos[0] + vel[0];

  resCost[0] = cost[0] + std::pow(force, 2);
  // resPos[0] = std::clamp(resPos[0], -1.2, 0.6);
}

class myStateValidityCheckerClass : public ob::StateValidityChecker
{
public:
  myStateValidityCheckerClass(const ob::SpaceInformationPtr &si) 
    : ob::StateValidityChecker(si)
  {
  }

  virtual bool isValid(const ob::State *state) const
  {
    // const auto startCS = start->as<ob::CompoundStateSpace::StateType>();
    // const double cost = startCS->as<ob::RealVectorStateSpace::StateType>(2)->values[0];
    return si_->satisfiesBounds(state);
  }
};

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string motionsFile;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
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

  float max_cost = 1e6;

  for (size_t iter = 0; iter < 10; ++iter) {

    std::cout << "iter " << iter << " w/ costbound " << max_cost << std::endl;

    // Create state space
    auto space(std::make_shared<ob::CompoundStateSpace>());
    space->addSubspace(std::make_shared<ob::RealVectorStateSpace>(1), 1.0); // position
    space->addSubspace(std::make_shared<ob::RealVectorStateSpace>(1), 0.5); // velocity
    space->addSubspace(std::make_shared<ob::RealVectorStateSpace>(1), 0.0); // cost

    // set bounds for position
    ob::RealVectorBounds boundsPos(1);
    boundsPos.setLow(0, -1.2);
    boundsPos.setHigh(0, 0.6);
    space->getSubspace(0)->as<ob::RealVectorStateSpace>()->setBounds(boundsPos);

    // set bounds for velocity
    ob::RealVectorBounds boundsVel(1);
    boundsVel.setLow(0, -0.07);
    boundsVel.setHigh(0, 0.07);
    space->getSubspace(1)->as<ob::RealVectorStateSpace>()->setBounds(boundsVel);

    // set bounds for cost
    ob::RealVectorBounds boundsCost(1);
    boundsCost.setLow(0, 0.0);
    boundsCost.setHigh(0, max_cost);
    space->getSubspace(2)->as<ob::RealVectorStateSpace>()->setBounds(boundsCost);

    // create control space
    auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 1));

    // set the bounds for the control space
    ob::RealVectorBounds cbounds(1);
    cbounds.setLow(-0.0015);
    cbounds.setHigh(0.0015);

    cspace->setBounds(cbounds);

    // construct an instance of space information from this state and control space
    auto si(std::make_shared<oc::SpaceInformation>(space, cspace));

    // set number of control steps
    si->setPropagationStepSize(1);
    si->setMinMaxControlDuration(1, 1);

    // set state propagator
    si->setStatePropagator(&propagate);

    si->setStateValidityChecker(std::make_shared<myStateValidityCheckerClass>(si));

    // create a problem instance
    auto pdef(std::make_shared<ob::ProblemDefinition>(si));

    // create and set a start state
    auto startState = si->allocState();
    auto startStateCS = startState->as<ob::CompoundStateSpace::StateType>();
    startStateCS->as<ob::RealVectorStateSpace::StateType>(0)->values[0] = -0.5;
    startStateCS->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = 0.0;
    startStateCS->as<ob::RealVectorStateSpace::StateType>(2)->values[0] = 0.0;
    pdef->addStartState(startState);
    si->freeState(startState);

    // set goal state
    auto goalState = si->allocState();
    auto goalStateCS = goalState->as<ob::CompoundStateSpace::StateType>();
    goalStateCS->as<ob::RealVectorStateSpace::StateType>(0)->values[0] = 0.45;
    goalStateCS->as<ob::RealVectorStateSpace::StateType>(1)->values[0] = 0.0;
    goalStateCS->as<ob::RealVectorStateSpace::StateType>(2)->values[0] = 0.0;
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
    // si->printSettings(std::cout);

    // print the problem settings
    // pdef->print(std::cout);

    // attempt to solve the problem within timelimit
    float timelimit = 10; 
    ob::PlannerStatus solved = planner->ob::Planner::solve(timelimit);

    std::cout << solved << std::endl;
    auto path = pdef->getSolutionPath()->as<oc::PathControl>();

    std::cout << path->getStateCount() << "," << path->getControlCount() << std::endl;
    assert(path->getStateCount() == path->getControlCount() + 1);

    auto stateCS = path->getStates().back()->as<ob::CompoundStateSpace::StateType>();
    float cost = stateCS->as<ob::RealVectorStateSpace::StateType>(2)->values[0];
    std::cout << "cost: " << cost << std::endl;
    if (solved == ob::PlannerStatus::EXACT_SOLUTION) {
      max_cost = cost;

      std::ofstream out(outputFile);
      out << "pos[m],vel[m/s],a" << std::endl;
      for (size_t i = 0; i < path->getControlCount(); ++i) {
        const auto state = path->getState(i);
        auto stateCS = state->as<ob::CompoundStateSpace::StateType>();
        double pos = stateCS->as<ob::RealVectorStateSpace::StateType>(0)->values[0];
        double vel = stateCS->as<ob::RealVectorStateSpace::StateType>(1)->values[0];

        double a = path->getControl(i)->as<ompl::control::RealVectorControlSpace::ControlType>()->values[0];
        out << pos << "," << vel << "," << a << std::endl;
      }
    }
  }

  return 0;
}
