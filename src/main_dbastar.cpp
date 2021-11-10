#include <fstream>
#include <iostream>
#include <algorithm>

#include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/program_options.hpp>
#include <boost/heap/d_ary_heap.hpp>

// OMPL headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/datastructures/NearestNeighborsSqrtApprox.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>

// #include "robotDubinsCar.h"
#include "robotCarFirstOrder.h"
// #include "robotCarSecondOrder.h"
#include "robotStatePropagator.hpp"
#include "fclStateValidityChecker.hpp"

namespace ob = ompl::base;
namespace oc = ompl::control;

ob::State* allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si, const YAML::Node& node)
{
  ob::State* state = si->allocState();
  std::vector<double> reals;
  for (const auto &value : node) {
    reals.push_back(value.as<double>());
  }
  si->getStateSpace()->copyFromReals(state, reals);
  return state;
}

class Motion
{
public:
  std::vector<ob::State*> states;

  size_t idx;
};

struct AStarNode
{
  AStarNode(const ob::State* state, float fScore, float gScore)
      : state(state), fScore(fScore), gScore(gScore) {}

  bool operator<(const AStarNode &other) const
  {
    // Sort order
    // 1. lowest fScore
    // 2. highest gScore

    // Our heap is a maximum heap, so we invert the comperator function here
    if (fScore != other.fScore)
    {
      return fScore > other.fScore;
    }
    else
    {
      return gScore < other.gScore;
    }
  }

  // friend std::ostream &operator<<(std::ostream &os, const Node &node)
  // {
  //   os << "state: " << node.state << " fScore: " << node.fScore
  //      << " gScore: " << node.gScore;
  //   return os;
  // }

  const ob::State* state;

  float fScore;
  float gScore;

  typename boost::heap::d_ary_heap<AStarNode, boost::heap::arity<2>,
                                   boost::heap::mutable_<true>>::handle_type
      handle;

  typename boost::heap::d_ary_heap<AStarNode, boost::heap::arity<2>,
                                   boost::heap::mutable_<true>>::handle_type
      came_from;

  size_t used_motion;
};

float heuristic(const ob::State *s)
{
  return 0;
}

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string motionsFile;
  float delta;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
    ("motions,m", po::value<std::string>(&motionsFile)->required(), "motions file (yaml)")
    ("delta", po::value<float>(&delta)->default_value(0.01), "discontinuity bound")
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

  // create and set a start state
  auto startState = allocAndFillState(si, robot_node["start"]);
  // si->freeState(startState);

  // set goal state
  auto goalState = allocAndFillState(si, robot_node["goal"]);
  // si->freeState(goalState);

  // load motion primitives
  YAML::Node motions_node = YAML::LoadFile(motionsFile);
  std::vector<Motion> motions;
  for (const auto& motion : motions_node) {
    Motion m;
    for (const auto& state : motion["states"]) {
      m.states.push_back(allocAndFillState(si, state));
    }
    m.idx = motions.size();
    motions.push_back(m);
  }

  // build kd-tree for motion primitives
  ompl::NearestNeighbors<Motion*>* T_m;
  if (si->getStateSpace()->isMetricSpace())
  {
    T_m = new ompl::NearestNeighborsGNATNoThreadSafety<Motion*>();
  } else {
    T_m = new ompl::NearestNeighborsSqrtApprox<Motion*>();
  }
  T_m->setDistanceFunction([si, motions](const Motion* a, const Motion* b) { return si->distance(a->states[0], b->states[0]); });

  for (auto& motion : motions) {
    T_m->add(&motion);
  }

  // db-A* search
  typedef typename boost::heap::d_ary_heap<AStarNode, boost::heap::arity<2>,
                                           boost::heap::mutable_<true>>
      open_t;

  open_t open;

  // kd-tree for nodes
  typedef std::pair<ob::State*, open_t::handle_type> nn_t;

  ompl::NearestNeighbors<nn_t> *T_n;
  if (si->getStateSpace()->isMetricSpace())
  {
    T_n = new ompl::NearestNeighborsGNATNoThreadSafety<nn_t>();
  }
  else
  {
    T_n = new ompl::NearestNeighborsSqrtApprox<nn_t>();
  }
  T_n->setDistanceFunction([si](const nn_t& a, const nn_t& b)
                           { return si->distance(a.first, b.first); });

  auto handle = open.push(AStarNode(startState, heuristic(startState), 0));
  (*handle).handle = handle;
  (*handle).used_motion = -1;

  T_n->add(std::make_pair<>(startState, handle));

  Motion fakeMotion;
  fakeMotion.idx = -1;
  fakeMotion.states.push_back(si->allocState());

  nn_t query_n;

  ob::State* tmpState = si->allocState();
  std::vector<Motion*> neighbors_m;
  std::vector<nn_t> neighbors_n;

  while (!open.empty())
  {
    AStarNode current = open.top();
    std::cout << "current";
    si->printState(current.state);
    if (si->distance(current.state, goalState) <= delta) {
      std::cout << "SOLUTION FOUND!!!!" << std::endl;

      auto handle = current.handle;
      while ((*handle).used_motion != -1) {
        std::cout << (*handle).used_motion << std::endl;
        si->printState((*handle).state);

        handle = (*handle).came_from;
      }

      break;
    }

    open.pop();

    // find relevant motions (within delta of current state)
    si->copyState(fakeMotion.states[0], current.state);
    robot->setPosition(fakeMotion.states[0], fcl::Vector3f(0,0,0));

    T_m->nearestR(&fakeMotion, delta, neighbors_m);

    // Loop over all potential applicable motions
    for (const Motion* motion : neighbors_m) {
      // Compute intermediate states and check their validity
      const auto current_pos = robot->getTransform(current.state).translation();

      bool motionValid = true;
      for (const auto& state : motion->states) {
        si->copyState(tmpState, state);
        const auto relative_pos = robot->getTransform(state).translation();
        robot->setPosition(tmpState, current_pos + relative_pos);

        // std::cout << "check";
        // si->printState(tmpState);

        if (!si->isValid(tmpState)) {
          motionValid = false;
          // std::cout << "invalid";
          break;
        }
      }

      // Skip this motion, if it isn't valid
      if (!motionValid) {
        continue;
      }
      std::cout << "valid " << si->distance(current.state, tmpState) << std::endl;
      si->printState(tmpState);

      // Check if we have this state (or any within delta) already
      query_n.first = tmpState;
      T_n->nearestR(query_n, delta, neighbors_n);

      std::cout << neighbors_n.size() << std::endl;

      float tentative_gScore = current.gScore + motion->states.size();
      if (neighbors_n.size() == 0)
      {
        std::cout << "new state";

        // new state -> add it to open and T_n
        auto new_state = si->cloneState(tmpState);
        auto handle = open.push(AStarNode(new_state, heuristic(new_state), tentative_gScore));
        (*handle).handle = handle;
        (*handle).came_from = current.handle;
        (*handle).used_motion = motion->idx;
        T_n->add(std::make_pair<>(new_state, handle));
      }
      else
      {
        // check if we have a better path now
        for (const nn_t& entry : neighbors_n) {
          auto handle = entry.second;
          float delta = (*handle).gScore - tentative_gScore;
          if (delta > 0) {
            std::cout << "improve score";
            si->printState((*handle).state);

            (*handle).gScore = tentative_gScore;
            (*handle).fScore -= delta;
            assert((*handle).fScore >= 0);
            (*handle).came_from = current.handle;
            (*handle).used_motion = motion->idx;
            open.increase(handle);
          }
        }
      }
    }

  }

  return 0;
}
