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

std::ofstream& printState(std::ofstream &stream, std::shared_ptr<ompl::control::SpaceInformation> si, const ob::State* state)
{
  std::vector<double> reals;
  si->getStateSpace()->copyToReals(reals, state);
  stream << "[";
  for (size_t d = 0; d < reals.size(); ++d)
  {
    stream << reals[d];
    if (d < reals.size() - 1)
    {
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}

oc::Control *allocAndFillControl(std::shared_ptr<ompl::control::SpaceInformation> si, const YAML::Node &node)
{
  oc::Control *control = si->allocControl();
  for (size_t idx = 0; idx < node.size(); ++idx)
  {
    double* address = si->getControlSpace()->getValueAddressAtIndex(control, idx);
    if (address) {
      *address = node[idx].as<double>();
    }
  }
  return control;
}

std::ofstream& printAction(std::ofstream &stream, std::shared_ptr<ompl::control::SpaceInformation> si, oc::Control *action)
{
  const size_t dim = si->getControlSpace()->getDimension();
  stream << "[";
  for (size_t d = 0; d < dim; ++d)
  {
    double *address = si->getControlSpace()->getValueAddressAtIndex(action, d);
    stream << *address;
    if (d < dim - 1)
    {
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}

class Motion
{
public:
  std::vector<ob::State*> states;
  std::vector<oc::Control*> actions;

  float cost;

  size_t idx;
};

// forward declaration
struct AStarNode;

struct compareAStarNode
{
  bool operator()(const AStarNode *a, const AStarNode *b) const;
};

// open type
typedef typename boost::heap::d_ary_heap<
    AStarNode *,
    boost::heap::arity<2>,
    boost::heap::compare<compareAStarNode>,
    boost::heap::mutable_<true>>
    open_t;

// Node type (used for open and explored states)
struct AStarNode
{
  const ob::State *state;

  float fScore;
  float gScore;

  const AStarNode* came_from;

  size_t used_motion;

  open_t::handle_type handle;
  bool is_in_open;
};

bool compareAStarNode::operator()(const AStarNode *a, const AStarNode *b) const
{
  // Sort order
  // 1. lowest fScore
  // 2. highest gScore

  // Our heap is a maximum heap, so we invert the comperator function here
  if (a->fScore != b->fScore)
  {
    return a->fScore > b->fScore;
  }
  else
  {
    return a->gScore < b->gScore;
  }
}

// struct AStarOpenNode
// {
//   AStarOpenNode(const AStarNode* node)
//       : node(node)
//   {
//   }

//   bool operator<(const AStarOpenNode &other) const
//   {
//     // Sort order
//     // 1. lowest fScore
//     // 2. highest gScore

//     // Our heap is a maximum heap, so we invert the comperator function here
//     if (node->fScore != other.node->fScore)
//     {
//       return node->fScore > other.node->fScore;
//     }
//     else
//     {
//       return node->gScore < other.node->gScore;
//     }
//   }

//   // friend std::ostream &operator<<(std::ostream &os, const Node &node)
//   // {
//   //   os << "state: " << node.state << " fScore: " << node.fScore
//   //      << " gScore: " << node.gScore;
//   //   return os;
//   // }

//   const AStarNode* node;

//   // const ob::State* state;

//   // float fScore;
//   // float gScore;

//   // typename boost::heap::d_ary_heap<AStarOpenNode, boost::heap::arity<2>,
//   //                                  boost::heap::mutable_<true>>::handle_type
//   //     handle;

//   // typename boost::heap::d_ary_heap<AStarNode, boost::heap::arity<2>,
//   //                                  boost::heap::mutable_<true>>::handle_type
//   //     came_from;

//   // size_t used_motion;
// };

float heuristic(std::shared_ptr<Robot> robot, const ob::State *s, const ob::State *g)
{
  // heuristic is the time it might take to get to the goal
  const auto current_pos = robot->getTransform(s).translation();
  const auto goal_pos = robot->getTransform(g).translation();
  float dist = (current_pos - goal_pos).norm();

  return dist * 0.5;
}

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string motionsFile;
  float delta;
  float maxCost;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
    ("motions,m", po::value<std::string>(&motionsFile)->required(), "motions file (yaml)")
    ("delta", po::value<float>(&delta)->default_value(0.01), "discontinuity bound")
    ("maxCost", po::value<float>(&maxCost)->default_value(std::numeric_limits<float>::infinity()), "cost bound")
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
    for (const auto& action : motion["actions"]) {
      m.actions.push_back(allocAndFillControl(si, action));
    }
    m.cost = m.actions.size();
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
  open_t open;

  // kd-tree for nodes
  ompl::NearestNeighbors<AStarNode*> *T_n;
  if (si->getStateSpace()->isMetricSpace())
  {
    T_n = new ompl::NearestNeighborsGNATNoThreadSafety<AStarNode*>();
  }
  else
  {
    T_n = new ompl::NearestNeighborsSqrtApprox<AStarNode*>();
  }
  T_n->setDistanceFunction([si](const AStarNode* a, const AStarNode* b)
                           { return si->distance(a->state, b->state); });

  auto start_node = new AStarNode();
  start_node->state = startState;
  start_node->gScore = 0;
  start_node->fScore = heuristic(robot, startState, goalState);
  start_node->came_from = nullptr;
  start_node->used_motion = -1;

  auto handle = open.push(start_node);
  start_node->handle = handle;
  start_node->is_in_open = true;

  T_n->add(start_node);

  Motion fakeMotion;
  fakeMotion.idx = -1;
  fakeMotion.states.push_back(si->allocState());

  AStarNode* query_n = new AStarNode();

  ob::State* tmpState = si->allocState();
  std::vector<Motion*> neighbors_m;
  std::vector<AStarNode*> neighbors_n;

  while (!open.empty())
  {
    AStarNode* current = open.top();
    // std::cout << "current";
    // si->printState(current->state);
    if (si->distance(current->state, goalState) <= delta) {
      std::cout << "SOLUTION FOUND!!!!" << std::endl;

      std::vector<const AStarNode*> result;

      const AStarNode* n = current;
      while (n != nullptr) {
        result.push_back(n);
        // std::cout << n->used_motion << std::endl;
        // si->printState(n->state);
        n = n->came_from;
      }
      std::reverse(result.begin(), result.end());

      std::ofstream out(outputFile);
      out << "result:" << std::endl;
      out << "  - states:" << std::endl;
      for (size_t i = 0; i < result.size() - 1; ++i)
      {
        // Compute intermediate states
        const auto node_state = result[i]->state;
        const fcl::Vector3f current_pos = robot->getTransform(node_state).translation();
        const auto &motion = motions.at(result[i+1]->used_motion);
        // skip last state each
        for (size_t k = 0; k < motion.states.size() - 1; ++k)
        {
          const auto state = motion.states[k];
          si->copyState(tmpState, state);
          const fcl::Vector3f relative_pos = robot->getTransform(state).translation();
          robot->setPosition(tmpState, current_pos + relative_pos);

          out << "      - ";
          printState(out, si, tmpState);
          out << std::endl;
        }
        out << std::endl;
      }
      out << "      - ";
      printState(out, si, result.back()->state);
      out << std::endl;
      out << "    actions:" << std::endl;
      for (size_t i = 0; i < result.size() - 1; ++i)
      {
        const auto &motion = motions[result[i+1]->used_motion];
        for (size_t k = 0; k < motion.actions.size(); ++k)
        {
          const auto& action = motion.actions[k];
          out << "      - ";
          printAction(out, si, action);
          out << std::endl;
        }
        out << std::endl;
      }

      return 0;
      break;
    }

    current->is_in_open = false;
    open.pop();

    // find relevant motions (within delta of current state)
    si->copyState(fakeMotion.states[0], current->state);
    robot->setPosition(fakeMotion.states[0], fcl::Vector3f(0,0,0));

    T_m->nearestR(&fakeMotion, delta, neighbors_m);
    // std::cout << "found " << neighbors_m.size() << " motions" << std::endl;
    // Loop over all potential applicable motions
    for (const Motion* motion : neighbors_m) {
      // Compute intermediate states and check their validity
      const auto current_pos = robot->getTransform(current->state).translation();
      float tentative_gScore = current->gScore + motion->cost;
      if (tentative_gScore > maxCost) {
        continue;
      }

      bool motionValid = true;
      for (const auto& state : motion->states)
      {
        // const auto& state = motion->states.back();
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
      // std::cout << "valid " << si->distance(current->state, tmpState) << std::endl;
      // si->printState(tmpState);

      // Check if we have this state (or any within delta) already
      query_n->state = tmpState;
      T_n->nearestR(query_n, delta, neighbors_n);

      // exclude state we came from (otherwise we never add motions that are less than delta away)
      // auto it = std::remove(neighbors_n.begin(), neighbors_n.end(), current);
      // neighbors_n.erase(it, neighbors_n.end());

      // std::cout << neighbors_n.size() << std::endl;

      if (neighbors_n.size() == 0)
      {
        // std::cout << "new state";

        // new state -> add it to open and T_n
        auto node = new AStarNode();
        node->state = si->cloneState(tmpState);
        node->gScore = tentative_gScore;
        node->fScore = tentative_gScore + heuristic(robot, node->state, goalState);
        node->came_from = current;
        node->used_motion = motion->idx;

        auto handle = open.push(node);
        node->handle = handle;
        T_n->add(node);
      }
      else
      {
        // check if we have a better path now
        for (AStarNode* entry : neighbors_n) {
          float delta = entry->gScore - tentative_gScore;
          if (delta > 0) {
            // std::cout << "improve score";
            // si->printState(entry->state);

            entry->gScore = tentative_gScore;
            entry->fScore -= delta;
            assert(entry->fScore >= 0);
            entry->came_from = current;
            entry->used_motion = motion->idx;
            if (entry->is_in_open) {
              open.increase(entry->handle);
            }
          }
        }
      }
    }

  }

  return 1;
}
