#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

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

#include "robots.h"
#include "robotStatePropagator.hpp"
#include "fclStateValidityChecker.hpp"
#include "fclHelper.hpp"

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

  std::shared_ptr<ShiftableDynamicAABBTreeCollisionManager<float>> collision_manager;
  std::vector<fcl::CollisionObjectf *> collision_objects;

  float cost;

  size_t idx;
  // std::string name;
  bool disabled;
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
  fcl::Vector3f used_offset;
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

float heuristic(std::shared_ptr<Robot> robot, const ob::State *s, const ob::State *g)
{
  // heuristic is the time it might take to get to the goal
  const auto current_pos = robot->getTransform(s).translation();
  const auto goal_pos = robot->getTransform(g).translation();
  float dist = (current_pos - goal_pos).norm();
  const float max_vel = 0.5; // m/s
  const float time = dist * max_vel;
  return time;
}

class DBAstar
{

};

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string motionsFile;
  float delta;
  float epsilon;
  float maxCost;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
    ("motions,m", po::value<std::string>(&motionsFile)->required(), "motions file (yaml)")
    ("delta", po::value<float>(&delta)->default_value(0.01), "discontinuity bound (negative to auto-compute with given k)")
    ("epsilon", po::value<float>(&epsilon)->default_value(1.0), "suboptimality bound")
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
      co->computeAABB();
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
  const auto &env_min = env["environment"]["min"];
  const auto &env_max = env["environment"]["max"];
  ob::RealVectorBounds position_bounds(env_min.size());
  for (size_t i = 0; i < env_min.size(); ++i) {
    position_bounds.setLow(i, env_min[i].as<double>());
    position_bounds.setHigh(i, env_max[i].as<double>());
  }
  std::shared_ptr<Robot> robot = create_robot(robotType, position_bounds);

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
    m.cost = m.actions.size() / 10.0f; // time in seconds
    m.idx = motions.size();
    // m.name = motion["name"].as<std::string>();

    // generate collision objects and collision manager
    for (const auto &state : m.states)
    {
      for (size_t part = 0; part < robot->numParts(); ++part) {
        const auto &transform = robot->getTransform(state, part);

        auto co = new fcl::CollisionObjectf(robot->getCollisionGeometry(part));
        co->setTranslation(transform.translation());
        co->setRotation(transform.rotation());
        co->computeAABB();
        m.collision_objects.push_back(co);
      }
    }
    m.collision_manager.reset(new ShiftableDynamicAABBTreeCollisionManager<float>());
    m.collision_manager->registerObjects(m.collision_objects);

    m.disabled = false;

    motions.push_back(m);
  }

  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(motions), std::end(motions), rng);
  for (size_t idx = 0; idx < motions.size(); ++idx) {
    motions[idx].idx = idx;
  }
  std::uniform_real_distribution<> dis_angle(0, 2 * M_PI);

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

  std::cout << "There are " << motions.size() << " motions!" << std::endl;


  //////////////////////////
  if (delta < 0) {
    Motion fakeMotion;
    fakeMotion.idx = -1;
    fakeMotion.states.push_back(si->allocState());
    std::vector<Motion *> neighbors_m;
    size_t num_desired_neighbors = (size_t)-delta;
    size_t num_samples = 100;

    auto state_sampler = si->allocStateSampler();
    float sum_delta = 0.0;
    for (size_t k = 0; k < num_samples; ++k) {
      state_sampler->sampleUniform(fakeMotion.states[0]);
      robot->setPosition(fakeMotion.states[0], fcl::Vector3f(0, 0, 0));

      T_m->nearestK(&fakeMotion, num_desired_neighbors, neighbors_m);

      float max_delta = si->distance(fakeMotion.states[0], neighbors_m.back()->states.front());
      sum_delta += max_delta;
    }
    float adjusted_delta = (sum_delta / num_samples) * 2;
    std::cout << "Automatically adjusting delta to: " << adjusted_delta << std::endl;
    delta = adjusted_delta;

  }
  //////////////////////////

  // if (false)
  {
    size_t num_duplicates = 0;
    Motion fakeMotion;
    fakeMotion.idx = -1;
    fakeMotion.states.push_back(si->allocState());
    std::vector<Motion *> neighbors_m;
    for (const auto& m : motions) {
      if (m.disabled) {
        continue;
      }

      si->copyState(fakeMotion.states[0], m.states[0]);
      T_m->nearestR(&fakeMotion, delta/2, neighbors_m);

      for (Motion* nm : neighbors_m) {
        if (nm == &m || nm->disabled) {
          continue;
        }
        float goal_delta = si->distance(m.states.back(), nm->states.back());
        if (goal_delta < delta/2) {
          // std::cout << nm->idx << " " << goal_delta << " " << delta/2 << std::endl;
          nm->disabled = true;
          ++num_duplicates;
        }
      }
    }
    std::cout << "There are " << num_duplicates << " duplicate motions!" << std::endl;

  }


  //////////////////////////

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
  start_node->fScore = epsilon * heuristic(robot, startState, goalState);
  start_node->came_from = nullptr;
  start_node->used_offset = fcl::Vector3f(0,0,0);
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

  float last_f_score = start_node->fScore;
  size_t expands = 0;
  while (!open.empty())
  {
    AStarNode* current = open.top();
    ++expands;
    if (expands % 1000 == 0) {
      std::cout << "expanded: " << expands << " open: " << open.size() << " nodes: " << T_n->size() << " f-score " << current->fScore << std::endl;
    }
    // if (expands > 100000) {
      // break;
    // }

    // std::cout << "fs " << current->fScore << " " << last_f_score << " " << current << std::endl;
    assert(current->fScore >= last_f_score);
    last_f_score = current->fScore;
    // std::cout << "current";
    // si->printState(current->state);
    if (si->distance(current->state, goalState) <= delta) {
      std::cout << "SOLUTION FOUND!!!! cost: " << current->gScore << std::endl;

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
      out << "delta: " << delta << std::endl;
      out << "epsilon: " << epsilon << std::endl;
      out << "cost: " << current->gScore << std::endl;
      out << "result:" << std::endl;
      out << "  - states:" << std::endl;
      for (size_t i = 0; i < result.size() - 1; ++i)
      {
        // Compute intermediate states
        const auto node_state = result[i]->state;
        const fcl::Vector3f current_pos = robot->getTransform(node_state).translation();
        const auto &motion = motions.at(result[i+1]->used_motion);
        out << "      # ";
        printState(out, si, node_state);
        out << std::endl;
        out << "      # motion " << motion.idx << " with cost " << motion.cost << std::endl;
        // skip last state each
        for (size_t k = 0; k < motion.states.size(); ++k)
        {
          const auto state = motion.states[k];
          si->copyState(tmpState, state);
          const fcl::Vector3f relative_pos = robot->getTransform(state).translation();
          robot->setPosition(tmpState, current_pos + result[i+1]->used_offset + relative_pos);

          if (k < motion.states.size() - 1) {
            out << "      - ";
          } else {
            out << "      # ";
          }
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
        out << "      # motion " << motion.idx << " with cost " << motion.cost << std::endl;
        for (size_t k = 0; k < motion.actions.size(); ++k)
        {
          const auto& action = motion.actions[k];
          out << "      - ";
          printAction(out, si, action);
          out << std::endl;
        }
        out << std::endl;
      }
      // statistics for the motions used
      std::map<size_t, size_t> motionsCount; // motionId -> usage count
      for (size_t i = 0; i < result.size() - 1; ++i)
      {
        auto motionId = result[i+1]->used_motion;
        auto iter = motionsCount.find(motionId);
        if (iter == motionsCount.end()) {
          motionsCount[motionId] = 1;
        } else {
          iter->second += 1;
        }
      }
      out << "    motion_stats:" << std::endl;
      for (const auto& kv : motionsCount) {
        out << "      " << motions[kv.first].idx << ": " << kv.second << std::endl;
      }

      // {
      //   T_n->list(neighbors_n);
      //   std::ofstream out("states.txt");
      //   for (AStarNode* entry : neighbors_n) {
      //     std::vector<double> reals;
      //     si->getStateSpace()->copyToReals(reals, entry->state);
      //     for (size_t d = 0; d < reals.size(); ++d) {
      //       out << reals[d];
      //       if (d < reals.size() - 1) {
      //         out << ",";
      //       }
      //     }
      //     out << "\n";

      //     std::vector<AStarNode*> nbhs;
      //     T_n->nearestK(entry, 2, nbhs);
      //     if (nbhs.size() > 1) {
      //       float dist = si->distance(entry->state, nbhs.back()->state);
      //       if (dist < delta / 2)
      //       {
      //         std::cout << "error?" << dist << " " << entry << " " << nbhs.back() << std::endl;
      //       }
      //     }
      //   }
      // }

      return 0;
      break;
    }

    current->is_in_open = false;
    open.pop();

    // find relevant motions (within delta/2 of current state)
    si->copyState(fakeMotion.states[0], current->state);
    robot->setPosition(fakeMotion.states[0], fcl::Vector3f(0,0,0));

    T_m->nearestR(&fakeMotion, delta/2, neighbors_m);
    // std::shuffle(std::begin(neighbors_m), std::end(neighbors_m), rng);

    // std::cout << "found " << neighbors_m.size() << " motions" << std::endl;
    // Loop over all potential applicable motions
    for (const Motion* motion : neighbors_m) {
      if (motion->disabled) {
        continue;
      }

#if 1
      fcl::Vector3f computed_offset(0, 0, 0);
#else
      float motion_dist = si->distance(fakeMotion.states[0], motion->states[0]);
      float translation_slack = delta/2 - motion_dist;
      assert(translation_slack >= 0);

      // ideally, solve the following optimization problem
      // min_translation fScore
      //     s.t. ||translation|| <= translation_slack // i.e., stay within delta/2
      //          no collisions

      const auto current_pos2 = robot->getTransform(current->state).translation();
      const auto goal_pos = robot->getTransform(goalState).translation();
      fcl::Vector3f computed_offset = (goal_pos - current_pos2).normalized() * translation_slack;

      // std::uniform_real_distribution<> dis_mag(0, translation_slack);
      // float angle = dis_angle(rng);
      // float mag = dis_mag(rng);
      // fcl::Vector3f computed_offset(mag * cos(angle), mag * sin(angle), 0);

      #ifndef NDEBUG
      {
        // check that the computed starting state stays within delta/2
        si->copyState(tmpState, motion->states.front());
        const auto current_pos = robot->getTransform(current->state).translation();
        const auto offset = current_pos + computed_offset;
        const auto relative_pos = robot->getTransform(tmpState).translation();
        robot->setPosition(tmpState, offset + relative_pos);
        std::cout << si->distance(tmpState, current->state)  << std::endl;
        assert(si->distance(tmpState, current->state) <= delta/2 + 1e-5);
      }
      #endif
#endif

      // compute estimated cost
      float tentative_gScore = current->gScore + motion->cost;
      // compute final state
      si->copyState(tmpState, motion->states.back());
      const auto current_pos = robot->getTransform(current->state).translation();
      const auto offset = current_pos + computed_offset;
      const auto relative_pos = robot->getTransform(tmpState).translation();
      robot->setPosition(tmpState, offset + relative_pos);
      // compute estimated fscore
      float tentative_hScore = epsilon * heuristic(robot, tmpState, goalState);
      float tentative_fScore = tentative_gScore + tentative_hScore;

      // skip motions that would exceed cost bound, or that are invalid
      if (tentative_fScore > maxCost || !si->satisfiesBounds(tmpState))
      {
        // std::cout << "skip " << tentative_fScore << " " << maxCost << std::endl;
        continue;
      }

      // Compute intermediate states and check their validity

      // auto start = std::chrono::steady_clock::now();
#if 0
      bool motionValid = true;
      for (const auto& state : motion->states)
      {
        // const auto& state = motion->states.back();
        si->copyState(tmpState, state);
        const auto relative_pos = robot->getTransform(state).translation();
        robot->setPosition(tmpState, offset + relative_pos);

        // std::cout << "check";
        // si->printState(tmpState);

        if (!si->isValid(tmpState)) {
          motionValid = false;
          // std::cout << "invalid";
          break;
        }
      }
      #else
      motion->collision_manager->shift(offset);
      fcl::DefaultCollisionData<float> collision_data;
      motion->collision_manager->collide(bpcm_env.get(), &collision_data, fcl::DefaultCollisionFunction<float>);
      bool motionValid = !collision_data.result.isCollision();
      motion->collision_manager->shift(-offset);

      // for (auto obj : motion->collision_objects) {
      //   obj->setTranslation(obj->getTranslation() + offset);
      // }
      // motion->collision_manager->update(motion->collision_objects);

      // fcl::DefaultCollisionData<float> collision_data;
      // motion->collision_manager->collide(bpcm_env.get(), &collision_data, fcl::DefaultCollisionFunction<float>);
      // bool motionValid = !collision_data.result.isCollision();

      // for (auto obj : motion->collision_objects) {
      //   obj->setTranslation(obj->getTranslation() - offset);
      // }

#endif
      // auto end = std::chrono::steady_clock::now();
      // size_t dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      // std::cout << "cc: " << dt << " ns\n";

      // Skip this motion, if it isn't valid
      if (!motionValid) {
        continue;
      }
      // std::cout << "valid " <<  std::endl;
      // si->printState(tmpState);

      // Check if we have this state (or any within delta/2) already
      query_n->state = tmpState;
      // avoid considering this an old state for very short motions
      // float motion_distance = si->distance(query_n->state, current->state);
      // const float eps = 1e-6;
      // float radius = std::min(delta/2, motion_distance-eps);
      float radius = delta/2;
      T_n->nearestR(query_n, radius, neighbors_n);
      // auto nearest = T_n->nearest(query_n);
      // float nearest_distance = si->distance(nearest->state, tmpState);

      // exclude state we came from (otherwise we never add motions that are less than delta away)
      // auto it = std::remove(neighbors_n.begin(), neighbors_n.end(), current);
      // neighbors_n.erase(it, neighbors_n.end());

      // std::cout << neighbors_n.size() << std::endl;

      if (neighbors_n.size() == 0)
      // if (nearest_distance > radius)
      {
        // new state -> add it to open and T_n
        auto node = new AStarNode();
        node->state = si->cloneState(tmpState);
        node->gScore = tentative_gScore;
        node->fScore = tentative_fScore;
        node->came_from = current;
        node->used_motion = motion->idx;
        node->used_offset = computed_offset;
        node->is_in_open = true;
        auto handle = open.push(node);
        node->handle = handle;
        T_n->add(node);

        // std::cout << "new state " << node->fScore << " " << node << std::endl;
      }
      else
      {
        // T_n->nearestR(query_n, radius, neighbors_n);
        // check if we have a better path now
        for (AStarNode* entry : neighbors_n) {
        // AStarNode* entry = nearest;
          assert(si->distance(entry->state, tmpState) <= delta);
          float delta_score = entry->gScore - tentative_gScore;
          if (delta_score > 0) {
            entry->gScore = tentative_gScore;
            entry->fScore -= delta_score;
            assert(entry->fScore >= 0);
            entry->came_from = current;
            entry->used_motion = motion->idx;
            entry->used_offset = computed_offset;
            if (entry->is_in_open) {
              open.increase(entry->handle);
              // std::cout << "improve score " << entry->fScore << std::endl;
            } else {
              // TODO: is this correct?
              auto handle = open.push(entry);
              entry->handle = handle;
              entry->is_in_open = true;
            }
          }
        }
      }
    }

  }

  query_n->state = goalState;
  const auto nearest = T_n->nearest(query_n);
  if (nearest->gScore == 0) {
    std::cout << "No solution found (not even approxmite)" << std::endl;
    return 1;
  }

  float nearest_distance = si->distance(nearest->state, goalState);
  std::cout << "Nearest to goal: " << nearest_distance << " (delta: " << delta << ")" << std::endl;

  std::cout << "Using approximate solution cost: " << nearest->gScore << std::endl;

  std::vector<const AStarNode*> result;

  const AStarNode* n = nearest;
  while (n != nullptr) {
    result.push_back(n);
    // std::cout << n->used_motion << std::endl;
    // si->printState(n->state);
    n = n->came_from;
  }
  std::reverse(result.begin(), result.end());

  std::ofstream out(outputFile);
  out << "delta: " << delta << std::endl;
  out << "epsilon: " << epsilon << std::endl;
  out << "cost: " << nearest->gScore << std::endl;
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (size_t i = 0; i < result.size() - 1; ++i)
  {
    // Compute intermediate states
    const auto node_state = result[i]->state;
    const fcl::Vector3f current_pos = robot->getTransform(node_state).translation();
    const auto &motion = motions.at(result[i+1]->used_motion);
    out << "      # ";
    printState(out, si, node_state);
    out << std::endl;
    out << "      # motion " << motion.idx << " with cost " << motion.cost << std::endl;
    // skip last state each
    for (size_t k = 0; k < motion.states.size(); ++k)
    {
      const auto state = motion.states[k];
      si->copyState(tmpState, state);
      const fcl::Vector3f relative_pos = robot->getTransform(state).translation();
      robot->setPosition(tmpState, current_pos + result[i+1]->used_offset + relative_pos);

      if (k < motion.states.size() - 1) {
        out << "      - ";
      } else {
        out << "      # ";
      }
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
    out << "      # motion " << motion.idx << " with cost " << motion.cost << std::endl;
    for (size_t k = 0; k < motion.actions.size(); ++k)
    {
      const auto& action = motion.actions[k];
      out << "      - ";
      printAction(out, si, action);
      out << std::endl;
    }
    out << std::endl;
  }
  // statistics for the motions used
  std::map<size_t, size_t> motionsCount; // motionId -> usage count
  for (size_t i = 0; i < result.size() - 1; ++i)
  {
    auto motionId = result[i+1]->used_motion;
    auto iter = motionsCount.find(motionId);
    if (iter == motionsCount.end()) {
      motionsCount[motionId] = 1;
    } else {
      iter->second += 1;
    }
  }
  out << "    motion_stats:" << std::endl;
  for (const auto& kv : motionsCount) {
    out << "      " << motions[kv.first].idx << ": " << kv.second << std::endl;
  }

  return 0;
}
