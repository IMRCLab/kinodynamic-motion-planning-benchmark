#include <algorithm>
// // #include <boost/graph/graphviz.hpp>
#include "Eigen/Core"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
//
#include <flann/flann.hpp>
// #include <msgpack.hpp>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <yaml-cpp/yaml.h>
//
// // #include <boost/functional/hash.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include <boost/program_options.hpp>
//
// // OMPL headers

#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/datastructures/NearestNeighbors.h>
// #include <ompl/datastructures/NearestNeighborsFLANN.h>
// #include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>
// #include <ompl/datastructures/NearestNeighborsSqrtApprox.h>
//
#include "fclHelper.hpp"
#include "fclStateValidityChecker.hpp"
// #include "ompl/base/ScopedState.h"
// #include "robotStatePropagator.hpp"
// #include "robots.h"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace po = boost::program_options;

using Sample = std::vector<double>;
using Sample_ = ob::State;

#include "croco_macros.hpp"

// boost stuff for the graph
// #include <boost/graph/adjacency_list.hpp>
// #include <boost/graph/dijkstra_shortest_paths.hpp>
// #include <boost/graph/graph_traits.hpp>
// #include <boost/graph/undirected_graph.hpp>
// #include <boost/property_map/property_map.hpp>

template <typename T>
void set_from_yaml(YAML::Node &node, T &var, const char *name) {

  if (YAML::Node parameter = node[name]) {
    var = parameter.as<T>();
  }
}

static inline double normalize_angle(double angle) {
  const double result = fmod(angle + M_PI, 2.0 * M_PI);
  if (result <= 0.0)
    return result + M_PI;
  return result - M_PI;
}

inline double norm_sq(double *x, size_t n) {
  double out = 0;
  for (size_t i = 0; i < n; i++) {
    out += x[i] * x[i];
  }
  return out;
}

void copyToArray(const ompl::base::StateSpacePtr &space, double *reals,
                 const ompl::base::State *source);

double _distance_angle(double a, double b);

double distance_squared_se2(const double *x, const double *y, const double *s);

// a + s * b
void add(const double *a, const double *b, size_t n, double s, double *out);

// a + sb * b + sc * c
void add2(const double *a, const double *b, const double *c, size_t n,
          double sb, double sc, double *out);

template <class T> struct L2Q {
  typedef bool is_kdtree_distance;

  typedef T ElementType;
  typedef typename flann::Accumulator<T>::Type ResultType;
  L2Q(std::function<double(const double *, const double *, size_t)> fun)
      : fun(fun) {}
  std::function<double(const double *, const double *, size_t)> fun;
  /**
   *  Compute the squared Euclidean distance between two vectors.
   *
   *	This is highly optimised, with loop unrolling, as it is one
   *	of the most expensive inner loops.
   *
   *	The computation of squared root at the end is omitted for
   *	efficiency.
   */
  template <typename Iterator1, typename Iterator2>
  ResultType operator()(Iterator1 a, Iterator2 b, size_t size,
                        ResultType worst_dist = -1) const {
    return fun(a, b, size);
  }

  /**
   *	Partial euclidean distance, using just one dimension. This is used by
   *the kd-tree when computing partial distances while traversing the tree.
   *
   *	Squared root is omitted for efficiency.
   */
  template <typename U, typename V>
  inline ResultType accum_dist(const U &a, const V &b, int) const {
    return std::sqrt((a - b) * (a - b));
  }
};

template <typename _T> class FLANNDistanceQ {
public:
  using ElementType = _T;
  using ResultType = double;
  typedef bool is_kdtree_distance;

  FLANNDistanceQ(
      const typename ompl::NearestNeighbors<_T>::DistanceFunction &distFun)
      : distFun_(distFun) {}

  template <typename Iterator1, typename Iterator2>
  ResultType operator()(Iterator1 a, Iterator2 b, size_t /*size*/,
                        ResultType /*worst_dist*/ = -1) const {
    return distFun_(*a, *b);
  }

protected:
  const typename ompl::NearestNeighbors<_T>::DistanceFunction &distFun_;
};

void print_vec(const double *a, size_t n, bool eof = true);

ob::State *
_allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si,
                   const std::vector<double> &reals);

ob::State *
allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si,
                  const YAML::Node &node);

std::ostream &printState(std::ostream &stream, const std::vector<double> &x);

std::ostream &printState(std::ostream &stream,
                         std::shared_ptr<ompl::control::SpaceInformation> si,
                         const ob::State *state,
                         bool add_brackets_comma = true);

// void ompl::base::StateSpace::copyFromReals(
//     State *destination, const std::vector<double> &reals) const;

void copyFromRealsControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                          oc::Control *out, const std::vector<double> &reals);

oc::Control *
allocAndFillControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                    const YAML::Node &node);

std::ostream &printAction(std::ostream &stream,
                          std::shared_ptr<ompl::control::SpaceInformation> si,
                          oc::Control *action);

struct HeuNodeWithIndex {

  int index;
  const ob::State *state;
  double dist;
};

struct HeuNode {
  const ob::State *state;
  double dist;
};

class Motion {
public:
  std::vector<ob::State *> states;
  std::vector<oc::Control *> actions;

  std::shared_ptr<ShiftableDynamicAABBTreeCollisionManager<float>>
      collision_manager;
  std::vector<fcl::CollisionObjectf *> collision_objects;

  float cost;

  size_t idx;
  // std::string name;
  bool disabled;
};

// forward declaration
struct AStarNode;

struct compareAStarNode {
  bool operator()(const AStarNode *a, const AStarNode *b) const;
};

// open type
typedef typename boost::heap::d_ary_heap<AStarNode *, boost::heap::arity<2>,
                                         boost::heap::compare<compareAStarNode>,
                                         boost::heap::mutable_<true>>
    open_t;

// Node type (used for open and explored states)
struct AStarNode {
  const ob::State *state;

  float fScore;
  float gScore;
  float hScore;

  const AStarNode *came_from;
  fcl::Vector3f used_offset;
  size_t used_motion;

  open_t::handle_type handle;
  bool is_in_open = false;
  bool valid = true;
};

float heuristic(std::shared_ptr<RobotOmpl> robot, const ob::State *s,
                const ob::State *g);

template <typename Fun, typename... Args>
auto timed_fun(Fun fun, Args &&...args) {
  auto tic = std::chrono::high_resolution_clock::now();
  auto out = fun(std::forward<Args>(args)...);
  auto tac = std::chrono::high_resolution_clock::now();
  return std::make_pair(
      out, std::chrono::duration<double, std::milli>(tac - tic).count());
}

using Ei = std::pair<int, double>;
using EdgeList = std::vector<std::pair<int, int>>;
using DistanceList = std::vector<double>;

void get_distance_all_vertices(const EdgeList &edge_list,
                               const DistanceList &distance_list,
                               double *dist_out, int *parents_out, int n,
                               int goal);

using Edge = std::pair<int, int>;
void backward_tree_with_dynamics(
    const std::vector<std::vector<double>> &data,
    std::vector<Motion> &primitives, std::vector<Edge> &edge_list,
    std::vector<double> &distance_list,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env,
    double delta_sq);

// std::vector<double>;
// std::vector<double>;

void generate_batch(std::function<void(double *, size_t)> free_sampler,
                    std::function<bool(Sample_ *)> checker,
                    size_t num_samples_trials, size_t dim,
                    std::shared_ptr<ompl::control::SpaceInformation> si,
                    std::vector<Sample_ *> &out);

struct SampleNode {
  Sample_ *x;
  double dist;
  int parent;
};

using HeuristicMap = std::vector<SampleNode>;

void compute_heuristic_map(const EdgeList &edge_list,
                           const DistanceList &distance_list,
                           const std::vector<Sample_ *> &batch_samples,
                           std::vector<SampleNode> &heuristic_map);

#if 0 
void build_heuristic_motions(
    const std::vector<Sample> &batch_samples /* goal should be last */,
    std::vector<SampleNode> &heuristic_map, std::vector<Motion> &motions,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env,
    double delta_sq) {

  EdgeList edge_list;
  DistanceList distance_list;

  auto out = timed_fun([&] {
    backward_tree_with_dynamics(batch_samples, motions, edge_list,
                                distance_list, bpcm_env, delta_sq);
    return 0;
  });
  std::cout << "Motions Backward time: " << out.second << std::endl;

  compute_heuristic_map(edge_list, distance_list, batch_samples, heuristic_map);
}
#endif

void write_heuristic_map(
    const std::vector<SampleNode> &heuristic_map,
    const std::shared_ptr<ompl::control::SpaceInformation> si,
    const char *filename = "heu_map.txt");

// bool check_edge_at_resolution(
//     const double *start, const double *goal, int n,
//     std::function<bool(const double *, size_t n)> check_fun,
//     double resolution = .2) {

bool check_edge_at_resolution(const Sample_ *start, const Sample_ *goal,
                              std::shared_ptr<RobotOmpl> robot,
                              double resolution = .2);

void build_heuristic_distance(const std::vector<Sample_ *> &batch_samples,
                              std::shared_ptr<RobotOmpl> robot,
                              std::vector<SampleNode> &heuristic_map,
                              double distance_threshold, double resolution);

double euclidean_distance_squared(const double *x, const double *y, size_t n);

double euclidean_distance_scale_squared(const double *x, const double *y,
                                        const double *s, size_t n);

double euclidean_distance(const double *x, const double *y, size_t n);

#if 0
double query_heuristic_map(
    const HeuristicMap &map, const std::vector<double> &x,
    std::function<double(const double *, const double *, size_t n)>
        distance_fun = euclidean_distance_squared) {

  std::vector<double> distances(map.size());
  std::transform(map.begin(), map.end(), distances.begin(), [&](const auto &y) {
    // TODO: if it is too far: then infinity!
    return distance_fun(y.x.data(), x.data(), x.size());
  });
  auto it = std::min_element(distances.begin(), distances.end());

  size_t index = std::distance(distances.begin(), it);
  return map[index].dist;
}
#endif

// dbastar:
// /home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/src/fclHelper.hpp:38:
// void
// ShiftableDynamicAABBTreeCollisionManager<S>::shift_recursive(fcl::detail::NodeBase<fcl::AABB<S>
// >*, fcl::Vector3<S_>&) [with S = floa t; fcl::Vector3<S_> =
// Eigen::Matrix<float, 3, 1>]: Assertion `node->bv.equal(obj->getAABB())'
// failed. fish: Job 1, './dbastar -i ../benchmark/unicy…' terminated by
// signal SIGABRT (Abort)

// TODO: add the angular velocity also!!

// T

// change the nearest neighbour search!!
// the sqrt is even slower
// check the implementation of RRT.

void print_matrix(std::ostream &out,
                  const std::vector<std::vector<double>> &data);

double heuristicCollisionsTree(ompl::NearestNeighbors<HeuNode *> *T_heu,
                               const ob::State *s,
                               std::shared_ptr<RobotOmpl> robot);

enum class Duplicate_detection {
  NO = 0,
  HARD = 1,
  SOFT = 2,
};

enum class Terminate_status {
  SOLVED = 0,
  MAX_EXPANDS = 1,
  EMPTY_QUEUE = 2,
  UNKNOWN = 3,
};


template <typename T>
void set_from_boostop(po::options_description &desc, T &var, const char *name) {
  desc.add_options()(name, po::value<T>(&var)->default_value(var));
}




struct Inout_db {

  std::string inputFile;
  std::string motionsFile;
  std::string outFile;
  std::string problem_name;
  double cost = -1;
  bool solved = 0;
  double cost_with_delta_time = -1;

  // const Eigen::IOFormat FMT(6, Eigen::DontAlignCols, ",", ",", "", "", "[",
  // "]");
  // Eigen::VectorXd start;
  // Eigen::VectorXd goal;
  // std::vector<Eigen::VectorXd> xs;
  // std::vector<Eigen::VectorXd> us;

  void print(std::ostream &out);

  void add_options(po::options_description &desc);

  void read_from_yaml(YAML::Node &node);

  void read_from_yaml(const char *file);
};

struct Options_db {

  float delta = .3;
  float epsilon = 1.;
  float alpha = .5;
  bool filterDuplicates = true;
  float maxCost = std::numeric_limits<float>::infinity();
  int new_heu = 0;
  size_t max_motions = std::numeric_limits<int>::max();
  double resolution = .2;
  double delta_factor_goal = 1;
  double cost_delta_factor = 0;
  int rebuild_every = 5000;
  size_t num_sample_trials = 3000;
  size_t max_expands = 1e6;
  bool cut_actions = false;
  int duplicate_detection_int = 0;
  bool use_landmarks = false;
  double factor_duplicate_detection = 2;
  double epsilon_soft_duplicate = 1.5;
  bool add_node_if_better = false;
  bool debug = false;
  bool add_after_expand = false; // this does not improve cost of closed nodes. it is fine if heu is admissible
  bool propagate_controls = false; // TODO: check what happens, in the style of Raul Shome

  void add_options(po::options_description &desc);

  void print(std::ostream &out);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
};

struct Result_db {

  bool feasible = false;
  double cost = -1;
  double cost_with_delta_time = -1;
  void print(std::ostream &out) {
    std::string be = "";
    std::string af = ": ";
    out << be << STR(feasible, af) << std::endl;
    out << be << STR(cost, af) << std::endl;
    out << be << STR(cost_with_delta_time, af) << std::endl;
  }
};

struct Time_benchmark {

  double time_nearestMotion = 0.0;
  double time_nearestNode = 0.0;
  double time_nearestNode_add = 0.0;
  double time_nearestNode_search = 0.0;
  double time_collisions = 0.0;
  double prepare_time = 0.0;
  double total_time = 0.;
  int expands = 0;
  int num_nn_motions = 0;
  int num_nn_states = 0;
  int num_col_motions = 0;
  int motions_tree_size = 0;
  int states_tree_size = 0;

  void write(std::ostream &out);
};

void generate_env(YAML::Node &env,
                  std::vector<fcl::CollisionObjectf *> &obstacles,
                  fcl::BroadPhaseCollisionManagerf *bpcm_env);

void load_motion_primitives(const std::string &motionsFile, RobotOmpl &robot,
                            std::vector<Motion> &motions, int max_motions,
                            bool cut_actions);

double automatic_delta(double delta_in, double alpha, RobotOmpl &robot,
                       ompl::NearestNeighbors<Motion *> &T_m);

void filte_duplicates(std::vector<Motion> &motions, double delta, double alpha,
                      RobotOmpl &robot, ompl::NearestNeighbors<Motion *> &T_m);

struct Heu_fun {
  virtual double h(const ompl::base::State *x) = 0;
  virtual ~Heu_fun() = default;
};

struct Heu_euclidean : Heu_fun {

  Heu_euclidean(std::shared_ptr<RobotOmpl> robot, ob::State *goal)
      : robot(robot), goal(goal) {}

  std::shared_ptr<RobotOmpl> robot;
  ob::State *goal;

  virtual double h(const ompl::base::State *x) override {
    return robot->cost_lower_bound(x, goal);
  }

  virtual ~Heu_euclidean() override{};
};

struct Heu_blind : Heu_fun {

  Heu_blind() {}

  virtual double h(const ompl::base::State *x) override { return 0; }

  virtual ~Heu_blind() override{};
};

struct Heu_roadmap : Heu_fun {

  std::shared_ptr<RobotOmpl> robot;
  ompl::NearestNeighbors<HeuNode *> *T_heu;

  Heu_roadmap(std::shared_ptr<RobotOmpl> robot,
              ompl::NearestNeighbors<HeuNode *> *T_heu)
      : robot(robot), T_heu(T_heu) {}

  virtual double h(const ompl::base::State *x) override {
    return heuristicCollisionsTree(T_heu, x, robot);
  }

  virtual ~Heu_roadmap() override{};
};

int solve(Options_db &options_db, Inout_db &result);
