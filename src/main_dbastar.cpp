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

#include "fclHelper.hpp"
#include "fclStateValidityChecker.hpp"
#include "ompl/base/ScopedState.h"
#include "robotStatePropagator.hpp"
#include "robots.h"

namespace ob = ompl::base;
namespace oc = ompl::control;

#define ERROR_WITH_INFO(msg)                                                   \
  throw std::runtime_error(__FILE__ + std::string(":") +                       \
                           std::to_string(__LINE__) + "\"" +                   \
                           std::string(msg) + "\"\n");

// boost stuff for the graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/property_map/property_map.hpp>

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
                 const ompl::base::State *source) {
  const auto &locations = space->getValueLocations();
  for (std::size_t i = 0; i < locations.size(); ++i)
    reals[i] = *space->getValueAddressAtLocation(source, locations[i]);
}

double distance_angle(double a, double b) {
  double result = b - a;
  if (result > M_PI)
    result -= 2 * M_PI;
  else if (result < -M_PI)
    result += 2 * M_PI;
  return result;
}

double distance_squared_se2(const double *x, const double *y, const double *s) {

  double d0 = s[0] * (x[0] - y[0]);
  double d1 = s[1] * (x[1] - y[1]);
  double d2 = s[2] * distance_angle(x[2], y[2]);
  return d0 * d0 + d1 * d1 + d2 * d2;
};

// a + s * b
void add(const double *a, const double *b, size_t n, double s, double *out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + s * b[i];
  }
}

// a + sb * b + sc * c
void add2(const double *a, const double *b, const double *c, size_t n,
          double sb, double sc, double *out) {
  for (size_t i = 0; i < n; i++) {
    out[i] = a[i] + sb * b[i] + sc * c[i];
  }
}

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

void print_vec(const double *a, size_t n, bool eof = true) {
  for (size_t i = 0; i < n; i++) {
    std::cout << a[i] << " ";
  }
  if (eof)
    std::cout << std::endl;
}

ob::State *
_allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si,
                   const std::vector<double> &reals) {
  ob::State *state = si->allocState();
  si->getStateSpace()->copyFromReals(state, reals);
  return state;
}

ob::State *
allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si,
                  const YAML::Node &node) {
  ob::State *state = si->allocState();
  std::vector<double> reals;
  for (const auto &value : node) {
    reals.push_back(value.as<double>());
  }
  si->getStateSpace()->copyFromReals(state, reals);
  return state;
}

std::ostream &printState(std::ostream &stream,
                         std::shared_ptr<ompl::control::SpaceInformation> si,
                         const ob::State *state) {
  std::vector<double> reals;
  si->getStateSpace()->copyToReals(reals, state);
  stream << "[";
  for (size_t d = 0; d < reals.size(); ++d) {
    stream << reals[d];
    if (d < reals.size() - 1) {
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}

oc::Control *
allocAndFillControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                    const YAML::Node &node) {
  oc::Control *control = si->allocControl();
  for (size_t idx = 0; idx < node.size(); ++idx) {
    double *address =
        si->getControlSpace()->getValueAddressAtIndex(control, idx);
    if (address) {
      *address = node[idx].as<double>();
    }
  }
  return control;
}

std::ofstream &printAction(std::ofstream &stream,
                           std::shared_ptr<ompl::control::SpaceInformation> si,
                           oc::Control *action) {
  const size_t dim = si->getControlSpace()->getDimension();
  stream << "[";
  for (size_t d = 0; d < dim; ++d) {
    double *address = si->getControlSpace()->getValueAddressAtIndex(action, d);
    stream << *address;
    if (d < dim - 1) {
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}

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
  bool is_in_open;
};

bool compareAStarNode::operator()(const AStarNode *a,
                                  const AStarNode *b) const {
  // Sort order
  // 1. lowest fScore
  // 2. highest gScore

  // Our heap is a maximum heap, so we invert the comperator function here
  if (a->fScore != b->fScore) {
    return a->fScore > b->fScore;
  } else {
    return a->gScore < b->gScore;
  }

  // return a->hScore > b->hScore;
}

float heuristic(std::shared_ptr<Robot> robot, const ob::State *s,
                const ob::State *g) {
  // heuristic is the time it might take to get to the goal
  const auto current_pos = robot->getTransform(s).translation();
  const auto goal_pos = robot->getTransform(g).translation();
  float dist = (current_pos - goal_pos).norm();
  const float max_vel = robot->maxSpeed(); // m/s
  const float time = dist / max_vel;
  return time;
}

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
                               int goal) {

  using graph_t = boost::adjacency_list<
      boost::listS, boost::vecS, boost::undirectedS, boost::no_property,
      boost::property<boost::edge_weight_t, double>>; // int or double?

  graph_t g(edge_list.data(), edge_list.data() + edge_list.size(),
            distance_list.data(), n);

  typedef boost::graph_traits<graph_t>::vertex_descriptor vertex_descriptor;

  std::vector<vertex_descriptor> p(num_vertices(g));
  vertex_descriptor s = vertex(goal, g);

  bool verbose = false;
  if (verbose)
    boost::write_graphviz(std::cout, g);

  dijkstra_shortest_paths(
      g, s,
      boost::predecessor_map(boost::make_iterator_property_map(
                                 p.begin(), get(boost::vertex_index, g)))
          .distance_map(boost::make_iterator_property_map(
              dist_out, get(boost::vertex_index, g))));

  boost::graph_traits<graph_t>::vertex_iterator vi, vend;
  for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {
    parents_out[*vi] = p[*vi];
  }

  if (verbose) {
    std::cout << "distances and parents:" << std::endl;
    boost::graph_traits<graph_t>::vertex_iterator vi, vend;
    for (boost::tie(vi, vend) = vertices(g); vi != vend; ++vi) {

      std::cout << "distance(" << *vi << ") = " << dist_out[*vi] << ", ";
      std::cout << "parent(" << *vi << ") = " << p[*vi] << std::endl;
    }
    std::cout << std::endl;
  }
}

using Edge = std::pair<int, int>;
void backward_tree_with_dynamics(
    const std::vector<std::vector<double>> &data,
    std::vector<Motion> &primitives, std::vector<Edge> &edge_list,
    std::vector<double> &distance_list,
    std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env,
    double delta_sq) {

  std::cout << "check with motions" << std::endl;

  std::function<double(const double *, const double *, size_t)> distance =
      [](const double *x, const double *y, size_t n) {
        if (n != 3) {
          throw std::runtime_error("n should be 3");
        }
        double s[3] = {1., 1., 1.};
        return distance_squared_se2(x, y, s);
      };

  int dim = 3;

  flann::Matrix<double> dataset(new double[data.size() * 3], data.size(), dim);
  for (size_t i = 0; i < data.size(); i++) {
    std::copy(data[i].begin(), data[i].end(), dataset[i]);
  }

  flann::KDTreeSingleIndex<L2Q<double>> index(
      dataset, flann::KDTreeSingleIndexParams(), L2Q<double>(distance));

  index.buildIndex();

  auto is_applicable = [&](auto &primitive, auto &point) {
    ob::State *last = primitive.states.back();
    auto lastTyped = last->as<ob::SE2StateSpace::StateType>();
    double d = distance_angle(point[2], lastTyped->getYaw());
    if (d * d < delta_sq)
      return true;
    else
      return false;
  };

  auto apply = [&](auto &primitive, auto &point, auto &point_out) {
    ob::State *first = primitive.states.front();
    ob::State *last = primitive.states.back();

    auto firstTyped = first->as<ob::SE2StateSpace::StateType>();
    auto lastTyped = last->as<ob::SE2StateSpace::StateType>();

    std::vector<double> v_first{firstTyped->getX(), firstTyped->getY(),
                                firstTyped->getYaw()}; //
    std::vector<double> v_last{lastTyped->getX(), lastTyped->getY(),
                               lastTyped->getYaw()}; //
                                                     //
    add2(v_first.data(), v_last.data(), point.data(), 3, -1., 1.,
         point_out.data());
    point_out[2] = normalize_angle(point_out[2]);
  };

  auto check_neighs = [&](auto &point, auto &idxs) {
    std::vector<std::vector<double>> v_dis;
    flann::Matrix<double> dataset(new double[3], 1, dim);
    std::copy(point.begin(), point.end(), dataset[0]);
    index.radiusSearch(dataset, idxs, v_dis, delta_sq, flann::SearchParams());
  };

  std::ofstream file("debug_collision.txt");

  if (true) {
    file << "dataset" << std::endl;
    for (size_t i = 0; i < data.size(); i++) {
      file << i << " ";
      auto &d = data.at(i);
      for (auto &s : d)
        file << s << " ";
      file << std::endl;
    }
  }
  file.close();
  std::cout << "closing file" << std::endl;

  bool verbose = false;
  bool debug = false;
  if (debug)
    file << "nodes" << std::endl;

  for (size_t i = 0; i < data.size(); i++) {
    for (size_t j = 0; j < primitives.size(); j++) {
      bool applicable = is_applicable(primitives[j], data[i]);
      if (applicable) {
        if (verbose) {
          std::cout << "motion " << j << "applicable in " << i << std::endl;
        }
        std::vector<double> point_out(3);
        apply(primitives[j], data[i], point_out);
        std::vector<std::vector<int>> idx;
        check_neighs(point_out, idx);
        if (idx.size()) {

          auto motion = &primitives[j];

          // TODO: continue here: I have to check for collisions!
          // Future: do this lazily? How and when?

          // ob::State* first = motion->states.front();
          // auto firstTyped = first->as<ob::SE2StateSpace::StateType>();
          // std::vector<double> v_first{ firstTyped->getX() ,
          // firstTyped->getY() , firstTyped->getYaw()   };  //
          //
          //
          fcl::Vector3f offset(point_out.at(0), point_out.at(1), 0);
          if (debug)
            file << "i " << i << " m " << j << " "
                 << "offset " << offset(0) << offset(1) << offset(2) << " ";
          auto out = timed_fun([&] {
            motion->collision_manager->shift(offset);
            fcl::DefaultCollisionData<float> collision_data;
            motion->collision_manager->collide(
                bpcm_env.get(), &collision_data,
                fcl::DefaultCollisionFunction<float>);
            bool motionValid = !collision_data.result.isCollision();
            motion->collision_manager->shift(-offset);
            return motionValid;
          });

          bool motionValid = out.first;
          if (debug)
            file << "valid " << motionValid << " ";

          if (motionValid) {
            if (debug)
              file << "neig ";
            for (size_t r = 0; r < idx.front().size(); r++) {
              auto &rr = idx.front()[r];
              if (verbose) {
                std::cout << "motion " << j << " from " << rr << " to " << i
                          << std::endl;
                print_vec(data.at(rr).data(), 3, false);
                std::cout << " --- "
                          << " ";
                print_vec(data.at(i).data(), 3);
              }

              if (debug)
                file << rr << " ";

              ob::State *first = primitives[j].states.front();
              ob::State *last = primitives[j].states.back();

              auto firstTyped = first->as<ob::SE2StateSpace::StateType>();
              auto lastTyped = last->as<ob::SE2StateSpace::StateType>();
              //
              std::vector<double> v_first{firstTyped->getX(),
                                          firstTyped->getY(),
                                          firstTyped->getYaw()}; //
              std::vector<double> v_last{lastTyped->getX(), lastTyped->getY(),
                                         lastTyped->getYaw()}; //
                                                               //
              if (verbose) {
                print_vec(v_first.data(), 3, false);
                std::cout << " --- "
                          << " ";
                print_vec(v_last.data(), 3);
              }

              edge_list.push_back({i, rr});
              distance_list.push_back(primitives[j].cost);
            }
          } else {
            // std::cout << "collision" << std::endl;
          }
        }
        if (debug)
          file << std::endl;
      }
    }
  }
  std::cout << "motion connection DONE" << std::endl;
}

using Sample = std::vector<double>;

void generate_batch(std::function<void(double *, size_t)> free_sampler,
                    std::function<bool(const double *, size_t)> checker,
                    size_t num_samples_trials, size_t dim,
                    std::vector<Sample> &out) {

  std::vector<std::vector<double>> data;
  size_t found_samples = 0;
  for (size_t i = 0; i < num_samples_trials; i++) {
    std::vector<double> x(dim);
    free_sampler(x.data(), dim);

    if (checker(x.data(), dim)) {
      found_samples++;
      out.push_back(x);
    }
  }

  std::cout << "Sampling DONE.  " << out.size() << " samples" << std::endl;
}

struct SampleNode {
  std::vector<double> x;
  double dist;
  int parent;
};

using HeuristicMap = std::vector<SampleNode>;

void compute_heuristic_map(const EdgeList &edge_list,
                           const DistanceList &distance_list,
                           const std::vector<Sample> &batch_samples,
                           std::vector<SampleNode> &heuristic_map) {
  std::vector<double> distances(batch_samples.size());
  std::vector<int> parents(batch_samples.size());

  auto out3 = timed_fun([&] {
    get_distance_all_vertices(edge_list, distance_list, distances.data(),
                              parents.data(), batch_samples.size(),
                              batch_samples.size() - 1);
    return 0;
  });

  std::cout << "time boost " << out3.second << std::endl;

  heuristic_map.clear();
  heuristic_map.resize(batch_samples.size());
  for (size_t i = 0; i < batch_samples.size(); i++) {
    heuristic_map.at(i) = {batch_samples.at(i), distances.at(i), parents.at(i)};
  }
}

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

void write_heuristic_map(const std::vector<SampleNode> &heuristic_map,
                         const char *filename = "heu_map.txt") {

  std::ofstream file(filename);
  for (auto &n : heuristic_map) {

    for (auto &e : n.x) {
      file << e << " ";
    }
    file << n.dist << " ";
    file << n.parent << " " << std::endl;
  }
}

bool check_edge_at_resolution(
    const double *start, const double *goal, int n,
    std::function<bool(const double *, size_t n)> check_fun,
    double resolution = .2) {

  bool valid_edge = true;
  double diff[n];
  add(goal, start, n, -1., diff);
  double len = std::sqrt(norm_sq(diff, n));
  int num_checks = len / resolution;

  for (size_t i = 1; i < num_checks; i++) {

    double proposal[n];
    add(start, diff, n, resolution * i, proposal);
    if (!check_fun(proposal, n)) {
      valid_edge = false;
      break;
    }
  }
  return valid_edge;
}

void build_heuristic_distance(
    const std::vector<Sample> &batch_samples /* goal should be last */,
    std::function<double(const double *, const double *, size_t)> distance_sq,
    std::function<double(const double *, const double *, size_t)> time,
    std::vector<SampleNode> &heuristic_map, size_t n,
    std::shared_ptr<ompl::control::SpaceInformation> si,
    double distance_threshold_sq,
    std::function<bool(const double *, size_t n)> check_fun,
    double resolution) {

  EdgeList edge_list;
  DistanceList distance_list;

  enum ComputeDist {
    Linear,
    TreeOMPL,
    TreeFlann,
  };

  ComputeDist mode = Linear;

  auto tic = std::chrono::high_resolution_clock::now();

  if (mode == Linear) {
    timed_fun([&] {
      for (size_t i = 0; i < batch_samples.size(); i++) {
        for (size_t j = i + 1; j < batch_samples.size(); j++) {
          auto p1 = batch_samples[i].data();
          auto p2 = batch_samples[j].data();
          double d = distance_sq(p1, p2, n);
          if (d < distance_threshold_sq &&
              check_edge_at_resolution(p1, p2, 3, check_fun, resolution)) {
            edge_list.push_back({i, j});
            double t = time(p1, p2, n);
            // std::cout <<  " t " << t << " d " << std::sqrt(d) << std::endl;
            distance_list.push_back(t);
          }
        }
      }
      return 0;
    });
  } else if (mode == TreeOMPL) {
    throw -1;

    std::cout << si->getStateSpace()->isMetricSpace() << std::endl;

    auto T_heu = std::make_unique<
        ompl::NearestNeighborsGNATNoThreadSafety<HeuNodeWithIndex *>>();
    auto distance_fun = [&](const HeuNodeWithIndex *a,
                            const HeuNodeWithIndex *b) {
      double realsa[3];
      double realsb[3];
      copyToArray(si->getStateSpace(), realsa, a->state);
      copyToArray(si->getStateSpace(), realsb, b->state);

      size_t n = 3;
      return distance_sq(realsa, realsb, n);
    };

    T_heu->setDistanceFunction(distance_fun);

    std::vector<HeuNodeWithIndex *> ptrs;
    for (size_t i = 0; i < batch_samples.size(); i++) {
      // TODO: is this memory leak?, stop bad code
      HeuNodeWithIndex *ptr = new HeuNodeWithIndex;
      ptr->state = _allocAndFillState(si, batch_samples[i]);
      ptr->index = i;
      ptrs.push_back(ptr);
    }

    T_heu->add(ptrs);

#if 1
    edge_list.clear();
    distance_list.clear();
#endif
    for (size_t i = 0; i < batch_samples.size(); i++) {
      auto &d = batch_samples.at(i);
      HeuNodeWithIndex ptr; // memory leak, stop bad code
      ptr.state = _allocAndFillState(si, d);

      std::vector<HeuNodeWithIndex *> out;
      T_heu->nearestR(&ptr, distance_threshold_sq, out);

      for (auto &s : out) {
        if (i < s->index) {
          auto p1 = batch_samples.at(i).data();
          auto p2 = batch_samples.at(s->index).data();
          if (check_edge_at_resolution(p1, p2, 3, check_fun, resolution)) {
            edge_list.push_back({i, s->index});
            double distance = distance_fun(&ptr, s);
            distance_list.push_back(std::sqrt(distance));
          }
        }
      }
    }
  } else if (mode == TreeFlann) {

    throw -1;
    int dim = 3;
    flann::Matrix<double> dataset(new double[batch_samples.size() * 3],
                                  batch_samples.size(), dim);
    for (size_t i = 0; i < batch_samples.size(); i++) {
      std::copy(batch_samples[i].begin(), batch_samples[i].end(), dataset[i]);
    }

    flann::KDTreeSingleIndex<L2Q<double>> index(
        dataset, flann::KDTreeSingleIndexParams(), L2Q<double>(distance_sq));

    index.buildIndex();

    std::vector<std::vector<double>> v_dis;
    std::vector<std::vector<int>> v_idx;

    index.radiusSearch(dataset, v_idx, v_dis, distance_threshold_sq,
                       flann::SearchParams());

    bool verbose = false;

    if (verbose) {
      std::cout << "results" << v_idx.size() << v_dis.size() << std::endl;

      for (size_t i = 0; i < v_idx.size(); i++) {
        std::cout << "query i " << i << std::endl;
        for (size_t j = 0; j < v_idx.at(i).size(); j++) {
          std::cout << "(" << v_idx.at(i).at(j) << " " << v_dis.at(i).at(j)
                    << ")"
                    << " ";
        }
        std::cout << std::endl;
      }
    }

    for (size_t i = 0; i < v_idx.size(); i++) {
      for (size_t j = 0; j < v_idx[i].size(); j++) {
        if (i < v_idx[i][j]) {

          double *p1 = dataset[i];
          double *p2 = dataset[v_idx[i][j]];

          if (check_edge_at_resolution(p1, p2, 3, check_fun, resolution)) {
            edge_list.push_back({i, v_idx[i][j]});
            distance_list.push_back(std::sqrt(v_dis[i][j]));
          }
        }
      }
    };
  } else {
    ERROR_WITH_INFO("unknown mode");
  }

  auto tac = std::chrono::high_resolution_clock::now();
  double total_time =
      std::chrono::duration<double, std::milli>(tac - tic).count();
  std::cout << "time building distance matrix " << total_time << std::endl;

  compute_heuristic_map(edge_list, distance_list, batch_samples, heuristic_map);
}

double euclidean_distance_squared(const double *x, const double *y, size_t n) {
  double out = 0;
  for (size_t i = 0; i < n; i++) {
    double d = x[i] - y[i];
    out += d * d;
  }
  return out;
}

double euclidean_distance_scale_squared(const double *x, const double *y,
                                        const double *s, size_t n) {
  double out = 0;
  for (size_t i = 0; i < n; i++) {
    double d = x[i] - y[i];
    out += s[i] * d * d;
  }
  return out;
}

double euclidean_distance(const double *x, const double *y, size_t n) {
  return std::sqrt(euclidean_distance_squared(x, y, n));
}

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
                  const std::vector<std::vector<double>> &data) {

  for (auto &v : data) {
    for (auto &e : v)
      out << e << " ";
    out << std::endl;
  }
}

double time_fun(const double *x, const double *y, size_t n) {
  if (n != 3) {
    throw std::runtime_error("n should be 3");
  }
  double s[3] = {1., 1., 0.};
  double max_vel = .5;
  double max_ang_vel = .5;
  double time_xy =
      std::sqrt(euclidean_distance_scale_squared(x, y, s, n)) / max_vel;
  double time_theta = std::fabs(distance_angle(x[2], y[2]) / max_ang_vel);
  return std::max(time_xy, time_theta);
};

double time_fun_si(std::shared_ptr<ompl::control::SpaceInformation> si,
                   const ob::State *a, const ob::State *b) {
  std::vector<double> g_std;
  std::vector<double> c_std;

  si->getStateSpace()->copyToReals(g_std, a);
  si->getStateSpace()->copyToReals(c_std, b);

  return time_fun(g_std.data(), c_std.data(), 3);
};

double
heuristicCollisionsTree(ompl::NearestNeighbors<HeuNode *> *T_heu,
                        const ob::State *s,
                        std::shared_ptr<ompl::control::SpaceInformation> si) {
  HeuNode node;
  node.state = s;
  // auto out = T_heu->nearest(&node);
  std::vector<HeuNode *> neighbors;
  double max_distance = 0.6;
  double min = 1e8;
  T_heu->nearestR(&node, max_distance, neighbors);
  for (const auto &p : neighbors) {
    double d = p->dist + time_fun_si(si, p->state, s);
    if (d < min) {
      min = d;
    }
  }
  return min;
}

// I need the same for the goal.
int main(int argc, char *argv[]) {

  auto tic = std::chrono::high_resolution_clock::now();

  double time_nearestMotion = 0.0;
  double time_nearestNode = 0.0;
  double time_nearestNode_add = 0.0;
  double time_nearestNode_search = 0.0;
  double time_collisions = 0.0;

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string motionsFile;
  float delta;
  float epsilon;
  float alpha;
  bool filterDuplicates;
  float maxCost;
  int new_heu;
  size_t max_motions;
  std::string outputFile;
  double resolution;
  double delta_factor_goal;
  bool add_cost_delta;
  int rebuild_every;

  desc.add_options()("help", "produce help message")(
      "input,i", po::value<std::string>(&inputFile)->required(),
      "input file (yaml)")("motions,m",
                           po::value<std::string>(&motionsFile)->required(),
                           "motions file (yaml)")(
      "resolution", po::value<double>(&resolution)->default_value(0.2),
      "resolution when computing the heuristic map")(
      "delta", po::value<float>(&delta)->default_value(0.1),
      "discontinuity bound (negative to auto-compute with given k)")(
      "epsilon", po::value<float>(&epsilon)->default_value(1.0),
      "suboptimality bound")(
      "add_cost_delta", po::value<bool>(&add_cost_delta)->default_value(false))(
      "rebuild_every", po::value<int>(&rebuild_every)->default_value(5000))(
      "heu", po::value<int>(&new_heu)->default_value(0),
      "heuristic {0:euclidean, 1:euclideanOBS, 2:motionOBS}")(
      "rgoal", po::value<double>(&delta_factor_goal)->default_value(1.))(
      "max_motions", po::value<size_t>(&max_motions)->default_value(INT_MAX),
      "")("alpha", po::value<float>(&alpha)->default_value(0.5),
          "alpha")("filterDuplicates",
                   po::value<bool>(&filterDuplicates)->default_value(true),
                   "filter duplicates")(
      "maxCost",
      po::value<float>(&maxCost)->default_value(
          std::numeric_limits<float>::infinity()),
      "cost bound")("output,o", po::value<std::string>(&outputFile)->required(),
                    "output file (yaml)");

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

  // load problem description
  YAML::Node env = YAML::LoadFile(inputFile);

  std::vector<fcl::CollisionObjectf *> obstacles;
  for (const auto &obs : env["environment"]["obstacles"]) {
    if (obs["type"].as<std::string>() == "box") {
      const auto &size = obs["size"];
      std::shared_ptr<fcl::CollisionGeometryf> geom;
      geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0));
      const auto &center = obs["center"];
      auto co = new fcl::CollisionObjectf(geom);
      co->setTranslation(
          fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
      co->computeAABB();
      obstacles.push_back(co);
    } else {
      throw std::runtime_error("Unknown obstacle type!");
    }
  }
  std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env(
      new fcl::DynamicAABBTreeCollisionManagerf());
  bpcm_env->registerObjects(obstacles);
  bpcm_env->setup();

  const auto &robot_node = env["robots"][0];
  auto robotType = robot_node["type"].as<std::string>();
  const auto &env_min = env["environment"]["min"];
  const auto &env_max = env["environment"]["max"];
  const auto &st = robot_node["start"];

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
  auto stateValidityChecker(
      std::make_shared<fclStateValidityChecker>(si, bpcm_env, robot));
  si->setStateValidityChecker(stateValidityChecker);

  // set the state propagator
  std::shared_ptr<oc::StatePropagator> statePropagator(
      new RobotStatePropagator(si, robot));
  si->setStatePropagator(statePropagator);

  si->setup();

  // create and set a start state
  auto startState = allocAndFillState(si, robot_node["start"]);
  // si->freeState(startState);

  // set goal state
  auto goalState = allocAndFillState(si, robot_node["goal"]);
  // si->freeState(goalState);

  // load motion primitives
  // YAML::Node motions_node = YAML::LoadFile(motionsFile);

  // load motions primitives
  std::ifstream is(motionsFile.c_str(), std::ios::in | std::ios::binary);
  // get length of file
  is.seekg(0, is.end);
  int length = is.tellg();
  is.seekg(0, is.beg);
  //
  msgpack::unpacker unpacker;
  unpacker.reserve_buffer(length);
  is.read(unpacker.buffer(), length);
  unpacker.buffer_consumed(length);
  msgpack::object_handle oh;
  unpacker.next(oh);
  msgpack::object msg_obj = oh.get();

  std::vector<Motion> motions;
  size_t num_states = 0;
  size_t num_invalid_states = 0;

  // create a robot with no position bounds
  ob::RealVectorBounds position_bounds_no_bound(env_min.size());
  position_bounds_no_bound.setLow(-1e6);
  position_bounds_no_bound.setHigh(1e6); // std::numeric_limits<double>::max());
  std::shared_ptr<Robot> robot_no_pos_bound =
      create_robot(robotType, position_bounds_no_bound);
  auto si_no_pos_bound = robot_no_pos_bound->getSpaceInformation();
  si_no_pos_bound->setPropagationStepSize(1);
  si_no_pos_bound->setMinMaxControlDuration(1, 1);
  si_no_pos_bound->setStateValidityChecker(stateValidityChecker);
  si_no_pos_bound->setStatePropagator(statePropagator);
  si_no_pos_bound->setup();

  if (msg_obj.type != msgpack::type::ARRAY) {
    throw msgpack::type_error();
  }
  for (size_t i = 0; i < msg_obj.via.array.size; ++i) {
    Motion m;
    // find the states
    auto item = msg_obj.via.array.ptr[i];
    if (item.type != msgpack::type::MAP) {
      throw msgpack::type_error();
    }
    // load the states
    for (size_t j = 0; j < item.via.map.size; ++j) {
      auto key = item.via.map.ptr[j].key.as<std::string>();
      if (key == "states") {
        auto val = item.via.map.ptr[j].val;
        for (size_t k = 0; k < val.via.array.size; ++k) {
          ob::State *state = si->allocState();
          std::vector<double> reals;
          val.via.array.ptr[k].convert(reals);
          si->getStateSpace()->copyFromReals(state, reals);
          m.states.push_back(state);
          if (!si_no_pos_bound->satisfiesBounds(m.states.back())) {
            si_no_pos_bound->enforceBounds(m.states.back());
            ++num_invalid_states;
            // si->printState(m.states.back());
          }
        }
        break;
      }
    }
    num_states += m.states.size();
    // load the actions
    for (size_t j = 0; j < item.via.map.size; ++j) {
      auto key = item.via.map.ptr[j].key.as<std::string>();
      if (key == "actions") {
        auto val = item.via.map.ptr[j].val;
        for (size_t k = 0; k < val.via.array.size; ++k) {
          oc::Control *control = si->allocControl();
          std::vector<double> reals;
          val.via.array.ptr[k].convert(reals);
          for (size_t idx = 0; idx < reals.size(); ++idx) {
            double *address =
                si->getControlSpace()->getValueAddressAtIndex(control, idx);
            if (address) {
              *address = reals[idx];
            }
          }
          m.actions.push_back(control);
        }
        break;
      }
    }
    m.cost = m.actions.size() * robot->dt(); // time in seconds
    m.idx = motions.size();
    // m.name = motion["name"].as<std::string>();

    // generate collision objects and collision manager
    for (const auto &state : m.states) {
      for (size_t part = 0; part < robot->numParts(); ++part) {
        const auto &transform = robot->getTransform(state, part);

        auto co = new fcl::CollisionObjectf(robot->getCollisionGeometry(part));
        co->setTranslation(transform.translation());
        co->setRotation(transform.rotation());
        co->computeAABB();
        m.collision_objects.push_back(co);
      }
    }
    m.collision_manager.reset(
        new ShiftableDynamicAABBTreeCollisionManager<float>());
    m.collision_manager->registerObjects(m.collision_objects);

    m.disabled = false;

    motions.push_back(m);
    if (motions.size() >= max_motions) {
      break;
    }
  }

  std::cout << "Info: " << num_invalid_states << " states are invalid of "
            << num_states << std::endl;
  std::cout << "Info: " << motions.size() << " num motions" << std::endl;

  auto rng = std::default_random_engine{};
  std::shuffle(std::begin(motions), std::end(motions), rng);
  for (size_t idx = 0; idx < motions.size(); ++idx) {
    motions[idx].idx = idx;
  }
  std::uniform_real_distribution<> dis_angle(0, 2 * M_PI);

  // build kd-tree for motion primitives
  ompl::NearestNeighbors<Motion *> *T_m;
  if (si->getStateSpace()->isMetricSpace()) {
    T_m = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
  } else {
    T_m = new ompl::NearestNeighborsSqrtApprox<Motion *>();
  }
  T_m->setDistanceFunction([si, motions](const Motion *a, const Motion *b) {
    return si->distance(a->states[0], b->states[0]);
  });

  std::vector<Motion *> motions_ptr(motions.size());
  std::transform(motions.begin(), motions.end(), motions_ptr.begin(),
                 [](auto &s) { return &s; });

  {
    auto out = timed_fun([&] {
      T_m->add(motions_ptr);
      return 0;
    });
    time_nearestMotion += out.second;
  }

  std::cout << "There are " << motions.size() << " motions!" << std::endl;
  std::cout << "Max cost is " << maxCost << std::endl;

  if (alpha <= 0 || alpha >= 1) {
    std::cerr << "Alpha needs to be between 0 and 1!" << std::endl;
    return 1;
  }

  //////////////////////////
  if (delta < 0) {
    Motion fakeMotion;
    fakeMotion.idx = -1;
    fakeMotion.states.push_back(si->allocState());
    std::vector<Motion *> neighbors_m;
    size_t num_desired_neighbors = (size_t)-delta;
    size_t num_samples = std::min<size_t>(1000, motions.size());

    auto state_sampler = si->allocStateSampler();
    float sum_delta = 0.0;
    for (size_t k = 0; k < num_samples; ++k) {
      do {
        state_sampler->sampleUniform(fakeMotion.states[0]);
      } while (!si->isValid(fakeMotion.states[0]));
      robot->setPosition(fakeMotion.states[0], fcl::Vector3f(0, 0, 0));

      T_m->nearestK(&fakeMotion, num_desired_neighbors + 1, neighbors_m);

      float max_delta = si->distance(fakeMotion.states[0],
                                     neighbors_m.back()->states.front());
      sum_delta += max_delta;
    }
    float adjusted_delta = (sum_delta / num_samples) / alpha;
    std::cout << "Automatically adjusting delta to: " << adjusted_delta
              << std::endl;
    delta = adjusted_delta;
  }
  //////////////////////////

  if (filterDuplicates) {
    size_t num_duplicates = 0;
    Motion fakeMotion;
    fakeMotion.idx = -1;
    fakeMotion.states.push_back(si->allocState());
    std::vector<Motion *> neighbors_m;
    for (const auto &m : motions) {
      if (m.disabled) {
        continue;
      }

      si->copyState(fakeMotion.states[0], m.states[0]);
      T_m->nearestR(&fakeMotion, delta * alpha, neighbors_m);

      for (Motion *nm : neighbors_m) {
        if (nm == &m || nm->disabled) {
          continue;
        }
        float goal_delta = si->distance(m.states.back(), nm->states.back());
        if (goal_delta < delta * (1 - alpha)) {
          nm->disabled = true;
          ++num_duplicates;
        }
      }
    }
    std::cout << "There are " << num_duplicates << " duplicate motions!"
              << std::endl;
  }

  //////////////////////////

  // db-A* search
  open_t open;

  // kd-tree for nodes
  ompl::NearestNeighbors<AStarNode *> *T_n;
  if (si->getStateSpace()->isMetricSpace()) {

    // all default except for rebalancing.
    unsigned int degree = 8;
    unsigned int minDegree = 4;
    unsigned int maxDegree = 12;
    unsigned int maxNumPtsPerLeaf = 50;
    unsigned int removedCacheSize = 500;
    bool rebalancing = true;

    T_n = new ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *>(
        degree, minDegree, maxDegree, maxNumPtsPerLeaf, removedCacheSize,
        rebalancing);

  } else {
    T_n = new ompl::NearestNeighborsSqrtApprox<AStarNode *>();
  }
  T_n->setDistanceFunction([si](const AStarNode *a, const AStarNode *b) {
    return si->distance(a->state, b->state);
  });

  HeuristicMap heuristic_map;
  size_t dim = 3;
  std::vector<Sample> batch_samples;

  if (new_heu == 1 || new_heu == 2) {
    std::vector<double> lb(dim);
    std::vector<double> ub(dim);

    std::default_random_engine re;

    for (size_t i = 0; i < 2; ++i) {
      lb[i] = env_min[i].as<double>();
      ub[i] = env_max[i].as<double>();
    }
    lb[2] = -M_PI;
    ub[2] = M_PI;

    std::vector<double> goal;
    std::vector<double> start;

    for (const auto &value : robot_node["goal"]) {
      goal.push_back(value.as<double>());
    }

    for (const auto &value : robot_node["start"]) {
      start.push_back(value.as<double>());
    }


    ob::State *_allocated_state = _allocAndFillState(si, {0, 0, 0});
    auto allocated_state = _allocated_state->as<ob::SE2StateSpace::StateType>();

    auto check_fun = [&](const double *x, size_t n) {
      // ob::SE2StateSpace::StateType state;
      // ompl::base::ScopedState<ob::SE2StateSpace>  state;
      allocated_state->setX(x[0]);
      allocated_state->setY(x[1]);
      allocated_state->setYaw(x[2]);
      return stateValidityChecker->isValid(allocated_state);
    };

    auto sample_fun = [&](double *x, size_t n) {
      for (size_t i = 0; i < n; i++) {
        std::uniform_real_distribution<double> uniform(lb[i], ub[i]);
        x[i] = uniform(re);
      };
    };

    size_t num_sample_trials = 3000;
    // TODO: make sure that in KINK I have enough
    generate_batch(sample_fun, check_fun, num_sample_trials, dim,
                   batch_samples);

    // add the start and the goal to the batch
    batch_samples.push_back(start);
    batch_samples.push_back(goal);

    if (new_heu == 1) {
      auto fun_dist_sq = [](const double *x, const double *y, size_t n) {
        if (n != 3) {
          throw std::runtime_error("n should be 3");
        }
        double s[3] = {1., 1., .1};
        return distance_squared_se2(x, y, s);
      };

      // TODO: choose one!
      double distance_threshold_sq = .45 * .45;
      distance_threshold_sq = .7 * .7;
      // distance_threshold_sq = 1.5 * 1.5;

      build_heuristic_distance(batch_samples /* goal should be last */,
                               fun_dist_sq, time_fun, heuristic_map, dim, si,
                               distance_threshold_sq, check_fun, resolution);

      // for (auto &dd : heuristic_map) {
      //   dd.dist *= 2;
      //   // because MAX VEL is 0.5
      // }

    } else if (new_heu == 2) {
      double backward_delta = std::min(2. * delta, .45);
      std::cout << "backward delta " << backward_delta << std::endl;
      build_heuristic_motions(batch_samples /* goal should be last */,
                              heuristic_map, motions, bpcm_env,
                              backward_delta * backward_delta);
    }

    write_heuristic_map(heuristic_map);
  }

  ompl::NearestNeighbors<HeuNode *> *T_heu;
  if (si->getStateSpace()->isMetricSpace()) {
    T_heu = new ompl::NearestNeighborsGNATNoThreadSafety<HeuNode *>();
  } else {
    T_heu = new ompl::NearestNeighborsSqrtApprox<HeuNode *>();
  }

  T_heu->setDistanceFunction([&](const HeuNode *a, const HeuNode *b) {
    return si->distance(a->state, b->state);
  });

  // TODO: check if it is necessary to rebalance afterwards
  for (size_t i = 0; i < heuristic_map.size(); i++) {
    HeuNode *ptr = new HeuNode; // memory leak, stop bad code
    ptr->state = _allocAndFillState(si, heuristic_map[i].x);
    ptr->dist = heuristic_map[i].dist;
    T_heu->add(ptr);
  }

  auto start_node = new AStarNode();
  start_node->state = startState;
  start_node->gScore = 0;

  if (new_heu == 0) {
    start_node->fScore = epsilon * heuristic(robot, startState, goalState);
    start_node->hScore = start_node->fScore;
  } else if (new_heu == 1 || new_heu == 2) {
    start_node->fScore =
        epsilon * heuristicCollisionsTree(T_heu, startState, si);
    start_node->hScore = start_node->fScore;
    std::cout << "goal heuristic "
              << epsilon * heuristicCollisionsTree(T_heu, goalState, si)
              << std::endl;
  }

  std::cout << "start_node heuristic " << start_node->fScore << std::endl;
  start_node->came_from = nullptr;
  start_node->used_offset = fcl::Vector3f(0, 0, 0);
  start_node->used_motion = -1;

  auto handle = open.push(start_node);
  start_node->handle = handle;
  start_node->is_in_open = true;

  // create a list of nodes that I check and introduce in tree -- debugging
  std::vector<std::vector<double>> _states_debug;
  {
    std::vector<double> _state;
    si->getStateSpace()->copyToReals(_state, start_node->state);
    _states_debug.push_back(_state);

    auto out = timed_fun([&] {
      T_n->add(start_node);
      return 0;
    });
    time_nearestNode += out.second;
    time_nearestNode_add += out.second;
  }

  Motion fakeMotion;
  fakeMotion.idx = -1;
  fakeMotion.states.push_back(si->allocState());

  AStarNode *query_n = new AStarNode();

  ob::State *tmpStateq = si->allocState();
  ob::State *tmpState = si->allocState();
  ob::State *tmpState2 = si->allocState();
  std::vector<Motion *> neighbors_m;
  std::vector<AStarNode *> neighbors_n;

  float last_f_score = start_node->fScore;
  size_t expands = 0;
  // clock start

  auto tac = std::chrono::high_resolution_clock::now();
  double prepare_time =
      std::chrono::duration<double, std::milli>(tac - tic).count();
  double total_time;

  // int rebuild_every = 100;
  // time_nearestNode: 1536.86
  // time_nearestNode_add: 1128.68
  // time_nearestNode_search: 408.176

  // 1000
  // time_nearestNode: 521.063
  // time_nearestNode_add: 125.819
  // time_nearestNode_search: 395.244
  //
  // 10000
  // time_nearestNode: 638.08
  // time_nearestNode_add: 18.8478
  // time_nearestNode_search: 619.232
  //
  //
  // int rebuild_every = 100000;
  // time_nearestNode: 904.296
  // time_nearestNode_add: 11.269
  // time_nearestNode_search: 893.027

  static_cast<ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(T_n)
      ->rebuildSize_ = 1e8;

  while (!open.empty()) {
    AStarNode *current = open.top();

    ++expands;
    if (expands % 1000 == 0 || expands == 1) {
      std::cout << "expanded: " << expands << " open: " << open.size()
                << " nodes: " << T_n->size() << " f-score " << current->fScore
                << " h-score " << current->hScore << " g-score "
                << current->gScore << std::endl;
    }

    if (expands % rebuild_every == 0) {

      auto out = timed_fun([&] {
        static_cast<ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(
            T_n)
            ->rebuildDataStructure();
        static_cast<ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(
            T_n)
            ->rebuildSize_ = 1e8;
        return 0;
      });
      time_nearestNode += out.second;
      time_nearestNode_add += out.second;
    }

    if (new_heu == 0)
      assert(current->fScore >= last_f_score);
    last_f_score = current->fScore;
    // std::cout << "current";
    // si->printState(current->state);
    if (si->distance(current->state, goalState) <= delta_factor_goal * delta) {

      double extra_time = time_fun_si(si, current->state, goalState);
      std::cout << "extra time " << extra_time << std::endl;

      std::cout << "SOLUTION FOUND!!!! cost: " << current->gScore << std::endl;
      std::cout << "SOLUTION FOUND!!!! with cost-to-goal: "
                << current->gScore + extra_time << std::endl;
      auto tac = std::chrono::high_resolution_clock::now();
      total_time = std::chrono::duration<double, std::milli>(tac - tic).count();

      std::vector<const AStarNode *> result;

      const AStarNode *n = current;
      while (n != nullptr) {
        result.push_back(n);
        // std::cout << n->used_motion << std::endl;
        // si->printState(n->state);
        n = n->came_from;
      }
      std::reverse(result.begin(), result.end());

      std::cout << "result size " << result.size() << std::endl;
      std::ofstream out_states("state_out.txt");
      print_matrix(out_states, _states_debug);

      std::cout << "writing output to " << outputFile << std::endl;
      std::ofstream out(outputFile);
      out << "delta: " << delta << std::endl;
      out << "epsilon: " << epsilon << std::endl;
      out << "cost: " << current->gScore << std::endl;
      out << "time: " << total_time << std::endl;
      out << "time_collisions: " << time_collisions << std::endl;
      out << "time_prepare: " << prepare_time << std::endl;
      out << "time_nearestMotion: " << time_nearestMotion << std::endl;
      out << "time_nearestNode: " << time_nearestNode << std::endl;
      out << "time_nearestNode_add: " << time_nearestNode_add << std::endl;
      out << "time_nearestNode_search: " << time_nearestNode_search
          << std::endl;
      out << "expands: " << expands << std::endl;
      out << "result:" << std::endl;
      out << "  - states:" << std::endl;
      double total_time = 0;

      si->copyState(tmpStateq, startState);

      double time_jumps = 0;

      for (size_t i = 0; i < result.size() - 1; ++i) {
        // Compute intermediate states
        const auto node_state = result[i]->state;
        const fcl::Vector3f current_pos =
            robot->getTransform(node_state).translation();
        const auto &motion = motions.at(result[i + 1]->used_motion);
        out << "      # ";
        printState(out, si, node_state);
        out << std::endl;
        out << "      # motion " << motion.idx << " with cost " << motion.cost
            << std::endl;
        // skip last state each

        for (size_t k = 0; k < motion.states.size(); ++k) {
          const auto state = motion.states[k];
          si->copyState(tmpState, state);
          const fcl::Vector3f relative_pos =
              robot->getTransform(state).translation();
          robot->setPosition(tmpState, current_pos +
                                           result[i + 1]->used_offset +
                                           relative_pos);

          if (k < motion.states.size() - 1) {
            if (k == 0) {
              out << "      # jump from ";
              printState(out, si, tmpStateq);
              out << " to ";
              printState(out, si, tmpState);
              out << " delta " << si->distance(tmpStateq, tmpState);
              double min_time = time_fun_si(si, tmpStateq, tmpState);
              time_jumps += min_time;
              out << " min time " << min_time;
              out << std::endl;
            }

            out << "      - ";

          } else {
            out << "      # ";
            si->copyState(tmpStateq, tmpState);
          }
          printState(out, si, tmpState);
          out << std::endl;
        }
        out << std::endl;
      }
      out << "      - ";

      std::cout << " time jumps " << time_jumps << std::endl;

      std::cout << "time jumps + extra_time " << time_jumps + extra_time
                << std::endl;

      // printing the last state
      printState(out, si, result.back()->state);
      out << std::endl;
      out << "    actions:" << std::endl;

      int action_counter = 0;
      for (size_t i = 0; i < result.size() - 1; ++i) {
        const auto &motion = motions[result[i + 1]->used_motion];
        out << "      # motion " << motion.idx << " with cost " << motion.cost
            << std::endl;
        for (size_t k = 0; k < motion.actions.size(); ++k) {
          const auto &action = motion.actions[k];
          out << "      - ";
          action_counter += 1;
          printAction(out, si, action);
          out << std::endl;
        }
        out << std::endl;
      }
      std::cout << "action counter " << action_counter << std::endl;
      // statistics for the motions used
      std::map<size_t, size_t> motionsCount; // motionId -> usage count
      for (size_t i = 0; i < result.size() - 1; ++i) {
        auto motionId = result[i + 1]->used_motion;
        auto iter = motionsCount.find(motionId);
        if (iter == motionsCount.end()) {
          motionsCount[motionId] = 1;
        } else {
          iter->second += 1;
        }
      }
      out << "    motion_stats:" << std::endl;
      for (const auto &kv : motionsCount) {
        out << "      " << motions[kv.first].idx << ": " << kv.second
            << std::endl;
      }

      // statistics on where the motion splits are
      out << "    splits:" << std::endl;
      for (size_t i = 0; i < result.size() - 1; ++i) {
        const auto &motion = motions.at(result[i + 1]->used_motion);
        out << "      - " << motion.states.size() - 1 << std::endl;
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
      //         std::cout << "error?" << dist << " " << entry << " " <<
      //         nbhs.back() << std::endl;
      //       }
      //     }
      //   }
      // }

      return 0;
      break;
    }

    current->is_in_open = false;
    open.pop();

    // std::cout << "top " << std::endl;
    // si->printState(current->state);

    // find relevant motions (within delta/2 of current state)
    si->copyState(fakeMotion.states[0], current->state);
    robot->setPosition(fakeMotion.states[0], fcl::Vector3f(0, 0, 0));

    {
      auto out = timed_fun([&] {
        T_m->nearestR(&fakeMotion, delta * alpha, neighbors_m);
        return 0;
      });
      time_nearestMotion += out.second;
    }
    // std::shuffle(std::begin(neighbors_m), std::end(neighbors_m), rng);

    // std::cout << "found " << neighbors_m.size() << " motions" << std::endl;
    // Loop over all potential applicable motions
    for (const Motion *motion : neighbors_m) {
      if (motion->disabled) {
        continue;
      }
      // si->printState(motion->states.front());

#if 1
      fcl::Vector3f computed_offset(0, 0, 0);
#else
      float motion_dist = si->distance(fakeMotion.states[0], motion->states[0]);
      float translation_slack = delta / 2 - motion_dist;
      assert(translation_slack >= 0);

      // ideally, solve the following optimization problem
      // min_translation fScore
      //     s.t. ||translation|| <= translation_slack // i.e., stay within
      //     delta/2
      //          no collisions

      const auto current_pos2 =
          robot->getTransform(current->state).translation();
      const auto goal_pos = robot->getTransform(goalState).translation();
      fcl::Vector3f computed_offset =
          (goal_pos - current_pos2).normalized() * translation_slack;

      // std::uniform_real_distribution<> dis_mag(0, translation_slack);
      // float angle = dis_angle(rng);
      // float mag = dis_mag(rng);
      // fcl::Vector3f computed_offset(mag * cos(angle), mag * sin(angle), 0);

#ifndef NDEBUG
      {
        // check that the computed starting state stays within delta/2
        si->copyState(tmpState, motion->states.front());
        const auto current_pos =
            robot->getTransform(current->state).translation();
        const auto offset = current_pos + computed_offset;
        const auto relative_pos = robot->getTransform(tmpState).translation();
        robot->setPosition(tmpState, offset + relative_pos);
        std::cout << si->distance(tmpState, current->state) << std::endl;
        assert(si->distance(tmpState, current->state) <= delta / 2 + 1e-5);
      }
#endif
#endif

      // compute estimated cost

      // tmpState + last_state

      float tentative_gScore = current->gScore + motion->cost;
      // compute final state
      si->copyState(tmpState, motion->states.back());
      const auto current_pos =
          robot->getTransform(current->state).translation();
      const auto offset = current_pos + computed_offset;
      const auto relative_pos = robot->getTransform(tmpState).translation();
      robot->setPosition(tmpState, offset + relative_pos);
      // compute estimated fscore

      // compute the delta.
      si->copyState(tmpState2, motion->states.front());
      robot->setPosition(tmpState2, current_pos);
      double delta_ = si->distance(tmpState2, current->state);
      // std::cout << "delta is " << delta_ << std::endl;
      double max_ang_vel = .5;
      double extra_time = delta_ / max_ang_vel;
      tentative_gScore += add_cost_delta * extra_time;

      float tentative_hScore;
      if (new_heu == 2 || new_heu == 1)
        tentative_hScore =
            epsilon * heuristicCollisionsTree(T_heu, tmpState, si);
      else
        tentative_hScore = epsilon * heuristic(robot, tmpState, goalState);

      float tentative_fScore = tentative_gScore + tentative_hScore;

      // skip motions that would exceed cost bound
      if (tentative_fScore > maxCost) {
        // std::cout << "skip b/c cost " << tentative_fScore << " " <<
        // tentative_gScore << " " << tentative_hScore << " " << maxCost <<
        // std::endl;
        continue;
      }
      // skip motions that are invalid
      if (!si->satisfiesBounds(tmpState)) {
        // std::cout << "skip invalid state" << std::endl;
        // si->printState(tmpState);
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

      bool motionValid;
      {
        auto out = timed_fun([&] {
          motion->collision_manager->shift(offset);
          fcl::DefaultCollisionData<float> collision_data;
          motion->collision_manager->collide(
              bpcm_env.get(), &collision_data,
              fcl::DefaultCollisionFunction<float>);
          bool motionValid = !collision_data.result.isCollision();
          motion->collision_manager->shift(-offset);
          return motionValid;
        });

        motionValid = out.first;
        time_collisions += out.second;
      }

      // for (auto obj : motion->collision_objects) {
      //   obj->setTranslation(obj->getTranslation() + offset);
      // }
      // motion->collision_manager->update(motion->collision_objects);

      // fcl::DefaultCollisionData<float> collision_data;
      // motion->collision_manager->collide(bpcm_env.get(), &collision_data,
      // fcl::DefaultCollisionFunction<float>); bool motionValid =
      // !collision_data.result.isCollision();

      // for (auto obj : motion->collision_objects) {
      //   obj->setTranslation(obj->getTranslation() - offset);
      // }

#endif
      // auto end = std::chrono::steady_clock::now();
      // size_t dt = std::chrono::duration_cast<std::chrono::nanoseconds>(end
      // - start).count(); std::cout << "cc: " << dt << " ns\n";

      // Skip this motion, if it isn't valid
      if (!motionValid) {
        // std::cout << "skip invalid motion" << std::endl;
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
      float radius = delta * (1 - alpha);

      std::vector<double> _state;
      si->getStateSpace()->copyToReals(_state, query_n->state);
      _states_debug.push_back(_state);

      {
        auto out = timed_fun([&] {
          T_n->nearestR(query_n, radius, neighbors_n);
          return 0;
        });
        time_nearestNode += out.second;
        time_nearestNode_search += out.second;
      }

      // auto nearest = T_n->nearest(query_n);
      // float nearest_distance = si->distance(nearest->state, tmpState);

      // exclude state we came from (otherwise we never add motions that are
      // less than delta away) auto it = std::remove(neighbors_n.begin(),
      // neighbors_n.end(), current); neighbors_n.erase(it,
      // neighbors_n.end());

      // std::cout << neighbors_n.size() << std::endl;

      if (neighbors_n.size() == 0)
      // if (nearest_distance > radius)
      {
        // new state -> add it to open and T_n
        auto node = new AStarNode();
        node->state = si->cloneState(tmpState);
        node->gScore = tentative_gScore;
        node->fScore = tentative_fScore;
        node->hScore = tentative_hScore;
        node->came_from = current;
        node->used_motion = motion->idx;
        node->used_offset = computed_offset;
        node->is_in_open = true;
        auto handle = open.push(node);
        node->handle = handle;

        {

          std::vector<double> _state;
          si->getStateSpace()->copyToReals(_state, node->state);
          _states_debug.push_back(_state);

          auto out = timed_fun([&] {
            T_n->add(node);
            return 0;
          });
          time_nearestNode += out.second;
          time_nearestNode_add += out.second;
        }
      } else {
        // T_n->nearestR(query_n, radius, neighbors_n);
        // check if we have a better path now
        for (AStarNode *entry : neighbors_n) {
          // AStarNode* entry = nearest;
          assert(si->distance(entry->state, tmpState) <= delta);
          double time_to_reach = time_fun_si(si, entry->state, tmpState);
          // std::cout  << "time to reach " << time_to_reach << std::endl;
          double tentative_gScore_ = tentative_gScore + time_to_reach;
          float delta_score = entry->gScore - (tentative_gScore_);

          if (delta_score > 0) {
            entry->gScore = tentative_gScore_;
            entry->fScore -= delta_score;
            entry->hScore = tentative_gScore_;
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
  // clock end
  tac = std::chrono::high_resolution_clock::now();
  total_time = std::chrono::duration<double, std::milli>(tac - tic).count();

  query_n->state = goalState;
  const auto nearest = T_n->nearest(query_n);
  if (nearest->gScore == 0) {
    std::cout << "No solution found (not even approxmite)" << std::endl;
    return 1;
  }

  float nearest_distance = si->distance(nearest->state, goalState);
  std::cout << "Nearest to goal: " << nearest_distance << " (delta: " << delta
            << ")" << std::endl;

  std::cout << "Using approximate solution cost: " << nearest->gScore
            << std::endl;

  std::vector<const AStarNode *> result;

  const AStarNode *n = nearest;
  while (n != nullptr) {
    result.push_back(n);
    // std::cout << n->used_motion << std::endl;
    // si->printState(n->state);
    n = n->came_from;
  }
  std::reverse(result.begin(), result.end());

  // TODO: We are copying the write output twice. Fix
  std::cout << "writing output to " << outputFile << std::endl;
  std::ofstream out(outputFile);
  out << "delta: " << delta << std::endl;
  out << "epsilon: " << epsilon << std::endl;
  out << "cost: " << nearest->gScore << std::endl;
  out << "time: " << total_time << std::endl;
  out << "time_collisions: " << time_collisions << std::endl;
  out << "time_nearestMotion: " << time_nearestMotion << std::endl;
  out << "time_nearestNode: " << time_nearestNode << std::endl;
  out << "time_nearestNode_add: " << time_nearestNode_add << std::endl;
  out << "time_nearestNode_search: " << time_nearestNode_search << std::endl;
  out << "expands: " << expands << std::endl;
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (size_t i = 0; i < result.size() - 1; ++i) {
    // Compute intermediate states
    const auto node_state = result[i]->state;
    const fcl::Vector3f current_pos =
        robot->getTransform(node_state).translation();
    const auto &motion = motions.at(result[i + 1]->used_motion);
    out << "      # ";
    printState(out, si, node_state);
    out << std::endl;
    out << "      # motion " << motion.idx << " with cost " << motion.cost
        << std::endl;
    // skip last state each
    for (size_t k = 0; k < motion.states.size(); ++k) {
      const auto state = motion.states[k];
      si->copyState(tmpState, state);
      const fcl::Vector3f relative_pos =
          robot->getTransform(state).translation();
      robot->setPosition(tmpState, current_pos + result[i + 1]->used_offset +
                                       relative_pos);

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
  for (size_t i = 0; i < result.size() - 1; ++i) {
    const auto &motion = motions[result[i + 1]->used_motion];
    out << "      # motion " << motion.idx << " with cost " << motion.cost
        << std::endl;
    for (size_t k = 0; k < motion.actions.size(); ++k) {
      const auto &action = motion.actions[k];
      out << "      - ";
      printAction(out, si, action);
      out << std::endl;
    }
    out << std::endl;
  }
  // statistics for the motions used
  std::map<size_t, size_t> motionsCount; // motionId -> usage count
  for (size_t i = 0; i < result.size() - 1; ++i) {
    auto motionId = result[i + 1]->used_motion;
    auto iter = motionsCount.find(motionId);
    if (iter == motionsCount.end()) {
      motionsCount[motionId] = 1;
    } else {
      iter->second += 1;
    }
  }
  out << "    motion_stats:" << std::endl;
  for (const auto &kv : motionsCount) {
    out << "      " << motions[kv.first].idx << ": " << kv.second << std::endl;
  }

  return 0;
}
