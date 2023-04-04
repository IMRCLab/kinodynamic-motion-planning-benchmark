#include "dbastar.hpp"

#include <boost/graph/graphviz.hpp>

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

#include "motions.hpp"
#include "ompl/base/ScopedState.h"
#include "robot_models.hpp"
#include "robots.h"

// boost stuff for the graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/property_map/property_map.hpp>

#include "general_utils.hpp"

namespace ob = ompl::base;
namespace oc = ompl::control;
namespace po = boost::program_options;

using Sample = std::vector<double>;
using Sample_ = ob::State;

const char *duplicate_detection_str[] = {"NO", "HARD", "SOFT"};

const char *terminate_status_str[] = {
    "SOLVED", "MAX_EXPANDS", "EMPTY_QUEUE", "MAX_TIME", "UNKNOWN",
};

Heu_roadmap::Heu_roadmap(std::shared_ptr<RobotOmpl> robot,
                         size_t num_sample_trials, ob::State *startState,
                         ob::State *goalState, double resolution)
    : robot(robot) {

  auto si = robot->getSpaceInformation();
  auto &stateValidityChecker = si->getStateValidityChecker();
  size_t dim = robot->diff_model->nx;

  auto check_fun = [&](Sample_ *x) { return stateValidityChecker->isValid(x); };

  auto sample_fun = [&](double *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
      Eigen::Map<Eigen::VectorXd> s(x, n);
      robot->diff_model->sample_uniform(s);
    };
  };

  generate_batch(sample_fun, check_fun, num_sample_trials, dim, si,
                 batch_samples);
  batch_samples.push_back(startState);
  batch_samples.push_back(goalState);

  {
    std::ofstream file_out("batch_out.txt");
    for (auto &x : batch_samples) {
      printState(file_out, si, x);
      file_out << std::endl;
    }
  }

  double distance_threshold = 1.;
  build_heuristic_distance(batch_samples /* goal should be last */, robot,
                           heuristic_map, distance_threshold, resolution);

  write_heuristic_map(heuristic_map, si);

  if (si->getStateSpace()->isMetricSpace()) {
    T_heu =
        std::make_shared<ompl::NearestNeighborsGNATNoThreadSafety<HeuNode *>>();
  } else {
    T_heu = std::make_shared<ompl::NearestNeighborsSqrtApprox<HeuNode *>>();
  }

  T_heu->setDistanceFunction([this](const HeuNode *a, const HeuNode *b) {
    return this->robot->getSpaceInformation()->distance(a->state, b->state);
  });

  for (size_t i = 0; i < heuristic_map.size(); i++) {
    HeuNode *ptr = new HeuNode; // memory leak, stop bad code
    ptr->state = heuristic_map[i].x;
    ptr->dist = heuristic_map[i].dist;
    T_heu->add(ptr);
  }
}

void copyToArray(const ompl::base::StateSpacePtr &space, double *reals,
                 const ompl::base::State *source) {
  const auto &locations = space->getValueLocations();
  for (std::size_t i = 0; i < locations.size(); ++i)
    reals[i] = *space->getValueAddressAtLocation(source, locations[i]);
}

double _distance_angle(double a, double b) {
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
  double d2 = s[2] * _distance_angle(x[2], y[2]);
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

std::ostream &printState(std::ostream &stream, const std::vector<double> &x) {
  // continue here
  stream << "[";
  for (size_t d = 0; d < x.size(); ++d) {
    stream << x[d];
    if (d < x.size() - 1) {
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}

std::ostream &printState(std::ostream &stream,
                         std::shared_ptr<ompl::control::SpaceInformation> si,
                         const ob::State *state, bool add_brackets_comma) {
  std::vector<double> reals;
  si->getStateSpace()->copyToReals(reals, state);
  if (add_brackets_comma)
    stream << "[";
  for (size_t d = 0; d < reals.size(); ++d) {
    stream << reals[d];
    if (d < reals.size() - 1) {
      if (add_brackets_comma)
        stream << ",";
      else
        stream << " ";
    }
  }
  if (add_brackets_comma)
    stream << "]";
  return stream;
}

std::ostream &printAction(std::ostream &stream,
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
void backward_tree_with_dynamics(const std::vector<std::vector<double>> &data,
                                 std::vector<Motion> &primitives,
                                 std::vector<Edge> &edge_list,
                                 std::vector<double> &distance_list,
                                 std::shared_ptr<RobotOmpl> robot,
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

  auto tic = std::chrono::high_resolution_clock::now();
  index.buildIndex();
  auto tac = std::chrono::high_resolution_clock::now();
  std::cout << "build index for heuristic "
            << std::chrono::duration<double, std::milli>(tac - tic).count()
            << std::endl;

  int num_nn_search = 0;

  auto is_applicable = [&](auto &primitive, auto &point) {
    ob::State *last = primitive.states.back();
    auto lastTyped = last->as<ob::SE2StateSpace::StateType>();
    double d = _distance_angle(point[2], lastTyped->getYaw());
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
    num_nn_search++;
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

  std::cout << "data SIZE " << data.size() << std::endl;
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
          fcl::Vector3d offset(point_out.at(0), point_out.at(1), 0);
          if (debug)
            file << "i " << i << " m " << j << " "
                 << "offset " << offset(0) << offset(1) << offset(2) << " ";
          auto out = timed_fun([&] {
            motion->collision_manager->shift(offset);
            fcl::DefaultCollisionData<double> collision_data;
            motion->collision_manager->collide(
                robot->diff_model->env.get(), &collision_data,
                fcl::DefaultCollisionFunction<double>);
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
  std::cout << "we used nn searches " << num_nn_search << std::endl;

  // it seems I am doing too nn: 2293399
}

// std::vector<double>;
// std::vector<double>;

void generate_batch(std::function<void(double *, size_t)> free_sampler,
                    std::function<bool(Sample_ *)> checker,
                    size_t num_samples_trials, size_t dim,
                    std::shared_ptr<ompl::control::SpaceInformation> si,
                    std::vector<Sample_ *> &out) {

  for (size_t i = 0; i < num_samples_trials; i++) {
    std::vector<double> x(dim);
    free_sampler(x.data(), dim);
    Sample_ *allocated_state = _allocAndFillState(si, x);
    if (checker(allocated_state)) {
      out.push_back(allocated_state);
    } else {
      si->freeState(allocated_state);
    }
  }

  std::cout << "Sampling DONE.  " << out.size() << " samples" << std::endl;
}

using HeuristicMap = std::vector<SampleNode>;

void compute_heuristic_map(const EdgeList &edge_list,
                           const DistanceList &distance_list,
                           const std::vector<Sample_ *> &batch_samples,
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
    const char *filename) {

  std::ofstream file(filename);
  for (auto &n : heuristic_map) {
    printState(file, si, n.x, false);
    file << " ";
    file << n.dist << " ";
    file << n.parent << " " << std::endl;
  }
}

// bool check_edge_at_resolution(
//     const double *start, const double *goal, int n,
//     std::function<bool(const double *, size_t n)> check_fun,
//     double resolution = .2) {

bool check_edge_at_resolution(const Sample_ *start, const Sample_ *goal,
                              std::shared_ptr<RobotOmpl> robot,
                              double resolution) {

  bool valid_edge = true;

  auto state = robot->getSpaceInformation()->allocState();

  double distance = robot->getSpaceInformation()->distance(start, goal);
  size_t ratio = distance / resolution;

  // distance = 3.3, resolution =  1.
  // ratio = 3
  // I want to check
  // 0. 1. 2. 3. 3.3
  // 0 and 3.3 are already checked.
  // I have to do 3 checks.
  // at 1. , 2. 3,.
  // Input has to be between 0 and 1.
  // 1 / 3.3 , 2 / 3.3 , 3 / 3.3
  for (size_t i = 0; i < ratio; i++) {
    double t = (i + 1) * resolution / distance;
    robot->geometric_interpolation(start, goal, t, state);
    if (!robot->getSpaceInformation()->isValid(state)) {
      valid_edge = false;
      break;
    }
  }
  robot->getSpaceInformation()->freeState(state);
  return valid_edge;
}

void build_heuristic_distance(const std::vector<Sample_ *> &batch_samples,
                              std::shared_ptr<RobotOmpl> robot,
                              std::vector<SampleNode> &heuristic_map,
                              double distance_threshold, double resolution) {

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
        std::cout << "i " << i << std::endl;
        for (size_t j = i + 1; j < batch_samples.size(); j++) {
          auto p1 = batch_samples.at(i);
          auto p2 = batch_samples.at(j);
          double d = robot->getSpaceInformation()->distance(p1, p2);
          if (d < distance_threshold &&
              check_edge_at_resolution(p1, p2, robot, resolution)) {
            edge_list.push_back({i, j});
            distance_list.push_back(robot->cost_lower_bound(p1, p2));
          }
        }
      }
      return 0;
    });
  } else if (mode == TreeOMPL) {

    std::cout << robot->getSpaceInformation()->getStateSpace()->isMetricSpace()
              << std::endl;

    auto T_heu = std::make_unique<
        ompl::NearestNeighborsGNATNoThreadSafety<HeuNodeWithIndex *>>();

    auto distance_fun = [&](const HeuNodeWithIndex *a,
                            const HeuNodeWithIndex *b) {
      return robot->getSpaceInformation()->distance(a->state, b->state);
    };

    T_heu->setDistanceFunction(distance_fun);

    std::vector<HeuNodeWithIndex *> ptrs;
    for (size_t i = 0; i < batch_samples.size(); i++) {
      // TODO: is this memory leak?, stop bad code
      HeuNodeWithIndex *ptr = new HeuNodeWithIndex;
      ptr->state = batch_samples.at(i);
      ptr->index = i;
      ptrs.push_back(ptr);
    }

    T_heu->add(ptrs);

    edge_list.clear();
    distance_list.clear();
    for (size_t i = 0; i < batch_samples.size(); i++) {
      auto &d = batch_samples.at(i);
      HeuNodeWithIndex ptr; // memory leak, stop bad code
      ptr.state = d;
      std::vector<HeuNodeWithIndex *> out;
      T_heu->nearestR(&ptr, distance_threshold, out);
      for (auto &s : out) {
        if (i < static_cast<size_t>(s->index)) {
          auto p1 = batch_samples.at(i);
          auto p2 = batch_samples.at(s->index);
          if (check_edge_at_resolution(p1, p2, robot, resolution)) {
            edge_list.push_back({i, s->index});
            distance_list.push_back(robot->cost_lower_bound(p1, p2));
          }
        }
      }
    }
  }
#if 0
  else if (mode == TreeFlann) {

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
  }
#endif
  else {

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
                  const std::vector<std::vector<double>> &data) {

  for (auto &v : data) {
    for (auto &e : v)
      out << e << " ";
    out << std::endl;
  }
}

double heuristicCollisionsTree(ompl::NearestNeighbors<HeuNode *> *T_heu,
                               const ob::State *s,
                               std::shared_ptr<RobotOmpl> robot) {
  HeuNode node;
  node.state = s;
  // auto out = T_heu->nearest(&node);
  std::vector<HeuNode *> neighbors;
  double max_distance = 0.5;
  double min = 1e8;
  CHECK(T_heu, AT);
  T_heu->nearestR(&node, max_distance, neighbors);
  for (const auto &p : neighbors) {
    double d = p->dist + robot->cost_lower_bound(p->state, s);
    if (d < min) {
      min = d;
    }
  }
  return min;
}

// 10 cm -> 0.1
//
// 10 degree - 0.174444 radians  ->  0.09
//
// I want 10 cm to be equivalent to 45 cm -> factor of 4 extra

// lets try to change the distance metric on SO2.

// I need the same for the goal.

#define NAMEOF(variable) #variable
#define VAR_WITH_NAME(variable) variable, #variable

struct Loader {

  bool use_boost = true; // boost (true) or yaml (false)
  void *source = nullptr;

  template <typename T> void set(T &x, const char *name) {

    if (use_boost) {
      po::options_description *p =
          static_cast<po::options_description *>(source);
      CHECK(p, AT);
      set_from_boostop(*p, x, name);
    } else {
      YAML::Node *p = static_cast<YAML::Node *>(source);
      CHECK(p, AT);
      set_from_yaml(*p, x, name);
    }
  }
};

// TODO: how to unify: write form boost and write from yaml?
void Options_dbastar::__load_data(void *source, bool boost) {

  Loader loader;
  loader.use_boost = boost;
  loader.source = source;

  loader.set(VAR_WITH_NAME(delta));
  loader.set(VAR_WITH_NAME(epsilon));
  loader.set(VAR_WITH_NAME(alpha));
  loader.set(VAR_WITH_NAME(filterDuplicates));
  loader.set(VAR_WITH_NAME(maxCost));
  loader.set(VAR_WITH_NAME(heuristic));
  loader.set(VAR_WITH_NAME(max_motions));
  loader.set(VAR_WITH_NAME(resolution));
  loader.set(VAR_WITH_NAME(delta_factor_goal));
  loader.set(VAR_WITH_NAME(cost_delta_factor));
  loader.set(VAR_WITH_NAME(rebuild_every));
  loader.set(VAR_WITH_NAME(num_sample_trials));
  loader.set(VAR_WITH_NAME(max_expands));
  loader.set(VAR_WITH_NAME(cut_actions));
  loader.set(VAR_WITH_NAME(duplicate_detection_int));
  loader.set(VAR_WITH_NAME(use_landmarks));
  loader.set(VAR_WITH_NAME(factor_duplicate_detection));
  loader.set(VAR_WITH_NAME(epsilon_soft_duplicate));
  loader.set(VAR_WITH_NAME(add_node_if_better));
  loader.set(VAR_WITH_NAME(debug));
  loader.set(VAR_WITH_NAME(add_after_expand));
  loader.set(VAR_WITH_NAME(motionsFile));
  loader.set(VAR_WITH_NAME(outFile));
  loader.set(VAR_WITH_NAME(search_timelimit));
}

void Options_dbastar::add_options(po::options_description &desc) {
  __load_data(&desc, true);
#if 0 
  set_from_boostop(desc, VAR_WITH_NAME(delta));
  set_from_boostop(desc, VAR_WITH_NAME(epsilon));
  set_from_boostop(desc, VAR_WITH_NAME(alpha));
  set_from_boostop(desc, VAR_WITH_NAME(filterDuplicates));
  set_from_boostop(desc, VAR_WITH_NAME(maxCost));
  set_from_boostop(desc, VAR_WITH_NAME(heuristic));
  set_from_boostop(desc, VAR_WITH_NAME(max_motions));
  set_from_boostop(desc, VAR_WITH_NAME(resolution));
  set_from_boostop(desc, VAR_WITH_NAME(delta_factor_goal));
  set_from_boostop(desc, VAR_WITH_NAME(cost_delta_factor));
  set_from_boostop(desc, VAR_WITH_NAME(rebuild_every));
  set_from_boostop(desc, VAR_WITH_NAME(num_sample_trials));
  set_from_boostop(desc, VAR_WITH_NAME(max_expands));
  set_from_boostop(desc, VAR_WITH_NAME(cut_actions));
  set_from_boostop(desc, VAR_WITH_NAME(duplicate_detection_int));
  set_from_boostop(desc, VAR_WITH_NAME(use_landmarks));
  set_from_boostop(desc, VAR_WITH_NAME(factor_duplicate_detection));
  set_from_boostop(desc, VAR_WITH_NAME(epsilon_soft_duplicate));
  set_from_boostop(desc, VAR_WITH_NAME(add_node_if_better));
  set_from_boostop(desc, VAR_WITH_NAME(debug));
  set_from_boostop(desc, VAR_WITH_NAME(add_after_expand));
  set_from_boostop(desc, VAR_WITH_NAME(motionsFile));
  set_from_boostop(desc, VAR_WITH_NAME(outFile));
  set_from_boostop(desc, VAR_WITH_NAME(search_timelimit));
#endif
}

void Options_dbastar::print(std::ostream &out, const std::string &be,
                            const std::string &af) const {

  out << be << STR(delta, af) << std::endl;
  out << be << STR(epsilon, af) << std::endl;
  out << be << STR(alpha, af) << std::endl;
  out << be << STR(filterDuplicates, af) << std::endl;
  out << be << STR(maxCost, af) << std::endl;
  out << be << STR(heuristic, af) << std::endl;
  out << be << STR(max_motions, af) << std::endl;
  out << be << STR(resolution, af) << std::endl;
  out << be << STR(delta_factor_goal, af) << std::endl;
  out << be << STR(cost_delta_factor, af) << std::endl;
  out << be << STR(rebuild_every, af) << std::endl;
  out << be << STR(num_sample_trials, af) << std::endl;
  out << be << STR(max_expands, af) << std::endl;
  out << be << STR(cut_actions, af) << std::endl;
  out << be << STR(duplicate_detection_int, af) << std::endl;
  out << be << STR(use_landmarks, af) << std::endl;
  out << be << STR(factor_duplicate_detection, af) << std::endl;
  out << be << STR(epsilon_soft_duplicate, af) << std::endl;
  out << be << STR(add_node_if_better, af) << std::endl;
  out << be << STR(debug, af) << std::endl;
  out << be << STR(add_after_expand, af) << std::endl;
  out << be << STR(motionsFile, af) << std::endl;
  out << be << STR(outFile, af) << std::endl;
  out << be << STR(search_timelimit, af) << std::endl;
}

void Options_dbastar::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Options_dbastar::read_from_yaml(YAML::Node &node) {

  if (node["options_dbastar"]) {
    __read_from_node(node["options_dbastar"]);
  } else {
    __read_from_node(node);
  }
}
void Options_dbastar::__read_from_node(const YAML::Node &node) {

  __load_data(&const_cast<YAML::Node &>(node), false);

#if 0
  set_from_yaml(node, VAR_WITH_NAME(delta));
  set_from_yaml(node, VAR_WITH_NAME(epsilon));
  set_from_yaml(node, VAR_WITH_NAME(alpha));
  set_from_yaml(node, VAR_WITH_NAME(filterDuplicates));
  set_from_yaml(node, VAR_WITH_NAME(maxCost));
  set_from_yaml(node, VAR_WITH_NAME(heuristic));
  set_from_yaml(node, VAR_WITH_NAME(max_motions));
  set_from_yaml(node, VAR_WITH_NAME(resolution));
  set_from_yaml(node, VAR_WITH_NAME(delta_factor_goal));
  set_from_yaml(node, VAR_WITH_NAME(cost_delta_factor));
  set_from_yaml(node, VAR_WITH_NAME(rebuild_every));
  set_from_yaml(node, VAR_WITH_NAME(num_sample_trials));
  set_from_yaml(node, VAR_WITH_NAME(max_expands));
  set_from_yaml(node, VAR_WITH_NAME(cut_actions));
  set_from_yaml(node, VAR_WITH_NAME(duplicate_detection_int));
  set_from_yaml(node, VAR_WITH_NAME(use_landmarks));
  set_from_yaml(node, VAR_WITH_NAME(factor_duplicate_detection));
  set_from_yaml(node, VAR_WITH_NAME(epsilon_soft_duplicate));
  set_from_yaml(node, VAR_WITH_NAME(add_node_if_better));
  set_from_yaml(node, VAR_WITH_NAME(debug));
  set_from_yaml(node, VAR_WITH_NAME(add_after_expand));
  set_from_yaml(node, VAR_WITH_NAME(motionsFile));
  set_from_yaml(node, VAR_WITH_NAME(outFile));
  set_from_yaml(node, VAR_WITH_NAME(search_timelimit));
#endif
}

void Time_benchmark::write(std::ostream &out) {

  std::string be = "";
  std::string af = ": ";

  out << be << STR(time_nearestMotion, af) << std::endl;
  out << be << STR(time_nearestNode, af) << std::endl;
  out << be << STR(time_nearestNode_add, af) << std::endl;
  out << be << STR(time_nearestNode_search, af) << std::endl;
  out << be << STR(time_collisions, af) << std::endl;
  out << be << STR(prepare_time, af) << std::endl;
  out << be << STR(total_time, af) << std::endl;
  out << be << STR(expands, af) << std::endl;
  out << be << STR(num_nn_motions, af) << std::endl;
  out << be << STR(num_nn_states, af) << std::endl;
  out << be << STR(num_col_motions, af) << std::endl;
  out << be << STR(motions_tree_size, af) << std::endl;
  out << be << STR(states_tree_size, af) << std::endl;
};

// void generate_env(YAML::Node &env,
//                   std::vector<fcl::CollisionObjectf *> &obstacles,
//                   fcl::BroadPhaseCollisionManagerf *bpcm_env) {
//
//   for (const auto &obs : env["environment"]["obstacles"]) {
//     double default_size = 1;
//     double default_pose = 0;
//
//     if (obs["type"].as<std::string>() == "box") {
//       const auto &size = obs["size"];
//       std::shared_ptr<fcl::CollisionGeometryf> geom;
//
//       if (size.size() == 2)
//         geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(),
//                                  default_size));
//       else
//         geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(),
//                                  size[2].as<float>()));
//
//       const auto &center = obs["center"];
//       auto co = new fcl::CollisionObjectf(geom);
//
//       if (size.size() == 2)
//         co->setTranslation(fcl::Vector3f(center[0].as<float>(),
//                                          center[1].as<float>(),
//                                          default_pose));
//       else
//         co->setTranslation(fcl::Vector3f(center[0].as<float>(),
//                                          center[1].as<float>(),
//                                          center[2].as<float>()));
//
//       co->computeAABB();
//       obstacles.push_back(co);
//     } else {
//       throw std::runtime_error("Unknown obstacle type!");
//     }
//   }
//
//   bpcm_env->registerObjects(obstacles);
//   bpcm_env->setup();
// }

void compute_col_shape(Motion &m, RobotOmpl &robot) {
  Eigen::VectorXd x(robot.diff_model->nx);
  for (auto &state : m.states) {
    robot.toEigen(state, x);
    auto &ts_data = robot.diff_model->ts_data;
    auto &col_geo = robot.diff_model->collision_geometries;
    robot.diff_model->transformation_collision_geometries(x, ts_data);

    for (size_t i = 0; i < ts_data.size(); i++) {
      auto &transform = ts_data.at(i);
      auto co = new fcl::CollisionObjectd(col_geo.at(i));
      co->setTranslation(transform.translation());
      co->setRotation(transform.rotation());
      co->computeAABB();
      m.collision_objects.push_back(co);
    }
  }

  m.collision_manager.reset(
      new ShiftableDynamicAABBTreeCollisionManager<double>());
  m.collision_manager->registerObjects(m.collision_objects);
};

void traj_to_motion(const Trajectory &traj, RobotOmpl &robot,
                    Motion &motion_out) {

  auto si = robot.getSpaceInformation();

  motion_out.states.resize(traj.states.size());

  std::transform(traj.states.begin(), traj.states.end(),
                 motion_out.states.begin(), [&](auto &x) {
                   // CSTR_V(x);
                   ob::State *state = si->allocState();
                   // printState(std::cout, si, state);
                   robot.fromEigen(state, x);
                   // printState(std::cout, si, state);
                   return state;
                 });

  motion_out.actions.resize(traj.actions.size());

  std::transform(traj.actions.begin(), traj.actions.end(),
                 motion_out.actions.begin(), [&](auto &a) {
                   oc::Control *action = si->allocControl();
                   control_from_eigen(a, si, action);
                   return action;
                 });

  compute_col_shape(motion_out, robot);
  motion_out.cost = traj.cost;
}

//
//
// new!
void load_motion_primitives_new(const std::string &motionsFile,
                                RobotOmpl &robot, std::vector<Motion> &motions,
                                int max_motions, bool cut_actions,
                                bool shuffle) {

  Trajectories trajs;

  trajs.load_file_boost(motionsFile.c_str());

  // CSTR_(max_motions);
  // CSTR_(trajs.data.size());
  if (max_motions < trajs.data.size())
    trajs.data.resize(max_motions);

  motions.resize(trajs.data.size());

  CSTR_(trajs.data.size());
  std::transform(trajs.data.begin(), trajs.data.end(), motions.begin(),
                 [&](const auto &traj) {
                   // traj.to_yaml_format(std::cout, "");
                   Motion m;
                   traj_to_motion(traj, robot, m);
                   return m;
                 });

  CHECK(motions.size(), AT);
  if (motions.front().cost > 1e5) {
    std::cout << "WARNING: motions have infinite cost." << std::endl;
    std::cout << "-- using default cost: TIME" << std::endl;
    for (auto &m : motions) {
      m.cost = robot.diff_model->ref_dt * m.actions.size();
    }
  }

  if (cut_actions) {
    NOT_IMPLEMENTED;
  }

  if (shuffle) {
    std::shuffle(std::begin(motions), std::end(motions),
                 std::default_random_engine{});
  }

  for (size_t idx = 0; idx < motions.size(); ++idx) {
    motions[idx].idx = idx;
  }
}

void load_motion_primitives(const std::string &motionsFile, RobotOmpl &robot,
                            std::vector<Motion> &motions, int max_motions,
                            bool cut_actions, bool shuffle) {

  auto si = robot.getSpaceInformation();
  // load motions primitives
  std::cout << "loading file: " << motionsFile << std::endl;
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

  // create a robot with no position bounds
  // ob::RealVectorBounds position_bounds_no_bound(env_min.size());
  // position_bounds_no_bound.setLow(-1e6);
  // position_bounds_no_bound.setHigh(1e6); //
  // std::numeric_limits<double>::max()); std::shared_ptr<RobotOmpl>
  // robot_no_pos_bound =
  //     create_robot_ompl_ompl_ompl(robotType, position_bounds_no_bound);
  // auto si_no_pos_bound = robot_no_pos_bound->getSpaceInformation();
  // si_no_pos_bound->setPropagationStepSize(1);
  // si_no_pos_bound->setMinMaxControlDuration(1, 1);
  // si_no_pos_bound->setStateValidityChecker(stateValidityChecker);
  // si_no_pos_bound->setStatePropagator(statePropagator);
  // si_no_pos_bound->setup();

  if (msg_obj.type != msgpack::type::ARRAY) {
    std::cout << STR_(msg_obj.type) << std::endl;
    throw msgpack::type_error();
  }

  std::cout << "Number of motions in file: " << msg_obj.via.array.size
            << std::endl;
  for (size_t i = 0; i < msg_obj.via.array.size; ++i) {
    bool good = true;
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

          // normalize quaternion?

          if (robot.getName() == "Quadrotor") {
            size_t i_q = 3;
            double norm = std::sqrt(reals.at(i_q + 0) * reals.at(i_q + 0) +
                                    reals.at(i_q + 1) * reals.at(i_q + 1) +
                                    reals.at(i_q + 2) * reals.at(i_q + 2) +
                                    reals.at(i_q + 3) * reals.at(i_q + 3));
            reals.at(i_q + 0) /= norm;
            reals.at(i_q + 1) /= norm;
            reals.at(i_q + 2) /= norm;
            reals.at(i_q + 3) /= norm;
          }

          si->getStateSpace()->copyFromReals(state, reals);
          m.states.push_back(state);
          // TODO: add and code check the primitive!! -- then skip the
          // position bounds
          //
          //
          // if
          // (!robot.getSpaceInformation()->satisfiesBounds(m.states.back()))
          // {
          //   std::cout << "Warning invalid state" << std::endl;
          //   si->printState(m.states.back());
          //   good = false;
          //   si->enforceBounds(m.states.back());
          //   si->printState(m.states.back());
          //   break;
          // }
        }
        break;
      }
    }
    // num_states += m.states.size();
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
    m.cost = m.actions.size() * robot.dt(); // time in seconds
    m.idx = motions.size();
    // m.name = motion["name"].as<std::string>();

    compute_col_shape(m, robot);

    m.disabled = false;
    if (good)
      motions.push_back(m);
    if (motions.size() >= static_cast<size_t>(max_motions)) {
      std::cout << "stop loading more motions -- reached max_motions "
                << max_motions << std::endl;
      break;
    }
  }

  std::vector<Motion> new_motions;
  if (cut_actions) {
    std::cout << "adding new motions by cutting " << std::endl;
    for (size_t im = 0; im < motions.size(); im++) {

      Motion &m_ref = motions.at(im);
      int num_states = m_ref.states.size();
      Motion m;

      std::copy_n(m_ref.states.begin(), num_states / 2,
                  std::back_inserter(m.states));
      std::copy_n(m_ref.actions.begin(), num_states / 2 - 1,
                  std::back_inserter(m.actions));

      m.cost = m.actions.size() * robot.dt(); // time in seconds
      m.idx = im + motions.size();

      Eigen::VectorXd x(robot.diff_model->nx);
      for (auto &state : m.states) {
        robot.toEigen(state, x);
        auto &ts_data = robot.diff_model->ts_data;
        auto &col_geo = robot.diff_model->collision_geometries;
        robot.diff_model->transformation_collision_geometries(x, ts_data);

        for (size_t i = 0; i < ts_data.size(); i++) {
          auto &transform = ts_data.at(i);
          auto co = new fcl::CollisionObjectd(col_geo.at(i));
          co->setTranslation(transform.translation());
          co->setRotation(transform.rotation());
          co->computeAABB();
          m.collision_objects.push_back(co);
        }
      }

      m.collision_manager.reset(
          new ShiftableDynamicAABBTreeCollisionManager<double>());
      m.collision_manager->registerObjects(m.collision_objects);

      new_motions.push_back(m);
    }

    motions.insert(motions.end(), new_motions.begin(), new_motions.end());
  }

  std::cout << "Motions loaded" << std::endl;
  std::cout << STR_(motions.size()) << std::endl;

  if (shuffle) {
    std::shuffle(std::begin(motions), std::end(motions),
                 std::default_random_engine{});
  }

  for (size_t idx = 0; idx < motions.size(); ++idx) {
    motions[idx].idx = idx;
  }
}

double automatic_delta(double delta_in, double alpha, RobotOmpl &robot,
                       ompl::NearestNeighbors<Motion *> &T_m) {
  Motion fakeMotion;
  fakeMotion.idx = -1;
  auto si = robot.getSpaceInformation();
  fakeMotion.states.push_back(si->allocState());
  std::vector<Motion *> neighbors_m;
  size_t num_desired_neighbors = (size_t)-delta_in;
  size_t num_samples = std::min<size_t>(1000, T_m.size());

  auto state_sampler = si->allocStateSampler();
  double sum_delta = 0.0;
  for (size_t k = 0; k < num_samples; ++k) {
    do {
      state_sampler->sampleUniform(fakeMotion.states[0]);
    } while (!si->isValid(fakeMotion.states[0]));
    robot.setPosition(fakeMotion.states[0], fcl::Vector3d(0, 0, 0));

    T_m.nearestK(&fakeMotion, num_desired_neighbors + 1, neighbors_m);

    double max_delta =
        si->distance(fakeMotion.states[0], neighbors_m.back()->states.front());
    sum_delta += max_delta;
  }
  double adjusted_delta = (sum_delta / num_samples) / alpha;
  std::cout << "Automatically adjusting delta to: " << adjusted_delta
            << std::endl;

  si->freeState(fakeMotion.states.back());

  return adjusted_delta;
}

void filte_duplicates(std::vector<Motion> &motions, double delta, double alpha,
                      RobotOmpl &robot, ompl::NearestNeighbors<Motion *> &T_m) {

  auto si = robot.getSpaceInformation();
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
    T_m.nearestR(&fakeMotion, delta * alpha, neighbors_m);

    for (Motion *nm : neighbors_m) {
      if (nm == &m || nm->disabled) {
        continue;
      }
      double goal_delta = si->distance(m.states.back(), nm->states.back());
      if (goal_delta < delta * (1 - alpha)) {
        nm->disabled = true;
        ++num_duplicates;
      }
    }
  }
  std::cout << "There are " << num_duplicates << " duplicate motions!"
            << std::endl;
}

void add_landmarks(ompl::NearestNeighbors<AStarNode *> *T_landmarks,
                   std::vector<AStarNode *> landmark_nodes,
                   std::shared_ptr<RobotOmpl> robot, size_t num_sample_trials) {

  CHECK(T_landmarks, AT);
  auto si = robot->getSpaceInformation();
  T_landmarks->setDistanceFunction([&](const AStarNode *a, const AStarNode *b) {
    return si->distance(a->state, b->state);
  });

  auto check_fun = [&](Sample_ *x) {
    return si->getStateValidityChecker()->isValid(x);
  };

  auto sample_fun = [&](double *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
      Eigen::Map<Eigen::VectorXd> s(x, n);
      robot->diff_model->sample_uniform(s);
    };
  };

  std::vector<Sample_ *> landmark_samples;

  generate_batch(sample_fun, check_fun, num_sample_trials, robot->nx, si,
                 landmark_samples);
  landmark_samples.push_back(robot->startState);
  landmark_samples.push_back(robot->goalState);

  landmark_nodes.resize(landmark_samples.size());

  double BIG_NUM = 1e8;
  for (size_t i = 0; i < landmark_samples.size(); i++) {
    auto ptr = new AStarNode;
    ptr->state = landmark_samples.at(i);
    ptr->gScore = BIG_NUM;
    ptr->hScore = BIG_NUM;
    ptr->fScore = BIG_NUM;
    landmark_nodes.at(i) = ptr;
  }
  T_landmarks->add(landmark_nodes);
}

void Motion::print(std::ostream &out,
                   std::shared_ptr<ompl::control::SpaceInformation> &si) {

  out << "states " << std::endl;
  for (auto &state : states) {
    printState(out, si, state);
    out << std::endl;
  }
  out << "actions: " << std::endl;
  for (auto &action : actions) {
    printAction(std::cout, si, action);
    out << std::endl;
  }

  STRY(cost, out, "", ": ");
  STRY(idx, out, "", ": ");
  STRY(disabled, out, "", ": ");
}

// continue here: cost lower bound for the quadcopter
void dbastar(const Problem &problem, const Options_dbastar &options_dbastar,
             Trajectory &traj_out, Out_info_db &out_info_db) {
  // TODO:
  // - disable motions should not be on the search tree!

  Options_dbastar options_dbastar_local = options_dbastar;

  std::cout << "*** options_dbastar_local ***" << std::endl;
  options_dbastar_local.print(std::cout);
  std::cout << "***" << std::endl;

  const std::string space6 = "      ";
  const std::string space4 = "    ";

  auto tic = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<std::vector<double>>> expansions;

  Time_benchmark time_bench;

  Duplicate_detection duplicate_detection = static_cast<Duplicate_detection>(
      options_dbastar_local.duplicate_detection_int);

  std::shared_ptr<RobotOmpl> robot = robot_factory_ompl(problem);

  auto si = robot->getSpaceInformation();
  std::cout << "Space info" << std::endl;
  si->printSettings(std::cout);
  std::cout << "***" << std::endl;

  auto startState = robot->startState;
  auto goalState = robot->goalState;

  std::vector<Motion> motions;
  if (options_dbastar_local.motions_ptr) {
    std::cout << "motions have alredy loaded " << std::endl;
    motions = *options_dbastar_local.motions_ptr;

  } else {
    std::cout << "loading motions ... " << std::endl;
    load_motion_primitives(options_dbastar_local.motionsFile, *robot, motions,
                           options_dbastar_local.max_motions,
                           options_dbastar_local.cut_actions, true);
  }

  if (options_dbastar_local.max_motions < motions.size())
    motions.resize(options_dbastar_local.max_motions);

  std::cout << "There are " << motions.size() << " motions!" << std::endl;

  size_t num_print_motions = 10;

  for (size_t i = 0; i < num_print_motions; i++) {
    motions.at(i).print(std::cout, si);
  }

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
    time_bench.time_nearestMotion += out.second;
  }

  if (options_dbastar_local.alpha <= 0 || options_dbastar_local.alpha >= 1) {
    ERROR_WITH_INFO("Alpha needs to be between 0 and 1!");
  }

  if (options_dbastar_local.delta < 0) {
    options_dbastar_local.delta = automatic_delta(
        options_dbastar_local.delta, options_dbastar_local.alpha, *robot, *T_m);
  }

  if (options_dbastar_local.filterDuplicates) {
    filte_duplicates(motions, options_dbastar_local.delta,
                     options_dbastar_local.alpha, *robot, *T_m);
  }

  std::shared_ptr<Heu_fun> h_fun;

  switch (options_dbastar_local.heuristic) {
  case 0: {
    h_fun = std::make_shared<Heu_euclidean>(robot, robot->goalState);
  } break;
  case 1: {
    h_fun = std::make_shared<Heu_roadmap>(
        robot, options_dbastar_local.num_sample_trials, robot->startState,
        robot->goalState, options_dbastar_local.resolution);
  } break;
  case -1: {
    h_fun = std::make_shared<Heu_blind>();
  } break;
  default: {
    ERROR_WITH_INFO("not implemented");
  }
  }

  ompl::NearestNeighbors<AStarNode *> *T_n;
  if (si->getStateSpace()->isMetricSpace()) {

    T_n = new ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *>();
    static_cast<ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(T_n)
        ->rebuildSize_ = 5000;

  } else {
    T_n = new ompl::NearestNeighborsSqrtApprox<AStarNode *>();
  }
  T_n->setDistanceFunction([si](const AStarNode *a, const AStarNode *b) {
    return si->distance(a->state, b->state);
  });

  std::vector<const Sample_ *> expanded;
  std::vector<std::vector<double>> found_nn;
  ompl::NearestNeighbors<AStarNode *> *T_landmarks;
  std::vector<AStarNode *> ptrs; // TODO: i should clean memory

  auto start_node = new AStarNode;
  start_node->gScore = 0;
  start_node->state = robot->startState;

  double hstart = h_fun->h(robot->startState);

  start_node->fScore = options_dbastar_local.epsilon * hstart;
  start_node->hScore = start_node->fScore;

  std::cout << "start_node heuristic g: " << start_node->gScore
            << " h: " << start_node->hScore << " f: " << start_node->fScore
            << std::endl;

  start_node->came_from = nullptr;
  start_node->used_offset = fcl::Vector3d(0, 0, 0);
  start_node->used_motion = -1;

  // db-A* search
  open_t open;

  auto handle = open.push(start_node);
  start_node->handle = handle;
  start_node->is_in_open = true;

  std::vector<std::vector<double>> _states_debug;

  Motion fakeMotion;
  fakeMotion.idx = -1;
  fakeMotion.states.push_back(si->allocState());

  AStarNode *query_n = new AStarNode();

  ob::State *tmpStateq = si->allocState();
  ob::State *tmpState = si->allocState();
  ob::State *tmpState_debug = si->allocState();
  ob::State *tmpState2 = si->allocState();
  std::vector<Motion *> neighbors_m;
  std::vector<AStarNode *> neighbors_n;

  double last_f_score = start_node->fScore;
  // size_t expands = 0;
  // clock start

  if (options_dbastar_local.use_landmarks) {

    T_landmarks = new ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *>();
    add_landmarks(T_landmarks, ptrs, robot,
                  options_dbastar_local.num_sample_trials);
    T_n = T_landmarks;
  }

  // int counter_nn = 0;

  AStarNode *solution = nullptr;

  bool check_average_min_d = true;
  double total_d = 0;
  int counter = 0;
  double average_min_d = 0;
  if (check_average_min_d) {
    for (auto &p : ptrs) {
      std::vector<AStarNode *> neighbors;
      T_n->nearestK(p, 2, neighbors);
      if (neighbors.front() != p) {
        ERROR_WITH_INFO("why?");
      }
      total_d += si->distance(neighbors.at(1)->state, p->state);
      counter++;
    }
    average_min_d = total_d / counter;
    std::cout << "average min distance " << average_min_d << std::endl;
  }

  Terminate_status status = Terminate_status::UNKNOWN;

  std::cout << "cost low bound at beginning is "
            << robot->cost_lower_bound(startState, goalState) << std::endl;

  bool print_queue = false;
  int print_every = 100;

  if (options_dbastar_local.add_node_if_better) {
    // didn't check if this works togehter
    CHECK_EQ(options_dbastar_local.duplicate_detection_int,
             static_cast<int>(Duplicate_detection::NO), AT);
  }

  double radius =
      options_dbastar_local.delta * (1. - options_dbastar_local.alpha);

  auto nearest_motion_timed = [&](auto &fakeMotion, auto &neighbors_m) {
    auto out = timed_fun([&] {
      T_m->nearestR(&fakeMotion,
                    options_dbastar_local.delta * options_dbastar_local.alpha,
                    neighbors_m);
      return 0;
    });
    time_bench.time_nearestMotion += out.second;
    time_bench.num_nn_motions++;
  };

  auto nearestR_state_timed = [&](auto &query_n, auto &neighbors_n) {
    auto _out = timed_fun([&] {
      T_n->nearestR(query_n, radius, neighbors_n);
      return 0;
    });
    time_bench.num_nn_states++;
    time_bench.time_nearestNode += _out.second;
    time_bench.time_nearestNode_search += _out.second;
  };

  auto is_motion_valid_timed = [&](auto &motion, auto &offset,
                                   bool &motionValid) {
    auto out = timed_fun([&] {
      motion->collision_manager->shift(offset);
      fcl::DefaultCollisionData<double> collision_data;
      motion->collision_manager->collide(robot->diff_model->env.get(),
                                         &collision_data,
                                         fcl::DefaultCollisionFunction<double>);
      bool motionValid = !collision_data.result.isCollision();
      motion->collision_manager->shift(-offset);
      return motionValid;
    });

    motionValid = out.first;
    time_bench.time_collisions += out.second;
    time_bench.num_col_motions++;
  };

  auto add_state_timed = [&](auto &node) {
    auto out = timed_fun([&] {
      T_n->add(node);
      return 0;
    });
    time_bench.time_nearestNode += out.second;
    time_bench.time_nearestNode_add += out.second;
  };

  if (!options_dbastar_local.use_landmarks &&
      !options_dbastar_local.add_after_expand) {
    if (options_dbastar_local.debug) {
      std::vector<double> _state;
      si->getStateSpace()->copyToReals(_state, start_node->state);
      _states_debug.push_back(_state);
    }
    add_state_timed(start_node);
  }

  if (options_dbastar_local.add_after_expand) {
    CHECK(!options_dbastar_local.add_node_if_better, AT);
  }

  // std::cout << "MOTIONS" << std::endl;
  // std::vector<Motion *> mss;
  // T_m->list(mss);
  // for (auto &ms : mss) {
  //   si->printState(ms->states.at(0));
  // }
  // std::cout << "END MOTIONS" << std::endl;

  auto tac = std::chrono::high_resolution_clock::now();
  time_bench.prepare_time =
      std::chrono::duration<double, std::milli>(tac - tic).count();

  static_cast<ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(T_n)
      ->rebuildSize_ = 1e8;

  auto start = std::chrono::steady_clock::now();

  auto get_time_stamp_ms = [&] {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };

  while (true) {

    if (static_cast<size_t>(time_bench.expands) >=
        options_dbastar_local.max_expands) {
      status = Terminate_status::MAX_EXPANDS;
      break;
    }

    if (open.empty()) {
      status = Terminate_status::EMPTY_QUEUE;
      break;
    }

    if (get_time_stamp_ms() > options_dbastar_local.search_timelimit) {
      status = Terminate_status::MAX_TIME;
      break;
    }

    if (print_queue) {
      std::cout << "QUEUE " << std::endl;

      for (auto &o : open) {
        std::cout << " *** " << std::endl;
        std::cout << o->fScore << std::endl;
        std::cout << o->gScore << std::endl;
        std::cout << o->hScore << std::endl;
        printState(std::cout, si, o->state);
        std::cout << "\n*** " << std::endl;
      }
      std::cout << "END QUEUE" << std::endl;
    }

    AStarNode *current = open.top();
    open.pop();
    // std::cout << "current state ";
    // printState(std::cout, si, current->state);
    // std::cout << std::endl;

    expanded.push_back(current->state);
    ++time_bench.expands;

    if (options_dbastar_local.add_after_expand) {
      // i have to check if state is novel

      std::vector<AStarNode *> neighbors_n;
      nearestR_state_timed(current, neighbors_n);
      if (neighbors_n.size()) {
        continue;
      } else {
        // add the node to the tree
        add_state_timed(current);
      }
    }

    if (time_bench.expands % print_every == 0 || time_bench.expands == 1) {
      std::cout << "expanded: " << time_bench.expands
                << " open: " << open.size() << " nodes: " << T_n->size()
                << " f-score " << current->fScore << " h-score "
                << current->hScore << " g-score " << current->gScore
                << std::endl;
    }

    if (time_bench.expands % options_dbastar_local.rebuild_every == 0 &&
        !options_dbastar_local.use_landmarks) {

      auto out = timed_fun([&] {
        auto p_derived = static_cast<
            ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(T_n);
        p_derived->rebuildDataStructure();
        p_derived->rebuildSize_ = 1e8;
        return 0;
      });
      time_bench.time_nearestNode += out.second;
      time_bench.time_nearestNode_add += out.second;
    }

    if (options_dbastar_local.heuristic == 0) {

      if (current->fScore < last_f_score) {
        std::cout << "expand: " << time_bench.expands << " state: ";
        si->printState(current->state);
        std::cout << "  -- WARNING, assert(current->fScore >= last_f_score) "
                  << current->fScore << " " << last_f_score << std::endl;
        // throw -1;
      }
    }
    last_f_score = current->fScore;
    // si->printState(current->state);
    // si->printState(goalState);
    if (si->distance(current->state, goalState) <=
        options_dbastar_local.delta_factor_goal * options_dbastar_local.delta) {
      solution = current;
      std::cout << "current state is " << std::endl;
      printState(std::cout, si, current->state);
      std::cout << std::endl;
      status = Terminate_status::SOLVED;
      std::cout << "SOLUTION FOUND!!!! cost: " << current->gScore << std::endl;
      break;
    }

    current->is_in_open = false;

    si->copyState(fakeMotion.states[0], current->state);

    if (robot->isTranslationInvariant())
      robot->setPosition(fakeMotion.states[0], fcl::Vector3d(0, 0, 0));

    nearest_motion_timed(fakeMotion, neighbors_m);

#if 0 
    std::cout << "current state state " << std::endl;
    si->printState(fakeMotion.states.front());
    std::cout << "reporting distances" << std::endl;
    std::vector<Motion *> mss;
    T_m->list(mss);
    std::cout << "mss.size() " << mss.size() << std::endl;
    double min_distance = std::numeric_limits<double>::max();

    for (auto &ms : mss) {
      si->printState(ms->states.at(0));
      double distance = si->distance(fakeMotion.states[0], ms->states.at(0));
      std::cout << "dist "
                << si->distance(fakeMotion.states[0], ms->states.at(0))
                << std::endl;
      if (distance < min_distance)
        min_distance = distance;
    }
    std::cout << "min distance is " << min_distance << std::endl;
    std::cout << "num neighbors " << std::endl;
    std::cout << STR_(neighbors_m.size()) << std::endl;
#endif

    for (const Motion *motion : neighbors_m) {
      if (motion->disabled) {
        continue;
      }

      fcl::Vector3d offset(0., 0., 0.);
      fcl::Vector3d computed_offset(0., 0., 0.);
      fcl::Vector3d current_pos(0., 0., 0.);
      double tentative_gScore = current->gScore + motion->cost;
      si->copyState(tmpState, motion->states.back());

      if (robot->isTranslationInvariant()) {
        current_pos = robot->getTransform(current->state).translation();
        offset = current_pos + computed_offset;
        const auto relative_pos = robot->getTransform(tmpState).translation();
        robot->setPosition(tmpState, offset + relative_pos);
      }

      if (!si->satisfiesBounds(tmpState)) {
        continue;
      }

      si->copyState(tmpState2, motion->states.front());
      if (robot->isTranslationInvariant()) {
        robot->setPosition(tmpState2, current_pos);
      }
      tentative_gScore += options_dbastar_local.cost_delta_factor *
                          robot->cost_lower_bound(tmpState2, current->state);
      double tentative_hScore = h_fun->h(tmpState);

      double tentative_fScore =
          tentative_gScore + options_dbastar_local.epsilon * tentative_hScore;

      if (tentative_fScore > options_dbastar_local.maxCost) {
        continue;
      }

      bool motionValid;

      is_motion_valid_timed(motion, offset, motionValid);

      if (!motionValid) {
        continue;
      }

      query_n->state = tmpState;

      if (options_dbastar_local.debug) {

        std::vector<double> _state;
        si->getStateSpace()->copyToReals(_state, query_n->state);
        _states_debug.push_back(_state);

        std::vector<std::vector<double>> motion_v;

        for (size_t k = 0; k < motion->states.size(); ++k) {
          const auto state = motion->states[k];
          si->copyState(tmpState_debug, state);
          if (robot->isTranslationInvariant()) {
            const fcl::Vector3d relative_pos =
                robot->getTransform(state).translation();
            robot->setPosition(tmpState_debug, current_pos + relative_pos);
          }
          std::vector<double> x;
          si->getStateSpace()->copyToReals(x, tmpState_debug);
          motion_v.push_back(x);
        }
        expansions.push_back(motion_v);
      }

      if (!options_dbastar_local.add_after_expand) {

        nearestR_state_timed(query_n, neighbors_n);

        if (neighbors_n.size() == 0 && !options_dbastar_local.use_landmarks) {

          bool add_node = true;

          if (duplicate_detection == Duplicate_detection::HARD ||
              duplicate_detection == Duplicate_detection::SOFT) {

            std::vector<AStarNode *> neighbors_n2;
            T_n->nearestR(query_n,
                          options_dbastar_local.factor_duplicate_detection *
                              radius,
                          neighbors_n2);

            if (neighbors_n2.size()) {
              if (duplicate_detection == Duplicate_detection::HARD) {
                add_node = false;
              } else if (duplicate_detection == Duplicate_detection::SOFT) {

                bool is_best = true;
                tentative_fScore = tentative_hScore + tentative_gScore;

                for (auto &n : neighbors_n2) {
                  if (n->fScore < tentative_fScore) {
                    is_best = false;
                    break;
                  }
                }

                if (!is_best) {
                  tentative_hScore *=
                      options_dbastar_local.epsilon_soft_duplicate;
                  tentative_fScore = tentative_hScore + tentative_gScore;
                }
              }
            }
          }

          if (add_node) {
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

            add_state_timed(node);
          }
        } else {
          // T_n->nearestR(query_n, radius, neighbors_n);
          // check if we have a better path now
          bool added = false;
          for (AStarNode *entry : neighbors_n) {

            if (options_dbastar_local.debug) {
              std::vector<double> landmark_;
              si->getStateSpace()->copyToReals(landmark_, entry->state);
              found_nn.push_back(landmark_);
            }

            // AStarNode* entry = nearest;
            assert(si->distance(entry->state, tmpState) <=
                   options_dbastar_local.delta);
            double extra_time_to_reach =
                options_dbastar_local.cost_delta_factor *
                robot->cost_lower_bound(entry->state, tmpState);
            double tentative_gScore_ = tentative_gScore + extra_time_to_reach;
            double delta_score = entry->gScore - tentative_gScore_;
            // double old_fscore = entry->fScore;
            // std::cout  << "time to reach " << time_to_reach << std::endl;

            if (delta_score > 0) {

              if (options_dbastar_local.add_node_if_better) {
                if (!entry->valid)
                  continue;
                if (!added) {

                  // std::cout << "adding state that is close"  << std::endl;
                  // si->printState(tmpState);

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

                  add_state_timed(node);
                  added = true;
                }
                // I invalidate the neighbour
                if (entry->is_in_open) {
                  entry->fScore = std::numeric_limits<double>::max();
                  entry->gScore = std::numeric_limits<double>::max();
                  entry->hScore = std::numeric_limits<double>::max();
                  entry->valid = false;
                  open.decrease(entry->handle);
                }
              } else {
                entry->gScore = tentative_gScore_;
                // entry->fScore = tentative_fScore;
                entry->hScore = h_fun->h(entry->state);
                // or entry state?
                // entry->hScore = h_fun->h(tmpState);
                entry->fScore = entry->gScore +
                                options_dbastar_local.epsilon * entry->hScore;
                assert(entry->fScore >= 0);
                entry->came_from = current;
                entry->used_motion = motion->idx;
                entry->used_offset = computed_offset;
                if (entry->is_in_open) {
                  // std::cout << "improve score  -- old: " << old_fscore
                  //           << " -- new -- " << entry->fScore << std::endl;
                  open.update(entry->handle); // increase?
                } else {
                  auto handle = open.push(entry);
                  entry->handle = handle;
                  entry->is_in_open = true;
                }
              }
            }
          }
        }
      } else {
        // I directly create a node and add to queue
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
      }
    }
  }
  // clock end
  tac = std::chrono::high_resolution_clock::now();
  time_bench.total_time =
      std::chrono::duration<double, std::milli>(tac - tic).count();

  std::cout << "search has ended " << std::endl;
  CSTR_(time_bench.expands);
  std::cout << "Terminate status: " << static_cast<int>(status) << " "
            << terminate_status_str[static_cast<int>(status)] << std::endl;

  time_bench.motions_tree_size = T_m->size();
  time_bench.states_tree_size = T_n->size();

  std::cout << "writing output to " << options_dbastar_local.outFile
            << std::endl;

  std::ofstream out(options_dbastar_local.outFile);
  out << "status: " << terminate_status_str[static_cast<int>(status)]
      << std::endl;

  out << "start: ";
  printState(out, si, startState);
  out << std::endl;
  out << "goal: ";
  printState(out, si, goalState);
  out << std::endl;
  out << "solved: " << !(!solution) << std::endl;

  options_dbastar_local.print(out);
  time_bench.write(out);

  if (options_dbastar_local.debug) {

    if (options_dbastar_local.heuristic == 1) {
      out << "batch:" << std::endl;
      auto fun_d = static_cast<Heu_roadmap *>(h_fun.get());
      for (auto &x : fun_d->batch_samples) {
        out << "      - ";
        printState(out, si, x);
        out << std::endl;
      }
    }

    out << "kdtree:" << std::endl;
    std::vector<AStarNode *> nodes_in_tree;
    T_n->list(nodes_in_tree);

    for (auto &x : nodes_in_tree) {
      out << space6 + "- ";
      printState(out, si, x->state);
      out << std::endl;
    }

    out << "expanded:" << std::endl;
    for (auto &x : expanded) {
      out << space6 + "- ";
      printState(out, si, x);
      out << std::endl;
    }

    out << "found_nn:" << std::endl;
    size_t max_nn = 1000;
    size_t it_nn = 0;
    for (auto &x : found_nn) {
      if (it_nn++ > max_nn)
        break;
      out << space6 + "- ";
      printState(out, x);
      out << std::endl;
    }

    out << "expansions:" << std::endl;
    size_t max_expansion = 1000;
    size_t it_expansion = 0;
    for (auto &e : expansions) {
      if (it_expansion++ > max_expansion)
        break;
      out << space6 + "- " << std::endl;
      for (auto &x : e) {
        out << space6 + "  - ";
        printState(out, x);
        out << std::endl;
      }
    }
  }

  double time_jumps = 0;
  double cost_with_jumps = 0;

  Trajectory traj;
  if (solution) {
    state_to_eigen(traj_out.start, si, startState);
    state_to_eigen(traj_out.goal, si, goalState);

    out << "cost: " << solution->gScore << std::endl;

    double extra_time_weighted =
        options_dbastar_local.cost_delta_factor *
        robot->cost_lower_bound(solution->state, goalState);
    std::cout << "extra time " << extra_time_weighted << std::endl;

    std::cout << "solution with extra time with cost-to-goal: "
              << solution->gScore + extra_time_weighted << std::endl;

    std::vector<const AStarNode *> result;

    const AStarNode *n = solution;
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

    out << "result:" << std::endl;
    out << "  - states:" << std::endl;

    si->copyState(tmpStateq, startState);

    for (size_t i = 0; i < result.size() - 1; ++i) {
      // Compute intermediate states
      const auto node_state = result[i]->state;
      fcl::Vector3d current_pos(0, 0, 0);

      if (robot->isTranslationInvariant())
        current_pos = robot->getTransform(node_state).translation();
      const auto &motion = motions.at(result[i + 1]->used_motion);
      out << space6 + "# ";
      printState(out, si, node_state); // debug
      out << std::endl;
      out << space6 + "# motion " << motion.idx << " with cost " << motion.cost
          << std::endl; // debug
      // skip last state each

      for (size_t k = 0; k < motion.states.size(); ++k) {
        const auto state = motion.states[k];
        si->copyState(tmpState, state);

        if (robot->isTranslationInvariant()) {
          const fcl::Vector3d relative_pos =
              robot->getTransform(state).translation();
          robot->setPosition(tmpState, current_pos +
                                           result[i + 1]->used_offset +
                                           relative_pos);
        }

        if (k < motion.states.size() - 1) {
          if (k == 0) {
            out << space6 + "# jump from ";
            printState(out, si, tmpStateq); // debug
            out << " to ";
            printState(out, si, tmpState);                         // debug
            out << " delta " << si->distance(tmpStateq, tmpState); // debug
            double min_time = robot->cost_lower_bound(tmpStateq, tmpState);
            time_jumps += min_time;
            out << " min time " << min_time; // debug
            out << std::endl;
          }

          out << space6 + "- ";
          Eigen::VectorXd x;
          state_to_eigen(x, si, tmpState);
          traj_out.states.push_back(x);

        } else {
          out << space6 + "# "; // debug
          si->copyState(tmpStateq, tmpState);
        }

        printState(out, si, tmpState);

        out << std::endl;
      }
      out << std::endl;
    }
    out << space6 + "- ";

    // printing the last state
    Eigen::VectorXd x;
    state_to_eigen(x, si, tmpState);
    traj_out.states.push_back(x);
    printState(out, si, result.back()->state);
    std::cout << " time jumps " << time_jumps << std::endl;
    out << std::endl;
    out << "    actions:" << std::endl;

    int action_counter = 0;
    for (size_t i = 0; i < result.size() - 1; ++i) {
      const auto &motion = motions[result[i + 1]->used_motion];
      out << space6 + "# motion " << motion.idx << " with cost " << motion.cost
          << std::endl; // debug
      for (size_t k = 0; k < motion.actions.size(); ++k) {
        const auto &action = motion.actions[k];
        out << space6 + "- ";
        action_counter += 1;
        printAction(out, si, action);
        Eigen::VectorXd x;
        control_to_eigen(x, si, action);
        traj_out.actions.push_back(x);
        out << std::endl;
      }
      out << std::endl;
    }
    // dts

    out << "result2:" << std::endl;
    out << "  - states:" << std::endl;

    // I want a solution that accounts for deltas
    // also, it should include the default

    // continue here
    double dt = robot->dt();

    si->copyState(tmpStateq, startState);

    //  First State
    si->copyState(tmpState, startState);

    std::vector<const Sample_ *> states2;
    std::vector<double> times;
    std::vector<oc::Control *> actions2;

    oc::Control *uzero = si->allocControl();

    copyFromRealsControl(
        si, uzero,
        std::vector<double>(robot->u_zero.data(),
                            robot->u_zero.data() + robot->u_zero.size()));

    std::cout << "action uzero" << std::endl;

    printAction(std::cout, si, uzero);
    std::cout << std::endl;

    for (size_t i = 0; i < result.size(); ++i) {

      const auto node_state = result[i]->state;
      if (i > 0) {
        // I also have to compute a distance
        double delta_time = robot->cost_lower_bound(node_state, tmpState);
        cost_with_jumps += delta_time;
        out << space6 + "# delta time " << delta_time << std::endl;
        double delta_i = si->distance(node_state, tmpState);
        out << space6 + "# delta " << delta_i << std::endl;
        assert(delta_i <= options_dbastar_local.delta);
      }

      out << space6 + "# this is node " << std::endl;
      out << space6 + "# time " << cost_with_jumps << std::endl;
      out << space6 + "- ";
      printState(out, si, node_state);
      out << std::endl;
      times.push_back(cost_with_jumps);

      auto uu = si->allocControl();
      si->copyControl(uu, uzero);
      actions2.push_back(uu);

      out << space6 + "# action ";
      printAction(out, si, uu);
      out << std::endl;

      auto x_ = si->allocState();
      si->copyState(x_, node_state);
      states2.push_back(x_);

      si->copyState(tmpState, node_state);

      if (i < result.size() - 1) {
        fcl::Vector3d current_pos(0, 0, 0);
        if (robot->isTranslationInvariant())
          current_pos = robot->getTransform(node_state).translation();
        const auto &motion = motions.at(result.at(i + 1)->used_motion);
        out << space6 + "# motion " << motion.idx << " with cost "
            << motion.cost << std::endl;

        for (size_t k = 0; k < motion.states.size(); ++k) {
          ompl::control::Control *u;

          if (k < motion.states.size() - 1) {
            u = motion.actions.at(k);
          } else {
            u = uzero;
          }

          const auto state = motion.states.at(k);
          si->copyState(tmpState, state);

          if (robot->isTranslationInvariant()) {
            const fcl::Vector3d relative_pos =
                robot->getTransform(state).translation();
            robot->setPosition(tmpState, current_pos +
                                             result[i + 1]->used_offset +
                                             relative_pos);
          }

          if (k == 0) {
            double delta_time = robot->cost_lower_bound(node_state, tmpState);
            cost_with_jumps += delta_time;
            out << space6 + "# delta time " << delta_time << std::endl;
            double delta_i = si->distance(node_state, tmpState);
            out << space6 + "# delta " << delta_i << std::endl;
            assert(delta_i <= options_dbastar_local.delta);
          } else {
            cost_with_jumps += dt;
          }
          out << space6 + "# time " << cost_with_jumps << std::endl;
          out << space6 + "- ";
          printState(out, si, tmpState);
          out << std::endl;

          auto x_ = si->allocState();
          si->copyState(x_, tmpState);
          states2.push_back(x_);

          times.push_back(cost_with_jumps);
          auto uu = si->allocControl();
          si->copyControl(uu, u);
          actions2.push_back(uu);

          out << space6 + "# action ";
          printAction(out, si, uu);
          out << std::endl;
        }
      }
    }
    if (true) {
      double delta_time = robot->cost_lower_bound(tmpState, goalState);
      cost_with_jumps += delta_time;
      out << space6 + "# delta time " << delta_time << std::endl;
      double delta_i = si->distance(goalState, tmpState);
      std::cout << "delta_i " << delta_i << std::endl;
      assert(delta_i < options_dbastar_local.delta);
      out << space6 + "# delta " << delta_i << std::endl;
      out << space6 + "# this is goal " << std::endl;
      out << space6 + "# time " << cost_with_jumps << std::endl;
      out << space6 + "- ";
      printState(out, si, goalState);
      out << std::endl;

      auto x_ = si->allocState();
      si->copyState(x_, tmpState);
      states2.push_back(x_);
      times.push_back(cost_with_jumps);

      // auto uu = si->allocControl();
      // si->copyControl(uu, uzero);
      // actions2.push_back(uu);

    } else {
      out << space6 + "# last state is already the goal" << std::endl;
    }

    out << space4 + "actions:" << std::endl;
    for (auto &a : actions2) {
      out << space6 + "- ";
      printAction(out, si, a);
      out << std::endl;
    }

    out << space4 + "times:" << std::endl;
    for (auto &t : times) {
      out << space6 + "- " << t << std::endl;
    }

    std::cout << "times size " << times.size() << std::endl;
    std::cout << "actions size " << actions2.size() << std::endl;
    std::cout << "states size " << states2.size() << std::endl;

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
    out << space4 + "motion_stats:" << std::endl;
    for (const auto &kv : motionsCount) {
      out << space6 + "" << motions[kv.first].idx << ": " << kv.second
          << std::endl;
    }

    // statistics on where the motion splits are
    out << space4 + "splits:" << std::endl;
    for (size_t i = 0; i < result.size() - 1; ++i) {
      const auto &motion = motions.at(result[i + 1]->used_motion);
      out << space6 + "- " << motion.states.size() - 1 << std::endl;
    }

    out << space4 + "graph_nodes:" << std::endl;
    for (auto &r : result) {
      out << space6 + "- ";
      printState(out, si, r->state);
      out << std::endl;
    }
  }
  std::cout << "WARNING: I am using cost with jumps!" << std::endl;
  traj_out.cost = cost_with_jumps;

  out_info_db.solved = solution;

  if (out_info_db.solved) {
    out_info_db.cost = solution->gScore;
    out_info_db.cost_with_delta_time = cost_with_jumps;
  } else {

    out_info_db.cost = -1;
    out_info_db.cost_with_delta_time = -1;
  }
}

void Out_info_db::print(std::ostream &out) {

  std::string be = "";
  std::string af = ": ";

  out << be << STR(cost, af) << std::endl;
  out << be << STR(cost_with_delta_time, af) << std::endl;
  out << be << STR(solved, af) << std::endl;
}
