
#include <algorithm>
#include <boost/graph/graphviz.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

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
#include "ompl/base/ScopedState.h"
#include "robots.h"

#if 0


namespace ob = ompl::base;
namespace oc = ompl::control;

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
    return (a - b) * (a - b);
  }
};

// boost stuff for the graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/property_map/property_map.hpp>

void load_matrix(std::istream &in, std::vector<std::vector<double>> &out) {
  out.clear();

  std::string line;
  while (std::getline(in, line)) {
    std::istringstream iss(line);
    std::string token;
    std::vector<double> v;
    while (std::getline(iss, token, ' ')) {
      v.push_back(std::stod(token));
    }
    out.push_back(v);
  }
}

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
}

double PI2 = 2 * 3.14159;
double PI = 3.14159;
double distance(const double *query, const double *ref) {

  // // size_t n = query.size();
  // assert(query.size() == ref.size());
  // double out = 0;

  double d0 = query[0] - ref[0];
  double d1 = query[1] - ref[1];
  double d2 = query[2] - ref[2];
  double ad2 = std::fabs(d2);
  ad2 = (ad2 > PI) ? 2.0 * PI - ad2 : ad2;
  // return std::sqrt(d0 * d0 + d1 * d1) + .5 * ad2;
  return std::sqrt(d0 * d0 + d1 * d1 + d2 * d2); // for comparing againts TREE

  // for (size_t i = 0; i < n; i++) {
  //   double d = query[i] - ref[i];
  //   out += d * d;
  // }
  // return std::sqrt(out);
}

void search_nearest_R(const std::vector<double> &query,
                      const std::vector<std::vector<double>> &data,
                      // const std::vector<double> &data,
                      double radius, std::vector<int> &neighbors_n) {

  int N = query.size();

  // for (size_t i = 0; i < data.size() / N ; i++) {
  for (size_t i = 0; i < data.size(); i++) {
    double d = distance(query.data(), data[i].data());
    if (d < radius) {
      neighbors_n.push_back(i);
    }
  }
}

void search_nearest_R(const std::vector<double> &query,
                      const std::vector<double> &data, double radius,
                      std::vector<int> &neighbors_n) {

  int N = query.size();

  for (size_t i = 0; i < data.size() / N; i++) {
    double d = distance(query.data(), &data[i * N]);
    if (d < radius) {
      neighbors_n.push_back(i);
    }
  }
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
  // double d2 = s[2] * distance_angle(x[2], y[2]);
  double d2 = s[2] * (x[2] - y[2]);
  return d0 * d0 + d1 * d1 + d2 * d2;
};

void search_nearest_R(const std::vector<double> &query,
                      flann::Matrix<double> dataset, int buffer_index,
                      std::vector<int> &nn, double R) {

  double scale[3] = {1., 1., 1.};

  for (size_t i = 0; i < buffer_index; i++) {
    double d = distance_squared_se2(query.data(), dataset[i], scale);
    if (d < R) {
      nn.push_back(i);
    }
  }
}

// create a naive linear time! with hardcoded distance function?
int main(int argc, char *argv[]) {

  // check the code of OMPL. Does it make sense to have a buffer?

  // ANN with R^2, without approximation
  // data size:158194
  // matrix size: 38286 : 37601 : 685
  // total_time:239.183
  // time_add:96.6327
  // time_search:132.63

  // OMPL
  // data size:158194
  // num points:37795
  // total_time:1288.72
  // time_add:19.9948
  // time_search:1241.14

  // script to evaluate the perfomance

  namespace po = boost::program_options;
  int buffer_size;

  po::options_description desc("Allowed options");
  std::string inputFile;
  int nn;
  double radius = .1;
  int num_points;
  int rebuild_size;
  desc.add_options()("help", "produce help message")(
      "input,i", po::value<std::string>(&inputFile)->required(),
      "input file (txt)")("radius,r",
                          po::value<double>(&radius)->default_value(.1),
                          "radius of nn search)")(
      "nn,n", po::value<int>(&nn)->default_value(0),
      "0: GNAT-OMPL 1: LINEAR-OMPL  2: LINEAR-NO-OMPL 3: KD-tree")(
      "buffer_size,b", po::value<int>(&buffer_size)->default_value(10000))(
      "rebuild_size", po::value<int>(&rebuild_size)->default_value(10000000))(
      "num", po::value<int>(&num_points)->default_value(50000),
      "max num points to consider");

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

  std::ifstream in_file(inputFile);

  std::vector<std::vector<double>> data;
  load_matrix(in_file, data);

  if (num_points < data.size())
    data.erase(data.begin() + num_points, data.end());

  // what to do next?
  // simulate adding one by one and doing R check with different algorithms.

  ompl::NearestNeighbors<AStarNode *> *T_n;

  switch (nn) {
  case 0: {
    unsigned int degree = 8;
    unsigned int minDegree = 4;
    unsigned int maxDegree = 12;
    unsigned int maxNumPtsPerLeaf = 50;
    unsigned int removedCacheSize = 500;
    bool rebalancing = true;

    T_n = new ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *>(
        degree, minDegree, maxDegree, maxNumPtsPerLeaf, removedCacheSize,
        rebalancing);
    //
    // T_n = new ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *>();
  } break;
  //
  //
  //     // std::cout << "rebuild_size " << rebuild_size << std::endl;
  //   // dynamic_cast<ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *>
  //   // *>(T_n)
  //   //     ->rebuildSize_ = rebuild_size;
  //   break;
  case 1:
    T_n = new ompl::NearestNeighborsLinear<AStarNode *>();
    break;
  case 2:
    break;
  case 3:
    break;
  default:
    ERROR_WITH_INFO("choose valide nn");
  }

  double time_add = 0;
  double time_search = 0;

  // I have to create a state

  auto robotType = "unicycle_first_order_0";
  double min[] = {0.0, 0.0};
  double max[] = {6, 6};
  ob::RealVectorBounds position_bounds(2);
  for (size_t i = 0; i < 2; i++) {
    position_bounds.setLow(i, min[i]);
    position_bounds.setHigh(i, max[i]);
  }
  std::shared_ptr<RobotOmpl> robot =
      create_robot_ompl(robotType, position_bounds);

  auto si = robot->getSpaceInformation();

  // set number of control steps
  si->setPropagationStepSize(1);
  si->setMinMaxControlDuration(1, 1);

  std::shared_ptr<oc::StatePropagator> statePropagator(
      new RobotOmplStatePropagator(si, robot));
  si->setStatePropagator(statePropagator);

  si->setup();

  ob::State *state = si->allocState();

  assert(data.size());
  si->getStateSpace()->copyFromReals(state, data.at(0));

  auto start_node = new AStarNode();
  start_node->state = state;
  start_node->gScore = 0;

  auto tic = std::chrono::high_resolution_clock::now();
  std::cout << "data size:" << data.size() << std::endl;

  if (nn == 2) {
    std::cout << "nn:2" << std::endl;

    ob::State *state_a = si->allocState();
    si->getStateSpace()->copyFromReals(state_a, data.at(0));

    ob::State *state_b = si->allocState();
    si->getStateSpace()->copyFromReals(state_b, data.at(2));

    double d = si->distance(state_a, state_b);
    // distance is:
    // "sqrt( 0.1751 ^ 2 + (2.88853 - 3) ^ 2)  + .5  * .558536 "

    for (auto &v : data.at(0)) {
      std::cout << v << " ";
    }
    std::cout << std::endl;
    for (auto &v : data.at(2)) {
      std::cout << v << " ";
    }
    std::cout << std::endl;

    std::cout << "d " << d << std::endl;
    // return 0;

    // how much time to do nn one by one
    // std::vector<std::vector<double>> matrix;
    // matrix.push_back(data.at(0));

    std::vector<double> single_matrix;
    single_matrix.reserve(50000 * 3);
    single_matrix.insert(single_matrix.end(), data.at(0).begin(),
                         data.at(0).end());

    for (size_t i = 1; i < data.size(); i++) {

      auto d = data.at(i);
      std::vector<int> neighbors_n;
      auto _tic = std::chrono::high_resolution_clock::now();

      search_nearest_R(d, single_matrix, radius, neighbors_n);
      // search_nearest_R(d, matrix, radius, neighbors_n);

      auto _tac = std::chrono::high_resolution_clock::now();
      time_search +=
          std::chrono::duration<double, std::milli>(_tac - _tic).count();

      if (!neighbors_n.size()) {
        _tic = std::chrono::high_resolution_clock::now();
        single_matrix.insert(single_matrix.end(), data.at(i).begin(),
                             data.at(i).end());
        // matrix.push_back(d);
        _tac = std::chrono::high_resolution_clock::now();
        time_add +=
            std::chrono::duration<double, std::milli>(_tac - _tic).count();
      }
      // add to tree
    }

    // std::cout << "matrix size: " << matrix.size() << std::endl;
    std::cout << "matrix size: " << single_matrix.size() / data.at(0).size()
              << std::endl;
  }

  else if (nn == 3) {
    // use flann

    std::function<double(const double *, const double *, size_t)> distance =

        [](const double *x, const double *y, size_t n) {
          if (n != 3) {
            throw std::runtime_error("n should be 3");
          }
          // double s[3] = {1., 1., .5};
          double s[3] = {1., 1., 1.};
          return distance_squared_se2(x, y, s);
        };

    // built index for doing nearest neigh Q

    int dim = 3;

    std::vector<std::vector<double>> matrix;
    matrix.push_back(data.at(0));

    flann::Matrix<double> ref_data(new double[3], 1, dim);
    // // for (size_t i = 0; i < data.size(); i++) {
    // std::copy(data[0].begin(), data[0].end(), dataset[0]);
    // }

    int buffer_index = 0;
    std::vector<int> chosen_index;
    chosen_index.push_back(0);
    flann::Matrix<double> buffer(new double[dim * buffer_size], buffer_size,
                                 dim);

    std::copy(data[0].begin(), data[0].end(), ref_data[0]);
    // add to the buffer

    flann::KDTreeSingleIndex<L2Q<double>> *index_ptr;

    index_ptr = new flann::KDTreeSingleIndex<L2Q<double>>(
        ref_data, flann::KDTreeSingleIndexParams(), L2Q<double>(distance));

    index_ptr->buildIndex();

    double delta_sq = radius * radius;
    for (size_t i = 1; i < data.size(); i++) {

      auto d = data.at(i);

      flann::Matrix<double> new_dataset(new double[dim], 1, dim);

      std::copy(d.begin(), d.end(), new_dataset[0]);

      auto _tic = std::chrono::high_resolution_clock::now();

      std::vector<std::vector<int>> idxs;
      std::vector<std::vector<double>> v_dis;

      std::vector<int> nn;

      if (index_ptr->size())
        index_ptr->radiusSearch(new_dataset, idxs, v_dis, delta_sq,
                                flann::SearchParams());

      if (buffer_index) {
        search_nearest_R(d, buffer, buffer_index, nn, delta_sq);
      }

      assert(idxs.size());
      assert(v_dis.size());

      // for (auto &idx : idxs) {
      //
      //   for (auto &jj : idx) {
      //     std::cout << jj << " ";
      //   }
      //   std::cout << std::endl;
      // }
      //
      // std::cout << "vdis" << std::endl;
      // for (auto &vdi : v_dis) {
      //
      //   for (auto &ii : vdi) {
      //     std::cout << ii << " ";
      //   }
      //   std::cout << std::endl;
      // }

      auto _tac = std::chrono::high_resolution_clock::now();
      time_search +=
          std::chrono::duration<double, std::milli>(_tac - _tic).count();

      if (!idxs.at(0).size() && !nn.size()) {
        _tic = std::chrono::high_resolution_clock::now();

        chosen_index.push_back(i);

        if (buffer_index < buffer_size) {
          std::copy(d.begin(), d.end(), buffer[buffer_index]);
          buffer_index++;
        }
        if (buffer_index == buffer_size) {
          // std::cout << "rebuilding " << std::endl;

          delete ref_data.ptr();
          delete index_ptr;

          ref_data = flann::Matrix<double>(
              new double[dim * chosen_index.size()], chosen_index.size(), dim);

          for (size_t i = 0; i < chosen_index.size(); i++) {
            auto e = chosen_index.at(i);
            std::copy(data.at(e).begin(), data.at(e).end(), ref_data[i]);
          }

          // change the tree.
          index_ptr = new flann::KDTreeSingleIndex<L2Q<double>>(
              ref_data, flann::KDTreeSingleIndexParams(),
              L2Q<double>(distance));
          index_ptr->buildIndex();
          buffer_index = 0;
        }

        _tac = std::chrono::high_resolution_clock::now();
        time_add +=
            std::chrono::duration<double, std::milli>(_tac - _tic).count();
      } else {
        // std::cout << "index " << i << " is repeated " << std::endl;
      }

      delete new_dataset.ptr();

      // add to tree
    }
    // std::cout << "matrix size: " << matrix.size() << std::endl;
    std::cout << "matrix size: " << index_ptr->size() + buffer_index << " : "
              << index_ptr->size() << " : " << buffer_index << std::endl;
  }

  else {

    T_n->add(start_node);

    T_n->setDistanceFunction([si](const AStarNode *a, const AStarNode *b) {
      return si->distance(a->state, b->state);
    });

    int max_buffer_size = buffer_size;

    bool use_buffer = false;
    std::vector<AStarNode *> buffer;

    for (size_t i = 1; i < data.size(); i++) {

      auto node = new AStarNode();
      ob::State *state = si->allocState();
      si->getStateSpace()->copyFromReals(state, data.at(i));
      node->state = state;
      // si->printState(node->state);

      // do nn search
      std::vector<AStarNode *> neighbors_n;

      auto _tic = std::chrono::high_resolution_clock::now();
      T_n->nearestR(node, radius, neighbors_n);

      if (buffer.size()) {
        for (auto &b : buffer) {
          if (si->distance(b->state, node->state) < radius) {
            neighbors_n.push_back(b);
          }
        }
      }

      // auto nn_ = T_n->nearest(node);
      // std::cout << "nearest " ;
      // si->printState(nn_->state) ;
      // std::cout << "neighbors_n:" << neighbors_n.size() << std::endl;
      auto _tac = std::chrono::high_resolution_clock::now();
      time_search +=
          std::chrono::duration<double, std::milli>(_tac - _tic).count();

      if (!neighbors_n.size()) {
        _tic = std::chrono::high_resolution_clock::now();
        if (use_buffer) {
          if (buffer.size() < buffer_size) {
            buffer.push_back(node);
            if (buffer.size() == max_buffer_size) {
              T_n->add(buffer);
              dynamic_cast<
                  ompl::NearestNeighborsGNATNoThreadSafety<AStarNode *> *>(T_n)
                  ->rebuildDataStructure();
              buffer.clear();
            }
          }
        } else {
          T_n->add(node);
        }
        _tac = std::chrono::high_resolution_clock::now();
        time_add +=
            std::chrono::duration<double, std::milli>(_tac - _tic).count();
      }
      // add to tree
    }
    std::cout << "num points:" << T_n->size() + buffer.size() << std::endl;
  }

  auto tac = std::chrono::high_resolution_clock::now();
  double total_time =
      std::chrono::duration<double, std::milli>(tac - tic).count();

  std::cout << "total_time:" << total_time << std::endl;
  std::cout << "time_add:" << time_add << std::endl;
  std::cout << "time_search:" << time_search << std::endl;
}

#endif
