// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// OMPL
#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>
#include <ompl/datastructures/NearestNeighborsSqrtApprox.h>

// local
#include "robotStatePropagator.hpp"
#include "robots.h"

#include "fcl/common/types.h"

class CollisionChecker {
public:
  CollisionChecker();

  ~CollisionChecker();
  void load(const std::string &filename);

  // TODO: use double precission in collisions

  // negative means collision
  std::tuple<double, fcl::Vector3<float>, fcl::Vector3<float>>
  distance(const std::vector<double> &state);
  std::tuple<double, std::vector<double>>
  distanceWithFDiffGradient(const std::vector<double> &state, 
double faraway_zero_gradient_bound = .1, double epsilon = 1e-4, 
std::vector<bool>* non_zero_flags = nullptr);

private:
  std::shared_ptr<fcl::CollisionGeometryf> geom_;
  std::shared_ptr<fcl::BroadPhaseCollisionManagerf> env_;
  std::shared_ptr<Robot> robot_;
  ompl::base::State *tmp_state_;
};
