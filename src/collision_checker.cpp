
#include "collision_checker.hpp"

namespace ob = ompl::base;

CollisionChecker::CollisionChecker() : tmp_state_(nullptr) {}

CollisionChecker::~CollisionChecker() {
  if (robot_ && tmp_state_) {
    auto si = robot_->getSpaceInformation();
    si->freeState(tmp_state_);
  }
}

void CollisionChecker::load(const std::string &filename) {
  if (robot_ && tmp_state_) {
    auto si = robot_->getSpaceInformation();
    si->freeState(tmp_state_);
  }

  YAML::Node env = YAML::LoadFile(filename);

  std::vector<fcl::CollisionObjectf *> obstacles;
  for (const auto &obs : env["environment"]["obstacles"]) {
    const auto &size = obs["size"];
    if (obs["type"].as<std::string>() == "box") {

      if (size.size() == 2) {
        std::shared_ptr<fcl::CollisionGeometryf> geom;
        geom.reset(
            new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0));
        const auto &center = obs["center"];
        auto co = new fcl::CollisionObjectf(geom);
        co->setTranslation(
            fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
        co->computeAABB();
        obstacles.push_back(co);
      } else {
        std::shared_ptr<fcl::CollisionGeometryf> geom;
        geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(),
                                 size[2].as<float>()));
        const auto &center = obs["center"];
        auto co = new fcl::CollisionObjectf(geom);
        co->setTranslation(fcl::Vector3f(center[0].as<float>(),
                                         center[1].as<float>(),
                                         center[2].as<float>()));
        co->computeAABB();
        obstacles.push_back(co);
      }
    } else if (obs["type"].as<std::string>() == "sphere") {

      if (size.size() == 2) {
        std::shared_ptr<fcl::CollisionGeometryf> geom;
        geom.reset(new fcl::Spheref(size[0].as<float>()));
        const auto &center = obs["center"];
        auto co = new fcl::CollisionObjectf(geom);
        co->setTranslation(
            fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
        co->computeAABB();
        obstacles.push_back(co);
      } else {
        std::shared_ptr<fcl::CollisionGeometryf> geom;
        geom.reset(new fcl::Spheref(size[0].as<float>()));
        const auto &center = obs["center"];
        auto co = new fcl::CollisionObjectf(geom);
        co->setTranslation(fcl::Vector3f(center[0].as<float>(),
                                         center[1].as<float>(),
                                         center[2].as<float>()));
        co->computeAABB();
        obstacles.push_back(co);
      }
    } else {
      throw std::runtime_error("Unknown obstacle type!");
    }
  }
  env_.reset(new fcl::DynamicAABBTreeCollisionManagerf());
  // std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env(new
  // fcl::NaiveCollisionManagerf());
  env_->registerObjects(obstacles);
  env_->setup();

  const auto &robot_node = env["robots"][0];
  auto robotType = robot_node["type"].as<std::string>();
  const auto &env_min = env["environment"]["min"];
  const auto &env_max = env["environment"]["max"];
  ob::RealVectorBounds position_bounds(env_min.size());
  for (size_t i = 0; i < env_min.size(); ++i) {
    position_bounds.setLow(i, env_min[i].as<double>());
    position_bounds.setHigh(i, env_max[i].as<double>());
  }
  robot_ = create_robot(robotType, position_bounds);

  auto si = robot_->getSpaceInformation();
  si->getStateSpace()->setup();

  tmp_state_ = si->allocState();
}

std::tuple<double, fcl::Vector3<float>, fcl::Vector3<float>>
CollisionChecker::distance(const std::vector<double> &state) {

  auto si = robot_->getSpaceInformation();
  si->getStateSpace()->copyFromReals(tmp_state_, state);

  std::vector<fcl::DefaultDistanceData<float>> distance_data(
      robot_->numParts());
  size_t min_idx = 0;
  for (size_t part = 0; part < robot_->numParts(); ++part) {
    const auto &transform = robot_->getTransform(tmp_state_, part);
    fcl::CollisionObjectf robot(
        robot_->getCollisionGeometry(part)); //, robot_->getTransform(state));
    robot.setTranslation(transform.translation());
    robot.setRotation(transform.rotation());
    robot.computeAABB();
    distance_data[part].request.enable_signed_distance = true;
    env_->distance(&robot, &distance_data[part],
                   fcl::DefaultDistanceFunction<float>);
    if (distance_data[part].result.min_distance <
        distance_data[min_idx].result.min_distance) {
      min_idx = part;
    }
  }

  return std::make_tuple(distance_data[min_idx].result.min_distance,
                         distance_data[min_idx].result.nearest_points[0],
                         distance_data[min_idx].result.nearest_points[1]);
}

// negative means collision
std::tuple<double, std::vector<double>>
CollisionChecker::distanceWithFDiffGradient(const std::vector<double> &state,
                                            double faraway_zero_gradient_bound,
                                            double epsilon,
                                            std::vector<bool> *non_zero_flags) {

  const auto dd = std::get<0>(distance(state));

  if (non_zero_flags) {
    assert(non_zero_flags->size() == state.size());
  }

  std::vector<double> out(state.size(), 0.);
  if (dd < faraway_zero_gradient_bound) {
    std::vector<double> eps(state.size(), 0.);
    for (size_t i = 0; i < state.size(); i++) {
      bool check_i = true;
      if (non_zero_flags)
        check_i = (*non_zero_flags)[i];
      if (check_i) {
        std::copy(state.begin(), state.end(), eps.begin());
        eps[i] += epsilon;
        auto dd_eps = std::get<0>(distance(eps));
        out[i] = (dd_eps - dd) / epsilon;
      }
    }
  }
  return std::make_tuple(dd, out);
}
