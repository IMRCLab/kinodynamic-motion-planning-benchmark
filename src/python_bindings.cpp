#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// local
#include "robotCarFirstOrder.h"
#include "robotCarSecondOrder.h"
#include "robotStatePropagator.hpp"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ob = ompl::base;
namespace oc = ompl::control;

class RobotHelper
{
public:
  RobotHelper(const std::string& robotType)
  {
    if (robotType == "car_first_order_0")
    {
      ob::RealVectorBounds position_bounds(2);
      position_bounds.setLow(-2);
      position_bounds.setHigh(2);
      robot_.reset(new RobotCarFirstOrder(
          position_bounds,
          /*w_limit*/ 0.5 /*rad/s*/,
          /*v_limit*/ 0.5 /* m/s*/));
    }
    else if (robotType == "car_second_order_0")
    {
      ob::RealVectorBounds position_bounds(2);
      position_bounds.setLow(-2);
      position_bounds.setHigh(2);
      robot_.reset(new RobotCarSecondOrder(
          position_bounds,
          /*v_limit*/ 0.5 /*m/s*/,
          /*w_limit*/ 0.5 /*rad/s*/,
          /*a_limit*/ 2.0 /*m/s^2*/,
          /*w_dot_limit*/ 2.0 /*rad/s^2*/
          ));
    }
    else
    {
      throw std::runtime_error("Unknown robot type!");
    }

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->setup();
    state_sampler_ = si->allocStateSampler();

    tmp_state_a_ = si->allocState();
    tmp_state_b_ = si->allocState();
  }

  ~RobotHelper()
  {
    auto si = robot_->getSpaceInformation();
    si->freeState(tmp_state_a_);
    si->freeState(tmp_state_b_);
  }

  float distance(const std::vector<double> &stateA, const std::vector<double> &stateB)
  {
    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_a_, stateA);
    si->getStateSpace()->copyFromReals(tmp_state_b_, stateB);
    return si->distance(tmp_state_a_, tmp_state_b_);
  }

  std::vector<double> sampleUniform()
  {
    state_sampler_->sampleUniform(tmp_state_a_);
    auto si = robot_->getSpaceInformation();
    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, tmp_state_a_);
    return reals;
  }

private: 
  std::shared_ptr<Robot> robot_;
  ob::StateSamplerPtr state_sampler_; 
  ob::State* tmp_state_a_;
  ob::State* tmp_state_b_;
};

class CollisionChecker
{
public:
  CollisionChecker()
    : tmp_state_(nullptr)
  {

  }

  ~CollisionChecker()
  {
    if (robot_ && tmp_state_) {
      auto si = robot_->getSpaceInformation();
      si->freeState(tmp_state_);
    }
  }

  void load(const std::string& filename)
  {
    if (robot_ && tmp_state_) {
      auto si = robot_->getSpaceInformation();
      si->freeState(tmp_state_);
    }

    YAML::Node env = YAML::LoadFile(filename);

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
    env_.reset(new fcl::DynamicAABBTreeCollisionManagerf());
    // std::shared_ptr<fcl::BroadPhaseCollisionManagerf> bpcm_env(new fcl::NaiveCollisionManagerf());
    env_->registerObjects(obstacles);
    env_->setup();

    const auto &robot_node = env["robots"][0];
    auto robotType = robot_node["type"].as<std::string>();
    if (robotType == "car_first_order_0")
    {
      ob::RealVectorBounds position_bounds(2);
      const auto &dims = env["environment"]["dimensions"];
      position_bounds.setLow(0);
      position_bounds.setHigh(0, dims[0].as<double>());
      position_bounds.setHigh(1, dims[1].as<double>());

      robot_.reset(new RobotCarFirstOrder(
          position_bounds,
          /*w_limit*/ 0.5 /*rad/s*/,
          /*v_limit*/ 0.5 /* m/s*/));
    }
    else if (robotType == "car_second_order_0")
    {
      ob::RealVectorBounds position_bounds(2);
      const auto &dims = env["environment"]["dimensions"];
      position_bounds.setLow(0);
      position_bounds.setHigh(0, dims[0].as<double>());
      position_bounds.setHigh(1, dims[1].as<double>());

      robot_.reset(new RobotCarSecondOrder(
          position_bounds,
          /*v_limit*/ 0.5 /*m/s*/,
          /*w_limit*/ 0.5 /*rad/s*/,
          /*a_limit*/ 2.0 /*m/s^2*/,
          /*w_dot_limit*/ 2.0 /*rad/s^2*/
          ));
    }
    else
    {
      throw std::runtime_error("Unknown robot type!");
    }

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->setup();

    tmp_state_ = si->allocState();
  }

  auto distance(const std::vector<double>& state)
  {

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_, state);

    const auto &transform = robot_->getTransform(tmp_state_);
    fcl::CollisionObjectf robot(robot_->getCollisionGeometry()); //, robot_->getTransform(state));
    robot.setTranslation(transform.translation());
    robot.setRotation(transform.rotation());
    fcl::DefaultDistanceData<float> distance_data;
    env_->distance(&robot, &distance_data, fcl::DefaultDistanceFunction<float>);

    return std::make_tuple(
      distance_data.result.min_distance,
      distance_data.result.nearest_points[0],
      distance_data.result.nearest_points[1]);
  }

private:
  std::shared_ptr<fcl::CollisionGeometryf> geom_;
  std::shared_ptr<fcl::BroadPhaseCollisionManagerf> env_;
  std::shared_ptr<Robot> robot_;
  ob::State *tmp_state_;
};

PYBIND11_MODULE(motionplanningutils, m)
{
  pybind11::class_<CollisionChecker>(m, "CollisionChecker")
      .def(pybind11::init())
      .def("load", &CollisionChecker::load)
      .def("distance", &CollisionChecker::distance);

  pybind11::class_<RobotHelper>(m, "RobotHelper")
      .def(pybind11::init<const std::string &>())
      .def("distance", &RobotHelper::distance)
      .def("sampleUniform", &RobotHelper::sampleUniform);
}
