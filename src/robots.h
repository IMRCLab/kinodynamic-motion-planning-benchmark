#pragma once

// OMPL headers
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

// FCL
#include <fcl/fcl.h>

class Robot {

public:
  Robot() {}

  virtual void propagate(const ompl::base::State *start,
                         const ompl::control::Control *control,
                         const double duration, ompl::base::State *result) = 0;

  virtual void geometric_interpolation(const ompl::base::State *from,
                                       const ompl::base::State *to, double t,
                                       ompl::base::State *out) {

    throw std::runtime_error("not implemented");
  }

  virtual fcl::Transform3f getTransform(const ompl::base::State *state,
                                        size_t part = 0) = 0;

  virtual void setPosition(ompl::base::State *state,
                           const fcl::Vector3f position) = 0;

  virtual size_t numParts() { return 1; }

  std::shared_ptr<fcl::CollisionGeometryf>
  getCollisionGeometry(size_t part = 0) {
    return geom_[part];
  }

  std::shared_ptr<ompl::control::SpaceInformation> getSpaceInformation() {
    return si_;
  }

  virtual double cost_lower_bound(const ompl::base::State *a,
                                  const ompl::base::State *b) const {
    throw std::runtime_error("not implemented");
  }

  float dt() const { return dt_; }

  float is2D() const { return is2D_; }

  float maxSpeed() const { return max_speed_; }

protected:
  std::vector<std::shared_ptr<fcl::CollisionGeometryf>> geom_;
  std::shared_ptr<ompl::control::SpaceInformation> si_;
  float dt_;
  bool is2D_;
  float max_speed_;

public:
  Eigen::VectorXd u_zero; // a default control
  Eigen::VectorXd x_ub;
  Eigen::VectorXd x_lb;
  Eigen::VectorXd u_ub;
  Eigen::VectorXd u_lb;
};

// Factory Method
std::shared_ptr<Robot>
create_robot(const std::string &robotType,
             const ompl::base::RealVectorBounds &positionBounds);
