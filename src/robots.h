#pragma once

// OMPL headers
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

// FCL
#include <fcl/fcl.h>

class Robot
{
public:
  Robot()
  {
  }

  virtual void propagate(
    const ompl::base::State *start,
    const ompl::control::Control *control,
    const double duration,
    ompl::base::State *result) = 0;

  virtual fcl::Transform3f getTransform(
    const ompl::base::State *state,
    size_t part = 0) = 0;

  virtual void setPosition(ompl::base::State* state, const fcl::Vector3f position) = 0;

  virtual size_t numParts()
  {
    return 1;
  }

  std::shared_ptr<fcl::CollisionGeometryf> getCollisionGeometry(size_t part = 0)
  {
    return geom_[part];
  }

  std::shared_ptr<ompl::control::SpaceInformation> getSpaceInformation()
  {
    return si_;
  }

protected:
  std::vector<std::shared_ptr<fcl::CollisionGeometryf>> geom_;
  std::shared_ptr<ompl::control::SpaceInformation> si_;
};

// Factory Method
std::shared_ptr<Robot> create_robot(
  const std::string& robotType,
  const ompl::base::RealVectorBounds& positionBounds);
