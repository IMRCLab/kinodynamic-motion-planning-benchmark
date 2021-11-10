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
    const ompl::base::State *state) = 0;

  // virtual void setPositionToZero(ompl::base::State* state) = 0;
  virtual void setPosition(ompl::base::State* state, const fcl::Vector3f position) = 0;

#if 0
  virtual bool is2D() = 0;

  virtual float stateActionCost(
     const ompl::base::State* /*state*/,
     const ompl::control::Control *control)
  {
    // default cost is just u^Tu

    // propagate cost
    unsigned int dim = si_->getControlSpace()->getDimension();

    const double* ctrl = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values;
    float sum = 0;
    for (unsigned int k = 0; k < dim; ++k) {
        sum += powf(ctrl[k], 2);
    }
    return sqrtf(sum);
  }
#endif

  std::shared_ptr<fcl::CollisionGeometryf> getCollisionGeometry()
  {
    return geom_;
  }

  std::shared_ptr<ompl::control::SpaceInformation> getSpaceInformation()
  {
    return si_;
  }

protected:
  std::shared_ptr<fcl::CollisionGeometryf> geom_;
  std::shared_ptr<ompl::control::SpaceInformation> si_;
};