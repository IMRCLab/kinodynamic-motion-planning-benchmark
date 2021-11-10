#pragma once

#include "robot.h"

class RobotCarFirstOrder : public Robot
{
public:
  RobotCarFirstOrder(
    const ompl::base::RealVectorBounds& position_bounds,
    float w_limit,
    float v_limit);

  void propagate(
    const ompl::base::State *start,
    const ompl::control::Control *control,
    const double duration,
    ompl::base::State *result) override;

  virtual fcl::Transform3f getTransform(
      const ompl::base::State *state) override;

  // virtual void setPositionToZero(ompl::base::State *state) override;
  virtual void setPosition(ompl::base::State *state, const fcl::Vector3f position) override;
};