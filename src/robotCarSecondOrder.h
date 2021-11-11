#pragma once

#include "robot.h"

class RobotCarSecondOrder : public Robot
{
public:
  RobotCarSecondOrder(
    const ompl::base::RealVectorBounds& position_bounds,
    float v_limit, // max velocity in m/s
    float w_limit, // max angular velocity in rad/s
    float a_limit, // max accelleration in m/s^2
    float w_dot_limit); // max angular acceleration in rad/s^2

  void propagate(
    const ompl::base::State *start,
    const ompl::control::Control *control,
    const double duration,
    ompl::base::State *result) override;

  virtual fcl::Transform3f getTransform(
    const ompl::base::State *state) override;

  virtual void setPosition(
    ompl::base::State *state,
    const fcl::Vector3f position) override;
};