#pragma once

#include "robot.h"

class RobotDubinsCar : public Robot
{
public:
  RobotDubinsCar(
    const ompl::base::RealVectorBounds& position_bounds,
    float w_limit,
    float v);

  void propagate(
    const ompl::base::State *start,
    const ompl::control::Control *control,
    const double duration,
    ompl::base::State *result) override;

  virtual fcl::Transform3f getTransform(
      const ompl::base::State *state) override;

protected:
  float v_; // fixed speed
  float w_limit_; // maximum magnitude of angular velocity control
};