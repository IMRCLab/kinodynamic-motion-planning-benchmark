#pragma once

#include "robot.h"

class RobotDubinsCar : public Robot
{
public:
  RobotDubinsCar(
    const ompl::base::RealVectorBounds& position_bounds,
    float u_min, float u_max,
    float v, float k);

  void propagate(
    const ompl::base::State *start,
    const ompl::control::Control *control,
    const double duration,
    ompl::base::State *result) override;

  virtual fcl::Transform3f getTransform(
      const ompl::base::State *state) override;

  bool is2D() override;

protected:
  float v_; // fixed speed
  float k_; // curvature parameter
  float z_;
};