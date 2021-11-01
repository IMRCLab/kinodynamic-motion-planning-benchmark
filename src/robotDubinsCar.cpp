#include "robotDubinsCar.h"

#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

RobotDubinsCar::RobotDubinsCar(
    const ompl::base::RealVectorBounds &position_bounds,
    float w_limit,
    float v)
    : Robot(), v_(v), w_limit_(w_limit)
{
  geom_.reset(new fcl::Boxf(0.5, 0.25, 1.0));

  auto space(std::make_shared<ob::SE2StateSpace>());
  space->setBounds(position_bounds);

  // create a control space
  // R^1: turning speed
  auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 1));

  // set the bounds for the control space
  ob::RealVectorBounds cbounds(1);
  cbounds.setLow(-w_limit_);
  cbounds.setHigh(w_limit_);

  cspace->setBounds(cbounds);

  // construct an instance of  space information from this control space
  si_ = std::make_shared<oc::SpaceInformation>(space, cspace);
}

void RobotDubinsCar::propagate(
  const ompl::base::State *start,
  const ompl::control::Control *control,
  const double duration,
  ompl::base::State *result)
{
  auto startTyped = start->as<ob::SE2StateSpace::StateType>();
  const double* ctrl = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values;

  auto resultTyped= result->as<ob::SE2StateSpace::StateType>();

  // use simple Euler integration
  float x = startTyped->getX();
  float y = startTyped->getY();
  float yaw = startTyped->getYaw();
  float remaining_time = duration;
  const float integration_dt = 0.1f;
  do {
    float dt = std::min(remaining_time, integration_dt);

    x += v_ * cosf(yaw) * dt;
    y += v_ * sinf(yaw) * dt;
    yaw += ctrl[0] * dt;

    remaining_time -= dt;
  } while (remaining_time >= integration_dt);

    // update result

  resultTyped->setX(x);
  resultTyped->setY(y);
  resultTyped->setYaw(yaw);

  // Normalize orientation
  ob::SO2StateSpace SO2;
  SO2.enforceBounds(resultTyped->as<ob::SO2StateSpace::StateType>(1));
}

fcl::Transform3f RobotDubinsCar::getTransform(
    const ompl::base::State *state)
{
  auto stateTyped = state->as<ob::SE2StateSpace::StateType>();

  fcl::Transform3f result;
  result = Eigen::Translation<float, 3>(fcl::Vector3f(stateTyped->getX(), stateTyped->getY(), 0));
  float yaw = stateTyped->getYaw();
  result.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
  return result;
}
