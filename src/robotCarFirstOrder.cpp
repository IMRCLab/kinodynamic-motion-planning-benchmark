#include "robotCarFirstOrder.h"

#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

RobotCarFirstOrder::RobotCarFirstOrder(
    const ompl::base::RealVectorBounds &position_bounds,
    float w_limit,
    float v_limit)
    : Robot()
{
  geom_.reset(new fcl::Boxf(0.5, 0.25, 1.0));

  auto space(std::make_shared<ob::SE2StateSpace>());
  space->setBounds(position_bounds);

  // create a control space
  // R^1: turning speed
  auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));

  // set the bounds for the control space
  ob::RealVectorBounds cbounds(2);
  cbounds.setLow(0, -v_limit);
  cbounds.setHigh(0, v_limit);
  cbounds.setLow(1,-w_limit);
  cbounds.setHigh(1,w_limit);

  cspace->setBounds(cbounds);

  // construct an instance of  space information from this control space
  si_ = std::make_shared<oc::SpaceInformation>(space, cspace);
}

void RobotCarFirstOrder::propagate(
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

    x += ctrl[0] * cosf(yaw) * dt;
    y += ctrl[0] * sinf(yaw) * dt;
    yaw += ctrl[1] * dt;

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

fcl::Transform3f RobotCarFirstOrder::getTransform(
    const ompl::base::State *state)
{
  auto stateTyped = state->as<ob::SE2StateSpace::StateType>();

  fcl::Transform3f result;
  result = Eigen::Translation<float, 3>(fcl::Vector3f(stateTyped->getX(), stateTyped->getY(), 0));
  float yaw = stateTyped->getYaw();
  result.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
  return result;
}

// void RobotCarFirstOrder::setPositionToZero(ompl::base::State *state)
// {
//   auto stateTyped = state->as<ob::SE2StateSpace::StateType>();
//   stateTyped->setX(0);
//   stateTyped->setY(0);
// }

void RobotCarFirstOrder::setPosition(ompl::base::State *state, const fcl::Vector3f position)
{
  auto stateTyped = state->as<ob::SE2StateSpace::StateType>();
  stateTyped->setX(position(0));
  stateTyped->setY(position(1));
}
