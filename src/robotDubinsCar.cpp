#include "robotDubinsCar.h"

#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

RobotDubinsCar::RobotDubinsCar(
    const ompl::base::RealVectorBounds &position_bounds,
    float u_min, float u_max,
    float v, float k)
    : Robot(), v_(v), k_(k)
{
  geom_.reset(new fcl::Boxf(0.5, 0.25, 1.0));

  auto space(std::make_shared<ob::SE2StateSpace>());
  space->setBounds(position_bounds);

  // create a control space
  // R^1: turning speed
  auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 1));

  // set the bounds for the control space
  ob::RealVectorBounds cbounds(1);
  cbounds.setLow(u_min);
  cbounds.setHigh(u_max);

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

  // update position
  resultTyped->setX(startTyped->getX() + v_ * cosf(startTyped->getYaw()) * duration);
  resultTyped->setY(startTyped->getY() + v_ * sinf(startTyped->getYaw()) * duration);

  // update yaw
  resultTyped->setYaw(startTyped->getYaw() + k_ * ctrl[0] * duration);

  // Normalize orientation between 0 and 2*pi
  ob::SO2StateSpace SO2;
  SO2.enforceBounds(resultTyped->as<ob::SO2StateSpace::StateType>(1));
}

fcl::Transform3f RobotDubinsCar::getTransform(
    const ompl::base::State *state)
{
  auto stateTyped = state->as<ob::SE2StateSpace::StateType>();

  fcl::Transform3f result;
  float yaw = stateTyped->getYaw();
  result.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
  result.translation() = fcl::Vector3f(stateTyped->getX(), stateTyped->getY(), 0);
  return result;
}

bool RobotDubinsCar::is2D()
{
  return true;
}