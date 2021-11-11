#include "robotCarSecondOrder.h"

#include <ompl/control/spaces/RealVectorControlSpace.h>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <ompl/tools/config/MagicConstants.h>

namespace ob = ompl::base;
namespace oc = ompl::control;

class StateSpace : public ob::CompoundStateSpace
{
public:
  class StateType : public ob::CompoundStateSpace::StateType
  {
  public:
    StateType() = default;

    double getX() const
    {
      return as<ob::RealVectorStateSpace::StateType>(0)->values[0];
    }

    double getY() const
    {
      return as<ob::RealVectorStateSpace::StateType>(0)->values[1];
    }

    double getYaw() const
    {
      return as<ob::SO2StateSpace::StateType>(1)->value;
    }

    double getVelocity() const
    {
      return as<ob::RealVectorStateSpace::StateType>(2)->values[0];
    }

    double getAngularVelocity() const
    {
      return as<ob::RealVectorStateSpace::StateType>(3)->values[0];
    }

    void setX(double x)
    {
      as<ob::RealVectorStateSpace::StateType>(0)->values[0] = x;
    }

    void setY(double y)
    {
      as<ob::RealVectorStateSpace::StateType>(0)->values[1] = y;
    }

    void setYaw(double yaw)
    {
      as<ob::SO2StateSpace::StateType>(1)->value = yaw;
    }

    void setVelocity(double velocity)
    {
      as<ob::RealVectorStateSpace::StateType>(2)->values[0] = velocity;
    }

    void setAngularVelocity(double angularVelocity)
    {
      as<ob::RealVectorStateSpace::StateType>(3)->values[0] = angularVelocity;
    }
  };

  StateSpace()
  {
    setName("CarSO" + getName());
    type_ = ob::STATE_SPACE_TYPE_COUNT + 0;
    addSubspace(std::make_shared<ob::RealVectorStateSpace>(2), 1.0); // position
    addSubspace(std::make_shared<ob::SO2StateSpace>(), 0.5);         // orientation
    addSubspace(std::make_shared<ob::RealVectorStateSpace>(1), 0.25); // velocity
    addSubspace(std::make_shared<ob::RealVectorStateSpace>(1), 0.25); // angular velocity
    lock();
  }

  ~StateSpace() override = default;

  void setPositionBounds(const ob::RealVectorBounds &bounds)
  {
    as<ob::RealVectorStateSpace>(0)->setBounds(bounds);
  }

  const ob::RealVectorBounds &getPositionBounds() const
  {
    return as<ob::RealVectorStateSpace>(0)->getBounds();
  }

  void setVelocityBounds(const ob::RealVectorBounds &bounds)
  {
    as<ob::RealVectorStateSpace>(2)->setBounds(bounds);
  }

  const ob::RealVectorBounds &getVelocityBounds() const
  {
    return as<ob::RealVectorStateSpace>(2)->getBounds();
  }

  void setAngularVelocityBounds(const ob::RealVectorBounds &bounds)
  {
    as<ob::RealVectorStateSpace>(3)->setBounds(bounds);
  }

  const ob::RealVectorBounds &getAngularVelocityBounds() const
  {
    return as<ob::RealVectorStateSpace>(3)->getBounds();
  }

  ob::State *allocState() const override
  {
    auto *state = new StateType();
    allocStateComponents(state);
    return state;
  }

  void freeState(ob::State *state) const override
  {
    CompoundStateSpace::freeState(state);
  }

  void registerProjections() override
  {
    class DefaultProjection : public ob::ProjectionEvaluator
    {
    public:
      DefaultProjection(const ob::StateSpace *space) : ob::ProjectionEvaluator(space)
      {
      }

      unsigned int getDimension() const override
      {
        return 2;
      }

      void defaultCellSizes() override
      {
        cellSizes_.resize(2);
        bounds_ = space_->as<ob::SE2StateSpace>()->getBounds();
        cellSizes_[0] = (bounds_.high[0] - bounds_.low[0]) / ompl::magic::PROJECTION_DIMENSION_SPLITS;
        cellSizes_[1] = (bounds_.high[1] - bounds_.low[1]) / ompl::magic::PROJECTION_DIMENSION_SPLITS;
      }

      void project(const ob::State *state, Eigen::Ref<Eigen::VectorXd> projection) const override
      {
        projection = Eigen::Map<const Eigen::VectorXd>(
            state->as<ob::SE2StateSpace::StateType>()->as<ob::RealVectorStateSpace::StateType>(0)->values, 2);
      }
    };

    registerDefaultProjection(std::make_shared<DefaultProjection>(this));
  }
};

RobotCarSecondOrder::RobotCarSecondOrder(
    const ompl::base::RealVectorBounds &position_bounds,
    float v_limit,      // max velocity in m/s
    float w_limit,      // max angular velocity in rad/s
    float a_limit,      // max accelleration in m/s^2
    float w_dot_limit) // max angular acceleration in rad/s^2
    : Robot()
{
  geom_.reset(new fcl::Boxf(0.5, 0.25, 1.0));

  auto space(std::make_shared<StateSpace>());
  space->setPositionBounds(position_bounds);

  ob::RealVectorBounds vel_bounds(1);
  vel_bounds.setLow(-v_limit);
  vel_bounds.setHigh(v_limit);
  space->setVelocityBounds(vel_bounds);

  ob::RealVectorBounds w_bounds(1);
  w_bounds.setLow(-w_limit);
  w_bounds.setHigh(w_limit);
  space->setAngularVelocityBounds(w_bounds);

  // create a control space
  // R^1: turning speed
  auto cspace(std::make_shared<oc::RealVectorControlSpace>(space, 2));

  // set the bounds for the control space
  ob::RealVectorBounds cbounds(2);
  cbounds.setLow(0, -a_limit);
  cbounds.setHigh(0, a_limit);
  cbounds.setLow(1,-w_dot_limit);
  cbounds.setHigh(1,w_dot_limit);

  cspace->setBounds(cbounds);

  // construct an instance of  space information from this control space
  si_ = std::make_shared<oc::SpaceInformation>(space, cspace);
}

void RobotCarSecondOrder::propagate(
  const ompl::base::State *start,
  const ompl::control::Control *control,
  const double duration,
  ompl::base::State *result)
{
  auto startTyped = start->as<StateSpace::StateType>();
  const double* ctrl = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values;

  auto resultTyped= result->as<StateSpace::StateType>();

  // use simple Euler integration
  float x = startTyped->getX();
  float y = startTyped->getY();
  float yaw = startTyped->getYaw();
  float v = startTyped->getVelocity();
  float w = startTyped->getAngularVelocity();
  float remaining_time = duration;
  const float integration_dt = 0.1f;
  do {
    float dt = std::min(remaining_time, integration_dt);

    x += v * cosf(yaw) * dt;
    y += v * sinf(yaw) * dt;
    yaw += w * dt;
    v += ctrl[0] * dt;
    w += ctrl[1] * dt;

    remaining_time -= dt;
  } while (remaining_time >= integration_dt);

    // update result

  resultTyped->setX(x);
  resultTyped->setY(y);
  resultTyped->setYaw(yaw);
  resultTyped->setVelocity(v);
  resultTyped->setAngularVelocity(w);

  // Normalize orientation
  ob::SO2StateSpace SO2;
  SO2.enforceBounds(resultTyped->as<ob::SO2StateSpace::StateType>(1));
}

fcl::Transform3f RobotCarSecondOrder::getTransform(
    const ompl::base::State *state)
{
  auto stateTyped = state->as<StateSpace::StateType>();

  fcl::Transform3f result;
  result = Eigen::Translation<float, 3>(fcl::Vector3f(stateTyped->getX(), stateTyped->getY(), 0));
  float yaw = stateTyped->getYaw();
  result.rotate(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()));
  return result;
}

void RobotCarSecondOrder::setPosition(ompl::base::State *state, const fcl::Vector3f position)
{
  auto stateTyped = state->as<ob::SE2StateSpace::StateType>();
  stateTyped->setX(position(0));
  stateTyped->setY(position(1));
}
