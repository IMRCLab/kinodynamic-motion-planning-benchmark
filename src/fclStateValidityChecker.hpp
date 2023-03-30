#pragma once

// #include "environment.h"
#include "robots.h"

#include <fcl/fcl.h>

#if 0 
class fclStateValidityChecker
  : public ompl::base::StateValidityChecker
{
public:
  fclStateValidityChecker(
      ompl::base::SpaceInformationPtr si,
      std::shared_ptr<fcl::BroadPhaseCollisionManagerd> environment,
      std::shared_ptr<RobotOmpl> robot)
      : StateValidityChecker(si)
      , environment_(environment)
      , robot_(robot)
  {
  }

  bool isValid(const ompl::base::State* state) const override
  {
    if (!si_->satisfiesBounds(state)) {
      return false;
    }

    for (size_t part = 0; part < robot_->numParts(); ++part) {
      const auto& transform = robot_->getTransform(state, part);
      fcl::CollisionObjectd robot(robot_->getCollisionGeometry(part)); //, robot_->getTransform(state));
      robot.setTranslation(transform.translation());
      robot.setRotation(transform.rotation());
      robot.computeAABB();
      fcl::DefaultCollisionData<double> collision_data;
      environment_->collide(&robot, &collision_data, fcl::DefaultCollisionFunction<float>);
      if (collision_data.result.isCollision()) {
        return false;
      }
    }

    return true;
  }

private:
  std::shared_ptr<fcl::BroadPhaseCollisionManagerf> environment_;
  std::shared_ptr<RobotOmpl> robot_;
};
#endif
