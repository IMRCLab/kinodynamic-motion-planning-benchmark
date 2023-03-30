#pragma once

#include "ompl/control/StatePropagator.h"
#include "robots.h"

class RobotOmplStatePropagator : public ompl::control::StatePropagator
{
public:
    RobotOmplStatePropagator(
        const ompl::control::SpaceInformationPtr& si,
        std::shared_ptr<RobotOmpl> robot)
        : ompl::control::StatePropagator(si)
        , robot_(robot)
    {
    }

    ~RobotOmplStatePropagator() override = default;

    void propagate(
        const ompl::base::State *state,
        const ompl::control::Control *control,
        double duration,
        ompl::base::State *result) const override
    {
        // propagate state
        robot_->propagate(state, control, duration, result);
    }

    bool canPropagateBackward() const override
    {
        return false;
    }

    bool canSteer() const override
    {
        return false;
    }

protected:
    std::shared_ptr<RobotOmpl> robot_;
};
