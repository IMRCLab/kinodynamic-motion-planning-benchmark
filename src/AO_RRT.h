#pragma once

#include <ompl/control/planners/rrt/RRT.h>

class AO_RRT_Implementation;

class AO_RRT : public ompl::base::Planner
{
public:
    /** \brief Constructor */
    AO_RRT(const ompl::control::SpaceInformationPtr &si);

    ~AO_RRT() override;

    /** \brief Continue solving for some amount of time. Return true if solution was found. */
    ompl::base::PlannerStatus solve(const ompl::base::PlannerTerminationCondition &ptc) override;

    /** \brief Clear datastructures. Call this function if the
                input data to the planner has changed and you do not
                want to continue planning */
    void clear() override;

    void setup() override;

    ompl::control::RRT& getLowLevelPlanner();

protected:
    AO_RRT_Implementation* impl_;
    // ompl::control::RRT rrt_;
};

// class AO_RRT : public ompl::control::RRT
// {
// public:
//     /** \brief Constructor */
//     AO_RRT(const ompl::control::SpaceInformationPtr &si);

//     ~AO_RRT() override;

// protected:
//     void removeStatesAboveCost(double maxCost);
// };
