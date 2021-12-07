#include "AO_RRT.h"

#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>

/////////////////////////////////////////////////////////////////////////
class CostStateSpace : public ompl::base::CompoundStateSpace
{
public:
  class StateType : public ompl::base::CompoundStateSpace::StateType
  {
  public:
    StateType() = default;

    const ompl::base::State *state() const
    {
      return as<ompl::base::State>(0);
    }

    ompl::base::State *state()
    {
      return as<ompl::base::State>(0);
    }

    double getCost() const
    {
      return as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];
    }

    void setCost(double cost)
    {
      as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0] = cost;
    }
  };

  CostStateSpace(const ompl::base::StateSpacePtr &stateSpace, double costWeight)
  {
    setName("Cost" + getName());
    type_ = ompl::base::STATE_SPACE_TYPE_COUNT + 0;
    addSubspace(stateSpace, 1.0);
    addSubspace(std::make_shared<ompl::base::RealVectorStateSpace>(1), costWeight);
    lock();
  }

  ~CostStateSpace() override = default;

  const ompl::base::StateSpace *getStateSpace() const
  {
    return as<ompl::base::StateSpace>(0);
  }

  ompl::base::StateSpace *getStateSpace()
  {
    return as<ompl::base::StateSpace>(0);
  }

  void setCostBound(double maxCost)
  {
    as<ompl::base::RealVectorStateSpace>(1)->setBounds(0, maxCost);
  }

  double getCostBound() const
  {
    return as<ompl::base::RealVectorStateSpace>(1)->getBounds().high[1];
  }

  void setCostWeight(double weight)
  {
    setSubspaceWeight(1, weight);
  }

  double getCostWeight() const
  {
    return getSubspaceWeight(1);
  }

  ompl::base::State *allocState() const override
  {
    auto *state = new StateType();
    allocStateComponents(state);
    return state;
  }

  void freeState(ompl::base::State *state) const override
  {
    CompoundStateSpace::freeState(state);
  }

  // void registerProjections() override;
};
/////////////////////////////////////////////////////////////////////////

class RRT_Internal : public ompl::control::RRT
{
public:
  /** \brief Constructor */
  RRT_Internal(const ompl::control::SpaceInformationPtr &si)
    : RRT(si)
  {
  }

  void removeStatesAboveCost(double maxCost)
  {
    // std::cout << "start cpy" << std::endl;
    std::vector<Motion *> motions;
    if (nn_)
    {
      nn_->list(motions);

      // removing is actually very slow; instead, clear NN data
      // structure and rebuilt
      nn_->clear();

      // std::cout << "cpy'ed " << motions.size() << std::endl;

      for (auto m : motions)
      {
        double cost = m->state->as<CostStateSpace::StateType>()->getCost();
        if (cost >= maxCost)
        {
          // std::cout << "rm w/ cost :" << cost << std::endl;
          si_->freeState(m->state);
          siC_->freeControl(m->control);
          delete m;
        }
        else
        {
          nn_->add(m);
        }
      }
    }
    lastGoalMotion_ = nullptr;
  }
};

/////////////////////////////////////////////////////////////////////////

class CostStateValidityChecker
    : public ompl::base::StateValidityChecker
{
public:
  CostStateValidityChecker(
      ompl::base::SpaceInformationPtr si,
      ompl::base::StateValidityCheckerPtr checker)
      : StateValidityChecker(si), checker_(checker)
  {
  }

  bool isValid(const ompl::base::State *state) const override
  {
    // This checks for the cost limit
    if (!si_->satisfiesBounds(state))
    {
      // double cost = state->as<CostStateSpace::StateType>()->getCost();
      // std::cout << "cost violated: " << cost << std::endl;
      return false;
    }

    auto stateTyped = state->as<CostStateSpace::StateType>();
    return checker_->isValid(stateTyped->state());
  }

private:
  ompl::base::StateValidityCheckerPtr checker_;
};

class CostStatePropagator : public ompl::control::StatePropagator
{
public:
  CostStatePropagator(
      const ompl::control::SpaceInformationPtr &si,
      const ompl::control::StatePropagatorPtr& prop)
      : ompl::control::StatePropagator(si), prop_(prop)
  {
  }

  void propagate(
      const ompl::base::State *state,
      const ompl::control::Control *control,
      double duration,
      ompl::base::State *result) const override
  {
    auto stateCS = state->as<CostStateSpace::StateType>();
    auto resultCS = result->as<CostStateSpace::StateType>();

    // propagate actual state
    prop_->propagate(stateCS->state(), control, duration, resultCS->state());

    // propagate cost
    double actionCost = duration;
    resultCS->setCost(stateCS->getCost() + actionCost);
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
  ompl::control::StatePropagatorPtr prop_;
};

/////////////////////////////////////////////////////////////////////////

class AO_RRT_Implementation
{
public:
  AO_RRT_Implementation(const ompl::control::SpaceInformationPtr &si)
    : rrt(si)
  {
  }

  RRT_Internal rrt;
};

/////////////////////////////////////////////////////////////////////////

AO_RRT::AO_RRT(const ompl::control::SpaceInformationPtr &si)
    : ompl::base::Planner(
          std::make_shared<ompl::control::SpaceInformation>(
              std::make_shared<CostStateSpace>(si->getStateSpace(), 0.01),
              si->getControlSpace()), "AO_RRT")
{
  auto siC = std::dynamic_pointer_cast<ompl::control::SpaceInformation>(si_);

  // copy some important settings from other statespace
  siC->setPropagationStepSize(si->getPropagationStepSize());
  siC->setMinMaxControlDuration(si->getMinControlDuration(), si->getMaxControlDuration());

  // set state validity checking for this space
  auto stateValidityChecker(std::make_shared<CostStateValidityChecker>(siC, si->getStateValidityChecker()));
  siC->setStateValidityChecker(stateValidityChecker);

  // set the state propagator
  std::shared_ptr<ompl::control::StatePropagator> statePropagator(new CostStatePropagator(siC, si->getStatePropagator()));
  siC->setStatePropagator(statePropagator);

  auto css = std::dynamic_pointer_cast<CostStateSpace>(si_->getStateSpace());
  css->setCostBound(1e6); // no cost bound for the first iteration

  siC->setup();

  impl_ = new AO_RRT_Implementation(siC);
}

AO_RRT::~AO_RRT()
{
  delete impl_;
}

ompl::base::PlannerStatus AO_RRT::solve(const ompl::base::PlannerTerminationCondition &ptc)
{
  auto css = std::dynamic_pointer_cast<CostStateSpace>(si_->getStateSpace());
  css->setCostBound(1e6); // no cost bound for the first iteration

  return impl_->rrt.solve(ptc);
}

void AO_RRT::clear()
{
  impl_->rrt.clear();
}

void AO_RRT::setup()
{
  impl_->rrt.setProblemDefinition(pdef_);
  impl_->rrt.setup();
}

ompl::control::RRT &AO_RRT::getLowLevelPlanner()
{
  return impl_->rrt;
}
