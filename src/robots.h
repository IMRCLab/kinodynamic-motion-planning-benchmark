#pragma once

// OMPL headers
#include "croco_macros.hpp"
#include "robot_models.hpp"
#include "motions.hpp"



#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

// FCL
#include <fcl/fcl.h>
#include <yaml-cpp/node/node.h>

struct Model_robot;

ompl::base::State *
_allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si,
                   const std::vector<double> &reals);

ompl::base::State *
allocAndFillState(std::shared_ptr<ompl::control::SpaceInformation> si,
                  const YAML::Node &node);

void copyFromRealsControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                          ompl::control::Control *out,
                          const std::vector<double> &reals);

ompl::control::Control *
allocAndFillControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                    const YAML::Node &node);

void copyToRealsControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                        ompl::control::Control *action,
                        std::vector<double> &reals);

void copyFromRealsControl(std::shared_ptr<ompl::control::SpaceInformation> si,
                          ompl::control::Control *out,
                          const std::vector<double> &reals);

void state_to_stream(std::ostream &out,
                     std::shared_ptr<ompl::control::SpaceInformation> si,
                     const ompl::base::State *state);

void state_to_eigen(Eigen::VectorXd &out,
                    std::shared_ptr<ompl::control::SpaceInformation> si,
                    const ompl::base::State *state);

void control_to_eigen(Eigen::VectorXd &out,
                      std::shared_ptr<ompl::control::SpaceInformation> si,
                      ompl::control::Control *control);

void control_from_eigen(const Eigen::VectorXd &out,
                        std::shared_ptr<ompl::control::SpaceInformation> si,
                        ompl::control::Control *control);

struct RobotOmpl {

  std::shared_ptr<Model_robot> diff_model;
  RobotOmpl(std::shared_ptr<Model_robot> diff_model);

  virtual ~RobotOmpl() {
    // TODO: erase state and goal!
  }

  // Eigen wrappers...

  size_t nx; // dim state
  size_t nu; // dim control

  Eigen::VectorXd xx; // data
  Eigen::VectorXd zz; // data
  Eigen::VectorXd yy; // data
  Eigen::VectorXd uu; // data

  virtual void geometric_interpolation(const ompl::base::State *from,
                                       const ompl::base::State *to, double t,
                                       ompl::base::State *out);

  virtual void propagate(const ompl::base::State *start,
                         const ompl::control::Control *control,
                         const double duration, ompl::base::State *result);

  virtual double cost_lower_bound(const ompl::base::State *x,
                                  const ompl::base::State *y);

  virtual void toEigen(const ompl::base::State *x_ompl,
                       Eigen::Ref<Eigen::VectorXd> x_eigen) {
    (void)x_ompl;
    (void)x_eigen;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void toEigenU(const ompl::control::Control *control,
                        Eigen::Ref<Eigen::VectorXd> u_eigen) {
    (void)control;
    (void)u_eigen;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void fromEigen(ompl::base::State *x_ompl,
                         const Eigen::Ref<const Eigen::VectorXd> &x_eigen) {
    (void)x_ompl;
    (void)x_eigen;
    ERROR_WITH_INFO("not implemented");
  }

  virtual fcl::Transform3d getTransform(const ompl::base::State *state,
                                        size_t part = 0) = 0;

  virtual void setPosition(ompl::base::State *state,
                           const fcl::Vector3d position) = 0;

  virtual size_t numParts() { return 1; }

  std::shared_ptr<ompl::control::SpaceInformation> getSpaceInformation() {
    return si_;
  }

  virtual void enforceBounds(ompl::base::State *) const {
    ERROR_WITH_INFO(" not implemented ");
  }

  double dt() const;

  bool is2D() const;

  bool isTranslationInvariant() const { return translation_invariant_; }

  std::string getName() const;

public:
  std::shared_ptr<ompl::control::SpaceInformation> si_;
  bool translation_invariant_ = true;
  std::string name_;
  Eigen::VectorXd u_zero;
  Eigen::VectorXd x_lb;
  Eigen::VectorXd x_ub;
  ompl::base::State *startState;
  ompl::base::State *goalState;

  // TODO: fix memory leaks!!!
};


struct RobotStateValidityChecker : public ompl::base::StateValidityChecker {
  std::shared_ptr<RobotOmpl> robot;
  mutable Eigen::VectorXd x_eigen;
  RobotStateValidityChecker(std::shared_ptr<RobotOmpl> robot);

  bool virtual isValid(const ompl::base::State *state) const override;
};

std::shared_ptr<RobotOmpl> robot_factory_ompl(const Problem &problem);

#pragma once

#include "ompl/control/StatePropagator.h"
#include "robots.h"

class RobotOmplStatePropagator : public ompl::control::StatePropagator {
public:
  RobotOmplStatePropagator(const ompl::control::SpaceInformationPtr &si,
                           std::shared_ptr<RobotOmpl> robot)
      : ompl::control::StatePropagator(si), robot_(robot) {}

  ~RobotOmplStatePropagator() override = default;

  void propagate(const ompl::base::State *state,
                 const ompl::control::Control *control, double duration,
                 ompl::base::State *result) const override {
    // propagate state
    robot_->propagate(state, control, duration, result);
  }

  bool canPropagateBackward() const override { return false; }

  bool canSteer() const override { return false; }

protected:
  std::shared_ptr<RobotOmpl> robot_;
};


