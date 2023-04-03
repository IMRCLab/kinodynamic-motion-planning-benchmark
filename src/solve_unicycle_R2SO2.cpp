///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#include <fstream>
#include <stdexcept>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

#include "pinocchio/multibody/liegroup/liegroup.hpp"

#include "crocoddyl/core/numdiff/action.hpp"

using namespace crocoddyl;

enum { Options = 0 };
typedef pinocchio::SpecialOrthogonalOperationTpl<2, double, Options>
    SO2_operation;

typedef SO2_operation::ConfigVector_t SO2_config;
typedef SO2_operation::TangentVector_t SO2_tangent;
typedef SO2_operation::JacobianMatrix_t SO2_Jacobian;

// lets define a new state

template <typename _Scalar>
class StateR2SO2_smallTpl : public StateAbstractTpl<_Scalar> {
public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  bool use_pin = true;

  explicit StateR2SO2_smallTpl() : StateAbstractTpl<Scalar>(3, 3){};
  virtual ~StateR2SO2_smallTpl() {}

  virtual VectorXs zero() const override {
    VectorXs out = VectorXs::Zero(nx_);
    // out(2) = 1.;
    return out;
  }

  virtual VectorXs rand() const override { return VectorXs::Random(nx_); }

  // x1 - x0
  virtual void diff(const Eigen::Ref<const VectorXs> &x0,
                    const Eigen::Ref<const VectorXs> &x1,
                    Eigen::Ref<VectorXs> dxout) const override {

    std::cout << "calling diff " << std::endl;
    std::cout << "x0 \n" << x0 << std::endl;
    std::cout << "x1 \n" << x1 << std::endl;
    dxout.template head<2>() = x1.template head<2>() - x0.template head<2>();
    if (use_pin) {
      SO2_operation aso2;
      SO2_config p0, p1;

      p0(0) = cos(x0(2));
      p0(1) = sin(x0(2));

      p1(0) = cos(x1(2));
      p1(1) = sin(x1(2));

      SO2_tangent dp;
      aso2.difference(p0, p1, dp);
      dxout(2) = dp(0);
    } else {

      assert(x1(2) <= M_PI);
      assert(x1(2) >= -M_PI);

      assert(x0(2) <= M_PI);
      assert(x0(2) >= -M_PI);

      double so2_diff = x1(2) - x0(2);
      if (so2_diff > M_PI) {
        so2_diff -= 2 * M_PI;
      }

      if (so2_diff < -M_PI) {
        so2_diff += 2 * M_PI;
      }

      dxout(2) = so2_diff;
    }
  }

  virtual void integrate(const Eigen::Ref<const VectorXs> &x,
                         const Eigen::Ref<const VectorXs> &dx,
                         Eigen::Ref<VectorXs> xout) const override {

    std::cout << "calling integrate " << std::endl;
    std::cout << "x" << std::endl;
    std::cout << x << std::endl;
    std::cout << "dx" << std::endl;
    std::cout << dx << std::endl;
    xout.template head<2>() = x.template head<2>() + dx.template head<2>();
    if (use_pin) {
      SO2_operation aso2;
      SO2_config p0, p1;
      SO2_tangent dp;
      dp(0) = dx(2);

      p0(0) = cos(x(2));
      p0(1) = sin(x(2));

      aso2.integrate(p0, dp, p1);
      xout.template tail<1>() << atan2(p1(1), p1(0));
    } else {
      assert(x(2) <= M_PI);
      assert(x(2) >= -M_PI);
      double so2_x = x(2) + dx(2);

      if (so2_x > M_PI) {
        so2_x -= 2 * M_PI;
      } else if (so2_x < -M_PI) {
        so2_x += 2 * M_PI;
      }
      assert(so2_x <= M_PI);
      assert(so2_x >= -M_PI);
      xout(2) = so2_x;
    }
  }
  virtual void Jdiff(const Eigen::Ref<const VectorXs> &x0,
                     const Eigen::Ref<const VectorXs> &x1,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const {

    if (use_pin) {
      Jfirst.template block<2, 2>(0, 0).diagonal().setConstant(-1.);
      Jsecond.template block<2, 2>(0, 0).diagonal().setConstant(1.);

      SO2_operation aso2;
      SO2_config p0, p1;
      SO2_Jacobian J0, J1;

      p0(0) = cos(x0(2));
      p0(1) = sin(x0(2));

      p1(0) = cos(x1(2));
      p1(1) = sin(x1(2));

      aso2.dDifference(p0, p1, J0, pinocchio::ARG0);
      aso2.dDifference(p0, p1, J1, pinocchio::ARG1);

      Jfirst.template block<1, 1>(2, 2) = J0;
      Jsecond.template block<1, 1>(2, 2) = J1;
    } else {
      Jfirst.diagonal().setConstant(-1.);
      Jsecond.diagonal().setConstant(1.);
    }
  }
  virtual void Jintegrate(const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &dx,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both,
                          const AssignmentOp = setto) const {
    std::cout << "calling Jintegrate" << std::endl;
    throw -1;
  }
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs> &x,
                                   const Eigen::Ref<const VectorXs> &dx,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const Jcomponent firstsecond) const {

    std::cout << "calling JintegrateTransport" << std::endl;
    throw -1;
  }

protected:
  using StateAbstractTpl<Scalar>::nx_;
  using StateAbstractTpl<Scalar>::ndx_;
  using StateAbstractTpl<Scalar>::nq_;
  using StateAbstractTpl<Scalar>::nv_;
  using StateAbstractTpl<Scalar>::lb_;
  using StateAbstractTpl<Scalar>::ub_;
  using StateAbstractTpl<Scalar>::has_limits_;
};

template <typename _Scalar>
class StateR2SO2Tpl : public StateAbstractTpl<_Scalar> {
public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit StateR2SO2Tpl() : StateAbstractTpl<Scalar>(4, 3){};
  virtual ~StateR2SO2Tpl() {}

  virtual VectorXs zero() const override {
    VectorXs out = VectorXs::Zero(nx_);
    out(2) = 1.;
    return out;
  }

  virtual VectorXs rand() const override { return VectorXs::Random(nx_); }

  virtual void diff(const Eigen::Ref<const VectorXs> &x0,
                    const Eigen::Ref<const VectorXs> &x1,
                    Eigen::Ref<VectorXs> dxout) const override {

    std::cout << "calling diff " << std::endl;
    std::cout << "x0 \n" << x0 << std::endl;
    std::cout << "x1 \n" << x1 << std::endl;
    dxout.template head<2>() = x1.template head<2>() - x0.template head<2>();
    SO2_operation aso2;

    SO2_config p0, p1;

    p0(0) = x0(2);
    p0(1) = x0(3);

    p1(0) = x1(2);
    p1(1) = x1(3);

    SO2_tangent dp;
    aso2.difference(p0, p1, dp);
    dxout(2) = dp(0);
  }

  virtual void integrate(const Eigen::Ref<const VectorXs> &x,
                         const Eigen::Ref<const VectorXs> &dx,
                         Eigen::Ref<VectorXs> xout) const override {

    std::cout << "calling integrate " << std::endl;
    std::cout << "x" << std::endl;
    std::cout << x << std::endl;
    std::cout << "dx" << std::endl;
    std::cout << dx << std::endl;
    xout.template head<2>() = x.template head<2>() + dx.template head<2>();

    SO2_operation aso2;
    SO2_config p0, p1;
    SO2_tangent dp;
    dp(0) = dx(2);
    p0(0) = x(2);
    p0(1) = x(3);

    aso2.integrate(p0, dp, p1);
    xout.template tail<2>() << p1(0), p1(1);

    std::cout << "xout" << std::endl;
    std::cout << xout << std::endl;
  }
  virtual void Jdiff(const Eigen::Ref<const VectorXs> &x0,
                     const Eigen::Ref<const VectorXs> &x1,
                     Eigen::Ref<MatrixXs> Jfirst, Eigen::Ref<MatrixXs> Jsecond,
                     const Jcomponent firstsecond = both) const {

    // implement this!!
    //
    Jfirst.template block<2, 2>(0, 0).diagonal().setConstant(-1.);
    Jsecond.template block<2, 2>(0, 0).diagonal().setConstant(1.);

    SO2_operation aso2;
    SO2_config p0, p1;
    SO2_Jacobian J0, J1;

    p0(0) = x0(2);
    p0(1) = x0(3);

    p1(0) = x1(2);
    p1(1) = x1(3);

    aso2.dDifference(p0, p1, J0, pinocchio::ARG0);
    aso2.dDifference(p0, p1, J1, pinocchio::ARG1);

    Jfirst.template block<1, 1>(2, 2) = J0;
    Jsecond.template block<1, 1>(2, 2) = J1;
  }
  virtual void Jintegrate(const Eigen::Ref<const VectorXs> &x,
                          const Eigen::Ref<const VectorXs> &dx,
                          Eigen::Ref<MatrixXs> Jfirst,
                          Eigen::Ref<MatrixXs> Jsecond,
                          const Jcomponent firstsecond = both,
                          const AssignmentOp = setto) const {
    std::cout << "calling Jintegrate" << std::endl;
    throw -1;
  }
  virtual void JintegrateTransport(const Eigen::Ref<const VectorXs> &x,
                                   const Eigen::Ref<const VectorXs> &dx,
                                   Eigen::Ref<MatrixXs> Jin,
                                   const Jcomponent firstsecond) const {

    std::cout << "calling JintegrateTransport" << std::endl;
    throw -1;
  }

protected:
  using StateAbstractTpl<Scalar>::nx_;
  using StateAbstractTpl<Scalar>::ndx_;
  using StateAbstractTpl<Scalar>::nq_;
  using StateAbstractTpl<Scalar>::nv_;
  using StateAbstractTpl<Scalar>::lb_;
  using StateAbstractTpl<Scalar>::ub_;
  using StateAbstractTpl<Scalar>::has_limits_;
};

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
// #include "crocoddyl/core/states/euclidean.hxx"

// #endif // CROCODDYL_CORE_STATES_EUCLIDEAN_HPP_

template <typename _Scalar>
struct ActionDataUnicycleR2SO2_smallTpl
    : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  template <template <typename Scalar> class Model>
  explicit ActionDataUnicycleR2SO2_smallTpl(Model<Scalar> *const model)
      : Base(model) {}
};

template <typename _Scalar>
struct ActionDataUnicycleR2SO2Tpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  template <template <typename Scalar> class Model>
  explicit ActionDataUnicycleR2SO2Tpl(Model<Scalar> *const model)
      : Base(model) {}
};

template <typename _Scalar>
class ActionModelUnicycleR2SO2_smallTpl
    : public ActionModelAbstractTpl<_Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataUnicycleR2SO2Tpl<Scalar> Data;
  typedef StateR2SO2_smallTpl<Scalar> State;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector2s Vector2s;

  ActionModelUnicycleR2SO2_smallTpl();
  virtual ~ActionModelUnicycleR2SO2_smallTpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);
  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data);

  const Vector2s &get_cost_weights() const;
  void set_cost_weights(const Vector2s &weights);

  Scalar get_dt() const;
  void set_dt(const Scalar dt);

  /**
   * @brief Print relevant information of the unicycle model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

protected:
  using Base::nu_;    //!< Control dimension
  using Base::state_; //!< Model of the state

private:
  Vector2s cost_weights_;
  Scalar dt_;
};

template <typename _Scalar>
class ActionModelUnicycleR2SO2Tpl : public ActionModelAbstractTpl<_Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataUnicycleR2SO2Tpl<Scalar> Data;
  typedef StateR2SO2Tpl<Scalar> State;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector2s Vector2s;

  ActionModelUnicycleR2SO2Tpl();
  virtual ~ActionModelUnicycleR2SO2Tpl();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);
  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);
  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data);

  const Vector2s &get_cost_weights() const;
  void set_cost_weights(const Vector2s &weights);

  Scalar get_dt() const;
  void set_dt(const Scalar dt);

  /**
   * @brief Print relevant information of the unicycle model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream &os) const;

protected:
  using Base::nu_;    //!< Control dimension
  using Base::state_; //!< Model of the state

private:
  Vector2s cost_weights_;
  Scalar dt_;
};

template <typename Scalar>
ActionModelUnicycleR2SO2_smallTpl<Scalar>::ActionModelUnicycleR2SO2_smallTpl()
    : ActionModelAbstractTpl<Scalar>(
          boost::make_shared<StateR2SO2_smallTpl<Scalar>>(), 2, 5),
      dt_(Scalar(0.1)) {
  cost_weights_ << Scalar(100.), Scalar(1.);
}

template <typename Scalar>
ActionModelUnicycleR2SO2_smallTpl<
    Scalar>::~ActionModelUnicycleR2SO2_smallTpl() {}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Scalar c = cos(x(2));
  const Scalar s = sin(x(2));

  // d->r.template head<3>() = cost_weights_[0] * x;
  d->r.template tail<2>() = cost_weights_[1] * u;
  d->cost = Scalar(.5) * d->r.dot(d->r);

  // i need a step

  // SO2_operation aso2_operation;
  // SO2_config so2_s, so2_g;
  // SO2_tangent so2_tangent;
  //
  // so2_tangent(0) = u(1) * dt_;
  // so2_s(0) = c;
  // so2_s(1) = s;

  // State* state = static_cast<State *>(this->get_state().get());
  State state;
  Eigen::Vector3d v(c * u(0), s * u(0), u(1));
  state.integrate(x, v * dt_, data->xnext);
  // state
  //
  // state->integrate(
  //
  //
  // aso2_operation.integrate(so2_s, so2_tangent, so2_g);
  // // std::cout << "d->xnext " << std::endl;
  // // std::cout << d->xnext << std::endl;
  // d->xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, so2_g(0),
  // so2_g(1);
}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  // d->r.template head<3>() = cost_weights_[0] * x;
  // d->r.template tail<2>().setZero();

  Eigen::Vector3d goal(0., 1., -M_PI);
  Eigen::Vector3d diff;

  State().diff(x, goal, diff);

  const Scalar w_x = cost_weights_[0] * cost_weights_[0];
  d->cost = w_x * .5 * diff.dot(diff);
}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  // std::cout << "d->Fx" << std::endl;
  // std::cout << d->Fx << std::endl;
  // std::cout << "d->Fu" << std::endl;
  // std::cout << d->Fu << std::endl;

  const Scalar c = cos(x(2));
  const Scalar s = sin(x(2));
  const Scalar w_u = cost_weights_[1] * cost_weights_[1];
  // d->Lx = x * w_x;
  d->Lu = u * w_u;
  // d->Lxx.diagonal().setConstant(w_x);
  d->Luu.diagonal().setConstant(w_u);

  d->Fx(0, 0) = 1;
  d->Fx(1, 1) = 1;
  d->Fx(2, 2) = 1;

  d->Fx(0, 2) = -s * u[0] * dt_;
  d->Fx(1, 2) = c * u[0] * dt_;

  //
  d->Fu(0, 0) = c * dt_;
  d->Fu(1, 0) = s * dt_;
  d->Fu(2, 1) = dt_;

  // SO2_operation aso2_operation;
  // SO2_config so2_s;
  // SO2_tangent so2_tangent;
  // so2_s(0) = c;
  // so2_s(1) = s;
  // so2_tangent(u(1) * dt_);

  // aso2_operation.integrate(so2_s, so2_tangent, so2_g);

  // void dIntegrate(const Eigen::MatrixBase<Config_t >  & q,
  //                 const Eigen::MatrixBase<Tangent_t>  & v,
  //                 const Eigen::MatrixBase<JacobianOut_t> & J,
  //                 const ArgumentPosition arg,
  //                 const AssignmentOperatorType op = SETTO) const;

  // SO2_Jacobian J;
  // aso2_operation.dIntegrate<pinocchio::ARG0>(so2_s, so2_tangent, J);
  // std::cout << "J1 \n" << J << std::endl;
  // aso2_operation.dIntegrate<pinocchio::ARG1>(so2_s, so2_tangent, J);
  // std::cout << "J2 \n" << J << std::endl;
}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Scalar w_x = cost_weights_[0] * cost_weights_[0];
  Eigen::Vector3d goal(0., 1., -M_PI);
  Eigen::Vector3d diff;
  State().diff(x, goal, diff);
  d->Lx = -1. * w_x * diff; // is this always true? -- YES.
  d->Lxx.diagonal().array() = w_x;
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelUnicycleR2SO2_smallTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool ActionModelUnicycleR2SO2_smallTpl<Scalar>::checkData(
    const boost::shared_ptr<ActionDataAbstract> &data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::print(std::ostream &os) const {
  os << "ActionModelUnicycleR2SO2Tpl {dt=" << dt_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s &
ActionModelUnicycleR2SO2_smallTpl<Scalar>::get_cost_weights() const {
  return cost_weights_;
}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::set_cost_weights(
    const typename MathBase::Vector2s &weights) {
  cost_weights_ = weights;
}

template <typename Scalar>
Scalar ActionModelUnicycleR2SO2_smallTpl<Scalar>::get_dt() const {
  return dt_;
}

template <typename Scalar>
void ActionModelUnicycleR2SO2_smallTpl<Scalar>::set_dt(const Scalar dt) {
  if (dt <= 0)
    throw_pretty("Invalid argument: dt should be strictly positive.");
  dt_ = dt;
}

//////////
////////
////////////
template <typename Scalar>
ActionModelUnicycleR2SO2Tpl<Scalar>::ActionModelUnicycleR2SO2Tpl()
    : ActionModelAbstractTpl<Scalar>(
          boost::make_shared<StateR2SO2Tpl<Scalar>>(), 2, 5),
      dt_(Scalar(0.1)) {
  cost_weights_ << Scalar(100.), Scalar(1.);
}

template <typename Scalar>
ActionModelUnicycleR2SO2Tpl<Scalar>::~ActionModelUnicycleR2SO2Tpl() {}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Scalar c = x(2);
  const Scalar s = x(3);

  // d->r.template head<3>() = cost_weights_[0] * x;
  d->r.template tail<2>() = cost_weights_[1] * u;
  d->cost = Scalar(.5) * d->r.dot(d->r);

  // i need a step

  // SO2_operation aso2_operation;
  // SO2_config so2_s, so2_g;
  // SO2_tangent so2_tangent;
  //
  // so2_tangent(0) = u(1) * dt_;
  // so2_s(0) = c;
  // so2_s(1) = s;

  // State* state = static_cast<State *>(this->get_state().get());
  State state;
  Eigen::Vector3d v(c * u(0), s * u(0), u(1));
  state.integrate(x, v * dt_, data->xnext);
  // state
  //
  // state->integrate(
  //
  //
  // aso2_operation.integrate(so2_s, so2_tangent, so2_g);
  // // std::cout << "d->xnext " << std::endl;
  // // std::cout << d->xnext << std::endl;
  // d->xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, so2_g(0),
  // so2_g(1);
}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::calc(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  // d->r.template head<3>() = cost_weights_[0] * x;
  // d->r.template tail<2>().setZero();

  Eigen::Vector4d goal(0., 1., -1, 0);
  Eigen::Vector3d diff;

  State().diff(x, goal, diff);

  const Scalar w_x = cost_weights_[0] * cost_weights_[0];
  d->cost = w_x * .5 * diff.dot(diff);
}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  // std::cout << "d->Fx" << std::endl;
  // std::cout << d->Fx << std::endl;
  // std::cout << "d->Fu" << std::endl;
  // std::cout << d->Fu << std::endl;

  const Scalar c = x(2);
  const Scalar s = x(3);
  const Scalar w_u = cost_weights_[1] * cost_weights_[1];
  // d->Lx = x * w_x;
  d->Lu = u * w_u;
  // d->Lxx.diagonal().setConstant(w_x);
  d->Luu.diagonal().setConstant(w_u);

  d->Fx(0, 0) = 1;
  d->Fx(1, 1) = 1;
  d->Fx(2, 2) = 1;

  d->Fx(0, 2) = -s * u[0] * dt_;
  d->Fx(1, 2) = c * u[0] * dt_;

  //
  d->Fu(0, 0) = c * dt_;
  d->Fu(1, 0) = s * dt_;
  d->Fu(2, 1) = dt_;

  // SO2_operation aso2_operation;
  // SO2_config so2_s;
  // SO2_tangent so2_tangent;
  // so2_s(0) = c;
  // so2_s(1) = s;
  // so2_tangent(u(1) * dt_);

  // aso2_operation.integrate(so2_s, so2_tangent, so2_g);

  // void dIntegrate(const Eigen::MatrixBase<Config_t >  & q,
  //                 const Eigen::MatrixBase<Tangent_t>  & v,
  //                 const Eigen::MatrixBase<JacobianOut_t> & J,
  //                 const ArgumentPosition arg,
  //                 const AssignmentOperatorType op = SETTO) const;

  // SO2_Jacobian J;
  // aso2_operation.dIntegrate<pinocchio::ARG0>(so2_s, so2_tangent, J);
  // std::cout << "J1 \n" << J << std::endl;
  // aso2_operation.dIntegrate<pinocchio::ARG1>(so2_s, so2_tangent, J);
  // std::cout << "J2 \n" << J << std::endl;
}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::calcDiff(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Scalar w_x = cost_weights_[0] * cost_weights_[0];
  Eigen::Vector4d goal(0., 1., -1., 0);
  Eigen::Vector3d diff;
  State().diff(x, goal, diff);
  d->Lx = -1. * w_x * diff; // is this always true? -- YES.
  d->Lxx.diagonal().array() = w_x;
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelUnicycleR2SO2Tpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool ActionModelUnicycleR2SO2Tpl<Scalar>::checkData(
    const boost::shared_ptr<ActionDataAbstract> &data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::print(std::ostream &os) const {
  os << "ActionModelUnicycleR2SO2Tpl {dt=" << dt_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector2s &
ActionModelUnicycleR2SO2Tpl<Scalar>::get_cost_weights() const {
  return cost_weights_;
}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::set_cost_weights(
    const typename MathBase::Vector2s &weights) {
  cost_weights_ = weights;
}

template <typename Scalar>
Scalar ActionModelUnicycleR2SO2Tpl<Scalar>::get_dt() const {
  return dt_;
}

template <typename Scalar>
void ActionModelUnicycleR2SO2Tpl<Scalar>::set_dt(const Scalar dt) {
  if (dt <= 0)
    throw_pretty("Invalid argument: dt should be strictly positive.");
  dt_ = dt;
}

int main(int argc, char *argv[]) {
#if 0
  bool CALLBACKS = false;
  unsigned int N = 200; // number of nodes
  unsigned int T = 5e3; // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1) {
    T = atoi(argv[1]);
  }

  // Creating the action models and warm point for the unicycle system
  Eigen::VectorXd x0 = Eigen::Vector3d(1., 0., 0.);
  boost::shared_ptr<crocoddyl::ActionModelAbstract> model =
      boost::make_shared<crocoddyl::ActionModelUnicycle>();
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::Vector2d::Zero());
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> runningModels(
      N, model);

  // Formulating the optimal control problem
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels, model);
  crocoddyl::SolverDDP ddp(problem);
  if (CALLBACKS) {
    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);
  }

  // Solving the optimal control problem
  Eigen::ArrayXd duration(T);
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    ddp.solve(xs, us, MAXITER);
    duration[i] = timer.get_duration();
  }

  double avrg_duration = duration.sum() / T;
  double min_duration = duration.minCoeff();
  double max_duration = duration.maxCoeff();
  std::cout << "  DDP.solve [ms]: " << avrg_duration << " (" << min_duration
            << "-" << max_duration << ")" << std::endl;

  // Running calc
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    problem->calc(xs, us);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " ("
            << min_duration << "-" << max_duration << ")" << std::endl;

  // Running calcDiff
  for (unsigned int i = 0; i < T; ++i) {
    crocoddyl::Timer timer;
    problem->calcDiff(xs, us);
    duration[i] = timer.get_duration();
  }

  avrg_duration = duration.sum() / T;
  min_duration = duration.minCoeff();
  max_duration = duration.maxCoeff();
  std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " ("
            << min_duration << "-" << max_duration << ")" << std::endl;

#endif

  {
    std::cout << "now in r2so2" << std::endl;
    Eigen::VectorXd x0 = Eigen::Vector4d(1., 0., 0, 1.);
    auto model = boost::make_shared<ActionModelUnicycleR2SO2Tpl<double>>();
    Eigen::Vector2d u(0, 0);
    {
      auto data = model->createData();
      std::cout << "model" << std::endl;
      model->calc(data, x0, u);
      model->calcDiff(data, x0, u);
      std::cout << "data->Fx" << std::endl;
      std::cout << data->Fx << std::endl;
      std::cout << "data->Fu" << std::endl;
      std::cout << data->Fu << std::endl;
      std::cout << "xnext\n" << data->xnext << std::endl;
    }

    auto model_diff = boost::make_shared<ActionModelNumDiffTpl<double>>(model);

    {
      auto data = model_diff->createData();
      std::cout << "model diff" << std::endl;
      model_diff->calc(data, x0, u);
      model_diff->calcDiff(data, x0, u);
      std::cout << "data->Fx" << std::endl;
      std::cout << data->Fx << std::endl;
      std::cout << "data->Fu" << std::endl;
      std::cout << data->Fu << std::endl;
      std::cout << "xnext\n" << data->xnext << std::endl;
    }

    //
    int N = 20;
    //
    std::vector<Eigen::VectorXd> xs(N + 1, x0);
    std::vector<Eigen::VectorXd> us(N, Eigen::Vector2d::Zero());
    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
        runningModels(N, model);

    boost::shared_ptr<crocoddyl::ShootingProblem> problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                       model);
    crocoddyl::SolverDDP ddp(problem);
    double cc = problem->calc(xs, us);
    auto xx = x0;
    Eigen::Vector2d uu(0, 0);

    auto data = model_diff->createData();

    StateR2SO2Tpl<double> state;
    std::cout << state.get_nx() << std::endl;  // 4
    std::cout << state.get_ndx() << std::endl; // 3
    std::cout << state.get_nq() << std::endl;  // 3
    std::cout << state.get_nv() << std::endl;  // 1

    Eigen::VectorXd x1 = Eigen::Vector4d(1., 1., 0., 1.);
    Eigen::Vector4d x2;
    Eigen::Vector3d dout;
    // x1 - x0
    state.diff(x0, x1, dout);
    // x0 + x2
    state.integrate(x0, dout, x2);

    Eigen::Vector3d dout2;
    state.diff(x1, x2, dout2);

    assert(dout2.norm() < 1e-12);

    state.diff(x2, x1, dout2);
    assert(dout2.norm() < 1e-12);

    std::cout << "x2 " << std::endl;

    for (size_t n = 0; n < N; n++) {
      model_diff->calc(data, xx, u);
      std::cout << "cost " << data->cost << std::endl;
      std::cout << " xnext " << std::endl;
      std::cout << data->xnext << std::endl;
      xx = data->xnext;
    }
    std::cout << "cc:" << cc << std::endl;
    int MAXITER = 10;

    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);

    ddp.solve(xs, us, MAXITER);

    auto xs_out = ddp.get_xs();
    auto us_out = ddp.get_us();
    std::ofstream file("__out.yaml");

    file << "xs: " << std::endl;
    for (auto &xs : xs_out) {
      file << "  - [";

      assert(xs.size());
      for (size_t i = 0; i < xs.size() - 1; i++) {
        file << xs(i) << ", ";
      }
      file << xs(xs.size() - 1) << "] " << std::endl;
    }

    file << "us: " << std::endl;
    for (auto &us : us_out) {
      file << "  - [";

      assert(us.size());
      for (size_t i = 0; i < us.size() - 1; i++) {
        file << us(i) << ", ";
      }
      file << us(us.size() - 1) << "] " << std::endl;
    }

    {
      StateR2SO2Tpl<double> state;
      Eigen::Vector4d x0(1, 1, 0, 1);
      Eigen::Vector4d x1(1, 1, cos(-3. * M_PI / 4.), sin(-3 * M_PI / 4));
      Eigen::Vector4d x2(1, 1, cos(3. * M_PI / 4.), sin(3 * M_PI / 4));
      Eigen::Vector4d x3(1, 1, cos(M_PI / 4.), sin(M_PI / 4));

      Eigen::Matrix3d Jfirst;
      Eigen::Matrix3d Jsecond;
      Jfirst.setZero();
      Jsecond.setZero();

      state.Jdiff(x0, x1, Jfirst, Jsecond);
      std::cout << "Jfirst:\n" << Jfirst << std::endl;
      std::cout << "Jsecond:\n" << Jsecond << std::endl;

      state.Jdiff(x0, x2, Jfirst, Jsecond);
      std::cout << "Jfirst:\n" << Jfirst << std::endl;
      std::cout << "Jsecond:\n" << Jsecond << std::endl;

      state.Jdiff(x0, x3, Jfirst, Jsecond);
      std::cout << "Jfirst:\n" << Jfirst << std::endl;
      std::cout << "Jsecond:\n" << Jsecond << std::endl;
    }
  }
  {
    std::cout << "now in r2so2 small" << std::endl;
    Eigen::VectorXd x0 = Eigen::Vector3d(1., 0., M_PI / 2.);
    auto model =
        boost::make_shared<ActionModelUnicycleR2SO2_smallTpl<double>>();
    Eigen::Vector2d u(0, 0);
    {
      auto data = model->createData();
      std::cout << "model" << std::endl;
      model->calc(data, x0, u);
      model->calcDiff(data, x0, u);
      std::cout << "data->Fx" << std::endl;
      std::cout << data->Fx << std::endl;
      std::cout << "data->Fu" << std::endl;
      std::cout << data->Fu << std::endl;
      std::cout << "xnext\n" << data->xnext << std::endl;
    }

    auto model_diff = boost::make_shared<ActionModelNumDiffTpl<double>>(model);

    {
      auto data = model_diff->createData();
      std::cout << "model diff" << std::endl;
      model_diff->calc(data, x0, u);
      model_diff->calcDiff(data, x0, u);
      std::cout << "data->Fx" << std::endl;
      std::cout << data->Fx << std::endl;
      std::cout << "data->Fu" << std::endl;
      std::cout << data->Fu << std::endl;
      std::cout << "xnext\n" << data->xnext << std::endl;
    }

    //
    int N = 20;
    //
    std::vector<Eigen::VectorXd> xs(N + 1, x0);
    std::vector<Eigen::VectorXd> us(N, Eigen::Vector2d::Zero());
    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
        runningModels(N, model);

    boost::shared_ptr<crocoddyl::ShootingProblem> problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                       model);
    crocoddyl::SolverDDP ddp(problem);
    double cc = problem->calc(xs, us);
    auto xx = x0;
    Eigen::Vector2d uu(0, 0);

    auto data = model_diff->createData();

    StateR2SO2_smallTpl<double> state;
    std::cout << state.get_nx() << std::endl;  // 4
    std::cout << state.get_ndx() << std::endl; // 3
    std::cout << state.get_nq() << std::endl;  // 3
    std::cout << state.get_nv() << std::endl;  // 1

    Eigen::VectorXd x1 = Eigen::Vector3d(1., 1., M_PI / 2.);
    Eigen::Vector3d x2;
    Eigen::Vector3d dout;
    // x1 - x0
    state.diff(x0, x1, dout);
    // x0 + x2
    state.integrate(x0, dout, x2);

    Eigen::Vector3d dout2;
    state.diff(x1, x2, dout2);

    assert(dout2.norm() < 1e-12);

    state.diff(x2, x1, dout2);
    assert(dout2.norm() < 1e-12);

    std::cout << "x2 " << std::endl;

    for (size_t n = 0; n < N; n++) {
      model_diff->calc(data, xx, u);
      std::cout << "cost " << data->cost << std::endl;
      std::cout << " xnext " << std::endl;
      std::cout << data->xnext << std::endl;
      xx = data->xnext;
    }
    std::cout << "cc:" << cc << std::endl;
    int MAXITER = 10;

    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);

    ddp.solve(xs, us, MAXITER);

    auto xs_out = ddp.get_xs();
    auto us_out = ddp.get_us();
    std::ofstream file("__out2.yaml");

    file << "xs: " << std::endl;
    for (auto &xs : xs_out) {
      file << "  - [";

      assert(xs.size());
      for (size_t i = 0; i < xs.size() - 1; i++) {
        file << xs(i) << ", ";
      }
      file << xs(xs.size() - 1) << "] " << std::endl;
    }

    file << "us: " << std::endl;
    for (auto &us : us_out) {
      file << "  - [";

      assert(us.size());
      for (size_t i = 0; i < us.size() - 1; i++) {
        file << us(i) << ", ";
      }
      file << us(us.size() - 1) << "] " << std::endl;
    }

    {
      StateR2SO2_smallTpl<double> state;
      Eigen::Vector3d x0(1, 1, M_PI / 2);
      Eigen::Vector3d x1(1, 1, -3. * M_PI / 4.);
      Eigen::Vector3d x2(1, 1, 3. * M_PI / 4.);
      Eigen::Vector3d x3(1, 1, M_PI / 4.);

      Eigen::Matrix3d Jfirst;
      Eigen::Matrix3d Jsecond;
      Jfirst.setZero();
      Jsecond.setZero();

      state.Jdiff(x0, x1, Jfirst, Jsecond);
      std::cout << "Jfirst:\n" << Jfirst << std::endl;
      std::cout << "Jsecond:\n" << Jsecond << std::endl;

      state.Jdiff(x0, x2, Jfirst, Jsecond);
      std::cout << "Jfirst:\n" << Jfirst << std::endl;
      std::cout << "Jsecond:\n" << Jsecond << std::endl;

      state.Jdiff(x0, x3, Jfirst, Jsecond);
      std::cout << "Jfirst:\n" << Jfirst << std::endl;
      std::cout << "Jsecond:\n" << Jsecond << std::endl;
    }
  }
  // working!!
}
