
#include <cmath>
#include <fstream>
#include <iostream>

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "Eigen/Core"
#include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include <boost/program_options.hpp>

#include "collision_checker.hpp"

using namespace crocoddyl;

// Eigen::Vector3d goal(1.9, .3, 0);
static double collision_weight = 100.;
// issue with derivatives...

template <class T> using ptr = boost::shared_ptr<T>;

template <typename T, typename... Args> auto mk(Args &&...args) {
  return boost::make_shared<T>(std::forward<Args>(args)...);
}

struct ActionDataQuim : public ActionDataAbstractTpl<double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
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
  explicit ActionDataQuim(Model<Scalar> *const model) : Base(model) {}
};

struct ActionDataQ : public ActionDataAbstractTpl<double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
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
  explicit ActionDataQ(Model<Scalar> *const model) : Base(model) {}
};

double diff_angle(double angle1, double angle2) {
  return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

struct Dynamics {

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) = 0;

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) = 0;

  size_t nx;
  size_t nu;
};

struct Dynamics_unicycle2 : Dynamics {

  double dt = .1;

  bool free_time;
  Dynamics_unicycle2(bool free_time = false) : free_time(free_time) {
    nx = 5;
    nu = 2;
    if (free_time)
      nu += 1;
  }

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) override {

    if (static_cast<std::size_t>(x.size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " +
                          std::to_string(nx) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu) + ")");
    }

    const double xx = x[0];
    const double y = x[1];
    const double yaw = x[2];
    const double v = x[3];
    const double w = x[4];

    const double c = cos(yaw);
    const double s = sin(yaw);

    const double a = u[0];
    const double w_dot = u[1];

    double dt_ = dt;
    if (free_time)
      dt_ *= u[2];

    const double v_next = v + a * dt_;
    const double w_next = w + w_dot * dt_;
    const double yaw_next = yaw + w * dt_;
    const double x_next = xx + v * c * dt_;
    const double y_next = y + v * s * dt_;

    xnext << x_next, y_next, yaw_next, v_next, w_next;
  };

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) override {

    if (static_cast<std::size_t>(x.size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " +
                          std::to_string(nx) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu) + ")");
    }

    const double c = cos(x[2]);
    const double s = sin(x[2]);

    const double v = x[3];
    Fx.setZero();
    Fu.setZero();

    double dt_ = dt;
    if (free_time)
      dt_ *= u[2];

    Fx(0, 0) = 1;
    Fx(1, 1) = 1;
    Fx(2, 2) = 1;
    Fx(3, 3) = 1;
    Fx(4, 4) = 1;

    Fx(0, 2) = -s * v * dt_;
    Fx(1, 2) = c * v * dt_;

    Fx(0, 3) = c * dt_;
    Fx(1, 3) = s * dt_;
    Fx(2, 4) = dt_;

    Fu(3, 0) = dt_;
    Fu(4, 1) = dt_;

    if (free_time) {

      const double a = u[0];
      const double w_dot = u[1];
      const double w = x[4];

      Fu(0, 2) = v * c * dt;
      Fu(1, 2) = v * s * dt;
      Fu(2, 2) = w * dt;
      Fu(3, 2) = a * dt;
      Fu(4, 2) = w_dot * dt;
    }
  }
};

struct Dynamics_unicycle : Dynamics {

  double dt = .1;
  bool free_time;

  Dynamics_unicycle(bool free_time = false) : free_time(free_time) {
    nx = 3;
    if (free_time)
      nu = 3;
    else
      nu = 2;
  }

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) override {

    if (static_cast<std::size_t>(x.size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " +
                          std::to_string(nx) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu) + ")");
    }

    const double c = cos(x[2]);
    const double s = sin(x[2]);

    double dt_ = dt;
    if (free_time)
      dt_ *= u[2];

    xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, x[2] + u[1] * dt_;
  };

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) override {

    if (static_cast<std::size_t>(x.size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " +
                          std::to_string(nx) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " +
                          std::to_string(nu) + ")");
    }

    const double c = cos(x[2]);
    const double s = sin(x[2]);

    double dt_ = dt;
    if (free_time)
      dt_ *= u[2];

    Fx.setZero();
    Fu.setZero();

    Fx(0, 0) = 1;
    Fx(1, 1) = 1;
    Fx(2, 2) = 1;
    Fx(0, 2) = -s * u[0] * dt_;
    Fx(1, 2) = c * u[0] * dt_;
    Fu(0, 0) = c * dt_;
    Fu(1, 0) = s * dt_;
    Fu(2, 1) = dt_;

    if (free_time) {
      Fu(0, 2) = c * u[0] * dt;
      Fu(1, 2) = s * u[0] * dt;
      Fu(2, 2) = u[1] * dt;
    }
  }
};

struct Cost {
  size_t nx;
  size_t nu;
  size_t nr;

  Cost(size_t nx, size_t nu, size_t nr) : nx(nx), nu(nu), nr(nr) {}
  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) = 0;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) = 0;

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) = 0;

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) = 0;
};

struct Col_cost : Cost {

  boost::shared_ptr<CollisionChecker> cl;

  double faraway_zero_gradient_bound = 5 * 1e-2;              // returns 0 gradient if distance is > than THIS.
  double epsilon = 1e-4; // finite diff
  std::vector<bool> non_zero_flags; // if not nullptr, non_zero_flags[i] = True
                                    // means compute gradient, non_zero_flags[i]
                                    // = false is no compute gradient

  Col_cost(size_t nx, size_t nu, size_t nr,
           boost::shared_ptr<CollisionChecker> cl)
      : Cost(nx, nu, nr), cl(cl) {}

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    std::vector<double> query{x.data(), x.data() + x.size()};
    double d = collision_weight * std::get<0>(cl->distance(query));
    auto out = Eigen::Matrix<double, 1, 1>(std::min(d, 0.));
    r = out;
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {
    calc(r, x, Eigen::VectorXd(nu));
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    std::vector<double> query{x.data(), x.data() + x.size()};
    auto out = cl->distanceWithFDiffGradient(
        query, faraway_zero_gradient_bound, epsilon,
        non_zero_flags.size() ? &non_zero_flags : nullptr);
    auto dist = collision_weight * std::get<0>(out);
    auto grad = std::get<1>(out);
    Eigen::VectorXd v =
        collision_weight * Eigen::VectorXd::Map(grad.data(), grad.size());
    if (dist <= 0) {
      Jx = v.transpose();
    } else {
      Jx.setZero();
    }
    Ju.setZero();
  };

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    auto Ju = Eigen::MatrixXd(1, 1);
    auto u = Eigen::VectorXd(1);
    calcDiff(Jx, Ju, x, u);
  }
};

struct Control_cost : Cost {

  Eigen::VectorXd u_weight;

  Control_cost(size_t nx, size_t nu, size_t nr, Eigen::VectorXd u_weight)
      : Cost(nx, nu, nr), u_weight(u_weight) {}

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    r = u.cwiseProduct(u_weight);
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {

    auto u = Eigen::VectorXd::Zero(nu);
    calc(r, x, u);
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    assert(static_cast<std::size_t>(Jx.rows()) == nr);
    assert(static_cast<std::size_t>(Ju.rows()) == nr);
    assert(static_cast<std::size_t>(Ju.cols()) == nu);
    Ju = u_weight.asDiagonal();
    Jx.setZero();
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    Eigen::VectorXd u(0);
    Eigen::MatrixXd Ju(0, 0);
    calcDiff(Jx, Ju, x, u);
  }
};

struct State_cost : Cost {

  Eigen::VectorXd x_weight;
  Eigen::VectorXd ref;

  State_cost(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &x_weight,
             const Eigen::VectorXd &ref)
      : Cost(nx, nu, nr), x_weight(x_weight), ref(ref) {}

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    r = (x - ref).cwiseProduct(x_weight);
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {

    assert(static_cast<std::size_t>(r.size()) == nr);
    r = (x - ref).cwiseProduct(x_weight);
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

    assert(static_cast<std::size_t>(Jx.rows()) == nr);
    assert(static_cast<std::size_t>(Ju.rows()) == nr);
    assert(static_cast<std::size_t>(Jx.cols()) == nx);
    assert(static_cast<std::size_t>(Ju.cols()) == nu);

    Jx = x_weight.asDiagonal();
    Ju.setZero();
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    assert(static_cast<std::size_t>(Jx.rows()) == nr);
    assert(static_cast<std::size_t>(Jx.cols()) == nx);

    Jx = x_weight.asDiagonal();
  }
};

struct All_cost : Cost {

  std::vector<boost::shared_ptr<Cost>> costs;

  All_cost(size_t nx, size_t nu, size_t nr,
           const std::vector<boost::shared_ptr<Cost>> &costs)
      : Cost(nx, nu, nr), costs(costs) {}

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    int index = 0;
    for (size_t i = 0; i < costs.size(); i++) {
      // fill the matrix
      auto &feat = costs.at(i);
      int _nr = feat->nr;
      feat->calc(r.segment(index, _nr), x, u);
      index += _nr;
    }
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);

    int index = 0;
    for (size_t i = 0; i < costs.size(); i++) {
      auto &feat = costs.at(i);
      int _nr = feat->nr;
      feat->calc(r.segment(index, _nr), x);
      index += _nr;
    }
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    // TODO: I shoudl only give the residuals...
    int index = 0;
    for (size_t i = 0; i < costs.size(); i++) {
      auto &feat = costs.at(i);
      int _nr = feat->nr;
      feat->calcDiff(Jx.block(index, 0, _nr, nx), Ju.block(index, 0, _nr, nu),
                     x, u);
      index += _nr;
    }
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    assert(static_cast<std::size_t>(x.size()) == nx);

    // TODO: I shoudl only give the residuals...
    int index = 0;
    for (size_t i = 0; i < costs.size(); i++) {
      auto &feat = costs.at(i);
      int _nr = costs.at(i)->nr;
      feat->calcDiff(Jx.block(index, 0, _nr, nx), x);
      index += _nr;
    }
  }
};

class ActionModelQ : public ActionModelAbstractTpl<double> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataQ Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector2s Vector2s;

  boost::shared_ptr<Dynamics> dynamics;
  boost::shared_ptr<Cost> features;
  size_t nx;
  size_t nu;
  size_t nr;

  ActionModelQ(boost::shared_ptr<Dynamics> dynamics,
               boost::shared_ptr<Cost> features)
      : Base(boost::make_shared<StateVectorTpl<Scalar>>(dynamics->nx),
             dynamics->nu, features->nr),
        dynamics(dynamics), features(features), nx(dynamics->nx),
        nu(dynamics->nu), nr(features->nr) {}

  virtual ~ActionModelQ(){};

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) {
    Data *d = static_cast<Data *>(data.get());

    dynamics->calc(d->xnext, x, u);
    features->calc(d->r, x, u);
    d->cost = Scalar(0.5) * d->r.dot(d->r);
  }

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) {

    Data *d = static_cast<Data *>(data.get());
    dynamics->calcDiff(d->Fx, d->Fu, x, u);
    // CHANGE THIS

    // create a matrix for the Jacobians

    Eigen::MatrixXd Jx = Eigen::MatrixXd::Zero(nr, nx);
    Eigen::MatrixXd Ju = Eigen::MatrixXd::Zero(nr, nu);

    features->calcDiff(Jx, Ju, x, u);

    data->Lx = d->r.transpose() * Jx;
    data->Lu = d->r.transpose() * Ju;
    data->Lxx = Jx.transpose() * Jx;
    data->Luu = Ju.transpose() * Ju;
    data->Lxu = Jx.transpose() * Ju;
  }

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x) {
    Data *d = static_cast<Data *>(data.get());
    features->calc(d->r, x);
    d->cost = Scalar(0.5) * d->r.dot(d->r);
  }

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x) {

    Data *d = static_cast<Data *>(data.get());
    // CHANGE THIS

    // create a matrix for the Jacobians

    Eigen::MatrixXd Jx = Eigen::MatrixXd::Zero(nr, nx);

    features->calcDiff(Jx, x);

    data->Lx = d->r.transpose() * Jx;
    data->Lxx = Jx.transpose() * Jx;
  }

  virtual boost::shared_ptr<ActionDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data) {

    boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
    if (d != NULL) {
      return true;
    } else {
      return false;
    }
  }

  virtual void print(std::ostream &os) const { os << "wrapper" << std::endl; }
};

class ActionModelQuim : public ActionModelAbstractTpl<double> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataQuim Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector2s Vector2s;

  ActionModelQuim(boost::shared_ptr<CollisionChecker> cl);

  boost::shared_ptr<CollisionChecker> cl;
  Eigen::Vector3d goal;

  virtual ~ActionModelQuim();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);
  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) {

    Data *d = static_cast<Data *>(data.get());

    const Scalar c = cos(x[2]);
    const Scalar s = sin(x[2]);
    const Scalar w_x = cost_weights_[0] * cost_weights_[0];
    const Scalar w_u = cost_weights_[1] * cost_weights_[1];
    d->Lx = (x - goal) * w_x;
    d->Lu = u * w_u;
    // why??
    d->Lxx.setZero();
    d->Luu.setZero();

    d->Lxx.diagonal().setConstant(w_x);
    d->Luu.diagonal().setConstant(w_u);

    d->Fx.setZero();
    d->Fu.setZero();

    d->Fx(0, 0) = 1;
    d->Fx(1, 1) = 1;
    d->Fx(2, 2) = 1;
    d->Fx(0, 2) = -s * u[0] * dt_;
    d->Fx(1, 2) = c * u[0] * dt_;
    d->Fu(0, 0) = c * dt_;
    d->Fu(1, 0) = s * dt_;
    d->Fu(2, 1) = dt_;

    // add my collision check

    std::vector<double> query{x[0], x[1], x[2]};
    auto out = cl->distanceWithFDiffGradient(query);
    auto dist = collision_weight * std::get<0>(out);
    auto grad = std::get<1>(out);
    Eigen::VectorXd v = collision_weight * Eigen::VectorXd::Map(grad.data(), 3);

    if (dist <= 0) {
      d->Lx += dist * v;
      d->Lxx += v * v.transpose();
    }
    d->Lxu.setZero();
  };
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x) {

    Data *d = static_cast<Data *>(data.get());

    const Scalar w_x = cost_weights_[0] * cost_weights_[0];
    d->Lx = (x - goal) * w_x;
    d->Lxx.setZero();
    d->Lxx.diagonal().setConstant(w_x);

    std::vector<double> query{x[0], x[1], x[2]};
    auto out = cl->distanceWithFDiffGradient(query);
    auto dist = collision_weight * std::get<0>(out);
    auto grad = std::get<1>(out);
    Eigen::VectorXd v = collision_weight * Eigen::VectorXd::Map(grad.data(), 3);

    if (dist <= 0.) {
      d->Lx += dist * v;
      d->Lxx += v * v.transpose();
    }

    d->Lxu.setZero();
  };
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

/* --- Details --------------------------------------------------------------
 */
/* --- Details --------------------------------------------------------------
 */
/* --- Details --------------------------------------------------------------
 */

using Scalar = double;

ActionModelQuim::ActionModelQuim(boost::shared_ptr<CollisionChecker> cl)
    : ActionModelAbstractTpl<Scalar>(
          boost::make_shared<StateVectorTpl<Scalar>>(3), 2, 6),
      dt_(Scalar(0.1)), cl(cl) {
  cost_weights_ << Scalar(10.), Scalar(1.);
}

ActionModelQuim::~ActionModelQuim() {}

std::vector<Eigen::Vector2d> obstacles{Eigen::Vector2d(.1, .1),
                                       Eigen::Vector2d(-.1, -.1)};

void ActionModelQuim::calc(
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

  const Scalar c = cos(x[2]);
  const Scalar s = sin(x[2]);
  d->xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, x[2] + u[1] * dt_;
  d->r.template head<3>() = cost_weights_[0] * (x - goal);
  d->r.segment(3, 2) = cost_weights_[1] * u;

  bool test = false;
  double dist = 0.;
  if (test) {
    double radius = .1;
    for (size_t i = 0; i < obstacles.size(); i++) {
      dist += std::min((x.head(2) - obstacles[i]).squaredNorm() - radius, 0.);
    }
  } else {
    // d < 0 is collision
    std::vector<double> query{x[0], x[1], x[2]};
    dist = std::min(std::get<0>(cl->distance(query)), 0.);
    // dist = std::get<0>(cl->distance(query));
  }
  d->r[5] = collision_weight * dist;

  d->cost = Scalar(0.5) * d->r.dot(d->r);
}

void ActionModelQuim::calc(
    const boost::shared_ptr<ActionDataAbstractTpl<Scalar>> &data,
    const Eigen::Ref<const VectorXs> &x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(state_->get_nx()) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  d->r.template head<3>() = cost_weights_[0] * (x - goal);
  d->r.segment(3, 2).setZero();

  bool test = false;
  double dist = 0.;
  if (test) {
    double radius = .1;
    for (size_t i = 0; i < obstacles.size(); i++) {
      dist += std::min((x.head(2) - obstacles[i]).squaredNorm() - radius, 0.);
    }
  } else {
    // d < 0 is collision
    std::vector<double> query{x[0], x[1], x[2]};
    dist = std::min(std::get<0>(cl->distance(query)), 0.);
  }
  d->r[5] = collision_weight * dist;
  d->cost = Scalar(0.5) * d->r.dot(d->r);
}

// void ActionModelQuim::calcDiff(const
// boost::shared_ptr<ActionDataAbstractTpl<Scalar> >& data,
//                                               const Eigen::Ref<const
//                                               VectorXs>& x, const
//                                               Eigen::Ref<const VectorXs>&
//                                               u)
//                                               {
//   if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " +
//                  std::to_string(state_->get_nx()) + ")");
//   }
//   if (static_cast<std::size_t>(u.size()) != nu_) {
//     throw_pretty("Invalid argument: "
//                  << "u has wrong dimension (it should be " +
//                  std::to_string(nu_) + ")");
//   }
//   Data* d = static_cast<Data*>(data.get());
//
//   const Scalar c = cos(x[2]);
//   const Scalar s = sin(x[2]);
//   const Scalar w_x = cost_weights_[0] * cost_weights_[0];
//   const Scalar w_u = cost_weights_[1] * cost_weights_[1];
//
//   double dist = 0 ;
//   double radius = 0;
//   // derivative of norm: || x || is   --> 1 / 2 /  || x || * 2  * x = x /
//   || x ||
//
//   Eigen::Vector3d D;
//   D.setZero();
//   for (size_t i = 0 ; i < obstacles.size() ; i++) {
//     dist = std::min( (x - obstacles[i]).squaredNorm() - radius , 0. );
//     if (dist  <  0)
//     {
//       D += 2 * (x - obstacles[i]);
//     }
//   }
//
//   d->Lx = x * w_x + D;
//   d->Lu = u * w_u;
//   d->Lxx.diagonal().setConstant(w_x);
//   d->Luu.diagonal().setConstant(w_u);
//   d->Fx(0, 2) = -s * u[0] * dt_;
//   d->Fx(1, 2) = c * u[0] * dt_;
//   d->Fu(0, 0) = c * dt_;
//   d->Fu(1, 0) = s * dt_;
//   d->Fu(2, 1) = dt_;
// }
//
// void ActionModelQuim::calcDiff(const
// boost::shared_ptr<ActionDataAbstractTpl<Scalar>>& data,
//                                               const Eigen::Ref<const
//                                               VectorXs>& x) {
//   if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
//     throw_pretty("Invalid argument: "
//                  << "x has wrong dimension (it should be " +
//                  std::to_string(state_->get_nx()) + ")");
//   }
//   Data* d = static_cast<Data*>(data.get());
//
//   const Scalar w_x = cost_weights_[0] * cost_weights_[0];
//   d->Lx = x * w_x;
//   d->Lxx.diagonal().setConstant(w_x);
// }

boost::shared_ptr<ActionDataAbstractTpl<Scalar>> ActionModelQuim::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

bool ActionModelQuim::checkData(
    const boost::shared_ptr<ActionDataAbstract> &data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

void ActionModelQuim::print(std::ostream &os) const {
  os << "ActionModelUnicycle {dt=" << dt_ << "}";
}

const typename MathBaseTpl<Scalar>::Vector2s &
ActionModelQuim::get_cost_weights() const {
  return cost_weights_;
}

void ActionModelQuim::set_cost_weights(
    const typename MathBase::Vector2s &weights) {
  cost_weights_ = weights;
}

Scalar ActionModelQuim::get_dt() const { return dt_; }

void ActionModelQuim::set_dt(const Scalar dt) {
  if (dt <= 0)
    throw_pretty("Invalid argument: dt should be strictly positive.");
  dt_ = dt;
}

int test(int argc, char *argv[]) {
  bool CALLBACKS = true;
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
  crocoddyl::SolverFDDP ddp(problem);
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
  return 0;
}

template <typename Base, typename Derived>
boost::shared_ptr<Base> cast(boost::shared_ptr<Derived> p) {
  return boost::static_pointer_cast<Base>(p);
}

template <typename Derived>
boost::shared_ptr<crocoddyl::ActionModelAbstract>
to_am_base(boost::shared_ptr<Derived> am) {
  return boost::static_pointer_cast<crocoddyl::ActionModelAbstract>(am);
};

void print_data(boost::shared_ptr<ActionDataAbstractTpl<double>> data) {
  std::cout << "***\n";
  std::cout << "xnext\n" << data->xnext << std::endl;
  std::cout << "Fx:\n" << data->Fx << std::endl;
  std::cout << "Fu:\n" << data->Fu << std::endl;
  std::cout << "r:" << data->r.transpose() << std::endl;
  std::cout << "cost:" << data->cost << std::endl;
  std::cout << "Lx:" << data->Lx.transpose() << std::endl;
  std::cout << "Lu:" << data->Lu.transpose() << std::endl;
  std::cout << "Lxx:\n" << data->Lxx << std::endl;
  std::cout << "Luu:\n" << data->Luu << std::endl;
  std::cout << "Lxu:\n" << data->Lxu << std::endl;
  std::cout << "***\n";
}

#if 0
void check_data() {

  auto model_run_numdiff =
      boost::make_shared<crocoddyl::ActionModelNumDiff>(model_run, true);
  model_run_numdiff->set_disturbance(1e-4);

  std::vector<ptr<crocoddyl::ActionModelAbstract>> runningModels_q(N, amq);

  // Formulating the optimal control problem
  ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                     model_terminal);

  Eigen::Vector3d x =
      boost::static_pointer_cast<Derived>(model_terminal)->goal +
      Eigen::Vector3d(2.2, .3, .1);
  {
    std::cout << "A" << std::endl;
    auto data = model_run->createData();
    model_run->checkData(data);

    model_run->calc(data, x, us.at(0));
    model_run->calcDiff(data, x, us.at(0));
    print_data(data);
  }
  {
    std::cout << "B" << std::endl;
    auto data = model_run_numdiff->createData();
    model_run_numdiff->calc(data, x, us.at(0));
    model_run_numdiff->calcDiff(data, x, us.at(0));
    print_data(data);
  }


}
#endif

void check_dyn(boost::shared_ptr<Dynamics> dyn, double eps) {

  int nx = dyn->nx;
  int nu = dyn->nu;

  Eigen::MatrixXd Fx(nx, nx);
  Eigen::MatrixXd Fu(nx, nu);
  Fx.setZero();
  Fu.setZero();

  Eigen::VectorXd x(nx);
  Eigen::VectorXd u(nu);
  x.setRandom();
  u.setRandom();

  dyn->calcDiff(Fx, Fu, x, u);

  // compute the same using finite diff

  Eigen::MatrixXd FxD(nx, nx);
  FxD.setZero();
  Eigen::MatrixXd FuD(nx, nu);
  FuD.setZero();

  Eigen::VectorXd xnext(nx);
  xnext.setZero();
  dyn->calc(xnext, x, u);
  for (size_t i = 0; i < nx; i++) {
    Eigen::MatrixXd xe;
    xe = x;
    xe(i) += eps;
    Eigen::VectorXd xnexte(nx);
    xnexte.setZero();
    dyn->calc(xnexte, xe, u);
    auto df = (xnexte - xnext) / eps;
    FxD.col(i) = df;
  }

  for (size_t i = 0; i < nu; i++) {
    Eigen::MatrixXd ue;
    ue = u;
    ue(i) += eps;
    Eigen::VectorXd xnexte(nx);
    xnexte.setZero();
    dyn->calc(xnexte, x, ue);
    auto df = (xnexte - xnext) / eps;
    FuD.col(i) = df;
  }

  std::cout << "Analytical " << std::endl;
  std::cout << "Fx\n" << Fx << std::endl;
  std::cout << "Fu\n" << Fu << std::endl;
  std::cout << "Finite Diff " << std::endl;
  std::cout << "Fx\n" << FxD << std::endl;
  std::cout << "Fu\n" << FuD << std::endl;

  assert((Fx - FxD).cwiseAbs().maxCoeff() < 10 * eps);
  assert((Fu - FuD).cwiseAbs().maxCoeff() < 10 * eps);
}

void quim_test(int argc, char *argv[]) {

  double th_stop = 1e-2;
  double init_reg = 1e-1;
  int max_iter = 30;
  bool CALLBACKS = true;
  bool use_finite_diff = false;

  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  std::string env_file;
  std::string init_guess;
  desc.add_options()("help", "produce help message")(
      "env,e", po::value<std::string>(&env_file)->required())(
      "init,i", po::value<std::string>(&init_guess)->required());

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return;
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return;
  }

  YAML::Node init = YAML::LoadFile(init_guess);
  YAML::Node env = YAML::LoadFile(env_file);

  std::vector<std::vector<double>> states;

  for (const auto &state : init["result"][0]["states"]) {
    std::vector<double> p;
    for (const auto &elem : state) {
      p.push_back(elem.as<double>());
    }
    states.push_back(p);
  }

  int N = states.size() - 1;

  std::vector<std::vector<double>> actions;
  for (const auto &state : init["result"][0]["actions"]) {
    std::vector<double> p;
    for (const auto &elem : state) {
      p.push_back(elem.as<double>());
    }
    actions.push_back(p);
  }

  std::vector<double> start, goal;
  for (const auto &e : env["robots"][0]["start"]) {
    start.push_back(e.as<double>());
  }

  for (const auto &e : env["robots"][0]["goal"]) {
    goal.push_back(e.as<double>());
  }

  int verbose = 1;

  if (verbose) {
    std::cout << "states " << std::endl;
    for (auto &x : states) {
      for (auto &e : x) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "actions " << std::endl;
    for (auto &x : actions) {
      for (auto &e : x) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
    }
  }

  auto cl = mk<CollisionChecker>();
  cl->load(env_file);

  bool repair_init_guess = true;

  if (repair_init_guess) {
    for (size_t i = 1; i < N + 1; i++) {
      states.at(i).at(2) =
          states.at(i - 1).at(2) +
          diff_angle(states.at(i).at(2), states.at(i - 1).at(2));
    }
  }

  if (verbose) {
    std::cout << "states " << std::endl;
    for (auto &x : states) {
      for (auto &e : x) {
        std::cout << e << " ";
      }
      std::cout << std::endl;
    }
  }

  std::string name = env["robots"][0]["type"].as<std::string>();

  size_t nx, nu;

  ptr<Cost> feats_run;
  ptr<Cost> feats_terminal;
  ptr<crocoddyl::ActionModelAbstract> am_run;
  ptr<crocoddyl::ActionModelAbstract> am_terminal;
  ptr<Dynamics> dyn;

  if (name == "unicycle_first_order_0") {
    nx = 3;
    nu = 2;
    dyn = mk<Dynamics_unicycle>();
    ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, cl);

    boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
        true, true, true};

    ptr<Cost> control_feature =
        mk<Control_cost>(nx, nu, nu, Eigen::Vector2d(1., 1.));
    ptr<Cost> state_feature =
        mk<State_cost>(nx, nu, nx, Eigen::Vector3d(100., 100., 100.),
                       Eigen::VectorXd::Map(goal.data(), goal.size()));

    feats_run =
        mk<All_cost>(nx, nu,
                     cl_feature->nr + 
                      control_feature->nr,
                     std::vector<ptr<Cost>>{
                       cl_feature, 
                       control_feature});

    feats_terminal = mk<All_cost>(nx, nu, state_feature->nr,
                                  std::vector<ptr<Cost>>{state_feature});

    am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));
    am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
    am_run->set_u_ub(Eigen::Vector2d(.5, .5));
    am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

  } else if (name == "unicycle_second_order_0") {
    nx = 5;
    nu = 2;
    dyn = mk<Dynamics_unicycle2>();
    ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
        true, true, true, false, false};

    ptr<Cost> control_feature =
        mk<Control_cost>(nx, nu, nu, Eigen::Vector2d(5., 5.));

    ptr<Cost> state_feature =
        mk<State_cost>(nx, nu, nx, 100. * Eigen::VectorXd::Ones(nx),
                       Eigen::VectorXd::Map(goal.data(), goal.size()));

    ptr<Cost> state_feature_reg = mk<State_cost>(
        nx, nu, nx, 0.0 * Eigen::VectorXd::Ones(nx), Eigen::VectorXd::Zero(nx));

    feats_run = mk<All_cost>(
        nx, nu, cl_feature->nr + control_feature->nr + state_feature_reg->nr,
        std::vector<ptr<Cost>>{cl_feature, control_feature, state_feature_reg});

    feats_terminal = mk<All_cost>(nx, nu, state_feature->nr,
                                  std::vector<ptr<Cost>>{state_feature});

    am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));
    am_run->set_u_lb(Eigen::Vector2d(-.25, -.25));
    am_run->set_u_ub(Eigen::Vector2d(.25, .25));
    am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

    // TODO: option to regularize w.r.t. intial guess.

  } else {
    throw -1;
  }

  Eigen::VectorXd x0 = Eigen::VectorXd::Map(start.data(), start.size());
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(nu));

  bool use_warmstart = true;

  if (use_warmstart) {
    for (size_t t = 0; t < N + 1; t++) {
      xs.at(t) = Eigen::VectorXd::Map(states.at(t).data(), states.at(t).size());
    }
    for (size_t t = 0; t < N; t++) {
      us.at(t) =
          Eigen::VectorXd::Map(actions.at(t).data(), actions.at(t).size());
    }
  }


  if (use_finite_diff) {
    auto lb = am_run->get_u_lb();
    auto ub = am_run->get_u_ub();
    am_run = mk<crocoddyl::ActionModelNumDiff>(am_run, true);
    boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_run)
        ->set_disturbance(1e-4);
    am_run->set_u_lb(lb);
    am_run->set_u_ub(ub);

    auto lbT = am_terminal->get_u_lb();
    auto ubT = am_terminal->get_u_ub();
    am_terminal = mk<crocoddyl::ActionModelNumDiff>(am_terminal, true);
    boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_terminal)
        ->set_disturbance(1e-4);
    am_terminal->set_u_lb(lbT);
    am_terminal->set_u_ub(ubT);
  }

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> amq_runs(
      N, am_run);

  ptr<crocoddyl::ShootingProblem> problem =
      mk<crocoddyl::ShootingProblem>(x0, amq_runs, am_terminal);

  crocoddyl::SolverBoxFDDP ddp(problem);
  ddp.set_th_stop(th_stop);

  if (CALLBACKS) {
    std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(mk<crocoddyl::CallbackVerbose>());
    ddp.setCallbacks(cbs);
  }

  {
    std::cout <<  problem->calc(xs, us) << std::endl;
    crocoddyl::Timer timer;
    problem->calc(xs, us) ; 
    std::cout << "calc time: " << timer.get_duration() << std::endl;
  }

  {
    crocoddyl::Timer timer;
    problem->calcDiff(xs, us); 
    std::cout << "calcDiff time: " << timer.get_duration() << std::endl;
  }

  crocoddyl::Timer timer;
  ddp.solve(xs, us, max_iter, false, init_reg);
  std::cout << "time: " << timer.get_duration()<< std::endl;

  std::cout << "solution" << std::endl;
  std::cout << problem->calc(ddp.get_xs(), ddp.get_us()) << std::endl;

  std::ofstream results_txt("out.txt");

  const Eigen::IOFormat fmt(6, Eigen::DontAlignCols, ",", " ", "", "", "", "");

  for (auto &x : ddp.get_xs()) {
    results_txt << x.transpose().format(fmt) << std::endl;
  }
  results_txt << "---" << std::endl;
  for (auto &u : ddp.get_us()) {
    results_txt << u.transpose().format(fmt) << std::endl;
  }

  // store in the good format

  // write the results.
  std::ofstream results_yaml("out_croco.yaml");
  // out << std::setprecision(std::numeric_limits<double>::digits10 + 1);
  results_yaml << "result:" << std::endl;
  results_yaml << "  - states:" << std::endl;
  for (auto &x : ddp.get_xs()) {
    results_yaml << "      - [" << x[0] << "," << x[1] << ","
                 << std::remainder(x[2], 2 * M_PI) << "]" << std::endl;
  }

  results_yaml << "    actions:" << std::endl;
  for (auto &u : ddp.get_us()) {
    results_yaml << "      - [" << u[0] << "," << u[1] << "]" << std::endl;
  }
}

void double_integ(int argc, char *argv[]) {
  // check the dynamics
  ptr<Dynamics> dyn = mk<Dynamics_unicycle2>();
  check_dyn(dyn, 1e-5);

  ptr<Dynamics> dyn_free_time = mk<Dynamics_unicycle2>(true);
  check_dyn(dyn_free_time, 1e-5);

  // generate the problem... to continue
}

void test_unifree() {

  ptr<Dynamics> dyn = mk<Dynamics_unicycle>();
  check_dyn(dyn, 1e-5);

  ptr<Dynamics> dyn_free_time = mk<Dynamics_unicycle>(true);
  check_dyn(dyn, 1e-5);
}

int main(int argc, char *argv[]) {

  // test(argc, argv);

  double_integ(argc, argv);

  test_unifree();

  // throw -1;

  quim_test(argc, argv);

  // double integrator

  //
}

// TODO: convert into tests
//
// (opti) ⋊> ~/s/w/k/build on feat_croco ⨯ make && ./test_croco --env
// ../benchmark/unicycle_first_order_0/bugtr ap_0.yaml --init
// ../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml

// (opti) ⋊> ~/s/w/k/build on feat_croco ⨯ make &&   ./test_croco --env
// ../benchmark/unicycle_first_order_0/par allelpark_0.yaml --init
// ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml
