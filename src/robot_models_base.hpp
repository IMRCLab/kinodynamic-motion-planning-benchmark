#pragma once
#include "Eigen/Core"
#include "croco_macros.hpp"
#include "fcl/broadphase/broadphase_collision_manager.h"
#include "general_utils.hpp"
#include "math_utils.hpp"
#include <algorithm>
// #include <boost/serialization/list.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/version.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>
#include <yaml-cpp/node/node.h>

struct Obstacle {
  std::string type;
  Eigen::VectorXd size;
  Eigen::VectorXd center;
};

using Transform3d = Eigen::Transform<double, 3, Eigen::Isometry>;

// p1 is in environemnt
// p2 is in robot
// d < 0 if there is collision (SDF)
struct CollisionOut {
  double distance;
  Eigen::Vector3d p1;
  Eigen::Vector3d p2;
  void write(std::ostream &out) const {
    out << STR_(distance) << std::endl;
    out << STR_V(p1) << std::endl;
    out << STR_V(p2) << std::endl;
  }
};

static std::vector<Eigen::VectorXd> DEFAULT_V;
static Eigen::VectorXd DEFAULT_E;

// next: time optimal linear, use so2 space, generate motion primitives
struct StateQ {

  size_t nx;
  size_t ndx;

  StateQ(size_t nx, size_t ndx) : nx(nx), ndx(ndx) {}

  virtual Eigen::VectorXd zero() const { ERROR_WITH_INFO("not implemented"); }

  virtual Eigen::VectorXd rand() const { ERROR_WITH_INFO("not implemented"); }

  virtual void diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                    const Eigen::Ref<const Eigen::VectorXd> &x1,
                    Eigen::Ref<Eigen::VectorXd> dxout) const {

    (void)x0;
    (void)x1;
    (void)dxout;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &dx,
                         Eigen::Ref<Eigen::VectorXd> xout) const {

    (void)x;
    (void)dx;
    (void)xout;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                     const Eigen::Ref<const Eigen::VectorXd> &x1,
                     Eigen::Ref<Eigen::MatrixXd> Jfirst,
                     Eigen::Ref<Eigen::MatrixXd> Jsecond) const {

    (void)x0;
    (void)x1;
    (void)Jfirst;
    (void)Jsecond;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void Jintegrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &dx,
                          Eigen::Ref<Eigen::MatrixXd> Jfirst,
                          Eigen::Ref<Eigen::MatrixXd> Jsecond) const {

    (void)x;
    (void)dx;
    (void)Jfirst;
    (void)Jsecond;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void JintegrateTransport(const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &dx,
                                   Eigen::Ref<Eigen::MatrixXd> Jin) const {
    (void)x;
    (void)dx;
    (void)Jin;

    ERROR_WITH_INFO("not implemented");
  }
};

struct CompoundState2 : StateQ {

  std::shared_ptr<StateQ> s1;
  std::shared_ptr<StateQ> s2;

  CompoundState2(std::shared_ptr<StateQ> s1, std::shared_ptr<StateQ> s2);
  virtual ~CompoundState2(){};

  virtual Eigen::VectorXd zero() const override;

  virtual Eigen::VectorXd rand() const override;

  virtual void diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                    const Eigen::Ref<const Eigen::VectorXd> &x1,
                    Eigen::Ref<Eigen::VectorXd> dxout) const override;

  virtual void integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &dx,
                         Eigen::Ref<Eigen::VectorXd> xout) const override;

  virtual void Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                     const Eigen::Ref<const Eigen::VectorXd> &x1,
                     Eigen::Ref<Eigen::MatrixXd> Jfirst,
                     Eigen::Ref<Eigen::MatrixXd> Jsecond) const override;
};

struct RnSOn : StateQ {

  size_t nR;
  size_t nSO2;
  const std::vector<size_t> so2_indices;
  RnSOn(size_t nR, size_t nSO2, const std::vector<size_t> &so2_indices)
      : StateQ(nR + nSO2, nR + nSO2), so2_indices(so2_indices) {
    CHECK_EQ(so2_indices.size(), nSO2, AT);
  }

  virtual ~RnSOn(){};

  virtual Eigen::VectorXd zero() const override;

  virtual Eigen::VectorXd rand() const override;

  virtual void diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                    const Eigen::Ref<const Eigen::VectorXd> &x1,
                    Eigen::Ref<Eigen::VectorXd> dxout) const override;

  virtual void integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &dx,
                         Eigen::Ref<Eigen::VectorXd> xout) const override;

  virtual void Jintegrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &dx,
                          Eigen::Ref<Eigen::MatrixXd> Jfirst,
                          Eigen::Ref<Eigen::MatrixXd> Jsecond) const override;

  virtual void Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                     const Eigen::Ref<const Eigen::VectorXd> &x1,
                     Eigen::Ref<Eigen::MatrixXd> Jfirst,
                     Eigen::Ref<Eigen::MatrixXd> Jsecond) const override;
};

struct Rn : StateQ {

  size_t nR;
  Rn(size_t nR) : StateQ(nR, nR) {}
  virtual ~Rn() {}

  virtual Eigen::VectorXd zero() const override;

  virtual Eigen::VectorXd rand() const override;

  virtual void diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                    const Eigen::Ref<const Eigen::VectorXd> &x1,
                    Eigen::Ref<Eigen::VectorXd> dxout) const override;

  virtual void integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &dx,
                         Eigen::Ref<Eigen::VectorXd> xout) const override;

  virtual void Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                     const Eigen::Ref<const Eigen::VectorXd> &x1,
                     Eigen::Ref<Eigen::MatrixXd> Jfirst,
                     Eigen::Ref<Eigen::MatrixXd> Jsecond) const override;

  virtual void Jintegrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &dx,
                          Eigen::Ref<Eigen::MatrixXd> Jfirst,
                          Eigen::Ref<Eigen::MatrixXd> Jsecond) const override;
  // const Jcomponent firstsecond = both, const AssignmentOp = setto) const;
};

struct Model_robot {

  size_t nx;
  size_t nu;
  size_t nx_pr; // the first nx_pr components are about position/orientation
  size_t nx_col = 0; // only the first nx_col variables have non zero gradient

  size_t nr_reg;
  size_t nr_ineq;

  bool invariance_reuse_col_shape = true;
  bool is_2d;
  size_t translation_invariance = 0; // e.g. 1, 2 , 3, ...
  std::vector<std::string> x_desc;
  std::vector<std::string> u_desc;
  std::string name;
  double ref_dt;

  Eigen::VectorXd u_ref; // used for cost
  Eigen::VectorXd u_0;   // used for init guess
  Eigen::VectorXd u_lb;
  Eigen::VectorXd u_ub;

  Eigen::VectorXd x_ub;
  Eigen::VectorXd x_lb;
  bool uniform_sampling_u = true;

  // TODO: transition towards this API. The robot model should include
  // regularization features/ineqs...
  virtual void regularization_cost(Eigen::Ref<Eigen::VectorXd> r,
                                   const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &u) {

    NOT_IMPLEMENTED;
  }

  virtual void
  regularization_cost_diff(Eigen::Ref<Eigen::MatrixXd> Jx,
                           Eigen::Ref<Eigen::MatrixXd> Ju,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) {

    NOT_IMPLEMENTED;
  }

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) {
    xout = xin;
  }

  // State
  std::shared_ptr<StateQ> state;

  // crocoddyl::StateAbstractTpl> state;

  Eigen::VectorXd u_weight;  // For optimization
  Eigen::VectorXd x_weightb; //

  Eigen::VectorXd distance_weights; // weights for the OMPL wrapper.
  Eigen::VectorXd r_weight;         // weights for state diff in optimization.

  Eigen::VectorXd __v;       // data
  Eigen::MatrixXd __Jv_x;    // data
  Eigen::MatrixXd __Jv_u;    // data
  Eigen::MatrixXd __Jfirst;  // data
  Eigen::MatrixXd __Jsecond; // data

  std::vector<Transform3d> ts_data;   // data
  std::vector<CollisionOut> col_outs; // data
  //
  //
  //
  Model_robot() = default;
  Model_robot(std::shared_ptr<StateQ> state, size_t nu);

  // Returns x_0 for optimization. Can depend on a reference point. Default:
  // return the ref point. Reasoning: in some systems, it is better to set the
  // orientation/velocities of x0 to zero
  virtual Eigen::VectorXd get_x0(const Eigen::VectorXd &x) { return x; }

  void set_position_ub(const Eigen::Ref<const Eigen::VectorXd> &p_ub) {
    CHECK_EQ(static_cast<size_t>(p_ub.size()), translation_invariance, AT);
    x_ub.head(translation_invariance) = p_ub;
  }

  void set_position_lb(const Eigen::Ref<const Eigen::VectorXd> &p_lb) {
    CHECK_EQ(static_cast<size_t>(p_lb.size()), translation_invariance, AT);
    x_lb.head(translation_invariance) = p_lb;
  }

  const Eigen::VectorXd &get_x_ub() { return x_ub; }
  const Eigen::VectorXd &get_x_lb() { return x_lb; }

  const size_t &get_translation_invariance() { return translation_invariance; }
  const size_t &get_nx() { return nx; }
  const size_t &get_nu() { return nu; }

  const Eigen::VectorXd &get_u_ub() { return u_ub; }
  const Eigen::VectorXd &get_u_lb() { return u_lb; }
  const Eigen::VectorXd &get_u_ref() { return u_ref; }

  std::vector<std::string> &get_x_desc() { return x_desc; }
  std::vector<std::string> &get_u_desc() { return u_desc; }

  virtual double cost(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &u) const;

  virtual double traj_cost(const std::vector<Eigen::VectorXd> &xs,
                           const std::vector<Eigen::VectorXd> &us) const;

  virtual void
  setPositionBounds(const Eigen::Ref<const Eigen::VectorXd> &p_lb,
                    const Eigen::Ref<const Eigen::VectorXd> &p_ub) {
    CHECK_EQ(p_lb.size(), p_ub.size(), AT);
    CHECK((static_cast<size_t>(p_lb.size()) == 2 ||
           static_cast<size_t>(p_lb.size()) == 3),
          AT);

    size_t p_nx = p_lb.size();

    x_ub.head(p_nx) = p_ub;
    x_lb.head(p_nx) = p_lb;
  }

  virtual void write_params(std::ostream &out) {
    (void)out;
    ERROR_WITH_INFO("not implemented");
  }

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
                     const Eigen::Ref<const Eigen::VectorXd> &d,
                     const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

  virtual bool is_control_valid(const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual bool is_state_valid(const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void stepR4(Eigen::Ref<Eigen::VectorXd> xnext,
                      const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

  virtual void stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

  virtual void constraintsIneq(Eigen::Ref<Eigen::VectorXd> r,
                               const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &u) {

    (void)r;
    (void)x;
    (void)u;

    NOT_IMPLEMENTED;
  }

  virtual void constraintsIneqDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                                   Eigen::Ref<Eigen::MatrixXd> Ju,
                                   const Eigen::Ref<const Eigen::VectorXd> x,
                                   const Eigen::Ref<const Eigen::VectorXd> &u) {

    (void)Jx;
    (void)Ju;
    (void)x;
    (void)u;

    NOT_IMPLEMENTED;
  }

  // virtual void stepDiffdt(Eigen::Ref<Eigen::MatrixXd> Fx,
  //                         Eigen::Ref<Eigen::MatrixXd> Fu,
  //                         const Eigen::Ref<const Eigen::VectorXd> &x,
  //                         const Eigen::Ref<const Eigen::VectorXd> &u,
  //                         double dt);

  virtual void stepDiff_with_v(Eigen::Ref<Eigen::MatrixXd> Fx,
                               Eigen::Ref<Eigen::MatrixXd> Fu,
                               Eigen::Ref<Eigen::VectorXd> __v,
                               const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &u,
                               double dt);

  // virtual void stepDiffdtX(Eigen::Ref<Eigen::MatrixXd> Fx,
  //                          Eigen::Ref<Eigen::MatrixXd> Fu,
  //                          const Eigen::Ref<const Eigen::VectorXd> &x,
  //                          const Eigen::Ref<const Eigen::VectorXd> &u,
  //                          double dt);

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y);

  virtual void rollout(const Eigen::Ref<const Eigen::VectorXd> &x0,
                       const std::vector<Eigen::VectorXd> &us,
                       std::vector<Eigen::VectorXd> &xs) {
    CHECK_EQ(us.size() + 1, xs.size(), AT);

    xs.at(0) = x0;
    for (size_t i = 0; i < us.size(); i++) {
      step(xs.at(i + 1), xs.at(i), us.at(i), ref_dt);
    }
  }

  virtual void transform_state(const Eigen::Ref<const Eigen::VectorXd> &p,
                               const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) {

    xout = xin;
    xout.head(translation_invariance) += p;
  }

  virtual void canonical_state(const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) {

    xout = xin;
    xout.head(translation_invariance).setZero();
  }

  virtual void offset(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> p) {

    CHECK_EQ(static_cast<size_t>(p.size()), translation_invariance, AT);
    p = xin.head(translation_invariance);
  }

  virtual size_t get_offset_dim() { return translation_invariance; }

  virtual void transform_primitive(const Eigen::Ref<const Eigen::VectorXd> &p,
                                   const std::vector<Eigen::VectorXd> &xs_in,
                                   const std::vector<Eigen::VectorXd> &us_in,
                                   std::vector<Eigen::VectorXd> &xs_out,
                                   std::vector<Eigen::VectorXd> &us_out) {

    // basic transformation is translation invariance
    CHECK_EQ(p.size(), translation_invariance, AT);
    // TODO: avoid memory allocation inside this function!!

    CHECK_EQ(us_out.size(), us_in.size(), AT);
    CHECK_EQ(xs_out.size(), xs_in.size(), AT);
    CHECK_EQ(xs_out.front().size(), xs_in.front().size(), AT);
    CHECK_EQ(us_out.front().size(), us_in.front().size(), AT);

    for (size_t i = 0; i < us_in.size(); i++) {
      us_out[i] = us_in[i];
    }

    if (translation_invariance) {
      for (size_t i = 0; i < xs_in.size(); i++) {
        xs_out[i] = xs_in[i];
        xs_out[i].head(translation_invariance) += p;
      }
    } else {
      for (size_t i = 0; i < xs_in.size(); i++) {
        xs_out[i] = xs_in[i];
      }
    }
  }

  // x1 - x0
  virtual void state_diff(Eigen::Ref<Eigen::VectorXd> r,
                          const Eigen::Ref<const Eigen::VectorXd> &x0,
                          const Eigen::Ref<const Eigen::VectorXd> &x1) {

    // lets just use state
    CHECK_EQ(r_weight.size(), r.size(), AT);
    CHECK_EQ(x0.size(), x1.size(), AT);
    state->diff(x0, x1, r);
    r.array() *= r_weight.array();
  }

  virtual void state_diffDiff(Eigen::Ref<Eigen::MatrixXd> Jx0,
                              Eigen::Ref<Eigen::MatrixXd> Jx1,
                              const Eigen::Ref<const Eigen::VectorXd> &x0,
                              const Eigen::Ref<const Eigen::VectorXd> &x1) {

    CHECK_EQ(x0.size(), x1.size(), AT);
    CHECK_EQ(Jx0.cols(), Jx1.cols(), AT);
    CHECK_EQ(Jx0.rows(), Jx1.rows(), AT);

    state->Jdiff(x0, x1, Jx0, Jx1);
    Jx0.diagonal().array() *= r_weight.array();
    Jx1.diagonal().array() *= r_weight.array();
  }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x);

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt);

  virtual double lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y);

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) { (void)x; }

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) {

    (void)x;
    (void)y;
    NOT_IMPLEMENTED;
  }

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) {

    (void)x;
    (void)y;
    NOT_IMPLEMENTED;
  }

  std::vector<std::shared_ptr<fcl::CollisionGeometryd>> collision_geometries;
  std::shared_ptr<fcl::BroadPhaseCollisionManagerd> env;

  virtual void collision_distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  CollisionOut &cout);

  // 1: No collision
  // 0: collision
  virtual bool collision_check(const Eigen::Ref<const Eigen::VectorXd> &x);

  // compute the Jacobians/Gradient using finite diff!
  // TODO: use Point-Point distance approximation to compute the gradient!

  virtual void
  collision_distance_diff(Eigen::Ref<Eigen::VectorXd> dd, double &f,
                          const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void transformation_collision_geometries(
      const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts);

  virtual ~Model_robot() = default;
};

void linearInterpolation(const Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &x, double t_query,
                         const StateQ &state, Eigen::Ref<Eigen::VectorXd> out,
                         Eigen::Ref<Eigen::VectorXd> Jx);

struct Interpolator {

  Eigen::VectorXd times;
  std::vector<Eigen::VectorXd> x;
  std::shared_ptr<StateQ> state;

  Interpolator(const Eigen::VectorXd &times,
               const std::vector<Eigen::VectorXd> &x)
      : Interpolator(times, x, std::make_shared<Rn>(x.front().size())) {}

  Interpolator(const Eigen::VectorXd &times,
               const std::vector<Eigen::VectorXd> &x,
               const std::shared_ptr<StateQ> &state)
      : times(times), x(x), state(state) {
    CHECK_EQ(static_cast<size_t>(times.size()), x.size(), AT);
  }

  void inline interpolate(double t_query, Eigen::Ref<Eigen::VectorXd> out,
                          Eigen::Ref<Eigen::VectorXd> J) {
    linearInterpolation(times, x, t_query, *state, out, J);
  }
};
