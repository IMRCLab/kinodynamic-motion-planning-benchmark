#include "pinocchio/math/fwd.hpp"
#include "pinocchio/multibody/liegroup/liegroup.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>
#include <yaml-cpp/node/iterator.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/node/parse.h>
#include <yaml-cpp/yaml.h>

#include "Eigen/Core"
#include "croco_macros.hpp"

#include <fcl/fcl.h>

#include "fcl/broadphase/broadphase_collision_manager.h"
#include "fcl/broadphase/broadphase_dynamic_AABB_tree.h"
#include "fcl/broadphase/default_broadphase_callbacks.h"
#include "fcl/geometry/shape/box.h"
#include "fcl/geometry/shape/sphere.h"
#include "general_utils.hpp"
#include "math_utils.hpp"
#include "robot_models_base.hpp"

using vstr = std::vector<std::string>;
using V2d = Eigen::Vector2d;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;
using Vxd = Eigen::VectorXd;
using V1d = Eigen::Matrix<double, 1, 1>;

double low__ = -std::sqrt(std::numeric_limits<double>::max());
double max__ = std::sqrt(std::numeric_limits<double>::max());

using namespace pinocchio;
// using namespace crocoddyl;

CompoundState2::CompoundState2(std::shared_ptr<StateQ> s1,
                               std::shared_ptr<StateQ> s2)
    : StateQ(s1->nx + s2->nx, s1->ndx + s2->ndx), s1(s1), s2(s2) {}

Eigen::VectorXd CompoundState2::zero() const {

  Eigen::VectorXd z1 = s1->zero();
  Eigen::VectorXd z2 = s2->zero();
  Eigen::VectorXd vec_joined(z1.size() + z2.size());

  vec_joined << z1, z2;
  return vec_joined;
}

Eigen::VectorXd CompoundState2::rand() const {

  Eigen::VectorXd z1 = s1->rand();
  Eigen::VectorXd z2 = s2->rand();
  Eigen::VectorXd vec_joined(z1.size() + z2.size());

  vec_joined << z1, z2;
  return vec_joined;
}

void CompoundState2::diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                          const Eigen::Ref<const Eigen::VectorXd> &x1,
                          Eigen::Ref<Eigen::VectorXd> dxout) const {

  s1->diff(x0.head(s1->nx), x1.head(s1->nx), dxout.head(s1->ndx));
  s2->diff(x0.tail(s2->nx), x1.tail(s2->nx), dxout.tail(s2->ndx));
}

void CompoundState2::integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &dx,
                               Eigen::Ref<Eigen::VectorXd> xout) const {

  s1->integrate(x.head(s1->nx), dx.head(s1->nx), xout.head(s1->ndx));
  s2->integrate(x.tail(s2->nx), dx.tail(s2->nx), xout.tail(s2->ndx));
}

void CompoundState2::Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                           const Eigen::Ref<const Eigen::VectorXd> &x1,
                           Eigen::Ref<Eigen::MatrixXd> Jfirst,
                           Eigen::Ref<Eigen::MatrixXd> Jsecond) const {

  s1->Jdiff(x0.head(s1->ndx), x1.head(s1->ndx),
            Jfirst.block(0, 0, s1->ndx, s1->ndx),
            Jsecond.block(0, 0, s1->ndx, s1->ndx));

  s2->Jdiff(x0.tail(s1->ndx), x1.tail(s1->ndx),
            Jfirst.block(s1->ndx, s1->ndx, s2->ndx, s2->ndx),
            Jsecond.block(s1->ndx, s1->ndx, s2->ndx, s2->ndx));
}

Eigen::VectorXd Rn::zero() const {
  Vxd out = Vxd::Zero(nx);
  return out;
}

Eigen::VectorXd Rn::rand() const {

  Vxd out = Vxd::Random(nx);
  return out;
}

void Rn::diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
              const Eigen::Ref<const Eigen::VectorXd> &x1,
              Eigen::Ref<Eigen::VectorXd> dxout) const {

  dxout = x1 - x0;
}

void Rn::integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &dx,
                   Eigen::Ref<Eigen::VectorXd> xout) const {

  xout = x + dx;
}

void Rn::Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
               const Eigen::Ref<const Eigen::VectorXd> &x1,
               Eigen::Ref<Eigen::MatrixXd> Jfirst,
               Eigen::Ref<Eigen::MatrixXd> Jsecond) const {

  (void)x0;
  (void)x1;
  Jfirst.diagonal().setConstant(-1.);
  Jsecond.diagonal().setConstant(1.);
}

void Rn::Jintegrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &dx,
                    Eigen::Ref<Eigen::MatrixXd> Jfirst,
                    Eigen::Ref<Eigen::MatrixXd> Jsecond) const {
  (void)x;
  (void)dx;
  Jfirst.diagonal().array() = 1;
  Jsecond.diagonal().array() = 1;
}

Eigen::VectorXd RnSOn::zero() const {
  Vxd out = Vxd::Zero(nx);
  return out;
}

Eigen::VectorXd RnSOn::rand() const {

  Vxd out = Vxd::Random(nx);

  for (auto &i : so2_indices) {
    out(i) *= M_PI;
  }
  return out;
}

void RnSOn::diff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                 const Eigen::Ref<const Eigen::VectorXd> &x1,
                 Eigen::Ref<Eigen::VectorXd> dxout) const {

  CHECK_EQ(x0.size(), x1.size(), AT);
  CHECK_EQ(dxout.size(), x1.size(), AT);
  dxout = x1 - x0;

  for (auto &i : so2_indices) {
    auto &diff = dxout(i);

    if (diff > M_PI) {
      diff -= 2 * M_PI;
    }

    if (diff < -M_PI) {
      diff += 2 * M_PI;
    }
  }
}

void RnSOn::integrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &dx,
                      Eigen::Ref<Eigen::VectorXd> xout) const {

  xout = x + dx;
  for (auto &i : so2_indices) {
    auto &so2_x = xout(i);
    so2_x = wrap_angle(so2_x);
    // if (so2_x > M_PI) {
    //   so2_x -= 2 * M_PI;
    // } else if (so2_x < -M_PI) {
    //   so2_x += 2 * M_PI;
    // }
  }
}

void RnSOn::Jintegrate(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &dx,
                       Eigen::Ref<Eigen::MatrixXd> Jfirst,
                       Eigen::Ref<Eigen::MatrixXd> Jsecond) const {
  (void)x;
  (void)dx;
  Jfirst.diagonal().array() = 1;
  Jsecond.diagonal().array() = 1;
}

void RnSOn::Jdiff(const Eigen::Ref<const Eigen::VectorXd> &x0,
                  const Eigen::Ref<const Eigen::VectorXd> &x1,
                  Eigen::Ref<Eigen::MatrixXd> Jfirst,
                  Eigen::Ref<Eigen::MatrixXd> Jsecond) const {

  (void)x0;
  (void)x1;
  Jfirst.diagonal().setConstant(-1.);
  Jsecond.diagonal().setConstant(1.);
}

// void Model_unicycle1_R2SO2::calcV(Eigen::Ref<Eigen::VectorXd> v,
//                                   const Eigen::Ref<const Eigen::VectorXd> &x,
//                                   const Eigen::Ref<const Eigen::VectorXd> &u)
//                                   {
//
//   const double c = cos(x[2]);
//   const double s = sin(x[2]);
//   v << c * u[0], s * u[0], u[1];
// }

Model_robot::Model_robot(std::shared_ptr<StateQ> state, size_t nu)
    : nx(state->nx), nu(nu), state(state) {

  u_ref.resize(nu);
  u_ref.setZero();

  u_0.resize(nu);
  u_0.setZero();

  u_lb.resize(nu);
  u_ub.resize(nu);

  u_ub.setConstant(1e8);
  u_lb.setConstant(-1e8);

  x_weightb.resize(nx);
  x_weightb.setZero();

  x_ub.resize(nx);
  x_lb.resize(nx);

  x_ub.setConstant(1e8);
  x_lb.setConstant(-1e8);

  __v.resize(nx);
  __Jv_x.resize(nx, nx);
  __Jv_u.resize(nx, nu);

  __v.setZero();
  __Jv_x.setZero();
  __Jv_u.setZero();

  // TODO: nx or ndx?
  __Jfirst.resize(state->ndx, state->ndx);
  __Jsecond.resize(state->ndx, state->ndx);

  __Jfirst.setZero();
  __Jsecond.setZero();

  // Eigen::MatrixXd __Jsecond;

  ts_data.resize(1);  // default is only once collision shape
  col_outs.resize(1); // default is only once collision shape

  std::cout << "init done" << std::endl;

  r_weight.resize(nx);
  r_weight.setOnes(); // default!
}

// default for collision with (x,y,theta)
void Model_robot::transformation_collision_geometries(
    const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {

  CHECK_GEQ(x.size(), 3, AT);
  CHECK_EQ(ts.size(), 1, AT);

  fcl::Transform3d result;
  result = Eigen::Translation<double, 3>(fcl::Vector3d(x(0), x(1), 0));
  result.rotate(Eigen::AngleAxisd(x(2), Eigen::Vector3d::UnitZ()));
  ts.at(0) = result;
}

void Model_robot::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  std::cout << nx << std::endl;
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
}

bool Model_robot::collision_check(const Eigen::Ref<const Eigen::VectorXd> &x) {

  assert(env);

  fcl::DefaultCollisionData<double> collision_data;

  transformation_collision_geometries(x, ts_data);
  CHECK_EQ(collision_geometries.size(), ts_data.size(), AT);
  assert(collision_geometries.size() == ts_data.size());
  CHECK_EQ(collision_geometries.size(), col_outs.size(), AT);
  assert(collision_geometries.size() == col_outs.size());

  for (size_t i = 0; i < collision_geometries.size(); i++) {

    fcl::Transform3d result = ts_data[i];
    assert(collision_geometries[i]);
    fcl::CollisionObject co(collision_geometries[i]);

    co.setTranslation(result.translation());
    co.setRotation(result.rotation());
    co.computeAABB();
    env->collide(&co, &collision_data, fcl::DefaultCollisionFunction<double>);
    if (collision_data.result.isCollision()) {
      return false;
    }
  }
  return true;
}

void Model_robot::collision_distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                     CollisionOut &cout) {

  if (env) {

    fcl::DefaultDistanceData<double> distance_data;

    // compute all tansforms

    transformation_collision_geometries(x, ts_data);
    CHECK_EQ(collision_geometries.size(), ts_data.size(), AT);
    assert(collision_geometries.size() == ts_data.size());
    CHECK_EQ(collision_geometries.size(), col_outs.size(), AT);
    assert(collision_geometries.size() == col_outs.size());

    for (size_t i = 0; i < collision_geometries.size(); i++) {

      fcl::Transform3d &result = ts_data[i];
      assert(collision_geometries[i]);
      fcl::CollisionObject co(collision_geometries[i]);

      co.setTranslation(result.translation());
      co.setRotation(result.rotation());
      co.computeAABB();
      distance_data.request.enable_signed_distance = true;
      env->distance(&co, &distance_data, fcl::DefaultDistanceFunction<double>);

      auto &col_out = col_outs.at(i);

      col_out.distance = distance_data.result.min_distance;
      col_out.p1 = distance_data.result.nearest_points[0];
      col_out.p2 = distance_data.result.nearest_points[1];
    }

    // decid eht

    bool return_only_min = true;
    if (return_only_min) {

      auto it = std::min_element(
          col_outs.begin(), col_outs.end(),
          [](auto &a, auto &b) { return a.distance < b.distance; });

      cout = *it; // copy only the min

    } else {
      ERROR_WITH_INFO("not implemented");
    }
  } else {
    cout.distance = max__;
    // struct CollisionOut {
    //   double distance;
    //   Eigen::Vector3d p1;
    //   Eigen::Vector3d p2;
  }
}

void Model_robot::collision_distance_diff(
    Eigen::Ref<Eigen::VectorXd> dd, double &f,
    const Eigen::Ref<const Eigen::VectorXd> &x) {
  // compute collision at current point

  CollisionOut c;
  assert(nx_col > 0);
  assert(nx_col <= x.size());

  collision_distance(x, c);
  f = c.distance;

  double eps = 1e-4; // TODO: evaluate which are valid values here!

  finite_diff_grad(
      [&](auto &y) {
        collision_distance(y.head(nx_col), c);
        return c.distance;
      },
      x.head(nx_col), dd.head(nx_col), eps);
}

bool Model_robot::is_control_valid(const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(u.size() == u_lb.size());
  assert(u.size() == u_ub.size());

  double d = check_bounds_distance(u, u_lb, u_ub);
  const double tol = 1e-12;
  assert(d >= 0);
  return d < tol;
}

bool Model_robot::is_state_valid(const Eigen::Ref<const Eigen::VectorXd> &x) {

  assert(x.size() == x_lb.size());
  assert(x.size() == x_ub.size());

  double d = check_bounds_distance(x, x_lb, x_ub);
  const double tol = 1e-12;
  assert(d >= 0);
  return d < tol;
}

// void Model_unicycle1_R2SO2::step(Eigen::Ref<Eigen::VectorXd> xnext,
//                                  const Eigen::Ref<const Eigen::VectorXd> &x,
//                                  const Eigen::Ref<const Eigen::VectorXd> &u,
//                                  double dt) {
//
//   Eigen::Vector3d v;
//   calcV(v, x, u);
//
//   using Scalar = double;
//   enum { Options = 0 };
//
//   typedef SpecialOrthogonalOperationTpl<2, Scalar, Options> SO2_operation;
//   SO2_operation asO2;
//   SO2_operation::ConfigVector_t pose_s;
//   SO2_operation::ConfigVector_t pose_g;
//   SO2_operation::TangentVector_t delta_pose;
//
//   const double c = cos(x[2]);
//   const double s = sin(x[2]);
//
//   delta_pose(0) = v(2) * dt;
//   pose_s(0) = c;
//   pose_s(1) = s;
//
//   asO2.integrate(pose_s, delta_pose, pose_g);
//   double angle_out = std::atan2(pose_g(1), pose_g(0));
//
//   xnext << x(0) + v(0) * dt, x(1) + v(1) * dt, angle_out;
// }
//
// // step diff
//
// void Model_unicycle1_se2::calcV(Eigen::Ref<Eigen::VectorXd> v,
//                                 const Eigen::Ref<const Eigen::VectorXd> &x,
//                                 const Eigen::Ref<const Eigen::VectorXd> &u) {
//
//   // enum { Options = 0 };
//   // typedef SpecialEuclideanOperationTpl<2, double, Options>
//   // SE2Operation; SE2Operation aSE2;
//
//   const double c = cos(x[2]);
//   const double s = sin(x[2]);
//   v << c * u[0], s * u[0], u[1];
// }
//
// void Model_unicycle1_se2::step(Eigen::Ref<Eigen::VectorXd> xnext,
//                                const Eigen::Ref<const Eigen::VectorXd> &x,
//                                const Eigen::Ref<const Eigen::VectorXd> &u,
//                                double dt) {
//
//   Eigen::Vector3d v;
//   calcV(v, x, u);
//   enum { Options = 0 };
//   typedef SpecialEuclideanOperationTpl<2, double, Options> SE2Operation;
//   SE2Operation aSE2;
//   SpecialEuclideanOperationTpl<2, double, Options>::ConfigVector_t pose_s,
//       pose_g;
//   SpecialEuclideanOperationTpl<2, double, Options>::TangentVector_t delta_u;
//
//   double c = std::cos(x(2));
//   double s = std::sin(x(2));
//   pose_s(0) = x(0);
//   pose_s(1) = x(1);
//   pose_s(2) = c;
//   pose_s(3) = s;
//
//   aSE2.integrate(pose_s, v * dt, pose_g);
//   double angle_out = std::atan2(pose_g(3), pose_g(2));
//
//   xnext(0) = pose_g(0);
//   xnext(1) = pose_g(1);
//   xnext(2) = angle_out;
// }

double Model_robot::cost(const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) const {
  // default cost is time

  (void)x;
  (void)u;
  return ref_dt;
}

double Model_robot::traj_cost(const std::vector<Eigen::VectorXd> &xs,
                              const std::vector<Eigen::VectorXd> &us) const {

  CHECK((xs.size() == us.size() || xs.size() == us.size() + 1), AT);
  double c = 0;

  for (size_t i = 0; i < us.size(); i++) {
    c += cost(xs[i], us[i]);
  }
  return c;
}

void Model_robot::calcV(Eigen::Ref<Eigen::VectorXd> v,
                        const Eigen::Ref<const Eigen::VectorXd> &d,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

  (void)v;
  (void)d;
  (void)u;

  ERROR_WITH_INFO("not implemented");
}

void Model_robot::step(Eigen::Ref<Eigen::VectorXd> xnext,
                       const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &u, double dt) {

  calcV(__v, x, u);
  // euler(xnext, x, __v, dt);
  state->integrate(x, __v * dt, xnext);
}

void Model_robot::stepR4(Eigen::Ref<Eigen::VectorXd> xnext,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u,
                         double dt) {

  runge4(
      xnext, x, u,
      [&](Eigen::Ref<Eigen::VectorXd> y,
          const Eigen::Ref<const Eigen::VectorXd> &x,
          const Eigen::Ref<const Eigen::VectorXd> &u) { calcV(y, x, u); },
      dt);
}

void Model_robot::stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                           Eigen::Ref<Eigen::MatrixXd> Fu,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u,
                           double dt) {

  calcDiffV(__Jv_x, __Jv_u, x, u);
  // euler_diff(Fx, Fu, dt, __Jv_x, __Jv_u);

  // Fx.diagonal()

  // Jy_x.noalias() = dt * Jv_x;
  // for (size_t i = 0; i < n; i++) {
  //   Jy_x(i, i) += 1;
  // }
  // Jy_u.noalias() = dt * Jv_u;

  calcV(__v, x, u); // todo: this is redundant
  state->Jintegrate(x, __v * dt, __Jfirst, __Jsecond);
  Fx += __Jfirst;
  Fx.noalias() += __Jsecond * dt * __Jv_x;
  Fu.noalias() += __Jsecond * dt * __Jv_u;
}

// void Model_robot::stepDiffdt(Eigen::Ref<Eigen::MatrixXd> Fx,
//                              Eigen::Ref<Eigen::MatrixXd> Fu,
//                              const Eigen::Ref<const Eigen::VectorXd> &x,
//                              const Eigen::Ref<const Eigen::VectorXd> &u,
//                              double dt) {
//   CHECK_EQ(nu, static_cast<size_t>(u.size()), AT);
//   CHECK_EQ(nx, static_cast<size_t>(x.size()), AT);
//   CHECK_EQ(nx, static_cast<size_t>(Fx.rows()), AT);
//   CHECK_EQ(nx, static_cast<size_t>(Fx.cols()), AT);
//   CHECK_EQ(nx, static_cast<size_t>(Fu.rows()), AT);
//   CHECK_EQ(static_cast<size_t>(Fu.cols()), nu + 1, AT);
//   calcDiffV(__Jv_x, __Jv_u, x, u);
//   euler_diff(Fx, Fu.block(0, 0, nx, nu), dt, __Jv_x, __Jv_u);
//   calcV(__v, x, u);
//   Fu.col(nu) = __v;
// }

void Model_robot::stepDiff_with_v(Eigen::Ref<Eigen::MatrixXd> Fx,
                                  Eigen::Ref<Eigen::MatrixXd> Fu,
                                  Eigen::Ref<Eigen::VectorXd> __v,
                                  const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &u,
                                  double dt) {
  CHECK_EQ(nu, static_cast<size_t>(u.size()), AT);
  CHECK_EQ(nx, static_cast<size_t>(x.size()), AT);
  CHECK_EQ(nx, static_cast<size_t>(Fx.rows()), AT);
  CHECK_EQ(nx, static_cast<size_t>(Fx.cols()), AT);
  CHECK_EQ(nx, static_cast<size_t>(Fu.rows()), AT);
  CHECK_EQ(static_cast<size_t>(Fu.cols()), nu, AT);

  calcV(__v, x, u);
  calcDiffV(__Jv_x, __Jv_u, x, u);
  euler_diff(Fx.block(0, 0, nx, nx), Fu.block(0, 0, nx, nu), dt, __Jv_x,
             __Jv_u);
}

void Model_robot::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                            Eigen::Ref<Eigen::MatrixXd> Jv_u,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u) {
  (void)Jv_x;
  (void)Jv_u;
  (void)x;
  (void)u;

  ERROR_WITH_INFO("not implemented");
}

double Model_robot::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                             const Eigen::Ref<const Eigen::VectorXd> &y) {
  return (x - y).norm(); // default distance
}

void Model_robot::interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                              const Eigen::Ref<const Eigen::VectorXd> &from,
                              const Eigen::Ref<const Eigen::VectorXd> &to,
                              double dt) {
  xt = from + dt * (to - from); // default interpolation
}

double
Model_robot::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &y) {

  (void)x;
  (void)y;
  ERROR_WITH_INFO("not implemented");
}

void linearInterpolation(const Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &x, double t_query,
                         const StateQ &state, Eigen::Ref<Eigen::VectorXd> out,
                         Eigen::Ref<Eigen::VectorXd> Jx) {

  CHECK(x.size(), AT);
  CHECK_EQ(x.front().size(), out.size(), AT);

  // double num_tolerance = 1e-8;
  // CHECK_GEQ(t_query + num_tolerance, times.head(1)(0), AT);
  assert(static_cast<size_t>(times.size()) == static_cast<size_t>(x.size()));

  if (times.size() == 1) {
    CHECK_EQ(x.size(), 1, AT);
    out = x.front();
    return;
  }

  size_t index = 0;
  if (t_query < times(0)) {
    std::cout << "WARNING: " << AT << std::endl;
    std::cout << "EXTRAPOLATION: " << t_query << " " << times(0) << "  "
              << t_query - times(0) << std::endl;
    index = 1;
  } else if (t_query >= times(times.size() - 1)) {
    std::cout << "WARNING: " << AT << std::endl;
    std::cout << "EXTRAPOLATION: " << t_query << " " << times(times.size() - 1)
              << "  " << t_query - times(times.size() - 1) << std::endl;
    index = times.size() - 1;
  } else {

    auto it = std::lower_bound(
        times.data(), times.data() + times.size(), t_query,
        [](const auto &it, const auto &value) { return it <= value; });

    index = std::distance(times.data(), it);

    const bool debug_bs = false;
    if (debug_bs) {
      size_t index2 = 0;
      for (size_t i = 0; i < static_cast<size_t>(times.size()); i++) {
        if (t_query < times(i)) {
          index2 = i;
          break;
        }
      }
      CHECK_EQ(index, index2, AT);
    }
  }

  //  i have to refactor this!

  // CSTR_V(times);
  // CSTR_(index);
  // CSTR_(t_query);

  double factor =
      (t_query - times(index - 1)) / (times(index) - times(index - 1));

  Eigen::VectorXd diff(state.ndx);
  Eigen::VectorXd x0 = x.at(index - 1);
  state.diff(x0, x.at(index), diff);

  Eigen::VectorXd dx = factor * diff;

  Eigen::MatrixXd J1(state.ndx, state.ndx);
  Eigen::MatrixXd J2(state.ndx, state.ndx);

  state.integrate(x0, dx, out);
  state.Jintegrate(x0, dx, J1, J2);

  Jx = J2 * diff / (times(index) - times(index - 1));
}
