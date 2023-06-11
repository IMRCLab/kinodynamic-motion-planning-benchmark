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
#include "quadpole_acceleration_auto.h"
#include "robot_models.hpp"

using vstr = std::vector<std::string>;
using V2d = Eigen::Vector2d;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;
using Vxd = Eigen::VectorXd;
using V1d = Eigen::Matrix<double, 1, 1>;

double RM_low__ = -std::sqrt(std::numeric_limits<double>::max());
double RM_max__ = std::sqrt(std::numeric_limits<double>::max());

using namespace pinocchio;
// using namespace crocoddyl;

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

std::vector<size_t> create_vector_so2_for_car(size_t num_trailers) {
  std::vector<size_t> out = {2};
  for (size_t i = 0; i < num_trailers; i++) {
    out.push_back(2 + i + 1);
  }
  return out;
}

Model_car_with_trailers::Model_car_with_trailers(const Car_params &params,
                                                 const Eigen::VectorXd &p_lb,
                                                 const Eigen::VectorXd &p_ub)
    : Model_robot(std::make_shared<RnSOn>(
                      2, 1 + params.num_trailers,
                      create_vector_so2_for_car(params.num_trailers)),
                  2),
      params(params) {

  nr_reg = 2;
  nr_ineq = 2 + params.num_trailers;

  name = "car_with_trailers";

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  translation_invariance = 2;
  nx_col = 3 + params.num_trailers;

  nx_pr = 3 + params.num_trailers;

  is_2d = true;
  ref_dt = params.dt;
  distance_weights = params.distance_weights;

  u_weight = V2d(.5, .5);

  x_desc = {"x [m]", "y [m]", "yaw [rad]"};
  u_desc = {"v [m/s]", "phi [rad]"};

  if (params.num_trailers > 1) {
    ERROR_WITH_INFO("not implemented");
  }

  if (params.num_trailers == 1) {
    x_desc.push_back("yaw2 [rad]");
  }

  u_lb << params.min_vel, -params.max_steering_abs;
  u_ub << params.max_vel, params.max_steering_abs;

  x_lb.setConstant(RM_low__);
  x_ub.setConstant(RM_max__);

  assert(params.shape == "box" && params.shape_trailer == "box");

  collision_geometries.emplace_back(
      std::make_shared<fcl::Boxd>(params.size[0], params.size[1], 1.0));
  for (size_t i = 0; i < static_cast<size_t>(params.hitch_lengths.size());
       ++i) {
    collision_geometries.emplace_back(std::make_shared<fcl::Boxd>(
        params.size_trailer[0], params.size_trailer[1], 1.0));
  }

  ts_data.resize(params.hitch_lengths.size() + 1);
  col_outs.resize(params.hitch_lengths.size() + 1);

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

void Model_car_with_trailers::constraintsIneq(
    Eigen::Ref<Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

  (void)u;
  CHECK_EQ(r.size(), 2, AT);
  double diff = x(2) - x(3);

  if (diff > M_PI) {
    diff -= 2 * M_PI;
  } else if (diff < -M_PI) {
    diff += 2 * M_PI;
  }

  //  -diff_max_abs < diff < diff_max_abs
  const double r1 = diff - params.diff_max_abs;
  const double r2 = -params.diff_max_abs - diff;
  r(0) = r1;
  r(1) = r2;
};

void Model_car_with_trailers::constraintsIneqDiff(
    Eigen::Ref<Eigen::MatrixXd> Jx, Eigen::Ref<Eigen::MatrixXd> Ju,
    const Eigen::Ref<const Eigen::VectorXd> x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {
  (void)u;
  CHECK_EQ(static_cast<size_t>(Jx.cols()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jx.rows()), 2, AT);

  Jx(0, 2) = 1;
  Jx(0, 3) = -1;
  Jx(1, 2) = -1;
  Jx(1, 3) = 1;
}

void Model_car_with_trailers::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));

  x(2) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
  if (params.num_trailers == 1) {
    double diff =
        params.diff_max_abs * Eigen::Matrix<double, 1, 1>::Random()(0);
    x(3) = x(2) + diff;
    x(3) = wrap_angle(x(3));
  }
}

void Model_car_with_trailers::transformation_collision_geometries(
    const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {

  fcl::Transform3d result;
  result = Eigen::Translation<double, 3>(fcl::Vector3d(x(0), x(1), 0));
  result.rotate(Eigen::AngleAxisd(x(2), Eigen::Vector3d::UnitZ()));
  ts.at(0) = result;

  if (params.hitch_lengths.size() == 0)
    ;
  else if (params.hitch_lengths.size() == 1) {
    fcl::Transform3d result;
    const double theta1 = x(3);
    fcl::Vector3d pos0(x(0), x(1), 0);
    fcl::Vector3d delta(cos(theta1), sin(theta1), 0);
    result =
        Eigen::Translation<double, 3>(pos0 - delta * params.hitch_lengths[0]);
    result.rotate(Eigen::AngleAxisd(theta1, Eigen::Vector3d::UnitZ()));
    ts.at(1) = result;
  } else {
    ERROR_WITH_INFO("not implemented");
  }
}

void Model_car_with_trailers::calcV(
    Eigen::Ref<Eigen::VectorXd> f, const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<size_t>(f.size()) == nx);
  assert(static_cast<size_t>(x.size()) == nx);
  assert(static_cast<size_t>(u.size()) == nu);

  const double &v = u(0);
  const double &phi = u(1);
  const double &yaw = x(2);

  const double &c = std::cos(yaw);
  const double &s = std::sin(yaw);

  f(0) = v * c;
  f(1) = v * s;
  f(2) = v / params.l * std::tan(phi);

  if (params.num_trailers) {
    CHECK_EQ(params.num_trailers, 1, AT);
    double d = params.hitch_lengths(0);
    double theta_dot = v / d;
    theta_dot *= std::sin(x(2) - x(3));
    f(3) = theta_dot;
  }
}

void Model_car_with_trailers::regularization_cost(
    Eigen::Ref<Eigen::VectorXd> r, const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {
  CHECK_EQ(r.size(), 2, AT);
  r = u;
}

void Model_car_with_trailers::regularization_cost_diff(
    Eigen::Ref<Eigen::MatrixXd> Jx, Eigen::Ref<Eigen::MatrixXd> Ju,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {
  (void)u;
  (void)x;
  (void)Jx;
  Ju.diagonal().setOnes();
}

void Model_car_with_trailers::calcDiffV(
    Eigen::Ref<Eigen::MatrixXd> Jv_x, Eigen::Ref<Eigen::MatrixXd> Jv_u,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

  CHECK_EQ(static_cast<size_t>(Jv_x.rows()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_u.rows()), nx, AT);

  CHECK_EQ(static_cast<size_t>(Jv_x.cols()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_u.cols()), nu, AT);

  CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
  CHECK_EQ(static_cast<size_t>(u.size()), nu, AT);

  const double &v = u(0);
  const double &phi = u(1);
  const double &yaw = x(2);

  const double &c = std::cos(yaw);
  const double &s = std::sin(yaw);

  Jv_x(0, 2) = -v * s;
  Jv_x(1, 2) = v * c;

  Jv_u(0, 0) = c;
  Jv_u(1, 0) = s;
  Jv_u(2, 0) = 1. / params.l * std::tan(phi);
  Jv_u(2, 1) = 1. * v / params.l / (std::cos(phi) * std::cos(phi));

  if (params.num_trailers) {
    CHECK_EQ(params.num_trailers, 1, AT);
    double d = params.hitch_lengths(0);
    // double theta_dot = v / d;
    // double theta_dot =  v / d * std::sin(x(2) - x(3));
    // xnext(3) = x(3) + theta_dot * dt_;
    Jv_x(3, 2) = v / d * std::cos(x(2) - x(3));
    Jv_x(3, 3) = -v / d * std::cos(x(2) - x(3));
    Jv_u(3, 0) = std::sin(x(2) - x(3)) / d;
  }
}

double
Model_car_with_trailers::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y) {
  CHECK_EQ(x.size(), 4, AT);
  CHECK_EQ(y.size(), 4, AT);
  assert(y(2) <= M_PI && y(2) >= -M_PI);
  assert(x(2) <= M_PI && x(2) >= -M_PI);
  double d = params.distance_weights(0) * (x.head<2>() - y.head<2>()).norm() +
             params.distance_weights(1) * so2_distance(x(2), y(2));
  if (params.num_trailers) {
    d += params.distance_weights(2) * so2_distance(x(3), y(3));
  }
  return d;
}

void Model_car_with_trailers::interpolate(
    Eigen::Ref<Eigen::VectorXd> xt,
    const Eigen::Ref<const Eigen::VectorXd> &from,
    const Eigen::Ref<const Eigen::VectorXd> &to, double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(xt.size() == 3);
  assert(from.size() == 3);
  assert(to.size() == 3);

  xt.head<2>() = from.head<2>() + dt * (to.head<2>() - from.head<2>());
  so2_interpolation(xt(2), from(2), to(2), dt);
  if (params.num_trailers) {
    so2_interpolation(xt(3), from(3), to(3), dt);
  }
}

double Model_car_with_trailers::lower_bound_time(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {
  double m = std::max((x.head<2>() - y.head<2>()).norm() / params.max_vel,
                      so2_distance(x(2), y(2)) / params.max_angular_vel);

  if (params.num_trailers) {
    m = std::max(m, so2_distance(x(3), y(3)) / params.max_angular_vel);
  }
  return m;
}

double lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &y) {
  return 0;
}

// Model_acrobot(
//   const Acrobot_params & acrobot_params = Acrobot_params());

Model_acrobot::Model_acrobot(const Acrobot_params &acrobot_params,
                             const Eigen::VectorXd &p_lb,
                             const Eigen::VectorXd &p_ub)
    : Model_robot(std::make_shared<RnSOn>(2, 2, std::vector<size_t>{0, 1}), 1),
      params(acrobot_params) {
  is_2d = false;
  translation_invariance = 0;
  invariance_reuse_col_shape = false;

  name = "acrobot";

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;
  nx_col = 2;
  nx_pr = 2;
  distance_weights.resize(4);
  distance_weights = params.distance_weights;

  u_lb = V1d(-params.max_torque);
  u_ub = V1d(params.max_torque);

  x_lb << RM_low__, RM_low__, -params.max_angular_vel, -params.max_angular_vel;
  x_ub << RM_max__, RM_max__, params.max_angular_vel, params.max_angular_vel;

  ref_dt = params.dt;

  u_weight = V1d(.5);
  x_weightb.resize(4);
  x_weightb << 0, 0, 50, 50;

  const double width = .1;

  collision_geometries.push_back(
      std::make_shared<fcl::Boxd>(params.l1, width, 1.0));

  collision_geometries.push_back(
      std::make_shared<fcl::Boxd>(params.l2, width, 1.0));

  ts_data.resize(2);
  col_outs.resize(2);

  // CHECK_EQ(p_lb.size(), 2, AT);
  // CHECK_EQ(p_ub.size(), 2, AT);
}

void Model_acrobot::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
  x(0) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
  x(1) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
}

void Model_acrobot::transformation_collision_geometries(
    const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {

  const double &q1 = x(0);
  const double &q2 = x(1);

  double offset = 3. * M_PI / 2.;
  Eigen::Vector2d p1 =
      params.lc1 * Eigen::Vector2d(cos(offset + q1), sin(offset + q1));
  Eigen::Vector2d pivot2 =
      params.l1 * Eigen::Vector2d(cos(offset + q1), sin(offset + q1));

  Eigen::Vector2d p2 =
      pivot2 + params.lc2 * Eigen::Vector2d(cos(offset + q1 + q2),
                                            sin(offset + q1 + q2));

  ts.at(0) = Eigen::Translation<double, 3>(fcl::Vector3d(p1(0), p1(1), 0));
  ts.at(0).rotate(Eigen::AngleAxisd(q1 + offset, Eigen::Vector3d::UnitZ()));

  ts.at(1) = Eigen::Translation<double, 3>(fcl::Vector3d(p2(0), p2(1), 0));
  ts.at(1).rotate(
      Eigen::AngleAxisd(q1 + q2 + offset, Eigen::Vector3d::UnitZ()));
}

double Model_acrobot::calcEnergy(const Eigen::Ref<const Eigen::VectorXd> &x) {

  const double &q1 = x(0);
  const double &q2 = x(1);
  const double &q1dot = x(2);
  const double &q2dot = x(3);

  const double &c1 = cos(q1);
  const double &c2 = cos(q2);
  const double &c12 = cos(q1 + q2);

  const double &m1 = params.m1;
  const double &m2 = params.m2;

  const double &I1 = params.I1;
  const double &I2 = params.I2;

  const double &l1 = params.l1;
  const double &lc1 = params.lc1;
  const double &lc2 = params.lc2;

  const double T1 = .5 * I1 * q1dot * q1dot;
  const double T2 =
      .5 * (m2 * l1 * l1 + I2 + 2. * m2 * l1 * lc2 * c2) * q1dot * q1dot +
      .5 * I2 * q2dot * q2dot + (I2 + m2 * l1 * lc2 * c2) * q1dot * q2dot;
  const double U = -m1 * g * lc1 * c1 - m2 * g * (l1 * c1 + lc2 * c12);

  return T1 + T2 + U;
}

void Model_acrobot::calcV(Eigen::Ref<Eigen::VectorXd> f,
                          const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &uu) {

  assert(static_cast<size_t>(f.size()) == nx);
  assert(static_cast<size_t>(x.size()) == nx);
  assert(static_cast<size_t>(uu.size()) == nu);

  const double &q1 = x(0);
  const double &q2 = x(1);
  const double &q1dot = x(2);
  const double &q2dot = x(3);
  const double &u = uu(0);

  const double &m1 = params.m1;
  const double &m2 = params.m2;

  const double &I1 = params.I1;
  const double &I2 = params.I2;

  const double &l1 = params.l1;
  const double &lc1 = params.lc1;
  const double &lc2 = params.lc2;

  double q1dotdot =
      (-I2 * (g * lc1 * m1 * sin(q1) +
              g * m2 * (l1 * sin(q1) + lc2 * sin(q1 + q2)) -
              2. * l1 * lc2 * m2 * q1dot * q2dot * sin(q2) -
              l1 * lc2 * m2 * pow(q2dot, 2.) * sin(q2)) +
       (I2 + l1 * lc2 * m2 * cos(q2)) *
           (g * lc2 * m2 * sin(q1 + q2) +
            l1 * lc2 * m2 * pow(q1dot, 2.) * sin(q2) - u)) /
      (I1 * I2 + I2 * pow(l1, 2.) * m2 -
       pow(l1, 2.) * pow(lc2, 2.) * pow(m2, 2.) * pow(cos(q2), 2.));
  double q2dotdot =
      ((I2 + l1 * lc2 * m2 * cos(q2)) *
           (g * lc1 * m1 * sin(q1) +
            g * m2 * (l1 * sin(q1) + lc2 * sin(q1 + q2)) -
            2. * l1 * lc2 * m2 * q1dot * q2dot * sin(q2) -
            l1 * lc2 * m2 * pow(q2dot, 2.) * sin(q2)) -
       (g * lc2 * m2 * sin(q1 + q2) + l1 * lc2 * m2 * pow(q1dot, 2.) * sin(q2) -
        u) *
           (I1 + I2 + pow(l1, 2.) * m2 + 2. * l1 * lc2 * m2 * cos(q2))) /
      (I1 * I2 + I2 * pow(l1, 2.) * m2 -
       pow(l1, 2.) * pow(lc2, 2.) * pow(m2, 2.) * pow(cos(q2), 2.));

  f(0) = x(2);
  f(1) = x(3);
  f(2) = q1dotdot;
  f(3) = q2dotdot;
}

void Model_acrobot::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                              Eigen::Ref<Eigen::MatrixXd> Jv_u,
                              const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &uu) {

  CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_x.cols()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_x.rows()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_u.rows()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_u.cols()), nu, AT);

  double q1dotdot_u;
  double q2dotdot_u;
  double q1dotdot_q1;
  double q2dotdot_q1;
  double q1dotdot_q2;
  double q2dotdot_q2;
  double q1dotdot_q2dot;
  double q2dotdot_q2dot;
  double q1dotdot_q1dot;
  double q2dotdot_q1dot;

  const double &q1 = x[0];
  const double &q2 = x[1];
  const double &q1dot = x[2];
  const double &q2dot = x[3];
  const double &u = uu[0];

  const double &m1 = params.m1;
  const double &m2 = params.m2;

  const double &I1 = params.I1;
  const double &I2 = params.I2;

  const double &l1 = params.l1;
  const double &lc1 = params.lc1;
  const double &lc2 = params.lc2;

  q1dotdot_u = (I2 + l1 * lc2 * m2 * cos(q2)) /
               (-I1 * I2 - I2 * pow(l1, 2) * m2 +
                pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q2dotdot_u = (I1 + I2 + pow(l1, 2) * m2 + 2 * l1 * lc2 * m2 * cos(q2)) /
               (I1 * I2 + I2 * pow(l1, 2) * m2 -
                pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q1dotdot_q1 = g *
                (I2 * l1 * m2 * cos(q1) + I2 * lc1 * m1 * cos(q1) -
                 l1 * pow(lc2, 2) * pow(m2, 2) * cos(q2) * cos(q1 + q2)) /
                (-I1 * I2 - I2 * pow(l1, 2) * m2 +
                 pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q2dotdot_q1 =
      g *
      (-lc2 * m2 * (I1 + I2 + pow(l1, 2) * m2 + 2 * l1 * lc2 * m2 * cos(q2)) *
           cos(q1 + q2) +
       (I2 + l1 * lc2 * m2 * cos(q2)) *
           (lc1 * m1 * cos(q1) + m2 * (l1 * cos(q1) + lc2 * cos(q1 + q2)))) /
      (I1 * I2 + I2 * pow(l1, 2) * m2 -
       pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q1dotdot_q2 =
      lc2 * m2 *
      (2 * pow(l1, 2) * lc2 * m2 *
           (I2 * (g * lc1 * m1 * sin(q1) +
                  g * m2 * (l1 * sin(q1) + lc2 * sin(q1 + q2)) -
                  2 * l1 * lc2 * m2 * q1dot * q2dot * sin(q2) -
                  l1 * lc2 * m2 * pow(q2dot, 2) * sin(q2)) -
            (I2 + l1 * lc2 * m2 * cos(q2)) *
                (g * lc2 * m2 * sin(q1 + q2) +
                 l1 * lc2 * m2 * pow(q1dot, 2) * sin(q2) - u)) *
           sin(q2) * cos(q2) +
       (I1 * I2 + I2 * pow(l1, 2) * m2 -
        pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2)) *
           (I2 * (-g * cos(q1 + q2) + 2 * l1 * q1dot * q2dot * cos(q2) +
                  l1 * pow(q2dot, 2) * cos(q2)) -
            l1 *
                (g * lc2 * m2 * sin(q1 + q2) +
                 l1 * lc2 * m2 * pow(q1dot, 2) * sin(q2) - u) *
                sin(q2) +
            (I2 + l1 * lc2 * m2 * cos(q2)) *
                (g * cos(q1 + q2) + l1 * pow(q1dot, 2) * cos(q2)))) /
      pow(I1 * I2 + I2 * pow(l1, 2) * m2 -
              pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2),
          2);
  q2dotdot_q2 =
      lc2 * m2 *
      (2 * pow(l1, 2) * lc2 * m2 *
           (-(I2 + l1 * lc2 * m2 * cos(q2)) *
                (g * lc1 * m1 * sin(q1) +
                 g * m2 * (l1 * sin(q1) + lc2 * sin(q1 + q2)) -
                 2 * l1 * lc2 * m2 * q1dot * q2dot * sin(q2) -
                 l1 * lc2 * m2 * pow(q2dot, 2) * sin(q2)) +
            (g * lc2 * m2 * sin(q1 + q2) +
             l1 * lc2 * m2 * pow(q1dot, 2) * sin(q2) - u) *
                (I1 + I2 + pow(l1, 2) * m2 + 2 * l1 * lc2 * m2 * cos(q2))) *
           sin(q2) * cos(q2) +
       (I1 * I2 + I2 * pow(l1, 2) * m2 -
        pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2)) *
           (2 * l1 *
                (g * lc2 * m2 * sin(q1 + q2) +
                 l1 * lc2 * m2 * pow(q1dot, 2) * sin(q2) - u) *
                sin(q2) -
            l1 *
                (g * lc1 * m1 * sin(q1) +
                 g * m2 * (l1 * sin(q1) + lc2 * sin(q1 + q2)) -
                 2 * l1 * lc2 * m2 * q1dot * q2dot * sin(q2) -
                 l1 * lc2 * m2 * pow(q2dot, 2) * sin(q2)) *
                sin(q2) -
            (I2 + l1 * lc2 * m2 * cos(q2)) *
                (-g * cos(q1 + q2) + 2 * l1 * q1dot * q2dot * cos(q2) +
                 l1 * pow(q2dot, 2) * cos(q2)) -
            (g * cos(q1 + q2) + l1 * pow(q1dot, 2) * cos(q2)) *
                (I1 + I2 + pow(l1, 2) * m2 + 2 * l1 * lc2 * m2 * cos(q2)))) /
      pow(I1 * I2 + I2 * pow(l1, 2) * m2 -
              pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2),
          2);
  q1dotdot_q2dot = 2 * I2 * l1 * lc2 * m2 * (q1dot + q2dot) * sin(q2) /
                   (I1 * I2 + I2 * pow(l1, 2) * m2 -
                    pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q2dotdot_q2dot = -2 * l1 * lc2 * m2 * (I2 + l1 * lc2 * m2 * cos(q2)) *
                   (q1dot + q2dot) * sin(q2) /
                   (I1 * I2 + I2 * pow(l1, 2) * m2 -
                    pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q1dotdot_q1dot = 2 * l1 * lc2 * m2 *
                   (I2 * q2dot + q1dot * (I2 + l1 * lc2 * m2 * cos(q2))) *
                   sin(q2) /
                   (I1 * I2 + I2 * pow(l1, 2) * m2 -
                    pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));
  q2dotdot_q1dot =
      -2 * l1 * lc2 * m2 *
      (I1 * q1dot + I2 * q1dot + I2 * q2dot + pow(l1, 2) * m2 * q1dot +
       2 * l1 * lc2 * m2 * q1dot * cos(q2) + l1 * lc2 * m2 * q2dot * cos(q2)) *
      sin(q2) /
      (I1 * I2 + I2 * pow(l1, 2) * m2 -
       pow(l1, 2) * pow(lc2, 2) * pow(m2, 2) * pow(cos(q2), 2));

  // fill the matrices
  Jv_u(2 + 0, 0) = q1dotdot_u;
  Jv_u(2 + 1, 0) = q2dotdot_u;

  Jv_x(2 + 0, 0) = q1dotdot_q1;
  Jv_x(2 + 0, 1) = q1dotdot_q2;
  Jv_x(2 + 0, 2) = q1dotdot_q1dot;
  Jv_x(2 + 0, 3) = q1dotdot_q2dot;

  Jv_x(2 + 1, 0) = q2dotdot_q1;
  Jv_x(2 + 1, 1) = q2dotdot_q2;
  Jv_x(2 + 1, 2) = q2dotdot_q1dot;
  Jv_x(2 + 1, 3) = q2dotdot_q2dot;

  Jv_x(0, 2) = 1;
  Jv_x(1, 3) = 1;
}

double Model_acrobot::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 4);
  assert(y.size() == 4);
  assert(y(0) <= M_PI && y(0) >= -M_PI);
  assert(x(1) <= M_PI && x(1) >= -M_PI);

  Eigen::Vector3d raw_d =
      Eigen::Vector3d(so2_distance(x(0), y(0)), so2_distance(x(1), y(1)),
                      (x.segment<2>(2) - y.segment<2>(2)).norm());

  return raw_d.dot(params.distance_weights);
}

void Model_acrobot::interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                                const Eigen::Ref<const Eigen::VectorXd> &from,
                                const Eigen::Ref<const Eigen::VectorXd> &to,
                                double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(xt.size() == 3);
  assert(from.size() == 3);
  assert(to.size() == 3);

  xt.tail<2>() = from.tail<2>() + dt * (to.tail<2>() - from.tail<2>());
  so2_interpolation(xt(0), from(0), to(0), dt);
  so2_interpolation(xt(1), from(1), to(1), dt);
}

double
Model_acrobot::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &y) {
  std::array<double, 5> maxs = {
      so2_distance(x(0), y(0)) / params.max_angular_vel,
      so2_distance(x(1), y(1)) / params.max_angular_vel,
      std::abs(x(2) - y(2)) / params.max_angular_acc,
      std::abs(x(3) - y(3)) / params.max_angular_acc};
  return *std::max_element(maxs.cbegin(), maxs.cend());
}

Model_unicycle1::Model_unicycle1(const Unicycle1_params &params,
                                 const Eigen::VectorXd &p_lb,
                                 const Eigen::VectorXd &p_ub)
    : Model_robot(std::make_shared<RnSOn>(2, 1, std::vector<size_t>{2}), 2),
      params(params) {
  is_2d = true;
  nx_col = 3;
  nx_pr = 3;
  translation_invariance = 2;

  u_ref << 0, 0;
  distance_weights = params.distance_weights;
  name = "unicycle1";

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  ref_dt = params.dt;
  std::cout << "in " << __FILE__ << ": " << __LINE__ << " -- " << STR_(ref_dt)
            << std::endl;
  x_desc = {"x[m]", "y[m]", "yaw[rad]"};
  u_desc = {"v[m/s]", "w[rad/s]"};
  u_lb << params.min_vel, params.min_angular_vel;
  u_ub << params.max_vel, params.max_angular_vel;
  x_ub.setConstant(RM_max__);
  x_lb.setConstant(RM_low__);

  u_weight.resize(2);
  u_weight.setConstant(.2);
  x_weightb = V3d::Zero();

  std::cout << "in " << __FILE__ << ": " << __LINE__ << std::endl;
  std::cout << STR_V(u_lb) << std::endl;
  std::cout << STR_V(u_ub) << std::endl;

  if (params.shape == "box") {
    collision_geometries.push_back(
        std::make_shared<fcl::Boxd>(params.size(0), params.size(1), 1.0));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

void Model_unicycle1::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
  x(2) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
}

void Model_unicycle1::calcV(Eigen::Ref<Eigen::VectorXd> v,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u) {

  CHECK_EQ(v.size(), 3, AT);
  CHECK_EQ(x.size(), 3, AT);
  CHECK_EQ(u.size(), 2, AT);

  const double c = cos(x[2]);
  const double s = sin(x[2]);
  v << c * u[0], s * u[0], u[1];
}

void Model_unicycle1::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                                Eigen::Ref<Eigen::MatrixXd> Jv_u,
                                const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(Jv_x.rows() == 3);
  assert(Jv_u.rows() == 3);

  assert(Jv_x.cols() == 3);
  assert(Jv_u.cols() == 2);

  assert(x.size() == 3);
  assert(u.size() == 2);

  const double c = cos(x[2]);
  const double s = sin(x[2]);

  Jv_x(0, 2) = -s * u[0];
  Jv_x(1, 2) = c * u[0];
  Jv_u(0, 0) = c;
  Jv_u(1, 0) = s;
  Jv_u(2, 1) = 1;
}

double Model_unicycle1::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                 const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 3);
  assert(y.size() == 3);
  assert(y[2] <= M_PI && y[2] >= -M_PI);
  assert(x[2] <= M_PI && x[2] >= -M_PI);
  return params.distance_weights(0) * (x.head<2>() - y.head<2>()).norm() +
         params.distance_weights(1) * so2_distance(x(2), y(2));
}

void Model_unicycle1::interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                                  const Eigen::Ref<const Eigen::VectorXd> &from,
                                  const Eigen::Ref<const Eigen::VectorXd> &to,
                                  double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(xt.size() == 3);
  assert(from.size() == 3);
  assert(to.size() == 3);

  xt.head<2>() = from.head<2>() + dt * (to.head<2>() - from.head<2>());
  so2_interpolation(xt(2), from(2), to(2), dt);
}

double
Model_unicycle1::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y) {
  double max_vel_abs =
      std::max(std::abs(params.max_vel), std::abs(params.min_vel));
  double max_angular_vel_abs = std::max(std::abs(params.max_angular_vel),
                                        std::abs(params.min_angular_vel));
  return std::max((x.head<2>() - y.head<2>()).norm() / max_vel_abs,
                  so2_distance(x(2), y(2)) / max_angular_vel_abs);
}

Model_quad3d::Model_quad3d(const Quad3d_params &params,

                           const Eigen::VectorXd &p_lb,
                           const Eigen::VectorXd &p_ub)

    : Model_robot(std::make_shared<Rn>(13), 4), params(params) {

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  if (params.motor_control) {
    u_0.setOnes();
  } else {
    u_0 << 1, 0, 0, 0;
  }

  translation_invariance = 3;
  invariance_reuse_col_shape = false;
  nx_col = 7;
  nx_pr = 7;
  is_2d = false;

  ref_dt = params.dt;
  distance_weights = params.distance_weights;

  arm = 0.707106781 * params.arm_length;
  u_nominal = params.m * g / 4.;

  if (params.motor_control) {
    B0 << 1, 1, 1, 1, -arm, -arm, arm, arm, -arm, arm, arm, -arm, -params.t2t,
        params.t2t, -params.t2t, params.t2t;
    B0 *= u_nominal;
    B0inv = B0.inverse();
  } else {
    B0.setIdentity();
    double nominal_angular_acceleration = 20;
    B0(0, 0) *= u_nominal * 4;
    B0(1, 1) *= nominal_angular_acceleration;
    B0(2, 2) *= nominal_angular_acceleration;
    B0(3, 3) *= nominal_angular_acceleration;
  }

  name = "quad3d";
  x_desc = {"x [m]",      "y [m]",      "z [m]",     "qx []",    "qy []",
            "qz []",      "qw []",      "vx [m/s]",  "vy [m/s]", "vz [m/s]",
            "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"};

  u_desc = {"f1 []", "f2 [], f3 [], f4 []"};

  Fu_selection.setZero();
  Fu_selection(2, 0) = 1.;

  // [ 0, 0, 0, 0]   [eta(0)]    =
  // [ 0, 0, 0, 0]   [eta(1)]
  // [ 1, 0, 0, 0]   [eta(2)]
  //                 [eta(3)]

  Ftau_selection.setZero();
  Ftau_selection(0, 1) = 1.;
  Ftau_selection(1, 2) = 1.;
  Ftau_selection(2, 3) = 1.;

  // [ 0, 1, 0, 0]   [eta(0)]    =
  // [ 0, 0, 1, 0]   [eta(1)]
  // [ 0, 0, 0, 1]   [eta(2)]
  //                 [eta(3)]

  Fu_selection_B0 = Fu_selection * B0;
  Ftau_selection_B0 = Ftau_selection * B0;

  // Bounds

  if (params.motor_control) {
    u_lb = Eigen::Vector4d(0, 0, 0, 0);
    u_ub =
        Eigen::Vector4d(params.max_f, params.max_f, params.max_f, params.max_f);
  } else {
    u_lb = params.u_lb;
    u_ub = params.u_ub;
  }

  x_lb.segment(0, 7) << RM_low__, RM_low__, RM_low__, RM_low__, RM_low__,
      RM_low__, RM_low__;
  x_lb.segment(7, 3) << -params.max_vel, -params.max_vel, -params.max_vel;
  x_lb.segment(10, 3) << -params.max_angular_vel, -params.max_angular_vel,
      -params.max_angular_vel;

  x_ub.segment(0, 7) << RM_max__, RM_max__, RM_max__, RM_max__, RM_max__,
      RM_max__, RM_max__;
  x_ub.segment(7, 3) << params.max_vel, params.max_vel, params.max_vel;
  x_ub.segment(10, 3) << params.max_angular_vel, params.max_angular_vel,
      params.max_angular_vel;

  // some precomputation
  inverseJ_v = params.J_v.cwiseInverse();

  inverseJ_M = inverseJ_v.asDiagonal();
  J_M = params.J_v.asDiagonal();

  inverseJ_skew = Skew(inverseJ_v);
  J_skew = Skew(params.J_v);

  m_inv = 1. / params.m;
  m = params.m;
  grav_v = Eigen::Vector3d(0, 0, -params.m * g);

  u_weight = V4d(.5, .5, .5, .5);
  x_weightb = 50. * Vxd::Ones(13);
  x_weightb.head(7) = Eigen::VectorXd::Zero(7);

  if (params.shape == "box") {
    collision_geometries.emplace_back(std::make_shared<fcl::Boxd>(
        params.size(0), params.size(1), params.size(2)));
  } else if (params.shape == "sphere") {
    collision_geometries.emplace_back(
        std::make_shared<fcl::Sphered>(params.size(0)));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

Eigen::VectorXd Model_quad3d::get_x0(const Eigen::VectorXd &x) {
  CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
  Eigen::VectorXd out(nx);
  out.setZero();
  out.head(3) = x.head(3);
  out(6) = 1.;
  return out;
}

void Model_quad3d::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  (void)x;
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
  x.segment(3, 4) = Eigen::Quaterniond::UnitRandom().coeffs();
}

void Model_quad3d::transformation_collision_geometries(
    const Eigen::Ref<const Eigen::VectorXd> &x, std::vector<Transform3d> &ts) {

  fcl::Transform3d result;
  result = Eigen::Translation<double, 3>(fcl::Vector3d(x(0), x(1), x(2)));
  result.rotate(Eigen::Quaterniond(x(3), x(4), x(5), x(6)));
  ts.at(0) = result;
}

void Model_quad3d::transform_primitive(
    const Eigen::Ref<const Eigen::VectorXd> &p,
    const std::vector<Eigen::VectorXd> &xs_in,
    const std::vector<Eigen::VectorXd> &us_in,
    std::vector<Eigen::VectorXd> &xs_out,
    std::vector<Eigen::VectorXd> &us_out) {

  CHECK((p.size() == 3 || 6), AT);

  if (p.size() == 3) {
    Model_robot::transform_primitive(p, xs_in, us_in, xs_out, us_out);
  } else {
    xs_out = xs_in;
    transform_state(p, xs_in.at(0), xs_out.at(0));
    rollout(xs_out.at(0), us_in, xs_out);
  }
}

void Model_quad3d::calcV(Eigen::Ref<Eigen::VectorXd> ff,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) {

  Eigen::Vector3d f_u;
  Eigen::Vector3d tau_u;

  Eigen::Vector4d eta = B0 * u;
  f_u << 0, 0, eta(0);
  tau_u << eta(1), eta(2), eta(3);

  Eigen::Vector4d q = x.segment(3, 4).head<4>().normalized();
  Eigen::Vector3d vel = x.segment(7, 3).head<3>();
  Eigen::Vector3d w = x.segment(10, 3).head<3>();

  auto fa_v = Eigen::Vector3d(0, 0, 0); // drag model
                                        //
                                        //
                                        // con

  auto const &J_v = params.J_v;

  Eigen::Vector3d a =
      m_inv * (grav_v + Eigen::Quaterniond(q)._transformVector(f_u) + fa_v);

  ff.head<3>() = vel;
  ff.segment<3>(3) = w;
  ff.segment<3>(7 - 1) = a;
  ff.segment<3>(10 - 1) =
      inverseJ_v.cwiseProduct((J_v.cwiseProduct(w)).cross(w) + tau_u);
}

void Model_quad3d::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                             Eigen::Ref<Eigen::MatrixXd> Jv_u,
                             const Eigen::Ref<const Eigen::VectorXd> &x,
                             const Eigen::Ref<const Eigen::VectorXd> &u) {

  // x = [ p , q , v , w ]

  Eigen::Vector3d f_u;
  Eigen::Vector3d tau_u;
  Eigen::Vector4d eta = B0 * u;
  f_u << 0, 0, eta(0);
  tau_u << eta(1), eta(2), eta(3);

  const Eigen::Vector4d &xq = x.segment<4>(3);
  Eigen::Ref<const Eigen::Vector3d> w = x.segment(10, 3).head<3>();
  Eigen::Vector3d y;
  auto const &J_v = params.J_v;
  Eigen::Vector4d q = x.segment(3, 4).head<4>().normalized();
  Eigen::Matrix3d R = Eigen::Quaterniond(q).toRotationMatrix();

  rotate_with_q(xq, f_u, y, data.Jx, data.Ja);

  Jv_x.block<3, 3>(0, 7).diagonal() = Eigen::Vector3d::Ones(); // dp / dv
  //
  //
  //
  // std::cout << "data.Jx\n" << data.Jx << std::endl;

  Jv_x.block<3, 4>(7 - 1, 3).noalias() = m_inv * data.Jx; // da / dq
  Jv_x.block<3, 3>(10 - 1, 10).noalias() =
      inverseJ_M * (Skew(J_v.cwiseProduct(w)) - Skew(w) * J_M); // daa / dw

  Jv_u.block<3, 4>(7 - 1, 0).noalias() = m_inv * R * Fu_selection_B0; // da / df
  Jv_u.block<3, 4>(10 - 1, 0).noalias() =
      inverseJ_M * Ftau_selection_B0; // daa / df
  //
  //
  // std::cout << "Jv_x \n" << Jv_x << std::endl;
}

void Model_quad3d::step(Eigen::Ref<Eigen::VectorXd> xnext,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u, double dt) {

  calcV(ff, x, u);

  Eigen::Ref<const Eigen::Vector3d> pos = x.head(3).head<3>();
  Eigen::Vector4d q = x.segment(3, 4).head<4>().normalized();
  Eigen::Ref<const Eigen::Vector3d> vel = x.segment(7, 3).head<3>();
  Eigen::Ref<const Eigen::Vector3d> w = x.segment(10, 3).head<3>();

  Eigen::Ref<Eigen::Vector3d> pos_next = xnext.head(3);
  Eigen::Ref<Eigen::Vector4d> q_next = xnext.segment(3, 4);
  Eigen::Ref<Eigen::Vector3d> vel_next = xnext.segment(7, 3);
  Eigen::Ref<Eigen::Vector3d> w_next = xnext.segment(10, 3);

  pos_next = pos + dt * ff.segment<3>(0);
  vel_next = vel + dt * ff.segment<3>(6);

  Eigen::Vector4d deltaQ;
  __get_quat_from_ang_vel_time(ff.segment<3>(3) * dt, deltaQ, nullptr);
  quat_product(q, deltaQ, q_next, nullptr, nullptr);
  w_next = w + dt * ff.segment<3>(9);
}

void Model_quad3d::stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                            Eigen::Ref<Eigen::MatrixXd> Fu,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u,
                            double dt) {

  calcDiffV(Jv_x, Jv_u, x, u);
  Fx.block<3, 3>(0, 0).diagonal() = Eigen::Vector3d::Ones();        // dp / dp
  Fx.block<3, 3>(0, 7) = dt * Jv_x.block<3, 3>(0, 7);               // dp / dv
  Fx.block<3, 3>(7, 7).diagonal() = Eigen::Vector3d::Ones();        // dv / dv
  Fx.block<3, 4>(7, 3).noalias() = dt * Jv_x.block<3, 4>(7 - 1, 3); // dv / dq
  Fx.block<3, 3>(10, 10).diagonal().setOnes();
  Fx.block<3, 3>(10, 10).noalias() += dt * Jv_x.block<3, 3>(10 - 1, 10);

  Fu.block<3, 4>(7, 0).noalias() = dt * Jv_u.block<3, 4>(7 - 1, 0);
  Fu.block<3, 4>(10, 0).noalias() = dt * Jv_u.block<3, 4>(10 - 1, 0);

  // Eigen::Vector3d y;
  // const Eigen::Vector4d &xq = x.segment<4>(3);
  //
  // rotate_with_q(xq, data.f_u, y, data.Jx, data.Ja);
  //
  // Fx.block<3, 4>(7, 3).noalias() = dt * m_inv * data.Jx;
  //
  // const Eigen::Vector3d &w = x.segment<3>(10);

  // q_next = qintegrate(Eigen::Quaterniond(q), w, dt).coeffs();
  // Eigen::Quaterniond deltaQ = get_quat_from_ang_vel_time(w *
  // dt);

  Eigen::Matrix<double, 4, 3> Jexp(4, 3);
  Eigen::Vector4d deltaQ;
  Eigen::Vector4d xq_normlized;
  Eigen::Matrix4d Jqnorm;
  Eigen::Matrix4d J1;
  Eigen::Matrix4d J2;
  Eigen::Vector4d yy;

  // QUATERNION....
  const Eigen::Vector4d &xq = x.segment<4>(3);
  Eigen::Ref<const Eigen::Vector3d> w = x.segment(10, 3).head<3>();

  __get_quat_from_ang_vel_time(w * dt, deltaQ, &Jexp);

  normalize(xq, xq_normlized, Jqnorm);
  quat_product(xq_normlized, deltaQ, yy, &J1, &J2);

  Fx.block<4, 4>(3, 3).noalias() = J1 * Jqnorm;
  Fx.block<4, 3>(3, 10) = J2 * Jexp * dt;
}

#if 0
void Model_quad3d::stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                            Eigen::Ref<Eigen::MatrixXd> Fu,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u,
                            double dt) {

  Eigen::Vector4d f = u_nominal * u;

  // todo: refactor this
  if (params.motor_control) {
    Eigen::Vector4d eta = B0 * f;
    data.f_u << 0, 0, eta(0);
    data.tau_u << eta(1), eta(2), eta(3);
  } else {
    CHECK(false, AT);
  }

  Eigen::Vector4d q = x.segment(3, 4).head<4>().normalized();

  Fx.block<3, 3>(0, 0).diagonal() = Eigen::Vector3d::Ones();
  Fx.block<3, 3>(0, 7).diagonal() = dt * Eigen::Vector3d::Ones();
  Fx.block<3, 3>(7, 7).diagonal() = Eigen::Vector3d::Ones();

  Eigen::Vector3d y;
  const Eigen::Vector4d &xq = x.segment<4>(3);

  rotate_with_q(xq, data.f_u, y, data.Jx, data.Ja);

  Fx.block<3, 4>(7, 3).noalias() = dt * m_inv * data.Jx;

  const Eigen::Vector3d &w = x.segment<3>(10);

  // q_next = qintegrate(Eigen::Quaterniond(q), w, dt).coeffs();
  // Eigen::Quaterniond deltaQ = get_quat_from_ang_vel_time(w *
  // dt);

  Eigen::Matrix<double, 4, 3> Jexp(4, 3);
  Eigen::Vector4d deltaQ;
  Eigen::Vector4d xq_normlized;
  Eigen::Matrix4d Jqnorm;
  Eigen::Matrix4d J1;
  Eigen::Matrix4d J2;
  Eigen::Vector4d yy;

  __get_quat_from_ang_vel_time(w * dt, deltaQ, &Jexp);

  normalize(xq, xq_normlized, Jqnorm);
  quat_product(xq_normlized, deltaQ, yy, &J1, &J2);

  Fx.block<4, 4>(3, 3).noalias() = J1 * Jqnorm;

  // angular velocity
  // for (size_t i = 10; i < 13; i++) {
  //   Eigen::MatrixXd xe;
  //   xe = x;
  //   xe(i) += eps;
  //   Eigen::VectorXd xnexte(nx);
  //   xnexte.setZero();
  //   calc(xnexte, xe, u);
  //   auto df = (xnexte - xnext) / eps;
  //   Fx.col(i) = df;
  // }

  // w_next =
  //     w + dt *
  //     inverseJ_v.cwiseProduct((J_v.cwiseProduct(w)).cross(w) +
  //     tau_u);

  // dw_next / d_tau = dt *  inverseJ_M

  // derivative of cross product

  // d / dx [ a x b ] = da / dx x b + a x db / dx

  // d / dx [ Jw x w ] = J x b + a x db / dx

  // J ( A x B ) = Skew(A) * JB - Skew(B) JA

  // J ( Kw x w ) = Skew(k w) * Id - Skew(w) * K

  auto const &J_v = params.J_v;
  Fx.block<4, 3>(3, 10) = J2 * Jexp * dt;
  Fx.block<3, 3>(10, 10).diagonal().setOnes();
  Fx.block<3, 3>(10, 10).noalias() +=
      dt * inverseJ_M * (Skew(J_v.cwiseProduct(w)) - Skew(w) * J_M);

  Eigen::Matrix3d R = Eigen::Quaterniond(q).toRotationMatrix();

  // std::cout << "setting FU" << std::endl;
  // std::cout << Fu << std::endl;
  Fu.block<3, 4>(7, 0).noalias() = u_nominal * dt * m_inv * R * Fu_selection_B0;
  Fu.block<3, 4>(10, 0).noalias() =
      u_nominal * dt * inverseJ_M * Ftau_selection_B0;
}
#endif

double Model_quad3d::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 13);
  assert(y.size() == 13);
  // std::cout << "quad3d distance" << std::endl;
  Eigen::Vector4d raw_d((x.head<3>() - y.head<3>()).norm(),
                        so3_distance(x.segment<4>(3), y.segment<4>(3)),
                        (x.segment<3>(7) - y.segment<3>(7)).norm(),
                        (x.segment<3>(10) - y.segment<3>(10)).norm());

  return raw_d.dot(params.distance_weights);
}

void Model_quad3d::interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                               const Eigen::Ref<const Eigen::VectorXd> &from,
                               const Eigen::Ref<const Eigen::VectorXd> &to,
                               double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(static_cast<size_t>(xt.size()) == nx);
  assert(static_cast<size_t>(from.size()) == nx);
  assert(static_cast<size_t>(to.size()) == nx);

  xt.head<3>() = from.head<3>() + dt * (to.head<3>() - from.head<3>());
  xt.tail<6>() = from.tail<6>() + dt * (to.tail<6>() - from.tail<6>());

  const Eigen::Quaterniond &q_s = Eigen::Quaterniond(from.segment<4>(3));
  const Eigen::Quaterniond &q_g = Eigen::Quaterniond(to.segment<4>(3));
  const Eigen::Quaterniond &q_ = q_s.slerp(dt, q_g);
  xt.segment<4>(3) = q_.coeffs();
}

double

Model_quad3d::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 4> maxs = {
      (x.head<3>() - y.head<3>()).norm() / params.max_vel,
      so3_distance(x.segment<4>(3), y.segment<4>(3)) / params.max_angular_vel,
      (x.segment<3>(7) - y.segment<3>(7)).norm() / params.max_acc,
      (x.segment<3>(10) - y.segment<3>(10)).norm() / params.max_angular_acc};
  return *std::max_element(maxs.cbegin(), maxs.cend());
}

double
Model_quad3d::lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 2> maxs = {
      (x.head<3>() - y.head<3>()).norm() / params.max_vel,
      so3_distance(x.segment<4>(3), y.segment<4>(3)) / params.max_angular_vel};
  return *std::max_element(maxs.cbegin(), maxs.cend());
}

double
Model_quad3d::lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 2> maxs = {
      (x.segment<3>(7) - y.segment<3>(7)).norm() / params.max_acc,
      (x.segment<3>(10) - y.segment<3>(10)).norm() / params.max_angular_acc};

  return *std::max_element(maxs.cbegin(), maxs.cend());
}

Model_quad2dpole::Model_quad2dpole(const Quad2dpole_params &params,
                                   const Eigen::VectorXd &p_lb,
                                   const Eigen::VectorXd &p_ub)

    : Model_robot(std::make_shared<RnSOn>(6, 2, std::vector<size_t>{2, 3}), 2),
      params(params) {
  is_2d = true;
  translation_invariance = 2;
  ref_dt = params.dt;
  name = "quad2dpole";
  invariance_reuse_col_shape = false;
  u_0.setOnes();
  distance_weights = params.distance_weights;

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  // TODO: check
  nx_col = 3;
  nx_pr = 3;
  x_desc = {"x [m]",      "y [m]",      "yaw[rad]", "q[rad]",
            "xdot [m/s]", "ydot [m/s]", "w[rad/s]", "vq [rad/s]"};
  u_desc = {"f1 []", "f2 []"};

  u_lb.setZero();
  u_ub.setConstant(params.max_f);

  x_lb << RM_low__, RM_low__, -params.yaw_max, RM_low__, -params.max_vel,
      -params.max_vel, -params.max_angular_vel, -params.max_angular_vel;

  x_ub << RM_max__, RM_max__, params.yaw_max, RM_max__, params.max_vel,
      params.max_vel, params.max_angular_vel, params.max_angular_vel;

  u_nominal = params.m * g / 2;

  u_weight = V2d(.5, .5);
  x_weightb = 10. * Vxd::Ones(8);
  x_weightb.head<4>() = V4d::Zero();

  if (params.shape == "box") {
    collision_geometries.push_back(
        std::make_shared<fcl::Boxd>(params.size(0), params.size(1), 1.0));
  } else if (params.shape == "sphere") {
    std::make_shared<fcl::Sphered>(params.size(0));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

Model_quad2d::Model_quad2d(const Quad2d_params &params,

                           const Eigen::VectorXd &p_lb,
                           const Eigen::VectorXd &p_ub)

    : Model_robot(std::make_shared<RnSOn>(5, 1, std::vector<size_t>{2}), 2),
      params(params) {
  is_2d = true;
  translation_invariance = 2;
  ref_dt = params.dt;
  name = "quad2d";
  invariance_reuse_col_shape = false;
  u_0.setOnes();
  distance_weights = params.distance_weights;

  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  nx_col = 3;
  nx_pr = 3;
  x_desc = {"x [m]",      "y [m]",      "yaw[rad]",
            "xdot [m/s]", "ydot [m/s]", "w[rad/s]"};
  u_desc = {"f1 []", "f2 []"};

  u_lb.setZero();
  u_ub.setConstant(params.max_f);

  x_lb << RM_low__, RM_low__, RM_low__, -params.max_vel, -params.max_vel,
      -params.max_angular_vel;
  x_ub << RM_max__, RM_max__, RM_max__, params.max_vel, params.max_vel,
      params.max_angular_vel;

  u_nominal = params.m * g / 2;

  u_weight = V2d(.5, .5);
  x_weightb = 10. * Vxd::Ones(6);
  x_weightb.head<3>() = V3d::Zero();

  if (params.shape == "box") {
    collision_geometries.push_back(
        std::make_shared<fcl::Boxd>(params.size(0), params.size(1), 1.0));
  } else if (params.shape == "sphere") {
    std::make_shared<fcl::Sphered>(params.size(0));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

void Model_quad2d::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
  x(2) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
}

void Model_quad2dpole::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
  x(2) = (params.yaw_max *
          Eigen::Matrix<double, 1, 1>::Random())(0); // yaw is restricted
  x(3) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
}

void Model_quad2dpole::calcV(Eigen::Ref<Eigen::VectorXd> v,
                             const Eigen::Ref<const Eigen::VectorXd> &x,
                             const Eigen::Ref<const Eigen::VectorXd> &u) {

  CHECK_EQ(v.size(), 8, AT);
  CHECK_EQ(x.size(), 8, AT);
  CHECK_EQ(u.size(), 2, AT);

  double data[6] = {params.I, params.m, params.m_p, params.l, params.r, g};
  double out[4];

  Eigen::Vector2d uu = u * u_nominal;

  quadpole_2d(x.data(), uu.data(), data, out, nullptr, nullptr);

  v.head(4) = x.segment(4, 4);
  v.segment(4, 4) << out[0], out[1], out[2], out[3];
}

void Model_quad2d::calcV(Eigen::Ref<Eigen::VectorXd> v,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) {

  CHECK_EQ(v.size(), 6, AT);
  CHECK_EQ(x.size(), 6, AT);
  CHECK_EQ(u.size(), 2, AT);

  const double &f1 = u_nominal * u(0);
  const double &f2 = u_nominal * u(1);
  const double &c = std::cos(x(2));
  const double &s = std::sin(x(2));

  const double &xdot = x(3);
  const double &ydot = x(4);
  const double &thetadot = x(5);

  const double &m_inv = 1. / params.m;
  const double &I_inv = 1. / params.I;

  double xdotdot = -m_inv * (f1 + f2) * s;
  double ydotdot = m_inv * (f1 + f2) * c - g;
  double thetadotdot = params.l * I_inv * (f1 - f2);

  if (params.drag_against_vel) {
    xdotdot -= m_inv * params.k_drag_linear * xdot;
    ydotdot -= m_inv * params.k_drag_linear * ydot;
    thetadotdot -= I_inv * params.k_drag_angular * thetadot;
  }

  v.head(3) = x.segment(3, 3);
  v.segment(3, 3) << xdotdot, ydotdot, thetadotdot;
}

void Model_quad2dpole::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                                 Eigen::Ref<Eigen::MatrixXd> Jv_u,
                                 const Eigen::Ref<const Eigen::VectorXd> &x,
                                 const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<size_t>(Jv_x.rows()) == 8);
  assert(static_cast<size_t>(Jv_u.rows()) == 8);

  assert(static_cast<size_t>(Jv_x.cols()) == 8);
  assert(static_cast<size_t>(Jv_u.cols()) == 2);

  assert(static_cast<size_t>(x.size()) == 8);
  assert(static_cast<size_t>(u.size()) == 2);

  CHECK_EQ(x.size(), 8, AT);
  CHECK_EQ(u.size(), 2, AT);

  double data[6] = {params.I, params.m, params.m_p, params.l, params.r, g};
  double out[4];
  double Jx[32];
  double Ju[8];

  Eigen::Vector2d uu = u * u_nominal;

  quadpole_2d(x.data(), uu.data(), data, out, Jx, Ju);

  // duu / du

  Jv_x.block(0, 4, 4, 4).setIdentity();

  // print_vec(Ju, 8);
  // print_vec(Jx, 32);

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 2; j++) {
      Jv_u(4 + i, j) = u_nominal * Ju[i * 2 + j];
      // Jv_u(4+i, j) =  Ju[i*2+4];
    }
  }

  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 8; j++) {
      Jv_x(4 + i, j) = Jx[i * 8 + j];
      // Jv_x(4+i, j) =  Jx[i*2+4]; ??
    }
  }
}

void Model_quad2d::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                             Eigen::Ref<Eigen::MatrixXd> Jv_u,
                             const Eigen::Ref<const Eigen::VectorXd> &x,
                             const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<size_t>(Jv_x.rows()) == 6);
  assert(static_cast<size_t>(Jv_u.rows()) == 6);

  assert(static_cast<size_t>(Jv_x.cols()) == 6);
  assert(static_cast<size_t>(Jv_u.cols()) == 2);

  assert(static_cast<size_t>(x.size()) == 6);
  assert(static_cast<size_t>(u.size()) == 2);

  const double &f1 = u_nominal * u(0);
  const double &f2 = u_nominal * u(1);
  const double &c = std::cos(x(2));
  const double &s = std::sin(x(2));

  const double &m_inv = 1. / params.m;
  const double &I_inv = 1. / params.I;

  Jv_x.block(0, 3, 3, 3).setIdentity();

  const double &d_xdotdot_dtheta = -m_inv * (f1 + f2) * c;
  const double &d_ydotdot_dtheta = m_inv * (f1 + f2) * (-s);

  Jv_x(3, 2) = d_xdotdot_dtheta;
  Jv_x(4, 2) = d_ydotdot_dtheta;

  Jv_u(3, 0) = -m_inv * s * u_nominal;
  Jv_u(3, 1) = -m_inv * s * u_nominal;

  Jv_u(4, 0) = m_inv * c * u_nominal;
  Jv_u(4, 1) = m_inv * c * u_nominal;

  Jv_u(5, 0) = params.l * I_inv * u_nominal;
  Jv_u(5, 1) = -params.l * I_inv * u_nominal;

  if (params.drag_against_vel) {
    Jv_x(3, 3) -= m_inv * params.k_drag_linear;
    Jv_x(4, 4) -= m_inv * params.k_drag_linear;
    Jv_x(5, 5) -= I_inv * params.k_drag_angular;
  }
}

double Model_quad2d::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 6);
  assert(y.size() == 6);
  assert(y[2] <= M_PI && y[2] >= -M_PI);
  assert(x[2] <= M_PI && x[2] >= -M_PI);

  Eigen::Vector4d raw_d(
      (x.head<2>() - y.head<2>()).norm(), so2_distance(x(2), y(2)),
      (x.segment<2>(3) - y.segment<2>(3)).norm(), std::fabs(x(5) - y(5)));
  return raw_d.dot(params.distance_weights);
}

double Model_quad2dpole::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 8);
  assert(y.size() == 8);
  assert(y[2] <= M_PI && y[2] >= -M_PI);
  assert(x[2] <= M_PI && x[2] >= -M_PI);

  assert(y[3] <= M_PI && y[3] >= -M_PI);
  assert(x[3] <= M_PI && x[3] >= -M_PI);

  Vector6d raw_d;
  raw_d << (x.head<2>() - y.head<2>()).norm(), so2_distance(x(2), y(2)),
      so2_distance(x(3), y(3)), (x.segment<2>(4) - y.segment<2>(4)).norm(),
      std::fabs(x(5) - y(5)), std::fabs(x(6) - y(6));

  return raw_d.dot(params.distance_weights);
}

void Model_quad2dpole::interpolate(
    Eigen::Ref<Eigen::VectorXd> xt,
    const Eigen::Ref<const Eigen::VectorXd> &from,
    const Eigen::Ref<const Eigen::VectorXd> &to, double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(xt.size() == 8);
  assert(from.size() == 8);
  assert(to.size() == 8);

  xt.head<2>() = from.head<2>() + dt * (to.head<2>() - from.head<2>());
  so2_interpolation(xt(2), from(2), to(2), dt);
  so2_interpolation(xt(3), from(3), to(3), dt);
  xt.segment<4>(4) =
      from.segment<4>(4) + dt * (to.segment<4>(4) - from.segment<4>(4));
}

void Model_quad2d::interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                               const Eigen::Ref<const Eigen::VectorXd> &from,
                               const Eigen::Ref<const Eigen::VectorXd> &to,
                               double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(xt.size() == 6);
  assert(from.size() == 6);
  assert(to.size() == 6);

  xt.head<2>() = from.head<2>() + dt * (to.head<2>() - from.head<2>());
  so2_interpolation(xt(2), from(2), to(2), dt);
  xt.segment<3>(3) =
      from.segment<3>(3) + dt * (to.segment<3>(3) - from.segment<3>(3));
}

double Model_quad2dpole::lower_bound_time_vel(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  NOT_IMPLEMENTED;
}

double
Model_quad2d::lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 3> maxs = {std::abs(x(3) - y(3)) / params.max_acc,
                                std::abs(x(4) - y(4)) / params.max_acc,
                                std::abs(x(5) - y(5)) / params.max_angular_acc};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

double Model_quad2dpole::lower_bound_time_pr(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  NOT_IMPLEMENTED;
}

double
Model_quad2d::lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 2> maxs = {
      (x.head<2>() - y.head<2>()).norm() / params.max_vel,
      so2_distance(x(2), y(2)) / params.max_angular_vel};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

double
Model_quad2dpole::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                                   const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 7> maxs = {
      (x.head<2>() - y.head<2>()).norm() / params.max_vel,
      so2_distance(x(2), y(2)) / params.max_angular_vel,
      so2_distance(x(3), y(3)) / params.max_angular_vel,
      std::abs(x(4) - y(4)) / params.max_acc,
      std::abs(x(5) - y(5)) / params.max_acc,
      std::abs(x(6) - y(6)) / params.max_angular_acc,
      std::abs(x(7) - y(7)) / params.max_angular_acc};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

double
Model_quad2d::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 5> maxs = {
      (x.head<2>() - y.head<2>()).norm() / params.max_vel,
      so2_distance(x(2), y(2)) / params.max_angular_vel,
      std::abs(x(3) - y(3)) / params.max_acc,
      std::abs(x(4) - y(4)) / params.max_acc,
      std::abs(x(5) - y(5)) / params.max_angular_acc};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

Model_unicycle2::Model_unicycle2(const Unicycle2_params &params,
                                 const Eigen::VectorXd &p_lb,
                                 const Eigen::VectorXd &p_ub)

    : Model_robot(std::make_shared<RnSOn>(4, 1, std::vector<size_t>{2}), 2),
      params(params) {

  name = "unicycle2";
  std::cout << "Model " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  distance_weights = params.distance_weights;
  nx_col = 3;
  nx_pr = 3;

  translation_invariance = 2;
  ref_dt = params.dt;
  x_desc = {"x [m]", "y [m]", "yaw[rad]", "v[m/s]", "w[rad/s]"};
  u_desc = {"a [m/s^2]", "aa[rad/s^2]"};

  u_lb << -params.max_acc_abs, -params.max_angular_acc_abs;
  u_ub << params.max_acc_abs, params.max_angular_acc_abs;

  x_ub << RM_max__, RM_max__, RM_max__, params.max_vel, params.max_angular_vel;
  x_lb << RM_low__, RM_low__, RM_low__, params.min_vel, params.min_angular_vel;

  u_weight.resize(2);
  u_weight.setConstant(.5);
  x_weightb.resize(5);
  x_weightb << 0, 0, 0, 200, 200;

  if (params.shape == "box") {
    collision_geometries.push_back(
        std::make_shared<fcl::Boxd>(params.size(0), params.size(1), 1.0));
  } else if (params.shape == "sphere") {
    collision_geometries.push_back(
        std::make_shared<fcl::Sphered>(params.size(0)));
  } else {
    ERROR_WITH_INFO("not implemented");
  }

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }
}

void Model_unicycle2::sample_uniform(Eigen::Ref<Eigen::VectorXd> x) {
  x = x_lb + (x_ub - x_lb)
                 .cwiseProduct(.5 * (Eigen::VectorXd::Random(nx) +
                                     Eigen::VectorXd::Ones(nx)));
  x(2) = (M_PI * Eigen::Matrix<double, 1, 1>::Random())(0);
}

void Model_unicycle2::calcV(Eigen::Ref<Eigen::VectorXd> f,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<size_t>(f.size()) == nx);
  assert(static_cast<size_t>(x.size()) == nx);
  assert(static_cast<size_t>(u.size()) == nu);

  const double yaw = x[2];
  const double vv = x[3];
  const double w = x[4];

  const double c = cos(yaw);
  const double s = sin(yaw);

  const double a = u[0];
  const double w_dot = u[1];

  f << vv * c, vv * s, w, a, w_dot;
}

void Model_unicycle2::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                                Eigen::Ref<Eigen::MatrixXd> Jv_u,
                                const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &u) {

  (void)u;
  assert(static_cast<size_t>(Jv_x.rows()) == nx);
  assert(static_cast<size_t>(Jv_u.rows()) == nx);

  assert(static_cast<size_t>(Jv_x.cols()) == nx);
  assert(static_cast<size_t>(Jv_u.cols()) == nu);

  assert(static_cast<size_t>(x.size()) == nx);
  assert(static_cast<size_t>(u.size()) == nu);

  const double c = cos(x[2]);
  const double s = sin(x[2]);

  const double v = x[3];

  Jv_x(0, 2) = -s * v;
  Jv_x(1, 2) = c * v;

  Jv_x(0, 3) = c;
  Jv_x(1, 3) = s;
  Jv_x(2, 4) = 1.;

  Jv_u(3, 0) = 1.;
  Jv_u(4, 1) = 1.;
}

double Model_unicycle2::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                                 const Eigen::Ref<const Eigen::VectorXd> &y) {
  assert(x.size() == 5);
  assert(y.size() == 5);
  assert(y[2] <= M_PI && y[2] >= -M_PI);
  assert(x[2] <= M_PI && x[2] >= -M_PI);
  Eigen::Vector4d raw_d = Eigen::Vector4d(
      (x.head<2>() - y.head<2>()).norm(), so2_distance(x(2), y(2)),
      std::abs(x(3) - y(3)), std::abs(x(4) - y(4)));
  return raw_d.dot(params.distance_weights);
}

void Model_unicycle2::interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                                  const Eigen::Ref<const Eigen::VectorXd> &from,
                                  const Eigen::Ref<const Eigen::VectorXd> &to,
                                  double dt) {
  assert(dt <= 1);
  assert(dt >= 0);

  assert(static_cast<size_t>(xt.size()) == nx);
  assert(static_cast<size_t>(from.size()) == nx);
  assert(static_cast<size_t>(to.size()) == nx);

  xt.head<2>() = from.head<2>() + dt * (to.head<2>() - from.head<2>());
  so2_interpolation(xt(2), from(2), to(2), dt);
  xt.tail<2>() = from.tail<2>() + dt * (to.tail<2>() - from.tail<2>());
}

double
Model_unicycle2::lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                                  const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 4> maxs = {
      (x.head<2>() - y.head<2>()).norm() / params.max_vel,
      so2_distance(x(2), y(2)) / params.max_angular_vel,
      std::abs(x(3) - y(3)) / params.max_acc_abs,
      std::abs(x(4) - y(4)) / params.max_angular_acc_abs};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

double Model_unicycle2::lower_bound_time_pr(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 2> maxs = {
      (x.head<2>() - y.head<2>()).norm() / params.max_vel,
      so2_distance(x(2), y(2)) / params.max_angular_vel};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

double Model_unicycle2::lower_bound_time_vel(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &y) {

  std::array<double, 2> maxs = {std::abs(x(3) - y(3)) / params.max_acc_abs,
                                std::abs(x(4) - y(4)) /
                                    params.max_angular_acc_abs};

  auto it = std::max_element(maxs.cbegin(), maxs.cend());
  return *it;
}

//
// refactor yaml and boost stuff.
//
void Unicycle1_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml_eigen(node, VAR_WITH_NAME(size));
  set_from_yaml_eigen(node, VAR_WITH_NAME(distance_weights));
}

void Quad2dpole_params::read_from_yaml(YAML::Node &node) {

  set_from_yaml(node, VAR_WITH_NAME(yaw_max));
  set_from_yaml(node, VAR_WITH_NAME(m_p));
  set_from_yaml(node, VAR_WITH_NAME(r));

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_f));
  set_from_yaml(node, VAR_WITH_NAME(dt));

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_acc));

  set_from_yaml(node, VAR_WITH_NAME(m));
  set_from_yaml(node, VAR_WITH_NAME(I));
  set_from_yaml(node, VAR_WITH_NAME(l));

  set_from_yaml(node, VAR_WITH_NAME(drag_against_vel));
  set_from_yaml(node, VAR_WITH_NAME(k_drag_linear));
  set_from_yaml(node, VAR_WITH_NAME(k_drag_angular));

  set_from_yaml(node, VAR_WITH_NAME(shape));

  set_from_yaml_eigen(node, VAR_WITH_NAME(size));
  set_from_yaml_eigen(node, VAR_WITH_NAME(distance_weights));
}

void Quad2d_params::read_from_yaml(YAML::Node &node) {

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_f));
  set_from_yaml(node, VAR_WITH_NAME(dt));

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_acc));

  set_from_yaml(node, VAR_WITH_NAME(m));
  set_from_yaml(node, VAR_WITH_NAME(I));
  set_from_yaml(node, VAR_WITH_NAME(l));

  set_from_yaml(node, VAR_WITH_NAME(drag_against_vel));
  set_from_yaml(node, VAR_WITH_NAME(k_drag_linear));
  set_from_yaml(node, VAR_WITH_NAME(k_drag_angular));

  set_from_yaml(node, VAR_WITH_NAME(shape));

  set_from_yaml_eigen(node, VAR_WITH_NAME(size));
  set_from_yaml_eigen(node, VAR_WITH_NAME(distance_weights));
}

void Car_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(l));
  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_steering_abs));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(num_trailers));
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml(node, VAR_WITH_NAME(shape_trailer));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(diff_max_abs));

  set_from_yaml_eigen(node, VAR_WITH_NAME(size));
  set_from_yaml_eigen(node, VAR_WITH_NAME(size_trailer));

  set_from_yaml_eigenx(node, VAR_WITH_NAME(distance_weights));
  set_from_yaml_eigenx(node, VAR_WITH_NAME(hitch_lengths));

  assert(num_trailers == hitch_lengths.size());
}

Model_car2::Model_car2(const Car2_params &params, const Eigen::VectorXd &p_lb,
                       const Eigen::VectorXd &p_ub)
    : Model_robot(std::make_shared<RnSOn>(4, 1, std::vector<size_t>{2}), 2),
      params(params) {

  name = "car2";
  std::cout << "Robot name " << name << std::endl;
  std::cout << "Parameters" << std::endl;
  this->params.write(std::cout);
  std::cout << "***" << std::endl;

  u_lb << -params.max_acc_abs, -params.max_steer_vel_abs;
  u_ub << params.max_acc_abs, params.max_steer_vel_abs;
  ref_dt = params.dt;

  u_weight = V2d(.5, .5);
  nx_pr = 3;

  x_lb << RM_low__, RM_low__, RM_low__, params.min_vel,
      -params.max_steering_abs;
  x_ub << RM_max__, RM_max__, RM_max__, params.max_vel, params.max_steering_abs;

  x_desc = {"x [m]", "y [m]", "yaw [rad]", "v[m/s]", "phi[rad]"};
  u_desc = {"a [m/s^2]", "phiw [rad/s]"};

  translation_invariance = 2;
  is_2d = true;
  nx_col = 3;

  if (p_lb.size() && p_ub.size()) {
    set_position_lb(p_lb);
    set_position_ub(p_ub);
  }

  collision_geometries.emplace_back(
      std::make_shared<fcl::Boxd>(params.size[0], params.size[1], 1.0));

  ts_data.resize(1);
  col_outs.resize(1);
};

double Model_car2::distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &y) {

  CHECK_EQ(x.size(), 5, AT);
  CHECK_EQ(y.size(), 5, AT);
  CHECK_LEQ(y(2), M_PI, AT);
  CHECK_GEQ(y(2), -M_PI, AT);
  CHECK_LEQ(x(2), M_PI, AT);
  CHECK_GEQ(x(2), -M_PI, AT);
  CHECK_EQ(params.distance_weights.size(), 4, AT);
  double d = params.distance_weights(0) * (x.head<2>() - y.head<2>()).norm() +
             params.distance_weights(1) * so2_distance(x(2), y(2)) +
             params.distance_weights(2) * std::abs(x(3) - y(3)) +
             params.distance_weights(3) * std::abs(x(4) - y(4));

  return d;
}

void Model_car2::calcV(Eigen::Ref<Eigen::VectorXd> f,
                       const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<size_t>(f.size()) == nx);
  assert(static_cast<size_t>(x.size()) == nx);
  assert(static_cast<size_t>(u.size()) == nu);

  const double &v = x(3);
  const double &phi = x(4);
  const double &yaw = x(2);

  const double &c = std::cos(yaw);
  const double &s = std::sin(yaw);

  f(0) = v * c;
  f(1) = v * s;
  f(2) = v / params.l * std::tan(phi);
  // f(3) = params.max_acc_abs * u(0);
  // f(4) = params.max_steer_vel_abs * u(1);
  f(3) = u(0);
  if (x(4) + u(1) * ref_dt > -params.max_steering_abs &&
      x(4) + u(1) * ref_dt < params.max_steering_abs)
    f(4) = u(1);
  else
    f(4) = 0;
};

void Model_car2::calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                           Eigen::Ref<Eigen::MatrixXd> Jv_u,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) {

  CHECK_EQ(static_cast<size_t>(Jv_x.rows()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_u.rows()), nx, AT);

  CHECK_EQ(static_cast<size_t>(Jv_x.cols()), nx, AT);
  CHECK_EQ(static_cast<size_t>(Jv_u.cols()), nu, AT);

  CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
  CHECK_EQ(static_cast<size_t>(u.size()), nu, AT);

  const double &v = x(3);
  const double &phi = x(4);
  const double &yaw = x(2);

  const double &c = std::cos(yaw);
  const double &s = std::sin(yaw);

  Jv_x(0, 2) = -v * s;
  Jv_x(1, 2) = v * c;

  Jv_x(0, 3) = c;
  Jv_x(1, 3) = s;
  Jv_x(2, 3) = 1. / params.l * std::tan(phi);
  Jv_x(2, 4) = 1. * v / params.l / (std::cos(phi) * std::cos(phi));

  Jv_u(3, 0) = 1.;
  Jv_u(4, 1) = 1.;
}

void Car2_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(l));
  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_steering_abs));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml(node, VAR_WITH_NAME(shape_trailer));
  set_from_yaml(node, VAR_WITH_NAME(dt));

  set_from_yaml(node, VAR_WITH_NAME(max_acc_abs));
  set_from_yaml(node, VAR_WITH_NAME(max_steer_vel_abs));

  set_from_yaml_eigen(node, VAR_WITH_NAME(size));

  set_from_yaml_eigenx(node, VAR_WITH_NAME(distance_weights));
}

void Car2_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Car_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Unicycle1_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Quad2dpole_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Quad2d_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Acrobot_params::read_from_yaml(YAML::Node &node) {

  set_from_yaml(node, VAR_WITH_NAME(l1));
  set_from_yaml(node, VAR_WITH_NAME(l2));
  set_from_yaml(node, VAR_WITH_NAME(lc1));
  set_from_yaml(node, VAR_WITH_NAME(lc2));
  set_from_yaml(node, VAR_WITH_NAME(m1));
  set_from_yaml(node, VAR_WITH_NAME(m2));
  set_from_yaml(node, VAR_WITH_NAME(I1));
  set_from_yaml(node, VAR_WITH_NAME(I2));

  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_acc));

  set_from_yaml(node, VAR_WITH_NAME(distance_weight_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_torque));

  set_from_yaml_eigen(node, VAR_WITH_NAME(distance_weights));
}

void Acrobot_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Unicycle2_params::write(std::ostream &out) const {

  const std::string be = "";
  const std::string af = ": ";

  out << be << STR(max_vel, af) << std::endl;
  out << be << STR(min_vel, af) << std::endl;
  out << be << STR(max_angular_vel, af) << std::endl;
  out << be << STR(min_angular_vel, af) << std::endl;
  out << be << STR(max_acc_abs, af) << std::endl;
  out << be << STR(max_angular_acc_abs, af) << std::endl;
  out << be << STR(dt, af) << std::endl;
  out << be << STR(shape, af) << std::endl;
  out << be << STR_V(distance_weights) << std::endl;
  out << be << STR_V(size) << std::endl;
  out << be << STR(filename, af) << std::endl;
}

void Unicycle2_params::read_from_yaml(YAML::Node &node) {

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(min_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc_abs));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_acc_abs));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml_eigen(node, VAR_WITH_NAME(distance_weights));
  set_from_yaml_eigen(node, VAR_WITH_NAME(size));
  set_from_yaml(node, VAR_WITH_NAME(shape));
}

void Unicycle2_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Quad3d_params::read_from_yaml(YAML::Node &node) {

  set_from_yaml(node, VAR_WITH_NAME(max_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_vel));
  set_from_yaml(node, VAR_WITH_NAME(max_acc));
  set_from_yaml(node, VAR_WITH_NAME(max_angular_acc));
  set_from_yaml(node, VAR_WITH_NAME(motor_control));
  set_from_yaml(node, VAR_WITH_NAME(m));
  set_from_yaml(node, VAR_WITH_NAME(g));
  set_from_yaml(node, VAR_WITH_NAME(max_f));
  set_from_yaml(node, VAR_WITH_NAME(arm_length));
  set_from_yaml(node, VAR_WITH_NAME(t2t));
  set_from_yaml(node, VAR_WITH_NAME(dt));
  set_from_yaml(node, VAR_WITH_NAME(shape));
  set_from_yaml_eigen(node, VAR_WITH_NAME(J_v));
  set_from_yaml_eigen(node, VAR_WITH_NAME(distance_weights));
  set_from_yaml_eigen(node, VAR_WITH_NAME(u_ub));
  set_from_yaml_eigen(node, VAR_WITH_NAME(u_lb));

  set_from_yaml_eigenx(node, VAR_WITH_NAME(size));
}

void Quad3d_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  filename = file;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

std::unique_ptr<Model_robot> robot_factory(const char *file) {

  // open the robot

  // get the dynamics

  std::cout << "Robot Factory: loading file: " << file << std::endl;
  YAML::Node node = YAML::LoadFile(file);

  assert(node["dynamics"]);
  std::string dynamics = node["dynamics"].as<std::string>();
  std::cout << STR_(dynamics) << std::endl;

  if (dynamics == "unicycle1") {
    return std::make_unique<Model_unicycle1>(file);
  } else if (dynamics == "unicycle2") {
    return std::make_unique<Model_unicycle2>(file);
  } else if (dynamics == "quad2d") {
    return std::make_unique<Model_quad2d>(file);
  } else if (dynamics == "quad3d") {
    return std::make_unique<Model_quad3d>(file);
  } else if (dynamics == "acrobot") {
    return std::make_unique<Model_acrobot>(file);
  } else if (dynamics == "car_with_trailers") {
    return std::make_unique<Model_car_with_trailers>(file);
  } else if (dynamics == "car2") {
    return std::make_unique<Model_car2>(file);
  } else if (dynamics == "quad2dpole") {
    return std::make_unique<Model_quad2dpole>(file);
  } else {
    ERROR_WITH_INFO("dynamics not implemented");
  }
}
