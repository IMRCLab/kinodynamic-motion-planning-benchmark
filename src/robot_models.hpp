#pragma once
#include "Eigen/Core"
#include "croco_macros.hpp"
#include "fcl/broadphase/broadphase_collision_manager.h"
#include "for_each_macro.hpp"
#include "general_utils.hpp"
#include "math_utils.hpp"
#include "robot_models_base.hpp"
#include <algorithm>
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

struct Car_params {

  Car_params(const char *file) { read_from_yaml(file); }
  Car_params() = default;

  size_t num_trailers = 1;
  double dt = .1;
  double l = .25;
  double max_vel = .5;
  double min_vel = -.1;
  double max_steering_abs = M_PI / 3.;
  double max_angular_vel = 10;
  double diff_max_abs = M_PI / 4;

  std::string shape = "box";
  std::string shape_trailer = "box";
  std::string filename = "";

  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);
  Eigen::Vector2d size_trailer = Eigen::Vector2d(.3, .25);
  Eigen::VectorXd distance_weights = Eigen::Vector3d(1, .5, .5);
  Eigen::VectorXd hitch_lengths = Eigen::Matrix<double, 1, 1>(.5);

#define CAR_PARAMS_INOUT                                                       \
  num_trailers, dt, l, max_vel, min_vel, max_steering_abs, max_angular_vel,    \
      diff_max_abs, shape, shape_trailer, filename, size, size_trailer,        \
      distance_weights, hitch_lengths

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

#define X(a) out << be << STR(a, af) << std::endl;
    APPLYXn(CAR_PARAMS_INOUT);
#undef X
  }
};

inline Eigen::Matrix<double, 5, 1> mk_v5(double x0, double x1, double x2,
                                         double x3, double x4) {

  Eigen::Matrix<double, 5, 1> out;

  out << x0, x1, x2, x3, x4;

  return out;
}
// TODO: I should be able to check if goal is inside the  bounds
// Each model shoudl have: "state_constraints"
// and "state_constraints_diff"

struct Car2_params {

  Car2_params(const char *file) { read_from_yaml(file); }
  Car2_params() = default;
  using Vector5d = Eigen::Matrix<double, 5, 1>;

  double dt = .1;
  double l = .25;
  double max_vel = .5;
  double min_vel = -.1;
  double max_steering_abs = M_PI / 3.;
  double max_angular_vel = 10; // for bounds
  double max_acc_abs = 2.;
  double max_steer_vel_abs = 2 * M_PI;

  std::string shape = "box";
  std::string shape_trailer = "box";
  std::string filename = "";

  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);
  Eigen::VectorXd distance_weights = Eigen::Vector4d(1, .5, .2, .2);

#define CAR2_PARAMS_INOUT                                                      \
  dt, l, max_vel, min_vel, max_steering_abs, max_angular_vel, max_acc_abs,     \
      max_steer_vel_abs, shape, shape_trailer, filename, size,                 \
      distance_weights

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

#define X(a) out << be << STR(a, af) << std::endl;
    APPLYXn(CAR2_PARAMS_INOUT);
#undef X
  }
};

struct Model_car2 : Model_robot {
  virtual ~Model_car2() = default;

  Car2_params params;

  Model_car2(const Car2_params &params = Car2_params(),
             const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
             const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override {
    (void)x;

    ERROR_WITH_INFO("not implemented");
  };

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> f,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override {

    (void)xt;
    (void)from;
    (void)to;
    (void)dt;
    ERROR_WITH_INFO("not implemented");
  }

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override {
    (void)x;
    (void)y;

    ERROR_WITH_INFO("not implemented");
  }

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override {

    (void)x;
    (void)y;
    NOT_IMPLEMENTED;
  }

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override {

    (void)x;
    (void)y;
    NOT_IMPLEMENTED;
  }
};

struct Model_car_with_trailers : Model_robot {
  virtual ~Model_car_with_trailers() = default;

  Car_params params;

  // Model_car_with_trailers(const char *file)
  //     : Model_car_with_trailers(Car_params(file)) {}

  Model_car_with_trailers(const Car_params &params = Car_params(),
                          const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                          const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout(2) = wrap_angle(xin(2));
    if (params.num_trailers) {
      xout(3) = wrap_angle(xin(3));
    }
  }

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> f,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  // r <= 0 means feasible
  virtual void
  constraintsIneq(Eigen::Ref<Eigen::VectorXd> r,
                  const Eigen::Ref<const Eigen::VectorXd> &x,
                  const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void
  constraintsIneqDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                      Eigen::Ref<Eigen::MatrixXd> Ju,
                      const Eigen::Ref<const Eigen::VectorXd> x,
                      const Eigen::Ref<const Eigen::VectorXd> &u) override;

  // r (the cost is then .5 r^2)
  virtual void
  regularization_cost(Eigen::Ref<Eigen::VectorXd> r,
                      const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void
  regularization_cost_diff(Eigen::Ref<Eigen::MatrixXd> Jx,
                           Eigen::Ref<Eigen::MatrixXd> Ju,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override {

    return lower_bound_time(x, y);
  }

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override {

    (void)x;
    (void)y;
    return 0;
  }

  virtual void transformation_collision_geometries(
      const Eigen::Ref<const Eigen::VectorXd> &x,
      std::vector<Transform3d> &ts) override;
};

struct Acrobot_params {

  Acrobot_params(const char *file) { read_from_yaml(file); }
  Acrobot_params() = default;

  double l1 = 1;
  double l2 = 1;
  double lc1 = l1 / 2.;
  double lc2 = l2 / 2.;
  double m1 = 1;
  double m2 = 1;
  double I1 = 1. / 3. * m1 * l1 * l1; // Inertia w.r.t to PIVOT
  double I2 = 1. / 3. * m2 * l2 * l2;
  double dt = .01;

  double max_angular_vel = 8;  // TODO: what to put here?
  double max_angular_acc = 10; // TODo: what to put here?

  double distance_weight_angular_vel = .2;
  double max_torque = 10;
  Eigen::Vector3d distance_weights = Eigen::Vector3d(.5, .5, .2);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  std::string filename = "";
  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(l1, af) << std::endl;
    out << be << STR(l2, af) << std::endl;
    out << be << STR(lc1, af) << std::endl;
    out << be << STR(lc2, af) << std::endl;
    out << be << STR(m1, af) << std::endl;
    out << be << STR(m2, af) << std::endl;
    out << be << STR(I1, af) << std::endl;
    out << be << STR(I2, af) << std::endl;
    out << be << STR(dt, af) << std::endl;

    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(max_angular_acc, af) << std::endl;
    out << be << STR(filename, af) << std::endl;

    out << be << STR(distance_weight_angular_vel, af) << std::endl;
    out << be << STR(max_torque, af) << std::endl;
    out << be << STR_VV(distance_weights, af) << std::endl;
  }
};

struct Model_acrobot : Model_robot {

  virtual ~Model_acrobot() = default;
  Acrobot_params params;
  double g = 9.81;

  Model_acrobot(const char *file,
                const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_acrobot(Acrobot_params(file), p_lb, p_ub) {}

  Model_acrobot(const Acrobot_params &acrobot_params = Acrobot_params(),
                const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) override {
    x(2) = 0;
    x(3) = 0;
  }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout(0) = wrap_angle(xin(0));
    xout(1) = wrap_angle(xin(1));
  }

  double calcEnergy(const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> f,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &uu) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &uu) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual void transformation_collision_geometries(
      const Eigen::Ref<const Eigen::VectorXd> &x,
      std::vector<Transform3d> &ts) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override {

    (void)x;
    (void)y;
    NOT_IMPLEMENTED;
  }

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override {

    (void)x;
    (void)y;
    NOT_IMPLEMENTED;
  }
};

// struct Model_unicycle1_se2 : Model_robot {
//
//   virtual ~Model_unicycle1_se2() = default;
//   Model_unicycle1_se2() : Model_robot(3, 2) { nx_col = 3; }
//
//   virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
//                      const Eigen::Ref<const Eigen::VectorXd> &x,
//                      const Eigen::Ref<const Eigen::VectorXd> &u) override;
//
//   virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
//                     const Eigen::Ref<const Eigen::VectorXd> &x,
//                     const Eigen::Ref<const Eigen::VectorXd> &u,
//                     double dt) override;
// };

// struct Model_unicycle1_R2SO2 : Model_robot {
//
//   virtual ~Model_unicycle1_R2SO2() = default;
//
//   Model_unicycle1_R2SO2() : Model_robot(3, 2) { nx_col = 3; }
//
//   virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
//                      const Eigen::Ref<const Eigen::VectorXd> &x,
//                      const Eigen::Ref<const Eigen::VectorXd> &u) override;
//
//   virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
//                     const Eigen::Ref<const Eigen::VectorXd> &x,
//                     const Eigen::Ref<const Eigen::VectorXd> &u,
//                     double dt) override;
// };

struct Unicycle1_params {
  Unicycle1_params(const char *file) { read_from_yaml(file); }
  Unicycle1_params() = default;

  double max_vel = .5;
  double min_vel = -.5;
  double max_angular_vel = .5;
  double min_angular_vel = -.5;
  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);
  Eigen::Vector2d distance_weights = Eigen::Vector2d(1, .5);
  std::string shape = "box";
  double dt = .1;
  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
  std::string filename = "";
  void inline write(std::ostream &out) const {
    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(min_vel, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(min_angular_vel, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR(dt, af) << std::endl;
    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(distance_weights, af) << std::endl;
    out << be << STR(filename, af) << std::endl;
  }
};

struct Model_unicycle1 : Model_robot {

  virtual ~Model_unicycle1() = default;

  Unicycle1_params params;

  Model_unicycle1(const char *file,
                  const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                  const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_unicycle1(Unicycle1_params(file), p_lb, p_ub) {}

  virtual void write_params(std::ostream &out) override { params.write(out); }

  Model_unicycle1(const Unicycle1_params &params = Unicycle1_params(),
                  const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                  const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout(2) = wrap_angle(xin(2));
  }

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override {

    return lower_bound_time(x, y);
  }

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override {

    (void)x;
    (void)y;
    return 0;
  }
};

struct Quad3d_params {

  Quad3d_params(const char *file) { read_from_yaml(file); }
  Quad3d_params() = default;

  double max_vel = 4;
  double max_angular_vel = 8;

  double max_acc = 25;
  double max_angular_acc = 20;

  bool motor_control = true;

  double m = 0.034; // kg
  double g = 9.81;
  double max_f = 1.3;        // thrust to weight ratio
  double arm_length = 0.046; // m
  double t2t = 0.006;        // thrust-to-torque ratio
  double dt = .01;
  std::string shape = "sphere";
  //
  Eigen::Vector4d distance_weights = Eigen::Vector4d(1, 1, .1, .1);
  Eigen::Vector4d u_ub;
  Eigen::Vector4d u_lb;

  Eigen::Vector3d J_v =
      Eigen::Vector3d(16.571710e-6, 16.655602e-6, 29.261652e-6);

  Eigen::VectorXd size = Eigen::Matrix<double, 1, 1>(.4);

  // continue here!!
  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  std::string filename = "";
  void write(std::ostream &out) {
    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(max_acc, af) << std::endl;
    out << be << STR(max_angular_acc, af) << std::endl;
    out << be << STR(motor_control, af) << std::endl;
    out << be << STR(m, af) << std::endl;
    out << be << STR(g, af) << std::endl;
    out << be << STR(max_f, af) << std::endl;
    out << be << STR(arm_length, af) << std::endl;
    out << be << STR(t2t, af) << std::endl;
    out << be << STR(dt, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR(filename, af) << std::endl;

    out << be << STR_VV(distance_weights, af) << std::endl;
    out << be << STR_VV(J_v, af) << std::endl;
    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(u_lb, af) << std::endl;
    out << be << STR_VV(u_ub, af) << std::endl;
  }
};

struct Model_quad3d : Model_robot {

  using Vector12d = Eigen::Matrix<double, 12, 1>;
  using Matrix34 = Eigen::Matrix<double, 3, 4>;

  virtual ~Model_quad3d() = default;

  struct Data {
    Eigen::Vector3d f_u;
    Eigen::Vector3d tau_u;
    Eigen::Matrix<double, 13, 1> xnext;
    Matrix34 Jx;
    Eigen::Matrix3d Ja;
  } data;

  Vector12d ff;
  Quad3d_params params;

  // Eigen::Matrix<double, 12, 13> Jv_x;
  // Eigen::Matrix<double, 12, 4> Jv_u;

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) override {
    x.segment<6>(7).setZero();
  }

  double arm;
  double g = 9.81;

  double u_nominal;
  double m_inv;
  double m;
  Eigen::Vector3d inverseJ_v;

  Eigen::Matrix3d inverseJ_M;
  Eigen::Matrix3d J_M;

  Eigen::Matrix3d inverseJ_skew;
  Eigen::Matrix3d J_skew;

  Eigen::Vector3d grav_v;

  Eigen::Matrix4d B0;
  Eigen::Matrix4d B0inv;

  Matrix34 Fu_selection;
  Matrix34 Ftau_selection;

  Matrix34 Fu_selection_B0;
  Matrix34 Ftau_selection_B0;

  const bool adapt_vel = true;

  Model_quad3d(const Model_quad3d &) = default;

  Model_quad3d(const char *file,
               const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
               const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_quad3d(Quad3d_params(file), p_lb, p_ub) {}

  Model_quad3d(const Quad3d_params &params = Quad3d_params(),
               const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
               const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {
    xout = xin;
    xout.segment<4>(3).normalize();
  }

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual Eigen::VectorXd get_x0(const Eigen::VectorXd &x) override;

  virtual void
  motorForcesFromThrust(Eigen::Ref<Eigen::VectorXd> f,
                        const Eigen::Ref<const Eigen::VectorXd> tm) {

    // Eigen::Vector4d eta = B0 * u_nominal * f;
    // f_u << 0, 0, eta(0);
    // tau_u << eta(1), eta(2), eta(3);
    f = B0inv * tm / u_nominal;
  }

  virtual void
  transform_primitive(const Eigen::Ref<const Eigen::VectorXd> &p,
                      const std::vector<Eigen::VectorXd> &xs_in,
                      const std::vector<Eigen::VectorXd> &us_in,
                      std::vector<Eigen::VectorXd> &xs_out,
                      std::vector<Eigen::VectorXd> &us_out) override;

  virtual void offset(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> p) override {
    CHECK_EQ(p.size(), 6, AT);
    if (adapt_vel) {
      p.head<3>() = xin.head<3>();
      p.tail<3>() = xin.segment<3>(7);
    } else {
      Model_robot::offset(xin, p);
    }
  }

  virtual size_t get_offset_dim() override { return adapt_vel ? 6 : 3; }

  virtual void canonical_state(const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) override {

    if (adapt_vel) {
      xout = xin;
      xout.head<3>().setZero();
      xout.segment<3>(7).setZero();
    } else {
      Model_robot::canonical_state(xin, xout);
    }
  }

  virtual void transform_state(const Eigen::Ref<const Eigen::VectorXd> &p,
                               const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) override {

    CHECK((p.size() == 3 || p.size() == 6), AT);
    if (p.size() == 3) {
      Model_robot::transform_state(p, xin, xout);
    } else if (p.size() == 6) {
      xout.head<3>() += p.head<3>();
      xout.segment<3>(7) += p.tail<3>();
    }
  }

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> f,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u,
                    double dt) override;

  virtual void stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u,
                        double dt) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual void transformation_collision_geometries(
      const Eigen::Ref<const Eigen::VectorXd> &x,
      std::vector<Transform3d> &ts) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override;
};

struct Quad2d_params {

  Quad2d_params(const char *file) { read_from_yaml(file); }

  Quad2d_params() = default;

  double max_f = 1.3;
  double dt = .01;

  double max_vel = 4;
  double max_angular_vel = 8;
  double max_acc = 25;
  double max_angular_acc = 25;

  // Big drone
  // double m = 2.5;
  // double I = 1.2;
  // double l = .5;

  // Crazy fly - style
  double m = 0.034;
  double I = 1e-4;
  double l = 0.1;

  bool drag_against_vel = false;
  double k_drag_linear = .0001;
  double k_drag_angular = .0001;
  std::string shape = "box";

  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);
  Eigen::Vector4d distance_weights = Eigen::Vector4d(1, .5, .2, .2);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  std::string filename = "";
  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(max_f, af) << std::endl;
    out << be << STR(dt, af) << std::endl;

    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(max_acc, af) << std::endl;
    out << be << STR(max_angular_acc, af) << std::endl;

    // Big drone
    // double m = 2.5;
    // double I = 1.2;
    // double l = .5;

    // Crazy fly - style
    out << be << STR(m, af) << std::endl;
    out << be << STR(I, af) << std::endl;
    out << be << STR(l, af) << std::endl;

    out << be << STR(drag_against_vel, af) << std::endl;
    out << be << STR(k_drag_linear, af) << std::endl;
    out << be << STR(k_drag_angular, af) << std::endl;
    out << be << STR(shape, af) << std::endl;

    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(distance_weights, af) << std::endl;
    out << be << STR(filename, af) << std::endl;
  }
};

struct Quad2dpole_params {

  using Vector6d = Eigen::Matrix<double, 6, 1>;
  Quad2dpole_params(const char *file) {
    distance_weights << 1, .5, .5, .2, .2, .2;
    read_from_yaml(file);
  }

  Quad2dpole_params() { distance_weights << 1, .5, .5, .2, .2, .2; }

  double yaw_max = 1.57;
  double max_f = 1.3;
  double dt = .01;

  double max_vel = 4;
  double max_angular_vel = 8;
  double max_acc = 25;
  double max_angular_acc = 25;

  double m_p = .5; // pendulum mass
  double r = 1;    // pendulum radius

  // Big drone
  // double m = 2.5;
  // double I = 1.2;
  // double l = .5;

  // Crazy fly - style
  double m = 0.034;
  double I = 1e-4;
  double l = 0.1;

  bool drag_against_vel = false;
  double k_drag_linear = .0001;
  double k_drag_angular = .0001;
  std::string shape = "box";

  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);

  // using Vector6d = Eigen::Matrix<double, 6, 1>;
  Vector6d distance_weights;

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  std::string filename = "";
  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(yaw_max, af) << std::endl;
    out << be << STR(max_f, af) << std::endl;
    out << be << STR(dt, af) << std::endl;
    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(max_acc, af) << std::endl;
    out << be << STR(max_angular_acc, af) << std::endl;
    out << be << STR(m_p, af) << std::endl;
    out << be << STR(r, af) << std::endl;
    out << be << STR(m, af) << std::endl;
    out << be << STR(I, af) << std::endl;
    out << be << STR(l, af) << std::endl;
    out << be << STR(drag_against_vel, af) << std::endl;
    out << be << STR(k_drag_linear, af) << std::endl;
    out << be << STR(k_drag_angular, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(distance_weights, af) << std::endl;
    out << be << STR(filename, af) << std::endl;
  }
};

struct Model_quad2dpole : Model_robot {

  using Vector6d = Eigen::Matrix<double, 6, 1>;
  virtual ~Model_quad2dpole() = default;
  Quad2dpole_params params;

  const double g = 9.81;
  double u_nominal;

  Model_quad2dpole(const char *file,
                   const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                   const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_quad2dpole(Quad2dpole_params(file), p_lb, p_ub) {}

  Model_quad2dpole(const Quad2dpole_params &params = Quad2dpole_params(),
                   const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                   const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {
    xout = xin;
    xout(2) = wrap_angle(xout(2));
    xout(3) = wrap_angle(xout(3));
  }

  virtual void transformation_collision_geometries(
      const Eigen::Ref<const Eigen::VectorXd> &x,
      std::vector<Transform3d> &ts) override;

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) override {
    x.segment(4, 4).setZero();
  }

  virtual Eigen::VectorXd get_x0(const Eigen::VectorXd &x) override {
    CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
    Eigen::VectorXd out(nx);
    out.setZero();
    out.head(2) = x.head(2);
    return out;
  }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void offset(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> p) override {

    CHECK((static_cast<size_t>(p.size()) == 4 ||
           static_cast<size_t>(p.size()) == 2),
          AT);

    if (p.size() == 4) {
      p.head(2) = xin.head(2);       // x,y
      p.tail(2) = xin.segment(4, 2); // vx,vy
    } else {
      NOT_IMPLEMENTED;
    }
  }

  virtual void canonical_state(const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout.head(2).setZero();
    xout.segment(4, 2).setZero();
  }

  virtual size_t get_offset_dim() override { return 4; }

  virtual void
  transform_primitive(const Eigen::Ref<const Eigen::VectorXd> &p,
                      const std::vector<Eigen::VectorXd> &xs_in,
                      const std::vector<Eigen::VectorXd> &us_in,
                      std::vector<Eigen::VectorXd> &xs_out,
                      std::vector<Eigen::VectorXd> &us_out) override {

    CHECK_EQ(us_out.size(), us_in.size(), AT);
    CHECK_EQ(xs_out.size(), xs_in.size(), AT);
    CHECK_EQ(xs_out.front().size(), xs_in.front().size(), AT);
    CHECK_EQ(us_out.front().size(), us_in.front().size(), AT);

    CHECK((p.size() == 2 || 4), AT);

    if (p.size() == 2) {
      Model_robot::transform_primitive(p, xs_in, us_in, xs_out, us_out);
    } else {

      for (size_t i = 0; i < us_in.size(); i++) {
        us_out[i] = us_in[i];
      }

      xs_out.front() = xs_in.front();
      xs_out.front().head(2) += p.head(2);
      xs_out.front().segment(4, 2) += p.tail(2);
      rollout(xs_out.front(), us_in, xs_out);
    }
  }
};

struct Model_quad2d : Model_robot {

  virtual ~Model_quad2d() = default;
  Quad2d_params params;

  const double g = 9.81;
  double u_nominal;

  Model_quad2d(const char *file,
               const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
               const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_quad2d(Quad2d_params(file), p_lb, p_ub) {}

  Model_quad2d(const Quad2d_params &params = Quad2d_params(),
               const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
               const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout(2) = wrap_angle(xin(2));
  }

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) override {
    x(3) = 0;
    x(4) = 0;
    x(5) = 0;
  }

  virtual Eigen::VectorXd get_x0(const Eigen::VectorXd &x) override {
    CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
    Eigen::VectorXd out(nx);
    out.setZero();
    out.head(2) = x.head(2);
    return out;
  }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void offset(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> p) override {

    CHECK((static_cast<size_t>(p.size()) == 4 ||
           static_cast<size_t>(p.size()) == 2),
          AT);

    if (p.size() == 4) {
      p.head(2) = xin.head(2);       // x,y
      p.tail(2) = xin.segment(3, 2); // vx,vy
    } else {
      NOT_IMPLEMENTED;
    }
  }

  virtual void canonical_state(const Eigen::Ref<const Eigen::VectorXd> &xin,
                               Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout.head(2).setZero();
    xout.segment(3, 2).setZero();
  }

  virtual size_t get_offset_dim() override { return 4; }

  virtual void
  transform_primitive(const Eigen::Ref<const Eigen::VectorXd> &p,
                      const std::vector<Eigen::VectorXd> &xs_in,
                      const std::vector<Eigen::VectorXd> &us_in,
                      std::vector<Eigen::VectorXd> &xs_out,
                      std::vector<Eigen::VectorXd> &us_out) override {

    CHECK((p.size() == 2 || 4), AT);

    CHECK_EQ(us_out.size(), us_in.size(), AT);
    CHECK_EQ(xs_out.size(), xs_in.size(), AT);
    CHECK_EQ(xs_out.front().size(), xs_in.front().size(), AT);
    CHECK_EQ(us_out.front().size(), us_in.front().size(), AT);

    if (p.size() == 2) {
      Model_robot::transform_primitive(p, xs_in, us_in, xs_out, us_out);
    } else {

      for (size_t i = 0; i < us_in.size(); i++) {
        us_out[i] = us_in[i];
      }

      xs_out.front() = xs_in.front();
      xs_out.front().head(2) += p.head(2);
      xs_out.front().segment(3, 2) += p.tail(2);
      rollout(xs_out.front(), us_in, xs_out);
    }
  }
};

struct Unicycle2_params {

  Unicycle2_params(const char *file) { read_from_yaml(file); }

  Unicycle2_params() = default;

  using Vector5d = Eigen::Matrix<double, 5, 1>;

  double max_vel = .5;
  double min_vel = -.5;
  double max_angular_vel = .5;
  double min_angular_vel = -.5;
  double max_acc_abs = .25;
  double max_angular_acc_abs = .25;
  double dt = .1;

  std::string shape = "box";
  Eigen::Vector4d distance_weights = Eigen::Vector4d(1., .5, .25, .25);
  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  std::string filename = "";
  void write(std::ostream &out) const;
};

struct Model_unicycle2 : Model_robot {

  virtual ~Model_unicycle2() = default;
  Unicycle2_params params;

  Model_unicycle2(const char *file,
                  const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                  const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_unicycle2(Unicycle2_params(file), p_lb, p_ub) {}

  Model_unicycle2(const Unicycle2_params &params = Unicycle2_params(),
                  const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                  const Eigen::VectorXd &p_ub = Eigen::VectorXd()

  );

  virtual void set_0_velocity(Eigen::Ref<Eigen::VectorXd> x) override {
    x(3) = 0;
    x(4) = 0;
  }

  virtual void ensure(const Eigen::Ref<const Eigen::VectorXd> &xin,
                      Eigen::Ref<Eigen::VectorXd> xout) override {

    xout = xin;
    xout(2) = wrap_angle(xin(2));
  }

  virtual void write_params(std::ostream &out) override { params.write(out); }

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> f,
                     const Eigen::Ref<const Eigen::VectorXd> &x,
                     const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual void calcDiffV(Eigen::Ref<Eigen::MatrixXd> Jv_x,
                         Eigen::Ref<Eigen::MatrixXd> Jv_u,
                         const Eigen::Ref<const Eigen::VectorXd> &x,
                         const Eigen::Ref<const Eigen::VectorXd> &u) override;

  virtual double distance(const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual void interpolate(Eigen::Ref<Eigen::VectorXd> xt,
                           const Eigen::Ref<const Eigen::VectorXd> &from,
                           const Eigen::Ref<const Eigen::VectorXd> &to,
                           double dt) override;

  virtual double
  lower_bound_time(const Eigen::Ref<const Eigen::VectorXd> &x,
                   const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_pr(const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &y) override;

  virtual double
  lower_bound_time_vel(const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &y) override;
};

std::unique_ptr<Model_robot>
robot_factory(const char *file, const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
              const Eigen::VectorXd &p_ub = Eigen::VectorXd());

inline std::string robot_type_to_path(const std::string &robot_type) {
  const std::string base_path = "../models/";
  const std::string suffix = ".yaml";
  return base_path + robot_type + suffix;
}
