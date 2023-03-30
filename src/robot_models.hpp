#pragma once
#include "Eigen/Core"
#include "croco_macros.hpp"
#include "fcl/broadphase/broadphase_collision_manager.h"
#include "general_utils.hpp"
#include "math_utils.hpp"
#include <algorithm>
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

struct Model_robot;

double check_u_bounds(const std::vector<Eigen::VectorXd> &us_out,
                      std::shared_ptr<Model_robot> model);

double check_x_bounds(const std::vector<Eigen::VectorXd> &xs_out,
                      std::shared_ptr<Model_robot> model);

void get_states_and_actions(const YAML::Node &data,
                            std::vector<Eigen::VectorXd> &states,
                            std::vector<Eigen::VectorXd> &actions);

struct Problem {
  using Vxd = Eigen::VectorXd;

  Problem(const char *file) { read_from_yaml(file); }
  Problem() = default;


  std::string name; // name of the proble: E.g. bugtrap-car1

  Eigen::VectorXd goal;
  Eigen::VectorXd start;

  Eigen::VectorXd p_lb; // position bounds
  Eigen::VectorXd p_ub; // position bounds

  std::vector<Obstacle> obstacles;
  std::string robotType;

  void read_from_yaml(const YAML::Node &env);

  void read_from_yaml(const char *file);

  void write_to_yaml(const char *file);
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
  size_t nx_col = 0; // only the first nx_col variables have non zero gradient

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

  virtual void calcV(Eigen::Ref<Eigen::VectorXd> v,
                     const Eigen::Ref<const Eigen::VectorXd> &d,
                     const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void step(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

  virtual void stepR4(Eigen::Ref<Eigen::VectorXd> xnext,
                      const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

  virtual void stepDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u, double dt);

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

  virtual void load_env_quim(const Problem &problem);

  virtual ~Model_robot() = default;
};

// struct Robot_params {
//
//   void virtual read_from_yaml(YAML::Node &node) {}
//   void virtual read_from_yaml(const char *file) {}
//
// }

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

  std::string shape = "box";
  std::string shape_trailer = "box";
  std::string filename = "";

  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);
  Eigen::Vector2d size_trailer = Eigen::Vector2d(.3, .25);
  Eigen::VectorXd distance_weights = Eigen::Vector3d(1, .5, .5);
  Eigen::VectorXd hitch_lengths = Eigen::Matrix<double, 1, 1>(.5);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(num_trailers, af) << std::endl;
    out << be << STR(dt, af) << std::endl;
    out << be << STR(l, af) << std::endl;
    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(min_vel, af) << std::endl;
    out << be << STR(max_steering_abs, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR(shape_trailer, af) << std::endl;
    out << be << STR(filename, af) << std::endl;

    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(size_trailer, af) << std::endl;
    out << be << STR_VV(distance_weights, af) << std::endl;
    out << be << STR_VV(hitch_lengths, af) << std::endl;
  }
};

inline Eigen::Matrix<double, 5, 1> mk_v5(double x0, double x1, double x2,
                                         double x3, double x4) {

  Eigen::Matrix<double, 5, 1> out;

  out << x0, x1, x2, x3, x4;

  return out;
}

struct Car2_params {

  Car2_params(const char *file) { read_from_yaml(file); }
  Car2_params() = default;
  using Vector5d = Eigen::Matrix<double, 5, 1>;

  double dt = .1;
  double l = .25;
  double max_vel = .5;
  double min_vel = -.1;
  double max_steering_abs = M_PI / 3.;
  double max_angular_vel = 10;
  double max_acc_abs = .1;
  double max_steer_vel_abs = M_PI;

  std::string shape = "box";
  std::string shape_trailer = "box";
  std::string filename = "";

  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);
  Eigen::VectorXd distance_weights = Eigen::Vector4d(1, .5, .2, .2);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);

  void inline write(std::ostream &out) const {

    const std::string be = "";
    const std::string af = ": ";

    out << be << STR(dt, af) << std::endl;
    out << be << STR(l, af) << std::endl;
    out << be << STR(max_vel, af) << std::endl;
    out << be << STR(min_vel, af) << std::endl;
    out << be << STR(max_steering_abs, af) << std::endl;
    out << be << STR(max_angular_vel, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR(shape_trailer, af) << std::endl;
    out << be << STR(filename, af) << std::endl;
    out << be << STR(max_acc_abs, af) << std::endl;
    out << be << STR(max_steer_vel_abs, af) << std::endl;
    out << be << STR_VV(size, af) << std::endl;
    out << be << STR_VV(distance_weights, af) << std::endl;
  }
};

struct Model_car2 : Model_robot {
  virtual ~Model_car2() = default;

  Car2_params params;

  // Model_car_with_trailers(const char *file)
  //     : Model_car_with_trailers(Car_params(file)) {}

  Model_car2(const Car2_params &params = Car2_params(),
             const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
             const Eigen::VectorXd &p_ub = Eigen::VectorXd());

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
};

struct Model_car_with_trailers : Model_robot {
  virtual ~Model_car_with_trailers() = default;

  Car_params params;

  // Model_car_with_trailers(const char *file)
  //     : Model_car_with_trailers(Car_params(file)) {}

  Model_car_with_trailers(const Car_params &params = Car_params(),
                          const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                          const Eigen::VectorXd &p_ub = Eigen::VectorXd());

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
  Eigen::Vector4d distance_weights = Eigen::Vector4d(1, 1, .2, .2);

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

  virtual void sample_uniform(Eigen::Ref<Eigen::VectorXd> x) override;

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

  Model_unicycle1(const Unicycle1_params &params = Unicycle1_params(),
                  const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
                  const Eigen::VectorXd &p_ub = Eigen::VectorXd());

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
  }
};

struct Model_quad3d : Model_robot {

  virtual ~Model_quad3d() = default;

  using Matrix34 = Eigen::Matrix<double, 3, 4>;

  struct Data {
    Eigen::Vector3d f_u;
    Eigen::Vector3d tau_u;
    Eigen::VectorXd xnext{13};
    Matrix34 Jx;
    Eigen::Matrix3d Ja;
  } data;

  Quad3d_params params;
  double arm;
  double g = 9.81;

  double u_nominal;
  double m_inv;
  Eigen::Vector3d inverseJ_v;

  Eigen::Matrix3d inverseJ_M;
  Eigen::Matrix3d J_M;

  Eigen::Matrix3d inverseJ_skew;
  Eigen::Matrix3d J_skew;

  Eigen::Vector3d grav_v;

  Eigen::Matrix4d B0;

  Matrix34 Fu_selection;
  Matrix34 Ftau_selection;

  Matrix34 Fu_selection_B0;
  Matrix34 Ftau_selection_B0;

  Model_quad3d(const char *file,
               const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
               const Eigen::VectorXd &p_ub = Eigen::VectorXd())
      : Model_quad3d(Quad3d_params(file), p_lb, p_ub) {}

  Model_quad3d(const Quad3d_params &params = Quad3d_params(),
               const Eigen::VectorXd &p_lb = Eigen::VectorXd(),
               const Eigen::VectorXd &p_ub = Eigen::VectorXd());

  virtual Eigen::VectorXd get_x0(const Eigen::VectorXd &x) override {
    CHECK_EQ(static_cast<size_t>(x.size()), nx, AT);
    Eigen::VectorXd out(nx);
    out.setZero();
    out.head(3) = x.head(3);
    out(6) = 1.;
    return out;
  }

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

  // Eigen::Vector2d u_ub = Eigen::Vector2d(.25, .25);
  // Eigen::Vector2d u_lb = Eigen::Vector2d(-.25, -.25);
  //
  // Vector5d x_ub = Vector5d(1e10, 1e10, M_PI, .5, .5);
  // Vector5d x_lb = Vector5d(-1e10, 1e10, M_PI, -.5, -.5);

  Eigen::Vector4d distance_weights = Eigen::Vector4d(1., .5, .25, .25);
  Eigen::Vector2d size = Eigen::Vector2d(.5, .25);

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
    out << be << STR(max_acc_abs, af) << std::endl;
    out << be << STR(max_angular_acc_abs, af) << std::endl;
    out << be << STR(dt, af) << std::endl;
    out << be << STR(shape, af) << std::endl;
    out << be << STR(distance_weights, af) << std::endl;
    out << be << STR(size, af) << std::endl;
    out << be << STR(filename, af) << std::endl;
  }
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
};

std::unique_ptr<Model_robot> robot_factory(const char *file);

double check_trajectory(const std::vector<Eigen::VectorXd> &xs_out,
                        const std::vector<Eigen::VectorXd> &us_out,
                        const Eigen::VectorXd &dt,
                        std::shared_ptr<Model_robot> model);

double check_cols(std::shared_ptr<Model_robot> model_robot,
                  const std::vector<Eigen::VectorXd> &xs);

struct Trajectory {

  double time_stamp; // when it was generated?
  double cost;
  bool feasible = 0;

  bool traj_feas = 0;
  bool goal_feas = 0;
  bool start_feas = 0;
  bool col_feas = 0;
  bool x_bounds_feas = 0;
  bool u_bounds_feas = 0;

  double max_jump = 0.;
  double max_collision = 0.;
  double goal_distance = 0.;
  double start_distance = 0.;
  double x_bound_distance = 0.;
  double u_bound_distance = 0.;

  Trajectory() = default;
  Trajectory(const char *file) { read_from_yaml(file); }

  Eigen::VectorXd start;
  Eigen::VectorXd goal;
  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> actions;
  size_t num_time_steps = 0; // use this if we want default init guess.
  Eigen::VectorXd times;

  void to_yaml_format(std::ostream &out, std::string prefix = "");

  void read_from_yaml(const YAML::Node &node);

  void read_from_yaml(const char *file);

  void check(std::shared_ptr<Model_robot> &robot);

  void update_feasibility(double traj_tol = 1e-2, double goal_tol = 1e-2,
                          double col_tol = 1e-2, double x_bound_tol = 1e-2,
                          double u_bound_tol = 1e-2);
};

double max_rollout_error(std::shared_ptr<Model_robot> robot,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us);

void resample_trajectory(std::vector<Eigen::VectorXd> &xs_out,
                         std::vector<Eigen::VectorXd> &us_out,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us,
                         const Eigen::VectorXd &ts, double ref_dt);
