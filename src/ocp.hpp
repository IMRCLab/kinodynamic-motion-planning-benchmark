#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>

#include "Eigen/Core"
#include <boost/program_options.hpp>

// #include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/states/euclidean.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"

#include "collision_checker.hpp"
#include "croco_macros.hpp"

template <class T> using ptr = boost::shared_ptr<T>;

template <typename T, typename... Args> auto mk(Args &&...args) {
  return boost::make_shared<T>(std::forward<Args>(args)...);
}

using namespace crocoddyl;

namespace po = boost::program_options;

template <typename T> bool __in(const std::vector<T> &v, const T &val) {
  return std::find(v.cbegin(), v.cend(), val) != v.cend();
}

typedef Eigen::Matrix<double, 3, 4> Matrix34;

template <typename T1, typename Fun>
bool __in_if(const std::vector<T1> &v, Fun fun) {
  return std::find_if(v.cbegin(), v.cend(), fun) != v.cend();
}

void inline finite_diff_jac(
    std::function<void(const Eigen::VectorXd &, Eigen::Ref<Eigen::VectorXd>)>
        fun,
    const Eigen::VectorXd &x, size_t nout, Eigen::Ref<Eigen::MatrixXd> J,
    double eps = 1e-6) {

  Eigen::VectorXd y(nout);

  fun(x, y);

  for (size_t i = 0; i < x.size(); i++) {
    Eigen::VectorXd xe = x;
    Eigen::VectorXd ye(nout);
    xe(i) += eps;
    fun(xe, ye);
    J.col(i) = (ye - y) / eps;
  }
};

void quat_product(const Eigen::Vector4d &p, const Eigen::Vector4d &q,
                  Eigen::Ref<Eigen::VectorXd> out,
                  Eigen::Ref<Eigen::Matrix4d> Jp,
                  Eigen::Ref<Eigen::Matrix4d> Jq);

void normalize(const Eigen::Ref<const Eigen::Vector4d> &q,
               Eigen::Ref<Eigen::Vector4d> y, Eigen::Ref<Eigen::Matrix4d> J);

// quaternion: real part last, and not normalized
void rotate_with_q(const Eigen::Ref<const Eigen::Vector4d> &x,
                   const Eigen::Ref<const Eigen::Vector3d> &a,
                   Eigen::Ref<Eigen::Vector3d> y, Eigen::Ref<Matrix34> Jx,
                   Eigen::Ref<Eigen::Matrix3d> Ja);

template <typename T>
void set_from_yaml(YAML::Node &node, T &var, const char *name) {

  if (YAML::Node parameter = node[name]) {
    var = parameter.as<T>();
  }
}

template <typename T>
void set_from_boostop(po::options_description &desc, T &var, const char *name) {
  // std::cout << var << std::endl;
  // std::cout << NAMEOF(var) << std::endl;
  desc.add_options()(name, po::value<T>(&var));
}

inline std::string get_time_stamp() {

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y--%H-%M-%S");
  auto str = oss.str();

  return str;
}

Eigen::Quaterniond qintegrate(const Eigen::Quaterniond &q,
                              const Eigen::Vector3d &omega, float dt);

struct Opti_params {

  bool CALLBACKS = true;
  std::string solver_name;
  bool use_finite_diff = false;
  bool use_warmstart = true;
  bool repair_init_guess = true;
  bool control_bounds = true;
  bool states_reg = false;
  int solver_id = 0;
  double disturbance =1e-4;

  double th_stop = 1e-2;
  double init_reg = 1e2;
  double th_acceptnegstep = .3;
  double noise_level = 1e-3; // factor on top of [-1., 1.]
  double k_linear = 10.;
  double k_contour = 10.;

  size_t max_iter = 50;
  size_t window_optimize = 20;
  size_t window_shift = 10;
  size_t max_mpc_iterations = 50;
  std::string debug_file_name = "debug_file.yaml";
  double weight_goal = 200.;
  double collision_weight = 100.;
  bool smooth_traj = false;
  bool shift_repeat = true;

  double tsearch_max_rate = 2;
  double tsearch_min_rate = .7;
  int tsearch_num_check = 20;

  void add_options(po::options_description &desc);
  void print(std::ostream &out);

  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
};

extern Opti_params opti_params;

void linearInterpolation(const Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &x, double t_query,
                         Eigen::Ref<Eigen::VectorXd> out,
                         Eigen::Ref<Eigen::VectorXd> Jx);

struct Interpolator {

  Eigen::VectorXd times;
  std::vector<Eigen::VectorXd> x;

  Interpolator(const Eigen::VectorXd &times,
               const std::vector<Eigen::VectorXd> &x)
      : times(times), x(x) {
    CHECK_EQ(times.size(), x.size(), AT);
  }

  void inline interpolate(double t_query, Eigen::Ref<Eigen::VectorXd> out,
                          Eigen::Ref<Eigen::VectorXd> J) {
    linearInterpolation(times, x, t_query, out, J);
  }
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

double inline diff_angle(double angle1, double angle2) {
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

  Eigen::VectorXd uref;

  virtual ~Dynamics(){};
};

struct Dummy_dynamics : Dynamics {

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  Dummy_dynamics(size_t t_nx, size_t t_nu) {
    nx = t_nx;
    nu = t_nu;
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u){};

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) {}
};

struct Dynamics_contour : Dynamics {

  ptr<Dynamics> dyn;
  bool accumulate = false;
  Dynamics_contour(ptr<Dynamics> dyn, bool accumulate);

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) override;

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) override;
};

struct Dynamics_unicycle2 : Dynamics {

  double dt = .1;

  bool free_time;
  Dynamics_unicycle2(bool free_time = false);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

struct Dynamics_car_with_trailers : Dynamics {

  double dt = .1;
  size_t num_trailers;

  double l = .25;
  Eigen::VectorXd hitch_lengths;

  bool free_time;
  Dynamics_car_with_trailers(const Eigen::VectorXd &hitch_lengths = {},
                             bool free_time = false);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

struct Dynamics_quadcopter2d : Dynamics {

  struct Dynamics_quadcopter2d_data {
    double xdotdot;
    double ydotdot;
    double thetadotdot;
  } data;

  double dt = .01;
  double g = 9.8;

  bool free_time;

  // Big drone
  // double m = 2.5;
  // double I = 1.2;
  // double l = .5;

  // Crazy fly - style
  const double m = 0.034;
  const double I = 1e-4;
  const double l = 0.1;

  double m_inv = 1. / m;
  double I_inv = 1. / I;

  bool drag_against_vel = true;
  double k_drag_linear = .02;
  double k_drag_angular = .01;
  const double u_nominal = m * g / 2.;

  Dynamics_quadcopter2d(bool free_time = false);

  void compute_acc(const Eigen::Ref<const VectorXs> &x,
                   const Eigen::Ref<const VectorXs> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

template <class Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
Skew(const Eigen::MatrixBase<Derived> &vec) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
  return (Eigen::Matrix<typename Derived::Scalar, 3, 3>() << 0.0, -vec[2],
          vec[1], vec[2], 0.0, -vec[0], -vec[1], vec[0], 0.0)
      .finished();
}

struct Dynamics_quadcopter3d : Dynamics {
  // theta = 0  is quadcopter in horizontal position
  // 2.5 * 9.8 / 2 =

  struct Data {
    Eigen::Vector3d f_u;
    Eigen::Vector3d tau_u;
    Eigen::VectorXd xnext{13};
    Matrix34 Jx;
    Eigen::Matrix3d Ja;
  } data;

  double dt = .01;
  double g = 9.8;

  bool free_time;
  bool force_control = true;

  // Big dron
  // double m = 2.5;
  // const Eigen::Vector3d J =
  //     Eigen::Vector3d(1.2, 1.2, 1.2);
  // double l = .5;
  // const float arm_length = 0.3; // m
  // const float arm = 0.707106781 * arm_length;
  // const float t2t = 0.006;   // thrust-to-torque ratio

  // Parameters based on Bitcraze Crazyflie 2.0
  double m = 0.034;               // kg
  const float arm_length = 0.046; // m
  const float arm = 0.707106781 * arm_length;
  const float t2t = 0.006; // thrust-to-torque ratio
  const Eigen::Vector3d J_v =
      Eigen::Vector3d(16.571710e-6, 16.655602e-6, 29.261652e-6);

  const double u_nominal = m * g / 4.;
  double m_inv = 1. / m;
  const Eigen::Vector3d inverseJ_v = J_v.cwiseInverse();

  const Eigen::Matrix3d inverseJ_M = inverseJ_v.asDiagonal();
  const Eigen::Matrix3d J_M = J_v.asDiagonal();

  const Eigen::Matrix3d inverseJ_skew = Skew(inverseJ_v);
  const Eigen::Matrix3d J_skew = Skew(J_v);

  Eigen::Vector3d grav_v = Eigen::Vector3d(0, 0, -m *g);

  Eigen::Matrix4d B0;

  Matrix34 Fu_selection;
  Matrix34 Ftau_selection;

  Matrix34 Fu_selection_B0;
  Matrix34 Ftau_selection_B0;

  Dynamics_quadcopter3d(bool free_time = false);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

struct Dynamics_unicycle : Dynamics {

  double dt = .1;
  bool free_time;

  Dynamics_unicycle(bool free_time = false);

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

enum class CostTYPE {
  linear,
  least_squares,
};

struct Cost {
  size_t nx;
  size_t nu;
  size_t nr;
  std::string name;
  CostTYPE cost_type = CostTYPE::least_squares;

  Eigen::VectorXd zero_u;
  Eigen::MatrixXd zero_Ju;

  Cost(size_t nx, size_t nu, size_t nr)
      : nx(nx), nu(nu), nr(nr), zero_u(nu), zero_Ju(nr, nu) {
    zero_u.setZero();
    zero_Ju.setZero();
  }
  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {

    throw std::runtime_error(AT);
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {

    throw std::runtime_error(AT);
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

    throw std::runtime_error(AT);
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    throw std::runtime_error(AT);
  }
  virtual std::string get_name() const { return name; }
};

struct Quaternion_cost : Cost {

  double k_quat = 5.; 
  Quaternion_cost(size_t nx, size_t nu);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);
};

struct Contour_cost_alpha_x : Cost {

  double k = 1.5; // bigger than 0
  Contour_cost_alpha_x(size_t nx, size_t nu);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

};

struct Contour_cost_alpha_u : Cost {

  double k = 1.5; // bigger than 0
  Contour_cost_alpha_u(size_t nx, size_t nu);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);
};

void finite_diff_cost(ptr<Cost> cost, Eigen::Ref<Eigen::MatrixXd> Jx,
                      Eigen::Ref<Eigen::MatrixXd> Ju, const Eigen::VectorXd &x,
                      const Eigen::VectorXd &u, const int nr);

struct Contour_cost_x : Cost {

  ptr<Interpolator> path;
  double weight = 20.;
  bool use_finite_diff = false;

  double last_query;
  Eigen::VectorXd last_out;
  Eigen::VectorXd last_J;

  Contour_cost_x(size_t nx, size_t nu, ptr<Interpolator> path);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct Contour_cost : Cost {

  ptr<Interpolator> path;
  double ref_alpha = 1.;
  double weight_alpha = 2.;
  double weight_diff = 20.;
  double weight_contour = 1.;
  double weight_virtual_control = 2.;
  bool use_finite_diff = false;

  double last_query;
  Eigen::VectorXd last_out;
  Eigen::VectorXd last_J;

  Contour_cost(size_t nx, size_t nu, ptr<Interpolator> path);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct Col_cost : Cost {

  boost::shared_ptr<CollisionChecker> cl;
  double margin = .03;
  double last_raw_d = 0;
  Eigen::VectorXd last_x;
  Eigen::VectorXd last_grad;

  // TODO: check that the sec_factor is actually save in a test
  double sec_factor = .1;

  // what about a log barrier function? -- maybe I get better gradients

  double faraway_zero_gradient_bound = 1.1 * margin;
  // returns 0 gradient if distance is > than THIS.
  double epsilon = 1e-3;            // finite diff
  std::vector<bool> non_zero_flags; // if not nullptr, non_zero_flags[i] = True
                                    // means compute gradient, non_zero_flags[i]
                                    // = false is no compute gradient

  size_t nx_effective;

  Col_cost(size_t nx, size_t nu, size_t nr,
           boost::shared_ptr<CollisionChecker> cl);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct Control_cost : Cost {

  Eigen::VectorXd u_weight;
  Eigen::VectorXd u_ref;

  Control_cost(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &u_weight,
               const Eigen::VectorXd &u_ref);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

// x - ub  <= 0

struct State_bounds : Cost {
  // weight * ( x - ub ) <= 0
  //
  // NOTE:
  // you can implement lower bounds
  // weight = -w
  // ub = lb
  // DEMO:
  // -w ( x - lb ) <= 0
  // w ( x - lb ) >= 0
  // w x >= w lb

  Eigen::VectorXd ub;
  Eigen::VectorXd weight;

  State_bounds(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &ub,
               const Eigen::VectorXd &weight);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};
struct State_cost : Cost {

  Eigen::VectorXd x_weight;
  Eigen::VectorXd ref;

  State_cost(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &x_weight,
             const Eigen::VectorXd &ref);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct All_cost : Cost {

  std::vector<boost::shared_ptr<Cost>> costs;

  All_cost(size_t nx, size_t nu, size_t nr,
           const std::vector<boost::shared_ptr<Cost>> &costs);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

size_t
get_total_num_features(const std::vector<boost::shared_ptr<Cost>> &features);

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
  std::vector<boost::shared_ptr<Cost>> features;
  size_t nx;
  size_t nu;
  size_t nr;

  Eigen::MatrixXd Jx;
  Eigen::MatrixXd Ju;
  // Eigen::VectorXd zero_u;
  // Eigen::MatrixXd zero_Ju;

  ActionModelQ(ptr<Dynamics> dynamics, const std::vector<ptr<Cost>> &features);

  virtual ~ActionModelQ();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);

  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data);

  virtual void print(std::ostream &os) const;
};

const Eigen::IOFormat FMT(6, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

extern double accumulated_time;

enum class SOLVER {
  traj_opt = 0,
  traj_opt_free_time = 1,
  traj_opt_smooth_then_free_time = 2,
  mpc = 3,
  mpcc = 4,
  mpcc2 = 5,
  traj_opt_mpcc = 6,
  mpc_nobound_mpcc = 7,
  mpcc_linear = 8,
  time_search_traj_opt = 9,
  mpc_adaptative = 10,
  traj_opt_free_time_proxi = 1,
  none = 11,
};

void PrintVariableMap(const boost::program_options::variables_map &vm,
                      std::ostream &out);

template <typename Derived>
boost::shared_ptr<crocoddyl::ActionModelAbstract>
to_am_base(boost::shared_ptr<Derived> am) {
  return boost::static_pointer_cast<crocoddyl::ActionModelAbstract>(am);
};

void print_data(boost::shared_ptr<ActionDataAbstractTpl<double>> data);

void check_dyn(boost::shared_ptr<Dynamics> dyn, double eps);

struct Generate_params {
  bool free_time = false;
  std::string name;
  size_t N;
  Eigen::VectorXd goal;
  Eigen::VectorXd start;
  ptr<CollisionChecker> cl;
  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> states_weights;
  std::vector<Eigen::VectorXd> actions;
  bool contour_control = false;
  ptr<Interpolator> interpolator = nullptr;
  double max_alpha = 100.;
  bool linear_contour = true;
  bool goal_cost = true;
  bool collisions = true;

  void print(std::ostream &out) const;
};

double max_rollout_error(ptr<Dynamics> dyn,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us);

bool check_feas(ptr<Cost> feat_col, const std::vector<Eigen::VectorXd> &xs,
                const std::vector<Eigen::VectorXd> &us,
                const Eigen::VectorXd &goal);

ptr<crocoddyl::ShootingProblem>
generate_problem(const Generate_params &gen_args, size_t &nx, size_t &nu);

bool check_dynamics(const std::vector<Eigen::VectorXd> &xs_out,
                    const std::vector<Eigen::VectorXd> &us_out,
                    ptr<Dynamics> dyn);

struct File_parser_inout {
  std::string problem_name;
  std::string init_guess;
  std::string env_file;
  bool new_format = false;
  std::string name;
  ptr<CollisionChecker> cl;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  Eigen::VectorXd start;
  Eigen::VectorXd goal;

  void add_options(po::options_description &desc);
  void print(std::ostream &out);
  void read_from_yaml(YAML::Node &node);
  void read_from_yaml(const char *file);
};

Eigen::VectorXd enforce_bounds(const Eigen::VectorXd &us,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub);

void read_from_file(File_parser_inout &inout);

void convert_traj_with_variable_time(const std::vector<Eigen::VectorXd> &xs,
                                     const std::vector<Eigen::VectorXd> &us,
                                     std::vector<Eigen::VectorXd> &xs_out,
                                     std::vector<Eigen::VectorXd> &us_out,
                                     const double &dt);

struct Result_opti {
  bool feasible = false;
  double cost = -1;
  std::string name;
  std::vector<Eigen::VectorXd> xs_out;
  std::vector<Eigen::VectorXd> us_out;

  void write_yaml(std::ostream &out);

  void write_yaml_db(std::ostream &out);
};

int inside_bounds(int i, int lb, int ub);

auto smooth_traj(const std::vector<Eigen::VectorXd> &us_init);

void solve_with_custom_solver(File_parser_inout &file_inout,
                              Result_opti &opti_out);

void compound_solvers(File_parser_inout file_inout, Result_opti &opti_out);

;
