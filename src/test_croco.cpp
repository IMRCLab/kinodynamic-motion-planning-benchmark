#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

// #include <boost/test/unit_test_suite.hpp>
// #define BOOST_TEST_DYN_LINK
// #include <boost/test/unit_test.hpp>

// see
// https://www.boost.org/doc/libs/1_81_0/libs/test/doc/html/boost_test/usage_variants.html
// #define BOOST_TEST_MODULE test module name

#define BOOST_TEST_MODULE test module name
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

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

#define NAMEOF(variable) #variable

// save data without the cluster stuff

#include <filesystem>
#include <random>
#include <regex>
#include <type_traits>

#include <filesystem>
#include <regex>

using namespace crocoddyl;
namespace po = boost::program_options;

// Eigen::Vector3d goal(1.9, .3, 0);
static double collision_weight = 100.;
// issue with derivatives...

const Eigen::IOFormat FMT(6, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define CHECK_EQ(A, B, msg)                                                    \
  if (!(A == B)) {                                                             \
    std::cout << "CHECK_EQ failed: '" << #A << "'=" << A << " '" << #B         \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_GEQ(A, B, msg)                                                   \
  if (!(A >= B)) {                                                             \
    std::cout << "CHECK_GEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_SEQ(A, B, msg)                                                   \
  if (!(A <= B)) {                                                             \
    std::cout << "CHECK_SEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

inline void PrintVariableMap(const boost::program_options::variables_map &vm,
                             std::ostream &out) {
  for (po::variables_map::const_iterator it = vm.cbegin(); it != vm.cend();
       it++) {
    out << "> " << it->first;
    if (((boost::any)it->second.value()).empty()) {
      out << "(empty)";
    }
    if (vm[it->first].defaulted() || it->second.defaulted()) {
      out << "(default)";
    }
    out << "=";

    bool is_char;
    try {
      boost::any_cast<const char *>(it->second.value());
      is_char = true;
    } catch (const boost::bad_any_cast &) {
      is_char = false;
    }
    bool is_str;
    try {
      boost::any_cast<std::string>(it->second.value());
      is_str = true;
    } catch (const boost::bad_any_cast &) {
      is_str = false;
    }

    auto &type = ((boost::any)it->second.value()).type();

    if (type == typeid(int)) {
      out << vm[it->first].as<int>() << std::endl;
    } else if (type == typeid(bool)) {
      out << vm[it->first].as<bool>() << std::endl;
    } else if (type == typeid(double)) {
      out << vm[it->first].as<double>() << std::endl;
    } else if (is_char) {
      out << vm[it->first].as<const char *>() << std::endl;
    } else if (is_str) {
      std::string temp = vm[it->first].as<std::string>();
      if (temp.size()) {
        out << temp << std::endl;
      } else {
        out << "true" << std::endl;
      }
    } else { // Assumes that the only remainder is vector<string>
      try {
        std::vector<std::string> vect =
            vm[it->first].as<std::vector<std::string>>();
        uint i = 0;
        for (std::vector<std::string>::iterator oit = vect.begin();
             oit != vect.end(); oit++, ++i) {
          out << "\r> " << it->first << "[" << i << "]=" << (*oit) << std::endl;
        }
      } catch (const boost::bad_any_cast &) {
        out << "UnknownType(" << ((boost::any)it->second.value()).type().name()
            << ")" << std::endl;
        assert(false);
      }
    }
  }
};

template <class T> using ptr = boost::shared_ptr<T>;

template <typename T, typename... Args> auto mk(Args &&...args) {
  return boost::make_shared<T>(std::forward<Args>(args)...);
}

void linearInterpolation(const Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &x, double t_query,
                         Eigen::Ref<Eigen::VectorXd> out,
                         Eigen::Ref<Eigen::VectorXd> Jx) {

  double num_tolerance = 1e-8;
  CHECK_GEQ(t_query + num_tolerance, times.head(1)(0), AT);

  if (t_query < times(0)) {
    std::cout << "WARNING EXTRAPOLATION" << AT << std::endl;
    // TODO: add extrapolation with the other side.
    throw -1;
  }

  size_t index = 0;

  // CHECK_SEQ(t_query, times.tail(1)(0) + num_tolerance, AT);
  if (t_query >= times(times.size() - 1)) {
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
      for (size_t i = 0; i < times.size(); i++) {
        if (t_query < times(i)) {
          index2 = i;
          break;
        }
      }
      CHECK_EQ(index, index2, AT);
    }
  }

  // std::cout << "index is " << index << std::endl;
  // std::cout << "size " << times.size() << std::endl;

  double factor =
      (t_query - times(index - 1)) / (times(index) - times(index - 1));
  // std::cout << "factor " << factor << std::endl;

  out = x.at(index - 1) + factor * (x.at(index) - x.at(index - 1));
  Jx = (x.at(index) - x.at(index - 1)) / (times(index) - times(index - 1));
}

struct Interpolator {

  Eigen::VectorXd times;
  std::vector<Eigen::VectorXd> x;

  // TODO
  Eigen::VectorXd last_query;
  Eigen::VectorXd last_f;
  Eigen::MatrixXd last_J;

  Interpolator(const Eigen::VectorXd &times,
               const std::vector<Eigen::VectorXd> &x)
      : times(times), x(x) {

    CHECK_EQ(times.size(), x.size(), AT);
  }

  void interpolate(double t_query, Eigen::Ref<Eigen::VectorXd> out,
                   Eigen::Ref<Eigen::VectorXd> J) {
    linearInterpolation(times, x, t_query, out, J);
  }
};

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
  Dynamics_contour(ptr<Dynamics> dyn, bool accumulate)
      : dyn(dyn), accumulate(accumulate) {
    nx = dyn->nx + 1;
    nu = dyn->nu + 1;
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

    dyn->calc(xnext.head(dyn->nx), x.head(dyn->nx), u.head(dyn->nu));

    if (accumulate)
      xnext(nx - 1) = x(nx - 1) + u(nu - 1); // linear model
    else
      xnext(nx - 1) = u(nu - 1); // linear model
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

    dyn->calcDiff(Fx.block(0, 0, dyn->nx, dyn->nx),
                  Fu.block(0, 0, dyn->nx, dyn->nu), x.head(dyn->nx),
                  u.head(dyn->nu));

    if (accumulate) {
      Fx(nx - 1, nx - 1) = 1.0;
      Fu(nx - 1, nu - 1) = 1.0;
    } else {
      Fu(nx - 1, nu - 1) = 1.0;
    }
  }
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
  std::string name;

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
  virtual std::string get_name() const { return name; }
};

struct Contour_cost : Cost {

  ptr<Interpolator> path;
  double ref_alpha = 1.;
  double weight_alpha = 2.;
  double weight_diff = 20.;
  double weight_contour = 1.;
  double weight_virtual_control = 2.;
  bool use_finite_diff = false;

  Contour_cost(size_t nx, size_t nu, ptr<Interpolator> path)
      : Cost(nx, nu, nx + 1), path(path) {
    name = "Contour";
  }
  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    assert(static_cast<std::size_t>(x.size()) == nx);

    double alpha = x(nx - 1);

    // x in contour
    Eigen::VectorXd tmp(nx - 1);
    Eigen::VectorXd J(nx - 1);
    path->interpolate(alpha, tmp, J);

    r.head(nx - 1) = weight_contour * weight_diff * (tmp - x.head(nx - 1));
    r(nx - 1) = weight_contour * weight_alpha * (alpha - ref_alpha);
    r(nx) = weight_virtual_control * u(nu - 1);

    // std::cout << "calc in contour " << r.transpose() << std::endl;
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

    // lets use finite diff
    if (use_finite_diff) {
      Eigen::VectorXd r_ref(nr);
      calc(r_ref, x, u);

      Ju.setZero();

      double eps = 1e-5;
      for (size_t i = 0; i < nu; i++) {
        Eigen::MatrixXd ue;
        ue = u;
        ue(i) += eps;
        Eigen::VectorXd r_e(nr);
        r_e.setZero();
        calc(r_e, x, ue);
        auto df = (r_e - r_ref) / eps;
        Ju.col(i) = df;
      }

      Jx.setZero();
      for (size_t i = 0; i < nx; i++) {
        Eigen::MatrixXd xe;
        xe = x;
        xe(i) += eps;
        Eigen::VectorXd r_e(nr);
        r_e.setZero();
        calc(r_e, xe, u);
        auto df = (r_e - r_ref) / eps;
        Jx.col(i) = df;
      }
    }

    else {
      Eigen::VectorXd tmp(nx - 1);
      Eigen::VectorXd J(nx - 1);
      path->interpolate(x(nx - 1), tmp, J);
      Jx.block(0, 0, nx - 1, nx - 1).diagonal() =
          -weight_contour * weight_diff * Eigen::VectorXd::Ones(nx - 1);
      Jx.block(0, nx - 1, nx - 1, 1) = weight_contour * weight_diff * J;
      Jx(nx - 1, nx - 1) = weight_contour * weight_alpha;
      Ju(nx, nu - 1) = weight_virtual_control;
    }
  };

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    auto Ju = Eigen::MatrixXd(nr, nu);
    auto u = Eigen::VectorXd(nu);
    calcDiff(Jx, Ju, x, u);
  }
};

struct Col_cost : Cost {

  boost::shared_ptr<CollisionChecker> cl;
  double margin = .03;
  double last_raw_d = 0;
  Eigen::VectorXd last_x;
  Eigen::VectorXd last_grad;

  // TODO: check that the sec_factor is save
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
           boost::shared_ptr<CollisionChecker> cl)
      : Cost(nx, nu, nr), cl(cl) {
    last_x = Eigen::VectorXd::Zero(nx);
    name = "collision";
    nx_effective = nx;
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);

    // std::vector<double> query{x.data(), x.data() + x.size()};
    std::vector<double> query{x.data(), x.data() + nx_effective};

    double raw_d;

    bool check_one = (x - last_x).squaredNorm() < 1e-8;
    bool check_two = (last_raw_d - margin) > 0 &&
                     (x - last_x).norm() < sec_factor * (last_raw_d - margin);

    std::cout << "checking collisions " << std::endl;

    if (check_one || check_two) {
      raw_d = last_raw_d;
    } else {
      raw_d = std::get<0>(cl->distance(query));
      last_x = x;
      last_raw_d = raw_d;
    }
    double d = collision_weight * (raw_d - margin);
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

    // std::vector<double> query{x.data(), x.data() + x.size()};
    std::vector<double> query{x.data(), x.data() + nx_effective};
    double raw_d;
    double d;
    Eigen::VectorXd v(nx);
    // CONSERVATIVE
    bool check_one =
        (x - last_x).squaredNorm() < 1e-8 && (last_raw_d - margin) > 0;
    bool check_two = (last_raw_d - margin) > 0 &&
                     (x - last_x).norm() < sec_factor * (last_raw_d - margin);

    if (check_one || check_two) {
      Jx.setZero();
    } else {
      auto out = cl->distanceWithFDiffGradient(
          query, faraway_zero_gradient_bound, epsilon,
          non_zero_flags.size() ? &non_zero_flags : nullptr);
      raw_d = std::get<0>(out);
      last_x = x;
      last_raw_d = raw_d;
      d = collision_weight * (raw_d - margin);
      auto grad = std::get<1>(out);
      v = collision_weight * Eigen::VectorXd::Map(grad.data(), grad.size());
      if (d <= 0) {
        Jx.block(0, 0, 1, nx_effective) = v.transpose();
      } else {
        Jx.setZero();
      }
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
  Eigen::VectorXd u_ref;

  Control_cost(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &u_weight,
               const Eigen::VectorXd &u_ref)
      : Cost(nx, nu, nr), u_weight(u_weight), u_ref(u_ref) {
    CHECK_EQ(u_weight.size(), nu, AT);
    CHECK_EQ(u_ref.size(), nu, AT);
    CHECK_EQ(nu, nr, AT);
    name = "control";
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
    // check that r
    assert(static_cast<std::size_t>(r.size()) == nr);
    assert(static_cast<std::size_t>(x.size()) == nx);
    assert(static_cast<std::size_t>(u.size()) == nu);
    r = (u - u_ref).cwiseProduct(u_weight);
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
      : Cost(nx, nu, nr), x_weight(x_weight), ref(ref) {
    name = "state";
  }

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
  bool verbose = false;
  if (verbose) {
    std::cout << "Analytical " << std::endl;
    std::cout << "Fx\n" << Fx << std::endl;
    std::cout << "Fu\n" << Fu << std::endl;
    std::cout << "Finite Diff " << std::endl;
    std::cout << "Fx\n" << FxD << std::endl;
    std::cout << "Fu\n" << FuD << std::endl;
  }

  BOOST_TEST((Fx - FxD).cwiseAbs().maxCoeff() < 10 * eps);
  BOOST_TEST((Fu - FuD).cwiseAbs().maxCoeff() < 10 * eps);
}

struct Opts {
  bool free_time = false;
  bool control_bounds = false;
  std::string name;
  size_t N;
  bool regularize_wrt_init_guess = false;
  bool use_finite_diff = false;
  Eigen::VectorXd goal;
  Eigen::VectorXd start;
  ptr<CollisionChecker> cl;
  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> actions;
  bool contour_control = false;
  ptr<Interpolator> interpolator = nullptr;
  double ref_alpha = 1.;
  double max_alpha = 100;
  double cost_alpha_multi = 1.;
  bool only_contour_last = true;
  Eigen::VectorXd alpha_refs;
  Eigen::VectorXd cost_alpha_multis;
  // Eigen::VectorXd u_lb;
  // Eigen::VectorXd u_ub;

  void print(std::ostream &out) const {
    auto pre = "";
    auto after = ": ";
    out << pre << "free_time" << after << free_time << std::endl;
    out << pre << "control_bounds" << after << control_bounds << std::endl;
    out << pre << "name" << after << name << std::endl;
    out << pre << "N" << after << N << std::endl;
    out << pre << "regularize_wrt_init_guess" << after
        << regularize_wrt_init_guess << std::endl;
    out << pre << "use_finite_diff" << after << use_finite_diff << std::endl;
    out << pre << "goal" << after << goal.transpose() << std::endl;
    out << pre << "start" << after << start.transpose() << std::endl;
    out << pre << "cl" << after << cl << std::endl;
    // out << pre << "states"<< after << states << std::endl;
    // out << pre << "actions"<< after << actions << std::endl;
    out << pre << "contour_control" << after << contour_control << std::endl;
    out << pre << "ref_alpha" << after << ref_alpha << std::endl;
    out << pre << "max_alpha" << after << max_alpha << std::endl;
    out << pre << "cost_alpha_multi" << after << cost_alpha_multi << std::endl;
    out << pre << "only_contour_last" << after << only_contour_last
        << std::endl;
    out << pre << "alpha_refs" << after << alpha_refs.transpose() << std::endl;
    out << pre << "cost_alpha_multis" << after << cost_alpha_multis.transpose()
        << std::endl;
  }
};

bool check_feas(ptr<Cost> feat_col, const std::vector<Eigen::VectorXd> &xs,
                const std::vector<Eigen::VectorXd> &us,
                const Eigen::VectorXd &goal) {

  double accumulated_c = 0;
  for (auto &x : xs) {
    Eigen::VectorXd out(1);
    feat_col->calc(out, x);
    accumulated_c += std::abs(out(0));
  }
  double dist_to_goal = (xs.back() - goal).norm();

  std::cout << "CROCO DONE " << std::endl;
  std::cout << "accumulated_c is " << accumulated_c << std::endl;
  std::cout << "distance to goal " << dist_to_goal << std::endl;

  double threshold_feas = .1;

  // Dist to goal: 1 cm is OK
  // Accumulated_c: 1 cm  ACCUM is OK
  bool feasible = 10 * (accumulated_c / collision_weight) + 10 * dist_to_goal <
                  threshold_feas;
  return feasible;
};

auto generate_problem(const Opts &opts, size_t &nx, size_t &nu) {

  std::cout << "**\nGenerating problem\n**\n" << std::endl;
  opts.print(std::cout);

  ptr<Cost> feats_terminal;
  ptr<crocoddyl::ActionModelAbstract> am_terminal;
  ptr<Dynamics> dyn;

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> amq_runs;
  Eigen::VectorXd goal_v = opts.goal;

  if (opts.regularize_wrt_init_guess && opts.contour_control) {
    CHECK_EQ(true, false, AT);
  }

  if (opts.name == "unicycle_first_order_0") {

    dyn = mk<Dynamics_unicycle>(opts.free_time);

    if (opts.contour_control) {
      dyn = mk<Dynamics_contour>(dyn, !opts.only_contour_last);
      // dyn = mk<Dynamics_Contour>(dyn, false);
    }

    nx = dyn->nx;
    nu = dyn->nu;

    ptr<Cost> control_feature;

    if (opts.free_time && !opts.contour_control)
      control_feature = mk<Control_cost>(
          nx, nu, nu, Eigen::Vector3d(.2, .2, 1.), Eigen::Vector3d(0., 0., .5));
    else if (!opts.free_time && !opts.contour_control)
      control_feature = mk<Control_cost>(nx, nu, nu, Eigen::Vector2d(.5, .5),
                                         Eigen::Vector2d::Zero());
    else if (!opts.free_time && opts.contour_control) {
      control_feature = mk<Control_cost>(
          nx, nu, nu, Eigen::Vector3d(.5, .5, .1), Eigen::Vector3d::Zero());
    } else {
      CHECK_EQ(true, false, AT);
    }

    for (size_t t = 0; t < opts.N; t++) {

      ptr<Cost> feats_run;
      ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, opts.cl);

      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true, true, true};

      if (opts.contour_control)
        boost::static_pointer_cast<Col_cost>(cl_feature)->nx_effective = nx - 1;

      if (opts.regularize_wrt_init_guess) {
        ptr<Cost> state_feature = mk<State_cost>(
            nx, nu, nx, Eigen::Vector3d(1., 1., 1), opts.states.at(t));
        feats_run = mk<All_cost>(
            nx, nu, cl_feature->nr + control_feature->nr + state_feature->nr,
            std::vector<ptr<Cost>>{cl_feature, state_feature, control_feature});
      } else if (opts.contour_control) {
        if (opts.only_contour_last) {
          feats_run =
              mk<All_cost>(nx, nu, cl_feature->nr + control_feature->nr,
                           std::vector<ptr<Cost>>{cl_feature, control_feature});
        }

        else {
          std::cout << "adding Contour to all " << std::endl;
          ptr<Contour_cost> Contour =
              mk<Contour_cost>(nx, nu, opts.interpolator);

          std::cout << "at time t, ref alpha is " << t << " "
                    << opts.alpha_refs(t) << std::endl;
          Contour->ref_alpha = opts.alpha_refs(t);
          // Contour->weight_alpha *= opts.cost_alpha_multis(t);
          Contour->weight_contour *= opts.cost_alpha_multis(t);

          feats_run = mk<All_cost>(
              nx, nu, cl_feature->nr + control_feature->nr + Contour->nr,
              std::vector<ptr<Cost>>{cl_feature, control_feature, Contour});
        }
        // std::cout << "WARNING "
        //           << "i have erased the collisions for now" << std::endl;
        // mk<All_cost>(nx, nu, control_feature->nr,
        //              std::vector<ptr<Cost>>{control_feature});
      } else {
        feats_run =
            mk<All_cost>(nx, nu, control_feature->nr + cl_feature->nr,
                         std::vector<ptr<Cost>>{cl_feature, control_feature});
        // feats_run = mk<All_cost>(nx, nu, control_feature->nr,
        //                          std::vector<ptr<Cost>>{control_feature});
      }

      auto am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));

      if (opts.control_bounds) {

        if (opts.free_time) {
          am_run->set_u_lb(Eigen::Vector3d(-.5, -.5, .4));
          am_run->set_u_ub(Eigen::Vector3d(.5, .5, 1.5));
        } else if (opts.contour_control) {
          am_run->set_u_lb(Eigen::Vector3d(-.5, -.5, 0.));
          am_run->set_u_ub(Eigen::Vector3d(.5, .5, opts.max_alpha));
        } else {
          // am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
          // am_run->set_u_ub(Eigen::Vector2d(.5, .5));
          // am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
          // am_run->set_u_ub(Eigen::Vector2d(.5, .5));

          am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
          am_run->set_u_ub(Eigen::Vector2d(.5, .5));
        }
      }

      amq_runs.push_back(am_run);
    }

    ptr<Cost> state_feature =
        mk<State_cost>(nx, nu, nx, 200. * Eigen::Vector3d::Ones(), opts.goal);

    if (opts.contour_control) {
      ptr<Contour_cost> Contour = mk<Contour_cost>(nx, nu, opts.interpolator);
      Contour->ref_alpha = opts.alpha_refs(opts.N);
      Contour->weight_contour *= opts.cost_alpha_multis(opts.N);

      feats_terminal =
          mk<All_cost>(nx, nu, Contour->nr, std::vector<ptr<Cost>>{Contour});

      am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

    } else {
      feats_terminal = mk<All_cost>(nx, nu, state_feature->nr,
                                    std::vector<ptr<Cost>>{state_feature});
      am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));
    }

  } else if (opts.name == "unicycle_second_order_0") {

#if 0 

    dyn = mk<Dynamics_unicycle2>(opts.free_time);
    nx = dyn->nx;
    nu = dyn->nu;
    ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, opts.cl);
    boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
        true, true, true, false, false};

    ptr<Cost> control_feature = mk<Control_cost>(
        nx, nu, nu, Eigen::Vector2d(.5, .5), Eigen::Vector2d::Zero());

    ptr<Cost> state_feature = mk<State_cost>(
        nx, nu, nx, 100. * Eigen::VectorXd::Ones(nx),
        Eigen::VectorXd::Map(opts.goal.data(), opts.goal.size()));

    ptr<Cost> feats_run =
        mk<All_cost>(nx, nu, cl_feature->nr + control_feature->nr,
                     std::vector<ptr<Cost>>{cl_feature, control_feature});

    feats_terminal = mk<All_cost>(nx, nu, state_feature->nr,
                                  std::vector<ptr<Cost>>{state_feature});

    am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));

    if (opts.control_bounds) {
      am_run->set_u_lb(Eigen::Vector2d(-.25, -.25));
      am_run->set_u_ub(Eigen::Vector2d(.25, .25));
    }
    am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

    // TODO
    amq_runs = std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>(
        opts.N, am_run);

    // TODO: option to regularize w.r.t. intial guess.
  //
  //
  //

#endif
  } else {
    throw -1;
  }

  if (opts.use_finite_diff) {
    std::cout << "using finite diff" << std::endl;

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
        amq_runs_diff(amq_runs.size());

    std::transform(
        amq_runs.begin(), amq_runs.end(), amq_runs_diff.begin(),
        [&](const auto &am_run) {
          auto am_rundiff = mk<crocoddyl::ActionModelNumDiff>(am_run, true);
          boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_rundiff)
              ->set_disturbance(1e-4);
          if (opts.control_bounds) {
            am_rundiff->set_u_lb(am_run->get_u_lb());
            am_rundiff->set_u_ub(am_run->get_u_ub());
          }
          return am_rundiff;
        });

    amq_runs = amq_runs_diff;

    auto am_terminal_diff =
        mk<crocoddyl::ActionModelNumDiff>(am_terminal, true);
    boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_terminal_diff)
        ->set_disturbance(1e-4);
    am_terminal = am_terminal_diff;
  }

  ptr<crocoddyl::ShootingProblem> problem =
      mk<crocoddyl::ShootingProblem>(opts.start, amq_runs, am_terminal);

  return problem;
};

enum class SOLVER {

  traj_opt = 0,
  traj_opt_free_time = 1,
  traj_opt_smooth_then_free_time = 2,
  mpc = 3,
  mpcc = 4,
  mpcc2 = 5,
  traj_opt_mpcc = 6,
  none = 7,
};

const char *SOLVER_txt[] = {"traj_opt",
                            "traj_opt_free_time",
                            "traj_opt_smooth_then_free_time",
                            "mpc",
                            "mpcc",
                            "mpcc2",
                            "traj_opt_mpcc",
                            "none"};

bool check_dynamics(const std::vector<Eigen::VectorXd> &xs_out,
                    const std::vector<Eigen::VectorXd> &us_out,
                    ptr<Dynamics> dyn) {

  double tolerance = 1e-4;
  CHECK_EQ(xs_out.size(), us_out.size() + 1, AT);

  size_t N = us_out.size();
  bool feasible = true;

  for (size_t i = 0; i < N; i++) {
    Eigen::VectorXd xnext(dyn->nx);

    auto &x = xs_out.at(i);
    auto &u = us_out.at(i);
    dyn->calc(xnext, x, u);

    if ((xnext - xs_out.at(i + 1)).norm() > tolerance) {
      std::cout << "Infeasible at " << i << std::endl;
      std::cout << xnext.transpose() << " " << xs_out.at(i + 1).transpose()
                << std::endl;
      feasible = false;
      break;
    }
  }

  return feasible;
}

struct File_parser_inout {
  std::string init_guess;
  std::string env_file;
  bool new_format = false;
  std::string name;
  ptr<CollisionChecker> cl;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  Eigen::VectorXd start;
  Eigen::VectorXd goal;

  void add_options(po::options_description &desc) {
    desc.add_options()("env", po::value<std::string>(&env_file)->required())(
        "waypoints", po::value<std::string>(&init_guess)->required())(
        "new_format", po::value<bool>(&new_format)->default_value(new_format));
  }
};

struct Opti_params {

  bool CALLBACKS = true;
  bool use_finite_diff = false;
  bool use_warmstart = true;
  bool free_time = false;
  bool repair_init_guess = true;
  bool regularize_wrt_init_guess = false;
  bool check_time = false;
  bool control_bounds = true;
  bool adaptative_goal_mpc = false;
  int solver_int = 0;

  double th_stop = 1e-2;
  double init_reg = 1e2;
  double th_acceptnegstep = .3;
  double noise_level = 1e-3; // factor on top of [-1., 1.]
  double alpha_rate = 1.5;   // I try to go X times faster

  int max_iter = 50;
  int num_steps_to_optimize = 20;
  int num_steps_to_move = 10;
  size_t max_mpc_iterations = 50;

  void add_options(po::options_description &desc) {

    desc.add_options()("free_time",
                       po::value<bool>(&free_time)->default_value(free_time))(
        "control_bounds",
        po::value<bool>(&control_bounds)->default_value(control_bounds))(
        "max_iter", po::value<int>(&max_iter)->default_value(max_iter))(
        "step_opt", po::value<int>(&num_steps_to_optimize)
                        ->default_value(num_steps_to_optimize))(
        "step_move",
        po::value<int>(&num_steps_to_move)->default_value(num_steps_to_move))(
        "solver", po::value<int>(&solver_int)->default_value(solver_int))(
        "use_warmstart",
        po::value<bool>(&use_warmstart)->default_value(use_warmstart))(
        "use_fdiff",
        po::value<bool>(&use_finite_diff)->default_value(use_finite_diff))(
        "alpha_rate",
        po::value<double>(&alpha_rate)->default_value(alpha_rate))(
        "reg", po::value<bool>(&regularize_wrt_init_guess)
                   ->default_value(regularize_wrt_init_guess));
  }
};

Eigen::VectorXd enforce_bounds(const Eigen::VectorXd &us,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub) {

  return us.cwiseMax(lb).cwiseMin(ub);
}

void read_from_file(File_parser_inout &inout) {

  double dt = .1;

  std::cout << "Warning, dt is hardcoded to: " << dt << std::endl;

  YAML::Node init = YAML::LoadFile(inout.init_guess);
  YAML::Node env = YAML::LoadFile(inout.env_file);

  // load the collision checker
  std::cout << "loading collision checker... " << std::endl;
  inout.cl = mk<CollisionChecker>();
  inout.cl->load(inout.env_file);
  std::cout << "DONE" << std::endl;

  inout.name = env["robots"][0]["type"].as<std::string>();

  std::vector<std::vector<double>> states;
  // std::vector<Eigen::VectorXd> xs_init;
  // std::vector<Eigen::VectorXd> us_init;

  size_t N;
  std::vector<std::vector<double>> actions;

  if (!inout.new_format) {
    for (const auto &state : init["result"][0]["states"]) {
      std::vector<double> p;
      for (const auto &elem : state) {
        p.push_back(elem.as<double>());
      }
      states.push_back(p);
    }

    N = states.size() - 1;

    for (const auto &state : init["result"][0]["actions"]) {
      std::vector<double> p;
      for (const auto &elem : state) {
        p.push_back(elem.as<double>());
      }
      actions.push_back(p);
    }

    inout.xs.resize(states.size());
    inout.us.resize(actions.size());

    std::transform(
        states.begin(), states.end(), inout.xs.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

    std::transform(
        actions.begin(), actions.end(), inout.us.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

  } else {

    std::cout << "Reading results in the new format " << std::endl;
    for (const auto &state : init["result2"][0]["states"]) {
      std::vector<double> p;
      for (const auto &elem : state) {
        p.push_back(elem.as<double>());
      }
      states.push_back(p);
    }

    N = states.size() - 1;

    for (const auto &action : init["result2"][0]["actions"]) {
      std::vector<double> p;
      for (const auto &elem : action) {
        p.push_back(elem.as<double>());
      }
      actions.push_back(p);
    }

    std::vector<double> _times;

    for (const auto &time : init["result2"][0]["times"]) {
      _times.push_back(time.as<double>());
    }

    // 0 ... 3.5
    // dt is 1.
    // 0 1 2 3 4

    // we use floor in the time to be more agressive
    std::cout << "DT hardcoded to .1 " << std::endl;

    double total_time = _times.back();

    int num_time_steps = std::ceil(total_time / dt);

    Eigen::VectorXd times = Eigen::VectorXd::Map(_times.data(), _times.size());

    std::vector<Eigen::VectorXd> _xs_init(states.size());
    std::vector<Eigen::VectorXd> _us_init(actions.size());

    std::vector<Eigen::VectorXd> xs_init_new;
    std::vector<Eigen::VectorXd> us_init_new;

    int nx = 3;
    int nu = 2;

    std::transform(
        states.begin(), states.end(), _xs_init.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

    std::transform(
        actions.begin(), actions.end(), _us_init.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

    auto ts =
        Eigen::VectorXd::LinSpaced(num_time_steps + 1, 0, num_time_steps * dt);

    std::cout << "taking samples at " << ts.transpose() << std::endl;

    for (size_t ti = 0; ti < num_time_steps + 1; ti++) {
      Eigen::VectorXd xout(nx);
      Eigen::VectorXd Jout(nx);

      if (ts(ti) > times.tail(1)(0))
        xout = _xs_init.back();
      else
        linearInterpolation(times, _xs_init, ts(ti), xout, Jout);
      xs_init_new.push_back(xout);
    }

    auto times_u = times.head(times.size() - 1);
    for (size_t ti = 0; ti < num_time_steps; ti++) {
      Eigen::VectorXd uout(nu);
      Eigen::VectorXd Jout(nu);
      if (ts(ti) > times_u.tail(1)(0))
        uout = _us_init.back();
      else
        linearInterpolation(times_u, _us_init, ts(ti), uout, Jout);
      us_init_new.push_back(uout);
    }

    N = num_time_steps;

    std::cout << "N: " << N << std::endl;
    std::cout << "us:  " << us_init_new.size() << std::endl;
    std::cout << "xs: " << xs_init_new.size() << std::endl;

    inout.xs = xs_init_new;
    inout.us = us_init_new;

    std::ofstream debug_file("debug.txt");

    for (auto &x : inout.xs) {
      debug_file << x.format(FMT) << std::endl;
    }
    debug_file << "---" << std::endl;

    for (auto &u : inout.us) {
      debug_file << u.format(FMT) << std::endl;
    }
  }

  std::vector<double> _start, _goal;
  for (const auto &e : env["robots"][0]["start"]) {
    _start.push_back(e.as<double>());
  }

  for (const auto &e : env["robots"][0]["goal"]) {
    _goal.push_back(e.as<double>());
  }

  inout.start = Eigen::VectorXd::Map(_start.data(), _start.size());
  inout.goal = Eigen::VectorXd::Map(_goal.data(), _goal.size());

  bool verbose = false;
  if (verbose) {

    std::cout << "states " << std::endl;
    for (auto &x : inout.xs)
      std::cout << x.format(FMT) << std::endl;

    std::cout << "actions " << std::endl;
    for (auto &u : inout.us)
      std::cout << u.format(FMT) << std::endl;
  }
}

struct Result_opti {
  bool feasible = false;
  double cost = -1;
  std::vector<Eigen::VectorXd> xs_out;
  std::vector<Eigen::VectorXd> us_out;
};

void solve_with_custom_solver(const Opti_params &opti_params,
                              File_parser_inout &file_inout,
                              Result_opti &opti_out) {

  double dt = .1;
  std::cout << "Warning: "
            << "dt is hardcoded to " << dt << std::endl;
  SOLVER solver = static_cast<SOLVER>(opti_params.solver_int);

  bool verbose = false;
  auto cl = file_inout.cl;
  auto xs_init = file_inout.xs;
  auto us_init = file_inout.us;
  size_t N = us_init.size();
  auto goal = file_inout.goal;
  auto start = file_inout.start;
  auto name = file_inout.name;

  if (opti_params.repair_init_guess) {
    std::cout << "WARNING: reparing init guess, annoying SO2" << std::endl;
    for (size_t i = 1; i < N + 1; i++) {
      xs_init.at(i)(2) = xs_init.at(i - 1)(2) +
                         diff_angle(xs_init.at(i)(2), xs_init.at(i - 1)(2));
    }
    goal(2) = xs_init.at(N)(2) + diff_angle(goal(2), xs_init.at(N)(2));

    std::cout << "goal is now (maybe updated) " << goal.transpose()
              << std::endl;
  }

  // if (free_time) {
  //   for (size_t i = 0; i < N; i++) {
  //     std::vector<double> new_u = actions.at(i);
  //     new_u.push_back(1.);
  //     actions.at(i) = new_u;
  //   }
  // }

  if (verbose) {
    std::cout << "states " << std::endl;
    for (auto &x : xs_init)
      std::cout << x.format(FMT) << std::endl;
  }

  bool feasible;
  std::vector<Eigen::VectorXd> xs_out;
  std::vector<Eigen::VectorXd> us_out;

  std::cout << "WARNING: "
            << "i modify last state to match goal" << std::endl;
  xs_init.back() = goal;

  std::ofstream debug_file_yaml("debug_file.yaml");
  debug_file_yaml << "name: " << name << std::endl;
  debug_file_yaml << "N: " << N << std::endl;
  debug_file_yaml << "start: " << start.format(FMT) << std::endl;
  debug_file_yaml << "goal: " << goal.format(FMT) << std::endl;
  debug_file_yaml << "xs0: " << std::endl;
  for (auto &x : xs_init)
    debug_file_yaml << "  - " << x.format(FMT) << std::endl;

  debug_file_yaml << "us0: " << std::endl;
  for (auto &x : us_init)
    debug_file_yaml << "  - " << x.format(FMT) << std::endl;

  if (solver == SOLVER::mpc || solver == SOLVER::mpcc) {
    // i could not stop when I reach the goal, only stop when I reach it with
    // the step move. Then, I would do the last at full speed? ( I hope :) )
    // Anyway, now is just fine

    CHECK_GEQ(opti_params.num_steps_to_optimize, opti_params.num_steps_to_move,
              AT);

    bool finished = false;

    std::vector<Eigen::VectorXd> xs_opt;
    std::vector<Eigen::VectorXd> us_opt;

    std::vector<Eigen::VectorXd> xs_init_rewrite = xs_init;
    std::vector<Eigen::VectorXd> us_init_rewrite = us_init;

    xs_opt.push_back(start);
    xs_init_rewrite.at(0) = start;

    debug_file_yaml << "opti:" << std::endl;

    auto times = Eigen::VectorXd::LinSpaced(xs_init.size(), 0,
                                            (xs_init.size() - 1) * dt);

    double max_alpha = times(times.size() - 1);

    ptr<Interpolator> path = mk<Interpolator>(times, xs_init);

    std::vector<Eigen::VectorXd> xs;
    std::vector<Eigen::VectorXd> us;

    double previous_alpha;

    Eigen::VectorXd previous_state = start;
    ptr<crocoddyl::ShootingProblem> problem;

    Eigen::VectorXd goal_mpc(3);

    bool is_last = false;

    double total_time = 0;
    size_t counter = 0;
    size_t total_iterations = 0;
    size_t num_steps_to_optimize_i = 0;

    while (!finished) {

      if (solver == SOLVER::mpc) {

        int remaining_steps = N - counter * opti_params.num_steps_to_move;

        num_steps_to_optimize_i =
            std::min(opti_params.num_steps_to_optimize, remaining_steps);

        int subgoal_index = 0;

        if (opti_params.adaptative_goal_mpc) {
          CHECK_EQ(true, false, AT);

          auto it =
              std::min_element(path->x.begin(), path->x.end(),
                               [&](const auto &a, const auto &b) {
                                 return (a - previous_state).squaredNorm() <=
                                        (b - previous_state).squaredNorm();
                               });

          size_t index = std::distance(path->x.begin(), it);
          std::cout << "starting with approx index " << index << std::endl;
          std::cout << "Non adaptative index would be: "
                    << counter * opti_params.num_steps_to_move << std::endl;

        } else {
          subgoal_index =
              counter * opti_params.num_steps_to_move + num_steps_to_optimize_i;
        }

        is_last = opti_params.num_steps_to_optimize > remaining_steps;
        auto start_i = previous_state;

        if (is_last)
          goal_mpc = goal;
        else
          goal_mpc = xs_init.at(subgoal_index);

        std::cout << "is_last:" << is_last << std::endl;
        std::cout << "counter:" << counter << std::endl;
        std::cout << "goal_i:" << goal_mpc.transpose() << std::endl;
        std::cout << "start_i:" << start_i.transpose() << std::endl;

        Opts opts{
            .free_time = false,
            .control_bounds = opti_params.control_bounds,
            .name = name,
            .N = num_steps_to_optimize_i,
            .regularize_wrt_init_guess = opti_params.regularize_wrt_init_guess,
            .use_finite_diff = opti_params.use_finite_diff,
            .goal = goal_mpc,
            .start = start_i,
            .cl = cl,
            .states = {},
            .actions = {},
        };

        size_t nx, nu;

        problem = generate_problem(opts, nx, nu);

        if (opti_params.use_warmstart) {

          // if (counter == 0) {
          xs = std::vector<Eigen::VectorXd>(
              xs_init_rewrite.begin() + counter * opti_params.num_steps_to_move,
              xs_init_rewrite.begin() +
                  counter * opti_params.num_steps_to_move +
                  num_steps_to_optimize_i + 1);
          us = std::vector<Eigen::VectorXd>(
              us_init_rewrite.begin() + counter * opti_params.num_steps_to_move,
              us_init_rewrite.begin() +
                  counter * opti_params.num_steps_to_move +
                  num_steps_to_optimize_i);
          // }

          // else {
          //
          //   // TODO: I have to take some from xopt from previous time.
          //   CHECK_EQ(xs_opt.size(), us_opt.size() + 1, AT);
          //
          //   auto xs_1 = std::vector<Eigen::VectorXd>(
          //       xs_opt.begin() + num_steps_to_move * counter, xs_opt.end());
          //
          //   CHECK_EQ(num_steps_to_move * counter, xs_opt.size(), AT);
          //
          //   auto xs_2 = std::vector<Eigen::VectorXd>(
          //       xs_init.begin() + counter * num_steps_to_move + 1,
          //       xs_init.begin() + counter * num_steps_to_move + 1 +
          //           num_steps_to_optimize_i);
          //
          //   xs_1.insert(xs_1.end(), xs_2.begin(), xs_2.end());
          //
          //   auto us_1 = std::vector<Eigen::VectorXd>(
          //       us_opt.begin() + num_steps_to_move * counter, us_opt.end());
          //
          //   CHECK_SEQ(counter * num_steps_to_move + num_steps_to_optimize_i,
          //             us_init.size(), AT);
          //
          //   auto us_2 = std::vector<Eigen::VectorXd>(
          //       us_init.begin() + counter * num_steps_to_move,
          //       us_init.begin() + counter * num_steps_to_move +
          //           num_steps_to_optimize_i);
          //
          //   us_1.insert(us_1.end(), us_2.begin(), us_2.end());
          //
          //   xs = xs_1;
          //   us = us_1;
          //
          //   std::cout << "x" << std::endl;
          //   for (auto &x : xs)
          //     std::cout << x.transpose() << std::endl;
          //
          //   std::cout << "u" << std::endl;
          //   for (auto &u : us)
          //     std::cout << u.transpose() << std::endl;
          //   std::cout << "done" << std::endl;
          // }

          CHECK_EQ(xs.size(), us.size() + 1, AT);
          CHECK_EQ(us.size(), num_steps_to_optimize_i, AT);

        } else {
          xs = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                            opts.start);
          us = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i,
                                            Eigen::VectorXd::Zero(nu));
        }

      } else if (solver == SOLVER::mpcc) {

        bool only_contour_last = true;
        const double threshold_alpha_close = 2.; //
        bool add_contour_to_all_when_close_to_gaol = true;
        const bool regularize_wrt_init_guess = false;

        num_steps_to_optimize_i = opti_params.num_steps_to_optimize;
        // approx alpha of first state
        auto it = std::min_element(
            path->x.begin(), path->x.end(), [&](const auto &a, const auto &b) {
              return (a - previous_state).squaredNorm() <=
                     (b - previous_state).squaredNorm();
            });

        size_t index = std::distance(path->x.begin(), it);
        std::cout << "starting with approx index " << index << std::endl;
        double alpha_of_first = path->times(index);
        std::cout << "alpha of first " << alpha_of_first << std::endl;

        double alpha_ref = alpha_of_first + opti_params.alpha_rate *
                                                num_steps_to_optimize_i * dt;

        // lets create a vector of alpha_refs
        Eigen::VectorXd alpha_refs = Eigen::VectorXd::LinSpaced(
            num_steps_to_optimize_i + 1, alpha_of_first,
            alpha_of_first + num_steps_to_optimize_i * dt);

        Eigen::VectorXd alpha_refs_goal = Eigen::VectorXd::LinSpaced(
            num_steps_to_optimize_i + 1, alpha_of_first,
            alpha_of_first +
                opti_params.alpha_rate * num_steps_to_optimize_i * dt);
        alpha_refs_goal = alpha_refs_goal.cwiseMin(max_alpha);
        alpha_refs = alpha_refs.cwiseMin(max_alpha);

        Eigen::VectorXd cost_alpha_multis =
            Eigen::VectorXd::Ones(num_steps_to_optimize_i + 1);
        cost_alpha_multis(0) =
            1.; // almost zero weight on alpha at the beginning

        std::cout << "vector of alpha refs  is " << std::endl;
        std::cout << alpha_refs.transpose() << std::endl;

        std::cout << "vector of alpha refs goals  is " << std::endl;
        std::cout << alpha_refs_goal.transpose() << std::endl;

        if (alpha_ref > max_alpha) {
          for (size_t i = 0; i < cost_alpha_multis.size(); i++) {
            if (alpha_refs_goal(i) >= max_alpha - 1e-4) {
              cost_alpha_multis(i) = 10.; //  Add higher alpha at the end
            }
          }
        }

        alpha_ref = std::min(alpha_ref, max_alpha);

        double cost_alpha_multi = 1.;
        int missing_indexes = times.size() - index;

        bool use_goal_reaching_formulation =
            std::fabs(alpha_of_first - max_alpha) < threshold_alpha_close;

        std::cout << "hardcoding " << std::endl;
        use_goal_reaching_formulation = true;
        add_contour_to_all_when_close_to_gaol = true;

        if (use_goal_reaching_formulation) {
          std::cout << "we are close to the goal" << std::endl;

#if 0
#endif
          if (add_contour_to_all_when_close_to_gaol) {
            only_contour_last = false;
            cost_alpha_multi = 1.;
          } else {
            std::cout << "alpha is close to the reference alpha " << std::endl;
            std::cout << std::fabs(alpha_of_first - max_alpha) << std::endl;
            std::cout << "adding very high cost to alpha" << std::endl;
            cost_alpha_multi = 100.;
          }
        }

        double cx_init;
        double cu_init;

        // if (!only_contour_last) {
        //   // the alpha_ref
        // }
        //
        // else {

        cx_init = std::min(alpha_of_first + 1.01 * num_steps_to_optimize_i * dt,
                           max_alpha - 1e-3);
        cu_init = cx_init;
        // }

        if (use_goal_reaching_formulation &&
            add_contour_to_all_when_close_to_gaol) {
          cx_init = alpha_of_first;
          cu_init = 0.;
        }

        std::cout << "cx_init:" << cx_init << std::endl;
        std::cout << "alpha_rate:" << opti_params.alpha_rate << std::endl;
        std::cout << "alpha_ref:" << alpha_ref << std::endl;
        std::cout << "max alpha " << max_alpha << std::endl;

        size_t nx, nu;
        const size_t _nx = 3;
        const size_t _nu = 2;

        Eigen::VectorXd start_ic(_nx + 1);
        start_ic.head(_nx) = previous_state.head(_nx);
        start_ic(_nx) = cx_init;

        Opts opts{
            .free_time = false,
            .control_bounds = opti_params.control_bounds,
            .name = name,
            .N = num_steps_to_optimize_i,
            .regularize_wrt_init_guess = regularize_wrt_init_guess,
            .use_finite_diff = opti_params.use_finite_diff,
            .goal = {},
            .start = start_ic,
            .cl = cl,
            .states = {},
            .actions = {},
            .contour_control = true,
            .interpolator = path,
            .ref_alpha = alpha_ref,
            .max_alpha = max_alpha,
            .cost_alpha_multi = cost_alpha_multi,
            .only_contour_last = only_contour_last,
            .alpha_refs = alpha_refs_goal,
            .cost_alpha_multis = cost_alpha_multis,
        };

        problem = generate_problem(opts, nx, nu);

        if (opti_params.use_warmstart) {
          // TODO: I need a more clever initialization. For example, using the
          // ones missing from last time, and then the default?

          std::cout << "warmstarting " << std::endl;
          size_t first_index = index;
          size_t last_index = index + num_steps_to_optimize_i;

          std::vector<Eigen::VectorXd> xs_i;
          std::vector<Eigen::VectorXd> us_i;

          if (last_index < xs_init.size() - 1) {
            std::cout << "last index is " << last_index << std::endl;
            // TODO: I should reuse the ones in OPT
            xs_i = std::vector<Eigen::VectorXd>(&xs_init.at(first_index),
                                                &xs_init.at(last_index) + 1);
            us_i = std::vector<Eigen::VectorXd>(&us_init.at(first_index),
                                                &us_init.at(last_index));
          } else if (first_index < xs_init.size() - 1) {

            // copy some indexes, and fill the others

            // 0 1 2 3 4

            // starting from index 3, i need five states: 3 4 4
            // size_t num_extra = (xs_init.size() - 1) - first_index;
            auto xs_1 = std::vector<Eigen::VectorXd>(
                xs_init.begin() + first_index, xs_init.end());

            // Eigen::VectorXd xref(4);
            // xref.head(3) = goal;
            // xref(3) = max_alpha;

            auto xs_2 = std::vector<Eigen::VectorXd>(
                num_steps_to_optimize_i + 1 - xs_1.size(), goal);

            xs_1.insert(xs_1.end(), xs_2.begin(), xs_2.end());

            auto us_1 = std::vector<Eigen::VectorXd>(
                us_init.begin() + first_index, us_init.end());

            // Eigen::VectorXd uref(3);
            // uref << 0., 0., 0.;

            auto us_2 = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i -
                                                         us_1.size(),
                                                     Eigen::VectorXd::Zero(2));

            us_1.insert(us_1.end(), us_2.begin(), us_2.end());

            xs_i = xs_1;
            us_i = us_1;

          }

          else {
            xs_i = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                                xs_init.at(first_index));
            us_i = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i,
                                                Eigen::VectorXd::Zero(2));
          }

          CHECK_EQ(xs_i.size(), num_steps_to_optimize_i + 1, AT);
          CHECK_EQ(us_i.size(), num_steps_to_optimize_i, AT);

          std::vector<Eigen::VectorXd> xcs_i(xs_i.size());
          std::vector<Eigen::VectorXd> ucs_i(us_i.size());

          bool old = false;
          if (old) {
            std::transform(xs_i.begin(), xs_i.end(), xcs_i.begin(),
                           [&_nx, &cx_init](auto &x) {
                             Eigen::VectorXd new_last(_nx + 1);
                             new_last.head(_nx) = x;
                             new_last(_nx) = cx_init;
                             return new_last;
                           });

            std::transform(us_i.begin(), us_i.end(), ucs_i.begin(),
                           [&_nu, &cu_init](auto &u) {
                             Eigen::VectorXd new_last(_nu + 1);
                             new_last.head(_nu) = u;
                             new_last(_nu) = cu_init;
                             return new_last;
                           });
          } else {

            for (size_t i = 0; i < xcs_i.size(); i++) {
              Eigen::VectorXd new_last(_nx + 1);
              new_last.head(_nx) = xs_i.at(i);
              new_last(_nx) = alpha_refs(i);
              xcs_i.at(i) = new_last;
            }

            for (size_t i = 0; i < ucs_i.size(); i++) {
              Eigen::VectorXd new_last(_nu + 1);
              new_last.head(_nu) = us_i.at(i);
              new_last(_nu) = alpha_refs(i + 1) - alpha_refs(i);
              ucs_i.at(i) = new_last;
            }
          }

          xs = xcs_i;
          us = ucs_i;

        } else {

          Eigen::VectorXd u0c(3);
          u0c.head(2).setZero();
          u0c(2) = cu_init;
          xs = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                            opts.start);
          us = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i, u0c);
          //
          // std::cout << "warmstarting the alpha's" << std::endl;
          //
          // for (size_t i = 0; i < xs.size(); i++) {
          //   xs.at(i)(3) = alpha_refs(i);
          // }
          //
          // for (size_t i = 0; i < us.size(); i++) {
          //   Eigen::VectorXd new_last(_nu + 1);
          //   us.at(i)(2) = alpha_refs(i + 1) - alpha_refs(i);
          // }
        }
      }

      crocoddyl::SolverBoxFDDP ddp(problem);
      ddp.set_th_stop(opti_params.th_stop);
      ddp.set_th_acceptnegstep(opti_params.th_acceptnegstep);

      if (opti_params.CALLBACKS) {
        std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
        cbs.push_back(mk<crocoddyl::CallbackVerbose>());
        ddp.setCallbacks(cbs);
      }

      // ENFORCING BOUNDS

      Eigen::VectorXd x_lb, x_ub, u_lb, u_ub;

      if (solver == SOLVER::mpcc) {
        double inf = 1e6;
        x_lb.resize(4);
        x_ub.resize(4);
        u_lb.resize(3);
        u_ub.resize(3);

        x_lb << -inf, -inf, -inf, 0.;
        x_ub << inf, inf, inf, max_alpha;
        u_lb << -inf, -inf, 0.;
        u_ub << inf, inf, inf;

      } else {
        double inf = 1e6;
        x_lb.resize(3);
        x_ub.resize(3);
        u_lb.resize(2);
        u_ub.resize(2);

        x_lb << -inf, -inf, -inf;
        x_ub << inf, inf, inf;
        u_lb << -inf, -inf;
        u_ub << inf, inf;
      }

      if (opti_params.noise_level > 0.) {
        for (size_t i = 0; i < xs.size(); i++) {
          xs.at(i) += opti_params.noise_level *
                      Eigen::VectorXd::Random(xs.front().size());
          xs.at(i) = enforce_bounds(xs.at(i), x_lb, x_ub);
        }

        for (size_t i = 0; i < us.size(); i++) {
          us.at(i) += opti_params.noise_level *
                      Eigen::VectorXd::Random(us.front().size());
          us.at(i) = enforce_bounds(us.at(i), u_lb, u_ub);
        }
      }

      crocoddyl::Timer timer;
      ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
      size_t iterations_i = ddp.get_iter();
      double time_i = timer.get_duration();
      std::cout << "time: " << time_i << std::endl;
      std::cout << "iterations: " << iterations_i << std::endl;
      total_time += time_i;
      total_iterations += iterations_i;
      std::vector<Eigen::VectorXd> xs_i_sol = ddp.get_xs();
      std::vector<Eigen::VectorXd> us_i_sol = ddp.get_us();

      previous_state = xs_i_sol.at(opti_params.num_steps_to_move).head(3);

      size_t copy_steps = 0;

      if (solver == SOLVER::mpcc) {
        // Eigen::VectorXd x_last = ddp.get_xs().back();
        // size_t final_index = num_steps_to_optimize_i;
        size_t final_index = opti_params.num_steps_to_move;

        double alpha_mpcc = ddp.get_xs().at(final_index)(3);
        Eigen::VectorXd x_last = ddp.get_xs().at(final_index);

        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "alpha_mpcc:" << alpha_mpcc << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "distance to goal: " << (x_last.head(3) - goal).norm()
                  << "\n**\n";

        auto fun_is_goal = [&](const auto &x) {
          return (x.head(3) - goal).norm() < 1e-2;
        };

        if (std::fabs(alpha_mpcc - times(times.size() - 1)) < 1. &&
            fun_is_goal(x_last)) {

          is_last = true;
          // check which is the first state that is close to goal

          std::cout << "checking first state that reaches the goal "
                    << std::endl;

          auto it = std::find_if(ddp.get_xs().begin(), ddp.get_xs().end(),
                                 [&](const auto &x) { return fun_is_goal(x); });

          assert(it != ddp.get_xs().end());

          num_steps_to_optimize_i = std::distance(ddp.get_xs().begin(), it);
          std::cout << "changing the number of steps to optimize(copy) to "
                    << num_steps_to_optimize_i << std::endl;
        }
      }

      if (is_last)
        copy_steps = num_steps_to_optimize_i;
      else
        copy_steps = opti_params.num_steps_to_move;

      for (size_t i = 0; i < copy_steps; i++)
        xs_opt.push_back(xs_i_sol.at(i + 1).head(3));

      for (size_t i = 0; i < copy_steps; i++)
        us_opt.push_back(us_i_sol.at(i).head(2));

      if (solver == SOLVER::mpc) {
        for (size_t i = 0; i < num_steps_to_optimize_i; i++)
          xs_init_rewrite.at(1 + counter * opti_params.num_steps_to_move + i) =
              xs_i_sol.at(i + 1).head(3);

        for (size_t i = 0; i < num_steps_to_optimize_i; i++)
          us_init_rewrite.at(counter * opti_params.num_steps_to_move + i) =
              us_i_sol.at(i).head(2);
      }

      debug_file_yaml << "  - xs0:" << std::endl;
      for (auto &x : xs)
        debug_file_yaml << "    - " << x.format(FMT) << std::endl;

      debug_file_yaml << "    us0:" << std::endl;
      for (auto &u : us)
        debug_file_yaml << "    - " << u.format(FMT) << std::endl;

      debug_file_yaml << "    xsOPT:" << std::endl;
      for (auto &x : xs_i_sol)
        debug_file_yaml << "    - " << x.format(FMT) << std::endl;

      debug_file_yaml << "    usOPT:" << std::endl;
      for (auto &u : us_i_sol)
        debug_file_yaml << "    - " << u.format(FMT) << std::endl;

      debug_file_yaml << "    start: " << xs.front().format(FMT) << std::endl;

      if (solver == SOLVER::mpc) {
        debug_file_yaml << "    goal: " << goal_mpc.format(FMT) << std::endl;
      } else if (solver == SOLVER::mpcc) {
        double alpha_mpcc = ddp.get_xs().back()(3);
        Eigen::VectorXd out(3);
        Eigen::VectorXd Jout(3);
        path->interpolate(alpha_mpcc, out, Jout);
        debug_file_yaml << "    alpha: " << alpha_mpcc << std::endl;
        debug_file_yaml << "    state_alpha: " << out.format(FMT) << std::endl;
      }

      CHECK_EQ(us_i_sol.size() + 1, xs_i_sol.size(), AT);

      // copy results

      if (is_last) {
        finished = true;
        std::cout << "finished: "
                  << "is_last" << std::endl;
      }

      counter++;

      if (counter > opti_params.max_mpc_iterations) {
        finished = true;
        std::cout << "finished: "
                  << "max mpc iterations" << std::endl;
      }
    }
    std::cout << "Total TIME: " << total_time << std::endl;
    std::cout << "Total Iterations: " << total_iterations << std::endl;

    xs_out = xs_opt;
    us_out = us_opt;

    debug_file_yaml << "xsOPT: " << std::endl;
    for (auto &x : xs_out)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    debug_file_yaml << "usOPT: " << std::endl;
    for (auto &u : us_out)
      debug_file_yaml << "  - " << u.format(FMT) << std::endl;

    // checking feasibility

    std::cout << "checking feasibility" << std::endl;
    size_t nx = 3;
    size_t nu = 2;
    ptr<Cost> feat_col = mk<Col_cost>(nx, nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(feat_col)->margin = 0.;
    Eigen::VectorXd goal_last = Eigen::VectorXd::Map(goal.data(), goal.size());

    bool feasible_ = check_feas(feat_col, xs_out, us_out, goal_last);

    ptr<Dynamics> dyn = mk<Dynamics_unicycle>();

    bool dynamics_feas = check_dynamics(xs_out, us_out, dyn);

    feasible = feasible_ && dynamics_feas;

    std::cout << "dynamics_feas: " << dynamics_feas << std::endl;

    std::cout << "feasible_ is " << feasible_ << std::endl;
    std::cout << "feasible is " << feasible << std::endl;
  } else if (solver == SOLVER::traj_opt ||
             solver == SOLVER::traj_opt_free_time ||
             solver == SOLVER::traj_opt_smooth_then_free_time) {

    Opts opts{
        .free_time = solver == SOLVER::traj_opt_free_time,
        .control_bounds = opti_params.control_bounds,
        .name = name,
        .N = N,
        .regularize_wrt_init_guess = opti_params.regularize_wrt_init_guess,
        .use_finite_diff = opti_params.use_finite_diff,
        .goal = goal,
        .start = start,
        .cl = cl,
        .states = xs_init,
        .actions = us_init,
    };

    size_t nx, nu;

    ptr<crocoddyl::ShootingProblem> problem = generate_problem(opts, nx, nu);
    std::vector<Eigen::VectorXd> xs(N + 1, opts.start);
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(nu));

    if (opti_params.use_warmstart) {
      xs = xs_init;
      us = us_init;
    }

    if (opti_params.noise_level > 0.) {
      for (size_t i = 0; i < xs.size(); i++) {
        xs.at(i) += opti_params.noise_level * Eigen::VectorXd::Random(nx);
      }

      for (size_t i = 0; i < us.size(); i++) {
        us.at(i) += opti_params.noise_level * Eigen::VectorXd::Random(nu);
      }
    }

    crocoddyl::SolverBoxFDDP ddp(problem);
    ddp.set_th_stop(opti_params.th_stop);
    ddp.set_th_acceptnegstep(opti_params.th_acceptnegstep);

    if (opti_params.CALLBACKS) {
      std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
      cbs.push_back(mk<crocoddyl::CallbackVerbose>());
      ddp.setCallbacks(cbs);
    }

    if (opti_params.check_time) {
      {
        crocoddyl::Timer timer;
        problem->calc(xs, us);
        std::cout << "calc time: " << timer.get_duration() << std::endl;
      }

      {
        crocoddyl::Timer timer;
        problem->calcDiff(xs, us);
        std::cout << "calcDiff time: " << timer.get_duration() << std::endl;
      }

      std::cout << "second time" << std::endl;
      {
        std::cout << problem->calc(xs, us) << std::endl;
        crocoddyl::Timer timer;
        problem->calc(xs, us);
        std::cout << "calc time: " << timer.get_duration() << std::endl;
      }

      {
        crocoddyl::Timer timer;
        problem->calcDiff(xs, us);
        std::cout << "calcDiff time: " << timer.get_duration() << std::endl;
      }
    }

    crocoddyl::Timer timer;
    ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
    std::cout << "time: " << timer.get_duration() << std::endl;

    // check the distance to the goal:
    ptr<Cost> feat_col = mk<Col_cost>(nx, nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(feat_col)->margin = 0.;

    feasible = check_feas(feat_col, ddp.get_xs(), ddp.get_us(), opts.goal);
    std::cout << "feasible is " << feasible << std::endl;

    std::cout << "solution" << std::endl;
    std::cout << problem->calc(xs, us) << std::endl;

    // I should have three modes:
    xs_out = ddp.get_xs();
    us_out = ddp.get_us();

    debug_file_yaml << "xsOPT: " << std::endl;
    for (auto &x : xs_out)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    debug_file_yaml << "usOPT: " << std::endl;
    for (auto &u : us_out)
      debug_file_yaml << "  - " << u.format(FMT) << std::endl;

    if (solver == SOLVER::traj_opt_smooth_then_free_time) {

      CHECK_EQ(opti_params.free_time, false, AT);

      std::cout << "repeating now with free time" << std::endl;
      opts.free_time = true;
      opts.regularize_wrt_init_guess = true;
      opts.states = xs_out;
      opts.actions = us_out;

      problem = generate_problem(opts, nx, nu);
      std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Zero(nx));
      std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(nu));

      std::cout << " nu " << nu << std::endl;
      for (size_t t = 0; t < N + 1; t++) {
        xs.at(t) = ddp.get_xs().at(t);
      }
      for (size_t t = 0; t < N; t++) {
        us.at(t) = Eigen::VectorXd::Ones(nu);
        us.at(t).head(nu - 1) = ddp.get_us().at(t);
      }

      for (auto &x : xs) {
        std::cout << x.transpose() << std::endl;
      }

      for (auto &u : us) {
        std::cout << u.transpose() << std::endl;
      }

      std::cout << "feasible is " << feasible << std::endl;

      std::cout << "before..." << std::endl;
      std::cout << "cost " << problem->calc(ddp.get_xs(), us) << std::endl;
      feasible = check_feas(feat_col, ddp.get_xs(), us, opts.goal);
      std::cout << "feasible is " << feasible << std::endl;

      ddp = crocoddyl::SolverBoxFDDP(problem);
      ddp.set_th_stop(opti_params.th_stop);
      ddp.set_th_acceptnegstep(opti_params.th_acceptnegstep);

      if (opti_params.CALLBACKS) {
        std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
        cbs.push_back(mk<crocoddyl::CallbackVerbose>());
        ddp.setCallbacks(cbs);
      }

      crocoddyl::Timer timer;
      ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
      std::cout << "time: " << timer.get_duration() << std::endl;

      feasible = check_feas(feat_col, ddp.get_xs(), ddp.get_us(), opts.goal);
      std::cout << "feasible is " << feasible << std::endl;

      double total_time = std::accumulate(
          ddp.get_us().begin(), ddp.get_us().end(), 0.,
          [&dt](auto &a, auto &b) { return a + dt * b.tail(1)(0); });

      std::cout << "original total time: " << dt * ddp.get_us().size()
                << std::endl;
      std::cout << "new total_time: " << total_time << std::endl;

      int num_time_steps = std::ceil(total_time / dt);
      std::cout << "number of time steps " << num_time_steps << std::endl;
      std::cout << "new total time " << num_time_steps * dt << std::endl;
      double scaling_factor = num_time_steps * dt / total_time;
      std::cout << "scaling factor " << scaling_factor << std::endl;
      assert(scaling_factor >= 1);
      CHECK_GEQ(scaling_factor, 1., AT);

      // now I have to sample at every dt
      // TODO: lOOK for Better solution than the trick with scaling

      auto times = Eigen::VectorXd(N + 1);
      times.setZero();
      for (size_t i = 1; i < times.size(); i++) {
        times(i) = times(i - 1) + dt * ddp.get_us().at(i - 1)(nu - 1);
      }
      std::cout << times.transpose() << std::endl;

      // TODO: be careful with SO(2)
      std::vector<Eigen::VectorXd> x_out, u_out;
      for (size_t i = 0; i < num_time_steps + 1; i++) {
        double t = i * dt / scaling_factor;
        Eigen::VectorXd out(nx);
        Eigen::VectorXd Jout(nx);
        linearInterpolation(times, ddp.get_xs(), t, out, Jout);
        x_out.push_back(out);
      }

      std::vector<Eigen::VectorXd> u_nx_orig(ddp.get_us().size());
      std::transform(ddp.get_us().begin(), ddp.get_us().end(),
                     u_nx_orig.begin(),
                     [&nu](auto &s) { return s.head(nu - 1); });

      for (size_t i = 0; i < num_time_steps; i++) {
        double t = i * dt / scaling_factor;
        Eigen::VectorXd out(nu - 1);
        std::cout << " i and time and num_time_steps is " << i << " " << t
                  << " " << num_time_steps << std::endl;
        Eigen::VectorXd J(nu - 1);
        linearInterpolation(times, u_nx_orig, t, out, J);
        u_out.push_back(out);
      }

      std::cout << "u out " << u_out.size() << std::endl;
      std::cout << "x out " << x_out.size() << std::endl;

      xs_out = x_out;
      us_out = u_out;
    };
  }

  std::ofstream results_txt("out.txt");

  for (auto &x : xs_out)
    results_txt << x.transpose().format(FMT) << std::endl;

  results_txt << "---" << std::endl;
  for (auto &u : us_out)
    results_txt << u.transpose().format(FMT) << std::endl;

  // store in the good format
  opti_out.feasible = feasible;
  opti_out.xs_out = xs_out;
  opti_out.us_out = us_out;
  opti_out.cost = us_out.size() * dt;
}

BOOST_AUTO_TEST_CASE(quim) {

  auto argv = boost::unit_test::framework::master_test_suite().argv;
  auto argc = boost::unit_test::framework::master_test_suite().argc;

  int verbose = 0;

  Opti_params opti_params;
  File_parser_inout file_inout;
  std::string config_file = "";
  std::string out;

  po::options_description desc("Allowed options");

  desc.add_options()("help", "produce help message");
  opti_params.add_options(desc);
  file_inout.add_options(desc);

  desc.add_options()("out", po::value<std::string>(&out)->required())(
      "config_file",
      po::value<std::string>(&config_file)->default_value(config_file));

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return;
    }

    if (config_file != "") {
      std::ifstream ifs{config_file};
      if (ifs)
        store(parse_config_file(ifs, desc), vm);
      else {
        std::cout << "cannont open config file: " << config_file << std::endl;
      }
      notify(vm);
    }

    std::cout << "***" << std::endl;
    std::cout << "variable map is: " << std::endl;
    PrintVariableMap(vm, std::cout);
    std::cout << "***" << std::endl;

  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return;
  }

  read_from_file(file_inout);

  Result_opti opti_out;
  solve_with_custom_solver(opti_params, file_inout, opti_out);

  // write the results.
  std::ofstream results_yaml(out);
  std::cout << "feasible is " << opti_out.feasible << std::endl;
  results_yaml << "result:" << std::endl;
  results_yaml << "  - states:" << std::endl;
  for (auto &x : opti_out.xs_out) {
    x(2) = std::remainder(x(2), 2 * M_PI);
    results_yaml << "      - " << x.format(FMT) << std::endl;
  }

  results_yaml << "    actions:" << std::endl;
  for (auto &u : opti_out.us_out) {
    results_yaml << "      - " << u.format(FMT) << std::endl;
  }

  std::cout << "warning: I need correct exit flag" << std::endl;
}

BOOST_AUTO_TEST_CASE(unicycle2) {
  // check the dynamics
  ptr<Dynamics> dyn = mk<Dynamics_unicycle2>();
  check_dyn(dyn, 1e-5);

  ptr<Dynamics> dyn_free_time = mk<Dynamics_unicycle2>(true);
  check_dyn(dyn_free_time, 1e-5);

  // generate the problem... to continue
}

BOOST_AUTO_TEST_CASE(test_unifree) {

  ptr<Dynamics> dyn = mk<Dynamics_unicycle>();
  check_dyn(dyn, 1e-5);

  ptr<Dynamics> dyn_free_time = mk<Dynamics_unicycle>(true);
  check_dyn(dyn, 1e-5);
}

// lets do integration!! For first order with and without
// use the reminder
// check with the integration.
//

BOOST_AUTO_TEST_CASE(test_a_easy_problem_with_numdiff) {}

BOOST_AUTO_TEST_CASE(easy_opti) {}

BOOST_AUTO_TEST_CASE(contour) {

  size_t num_time_steps = 9;
  double dt = .1;
  Eigen::VectorXd ts =
      Eigen::VectorXd::LinSpaced(num_time_steps + 1, 0, num_time_steps * dt);

  Eigen::MatrixXd xs(10, 3);
  xs << 0.707620811535916, 0.510258911240815, 0.417485437023409,
      0.603422256426978, 0.529498282727551, 0.270351549348981,
      0.228364197569334, 0.423745615677815, 0.637687289287490,
      0.275556796335168, 0.350856706427970, 0.684295784598905,
      0.514519311047655, 0.525077224890754, 0.351628308305896,
      0.724152914315666, 0.574461155457304, 0.469860285484058,
      0.529365063753288, 0.613328702656816, 0.237837040141739,
      0.522469395136878, 0.619099658652895, 0.237139665242069,
      0.677357023849552, 0.480655768435853, 0.422227610314397,
      0.247046593173758, 0.380604672404750, 0.670065791405019;

  std::vector<Eigen::VectorXd> xs_vec(10, Eigen::VectorXd(3));

  size_t i = 0;
  for (size_t i = 0; i < 10; i++)
    xs_vec.at(i) = xs.row(i);

  ptr<Interpolator> path = mk<Interpolator>(ts, xs_vec);

  size_t nx = 4;
  size_t nu = 3;

  Eigen::VectorXd out(3);
  Eigen::VectorXd J(3);

  path->interpolate(0.001, out, J);
  path->interpolate(dt + 1e-3, out, J);
  path->interpolate(num_time_steps * dt + .1, out, J);
  path->interpolate(0., out, J);
  path->interpolate(dt, out, J);
  path->interpolate(2 * dt + .1, out, J);
  // add some test?

  ptr<Contour_cost> cost = mk<Contour_cost>(nx, nu, path);

  Eigen::VectorXd x(4);
  x << 2., -1., .3, 0.34;

  Eigen::VectorXd u(3);
  u << 2., -1, .3;

  Eigen::VectorXd r(cost->nr);

  std::cout << "x " << x << std::endl;
  cost->calc(r, x);

  std::cout << "r" << r.transpose() << std::endl;

  Eigen::MatrixXd Jxdif(cost->nr, cost->nx);
  Eigen::MatrixXd Judif(cost->nr, cost->nu);

  cost->use_finite_diff = true;
  cost->calcDiff(Jxdif, Judif, x, u);

  std::cout << "Ju" << Judif << std::endl;
  std::cout << "Jx" << Jxdif << std::endl;

  Eigen::MatrixXd Jx(cost->nr, cost->nx);
  Eigen::MatrixXd Ju(cost->nr, cost->nu);

  cost->use_finite_diff = false;
  cost->calcDiff(Jx, Ju, x, u);

  std::cout << "Ju" << Ju << std::endl;
  std::cout << "Jx" << Jx << std::endl;
  std::cout << "Judif" << Judif << std::endl;

  BOOST_TEST((Jx - Jxdif).norm());
  BOOST_TEST((Ju - Judif).norm());
}

// continue here:
// bounds,
// combine solvers,
// integration (finally!)

BOOST_AUTO_TEST_CASE(linear_interpolation) {

  Eigen::VectorXd ts = Eigen::VectorXd::LinSpaced(10, 0, 9);

  std::vector<Eigen::VectorXd> xs_vec(10, Eigen::VectorXd(2));

  for (size_t i = 0; i < 10; i++) {
    Eigen::VectorXd x(2);
    x << i, 2 * i;
    xs_vec.at(i) = x;
  }

  ptr<Interpolator> path = mk<Interpolator>(ts, xs_vec);

  Eigen::VectorXd x(2);
  Eigen::VectorXd J(2);

  for (size_t i = 0; i < 10; i++) {
    path->interpolate(ts(i), x, J);
    std::cout << x << std::endl;
    BOOST_TEST((x - xs_vec.at(i)).norm() < 1e-12);
  }

  path->interpolate(10, x, J);
  std::cout << x << std::endl;
  path->interpolate(11, x, J);
  std::cout << x << std::endl;

  path->interpolate(.5, x, J);
  std::cout << x << std::endl;
  path->interpolate(.1, x, J);
  std::cout << x << std::endl;

  path->interpolate(.89, x, J);
  std::cout << x << std::endl;

  // add some test?
}

// Strategy:
// We optimize with the MPC controller without bounds, using the time with the
// deltas. We optimize with the path tracking formulation with alpha_rate = 1.
// We optimize with the path tracking formulation with alpha = 1.5.

// Next steps: lets get a solution from db-astar with deltas.
// and lets check for the full pipeline.
// Should it be in python?
//
//
//

// this works.
// (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make &&   ./test_croco
// --run_test=quim
// --  --env ../benchmark/unicycle_first_order_0/parallelpark_0.yaml --w
// aypoints ../build_debug/qdbg/result_dbastar_newresult.yaml   --reg 1 --out
// out.yaml --new_format true --control_bounds 0  --solver 3
