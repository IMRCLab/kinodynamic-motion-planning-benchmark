#include "generate_primitives.hpp"
#include "robot_models.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

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

// #include "collision_checker.hpp"
#include "croco_macros.hpp"

// save data without the cluster stuff

#include <filesystem>
#include <random>
#include <regex>
#include <type_traits>

#include <filesystem>
#include <regex>

#include "croco_models.hpp"
#include "motions.hpp"
#include "ocp.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>

using namespace std;

Eigen::VectorXd default_vector;

using namespace crocoddyl;

// Eigen::Vector3d goal(1.9, .3, 0);
// issue with derivatives...

template <typename T> T scalarFunctionOne(T const &x) {
  return 2 * x * x + 3 * x + 1;
};

void checkFunctionOne(double &x, double &dfdx) { dfdx = 4 * x + 3; }

template <typename T> T scalarFunctionTwo(T const &x, T const &y) {
  return 2 * x * x + 3 * x + 3 * x * y * y + 2 * y + 1;
};

void checkFunctionTwo(double &x, double &y, double &dfdx, double &dfdy) {
  dfdx = 4 * x + 3 + 3 * y * y;
  dfdy = 6 * x * y + 2;
}

struct my_f {

  Eigen::VectorXd operator()(const Eigen::VectorXd &x) { return x; }
};

template <typename Scalar_, int NX = Eigen::Dynamic, int NY = Eigen::Dynamic>
struct Func {
  typedef Scalar_ Scalar;
  enum { InputsAtCompileTime = NX, ValuesAtCompileTime = NY };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>
      JacobianType;

  int m_inputs, m_values;

  Func() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Func(int inputs_, int values_) : m_inputs(inputs_), m_values(values_) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  template <typename T>
  void operator()(const Eigen::Matrix<T, InputsAtCompileTime, 1> &x,
                  Eigen::Matrix<T, ValuesAtCompileTime, 1> *_v) const {
    Eigen::Matrix<T, ValuesAtCompileTime, 1> &v = *_v;
    v = 2 * x;
  }
};

struct Func2 {
  typedef double Scalar;
  enum { InputsAtCompileTime = 3, ValuesAtCompileTime = 3 };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>
      JacobianType;

  int m_inputs, m_values;

  Func2() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
  Func2(int inputs_, int values_) : m_inputs(inputs_), m_values(values_) {}

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  template <typename T>
  void operator()(const Eigen::Matrix<T, InputsAtCompileTime, 1> &x,
                  Eigen::Matrix<T, ValuesAtCompileTime, 1> *_v) const {
    Eigen::Matrix<T, ValuesAtCompileTime, 1> &v = *_v;

    auto &&q = x(0);
    v(0) = q;
    v(1) = x(1);
    v(2) = x(2) + q + x(1);
  }
};

struct Func4 {
  typedef double Scalar;
  enum {
    InputsAtCompileTime = Eigen::Dynamic,
    ValuesAtCompileTime = Eigen::Dynamic
  };
  typedef Eigen::Matrix<Scalar, InputsAtCompileTime, 1> InputType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
  typedef Eigen::Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime>
      JacobianType;

  int m_inputs, m_values;

  Func4(int inputs_, int values_)
      : m_inputs(inputs_), m_values(values_), J_(3), inverseJ_(3) {
    J_ << 16.571710e-6, 16.655602e-6, 29.261652e-6;
    inverseJ_ << 1 / J_(0), 1 / J_(1), 1 / J_(2);
  }

  int inputs() const { return m_inputs; }
  int values() const { return m_values; }

  Eigen::VectorXd J_;
  Eigen::VectorXd inverseJ_;

  template <typename T>
  void operator()(const Eigen::Matrix<T, InputsAtCompileTime, 1> &x,
                  Eigen::Matrix<T, ValuesAtCompileTime, 1> *_v) const {
    Eigen::Matrix<T, ValuesAtCompileTime, 1> &v = *_v;

    auto &&pos_in = x.segment(0, 3);
    auto &&q_in = x.segment(0, 4);
    auto &&vel_in = x.segment(7, 3);
    auto &&omega_in = x.segment(10, 3);
    auto &&force_in = x.segment(13, 4);

    auto &&pos_out = v.segment(0, 3);
    auto &&q_out = v.segment(0, 4);
    auto &&vel_out = v.segment(10, 3);
    auto &&omega_out = v.segment(7, 3);

    pos_out = pos_in + vel_in;
    vel_out = vel_in + force_in.template head<3>();
    omega_out = J_.cwiseProduct(omega_in).template head<3>().cross(
        omega_in.template head<3>());
    q_out = q_in + force_in;
  }
};

BOOST_AUTO_TEST_CASE(eigen_derivatives) {

  double x, y, z, f, g, dfdx, dgdy, dgdz;
  (void)f;
  (void)g;
  Eigen::AutoDiffScalar<Eigen::VectorXd> xA, yA, zA, fA, gA;

  cout << endl << "Testing scalar function with 1 input..." << endl;

  xA.value() = 1;
  xA.derivatives() = Eigen::VectorXd::Unit(1, 0);

  fA = scalarFunctionOne(xA);

  cout << "  AutoDiff:" << endl;
  cout << "    Function output: " << fA.value() << endl;
  cout << "    Derivative: " << fA.derivatives() << endl;

  x = 1;
  checkFunctionOne(x, dfdx);

  cout << "  Hand differentiation:" << endl;
  cout << "    Derivative: " << dfdx << endl << endl;

  cout << "Testing scalar function with 2 inputs..." << endl;

  yA.value() = 1;
  zA.value() = 2;

  yA.derivatives() = Eigen::VectorXd::Unit(2, 0);
  zA.derivatives() = Eigen::VectorXd::Unit(2, 1);

  gA = scalarFunctionTwo(yA, zA);

  cout << "  AutoDiff:" << endl;
  cout << "    Function output: " << gA.value() << endl;
  cout << "    Derivative: " << gA.derivatives()[0] << ", "
       << gA.derivatives()[1] << endl;

  y = 1;
  z = 2;
  checkFunctionTwo(y, z, dgdy, dgdz);

  cout << "  Hand differentiation:" << endl;
  cout << "    Derivative: " << dgdy << ", " << dgdz << endl;

  {
    using F = Func<double, 2, 2>;
    F f;

    typename F::InputType x = F::InputType::Random(f.inputs());
    typename F::ValueType y(f.values()), yref(f.values());
    typename F::JacobianType j(f.values(), f.inputs()),
        jref(f.values(), f.inputs());

    jref.setZero();
    yref.setZero();

    // f(x, &yref, &jref);
    //     std::cerr << y.transpose() << "\n\n";;
    //     std::cerr << j << "\n\n";;

    j.setZero();
    y.setZero();
    Eigen::AutoDiffJacobian<F> autoj(f);
    autoj(x, &y, &j);
    // autoj(x, &y, &j);
    std::cout << y.transpose() << "\n\n";
    ;
    std::cout << j << "\n\n";
    ;

    // VERIFY_IS_APPROX(y, yref);
    // VERIFY_IS_APPROX(j, jref);
  }

  {
    using F = Func2;
    F f;

    typename F::InputType x = F::InputType::Random(f.inputs());
    typename F::ValueType y(f.values()), yref(f.values());
    typename F::JacobianType j(f.values(), f.inputs()),
        jref(f.values(), f.inputs());

    jref.setZero();
    yref.setZero();

    // f(x, &yref, &jref);
    //     std::cout << y.transpose() << "\n\n";;
    //     std::cout << j << "\n\n";;

    j.setZero();
    y.setZero();
    Eigen::AutoDiffJacobian<F> autoj(f);
    autoj(x, &y, &j);
    // autoj(x, &y, &j);
    std::cout << y.transpose() << "\n\n";
    ;
    std::cout << j << "\n\n";
    ;

    // VERIFY_IS_APPROX(y, yref);
    // VERIFY_IS_APPROX(j, jref);
  }

  {

    using F = Func4;
    F f(17, 13);

    typename F::InputType x = F::InputType::Random(f.inputs());
    typename F::ValueType y(f.values()), yref(f.values());
    typename F::JacobianType j(f.values(), f.inputs()),
        jref(f.values(), f.inputs());

    jref.setZero();
    yref.setZero();

    // f(x, &yref, &jref);
    //     std::cerr << y.transpose() << "\n\n";;
    //     std::cerr << j << "\n\n";;

    j.setZero();
    y.setZero();
    Eigen::AutoDiffJacobian<F> autoj(f);
    autoj(x, &y, &j);
    // autoj(x, &y, &j);
    std::cout << y.transpose() << "\n\n";
    ;
    std::cout << j << "\n\n";
    ;
  }
}

BOOST_AUTO_TEST_CASE(t_acrobot) {
  double tol = 1e-7;
  double margin_rate = 100;
  {
    auto dyn = mk<Dynamics>(mks<Model_acrobot>());
    check_dyn(dyn, tol, default_vector, default_vector, margin_rate);
  }

  {
    auto dyn = mk<Dynamics>(mks<Model_acrobot>(), Control_Mode::default_mode);
    check_dyn(dyn, tol, default_vector, default_vector, margin_rate);
  }
}

BOOST_AUTO_TEST_CASE(acrobot_rollout) {
  size_t T = 100; // one second
  // auto dyn = mk<Dynamics_acrobot>();
  auto model = mk<Model_acrobot>();

  std::vector<Eigen::VectorXd> us;
  std::vector<Eigen::VectorXd> xs;
  for (size_t i = 0; i < T; i++) {
    us.push_back(model->params.max_torque * .01 * Eigen::VectorXd::Random(1));
  }

  // rollout

  Eigen::VectorXd xnext(4);
  Eigen::VectorXd xold(4);
  xold << 0, 0, 0, 0;
  xs.push_back(xold);
  double dt = .01;

  for (size_t i = 0; i < T; i++) {
    std::cout << "u " << us.at(i).format(FMT) << std::endl;
    std::cout << "xold " << xold.format(FMT) << std::endl;
    model->step(xnext, xold, us.at(i), dt);
    std::cout << "xnext " << xnext.format(FMT) << std::endl;
    xold = xnext;
    xs.push_back(xold);
  }

  std::cout << "final state" << xs.back().format(FMT) << std::endl;
}

BOOST_AUTO_TEST_CASE(acrobot_rollout_free) {
  auto model = mk<Model_acrobot>();
  double dt = .01;
  size_t T = 1. / dt; // one second

  std::vector<Eigen::VectorXd> us;
  std::vector<Eigen::VectorXd> xs;
  for (size_t i = 0; i < T; i++) {
    us.push_back(Eigen::VectorXd::Zero(1));
  }

  // rollout

  Eigen::VectorXd xnext(4);
  Eigen::VectorXd xold(4);
  {
    xold << 2.8, 0, 0, 0;
    xs.push_back(xold);
    double original_energy = model->calcEnergy(xold);
    for (size_t i = 0; i < T; i++) {
      std::cout << "i: " << i << std::endl;
      std::cout << "u " << us.at(i).format(FMT) << std::endl;
      std::cout << "xold " << xold.format(FMT) << std::endl;
      model->step(xnext, xold, us.at(i), dt);
      std::cout << "xnext " << xnext.format(FMT) << std::endl;
      xold = xnext;
      model->calcEnergy(xold);
      xs.push_back(xold);
    }

    double last_energy = model->calcEnergy(xs.back());
    BOOST_TEST(std::abs(original_energy - last_energy) <
               4); // euler integration is very bad here!!
  }

  {
    std::cout << "R4 " << std::endl;
    xs.clear();
    xold << 2.8, 0, 0, 0;
    double original_energy = model->calcEnergy(xold);
    for (size_t i = 0; i < T; i++) {
      std::cout << "i: " << i << std::endl;
      std::cout << "u " << us.at(i).format(FMT) << std::endl;
      std::cout << "xold " << xold.format(FMT) << std::endl;
      model->stepR4(xnext, xold, us.at(i), dt);
      std::cout << "xnext " << xnext.format(FMT) << std::endl;
      xold = xnext;
      model->calcEnergy(xold);
      xs.push_back(xold);
    }
    double last_energy = model->calcEnergy(xs.back());
    BOOST_TEST(std::abs(original_energy - last_energy) < 1e-2);
  }

  // std::cout << "final state" << xs.back().format(FMT) << std::endl;

  // dyn->max_torque =
}

BOOST_AUTO_TEST_CASE(quad2d) {
  double tol = 1e-7;
  {
    auto dyn = mk<Dynamics>(mks<Model_quad2d>());
    check_dyn(dyn, tol);
  }
  {
    auto dyn_free_time =
        mk<Dynamics>(mks<Model_quad2d>(), Control_Mode::free_time);

    Eigen::VectorXd x(6);
    x.setRandom();

    Eigen::VectorXd u(3);
    u.setRandom();
    u(2) = std::fabs(u(2));

    check_dyn(dyn_free_time, tol, x, u);
  }

  // now with drag
  {

    auto model = mks<Model_quad2d>();
    model->params.drag_against_vel = true;
    auto dyn = mk<Dynamics>(model);

    check_dyn(dyn, tol);
  }

  {

    Eigen::VectorXd x(6);
    x.setRandom();

    Eigen::VectorXd u(3);
    u.setRandom();
    u(2) = std::fabs(u(2));

    auto model = mks<Model_quad2d>();
    model->params.drag_against_vel = true;
    auto dyn = mk<Dynamics>(model);

    auto dyn_free_time = mk<Dynamics>(model, Control_Mode::free_time);
    check_dyn(dyn_free_time, tol, x, u);
  }
}

BOOST_AUTO_TEST_CASE(t_car2) {
  {
    ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_car2>());
    check_dyn(dyn, 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(car_trailer) {
  {
    ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_car_with_trailers>());
    check_dyn(dyn, 1e-5);
  }
  {
    ptr<Dynamics> dyn_free_time =
        mk<Dynamics>(mks<Model_car_with_trailers>(), Control_Mode::free_time);

    Eigen::VectorXd x(dyn_free_time->nx);
    Eigen::VectorXd u(dyn_free_time->nu);
    x.setRandom();
    u.setRandom();
    u(u.size() - 1) = std::fabs(u(u.size() - 1));

    check_dyn(dyn_free_time, 1e-5, x, u);
  }

  {
    ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_car_with_trailers>());
    check_dyn(dyn, 1e-5);
  }

  {

    ptr<Dynamics> dyn_free_time =
        mk<Dynamics>(mks<Model_car_with_trailers>(), Control_Mode::free_time);

    Eigen::VectorXd x(dyn_free_time->nx);
    Eigen::VectorXd u(dyn_free_time->nu);
    x.setRandom();
    u.setRandom();
    u(u.size() - 1) = std::fabs(u(u.size() - 1));

    check_dyn(dyn_free_time, 1e-5, x, u);
  }
}

BOOST_AUTO_TEST_CASE(t_qintegrate) {

  Eigen::Quaterniond q = Eigen::Quaterniond(0, 0, 0, 1);
  double dt = .01;
  Eigen::Vector3d omega{0, 0, 1};

  Eigen::Vector4d out;
  Eigen::Vector4d ye;
  Eigen::Vector4d deltaQ;
  __get_quat_from_ang_vel_time(omega * dt, deltaQ, nullptr);
  quat_product(q.coeffs(), deltaQ, out, nullptr, nullptr);

  std::cout << "out\n" << out << std::endl;

  Eigen::MatrixXd JqD(4, 4);
  double eps = 1e-6;
  for (size_t i = 0; i < 4; i++) {
    Eigen::Vector4d qe;
    // Eigen::Vector3d ye;
    qe = q.coeffs();
    qe(i) += eps;
    // qe.normalize();

    Eigen::Vector4d ye;
    Eigen::Vector4d deltaQ;
    __get_quat_from_ang_vel_time(omega * dt, deltaQ, nullptr);
    quat_product(qe, deltaQ, ye, nullptr, nullptr);
    auto df = (ye - out) / eps;
    JqD.col(i) = df;
  }

  Eigen::MatrixXd JomegaD(4, 3);
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d omegae;
    omegae = omega;
    omegae(i) += eps;
    Eigen::Vector4d ye;
    Eigen::Vector4d deltaQ;
    __get_quat_from_ang_vel_time(omegae * dt, deltaQ, nullptr);
    quat_product(q.coeffs(), deltaQ, ye, nullptr, nullptr);

    auto df = (ye - out) / eps;
    JomegaD.col(i) = df;
  }

  std::cout << "omega" << std::endl;
  std::cout << JomegaD << std::endl;
  std::cout << "q" << std::endl;
  std::cout << JqD << std::endl;
  // TODO: check the diffs against analytic!!
}

BOOST_AUTO_TEST_CASE(t_quat_product) {

  Eigen::Vector4d p{1, 2, 3, 4};
  p.normalize();

  Eigen::Vector4d q{1, .2, .3, .4};
  p.normalize();

  Eigen::Vector4d out;
  Eigen::Matrix4d Jp;
  Eigen::Matrix4d Jq;

  quat_product(p, q, out, &Jp, &Jq);

  Eigen::Quaterniond out_eigen = Eigen::Quaterniond(p) * Eigen::Quaterniond(q);

  bool check1 = (out_eigen.coeffs() - out).cwiseAbs().maxCoeff() < 1e-10;

  if (!check1) {

    std::cout << "out_eigen" << std::endl;
    std::cout << out_eigen.coeffs() << std::endl;
    std::cout << "out" << std::endl;
    std::cout << out << std::endl;
    std::cout << "out_eigen.coeffs() - out" << std::endl;
    std::cout << out_eigen.coeffs() - out << std::endl;
  }

  CHECK(check1, AT);

  Eigen::Matrix4d __Jp;
  Eigen::Matrix4d __Jq;

  Eigen::Matrix4d JpD;
  Eigen::Matrix4d JqD;

  finite_diff_jac(
      [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
        quat_product(x, q, y, &__Jp, &__Jq);
      },
      p, 4, JpD);

  finite_diff_jac(
      [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
        quat_product(p, x, y, &__Jp, &__Jq);
      },
      p, 4, JqD);

  double eps = 1e-6;

  bool check2 = (Jq - JqD).cwiseAbs().maxCoeff() < 10 * eps;

  if (!check2) {

    std::cout << "Jq" << std::endl;
    std::cout << Jq << std::endl;
    std::cout << "JqD" << std::endl;
    std::cout << JqD << std::endl;
    std::cout << "Jq - JqD" << std::endl;
    std::cout << Jq - JqD << std::endl;
  }
  CHECK(check2, AT);

  bool check3 = (Jp - JpD).cwiseAbs().maxCoeff() < 10 * eps;

  if (!check3) {

    std::cout << "Jp" << std::endl;
    std::cout << Jp << std::endl;
    std::cout << "JpD" << std::endl;
    std::cout << JpD << std::endl;
    std::cout << "Jp - JpD" << std::endl;
    std::cout << Jp - JpD << std::endl;
  }
  CHECK(check3, AT);
}

BOOST_AUTO_TEST_CASE(exp_map_quat) {

  using Matrix43d = Eigen::Matrix<double, 4, 3>;
  {
    Eigen::Vector3d v(-.1, .1, .2);
    Eigen::Vector4d q;
    Matrix43d J(4, 3);
    Matrix43d Jd(4, 3);

    __get_quat_from_ang_vel_time(v, q, &J);

    std::cout << "q " << q.format(FMT) << std::endl;

    Eigen::Quaterniond ref;
    ref = get_quat_from_ang_vel_time(v);
    std::cout << "ref " << ref.coeffs().format(FMT) << std::endl;

    finite_diff_jac(
        [&](const Eigen::VectorXd &xx, Eigen::Ref<Eigen::VectorXd> y) {
          return __get_quat_from_ang_vel_time(xx, y);
        },
        v, 4, Jd, 1e-8);

    std::cout << "error \n"
              << (Jd - J) << std::endl
              << "norm " << (Jd - J).norm() << std::endl;

    BOOST_TEST((ref.coeffs() - q).norm() <= 1e-7);
    BOOST_TEST((Jd - J).norm() <= 1e-7);
  }

  {
    Eigen::Vector3d v(-.1, .1, .2);
    v *= 1e-12;
    Eigen::Vector4d q;

    Matrix43d J(4, 3);
    Matrix43d Jd(4, 3);
    __get_quat_from_ang_vel_time(v, q, &J);

    std::cout << "q " << q.format(FMT) << std::endl;

    Eigen::Quaterniond ref;
    ref = get_quat_from_ang_vel_time(v);
    std::cout << "ref " << ref.coeffs().format(FMT) << std::endl;

    finite_diff_jac(
        [&](const Eigen::VectorXd &xx, Eigen::Ref<Eigen::VectorXd> y) {
          return __get_quat_from_ang_vel_time(xx, y);
        },
        v, 4, Jd, 1e-9);

    std::cout << "error \n"
              << (Jd - J) << std::endl
              << "norm " << (Jd - J).norm() << std::endl;

    BOOST_TEST((Jd - J).norm() <= 1e-7);
    BOOST_TEST((ref.coeffs() - q).norm() <= 1e-7);
  }

  {
    Eigen::Vector3d v(0., 0., 0.);
    v *= 1e-12;
    Eigen::Vector4d q;

    Matrix43d J(4, 3);
    Matrix43d Jd(4, 3);
    __get_quat_from_ang_vel_time(v, q, &J);

    std::cout << "q " << q.format(FMT) << std::endl;

    Eigen::Quaterniond ref;
    ref = get_quat_from_ang_vel_time(v);
    std::cout << "ref " << ref.coeffs().format(FMT) << std::endl;

    finite_diff_jac(
        [&](const Eigen::VectorXd &xx, Eigen::Ref<Eigen::VectorXd> y) {
          return __get_quat_from_ang_vel_time(xx, y);
        },
        v, 4, Jd, 1e-13);

    std::cout << Jd << std::endl;
    std::cout << J << std::endl;
    std::cout << "error \n"
              << (Jd - J) << std::endl
              << "norm " << (Jd - J).norm() << std::endl;

    BOOST_TEST((Jd - J).norm() <= 1e-7);
    BOOST_TEST((ref.coeffs() - q).norm() <= 1e-7);
  }
}

BOOST_AUTO_TEST_CASE(quad3d_bench_time) {

  size_t N = 1000;
  ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_quad3d>());

  Eigen::VectorXd xnext(13);
  xnext.setZero();

  Eigen::VectorXd x(13);
  Eigen::VectorXd u(4);

  x.setRandom();
  u.setRandom();

  int nx = dyn->nx;
  int nu = dyn->nu;

  Eigen::MatrixXd Fx(nx, nx);
  Eigen::MatrixXd Fu(nx, nu);
  Fx.setZero();
  Fu.setZero();

  {
    crocoddyl::Timer timer;
    for (size_t i = 0; i < N; i++) {
      dyn->calc(xnext, x, u);
    }
    std::cout << timer.get_duration() << std::endl;
  }

  {
    crocoddyl::Timer timer;
    for (size_t i = 0; i < N; i++) {
      dyn->calcDiff(Fx, Fu, x, u);
    }
    std::cout << timer.get_duration() << std::endl;
  }

  {
    crocoddyl::Timer timer;

    Eigen::MatrixXd Fxd(nx, nx);
    Eigen::MatrixXd Fud(nx, nu);
    Fxd.setZero();
    Fud.setZero();
    for (size_t i = 0; i < N; i++) {
      finite_diff_jac(
          [&](const Eigen::VectorXd &xx, Eigen::Ref<Eigen::VectorXd> y) {
            return dyn->calc(y, xx, u);
          },
          x, 13, Fxd);

      finite_diff_jac(
          [&](const Eigen::VectorXd &xx, Eigen::Ref<Eigen::VectorXd> y) {
            return dyn->calc(y, x, xx);
          },
          u, 13, Fud);
    }
    std::cout << timer.get_duration() << std::endl;
  }
}

BOOST_AUTO_TEST_CASE(quad3d) {
  ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_quad3d>());
  check_dyn(dyn, 1e-6);
}

BOOST_AUTO_TEST_CASE(quad3d_ompl) {
  ptr<Dynamics> dyn =
      mk<Dynamics>(mks<Model_quad3d>("../models/quad3d_omplapp.yaml"));
  check_dyn(dyn, 1e-6);
}

BOOST_AUTO_TEST_CASE(unicycle2) {
  ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_unicycle2>());
  check_dyn(dyn, 1e-5);

  {
    Eigen::VectorXd x(5);
    x.setRandom();

    Eigen::VectorXd u(3);
    u.setRandom();
    u(2) = std::fabs(u(2));

    ptr<Dynamics> dyn_free_time =
        mk<Dynamics>(mks<Model_unicycle2>(), Control_Mode::free_time);
    check_dyn(dyn_free_time, 1e-5, x, u);
  }
}

BOOST_AUTO_TEST_CASE(test_unifree) {

  ptr<Dynamics> dyn =
      mk<Dynamics>(mks<Model_unicycle1>(), Control_Mode::default_mode);

  check_dyn(dyn, 1e-5);
  ptr<Dynamics> dyn_free_time =
      mk<Dynamics>(mks<Model_unicycle1>(), Control_Mode::free_time);
  check_dyn(dyn, 1e-5);
}

#if 0 
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
#endif

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

  // TODO: add some test!
}

#if 0
BOOST_AUTO_TEST_CASE(traj_opt_no_bounds) {

  options_trajopt = Options_trajopt();

  Options_trajopt options_trajopt;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_bugtrap_0_sol0.yaml";

  options_trajopt.control_bounds = 0;
  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt);
  options_trajopt.use_warmstart = 1;
  options_trajopt.max_iter = 50;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt,sol, result);
  BOOST_TEST(result.feasible);
}
#endif

BOOST_AUTO_TEST_CASE(easy_quad_ompl) {

  Problem problem1("../benchmark/quadrotor_ompl/empty_easy_ompl.yaml");
  Problem problem2("../benchmark/quadrotor_ompl/empty_easy_2_ompl.yaml");

  size_t i = 0;
  for (auto &ptr : {&problem1, &problem2}) {

    auto &problem = *ptr;
    Trajectory traj_in, traj_out;
    traj_in.num_time_steps = 300;

    Options_trajopt options_trajopt;
    options_trajopt.solver_id = 0;
    options_trajopt.smooth_traj = false;
    options_trajopt.weight_goal = 200;
    options_trajopt.max_iter = 100;

    Result_opti opti_out;

    trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                            opti_out);

    {
      std::ofstream out("out_opt_" + std::to_string(i) + ".yaml");
      traj_out.to_yaml_format(out);
      i++;
    }

    BOOST_TEST(opti_out.feasible);
  }
}

BOOST_AUTO_TEST_CASE(recovery_quad3d_ompl) {

  Problem problem("../benchmark/quadrotor_ompl/recovery_ompl.yaml");

  Trajectory traj_in, traj_out;
  traj_in.num_time_steps = 400;

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 0;
  options_trajopt.smooth_traj = true;
  options_trajopt.weight_goal = 300;
  options_trajopt.max_iter = 200;
  options_trajopt.ref_x0 = 1;
  options_trajopt.control_bounds = 1;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt_.yaml");
    traj_out.to_yaml_format(out);
  }

  BOOST_TEST(opti_out.feasible);
}

BOOST_AUTO_TEST_CASE(jac_quadrotor) {

  Options_trajopt options_trajopt;
  options_trajopt.disturbance = 1e-7;
  size_t nx(13), nu(4);
  size_t N = 5;

  Eigen::VectorXd goal = Eigen::VectorXd::Zero(13);
  goal(6) = 1;
  goal(0) = .1;
  goal(1) = .1;
  goal(2) = .1;
  Eigen::VectorXd start = Eigen::VectorXd::Zero(13);
  start(6) = 1;
  start += .1 * Eigen::VectorXd::Random(13);
  start.segment(3, 4).normalize();

  std::vector<Eigen::VectorXd> xs(N + 1);
  std::vector<Eigen::VectorXd> us(N);

  for (auto &x : xs) {
    x = start;
    x += .1 * Eigen::VectorXd::Random(13);
    start.segment(3, 4).normalize();
  }

  for (auto &u : us) {
    u = Eigen::VectorXd::Zero(4);
    u += .1 * Eigen::VectorXd::Random(4);
  }

  //
  //
  //
  //

  // auto model_robot = mks<Model_quad3d>();

  for (auto &model_robot :
       {mks<Model_quad3d>(), mks<Model_quad3d>("../model/quad3d_v0.yaml"),
        mks<Model_quad3d>("../models/quad3d_omplapp.yaml")}) {

    CSTR_(model_robot->params.filename);
    Generate_params gen_args;
    gen_args.name = "quadrotor_0";
    gen_args.N = N;
    options_trajopt.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;

    auto problem = generate_problem(gen_args, options_trajopt, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    options_trajopt.use_finite_diff = 1;
    auto problem_diff = generate_problem(gen_args, options_trajopt, nx, nu);
    problem_diff->calc(xs, us);
    problem_diff->calcDiff(xs, us);
    auto data_running_diff = problem_diff->get_runningDatas();
    auto data_terminal_diff = problem_diff->get_terminalData();

    double tol = 1e-3;

    // BOOST_CHECK((
    std::cout << "data_terminal_diff->Lx" << std::endl;
    std::cout << data_terminal_diff->Lx << std::endl;
    std::cout << "data_terminal->Lx" << std::endl;
    std::cout << data_terminal->Lx << std::endl;

    // - data_terminal->Lx).isZero(tol));
    // BOOST_CHECK((data_terminal_diff->Lx -
    // data_terminal->Lx).isZero(tol));
    //
    BOOST_CHECK(
        check_equal(data_terminal_diff->Lx, data_terminal->Lx, tol, tol));
    BOOST_CHECK(
        check_equal(data_terminal_diff->Lxx, data_terminal->Lxx, tol, tol));
    BOOST_CHECK(
        check_equal(data_terminal_diff->Lxu, data_terminal->Lxu, tol, tol));
    BOOST_CHECK(
        check_equal(data_terminal_diff->Luu, data_terminal->Luu, tol, tol));

    BOOST_CHECK_EQUAL(data_running_diff.size(), data_running.size());

    for (size_t i = 0; i < data_running_diff.size(); i++) {
      std::cout << "check " << i << std::endl;
      auto &d = data_running.at(i);
      auto &d_diff = data_running_diff.at(i);

      BOOST_CHECK(check_equal(d_diff->Lx, d->Lx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Lu, d->Lu, tol, tol));

      std::cout << "d->Lu" << std::endl;
      std::cout << d->Lu << std::endl;
      std::cout << "d->Lu diff" << std::endl;
      std::cout << d_diff->Lu << std::endl;

      BOOST_CHECK(check_equal(d_diff->Fx, d->Fx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fu, d->Fu, tol, tol));
      std::cout << "d_diff->Lxx" << std::endl;
      std::cout << d_diff->Lxx << std::endl;
      std::cout << "d->Lxx" << std::endl;
      std::cout << d->Lxx << std::endl;
      BOOST_CHECK(check_equal(d_diff->Lxx, d->Lxx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Lxu, d->Lxu, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Luu, d->Luu, tol, tol));
    }
  }
}

BOOST_AUTO_TEST_CASE(test_jacobians) {

  Options_trajopt options_trajopt;
  options_trajopt.disturbance = 1e-5;
  size_t nx(0), nu(0);

  size_t N = 5;
  Eigen::VectorXd alpha_refs = Eigen::VectorXd::Random(N + 1);
  Eigen::VectorXd cost_alpha_multis = Eigen::VectorXd::Random(N + 1);

  std::vector<Eigen::VectorXd> path;
  path.push_back(Eigen::VectorXd::Random(3));
  path.push_back(Eigen::VectorXd::Random(3));

  std::shared_ptr<Model_robot> model_robot = mks<Model_unicycle1>();

  ptr<Interpolator> interpolator =
      mk<Interpolator>(Eigen::Vector2d({-2., 3.}), path, model_robot->state);
  {

    Eigen::VectorXd goal = Eigen::VectorXd::Random(3);
    Eigen::VectorXd start = Eigen::VectorXd::Random(4);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(4));
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(3));

    Generate_params gen_args;
    gen_args.name = "unicycle1_v0";
    gen_args.N = 5;
    options_trajopt.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    // gen_args.cl = cl;
    gen_args.model_robot = model_robot;
    gen_args.contour_control = true;
    gen_args.interpolator = interpolator;

    auto problem = generate_problem(gen_args, options_trajopt, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    options_trajopt.use_finite_diff = 1;
    auto problem_diff = generate_problem(gen_args, options_trajopt, nx, nu);
    problem_diff->calc(xs, us);
    problem_diff->calcDiff(xs, us);
    auto data_running_diff = problem_diff->get_runningDatas();
    auto data_terminal_diff = problem_diff->get_terminalData();

    double tol = 1e-3;

    // - data_terminal->Lx).isZero(tol));
    // BOOST_CHECK((data_terminal_diff->Lx -
    // data_terminal->Lx).isZero(tol));
    BOOST_CHECK(
        check_equal(data_terminal_diff->Lx, data_terminal->Lx, tol, tol));

    BOOST_WARN(
        check_equal(data_terminal_diff->Lxx, data_terminal->Lxx, tol, tol));

    std::cout << data_terminal_diff->Lx << std::endl;
    std::cout << data_terminal->Lx << std::endl;

    BOOST_CHECK_EQUAL(data_running_diff.size(), data_running.size());

    for (size_t i = 0; i < data_running_diff.size(); i++) {
      auto &d = data_running.at(i);
      auto &d_diff = data_running_diff.at(i);

      BOOST_CHECK(check_equal(d_diff->Lx, d->Lx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Lu, d->Lu, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fx, d->Fx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fu, d->Fu, tol, tol));

      BOOST_WARN(check_equal(d_diff->Lxx, d->Lxx, tol, tol));
      BOOST_WARN(check_equal(d_diff->Lxu, d->Lxu, tol, tol));
      BOOST_WARN(check_equal(d_diff->Luu, d->Luu, tol, tol)); // NOTE: it
      // will give false because of Finite diff use gauss newton
    }
  }

#if 1
  {

    Generate_params gen_args;

    Eigen::VectorXd goal = Eigen::VectorXd::Random(3);
    Eigen::VectorXd start = Eigen::VectorXd::Random(3);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(3));
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(2));

    gen_args.name = "unicycle_first_order_0";
    gen_args.N = 5;
    options_trajopt.use_finite_diff = false;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;
    gen_args.contour_control = false;

    auto problem = generate_problem(gen_args, options_trajopt, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    options_trajopt.use_finite_diff = true;
    auto problem_diff = generate_problem(gen_args, options_trajopt, nx, nu);
    problem_diff->calc(xs, us);
    problem_diff->calcDiff(xs, us);
    auto data_running_diff = problem_diff->get_runningDatas();
    auto data_terminal_diff = problem_diff->get_terminalData();

    double tol = 1e-3;

    std::cout << "terminal " << std::endl;

    BOOST_CHECK(
        check_equal(data_terminal_diff->Lx, data_terminal->Lx, tol, tol));
    BOOST_CHECK_EQUAL(data_running_diff.size(), data_running.size());

    for (size_t i = 0; i < data_running_diff.size(); i++) {
      auto &d = data_running.at(i);
      auto &d_diff = data_running_diff.at(i);

      BOOST_CHECK(check_equal(d_diff->Lx, d->Lx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Lu, d->Lu, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fx, d->Fx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fu, d->Fu, tol, tol));
    }
  }
  {

    Generate_params gen_args;

    Eigen::VectorXd goal = Eigen::VectorXd::Random(3);
    Eigen::VectorXd start = Eigen::VectorXd::Random(4);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(4));
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(3));

    gen_args.name = "unicycle_first_order_0";
    gen_args.N = 5;
    options_trajopt.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;
    gen_args.interpolator = interpolator;
    gen_args.contour_control = true;
    gen_args.linear_contour = true;

    auto problem = generate_problem(gen_args, options_trajopt, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    options_trajopt.use_finite_diff = 1;
    auto problem_diff = generate_problem(gen_args, options_trajopt, nx, nu);
    problem_diff->calc(xs, us);
    problem_diff->calcDiff(xs, us);
    auto data_running_diff = problem_diff->get_runningDatas();
    auto data_terminal_diff = problem_diff->get_terminalData();

    double tol = 1e-3;

    std::cout << "terminal " << std::endl;

    BOOST_CHECK(
        check_equal(data_terminal_diff->Lx, data_terminal->Lx, tol, tol));
    BOOST_CHECK_EQUAL(data_running_diff.size(), data_running.size());

    for (size_t i = 0; i < data_running_diff.size(); i++) {
      auto &d = data_running.at(i);
      auto &d_diff = data_running_diff.at(i);

      BOOST_CHECK(check_equal(d_diff->Lx, d->Lx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Lu, d->Lu, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fx, d->Fx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Fu, d->Fu, tol, tol));
    }
  }

#endif

  {

    Generate_params gen_args;

    Eigen::VectorXd goal = Eigen::VectorXd::Random(3);
    Eigen::VectorXd start = Eigen::VectorXd::Random(4);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(4));

    for (auto &x : xs) {
      x.tail(1)(0) = std::abs(x.tail(1)(0));
    }

    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(3));
    for (auto &u : us) {
      u.tail(1)(0) = std::abs(u.tail(1)(0));
    }

    gen_args.name = "unicycle_first_order_0";
    gen_args.N = 5;
    options_trajopt.use_finite_diff = false;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;
    gen_args.free_time = true;
    gen_args.free_time_linear = true;

    auto problem = generate_problem(gen_args, options_trajopt, nx, nu);
    // now with finite diff
    options_trajopt.use_finite_diff = true;
    auto problem_diff = generate_problem(gen_args, options_trajopt, nx, nu);

    bool equal = check_problem(problem, problem_diff, xs, us);
    BOOST_WARN(equal); // TODO: hard check on gradient.
    // Warn on hessian
  }
}

BOOST_AUTO_TEST_CASE(state_bounds) {

  Eigen::VectorXd ub = Eigen::VectorXd::Ones(4);

  Eigen::VectorXd weight_b(4);
  weight_b << 1., 2., 3., 4.;

  Eigen::VectorXd x(4);
  x << .5, .9, 1.1, 1.2;

  Eigen::VectorXd u(4);
  u << .5, .9, 1.1, 1.2;

  int nr = 4;
  int nx = 4;
  int nu = 4;

  ptr<Cost> state_bounds = mk<State_bounds>(4, 4, 4, ub, weight_b);

  Eigen::MatrixXd Jx(nr, nx);
  Eigen::MatrixXd Ju(nr, nu);

  Eigen::MatrixXd Jxdiff(nr, nx);
  Eigen::MatrixXd Judiff(nr, nu);

  Eigen::VectorXd r(4);

  state_bounds->calc(r, x, u);

  Eigen::VectorXd Lx(4);
  Eigen::VectorXd Lu(4);
  Eigen::MatrixXd Lxx(4, 4);
  Eigen::MatrixXd Luu(4, 4);
  Eigen::MatrixXd Lxu(4, 4);

  Eigen::VectorXd Lxdiff(4);
  Eigen::VectorXd Ludiff(4);
  Eigen::MatrixXd Lxxdiff(4, 4);
  Eigen::MatrixXd Luudiff(4, 4);
  Eigen::MatrixXd Lxudiff(4, 4);

  Lx.setZero();
  Lu.setZero();
  Lxx.setZero();
  Luu.setZero();
  Lxu.setZero();

  state_bounds->calcDiff(Lx, Lu, Lxx, Luu, Lxu, x, u);

  finite_diff_grad(
      [&](auto &y) {
        state_bounds->calc(r, y, u);
        return .5 * r.dot(r);
      },
      x, Lxdiff);

  finite_diff_grad(
      [&](auto &y) {
        state_bounds->calc(r, x, y);
        return .5 * r.dot(r);
      },
      u, Ludiff);

  finite_diff_hess(
      [&](auto &y) {
        state_bounds->calc(r, y, u);
        return .5 * r.dot(r);
      },
      x, Lxxdiff);

  finite_diff_hess(
      [&](auto &y) {
        state_bounds->calc(r, x, y);
        return .5 * r.dot(r);
      },
      u, Luudiff);

  double tol = 1e-3;
  BOOST_CHECK(check_equal(Lx, Lxdiff, tol, tol));
  BOOST_CHECK(check_equal(Lu, Ludiff, tol, tol));
  BOOST_CHECK(check_equal(Lxx, Lxxdiff, tol, tol));
  BOOST_CHECK(check_equal(Luu, Luudiff, tol, tol));
}

BOOST_AUTO_TEST_CASE(contour_park_raw) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "parallelpark_0.yaml");

  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  options_trajopt.control_bounds = 1;
  options_trajopt.use_finite_diff = 0;
  options_trajopt.k_linear = 10.;
  options_trajopt.k_contour = 10.;

  options_trajopt.use_warmstart = true;
  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.max_iter = 50;
  options_trajopt.weight_goal = 100;

  options_trajopt.window_shift = 10;
  options_trajopt.window_optimize = 40;
  options_trajopt.smooth_traj = 1;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST(result.feasible);
}

BOOST_AUTO_TEST_CASE(contour_park_easy) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "parallelpark_0.yaml");

  Trajectory init_guess("../build_debug/smooth_park_debug.yaml");

  options_trajopt.control_bounds = true;
  options_trajopt.use_finite_diff = false;
  options_trajopt.k_linear = 10.;
  options_trajopt.k_contour = 2.;
  options_trajopt.weight_goal = 300;

  options_trajopt.use_warmstart = true;
  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.max_iter = 20;

  Result_opti result;
  Trajectory sol;

  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST(result.feasible);

  std::string filename = "out.yaml";
  std::cout << "writing results to:" << filename << std::endl;
  std::ofstream file_out(filename);
  result.write_yaml_db(file_out);
}

BOOST_AUTO_TEST_CASE(parallel_small_step_good_init_guess) {
  Options_trajopt options_trajopt;

  Problem problem("../benchmark/unicycle_first_order_0/"
                  "parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_parallelpark_mpcc.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 2;
  options_trajopt.window_optimize = 35;
  options_trajopt.smooth_traj = 1;
  options_trajopt.k_linear = 20.;
  options_trajopt.k_contour = 10.;
  options_trajopt.weight_goal = 200;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  // NOTE cost should be 3.2
  BOOST_TEST_CHECK(result.cost <= 3.3);
}

BOOST_AUTO_TEST_CASE(bugtrap_so2_hard) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "bugtrap_0.yaml");
  Trajectory init_guess(
      "../test/unicycle_first_order_0/guess_bugtrap_0_sol1.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 10;
  options_trajopt.window_optimize = 40;
  options_trajopt.smooth_traj = 0;
  options_trajopt.k_linear = 10.;
  options_trajopt.k_contour = 10.;
  options_trajopt.weight_goal = 200;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 27.91);
}

BOOST_AUTO_TEST_CASE(bugtrap_so2_easy) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "bugtrap_0.yaml");
  Trajectory init_guess(
      "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 10;
  options_trajopt.window_optimize = 40;
  options_trajopt.smooth_traj = 1;
  options_trajopt.k_linear = 10.;
  options_trajopt.k_contour = 10.;
  options_trajopt.weight_goal = 200;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 24.2);
}

BOOST_AUTO_TEST_CASE(bugtrap_bad_init_guess) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "bugtrap_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_bugtrap_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 10;
  options_trajopt.window_optimize = 40;
  options_trajopt.smooth_traj = 1;
  options_trajopt.k_linear = 10.;
  options_trajopt.k_contour = 10.;
  options_trajopt.weight_goal = 200;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 23.0);
}

BOOST_AUTO_TEST_CASE(bugtrap_good_init_guess) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "bugtrap_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_bugtrap_0_mpcc.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 10;
  options_trajopt.window_optimize = 40;
  options_trajopt.smooth_traj = 0;
  options_trajopt.k_linear = 10.;
  options_trajopt.k_contour = 10.;
  options_trajopt.weight_goal = 200;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 23.0);
}

BOOST_AUTO_TEST_CASE(parallel_free_time) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt_free_time);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  std::cout << "cost is " << result.cost << std::endl;
  std::cout << "feasible is " << result.feasible << std::endl;
  BOOST_TEST(result.feasible);
  BOOST_TEST(result.cost <= 3.3);
}

BOOST_AUTO_TEST_CASE(parallel_free_time_linear) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_parallelpark_0_sol0.yaml");
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  {

    options_trajopt.solver_id =
        static_cast<int>(SOLVER::traj_opt_free_time_proxi_linear);

    Result_opti result;
    Trajectory sol;
    trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
    BOOST_TEST(result.success);
    std::cout << "cost is " << result.cost << std::endl;
    BOOST_TEST(result.cost <= 3.3);
  }

  {
    options_trajopt.solver_id =
        static_cast<int>(SOLVER::traj_opt_free_time_linear);

    Result_opti result;
    Trajectory sol;
    trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
    BOOST_TEST_CHECK(result.feasible);
    std::cout << "cost is " << result.cost << std::endl;
    BOOST_TEST_CHECK(result.cost <= 3.3);
  }
}

BOOST_AUTO_TEST_CASE(parallel_bad_init_guess) {

  Options_trajopt options_trajopt;

  Problem problem("../benchmark/unicycle_first_order_0/"
                  "parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 5;
  options_trajopt.window_optimize = 30;
  options_trajopt.smooth_traj = 1;
  options_trajopt.k_linear = 5.;
  options_trajopt.k_contour = 10.;
  options_trajopt.weight_goal = 200;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 3.6);
}

BOOST_AUTO_TEST_CASE(parallel_search_time) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  // (opti) ⋊> ~/s/w/k/build_debug on dev ⨯
  // make -j4 &&  ./test_croco
  // --run_test=quim --  --env
  // ../benchmark/uni
  // cycle_first_order_0/parallelpark_0.yaml
  // --waypoints
  // ../test/unicycle_first_order_0/guess_parallelpark_0_so
  // l0.yaml  --out out.yaml --solver 9
  // --control_bounds 1 --use_warmstart 1 >
  // quim.txt

  options_trajopt.solver_id = static_cast<int>(SOLVER::time_search_traj_opt);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 3.3);
}

BOOST_AUTO_TEST_CASE(bugtrap_bad_mpc_adaptative) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/"
                  "bugtrap_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_bugtrap_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpc_adaptative);

  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 20;
  options_trajopt.window_optimize = 50;
  options_trajopt.weight_goal = 100;
  options_trajopt.noise_level = 0.;
  options_trajopt.smooth_traj = true;
  options_trajopt.shift_repeat = false;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 27.);
}

BOOST_AUTO_TEST_CASE(kink_mpcc) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_first_order_0/kink_0.yaml");
  Trajectory init_guess("../test/unicycle_first_order_0/"
                        "guess_kink_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.window_shift = 20;
  options_trajopt.window_optimize = 40;
  options_trajopt.weight_goal = 100;
  options_trajopt.k_linear = 20;
  options_trajopt.k_contour = 10;
  options_trajopt.smooth_traj = 1;
  options_trajopt.max_iter = 30;
  options_trajopt.shift_repeat = false;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 14.);
}

BOOST_AUTO_TEST_CASE(eigen0) {
  Eigen::MatrixXd a = Eigen::MatrixXd::Zero(4, 4);
  auto &&jx = a.block(0, 0, 2, 2);
  jx = Eigen::MatrixXd::Ones(2, 2);
  std::cout << a << std::endl;
}

BOOST_AUTO_TEST_CASE(t_normalize) {

  Eigen::Vector4d q(1, 2, 1, 2.);
  Eigen::Vector4d y;
  Eigen::Matrix4d Jq;

  normalize(q, y, Jq);

  Eigen::Matrix4d __Jq;
  Eigen::MatrixXd JqD(4, 4);
  double eps = 1e-6;

  finite_diff_jac(
      [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
        return normalize(x, y, __Jq);
      },
      q, 4, JqD);

  bool check2 = (Jq - JqD).cwiseAbs().maxCoeff() < 10 * eps;

  if (!check2) {
    std::cout << "Jq" << std::endl;
    std::cout << Jq << std::endl;
    std::cout << "JqD" << std::endl;
    std::cout << JqD << std::endl;
    std::cout << "Jq - JqD" << std::endl;
    std::cout << Jq - JqD << std::endl;
    CHECK(((Jq - JqD).cwiseAbs().maxCoeff() < 10 * eps), AT);
  }
}

// TODO: use
//  R = exp(theta)
//  Exp(theta) + dtheta
//  I can use theta and dtheta as my parametrization.
//  Then, I can compute the derivatives and

BOOST_AUTO_TEST_CASE(matrix_rotation) {
  // very big error. Compute the rotation of a
  // vector. check with finite diff.

  Eigen::MatrixXd Jq(3, 4);
  Eigen::Matrix3d Ja;

  // Eigen::Vector4d q(1, 2, 1, 2.);
  Eigen::Vector4d q(0, 0, 0, 1.);
  q.normalize();
  Eigen::Vector3d a(1, 2, 3);
  Eigen::Vector3d y;

  rotate_with_q(q, a, y, Jq, Ja);

  // with finite diff

  Eigen::MatrixXd __Jq(3, 4);
  Eigen::Matrix3d __Ja;
  Eigen::MatrixXd JqD(3, 4);
  double eps = 1e-6;
  for (size_t i = 0; i < 4; i++) {
    Eigen::Vector4d qe;
    Eigen::Vector3d ye;
    qe = q;
    qe(i) += eps;
    // qe.normalize();
    rotate_with_q(qe, a, ye, __Jq, __Ja);
    auto df = (ye - y) / eps;
    std::cout << "ye " << ye << std::endl;
    JqD.col(i) = df;
  }

  Eigen::Matrix3d JaD;
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d ae;
    Eigen::Vector3d ye;
    ae = a;
    ae(i) += eps;
    rotate_with_q(q, ae, ye, __Jq, __Ja);
    auto df = (ye - y) / eps;
    JaD.col(i) = df;
  }

  bool check1 = (Ja - JaD).cwiseAbs().maxCoeff() < 10 * eps;
  bool check2 = (Jq - JqD).cwiseAbs().maxCoeff() < 10 * eps;

  if (!check1) {
    std::cout << "Ja" << std::endl;
    std::cout << Ja << std::endl;
    std::cout << "JaD" << std::endl;
    std::cout << JaD << std::endl;
    std::cout << "Ja - JaD" << std::endl;
    std::cout << Ja - JaD << std::endl;
    CHECK(((Ja - JaD).cwiseAbs().maxCoeff() < 10 * eps), AT);
  }

  if (!check2) {
    std::cout << "Jq" << std::endl;
    std::cout << Jq << std::endl;
    std::cout << "JqD" << std::endl;
    std::cout << JqD << std::endl;
    std::cout << "Jq - JqD" << std::endl;
    std::cout << Jq - JqD << std::endl;
    CHECK(((Jq - JqD).cwiseAbs().maxCoeff() < 10 * eps), AT);
  }
}

BOOST_AUTO_TEST_CASE(second_order_park_traj_opt) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_second_order_0/"
                  "parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_second_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.weight_goal = 100;
  options_trajopt.max_iter = 50;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 10.);
}

BOOST_AUTO_TEST_CASE(second_order_park_time) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_second_order_0/"
                  "parallelpark_0.yaml");
  Trajectory init_guess("../test/unicycle_second_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt_free_time);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.weight_goal = 100;
  options_trajopt.max_iter = 50;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 11.);
}

BOOST_AUTO_TEST_CASE(quadrotor_0_recovery) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/quadrotor_0/"
                  "empty_test_recovery_welf.yaml");
  // auto model_robot = mks<Model_quad3d>();
  options_trajopt.solver_id = 0;
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.weight_goal = 300;
  options_trajopt.max_iter = 400;
  options_trajopt.noise_level = 1e-6;
  options_trajopt.ref_x0 = 1;
  options_trajopt.use_finite_diff = 0;

  Result_opti result;
  Trajectory sol;
  Trajectory init_guess; // TODO: what to do in this cases?
  init_guess.num_time_steps = 300;

  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 5.);
}

BOOST_AUTO_TEST_CASE(t_recovery_with_smooth) {

  Problem problem("../benchmark/quadrotor_0/"
                  "empty_test_recovery_welf.yaml");
  Trajectory init_guess; // TODO: use T!!
  init_guess.num_time_steps = 400;

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt_no_bound_bound);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = 1;
  options_trajopt.weight_goal = 100;
  options_trajopt.max_iter = 1000;
  options_trajopt.noise_level = 1e-4;
  options_trajopt.ref_x0 = true;
  options_trajopt.smooth_traj = 1;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 4.);
}

BOOST_AUTO_TEST_CASE(t_slerp) {

  Eigen::Vector4d a(1, 0, 0, 0);
  Eigen::Vector4d v(0, 0, 0, 1);

  Eigen::Vector4d out(0, 0, 0, 1);

  std::cout << Eigen::Quaterniond(a).slerp(0, Eigen::Quaterniond(v)).coeffs()
            << std::endl;
  std::cout << Eigen::Quaterniond(a).slerp(0.5, Eigen::Quaterniond(v)).coeffs()
            << std::endl;
  std::cout << Eigen::Quaterniond(a).slerp(1., Eigen::Quaterniond(v)).coeffs()
            << std::endl;
}

BOOST_AUTO_TEST_CASE(mpc_controller) {

  Options_trajopt options_trajopt;
  bool half = true;
  Problem problem;
  Trajectory init_guess; // TODO: use T!!

  if (!half) {
    problem.read_from_yaml("../benchmark/unicycle_first_order_0/"
                           "check_region_of_attraction.yaml");
    init_guess.num_time_steps = 40;
  } else {
    problem.read_from_yaml("../benchmark/unicycle_first_order_0/"
                           "check_region_of_attraction_half."
                           "yaml");
    init_guess.num_time_steps = 30;
  }

  options_trajopt.solver_id = static_cast<int>(SOLVER::traj_opt);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = true;
  options_trajopt.weight_goal = 200;
  options_trajopt.smooth_traj = 1;
  options_trajopt.max_iter = 40;

  Result_opti result;
  Trajectory sol;

  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 40.);

  std::ofstream ref_out("ref_out-h" + std::to_string(half) + ".yaml");
  result.write_yaml_db(ref_out);

  // # controller
  size_t num_samples = 5;

  double noise_magnitude = .6;

  Eigen::VectorXd start = problem.start;

  struct Data_out {
    bool feasible = 0;
    Eigen::VectorXd start;
    std::vector<Eigen::VectorXd> xs;
    std::vector<Eigen::VectorXd> us;
  };

  std::vector<int> max_iter_array{2, 5, 10};

  for (const auto &max_iter : max_iter_array) {
    std::vector<Data_out> datas;

    for (size_t i = 0; i < num_samples; i++) {

      Eigen::Vector3d new_start =
          start + noise_magnitude * Eigen::VectorXd::Random(start.size());

      Result_opti resulti;
      Trajectory soli;
      Trajectory init_guess;
      init_guess.states = result.xs_out;
      init_guess.actions = result.us_out;
      problem.start = new_start;
      options_trajopt.max_iter = max_iter;
      options_trajopt.th_acceptnegstep = 2.;
      options_trajopt.smooth_traj = false;

      trajectory_optimization(problem, init_guess, options_trajopt, soli,
                              resulti);

      Data_out data_out{
          .feasible = resulti.feasible,
          .start = new_start,
          .xs = resulti.xs_out,
          .us = resulti.us_out,
      };
      datas.push_back(data_out);
    }
    // save to file
    //
    std::ofstream out("mpc_attraction-" +
                      std::to_string(options_trajopt.max_iter) + "-h" +
                      std::to_string(half) + ".yaml");
    std::string space2 = "  ";
    std::string space4 = "    ";

    for (auto &d : datas) {

      out << "- " << std::endl;
      out << space2 << "feasible: " << d.feasible << std::endl;
      out << space2 << "start: " << d.start.format(FMT) << std::endl;
      out << space2 << "states: " << std::endl;
      for (auto &s : d.xs) {
        out << space4 << "- " << s.format(FMT) << std::endl;
      }
      out << space2 << "actions: " << std::endl;
      for (auto &s : d.us) {
        out << space4 << "- " << s.format(FMT) << std::endl;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(park_second_mpcc) {

  Options_trajopt options_trajopt;
  Problem problem("../benchmark/unicycle_second_order_0/"
                  "parallelpark_0.yaml");

  Trajectory init_guess("../test/unicycle_second_order_0/"
                        "guess_parallelpark_0_sol0.yaml");

  options_trajopt.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  options_trajopt.control_bounds = 1;
  options_trajopt.use_warmstart = true;
  options_trajopt.k_linear = 50;
  options_trajopt.k_contour = 100;
  options_trajopt.weight_goal = 200;
  options_trajopt.smooth_traj = 1;
  options_trajopt.max_iter = 30;
  options_trajopt.window_optimize = 40;
  options_trajopt.window_shift = 20;
  options_trajopt.shift_repeat = 0;

  Result_opti result;
  Trajectory sol;
  trajectory_optimization(problem, init_guess, options_trajopt, sol, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 12.);
}

// BOOST_AUTO_TEST_CASE(step_with_se2) {
//
//   Eigen::Vector3d x(0, 0, 0);
//   Eigen::Vector2d u(1, 1);
//   Eigen::Vector3d xnext;
//   double dt = .1;
//
//   std::cout << " x " << x.format(FMT) << std::endl;
//
//   {
//     Model_unicycle1_se2 model_se2;
//
//     model_se2.step(xnext, x, u, dt);
//     std::cout << xnext.format(FMT) << std::endl;
//   }
//
//   {
//     Model_unicycle1_R2SO2 model_se2;
//     model_se2.step(xnext, x, u, dt);
//     std::cout << xnext.format(FMT) << std::endl;
//   }
//   {
//     Model_unicycle1 model;
//     model.step(xnext, x, u, dt);
//     std::cout << xnext.format(FMT) << std::endl;
//   }
//
//   x << 0, 0, 3.14;
//
//   std::cout << " x " << x.format(FMT) << std::endl;
//   {
//     Model_unicycle1_se2 model_se2;
//
//     model_se2.step(xnext, x, u, dt);
//     std::cout << xnext.format(FMT) << std::endl;
//   }
//
//   {
//     Model_unicycle1_R2SO2 model_se2;
//     model_se2.step(xnext, x, u, dt);
//     std::cout << xnext.format(FMT) << std::endl;
//   }
//   {
//     Model_unicycle1 model;
//     model.step(xnext, x, u, dt);
//     std::cout << xnext.format(FMT) << std::endl;
//   }
// }

BOOST_AUTO_TEST_CASE(col_unicycle) {

  const char *env = "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";

  Problem problem;
  problem.read_from_yaml(env);

  auto unicycle = Model_unicycle1();
  load_env_quim(unicycle, problem);

  Eigen::Vector3d x(.7, .8, 0);

  CollisionOut col;

  unicycle.collision_distance(x, col);

  BOOST_CHECK(std::fabs(col.distance - .25) < 1e-7);

  x = Eigen::Vector3d(1.9, .3, 0);
  unicycle.collision_distance(x, col);

  BOOST_CHECK(std::fabs(col.distance - .3) < 1e-7);

  col.write(std::cout);

  x = Eigen::Vector3d(1.5, .3, .1);
  unicycle.collision_distance(x, col);
  col.write(std::cout);

  BOOST_CHECK(std::fabs(col.distance - (-0.11123)) < 1e-5);
}

// derivatives?

BOOST_AUTO_TEST_CASE(col_car_with_trailer) {

  const char *env =
      "../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml";
  auto car = Model_car_with_trailers();

  Problem problem;
  problem.read_from_yaml(env);

  load_env_quim(car, problem);

  Eigen::Vector4d x(3.4, 3, 3.14, 3.14);
  CollisionOut col;
  car.collision_distance(x, col);
  col.write(std::cout);

  x = Eigen::Vector4d(5.2, 3, 1.55, 1.55);
  car.collision_distance(x, col);
  col.write(std::cout);

  x = Eigen::Vector4d(3.6, 1, 1.55, 1.55);
  car.collision_distance(x, col);
  col.write(std::cout);

  x = Eigen::Vector4d(5.2, 3, 1.55, .3);
  car.collision_distance(x, col);
  col.write(std::cout);
}

BOOST_AUTO_TEST_CASE(col_quad3d) {

  const char *env = "../benchmark/quadrotor_0/quad_one_obs.yaml";
  Problem problem;
  problem.read_from_yaml(env);

  Eigen::VectorXd x(13);
  auto quad3d = Model_quad3d();
  load_env_quim(quad3d, problem);
  // quad3d.load_env_quim(problem);
  CollisionOut col;

  x << 1., 1., 1., 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
  quad3d.collision_distance(x, col);
  col.write(std::cout);

  BOOST_TEST(std::fabs(col.distance - 0.30713) < 1e-5);

  x << 1.2, 1.5, 2., 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
  quad3d.collision_distance(x, col);
  col.write(std::cout);

  BOOST_TEST(std::fabs(col.distance - (-0.0999999)) < 1e-5);
  x << 5., 5., 1., 0, 0, 0, 1, 0, 0, 0, 0, 0, 0;
  quad3d.collision_distance(x, col);
  col.write(std::cout);
  BOOST_TEST(std::fabs(col.distance - (0.307206)) < 1e-5);
}

BOOST_AUTO_TEST_CASE(col_acrobot) {

  Problem problem;
  const char *env = "../benchmark/acrobot/swing_up_obs.yaml";
  problem.read_from_yaml(env);

  Eigen::Vector4d x;
  auto acrobot = Model_acrobot();
  load_env_quim(acrobot, problem);

  CollisionOut col;
  x << 0, 0, 0, 0;
  acrobot.collision_distance(x, col);
  col.write(std::cout);
  double tol = 1e-5;

  BOOST_TEST(std::fabs(col.distance - 1.59138) < tol);

  x << 3.14159, 0, 0, 0;
  acrobot.collision_distance(x, col);
  col.write(std::cout);
  BOOST_TEST(std::fabs(col.distance - 1.1) < tol);

  x << M_PI / 2., M_PI / 2., 0, 0;
  acrobot.collision_distance(x, col);
  col.write(std::cout);
  BOOST_TEST(std::fabs(col.distance - 0.180278) < tol);
}

BOOST_AUTO_TEST_CASE(check_gradient_of_col) { // TODO
}

BOOST_AUTO_TEST_CASE(col_quad3d_v2) {

  Problem problem("../benchmark/quadrotor_0/obstacle_flight.yaml");

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path(problem.robotType).c_str());
  load_env_quim(*robot, problem);

  Eigen::VectorXd x(13);
  x.setZero();
  x.head(3) << -.1, -1.3, 1.;
  x(6) = 1.;

  CollisionOut out;
  robot->collision_distance(x, out);

  out.write(std::cout);
};

BOOST_AUTO_TEST_CASE(check_traj) {

  Problem problem("../benchmark/quadrotor_0/obstacle_flight.yaml");

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path(problem.robotType).c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  double u_nominal = robot_derived->u_nominal;
  {

    Trajectory traj_raw;

    traj_raw.read_from_yaml(
        "../data_welf_yaml/result_obstacle_flight_CASADI.yaml");

    std::cout << "Printing Traj " << std::endl;
    traj_raw.to_yaml_format(std::cout);

    std::cout << "*** PARAMS ***" << std::endl;
    robot->write_params(std::cout);
    std::cout << "*** PARAMS ***" << std::endl;

    Trajectory traj = from_welf_to_quim(traj_raw, u_nominal);

    traj.check(robot);
    traj.update_feasibility(3e-2);

    BOOST_TEST(traj.feasible);
  }

#if 1
  // {
  //
  //   Trajectory traj_raw;
  //
  //   traj_raw.read_from_yaml("traj_welf2.yaml");
  //
  //   std::cout << "Printing Traj " << std::endl;
  //   traj_raw.to_yaml_format(std::cout);
  //
  //   std::cout << "*** PARAMS ***" << std::endl;
  //   robot->write_params(std::cout);
  //   std::cout << "*** PARAMS ***" << std::endl;
  //
  //   Trajectory traj = from_welf_to_quim(traj_raw, u_nominal);
  //
  //   traj.check(robot);
  //   traj.update_feasibility(1e-2);
  //
  //   BOOST_TEST(traj.feasible);
  // }

  // now lets solve it!
  {
    std::cout << "solve using super good init guess" << std::endl;
    Trajectory traj_croco;
    Options_trajopt options_trajopt;
    Result_opti opti_out;

    Trajectory init_guess_raw;
    init_guess_raw.read_from_yaml(
        "../data_welf_yaml/result_obstacle_flight_CASADI.yaml");

    Trajectory traj_init_guess = from_welf_to_quim(init_guess_raw, u_nominal);

    {
      std::ofstream out_file("traj_welf_casadi_quimf.yaml");
      traj_init_guess.to_yaml_format(out_file);
    }

    trajectory_optimization(problem, traj_init_guess, options_trajopt,
                            traj_croco, opti_out);
    BOOST_TEST(traj_croco.feasible);
    BOOST_TEST(opti_out.feasible);

    {
      std::ofstream out_file("traj_welf_quimf.yaml");
      traj_croco.to_yaml_format(out_file);
    }

    Trajectory out_welf = from_quim_to_welf(traj_croco, u_nominal);

    out_welf.to_yaml_format(std::cout);

    std::ofstream out_file("traj_welf.yaml");
    out_welf.to_yaml_format(out_file);
  }
  {
    std::cout << "solve using the init guess from Welf" << std::endl;

    Trajectory traj_croco;
    Options_trajopt options_trajopt;
    Result_opti opti_out;

    Trajectory init_guess_raw;
    init_guess_raw.read_from_yaml(
        "../data_welf_yaml/guess_obstacle_flight.yaml");

    Trajectory traj_init_guess = from_welf_to_quim(init_guess_raw, u_nominal);
    trajectory_optimization(problem, traj_init_guess, options_trajopt,
                            traj_croco, opti_out);

    BOOST_TEST(traj_croco.feasible);
    BOOST_TEST(opti_out.feasible);

    Trajectory out_welf = from_quim_to_welf(traj_croco, u_nominal);

    {
      std::ofstream out_file("traj_welf2_quimf.yaml");
      traj_croco.to_yaml_format(out_file);
    }

    std::ofstream out_file("traj_welf2.yaml");
    out_welf.to_yaml_format(out_file);
  }
#endif
}

BOOST_AUTO_TEST_CASE(solve_obs) {}

BOOST_AUTO_TEST_CASE(check_car2_alex) {

  std::shared_ptr<Model_robot> car2 = std::make_shared<Model_car2>();

  auto result = "../test/car2/trajectory_alexander.yaml";

  std::cout << "loading file: " << result << std::endl;
  YAML::Node node = YAML::LoadFile(result);

  std::vector<Eigen::VectorXd> __xs = yaml_node_to_xs(node["states"]);
  std::vector<Eigen::VectorXd> __us = yaml_node_to_xs(node["actions"]);

  __us.pop_back();

  std::vector<Eigen::VectorXd> xs(__xs.size());

  std::transform(__xs.begin(), __xs.end(), xs.begin(), [](auto &v) {
    Eigen::VectorXd out(5);
    out = v.head(5);
    out(2) = wrap_angle(out(2));
    return out;
  });

  Trajectory traj;

  traj.states = xs;
  traj.actions = __us;

  traj.to_yaml_format(std::cout);

  traj.check(car2);
  traj.to_yaml_format(std::cout);
}

BOOST_AUTO_TEST_CASE(test_cli) {

  std::string cmd =
      "make croco_main && ./croco_main --solver_id 0  --env_file  "
      "../benchmark/acrobot/swing_up_empty.yaml   --init_guess "
      "../benchmark/acrobot/swing_up_empty_guess_db.yaml "
      "--max_iter 400  --out buu.yaml --weight_goal 1000";

  int out = std::system(cmd.c_str());
  BOOST_TEST(out == 0);
}

BOOST_AUTO_TEST_CASE(test_welf2) {

  Trajectory traj;
  traj.read_from_yaml("../data_welf/rollout.yaml");

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path("quad3d_v2").c_str());

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  // double u_nominal =
  // robot_derived->u_nominal = 1;

  double u_nominal = robot_derived->u_nominal;

  Trajectory traj_init_guess = from_welf_to_quim(traj, u_nominal);

  traj_init_guess.check(robot);

  {

    Trajectory for_welf_check = from_quim_to_welf(traj_init_guess, u_nominal);
    std::ofstream tmp("../data_welf/rollout_double_conversion.yaml");
    for_welf_check.to_yaml_format(tmp);

    Trajectory traj;
    traj.read_from_yaml("../data_welf/rollout_double_conversion.yaml");
    Trajectory traj_init_guess = from_welf_to_quim(traj, u_nominal);
    traj_init_guess.check(robot);
  }
}

BOOST_AUTO_TEST_CASE(test_collisions) {

  Trajectory traj_w, traj;
  traj_w.read_from_yaml("../data_welf/traj_welf2_from_quim.yaml");
  Problem problem("../benchmark/quadrotor_0/obstacle_flight.yaml");

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path("quad3d_v1").c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  traj = from_welf_to_quim(traj_w, robot_derived->u_nominal);

  traj.check(robot);

  std::vector<Eigen::Vector3d> centers{
      {-0.8, 0.4, 0.2},  {0., 0.6, 0.5}, {0.8, -0.3, 0.2},
      {-0.5, 0.2, 1.1},  {0., -0.4, 1.}, {0.5, 1., 0.8},
      {-0.8, -0.8, 1.1}, {0., 0.4, 1.5}, {0.8, 0., 1.8}};

  Eigen::VectorXd x = traj.states.at(40);

  std::cout << "manual check" << std::endl;
  for (auto &c : centers) {
    std::cout << (x.head(3) - c).norm() << std::endl;
  }

  CollisionOut out;

  robot->collision_distance(x, out);
  out.write(std::cout);
}

BOOST_AUTO_TEST_CASE(t_serialization) {

  Trajectory traj1, traj2;

  auto v1 = Eigen::Vector3d(0, 0, 1);
  auto v2 = Eigen::Vector3d(0, 1, 1);
  auto v3 = Eigen::Vector3d(1, 0, 1);

  auto a1 = Eigen::Vector2d(0, 0);
  auto a2 = Eigen::Vector2d(0, 1);

  traj1.states = std::vector<Eigen::VectorXd>{v1, v2, v3};
  traj1.actions = std::vector<Eigen::VectorXd>{a1, a2};

  std::filesystem::create_directory("/tmp/croco/");

  auto filename = "/tmp/croco/test_serialize.bin";

  traj1.save_file_boost(filename);

  traj2.load_file_boost(filename);

  BOOST_TEST(traj1.distance(traj2) < 1e-10);
  BOOST_TEST(traj2.distance(traj1) < 1e-10);

  Trajectories trajs_A{.data = {traj1, traj2}};
  Trajectories trajs_B{.data = {traj1, traj2}};

  auto filename_trajs = "/tmp/croco/test_serialize_trajs.bin";

  trajs_A.save_file_boost(filename_trajs);
  trajs_B.load_file_boost(filename_trajs);

  BOOST_TEST(trajs_A.data.at(0).distance(trajs_B.data.at(0)) < 1e-10);
  BOOST_TEST(trajs_A.data.at(1).distance(trajs_B.data.at(1)) < 1e-10);
}

BOOST_AUTO_TEST_CASE(t_welf_recovery_paper) {

  Problem problem("../benchmark/quadrotor_0/recovery_paper.yaml");

  Trajectory traj_w, traj_q, traj_oq, traj_ow;
  traj_w.read_from_yaml("../data_welf_yaml/guess_recovery_flight.yaml");
  // traj_w.read_from_yaml("../data_welf_yaml/guesses/guess_low_noise/"
  //                       "initial_guess_recovery_flight_7.yaml");
  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path("quad3d_v3").c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  traj_q = from_welf_to_quim(traj_w, robot_derived->u_nominal);

  {
    std::ofstream out("init_guess.yaml");
    traj_q.to_yaml_format(out);
  }

  Options_trajopt options_trajopt;
  options_trajopt.weight_goal = 400; // 200-400
  options_trajopt.max_iter = 200;
  options_trajopt.smooth_traj = true;
  options_trajopt.noise_level = 1e-7;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_q, options_trajopt, traj_oq, opti_out);
  BOOST_TEST(traj_oq.feasible);
  BOOST_TEST(opti_out.feasible);

  std::cout << "optimazation done" << std::endl;
  opti_out.write_yaml(std::cout);

  {
    std::ofstream tmp("../data_welf_yaml/croco_recovery_q.yaml");
    traj_oq.to_yaml_format(tmp);
  }

  traj_ow = from_quim_to_welf(traj_oq, robot_derived->u_nominal);
  std::ofstream tmp("../data_welf_yaml/croco_recovery.yaml");
  traj_ow.to_yaml_format(tmp);
}

BOOST_AUTO_TEST_CASE(t_welf_recovery) {

  Problem problem("../benchmark/quadrotor_0/recovery_paper.yaml");

  Trajectory traj_w, traj_q, traj_oq, traj_ow;
  traj_w.read_from_yaml("../data_welf_yaml/guess_recovery_flight.yaml");

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path("quad3d_v3").c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  traj_q = from_welf_to_quim(traj_w, robot_derived->u_nominal);

  Options_trajopt options_trajopt;
  options_trajopt.weight_goal = 400;
  options_trajopt.max_iter = 200;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_q, options_trajopt, traj_oq, opti_out);
  BOOST_TEST(traj_oq.feasible);
  BOOST_TEST(opti_out.feasible);

  std::cout << "optimazation done" << std::endl;
  opti_out.write_yaml(std::cout);

  {
    std::ofstream tmp("../data_welf_yaml/croco_recovery_q.yaml");
    traj_oq.to_yaml_format(tmp);
  }

  traj_ow = from_quim_to_welf(traj_oq, robot_derived->u_nominal);
  std::ofstream tmp("../data_welf_yaml/croco_recovery.yaml");
  traj_ow.to_yaml_format(tmp);
}

BOOST_AUTO_TEST_CASE(t_welf_simple) {

  Problem problem("../benchmark/quadrotor_0/simple_flight.yaml");
  Trajectory traj_w, traj_q, traj_oq, traj_ow;
  traj_w.read_from_yaml("../data_welf_yaml/guess_simple_flight.yaml");

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path("quad3d_v4").c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  traj_q = from_welf_to_quim(traj_w, robot_derived->u_nominal);

  Options_trajopt options_trajopt;
  options_trajopt.weight_goal = 800;
  options_trajopt.max_iter = 200;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_q, options_trajopt, traj_oq, opti_out);
  BOOST_TEST(traj_oq.feasible);
  BOOST_TEST(opti_out.feasible);

  std::cout << "optimazation done" << std::endl;
  opti_out.write_yaml(std::cout);

  {
    std::ofstream tmp("../data_welf_yaml/croco_simple_q.yaml");
    traj_oq.to_yaml_format(tmp);
  }

  traj_ow = from_quim_to_welf(traj_oq, robot_derived->u_nominal);
  std::ofstream tmp("../data_welf_yaml/croco_simple.yaml");
  traj_ow.to_yaml_format(tmp);
}

BOOST_AUTO_TEST_CASE(t_welf_flip) {

  // Problem problem("../benchmark/quadrotor_0/flip.yaml");
  Problem problem("../benchmark/quadrotor_0/flip.yaml");
  Trajectory traj_w, traj_q, traj_oq, traj_ow;
  traj_w.read_from_yaml("../data_welf_yaml/result_flip_CASADI.yaml");
  // traj_w.read_from_yaml("../data_welf_yaml/guess_flip.yaml");
  // result_flip_CASADI.yaml"); //

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path(problem.robotType).c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  traj_q = from_welf_to_quim(traj_w, robot_derived->u_nominal);

  traj_q.check(robot);

  {
    std::ofstream tmp("../data_welf_yaml/guess_flip_q.yaml");
    traj_q.to_yaml_format(tmp);
  }

  Options_trajopt options_trajopt;
  options_trajopt.weight_goal = 200;
  options_trajopt.max_iter = 200;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_q, options_trajopt, traj_oq, opti_out);
  BOOST_TEST(traj_oq.feasible);
  BOOST_TEST(opti_out.feasible);

  std::cout << "optimazation done" << std::endl;
  opti_out.write_yaml(std::cout);

  {
    std::ofstream tmp("../data_welf_yaml/croco_flip_q.yaml");
    traj_oq.to_yaml_format(tmp);
  }

  traj_ow = from_quim_to_welf(traj_oq, robot_derived->u_nominal);
  std::ofstream tmp("../data_welf_yaml/croco_flip.yaml");
  traj_ow.to_yaml_format(tmp);
}

BOOST_AUTO_TEST_CASE(t_welf_loop) {

  // Problem problem("../benchmark/quadrotor_0/flip.yaml");
  Problem problem("../benchmark/quadrotor_0/loop.yaml");
  Trajectory traj_w, traj_q, traj_oq, traj_ow;
  // traj_w.read_from_yaml("../data_welf_yaml/result_flip_CASADI.yaml");
  traj_w.read_from_yaml("../data_welf_yaml/guess_loop.yaml");
  // result_flip_CASADI.yaml"); //

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path(problem.robotType).c_str());
  load_env_quim(*robot, problem);

  std::shared_ptr<Model_quad3d> robot_derived =
      std::dynamic_pointer_cast<Model_quad3d>(robot);

  traj_q = from_welf_to_quim(traj_w, robot_derived->u_nominal);

  traj_q.check(robot);

  {
    std::ofstream tmp("../data_welf_yaml/guess_loop_q.yaml");
    traj_q.to_yaml_format(tmp);
  }

  Options_trajopt options_trajopt;
  options_trajopt.weight_goal = 200;
  options_trajopt.max_iter = 200;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_q, options_trajopt, traj_oq, opti_out);
  BOOST_TEST(traj_oq.feasible);
  BOOST_TEST(opti_out.feasible);

  std::cout << "optimazation done" << std::endl;
  opti_out.write_yaml(std::cout);

  {
    std::ofstream tmp("../data_welf_yaml/croco_loop_q.yaml");
    traj_oq.to_yaml_format(tmp);
  }

  traj_ow = from_quim_to_welf(traj_oq, robot_derived->u_nominal);
  std::ofstream tmp("../data_welf_yaml/croco_loop.yaml");
  traj_ow.to_yaml_format(tmp);
}

BOOST_AUTO_TEST_CASE(load_my) {

  Problem problem("../benchmark/quadrotor_0/flip.yaml");
  Trajectory traj_w, traj_q, traj_oq, traj_ow;
  traj_w.read_from_yaml("../data_welf_yaml/croco_flip_q.yaml");
  // traj_w.read_from_yaml("../data_welf_yaml/guess_flip.yaml");
  // result_flip_CASADI.yaml"); //

  for (auto &s : traj_w.states) {
    s.segment(3, 4).normalize();
  }

  std::shared_ptr<Model_robot> robot =
      robot_factory(robot_type_to_path(problem.robotType).c_str());
  load_env_quim(*robot, problem);
  traj_w.check(robot);
}

BOOST_AUTO_TEST_CASE(t_data_for_alex) {

  CSTR_("hello wordld");
  auto file = "../data_alex/starting_config.yaml";
  YAML::Node node = YAML::LoadFile(file);

  // x,y,theat,v,phi,vg,phig
  std::vector<Eigen::VectorXd> alex_starts;

  for (const auto &n : node["start_configs"]) {
    auto v = n.as<std::vector<double>>();
    alex_starts.push_back(Eigen::VectorXd::Map(v.data(), v.size()));
  }

  const char *dynamics = "car2_v0";

  std::vector<Eigen::VectorXd> starts(alex_starts.size());
  std::vector<Eigen::VectorXd> goals(alex_starts.size());

  std::transform(alex_starts.begin(), alex_starts.end(), starts.begin(),
                 [](auto &v) {
                   Eigen::VectorXd out(5);
                   out = v.head(5);
                   out(2) = wrap_angle(out(2));
                   return out;
                 });

  std::transform(alex_starts.begin(), alex_starts.end(), goals.begin(),
                 [](auto &v) {
                   Eigen::VectorXd out(5);
                   out.setZero();
                   out.tail(2) = v.tail(2);
                   return out;
                 });

  int ref_time_steps = 100;
  Options_trajopt options_trajopt;
  Trajectories trajectories, trajs_opt;

  CSTR_(starts.size());
  // solve the problem
  for (size_t i = 0; i < starts.size(); i++) {
    Eigen::VectorXd &goal = goals.at(i);
    Eigen::VectorXd &start = starts.at(i);

    CSTR_V(start);
    CSTR_V(goal);

    Problem problem;
    problem.goal = goal;
    problem.start = start;
    problem.robotType = dynamics;

    Trajectory init_guess;
    init_guess.num_time_steps = int(ref_time_steps);

    Trajectory traj;
    Result_opti opti_out;

    trajectory_optimization(problem, init_guess, options_trajopt, traj,
                            opti_out);
    if (opti_out.feasible) {
      CHECK(traj.states.size(), AT);
      traj.start = traj.states.front();
      traj.goal = traj.states.back();
      trajectories.data.push_back(traj);
    }
  }

  trajectories.save_file_yaml("car2_v0_alex_100.yaml");
  CSTR_(trajectories.data.size());

  options_trajopt.solver_id = 14;
  Options_primitives options_primitives;
  improve_motion_primitives(options_trajopt, trajectories, dynamics, trajs_opt,
                            options_primitives);

  trajs_opt.save_file_yaml("car2_v0_alex_opt.yaml");
  CSTR_(trajs_opt.data.size());
}

BOOST_AUTO_TEST_CASE(check_all_init_guess) {

  //

  // load init guess

  namespace fs = std::filesystem;

  auto path = "../data_welf_yaml/guesses/guess_low_noise/";
  // auto path = "../data_welf_yaml/guesses/guess_middle_noise/";
  // auto path = "../data_welf_yaml/guesses/guess_high_noise/";
  // guess_middle_noise/";
  // guess_low_noise/";

  std::vector<std::string> paths;

  Problem problem("../benchmark/quadrotor_0/recovery_paper.yaml");

  for (const auto &entry : fs::directory_iterator(path)) {
    paths.push_back(entry.path());
  }

  Options_trajopt options_trajopt;
  options_trajopt.welf_format = true;
  options_trajopt.max_iter = 350;
  options_trajopt.weight_goal = 300;
  options_trajopt.smooth_traj =
      true; // important: set to true, because the guess are bad!

  size_t solved = 0;
  size_t trials = 0;

  for (const auto &p : paths) {

    Trajectory traj_in, traj_out;
    Result_opti opti_out;
    std::cout << "using init guess " << std::endl;
    traj_in.read_from_yaml(p.c_str());
    trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                            opti_out);
    trials++;
    if (traj_out.feasible) {
      solved++;
    }
  }

  CSTR_(solved);
  CSTR_(trials);
  CSTR_(double(solved) / trials);
}

BOOST_AUTO_TEST_CASE(t_diff_angle) {
  // TODO
}

BOOST_AUTO_TEST_CASE(t_alexander_v2) {

  std::shared_ptr<Model_robot> car2 =
      std::make_shared<Model_car2>("../models/car2_v0.yaml");

  auto result =
      "/home/quim/Downloads/trajectory_config_1681304908.0324593.yaml";

  std::cout << "loading file: " << result << std::endl;
  YAML::Node node = YAML::LoadFile(result);

  std::vector<Eigen::VectorXd> __xs = yaml_node_to_xs(node["states"]);
  std::vector<Eigen::VectorXd> __us = yaml_node_to_xs(node["actions"]);

  __us.pop_back();

  std::vector<Eigen::VectorXd> xs(__xs.size());

  std::transform(__xs.begin(), __xs.end(), xs.begin(), [](auto &v) {
    Eigen::VectorXd out(5);
    out = v.head(5);
    out(2) = wrap_angle(out(2));
    return out;
  });

  Trajectory traj;

  traj.states = xs;
  traj.actions = __us;

  traj.to_yaml_format(std::cout);
  traj.check(car2);

  // not matching exactly because he clips one of the quantities

  {
    std::cout << "SOLVING for pose " << std::endl;

    auto file = "/home/quim/Downloads/starting_poses_20_steps.yaml";

    YAML::Node node = YAML::LoadFile(file);

    // x,y,theat,v,phi,vg,phig
    std::vector<Eigen::VectorXd> alex_starts;

    for (const auto &n : node["training_set"]) {
      auto v = n.as<std::vector<double>>();
      alex_starts.push_back(Eigen::VectorXd::Map(v.data(), v.size()));
    }

    const char *dynamics = "car2_v0";

    std::vector<Eigen::VectorXd> starts(alex_starts.size());
    std::vector<Eigen::VectorXd> goals(alex_starts.size());

    std::transform(alex_starts.begin(), alex_starts.end(), starts.begin(),
                   [](auto &v) {
                     Eigen::VectorXd out(5);
                     out = v.head(5);
                     out(2) = wrap_angle(out(2));
                     return out;
                   });

    std::transform(alex_starts.begin(), alex_starts.end(), goals.begin(),
                   [](auto &v) {
                     Eigen::VectorXd out(5);
                     out.setZero();
                     out.tail(2) = v.tail(2);
                     return out;
                   });

    int ref_time_steps = 20;
    Options_trajopt options_trajopt;
    Trajectories trajectories, trajs_opt;

    CSTR_(starts.size());
    // solve the problem
    for (size_t i = 0; i < starts.size(); i++) {
      Eigen::VectorXd &goal = goals.at(i);
      Eigen::VectorXd &start = starts.at(i);

      CSTR_V(start);
      CSTR_V(goal);

      Problem problem;
      problem.goal = goal;
      problem.start = start;
      problem.robotType = dynamics;

      Trajectory init_guess;
      init_guess.num_time_steps = int(ref_time_steps);

      Trajectory traj;
      Result_opti opti_out;

      trajectory_optimization(problem, init_guess, options_trajopt, traj,
                              opti_out);
      if (opti_out.feasible) {
        CHECK(traj.states.size(), AT);
        traj.start = traj.states.front();
        traj.goal = traj.states.back();
        trajectories.data.push_back(traj);
      }
    }

    trajectories.save_file_yaml("car2_v0_starting_poses_20.yaml");
    CSTR_(trajectories.data.size());
    CSTR_(starts.size());
  }
}

BOOST_AUTO_TEST_CASE(alex_3) {

  std::vector<std::string> files;
  auto in_folder = "../src/alex/trajectories/";
  for (const auto &entry : std::filesystem::directory_iterator(in_folder)) {
    if (entry.is_regular_file())
      files.push_back(entry.path());
  }

  std::shared_ptr<Model_robot> car2 =
      std::make_shared<Model_car2>("../models/car2_v0.yaml");

  for (auto &f : files) {

    YAML::Node node = YAML::LoadFile(f);

    std::vector<Eigen::VectorXd> __xs = yaml_node_to_xs(node["states"]);
    std::vector<Eigen::VectorXd> __us = yaml_node_to_xs(node["actions"]);

    __us.pop_back();

    std::vector<Eigen::VectorXd> xs(__xs.size());

    std::transform(__xs.begin(), __xs.end(), xs.begin(), [](auto &v) {
      Eigen::VectorXd out(5);
      out = v.head(5);
      out(2) = wrap_angle(out(2));
      return out;
    });

    Trajectory traj;

    traj.states = xs;
    traj.actions = __us;

    traj.start = traj.states.front();
    traj.goal = traj.states.back();

    traj.to_yaml_format(std::cout);
    traj.check(car2);
  }
}

BOOST_AUTO_TEST_CASE(test_acrobot_mpcc) {

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../unittest/traj_db_acrobot_swingup.yaml");
  Problem problem("../benchmark/acrobot/swing_up_empty.yaml");
  std::shared_ptr<Model_robot> robot =
      std::make_shared<Model_acrobot>("../models/acrobot_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(robot, true);

  Options_trajopt options_trajopt;
  // options_trajopt.x_b
  options_trajopt.solver_id = 8;
  options_trajopt.window_optimize = 300;
  options_trajopt.window_shift = 100;
  options_trajopt.max_iter = 100;
  options_trajopt.weight_goal = 300;
  options_trajopt.smooth_traj = true;
  options_trajopt.k_linear = 10;
  options_trajopt.k_contour = 10;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);
}

BOOST_AUTO_TEST_CASE(test_quad2d_recovery) {

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../unittest/traj_db_quad2d_recovery.yaml");
  Problem problem("../benchmark/quad2d/quad2d_recovery_wo_obs.yaml");

  std::shared_ptr<Model_robot> robot =
      std::make_shared<Model_quad2d>("../models/quad2d_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(robot, true);

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 8;
  options_trajopt.window_optimize = 300;
  options_trajopt.window_shift = 100;
  options_trajopt.max_iter = 100;
  options_trajopt.weight_goal = 200;
  options_trajopt.smooth_traj = true;
  options_trajopt.k_linear = 100;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);
}

BOOST_AUTO_TEST_CASE(test_acrobot_db) {

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../build/traj_db.yaml");
  Problem problem("../benchmark/acrobot/swing_up_empty.yaml");

  std::shared_ptr<Model_robot> acrobot =
      std::make_shared<Model_acrobot>("../models/acrobot_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(acrobot, true);

  Options_trajopt options_trajopt;
  options_trajopt.max_iter = 100;
  options_trajopt.weight_goal = 1000;
  options_trajopt.smooth_traj = true;
  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(test_quad3d_mpcc) {

  // the init guess is good. why the mpcc produces strange trajectory?

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../build/traj_db_fwduGj.yaml");
  Problem problem("../benchmark/quadrotor_0/empty_0_easy.yaml");

  for (auto &s : traj_in.states) {
    s.segment<4>(3).normalize();
  }
  CSTR_(traj_in.states.size());

  std::shared_ptr<Model_robot> acrobot =
      std::make_shared<Model_quad3d>("../models/quad3d_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(acrobot, true);

  Options_trajopt options_trajopt;
  options_trajopt.max_iter = 100;
  options_trajopt.weight_goal = 500;
  options_trajopt.smooth_traj = true;
  options_trajopt.solver_id = 8;
  options_trajopt.window_optimize = 100;
  options_trajopt.window_shift = 50;
  options_trajopt.k_linear = 100;
  options_trajopt.noise_level = 1e-8;
  options_trajopt.max_mpc_iterations = 10;
  // options_trajopt.k_contour = 50;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(test_quad3d_mpcc_oneobs) {

  // TODO: not working.
  // TODO: issue with distance with quaternions!!
  // maybe use smaller delta?
  // or scaled distance function? -- decide if I should use everywhere

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../unittest/traj_db_quad3d_oneobs.yaml");
  Problem problem("../benchmark/quadrotor_0/quad_one_obs.yaml");

  for (auto &s : traj_in.states) {
    s.segment<4>(3).normalize();
  }
  CSTR_(traj_in.states.size());

  std::shared_ptr<Model_robot> acrobot =
      std::make_shared<Model_quad3d>("../models/quad3d_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(acrobot, true);

  Options_trajopt options_trajopt;
  options_trajopt.max_iter = 100;
  options_trajopt.weight_goal = 300;
  options_trajopt.smooth_traj = true;
  options_trajopt.solver_id = 8;
  options_trajopt.window_optimize = 100;
  options_trajopt.window_shift = 50;
  options_trajopt.k_linear = 50;
  options_trajopt.k_contour = 10;
  options_trajopt.noise_level = 1e-8;
  options_trajopt.max_mpc_iterations = 20;
  // options_trajopt.k_contour = 50;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(test_quad3d_recovery_mpcc) {

  // TODO: not working.
  // TODO: issue with distance with quaternions!!
  // maybe use smaller delta?
  // or scaled distance function? -- decide if I should use everywhere

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../unittest/traj_db_recovery.yaml");
  Problem problem("../benchmark/quadrotor_0/recovery.yaml");

  for (auto &s : traj_in.states) {
    s.segment<4>(3).normalize();
  }
  CSTR_(traj_in.states.size());

  std::shared_ptr<Model_robot> acrobot =
      std::make_shared<Model_quad3d>("../models/quad3d_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(acrobot, true);

  Options_trajopt options_trajopt;
  options_trajopt.max_iter = 100;
  options_trajopt.weight_goal = 300;
  options_trajopt.smooth_traj = true;
  options_trajopt.solver_id = 8;
  options_trajopt.window_optimize = 100;
  options_trajopt.window_shift = 50;
  options_trajopt.k_linear = 50;
  options_trajopt.k_contour = 10;
  options_trajopt.noise_level = 1e-8;
  options_trajopt.max_mpc_iterations = 20;
  // options_trajopt.k_contour = 50;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(test_quad2d_recovery_new) {

  // Trajectory traj_in, traj_out;
  // traj_in.read_from_yaml("../unittest/traj_db_quad2d_recovery.yaml");

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../unittest/traj_db_quad2d_recovery_2.yaml");

  Problem problem("../benchmark/quad2d/quad2d_recovery_wo_obs.yaml");

  std::shared_ptr<Model_robot> robot =
      std::make_shared<Model_quad2d>("../models/quad2d_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(robot, true);

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 0;
  options_trajopt.max_iter = 200;
  options_trajopt.weight_goal = 200;
  options_trajopt.smooth_traj = true;
  options_trajopt.linear_search = true;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(uni1_mpc) {

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../unittest/traj_db_uni1_bugtrap.yaml");
  Problem problem("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");

  std::shared_ptr<Model_robot> robot =
      std::make_shared<Model_unicycle1>("../models/unicycle1_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(robot, true);

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 10;
  options_trajopt.smooth_traj = true;
  // smooth_traj
  options_trajopt.window_optimize = 40;
  options_trajopt.window_shift = 20;
  options_trajopt.max_iter = 20;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(uni2_mpc) {

  Trajectory traj_in, traj_out;
  traj_in.read_from_yaml("../build/trajdb_zHo6Qf.yaml");
  Problem problem("../benchmark/unicycle_second_order_0/bugtrap_0.yaml");

  std::shared_ptr<Model_robot> robot =
      std::make_shared<Model_unicycle2>("../models/unicycle2_v0.yaml");

  traj_in.start = problem.start;
  traj_in.goal = problem.goal;
  traj_in.check(robot, true);

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 10;
  options_trajopt.smooth_traj = true;
  // smooth_traj
  options_trajopt.window_optimize = 40;
  options_trajopt.window_shift = 15;
  options_trajopt.max_iter = 20;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(quadpole2d_dyn) {
  ptr<Dynamics> dyn = mk<Dynamics>(mks<Model_quad2dpole>());
  check_dyn(dyn, 1e-6, Eigen::VectorXd(), Eigen::VectorXd(), 200);
  // NICE!! :)
}

BOOST_AUTO_TEST_CASE(quadpole2d_problem) {

  Problem problem("../benchmark/quad2dpole/empty_0.yaml");

  Trajectory traj_in, traj_out;
  traj_in.num_time_steps = 300;

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 0;
  options_trajopt.smooth_traj = false;
  options_trajopt.weight_goal = 200;
  options_trajopt.max_iter = 100;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(quadpole2d_problem_hover) {

  Problem problem("../benchmark/quad2dpole/hover_up_0.yaml");

  Trajectory traj_in, traj_out;
  traj_in.num_time_steps = 300;

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 0;
  options_trajopt.smooth_traj = false;
  options_trajopt.weight_goal = 200;
  options_trajopt.max_iter = 100;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}

BOOST_AUTO_TEST_CASE(quadpole2d_problem_hard) {

  Problem problem("../benchmark/quad2dpole/up.yaml");

  Trajectory traj_in, traj_out;
  traj_in.num_time_steps = 500;

  Options_trajopt options_trajopt;
  options_trajopt.solver_id = 0;
  options_trajopt.smooth_traj = false;
  options_trajopt.weight_goal = 100;
  options_trajopt.max_iter = 10000;
  options_trajopt.noise_level = 1e-2;
  // options_trajopt.use_finite_diff = true;

  Result_opti opti_out;

  trajectory_optimization(problem, traj_in, options_trajopt, traj_out,
                          opti_out);

  {
    std::ofstream out("out_opt.yaml");
    traj_out.to_yaml_format(out);
  }
}
