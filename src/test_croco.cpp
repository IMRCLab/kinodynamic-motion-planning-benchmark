#include "robot_models.hpp"
#include <algorithm>
#include <cmath>
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

#include "ocp.hpp"
#include "croco_models.hpp"
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
  int T = 100; // one second
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
  int T = 1. / dt; // one second

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

  // add some test?
}

#if 0
BOOST_AUTO_TEST_CASE(traj_opt_no_bounds) {

  opti_params = Opti_params();

  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_bugtrap_0_sol0.yaml";

  opti_params.control_bounds = 0;
  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
  opti_params.use_warmstart = 1;
  opti_params.max_iter = 50;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST(result.feasible);
}
#endif

BOOST_AUTO_TEST_CASE(jac_quadrotor) {

  opti_params = Opti_params();
  opti_params.disturbance = 1e-7;
  size_t nx(13), nu(4);
  size_t N = 5;


  {

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

    auto model_robot = mks<Model_quad3d>();

    Generate_params gen_args;
    gen_args.name = "quadrotor_0";
    gen_args.N = N;
    opti_params.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;

    auto problem = generate_problem(gen_args, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    opti_params.use_finite_diff = 1;
    auto problem_diff = generate_problem(gen_args, nx, nu);
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

  opti_params = Opti_params();
  opti_params.disturbance = 1e-5;
  size_t nx(0), nu(0);

  size_t N = 5;
  Eigen::VectorXd alpha_refs = Eigen::VectorXd::Random(N + 1);
  Eigen::VectorXd cost_alpha_multis = Eigen::VectorXd::Random(N + 1);

  std::vector<Eigen::VectorXd> path;
  path.push_back(Eigen::VectorXd::Random(3));
  path.push_back(Eigen::VectorXd::Random(3));

  ptr<Interpolator> interpolator =
      mk<Interpolator>(Eigen::Vector2d({-2., 3.}), path);



  std::shared_ptr<Model_robot> model_robot = mks<Model_unicycle1>();
  {

    Eigen::VectorXd goal = Eigen::VectorXd::Random(3);
    Eigen::VectorXd start = Eigen::VectorXd::Random(4);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(4));
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(3));

    Generate_params gen_args;
    gen_args.name = "unicycle1_v0";
    gen_args.N = 5;
    opti_params.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    // gen_args.cl = cl;
    gen_args.model_robot = model_robot;
    gen_args.contour_control = true;
    gen_args.interpolator = interpolator;

    auto problem = generate_problem(gen_args, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    opti_params.use_finite_diff = 1;
    auto problem_diff = generate_problem(gen_args, nx, nu);
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

    BOOST_CHECK(
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

      BOOST_CHECK(check_equal(d_diff->Lxx, d->Lxx, tol, tol));
      BOOST_CHECK(check_equal(d_diff->Lxu, d->Lxu, tol, tol));
      // BOOST_CHECK(check_equal(d_diff->Luu, d->Luu, tol, tol)); //NOTE: it
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
    opti_params.use_finite_diff = false;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;
    gen_args.contour_control = false;

    auto problem = generate_problem(gen_args, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    opti_params.use_finite_diff = true;
    auto problem_diff = generate_problem(gen_args, nx, nu);
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
    opti_params.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.model_robot = model_robot;
    gen_args.interpolator = interpolator;
    gen_args.contour_control = true;
    gen_args.linear_contour = true;

    auto problem = generate_problem(gen_args, nx, nu);
    problem->calc(xs, us);
    problem->calcDiff(xs, us);
    auto data_running = problem->get_runningDatas();
    auto data_terminal = problem->get_terminalData();

    // now with finite diff
    opti_params.use_finite_diff = 1;
    auto problem_diff = generate_problem(gen_args, nx, nu);
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

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

  opti_params.control_bounds = 1;
  opti_params.use_finite_diff = 0;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 10.;

  opti_params.use_warmstart = true;
  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.max_iter = 50;
  opti_params.weight_goal = 100;

  opti_params.window_shift = 10;
  opti_params.window_optimize = 40;
  opti_params.smooth_traj = 1;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST(result.feasible);
}

BOOST_AUTO_TEST_CASE(contour_park_easy) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../build_debug/smooth_park_debug.yaml";

  opti_params.control_bounds = true;
  opti_params.use_finite_diff = false;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 2.;
  opti_params.weight_goal = 300;

  opti_params.use_warmstart = true;
  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.max_iter = 20;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST(result.feasible);

  std::string filename = "out.yaml";
  std::cout << "writing results to:" << filename << std::endl;
  std::ofstream file_out(filename);
  result.write_yaml_db(file_out);
}

BOOST_AUTO_TEST_CASE(parallel_small_step_good_init_guess) {
  opti_params = Opti_params();

  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_parallelpark_mpcc.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 2;
  opti_params.window_optimize = 35;
  opti_params.smooth_traj = 1;
  opti_params.k_linear = 20.;
  opti_params.k_contour = 10.;
  opti_params.weight_goal = 200;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  // NOTE cost should be 3.2
  BOOST_TEST_CHECK(result.cost <= 3.3);
}


BOOST_AUTO_TEST_CASE(bugtrap_so2_hard) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/guess_bugtrap_0_sol1.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 10;
  opti_params.window_optimize = 40;
  opti_params.smooth_traj = 0;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 10.;
  opti_params.weight_goal = 200;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 27.91);
}


BOOST_AUTO_TEST_CASE(bugtrap_so2_easy) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 10;
  opti_params.window_optimize = 40;
  opti_params.smooth_traj = 1;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 10.;
  opti_params.weight_goal = 200;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 24.2);
}





BOOST_AUTO_TEST_CASE(bugtrap_bad_init_guess) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_bugtrap_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 10;
  opti_params.window_optimize = 40;
  opti_params.smooth_traj = 1;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 10.;
  opti_params.weight_goal = 200;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 23.0);
}

BOOST_AUTO_TEST_CASE(bugtrap_good_init_guess) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_bugtrap_0_mpcc.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 10;
  opti_params.window_optimize = 40;
  opti_params.smooth_traj = 0;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 10.;
  opti_params.weight_goal = 200;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  BOOST_TEST_CHECK(result.cost <= 23.0);
}

BOOST_AUTO_TEST_CASE(parallel_free_time) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt_free_time);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 3.3);
}

BOOST_AUTO_TEST_CASE(parallel_bad_init_guess) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 5;
  opti_params.window_optimize = 30;
  opti_params.smooth_traj = 1;
  opti_params.k_linear = 5.;
  opti_params.k_contour = 10.;
  opti_params.weight_goal = 200;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 3.6);
}

BOOST_AUTO_TEST_CASE(parallel_search_time) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

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

  opti_params.solver_id = static_cast<int>(SOLVER::time_search_traj_opt);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 3.3);
}

BOOST_AUTO_TEST_CASE(bugtrap_bad_mpc_adaptative) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "bugtrap_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_bugtrap_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpc_adaptative);

  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 20;
  opti_params.window_optimize = 50;
  opti_params.weight_goal = 100;
  opti_params.noise_level = 0.;
  opti_params.smooth_traj = true;
  opti_params.shift_repeat = false;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 27.);
}

BOOST_AUTO_TEST_CASE(kink_mpcc) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                        "kink_0.yaml";
  file_inout.init_guess = "../test/unicycle_first_order_0/"
                          "guess_kink_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.window_shift = 20;
  opti_params.window_optimize = 40;
  opti_params.weight_goal = 100;
  opti_params.k_linear = 20;
  opti_params.k_contour = 10;
  opti_params.smooth_traj = 1;
  opti_params.max_iter = 30;
  opti_params.shift_repeat = false;

  Result_opti result;
  compound_solvers(file_inout, result);
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

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_second_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_second_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.weight_goal = 100;
  opti_params.max_iter = 50;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 10.);
}

BOOST_AUTO_TEST_CASE(second_order_park_time) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_second_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_second_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt_free_time);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.weight_goal = 100;
  opti_params.max_iter = 50;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 11.);
}

BOOST_AUTO_TEST_CASE(quadrotor_0_recovery) {

  opti_params = Opti_params();

  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/quadrotor_0/"
                        "empty_test_recovery_welf.yaml";

  // auto model_robot = mks<Model_quad3d>();
  opti_params.solver_id = 0;
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = 1;
  opti_params.weight_goal = 300;
  opti_params.max_iter = 400;
  opti_params.noise_level = 1e-3;
  file_inout.T = 300;
  opti_params.ref_x0 = 1;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 5.);
}

BOOST_AUTO_TEST_CASE(t_recovery_with_smooth) {

  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/quadrotor_0/"
                        "empty_test_recovery_welf.yaml";
  file_inout.T = 400;

  Opti_params opti;
  opti.solver_id = 12;
  opti.control_bounds = 1;
  opti.use_warmstart = 1;
  opti.weight_goal = 100;
  opti.max_iter = 2000;
  opti.noise_level = 1e-4;
  opti.ref_x0 = true;
  opti.smooth_traj = 1;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 4.);

  //     ./croco_main   --env
  //     ../benchmark/quadrotor_0/empty_test_recovery_welf.yaml
  //     --out out.yaml --solver_id 12
  //     --control_bounds 1 --use_warmstart 1
  //     --weight_goal 100. --max_iter 2000 --
  // noise_level 1e-4 --use_finite_diff 0 --T
  // 400 --ref_x0 1 --smooth_traj 1
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

  opti_params = Opti_params();
  File_parser_inout file_inout;
  bool half = true;

  if (!half) {
    file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                          "check_region_of_attraction.yaml";
    file_inout.T = 40;
  } else {
    file_inout.env_file = "../benchmark/unicycle_first_order_0/"
                          "check_region_of_attraction_half."
                          "yaml";
    file_inout.T = 30;
  }

  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = true;
  opti_params.weight_goal = 200;
  opti_params.smooth_traj = 1;
  opti_params.max_iter = 40;

  Result_opti result;
  CSTR_(file_inout.name);
  compound_solvers(file_inout, result);
  BOOST_TEST_CHECK(result.feasible);
  std::cout << "cost is " << result.cost << std::endl;
  BOOST_TEST_CHECK(result.cost <= 40.);

  std::ofstream ref_out("ref_out-h" + std::to_string(half) + ".yaml");
  result.write_yaml_db(ref_out);

  // # controller
  size_t num_samples = 100;

  double noise_magnitude = .6;

  Eigen::VectorXd start = file_inout.start;

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
      file_inout.xs = result.xs_out;
      file_inout.us = result.us_out;
      file_inout.start = new_start;
      opti_params.max_iter = max_iter;
      opti_params.th_acceptnegstep = 2.;
      opti_params.smooth_traj = false;
      solve_with_custom_solver(file_inout, resulti);

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
    std::ofstream out("mpc_attraction-" + std::to_string(opti_params.max_iter) +
                      "-h" + std::to_string(half) + ".yaml");
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

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_second_order_0/"
                        "parallelpark_0.yaml";
  file_inout.init_guess = "../test/unicycle_second_order_0/"
                          "guess_parallelpark_0_sol0.yaml";

  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.control_bounds = 1;
  opti_params.use_warmstart = true;
  opti_params.k_linear = 50;
  opti_params.k_contour = 100;
  opti_params.weight_goal = 200;
  opti_params.smooth_traj = 1;
  opti_params.max_iter = 30;
  opti_params.window_optimize = 40;
  opti_params.window_shift = 20;
  opti_params.shift_repeat = 0;

  Result_opti result;
  compound_solvers(file_inout, result);
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

  auto unicycle = Model_unicycle1();
  unicycle.load_env_quim(env);
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
  car.load_env_quim(env);

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

  Eigen::VectorXd x(13);
  auto quad3d = Model_quad3d();
  quad3d.load_env_quim(env);
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

  const char *env = "../benchmark/acrobot/swing_up_obs.yaml";

  Eigen::Vector4d x;
  auto acrobot = Model_acrobot();
  acrobot.load_env_quim(env);

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

BOOST_AUTO_TEST_CASE(check_traj) {

  // const char *env;
  const char *result = "../data_welf_yaml/result_obstacle_flight_SCVX.yaml";
  const char *robot_params = "../models/quad3d_v1.yaml";

  std::cout << "loading file: " << result << std::endl;
  YAML::Node node = YAML::LoadFile(result);

  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<Eigen::VectorXd> __xs =
      yaml_node_to_xs(node["result"][0]["states"]);
  std::vector<Eigen::VectorXd> __us =
      yaml_node_to_xs(node["result"][0]["actions"]);

  Quad3d_params quad_params;

  std::cout << "quad parameters -- default " << std::endl;
  quad_params.write(std::cout);

  quad_params.read_from_yaml(robot_params);

  std::cout << "quad parameters " << std::endl;
  quad_params.write(std::cout);

  auto model_robot = mks<Model_quad3d>(quad_params);

  model_robot->u_nominal = 1;

  auto dynamics = mk<Dynamics>(model_robot);

  xs.resize(__xs.size());
  us.resize(__us.size());

  assert(__xs.size() == __us.size() + 1);

  assert(__xs.size());
  Eigen::VectorXd tmp(__xs.front().size());

  std::transform(__xs.begin(), __xs.end(), xs.begin(), [&](auto &x) {
    from_welf_format(x, tmp);
    tmp.segment(3, 4).normalize();
    return tmp;
  });

  assert(__us.size());
  Eigen::VectorXd u_tmp(__us.front().size());
  std::transform(__us.begin(), __us.end(), us.begin(), [&](auto &x) {
    u_tmp = x / model_robot->u_nominal;
    return u_tmp;
  });

  Eigen::VectorXd dts(us.size());
  dts.setConstant(model_robot->ref_dt);

  bool flag = check_trajectory(xs, us, dts, model_robot, 1e-2);

  std::cout << "flag is " << flag << std::endl;
  BOOST_TEST(flag);

  // check
}
