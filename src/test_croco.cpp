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

#include "collision_checker.hpp"
#include "croco_macros.hpp"

// save data without the cluster stuff

#include <filesystem>
#include <random>
#include <regex>
#include <type_traits>

#include <filesystem>
#include <regex>

#include "ocp.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <unsupported/Eigen/AutoDiff>

using namespace std;

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

BOOST_AUTO_TEST_CASE(quad2d) {
  {
    auto dyn = mk<Dynamics_quadcopter2d>();
    dyn->drag_against_vel = false;
    check_dyn(dyn, 1e-5);
    auto dyn_free_time = mk<Dynamics_quadcopter2d>(true);
    dyn_free_time->drag_against_vel = false;
    check_dyn(dyn_free_time, 1e-5);
  }

  // now with drag

  {
    auto dyn = mk<Dynamics_quadcopter2d>();
    dyn->drag_against_vel = true;
    check_dyn(dyn, 1e-5);
    auto dyn_free_time = mk<Dynamics_quadcopter2d>(true);
    dyn_free_time->drag_against_vel = true;
    check_dyn(dyn_free_time, 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(car_trailer) {
  {
    std::cout << "without trailers " << std::endl;
    ptr<Dynamics> dyn = mk<Dynamics_car_with_trailers>();
    check_dyn(dyn, 1e-5);
    ptr<Dynamics> dyn_free_time =
        mk<Dynamics_car_with_trailers>(Eigen::VectorXd(), true);
    check_dyn(dyn_free_time, 1e-5);
  }

  {
    std::cout << "with 1 trailer" << std::endl;
    Eigen::VectorXd c(1);
    c << .5;
    ptr<Dynamics> dyn = mk<Dynamics_car_with_trailers>(c, false);
    check_dyn(dyn, 1e-5);

    ptr<Dynamics> dyn_free_time = mk<Dynamics_car_with_trailers>(c, true);
    check_dyn(dyn_free_time, 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(t_qintegrate) {

  Eigen::Quaterniond q = Eigen::Quaterniond(0, 0, 0, 1);
  double dt = .01;
  Eigen::Vector3d omega{0, 0, 1};
  auto out = qintegrate(q, omega, dt);

  std::cout << "out\n" << out.coeffs() << std::endl;

  Eigen::MatrixXd JqD(4, 4);
  double eps = 1e-6;
  for (size_t i = 0; i < 4; i++) {
    Eigen::Vector4d qe;
    // Eigen::Vector3d ye;
    qe = q.coeffs();
    qe(i) += eps;
    // qe.normalize();
    auto ye = qintegrate(Eigen::Quaterniond(qe), omega, dt);
    auto df = (ye.coeffs() - out.coeffs()) / eps;
    JqD.col(i) = df;
  }

  Eigen::MatrixXd JomegaD(4, 3);
  for (size_t i = 0; i < 3; i++) {
    Eigen::Vector3d omegae;
    omegae = omega;
    omegae(i) += eps;
    auto ye = qintegrate(q, omegae, dt);
    auto df = (ye.coeffs() - out.coeffs()) / eps;
    JomegaD.col(i) = df;
  }

  std::cout << "omega" << std::endl;
  std::cout << JomegaD << std::endl;
  std::cout << "q" << std::endl;
  std::cout << JqD << std::endl;
}

BOOST_AUTO_TEST_CASE(t_quat_product) {

  Eigen::Vector4d p{1, 2, 3, 4};
  p.normalize();

  Eigen::Vector4d q{1, .2, .3, .4};
  p.normalize();

  Eigen::Vector4d out;
  Eigen::Matrix4d Jp;
  Eigen::Matrix4d Jq;

  quat_product(p, q, out, Jp, Jq);

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
        quat_product(x, q, y, __Jp, __Jq);
      },
      p, 4, JpD);

  finite_diff_jac(
      [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
        quat_product(p, x, y, __Jp, __Jq);
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

BOOST_AUTO_TEST_CASE(quad3d_bench_time) {

  size_t N = 1000;
  ptr<Dynamics> dyn = mk<Dynamics_quadcopter3d>();

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
}

BOOST_AUTO_TEST_CASE(quad3d) {
  ptr<Dynamics> dyn = mk<Dynamics_quadcopter3d>();
  check_dyn(dyn, 1e-5);
}

BOOST_AUTO_TEST_CASE(unicycle2) {
  ptr<Dynamics> dyn = mk<Dynamics_unicycle2>();
  check_dyn(dyn, 1e-5);
  ptr<Dynamics> dyn_free_time = mk<Dynamics_unicycle2>(true);
  check_dyn(dyn_free_time, 1e-5);
}

BOOST_AUTO_TEST_CASE(test_unifree) {
  ptr<Dynamics> dyn = mk<Dynamics_unicycle>();
  check_dyn(dyn, 1e-5);
  ptr<Dynamics> dyn_free_time = mk<Dynamics_unicycle>(true);
  check_dyn(dyn, 1e-5);
}

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

BOOST_AUTO_TEST_CASE(traj_opt_no_bounds) {

  opti_params = Opti_params();

  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/bugtrap_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml";

  opti_params.control_bounds = 0;
  opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
  opti_params.use_warmstart = 1;
  opti_params.max_iter = 50;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST(result.feasible);
}

bool check_equal(Eigen::MatrixXd A, Eigen::MatrixXd B, double rtol,
                 double atol) {

  CHECK_EQ(A.rows(), B.rows(), AT);
  CHECK_EQ(A.cols(), B.cols(), AT);

  auto dif = (A - B).cwiseAbs();
  auto max_cwise = A.cwiseAbs().cwiseMax(B.cwiseAbs());

  auto tmp = dif - (rtol * max_cwise +
                    atol * Eigen::MatrixXd::Ones(A.rows(), A.cols()));
  // all element in tmp shoulb be negative
  return (tmp.array() <= 0).all();
}

BOOST_AUTO_TEST_CASE(jac_quadrotor) {
  opti_params = Opti_params();
  opti_params.disturbance = 1e-7;
  size_t nx(13), nu(4);
  size_t N = 5;

  ptr<CollisionChecker> cl = nullptr;

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

    // std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(4));

    // issue with the goal!!
    //
    Generate_params gen_args;
    gen_args.name = "quadrotor_0";
    gen_args.N = N;
    opti_params.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.cl = cl;

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
    // BOOST_CHECK((data_terminal_diff->Lx - data_terminal->Lx).isZero(tol));
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

  auto cl = mk<CollisionChecker>();
  cl->load("../benchmark/unicycle_first_order_0/bugtrap_0.yaml");

  {

    Eigen::VectorXd goal = Eigen::VectorXd::Random(4);
    Eigen::VectorXd start = Eigen::VectorXd::Random(4);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(4));
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(3));

    // issue with the goal!!
    //
    Generate_params gen_args;
    gen_args.name = "unicycle_first_order_0";
    gen_args.N = 5;
    opti_params.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.cl = cl;
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

    // BOOST_CHECK((
    std::cout << data_terminal_diff->Lx << std::endl;
    std::cout << data_terminal->Lx << std::endl;

    // - data_terminal->Lx).isZero(tol));
    // BOOST_CHECK((data_terminal_diff->Lx - data_terminal->Lx).isZero(tol));
    BOOST_CHECK(
        check_equal(data_terminal_diff->Lx, data_terminal->Lx, tol, tol));
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
    }
  }

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
    gen_args.cl = cl;
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

    Eigen::VectorXd goal = Eigen::VectorXd::Random(4);
    Eigen::VectorXd start = Eigen::VectorXd::Random(4);

    std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Random(4));
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Random(3));

    gen_args.name = "unicycle_first_order_0";
    gen_args.N = 5;
    opti_params.use_finite_diff = 0;
    gen_args.goal = goal;
    gen_args.start = start;
    gen_args.cl = cl;
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
  state_bounds->calcDiff(Jx, Ju, x, u);

  std::cout << "r: " << r.format(FMT) << std::endl;

  finite_diff_cost(state_bounds, Jxdiff, Judiff, x, u, nr);

  double tol = 1e-3;
  BOOST_CHECK(check_equal(Jx, Jxdiff, tol, tol));
  BOOST_CHECK(check_equal(Ju, Judiff, tol, tol));
  // compute the jacobians with finite diff
}

BOOST_AUTO_TEST_CASE(contour_park_raw) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml";

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
  file_inout.env_file =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  file_inout.init_guess = "../build_debug/smooth_park_debug.yaml";

  opti_params.control_bounds = true;
  opti_params.use_finite_diff = false;
  opti_params.k_linear = 10.;
  opti_params.k_contour = 2.;

  opti_params.use_warmstart = true;
  opti_params.solver_id = static_cast<int>(SOLVER::mpcc_linear);
  opti_params.max_iter = 50;

  Result_opti result;
  compound_solvers(file_inout, result);
  BOOST_TEST(result.feasible);
}

BOOST_AUTO_TEST_CASE(parallel_small_step_good_init_guess) {
  opti_params = Opti_params();

  File_parser_inout file_inout;
  file_inout.env_file =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_parallelpark_mpcc.yaml";

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

BOOST_AUTO_TEST_CASE(bugtrap_bad_init_guess) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file = "../benchmark/unicycle_first_order_0/bugtrap_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml";

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
  file_inout.env_file = "../benchmark/unicycle_first_order_0/bugtrap_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_bugtrap_0_mpcc.yaml";

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
  file_inout.env_file =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml";

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
  file_inout.env_file =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml";

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
  file_inout.env_file =
      "../benchmark/unicycle_first_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml";

  // (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&  ./test_croco
  // --run_test=quim --  --env ../benchmark/uni
  // cycle_first_order_0/parallelpark_0.yaml  --waypoints
  // ../test/unicycle_first_order_0/guess_parallelpark_0_so l0.yaml  --out
  // out.yaml --solver 9 --control_bounds 1 --use_warmstart 1 > quim.txt

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
  file_inout.env_file = "../benchmark/unicycle_first_order_0/bugtrap_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml";

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
  file_inout.env_file = "../benchmark/unicycle_first_order_0/kink_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_first_order_0/guess_kink_0_sol0.yaml";

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
  // very big error. Compute the rotation of a vector.
  // check with finite diff.

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
  file_inout.env_file =
      "../benchmark/unicycle_second_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml";

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
  file_inout.env_file =
      "../benchmark/unicycle_second_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml";

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

BOOST_AUTO_TEST_CASE(park_second_mpcc) {

  opti_params = Opti_params();
  File_parser_inout file_inout;
  file_inout.env_file =
      "../benchmark/unicycle_second_order_0/parallelpark_0.yaml";
  file_inout.init_guess =
      "../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml";

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

// this should be a very good initial guess for parallel park -> almost time
// optimal "paralle_second_free_time_feasible.yaml"

BOOST_AUTO_TEST_CASE(park_second_contour) {
  // ./main_croco --env
  // ../benchmark/unicycle_second_order_0/parallelpark_0.yaml
  // --waypoints
  //  paralle_second_free_time.yaml   --out out.yaml --solver 10
  //  --control_bounds 1 --use_warmstart 1 --step_move 5 --step_opt 40
  //  --k_linear 20 --k_contour 10
  // 0 --weight_goal 100 --smooth 0 --max_iter 50
}

BOOST_AUTO_TEST_CASE(park_second_contour_A) {
  // (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&    ./main_croco --env
  // ../benchmark/unicycle_second_order_0/parallelpark_0.yaml   --waypoints
  // ../test/u nicycle_second_order_0/guess_parallelpark_0_sol0.yaml --out
  // out.yaml --solver 8 --control_bounds 1 --use_warmstart 1 --step_move 10
  // --step_opt 40 --k_lin ear 10 --k_contour 100 --weight_goal 200 --smooth 1
  // --max_iter 30
}

BOOST_AUTO_TEST_CASE(park_second_order_mpcA) {
  // (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 && gdb --args
  // ./main_croco
  // --env ../benchmark/unicycle_second_order_0/parallelpark_0.yaml
  // --waypoints
  // ../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml   --out
  // out.yaml --solver 10 --control_bounds 1 --use_warmstart 1 --step_move 20
  // --step_opt 4 0 --k_linear 20 --k_contour 10 --weight_goal 100 --smooth 0
  // --max_iter 50
}

// continue here:

// check how can i traverse this trajectory faster.
// linear_mpcc_parallel_slow.yaml

// this is bad
// (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&   ./test_croco
// --run_test=quim --  --env
// ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  --wa ypoints
// ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml  --out
// out.yaml
// --solver 8 --control_bounds 1 --use_warmstart 1 --step_move 10 --st ep_opt
// 35
// --alpha_rate 1. --use_fdif 0 --k_linear 20.  --k_contour 10. --weight_goal
// 200.

// this is good
// (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&   ./test_croco
// --run_test=quim --  --env
// ../benchmark/unicycle_first_order_0/parallelpark_0.yaml  --wa ypoints
// ../test/unicycle_first_order_0/guess_parallelpark_0_sol0.yaml  --out
// out.yaml
// --solver 8 --control_bounds 1 --use_warmstart 1 --step_move 35 --st ep_opt
// 35
// --alpha_rate 1. --use_fdif 0 --k_linear 20.  --k_contour 10. --weight_goal
// 200.

// what can I do when I now I can get to the goal?

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
