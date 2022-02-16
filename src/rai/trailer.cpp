#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include "car_utils.h"
#include <Kin/F_pose.h>
#include <Kin/kin.h>
#include <cassert>

struct Trailer : Feature {
  // IMPORTANT: See Quim's Drawing:

  double d1;

  Trailer(double d1) : d1(d1) {}

  void phi2(arr &y, arr &J, const FrameL &F) {

    CHECK_EQ(F.nd, 2, "need two frames");

    CHECK_EQ(F(0, 0)->joint->type, rai::JT_transXYPhi, "");
    CHECK_EQ(F(0, 1)->joint->type, rai::JT_hingeZ, "");
    CHECK_EQ(F(1, 0)->joint->type, rai::JT_transXYPhi, "");
    CHECK_EQ(F(1, 1)->joint->type, rai::JT_hingeZ, "");

    y.resize(1);

    // robot
    arr r, Jr;
    arr rdot, Jrdot;
    F_qItself().setOrder(0).eval(r, Jr, FrameL{F(1, 0)}.reshape(1, -1));
    F_qItself().setOrder(1).eval(rdot, Jrdot,
                                 FrameL{F(0, 0), F(1, 0)}.reshape(-1, 1));

    // trailer
    arr ang_vel, Jang_vel;
    F_AngVel().phi2(ang_vel, Jang_vel, FrameL{F(0, 0), F(1, 0)}.reshape(-1, 1));

    // trailer
    arr t, Jt;
    arr tdot, Jtdot;
    F_qItself().setOrder(0).eval(t, Jt, FrameL{F(1, 1)}.reshape(1, -1));
    F_qItself().setOrder(1).eval(
        tdot, Jtdot, FrameL{F(0, 1), F(1, 1)}.reshape(-1, 1)); // break here!!

    // get First car speed
    arr vel, Jvel;
    FrameL Fvel = {F(0, 0), F(1, 0)}; // First column
    get_speed(vel, Jvel, Fvel.reshape(-1, 1));

    double ct = std::cos(t(0));
    double st = std::sin(t(0));
    // y(0) = tdot(0) + rdot(2) - vel(0) / d1 * ct;
    y(0) = tdot(0) -ang_vel(2) - vel(0) / d1 * ct;

    if (!!J) {
      // ROWS = 1 equations ; COLUMNS= 1 t +  1 tdot + 1 vel + 3 rdot
      arr Jl;
      Jl.resize(1, 6);
      Jl.setZero(0);
      Jl(0, 0) = vel(0) / d1 * st;
      Jl(0, 1) = 1;
      Jl(0, 2) = -1 / d1 * ct;
      // Jl(0, 5) = 1;
      Jl(0, 5) = -1;

      arr block_, block_2, block;
      block_.setBlockMatrix(Jt, Jtdot);
      block_2.setBlockMatrix(block_, Jvel);
      // block.setBlockMatrix(block_2, Jrdot);
      block.setBlockMatrix(block_2, Jang_vel);
      J = Jl * block;
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    return 1;
  }
};

struct FirstCarRotation : Feature {

  double L;
  FirstCarRotation(double L) : L(L) {}

  void phi2(arr &y, arr &J, const FrameL &F) {

    CHECK_EQ(F.nd, 2, "need two frames");

    CHECK_EQ(F(0, 0)->joint->type, rai::JT_transXYPhi, "");
    CHECK_EQ(F(0, 1)->joint->type, rai::JT_hingeZ, "");
    CHECK_EQ(F(1, 0)->joint->type, rai::JT_transXYPhi, "");
    CHECK_EQ(F(1, 1)->joint->type, rai::JT_hingeZ, "");

    bool verbose = false;
    if (verbose) {
      std::cout << "frames " << std::endl;
      std::cout << F.nd << std::endl;
      std::cout << F(0, 0)->name << std::endl; // robot
      std::cout << F(0, 1)->name << std::endl; // front wheel
      for (auto &f : F) {
        std::cout << f->name << std::endl;
      }

      std::cout << "**** " << std::endl;
    }

    // robot
    arr r, Jr;
    arr rdot, Jrdot;
    F_qItself().setOrder(0).eval(r, Jr, FrameL{F(1, 0)}.reshape(1, -1));
    F_qItself().setOrder(1).eval(rdot, Jrdot,
                                 FrameL{F(0, 0), F(1, 0)}.reshape(-1, 1));

    // NOTE: double check
    arr ang_vel, Jang_vel;
    F_AngVel().phi2(ang_vel, Jang_vel, FrameL{F(0, 0), F(1, 0)}.reshape(-1, 1));

    // NOTE: rdot(2) = -ang_vel(2). why?
    // std::cout << "rdot(2) " << rdot(2) << std::endl;
    // std::cout << "ang_vel " << ang_vel << std::endl;

    // phi
    arr phi;
    arr Jphi;
    F_qItself().setOrder(0).eval(phi, Jphi, F(1, {1, 1}).reshape(1, -1));

    arr vel, Jvel;
    FrameL Fvel = {F(0, 0), F(1, 0)}; // First column
    get_speed(vel, Jvel, Fvel.reshape(-1, 1));

    y.resize(1);
    double tphi = std::tan(phi(0));
    double cphi = std::cos(phi(0));
    // y(0) = rdot(2) - vel(0) / L * tphi;
    y(0) = -1 * ang_vel(2) - vel(0) / L * tphi;

    if (!!J) {
      // ROWS = 1 equations ; COLUMNS= 3 rdot + 1 vel + 1 phi
      arr Jl;
      Jl.resize(1, 5);
      // Jl(0, 2) = 1;             // w.r.t rdot(2)
      Jl(0, 2) = -1;             // w.r.t angvel(2)
      Jl(0, 3) = -1 / L * tphi; //
      Jl(0, 4) = -vel(0) / (L * cphi * cphi);

      arr block_, block;
      block_.setBlockMatrix(Jang_vel, Jvel);
      // block_.setBlockMatrix(Jrdot, Jvel);
      block.setBlockMatrix(block_, Jphi);

      J = Jl * block;
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    return 1;
  }
};

static arrA getPath_qAll_with_prefix(KOMO &komo, int order) {
  arrA q(komo.T + order);
  for (int t = -order; t < int(komo.T); t++) {
    q(t + order) = komo.getConfiguration_qAll(t);
  }
  return q;
}

static double velocity(const arrA &results, int t, double dt) {
  const double tol = 0.1; // tolerance to avoid division by zero

  arr v = results(t) - results(t - 1);
  double theta = results(t)(2);

  double c_theta = cos(theta);
  double s_theta = sin(theta);
  double speed;

  if (fabs(c_theta) > tol) {
    speed = v(0) / c_theta;
  } else {
    speed = v(1) / s_theta;
  }

  return speed / dt;
}

int main_trailer() {

  const double L = .4;  // distance  rear-front wheels
  const double d1 = .5; // distance between car centers

  // path to *.g file
  rai::String model_file =
      rai::getParameter<rai::String>("model", STRING("none"));

  // path to initial guess file (yaml)
  rai::String waypoints_file =
      rai::getParameter<rai::String>("waypoints", STRING("none"));

  int N = rai::getParameter<int>("N", -1);

  bool display = rai::getParameter<bool>("display", false);
  int animate = rai::getParameter<int>("animate", 0);

  // path to output file (*.yaml)
  rai::String out_file =
      rai::getParameter<rai::String>("out", STRING("out.yaml"));

  rai::String env_file = rai::getParameter<rai::String>("env", STRING("none"));

  rai::Configuration C;
  C.addFile(model_file);

  int order = rai::getParameter<int>("order", 1);
  std::cout << "Car order: " << order << std::endl;

  KOMO komo;
  komo.setModel(C, true);

  auto robot_collision = "R_robot_shape";
  auto car_name = "R_robot";
  auto goal_name = "GOAL_robot";
  auto arm_name = "R_arm";
  auto wheel_name = "R_front_wheel";
  auto trailer_name = "R_trailer";
  auto trailer_goal = "GOAL_trailer";

  arrA waypoints;
  if (waypoints_file != "none") {
    // load initial guess
    YAML::Node env = YAML::LoadFile((const char *)waypoints_file);
    const auto &node = env["result"][0];
    size_t num_states = node["states"].size();

    // the initial guess has states: x,y,theta0,theta1v and actions: v, phi
    // KOMO uses 5 states: x,y,theta0,phi,theta1q

    double latest_theta = 3.14;
    for (size_t i = 1; i < num_states; ++i) {
      const auto &state = node["states"][i];

      auto x = state[0].as<double>();
      auto y = state[1].as<double>();
      auto theta0 = state[2].as<double>();

      //
      double theta0_plus = theta0 + 2. * M_PI;
      double theta0_minus = theta0 - 2. * M_PI;
      // check the difference in abs value
      // double dif = std::abs(latest_theta - theta0);
      // double dif_plus = std::abs(latest_theta - theta0_plus);
      // double dif_minus = std::abs(latest_theta - theta0_minus);

      // if (dif_plus < dif) {
      //   theta0 = theta0_plus;
      //   dif = dif_plus;
      // }
      // if (dif_minus < dif) {
      //   theta0 = theta0_minus;
      //   dif = dif_minus;
      // }

      auto theta1v = state[3].as<double>();

      // see drawing
      double theta1q = M_PI / 2 - theta0 + theta1v;
      // put beween -pi, pi
      theta1q = std::atan2(sin(theta1q), cos(theta1q));
      double phi = 0;
      if (node["actions"]) {
        const auto &action = env["result"][0]["actions"][i - 1];
        phi = action[1].as<double>();
      }
      waypoints.append({x, y, theta0, phi, theta1q});
      latest_theta = theta0;
    }
    N = waypoints.N;
  }
  double dt = 0.1;
  double duration_phase = N * dt;
  komo.setTiming(1, N, duration_phase, order);

  // komo.add_qControlObjective({}, order, .5);
  // komo.add_qControlObjective({}, order, .5);

  bool regularize_traj = false;
  if (regularize_traj && waypoints_file != "none") {
    double scale_regularization = .1; // try different scales
    int it = 1;
    // ways -> N+1
    // N
    for (const arr &a : waypoints) // i take from index=1 because we
                                   // are ignoring the first waypoint.
    {
      komo.addObjective(double(it) * arr{1. / N, 1. / N}, FS_qItself, {},
                        OT_sos, {scale_regularization}, a, 0);
      it++;
    }
  }

  komo.addObjective({}, FS_qItself, {arm_name}, OT_sos, {.1}, {}, 1);
  komo.addObjective({}, FS_qItself, {wheel_name}, OT_sos, {.1}, {}, 1);
  komo.addObjective({}, make_shared<F_LinAngVel>(), {car_name}, OT_sos, {.1},
  {},
                    1);

  // komo.addObjective({}, make_shared<F_AngVel>(), {car_name}, OT_sos, {.1},
  // {},
  //                   1);

  // komo.addObjective({}, make_shared<F_AngVel>(), {arm_name}, OT_sos, {.1},
  // {},
  //                   1);

  // komo.addObjective({}, make_shared<F_AngVel>(), {wheel_name}, OT_sos, {.1},
  // {},
  //                   1);

  // komo.addObjective({}, make_shared<UnicycleDynamics>(), {car_name}, OT_eq,
  //                   {1e1}, {0}, 1);

  // add the goal
  komo.addObjective({1., 1.}, FS_poseDiff, {car_name, goal_name}, OT_eq, {1e2});
  komo.addObjective({1., 1.}, FS_poseDiff, {trailer_name, trailer_goal}, OT_eq,
                    {1e2});

  // add collisions
  StringA obstacles;
  for (auto &frame : C.frames) {
    std::cout << *frame << std::endl;
    if (frame->shape && frame->name.startsWith("obs")) {
      obstacles.append(frame->name);
    }
  }

  for (auto &obs : obstacles) {
    komo.addObjective({}, FS_distance, {robot_collision, obs}, OT_ineq, {1e2});
    komo.addObjective({}, FS_distance, {trailer_name, obs}, OT_ineq, {1e2});
  }

  // Workspace bounds
  YAML::Node env = YAML::LoadFile((const char *)env_file);
  double x_min = env["environment"]["min"][0].as<double>();
  double y_min = env["environment"]["min"][1].as<double>();
  double x_max = env["environment"]["max"][0].as<double>();
  double y_max = env["environment"]["max"][1].as<double>();

  komo.addObjective({}, FS_position, {car_name}, OT_ineq, {1, 1, 0}, {x_max, y_max, 0});
  komo.addObjective({}, FS_position, {car_name}, OT_ineq, {-1, -1, 0}, {x_min, y_min, 0});

  if (order == 1) {

    const double min_velocity = -0.1; // m/s
    const double max_velocity = 0.5; // m/s
    const double min_phi = -M_PI / 3;
    const double max_phi = M_PI / 3;
    const double min_theta1quim = M_PI / 2. - M_PI / 4.;
    const double max_theta1quim = M_PI / 2. + M_PI / 4.;

    // Linear velocity First Car
    komo.addObjective({}, make_shared<UnicycleDynamics>(), {car_name}, OT_eq,
                      {10}, {0}, 1);

    // Rotation First Car
    komo.addObjective({}, make_shared<FirstCarRotation>(L),
                      {car_name, wheel_name}, OT_eq, {10}, {0}, 1);

    // Rotation Trailer
    komo.addObjective({}, make_shared<Trailer>(d1), {car_name, arm_name}, OT_eq,
                      {10}, {0}, 1);

    // Bound Linear Velocity
    komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name}, OT_ineq,
                      {10}, {max_velocity}, 1);

    komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name}, OT_ineq,
                      {-10}, {min_velocity}, 1);

    // Bound angle on wheel
    komo.addObjective({}, FS_qItself, {wheel_name}, OT_ineq, {10}, {max_phi},
                      -1);

    komo.addObjective({}, FS_qItself, {wheel_name}, OT_ineq, {-10}, {min_phi},
                      -1);

    // bound angle between car and trailer
    komo.addObjective({}, FS_qItself, {arm_name}, OT_ineq, {10},
                      {max_theta1quim}, 0);
    komo.addObjective({}, FS_qItself, {arm_name}, OT_ineq, {-10},
                      {min_theta1quim}, 0);

  } else {
    NIY;
  }

  komo.run_prepare(0.02); // TODO: is this necessary?
  if (waypoints_file != "none") {
    komo.initWithWaypoints(waypoints, N);

    bool report_before = true;
    if (report_before) {
      std::cout << "report before solve" << std::endl;
      auto sparse = komo.nlp_SparseNonFactored();
      arr phi;
      sparse->evaluate(phi, NoArr, komo.x);

      komo.reportProblem();
      rai::Graph report = komo.getReport(display, 0, std::cout);
      std::cout << "report " << report << std::endl;
      if (display) {
        komo.view_play(true, 0.3);
      }
    }
  }

  bool check_gradients = false;
  if (check_gradients) {
    std::cout << "checking gradients" << std::endl;
    // TODO: to avoid singular points, I shoud add noise before
    // checking the gradients.
    komo.run_prepare(.1);
    komo.checkGradients();
    std::cout << "done " << std::endl;
  }

  // komo.run_prepare(0.01);
  // komo.reportProblem();

  if (display) {
    // komo.view(true);
    // komo.view_play(true);
    // komo.view_play(true, 1,"vid/car");
    komo.plotTrajectory();

    do {
      cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
  }
  // throw -1;

  komo.run();

  komo.reportProblem();

  if (display) {
    komo.view(true);
    komo.view_play(true);
    // komo.view_play(true, 1,"vid/car");
    komo.plotTrajectory();

    do {
      cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
  }

  auto report = komo.getReport(display, 0, std::cout);
  std::cout << "report " << report << std::endl;
  double ineq = report.get<double>("ineq") / komo.T;
  double eq = report.get<double>("eq") / komo.T;
  if (ineq > 0.01 || eq > 0.01) {
    // Optimization failed (constraint violations)
    std::cout << "Optimization failed (constraint violation)!" << std::endl;
    return 1;
  }

  // write the results.
  arrA results = getPath_qAll_with_prefix(komo, order);
  std::cout << "results: " << std::endl;
  std::cout << C.getFrameNames() << std::endl;
  std::cout << results << std::endl;
  std::cout << "(N,T): " << results.N << " " << komo.T << std::endl;

  std::ofstream out(out_file);
  // out << std::setprecision(std::numeric_limits<double>::digits10 + 1);
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (size_t t = order - 1; t < results.N; ++t) {
    auto &v = results(t);
    // v(2) = theta0
    // v(4) = theta1q
    // theta1v = theta1q - 90 + theta0
    out << "      - [" << v(0) << "," << v(1) << ","
        << std::remainder(v(2), 2 * M_PI) << ","
        << std::remainder(v(4) - M_PI / 2 + v(2), 2 * M_PI) << "]" << std::endl;
  }
  out << "    actions:" << std::endl;
  for (size_t t = order; t < results.N; ++t) {
    auto &v = results(t);
    out << "      - [" << velocity(results, t, dt) << "," << v(3) << "]"
        << std::endl;
  }

  return 0;
}
