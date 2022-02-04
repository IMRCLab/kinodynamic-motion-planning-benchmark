#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include "car_utils.hpp"
#include <Kin/kin.h>
#include <cassert>

struct Trailer : Feature {
  // IMPORTANT: See Quim's Drawing:

  double d1;

  Trailer(double d1) : d1(d1) {}

  void phi2(arr &y, arr &J, const FrameL &F) {

    CHECK_EQ(F.nd, 2, "need two frames");
    std::cout << "ho" << std::endl;

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
    y(0) = tdot(0) + rdot(2) - vel(0) / d1 * ct;

    if (!!J) {
      // ROWS = 1 equations ; COLUMNS= 1 t +  1 tdot + 1 vel + 3 rdot
      arr Jl;
      Jl.resize(1, 6);
      Jl.setZero(0);
      Jl(0, 0) = vel(0) / d1 * st;
      Jl(0, 1) = 1;
      Jl(0, 2) = -1 / d1 * ct;
      Jl(0, 5) = 1;

      arr block_, block_2, block;
      block_.setBlockMatrix(Jt, Jtdot);
      block_2.setBlockMatrix(block_, Jvel);
      block.setBlockMatrix(block_2, Jrdot);
      J = Jl * block;
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    std::cout << "dim is " << 1 << std::endl;
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

    // phi
    arr phi;
    arr Jphi;
    F_qItself().setOrder(0).eval(phi, Jphi, F(1, {1, 1}).reshape(1, -1));

    std::cout << "r " << r << std::endl;
    std::cout << "Jr " << Jr << std::endl;
    std::cout << "rdot " << rdot << std::endl;
    std::cout << "Jdot " << Jrdot << std::endl;
    std::cout << "phi " << phi << std::endl;
    std::cout << "Jphi " << Jphi << std::endl;

    arr vel, Jvel;
    FrameL Fvel = {F(0, 0), F(1, 0)}; // First column
    get_speed(vel, Jvel, Fvel.reshape(-1, 1));

    y.resize(1);
    double tphi = std::tan(phi(0));
    double cphi = std::cos(phi(0));
    y(0) = rdot(2) - vel(0) / L * tphi;

    if (!!J) {
      // ROWS = 1 equations ; COLUMNS= 3 rdot + 1 vel + 1 phi
      arr Jl;
      Jl.resize(1, 5);
      Jl(0, 2) = 1;             // w.r.t rdot(2)
      Jl(0, 3) = -1 / L * tphi; //
      Jl(0, 4) = -vel(0) / (L * cphi * cphi);

      arr block_, block;
      block_.setBlockMatrix(Jrdot, Jvel);
      block.setBlockMatrix(block_, Jphi);

      J = Jl * block;
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    return 1;
  }
};

arrA getPath_qAll_with_prefix(KOMO &komo, int order) {
  arrA q(komo.T + order);
  for (int t = -order; t < int(komo.T); t++) {
    q(t + order) = komo.getConfiguration_qAll(t);
  }
  return q;
}

int main(int argn, char **argv) {

  const double L = .4;  // distance  rear-front wheels
  const double d1 = .5; // distance between car centers
  rai::initCmdLine(argn, argv);
  rnd.clockSeed();

  // path to *.g file
  rai::String model_file =
      rai::getParameter<rai::String>("model", STRING("none"));

  // path to initial guess file (yaml)
  rai::String waypoints_file =
      rai::getParameter<rai::String>("waypoints", STRING("none"));

  bool display = rai::getParameter<bool>("display", false);
  int animate = rai::getParameter<int>("animate", 0);

  // path to output file (*.yaml)
  rai::String out_file =
      rai::getParameter<rai::String>("out", STRING("out.yaml"));

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

  // int N = 50;
  int N = 50;
  double dt = 0.1;
  double duration_phase = N * dt;
  komo.setTiming(1, N, duration_phase, order);

  komo.add_qControlObjective({}, order, 1);

  double action_factor = rai::getParameter<double>("action_factor", 1.0);

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

  if (order == 1) {

    const double max_velocity = 0.5 * action_factor - 0.01; // m/s
    double max_phi = 45 * 3.14159 / 180;

    // Linear velocity First Car
    komo.addObjective({}, make_shared<UnicycleDynamics>(), {car_name}, OT_eq,
                      {1e1}, {0}, 1);

    // Rotation First Car
    komo.addObjective({}, make_shared<FirstCarRotation>(L),
                      {car_name, wheel_name}, OT_eq, {1e1}, {0}, 1);

    // Rotation Trailer
    komo.addObjective({}, make_shared<Trailer>(d1), {car_name, arm_name}, OT_eq,
                      {1e1}, {0}, 1);

    // Bound Linear Velocity
    komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name}, OT_ineq,
                      {1}, {max_velocity}, 1);

    komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name}, OT_ineq,
                      {-1}, {-max_velocity}, 1);

    // Bound angle on wheel
    komo.addObjective({}, FS_qItself, {wheel_name}, OT_ineq, {1}, {max_phi}, 1);

    komo.addObjective({}, FS_qItself, {wheel_name}, OT_ineq, {-1}, {-max_phi},
                      1);

    // TODO: Wolfgang, do you want bounds on the angular velocity?

  } else {
    NIY;
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

  komo.run_prepare(0.01);
  komo.reportProblem();

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
    out << "      - [" << v(0) << "," << v(1) << ","
        << std::remainder(v(2), 2 * M_PI) << ","
        << std::remainder(v(2) + M_PI / 2 + v(4) - M_PI, 2 * M_PI) << "]"
        << std::endl;
  }
  // out << "    actions:" << std::endl;
  // for (size_t t = order; t < results.N; ++t) {
  //   out << "      - [" << velocity(results, t, dt) << ","
  //       << angularVelocity(results, t, dt) << "]" << std::endl;
  // }

  return 0;
}
