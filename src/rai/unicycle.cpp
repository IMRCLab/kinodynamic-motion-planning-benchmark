#include "Core/util.h"

#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include <Kin/kin.h>
#include <cassert>
#include <cmath>
#include <iostream>
//#include <iomanip>

#include <yaml-cpp/yaml.h>
#include "car_utils.h"



static double velocity(const arrA& results, int t, double dt) {
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

static double acceleration(const arrA &results, int t, double dt) {
  double vel_now = velocity(results, t, dt);
  double vel_before = velocity(results, t - 1, dt);
  double acc = (vel_now - vel_before) / dt;
  return acc;
}

static double angularVelocity(const arrA &results, int t, double dt) {
  arr delta = results(t) - results(t - 1);
  double angular_change = atan2(sin(delta(2)), cos(delta(2)));
  double angular_velocity = angular_change / dt; // rad/s
  return angular_velocity;
}

static double angularAcceleration(const arrA &results, int t, double dt) {
  double omega_now = angularVelocity(results, t, dt);
  double omega_before = angularVelocity(results, t - 1, dt);
  double omega_dot = (omega_now - omega_before) / dt;
  return omega_dot;
}

static arrA getPath_qAll_with_prefix(KOMO &komo, int order) {
  arrA q(komo.T + order);
  for (int t = -order; t < int(komo.T); t++) {
    q(t + order) = komo.getConfiguration_qAll(t);
  }
  return q;
}

// usage:
// EXECUTABLE -model FILE_G -waypoints FILE_WAY -one_every ONE_EVERY_N
// -display {0,1} -out OUT_FILE OUT_FILE -animate  {0,1,2} -order {0,1,2}

// OUT_FILE: Write down the trajectory
// ONE_EVERY_N: take only one every N waypoints

int main_unicycle(float min_v, float max_v, float min_w, float max_w) {

  rai::String model_file =
      rai::getParameter<rai::String>("model", STRING("none"));
  rai::String waypoints_file =
      rai::getParameter<rai::String>("waypoints", STRING("none"));

  int N = rai::getParameter<int>("N", -1);

  bool display = rai::getParameter<bool>("display", false);
  int animate = rai::getParameter<int>("animate", 0);
  int order = rai::getParameter<int>("order", 0);
  rai::String out_file =
      rai::getParameter<rai::String>("out", STRING("out.yaml"));

  rai::String env_file = rai::getParameter<rai::String>("env", STRING("none"));

  rai::String mode = rai::getParameter<rai::String>("mode", STRING("default"));


  enum CAR_ORDER {
    ZERO = 0, // no bounds
    ONE = 1,  // bounds velocity
    TWO = 2,  // bound acceleration
  };
  CAR_ORDER car_order = static_cast<CAR_ORDER>(order);
  std::cout << "Car order: " << order << std::endl;

  arrA waypoints;
  if (waypoints_file != "none") {
    waypoints = load_waypoints(waypoints_file);

    if (car_order == TWO) {
      arrA waypointsS;
      waypointsS.append(waypoints(0));
      // waypointsS.append(waypoints(0));
      for (size_t i = 0; i < waypoints.N; i++) {
        waypointsS.append(waypoints(i));
      }
      waypointsS.append(waypoints(-1));
      waypoints = waypointsS;
    }

    std::cout << "Warning: we will skip the first waypoint" << std::endl;
    std::cout << "waypoints are (including the first)" << std::endl;
    std::cout << waypoints << std::endl;

    int Nplus1 = waypoints.N;
    N = Nplus1 - 1;
  }

  // load G file
  rai::Configuration C;
  C.addFile(model_file);

  // NOTE: collision constraints are added one by one between the robot
  // and the objects that contain the keyworkd "contact"
  // but not "robot0" itself
  StringA obstacles;
  for (auto &frame : C.frames) {
    std::cout << *frame << std::endl;
    if (frame->shape) {
      if (frame->shape->cont) {
        if (strcmp(frame->name, "robot0") != 0) {
          obstacles.append(frame->name);
        }
      }
    }
  }

  // create optimization problem
  KOMO komo;
  komo.setModel(C, true);

  std::cout << "N " << N << std::endl;

  if (N == 0) {
    return 1;
  }

  double dt = 0.1;
  double duration_phase = N * dt;
  komo.setTiming(1, N, duration_phase, order);

  if (order == 2) {
    komo.add_qControlObjective({}, 2, .1);
    // NOTE: we could also add cost on the velocity
  }

  if (order == 1) {
    komo.add_qControlObjective({}, 1, .1);
  }

  bool regularize_traj = false;
  if (regularize_traj && waypoints_file != "none") {
    double scale_regularization = .1; // try different scales
    int it = 1;
    // ways -> N+1
    // N
    for (const arr &a : waypoints({1, -1})) // i take from index=1 because we
                                            // are ignoring the first waypoint.
    {
      komo.addObjective(double(it) * arr{1. / N, 1. / N}, FS_qItself,
                        {"robot0"}, OT_sos, {scale_regularization}, a, 0);
      it++;
    }
  }

  // I assume names robot0 and goal0 in the .g file
  if (mode != "dynamics_check") {
    komo.addObjective({1., 1.}, FS_poseDiff, {"robot0", "goal0"}, OT_eq, {1e2});
    // komo.addObjective({1., 1.}, FS_poseDiff, {"robot0", "goal0"}, OT_sos, {1e1});
  }

  // Note: if you want position constraints on the first variable.
  // komo.addObjective({1./N, 1./N}, FS_poseDiff, {"robot0", "start0"}, OT_eq,
  // {1e2});

  // Workspace bounds
  YAML::Node env = YAML::LoadFile((const char *)env_file);
  double x_min = env["environment"]["min"][0].as<double>();
  double y_min = env["environment"]["min"][1].as<double>();
  double x_max = env["environment"]["max"][0].as<double>();
  double y_max = env["environment"]["max"][1].as<double>();

  komo.addObjective({}, FS_position, {"robot0"}, OT_ineq, {1, 1, 0}, {x_max, y_max, 0});
  komo.addObjective({}, FS_position, {"robot0"}, OT_ineq, {-1, -1, 0}, {x_min, y_min, 0});


  if (car_order == ONE) {
    // robot dynamics
    komo.addObjective({}, make_shared<UnicycleDynamics>(), {"robot0"}, OT_eq,
                      {1e1}, {0}, 1);

    // angular velocity limit
    komo.addObjective({}, make_shared<UnicycleAngularVelocity>(), {"robot0"},
                      OT_ineq, {10}, {max_w}, 1);

    komo.addObjective({}, make_shared<UnicycleAngularVelocity>(), {"robot0"},
                      OT_ineq, {-10}, {min_w}, 1);

    // velocity limit
    komo.addObjective({}, make_shared<UnicycleVelocity>(), {"robot0"}, OT_ineq,
                      {10}, {max_v}, 1);

    komo.addObjective({}, make_shared<UnicycleVelocity>(), {"robot0"}, OT_ineq,
                      {-10}, {min_v}, 1);
  }
  if (car_order == TWO) {
    const double max_acceleration = 0.25 - 0.01; // m s^-2
    const double max_wdot = 0.25 - 0.01;
    const double max_velocity = 0.5 - 0.01; // m/s
    const double max_omega = 0.5 - 0.01;    // rad/s

    // robot dynamics
    komo.addObjective({2. / N, 1.}, make_shared<UnicycleDynamics>(), {"robot0"},
                      OT_eq, {1e1}, {0}, 1);
    // angular velocity limit
    komo.addObjective({2./N,1.}, make_shared<UnicycleAngularVelocity>(), {"robot0"},
                      OT_ineq, {1}, {max_omega}, 1);

    komo.addObjective({2./N,1.}, make_shared<UnicycleAngularVelocity>(), {"robot0"},
                      OT_ineq, {-1}, {-max_omega}, 1);

    // velocity limit
    komo.addObjective({2. / N, 1.}, make_shared<UnicycleVelocity>(), {"robot0"},
                      OT_ineq, {1}, {max_velocity}, 1);

    komo.addObjective({2. / N, 1.}, make_shared<UnicycleVelocity>(), {"robot0"},
                      OT_ineq, {-1}, {-max_velocity}, 1);

    // NOTE: UnicycleAcceleration returns v - v'
    // a = (v - v') / dt
    // so use  amax * dt as limit

    komo.addObjective({3. / N, 1.}, make_shared<UnicycleAcceleration>(),
                      {"robot0"}, OT_ineq, {10}, {max_acceleration * dt}, 2);

    komo.addObjective({3. / N, 1.}, make_shared<UnicycleAcceleration>(),
                      {"robot0"}, OT_ineq, {-10}, {-max_acceleration * dt}, 2);

    // angular acceleration control limit
    komo.addObjective({3./N,1.}, make_shared<UnicycleAngularAcceleration>(), {"robot0"},
                      OT_ineq, {10}, {max_wdot}, 2);

    komo.addObjective({3./N,1.}, make_shared<UnicycleAngularAcceleration>(), {"robot0"},
                      OT_ineq, {-10}, {-max_wdot}, 2);


    // contraints: zero velocity start and end
    // NOTE: {0,0} seems to be ok for velocity.
    // But why not {1/N,1/N},as in the position case?

    double v0 = env["robots"][0]["start"][3].as<double>();
    double w0 = env["robots"][0]["start"][4].as<double>();

    komo.addObjective({2. / N, 2. / N}, FS_poseDiff, {"robot0", "start0"},
                      OT_eq, {10});

    komo.addObjective({2. / N, 2. / N}, make_shared<UnicycleVelocity>(),
                      {"robot0"}, OT_eq, {10}, {v0}, 1);
    komo.addObjective({2. / N, 2. / N}, FS_qItself, {"robot0"}, OT_eq,
                      {0, 0, 10}, {0, 0, w0}, 1);

    double vf = env["robots"][0]["goal"][3].as<double>();
    double wf = env["robots"][0]["goal"][4].as<double>();

    komo.addObjective({1., 1.}, make_shared<UnicycleVelocity>(), {"robot0"},
                      OT_eq, {10}, {vf}, 1);
    komo.addObjective({1., 1.}, FS_qItself, {"robot0"}, OT_eq, {0, 0, 10},
                      {0, 0, wf}, 1);
    // komo.addObjective({1., 1.}, make_shared<UnicycleVelocity>(), {"robot0"},
    //                   OT_sos, {10}, {vf}, 1);
    // komo.addObjective({1., 1.}, FS_qItself, {"robot0"}, OT_sos, {0, 0, 10},
    //                   {0, 0, wf}, 1);

    // komo.addObjective({1./N,1./N}, FS_qItself, {"robot0"}, OT_eq,
    // {10},{},1); komo.addObjective({1,1}, FS_qItself, {"robot0"}, OT_eq,
    // {10},{},1);

    // Regularization: go slow at beginning and end
    // komo.addObjective({0,.1}, FS_qItself, {"robot0"}, OT_sos, {},{},1);
    // komo.addObjective({.9,1}, FS_qItself, {"robot0"}, OT_sos, {},{},1);
  }

  for (auto &obs : obstacles) {
    komo.addObjective({}, FS_distance, {"robot0", obs}, OT_ineq, {1e2});
  }

  komo.run_prepare(0.1); // TODO: is this necessary?
  // komo.checkGradients();
  std::cout << "done" << std::endl;
  if (waypoints_file != "none") {
    komo.initWithWaypoints(waypoints({1, -1}), N);

    // komo.run_prepare(0.1); // TODO: is this necessary?
    // komo.checkGradients();
    std::cout << "done" << std::endl;
    // throw -1;

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

      if (mode == "dynamics_check") {
        double ineq = report.get<double>("ineq") / komo.T;
        double eq = report.get<double>("eq") / komo.T;

        if (ineq <= 1e-5 && eq <= 1e-3) {
          return 0;
        }
        std::cout << "Dynamics check failed! Rerun with animate=1 and display=1 to debug! ineq="
                  << ineq << " eq=" 
                  << eq << std::endl;
        return 1;
      }
    }
  }
  // return 5;

  bool check_gradients = false;
  if (check_gradients) {
    std::cout << "checking gradients" << std::endl;
    // TODO: to avoid singular points, I shoud add noise before
    // checking the gradients.
    komo.checkGradients();
    std::cout << "done " << std::endl;
  }

  if (animate) {
    komo.opt.animateOptimization = animate;
  }

  // TODO: in final benchmark, check which is the optimal value of inital
  // noise.
  komo.optimize(0.1);

  std::cout << "report after solve" << std::endl;
  komo.reportProblem();

  auto report = komo.getReport(display, 0, std::cout);
  std::cout << "report " << report << std::endl;
  // std::cout << "ineq: " << report.getValuesOfType<double>("ineq") <<
  // std::endl;
  double ineq = report.get<double>("ineq") / komo.T;
  double eq = report.get<double>("eq") / komo.T;

  if (display) {
    // komo.view(true);
    komo.view_play(true);
    // komo.view_play(false, .3, "z.vid/");
  }

  arrA results = getPath_qAll_with_prefix(komo, order);

  std::cout << "results: " << std::endl;
  std::cout << results << std::endl;
  std::cout << "(N,T): " << results.N << " " << komo.T << std::endl;

  // write the results.
  std::ofstream out(out_file);
  // out << std::setprecision(std::numeric_limits<double>::digits10 + 1);
  out << "result:" << std::endl;
  if (car_order == ZERO) {
    out << "  - states:" << std::endl;
    for (size_t t = order - 1; t < results.N; ++t) {
      auto &v = results(t);
      out << "      - [" << v(0) << "," << v(1) << ","
          << std::remainder(v(2), 2 * M_PI) << "]" << std::endl;
    }
  } else if (car_order == ONE) {
    out << "  - states:" << std::endl;
    for (size_t t = order - 1; t < results.N; ++t) {
      auto &v = results(t);
      out << "      - [" << v(0) << "," << v(1) << ","
          << std::remainder(v(2), 2 * M_PI) << "]" << std::endl;
    }
    out << "    actions:" << std::endl;
    for (size_t t = order; t < results.N; ++t) {
      out << "      - [" << velocity(results, t, dt) << ","
          << angularVelocity(results, t, dt) << "]" << std::endl;
    }
  } else if (car_order == TWO) {
    out << "  - states:" << std::endl;
    for (size_t t = order - 1 + 2; t < results.N; ++t) {
      const auto &v = results(t);
      out << "      - [" << v(0) << "," << v(1) << ","
          << std::remainder(v(2), 2 * M_PI) << "," << velocity(results, t, dt)
          << "," << angularVelocity(results, t, dt) << "]" << std::endl;
    }
    out << "    actions:" << std::endl;
    for (size_t t = order + 2; t < results.N; ++t) {
      out << "      - [" << acceleration(results, t, dt) << ","
          << angularAcceleration(results, t, dt) << "]" << std::endl;
    }
  }

  if (ineq > 0.01 || eq > 0.01) {
    // Optimization failed (constraint violations)
    std::cout << "Optimization failed (constraint violation)!" << std::endl;
    return 1;
  }

  return 0;
}
