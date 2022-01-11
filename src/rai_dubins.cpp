#include "Core/util.h"

#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include <Kin/kin.h>
#include <cassert>
#include <cmath>
#include <iostream>

#include <yaml-cpp/yaml.h>

/* xdot = V cos ( theta ) */
/* ydot = V sin ( theta ) */
/* theta_dot = u */

/* V is car speed */
/* u is change of angular rate */

struct Dubins2 : Feature {
  // Model dubins with 2 non linear equations:
  // V cos(theta) - xdot = 0
  // V sin(theta) - ydot = 0
  void phi2(arr &y, arr &J, const FrameL &F);
  uint dim_phi2(const FrameL &F) {
    (void)F;
    return 2;
  }
};

void Dubins2::phi2(arr &y, arr &J, const FrameL &F) {

  const double tol = 1e-4; // tolerance to avoid division by zero

  // implementation only for rai::JT_transXYPhi!
  for (auto &f : F) {
    assert(f->joint->type == rai::JT_transXYPhi);
  }

  // p: position = [x,y,theta]
  // v: velocity = [vx,vy,vtheta]
  arr p, v, Jp, Jv;
  F_qItself().setOrder(0).eval(p, Jp, F[1].reshape(1, -1));
  F_qItself().setOrder(1).eval(v, Jv, F);
  double theta = p(2);

  double c_theta = cos(theta);
  double s_theta = sin(theta);
  double speed;
  int mode;
  if (fabs(c_theta) > tol) {
    mode = 1;
    speed = v(0) / c_theta;
  } else {
    mode = 2;
    speed = v(1) / s_theta;
  }

  // feature is y
  y.resize(2);
  y(0) = c_theta * speed - v(0); // cos V - xdot = 0
  y(1) = s_theta * speed - v(1); // sin V - ydot = 0

  // compute Jacobian
  if (!!J) {
    arr Jl;
    Jl.resize(2, 6); // ROWS = 2 equations ; COLUMNS= 3 position * 3 velocities
    Jl.setZero();
    if (mode == 1) {
      // w.r.t theta
      Jl(0, 2) = 0;
      Jl(1, 2) = v(0) * 1 / c_theta * 1 / c_theta;
      // w.r.t vx
      Jl(0, 3) = 0;
      Jl(1, 3) = s_theta / c_theta;
      // w.r.t vy
      Jl(0, 4) = 0;
      Jl(1, 4) = -1;
    } else if (mode == 2) {
      // w.r.t theta
      Jl(0, 2) = -v(1) * 1 / s_theta * 1 / s_theta;
      Jl(1, 2) = 0;
      // w.r.t vx
      Jl(0, 3) = -1;
      Jl(1, 3) = 0;
      // w.r.t vy
      Jl(0, 4) = c_theta / s_theta;
      Jl(1, 4) = 0;
    }

    arr JBlock;
    JBlock.setBlockMatrix(Jp, Jv);
    J = Jl * JBlock;
  }
}

struct CarFirstOrderVelocity : Feature {
  uint dim_phi2(const FrameL &) { return 1; }

  void phi2(arr &y, arr &J, const FrameL &F) {
    const double tol = 1e-4; // tolerance to avoid division by zero

    // implementation only for rai::JT_transXYPhi!
    for (auto &f : F) {
      assert(f->joint->type == rai::JT_transXYPhi);
    }

    // p: position = [x,y,theta]
    // v: velocity = [vx,vy,vtheta]
    arr p, v, Jp, Jv;
    F_qItself().setOrder(0).eval(p, Jp, F[1].reshape(1, -1));
    F_qItself().setOrder(1).eval(v, Jv, F);

    double theta = p(2);
    double c_theta = cos(theta);
    double s_theta = sin(theta);
    double speed;
    int mode;
    if (fabs(c_theta) > tol) {
      mode = 1;
      speed = v(0) / c_theta;
    } else {
      mode = 2;
      speed = v(1) / s_theta;
    }

    // feature is y
    y.resize(1);
    y(0) = speed;

    // compute Jacobian
    if (!!J) {
      arr Jl;
      // ROWS = 1 equations ; COLUMNS= 3 position + 3 velocities
      Jl.resize(1, 6);
      Jl.setZero();
      if (mode == 1) {
        // w.r.t theta
        Jl(0, 2) = v(0) * s_theta / (c_theta * c_theta);
        // w.r.t vx
        Jl(0, 3) = 1 / c_theta;
        // w.r.t vy
        Jl(0, 4) = 0;
      } else if (mode == 2) {
        // w.r.t theta
        Jl(0, 2) = -v(1) * c_theta / (s_theta * s_theta);
        // w.r.t vx
        Jl(0, 3) = 0;
        // w.r.t vy
        Jl(0, 4) = 1 / s_theta;
      }
      arr JBlock;
      JBlock.setBlockMatrix(Jp, Jv);
      J = Jl * JBlock;
    }
  }
};

struct CarSecondOrderAcceleration : Feature {
  uint dim_phi2(const FrameL &) { return 1; }

  void phi2(arr &y, arr &J, const FrameL &F) {

    if (order != 2)
      throw std::runtime_error("error");

    for (auto &f : F) {
      assert(f->joint->type == rai::JT_transXYPhi);
    }

    arr p, v, Jp, Jv;
    arr pprev, vprev, Jpprev, Jvprev;

    F_qItself().setOrder(0).eval(p, Jp, F[2].reshape(1, -1));
    F_qItself().setOrder(1).eval(v, Jv, F({1, 2}));
    F_qItself().setOrder(0).eval(pprev, Jpprev, F[1].reshape(1, -1));
    F_qItself().setOrder(1).eval(vprev, Jvprev, F({0, 1}));

    const double tol = 1e-4; // tolerance to avoid division by zero

    double theta = p(2);
    double c_theta = cos(theta);
    double s_theta = sin(theta);
    double speed;
    int mode;
    if (fabs(c_theta) > tol)
    {
      mode = 1;
      speed = v(0) / c_theta;
    }
    else
    {
      mode = 2;
      speed = v(1) / s_theta;
    }

    double thetaprev = pprev(2);
    double c_thetaprev = cos(thetaprev);
    double s_thetaprev = sin(thetaprev);
    double speedprev;
    int modeprev;
    if (fabs(c_thetaprev) > tol)
    {
      modeprev = 1;
      speedprev = vprev(0) / c_thetaprev;
    }
    else
    {
      modeprev = 2;
      speedprev = vprev(1) / s_thetaprev;
    }

    y.resize(1);
    y(0) = speed - speedprev;

    if (!!J) {
      arr Jl;
      // ROWS = 1 equations ; COLUMNS= 3 p + 3 v + 3 pprev + 3 vprev
      Jl.resize(1, 12);
      Jl.setZero();
      if (mode == 1)
      {
        // w.r.t theta
        Jl(0, 2) = v(0) * s_theta / (c_theta * c_theta);
        // w.r.t vx
        Jl(0, 3) = 1 / c_theta;
        // w.r.t vy
        Jl(0, 4) = 0;
      }
      else if (mode == 2)
      {
        // w.r.t theta
        Jl(0, 2) = -v(1) * c_theta / (s_theta * s_theta);
        // w.r.t vx
        Jl(0, 3) = 0;
        // w.r.t vy
        Jl(0, 4) = 1 / s_theta;
      }

      if (modeprev == 1)
      {
        // w.r.t theta
        Jl(0, 2+6) = -vprev(0) * s_thetaprev / (c_thetaprev * c_thetaprev);
        // w.r.t vx
        Jl(0, 3+6) = -1 / c_thetaprev;
        // w.r.t vy
        Jl(0, 4+6) = -0;
      }
      else if (modeprev == 2)
      {
        // w.r.t theta
        Jl(0, 2+6) = vprev(1) * c_thetaprev / (s_thetaprev * s_thetaprev);
        // w.r.t vx
        Jl(0, 3+6) = -0;
        // w.r.t vy
        Jl(0, 4+6) = -1 / s_thetaprev;
      }

      arr JBlock;
      // JBlock.setBlockMatrix(Jp, Jv, Jpprev, Jvprev);
      // std::cout << JBlock << std::endl;
      JBlock.resize(Jp.d0 + Jv.d0 + Jpprev.d0 + Jvprev.d0, Jp.d1);
      JBlock.setMatrixBlock(Jp, 0, 0);
      JBlock.setMatrixBlock(Jv, Jp.d0, 0);
      JBlock.setMatrixBlock(Jpprev, Jp.d0 + Jv.d0, 0);
      JBlock.setMatrixBlock(Jvprev, Jp.d0 + Jv.d0 + Jpprev.d0, 0);

      J = Jl * JBlock;
    }
  }
};

arrA load_waypoints(const char *filename) {

  // load initial guess
  YAML::Node env = YAML::LoadFile(filename);

  // load states
  arrA waypoints;
  for (const auto &state : env["result"][0]["states"]) {
    arr stateArray;
    for (const auto &elem : state) {
      stateArray.append(elem.as<double>());
      // We only care about pose, not higher-order derivatives
      if (stateArray.N == 3) {
        break;
      }
    }
    waypoints.append(stateArray);
  }
  return waypoints;
}

double velocity(const arrA& results, int t, double dt)
{
  const double tol = 1e-4; // tolerance to avoid division by zero

  arr v = results(t) - results(t-1);
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

double acceleration(const arrA &results, int t, double dt)
{
  double vel_now = velocity(results, t, dt);
  double vel_before = velocity(results, t - 1, dt);
  double acc = (vel_now - vel_before) / dt;
  return acc;
}

double angularVelocity(const arrA &results, int t, double dt)
{
  arr delta = results(t) - results(t - 1);
  double angular_change = atan2(sin(delta(2)), cos(delta(2)));
  double angular_velocity = angular_change / dt; // rad/s
  return angular_velocity;
}

double angularAcceleration(const arrA &results, int t, double dt)
{
  double omega_now = angularVelocity(results, t, dt);
  double omega_before = angularVelocity(results, t - 1, dt);
  double omega_dot = (omega_now - omega_before) / dt;
  return omega_dot;
}

arrA getPath_qAll_with_prefix(KOMO& komo, int order) {
  arrA q(komo.T+order);
  for(int t=-order; t<int(komo.T); t++) q(t+order) = komo.getConfiguration_qAll(t);
  return q;
}

    // usage:
    // EXECUTABLE -model FILE_G -waypoints FILE_WAY -one_every ONE_EVERY_N -display
    // {0,1} -out OUT_FILE OUT_FILE -animate  {0,1,2} -order {0,1,2}

    // OUT_FILE: Write down the trajectory
    // ONE_EVERY_N: take only one every N waypoints

int main(int argn, char **argv) {

  rai::initCmdLine(argn, argv);
  rnd.clockSeed();

  rai::String model_file =
      rai::getParameter<rai::String>("model", STRING("none"));
  rai::String waypoints_file =
      rai::getParameter<rai::String>("waypoints", STRING("none"));
  int one_every = int(rai::getParameter<double>("one_every", 1));

  if (one_every < 1) {
    throw std::runtime_error("one every should be >= 1");
  }

  bool display = rai::getParameter<bool>("display", false);
  int animate = rai::getParameter<int>("animate", 0);
  int order = rai::getParameter<int>("order", 0);
  rai::String out_file =
      rai::getParameter<rai::String>("out", STRING("out.yaml"));

  arrA waypoints = load_waypoints(waypoints_file);

  // downsample
  if (one_every > 1) {
    arrA waypointsS;
    for (size_t i = 0; i < waypoints.N; i++) {
      if (i % one_every == 0) {
        waypointsS.append(waypoints(i));
      }
    }
    waypoints = waypointsS;
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

  std::cout << "Warning: we will skip the first waypoint" << std::endl;
  std::cout << "waypoints are (including the first)" << std::endl;
  std::cout << waypoints << std::endl;
  // create optimization problem
  KOMO komo;
  komo.setModel(C, true);
  int Nplus1 = waypoints.N;
  int N = Nplus1 - 1;
  std::cout << "Nplus1 " << Nplus1 << std::endl;
  std::cout << "N " << N << std::endl;

  if (N==0) {
    return 1;
  }


  double dt = 0.1;
  double duration_phase = N * dt;
  komo.setTiming(1,N, duration_phase, order);

  if (order == 2)
  {
    komo.add_qControlObjective({}, 2, .1);
    // NOTE: we could also add cost on the velocity
  }

  if (order==1)
  {
    komo.add_qControlObjective({}, 1, .1);
  }

  // I assume names robot0 and goal0 in the .g file
  komo.addObjective({1., 1.}, FS_poseDiff, {"robot0", "goal0"}, OT_eq, {1e2});

  // Note: if you want position constraints on the first variable.
  // komo.addObjective({1./N, 1./N}, FS_poseDiff, {"robot0", "start0"}, OT_eq, {1e2});

  // robot dynamics
  komo.addObjective({}, make_shared<Dubins2>(), {"robot0"}, OT_eq, {1e1}, {0},
                    1);
  enum CAR_ORDER {
    ZERO = 0, // no bounds
    ONE = 1, // bounds velocity
    TWO = 2, // bound acceleration
  };
  CAR_ORDER car_order = static_cast<CAR_ORDER>(order);
  std::cout << "Car order: " << order << std::endl;

  if (car_order >= ONE) {
    const double max_velocity = 0.5; // m/s
    const double max_omega = 0.5; // rad/s

    komo.addObjective({}, FS_qItself, {"robot0"}, OT_ineq, {0, 0, 1},
                      {0, 0, max_omega}, 1);
    komo.addObjective({}, FS_qItself, {"robot0"}, OT_ineq, {0, 0, -1},
                      {0, 0, -max_omega}, 1);
    // velocity limit
    komo.addObjective({}, make_shared<CarFirstOrderVelocity>(), {"robot0"},
                      OT_ineq, {1}, {max_velocity}, 1);

    komo.addObjective({}, make_shared<CarFirstOrderVelocity>(), {"robot0"},
                      OT_ineq, {-1}, {-max_velocity}, 1);
  }
  if (car_order >= TWO) {
    double max_acceleration = 2; // m s^-2
    double max_wdot = 2;

    // NOTE: CarSecondOrderAcceleration returns v - v'
    // a = (v - v') / dt
    // so use  amax * dt as limit

    komo.addObjective({}, make_shared<CarSecondOrderAcceleration>(), {"robot0"},
                      OT_ineq, {1}, {max_acceleration * dt}, 2);

    komo.addObjective({}, make_shared<CarSecondOrderAcceleration>(), {"robot0"},
                      OT_ineq, {-1}, {-max_acceleration * dt}, 2);

    komo.addObjective({}, FS_qItself, {"robot0"}, OT_ineq, {0, 0, 1},
                      {0, 0, max_wdot}, 2);

    komo.addObjective({}, FS_qItself, {"robot0"}, OT_ineq, {0, 0, -1},
                      {0, 0, -max_wdot}, 2);

    // contraints: zero velocity start and end
    // NOTE: {0,0} seems to be ok for velocity. 
    // But why not {1/N,1/N},as in the position case?
    // komo.addObjective({0,0}, FS_qItself, {"robot0"}, OT_eq, {10},{},1);
    komo.addObjective({1./N,1./N}, FS_qItself, {"robot0"}, OT_eq, {10},{},1);
    komo.addObjective({1,1}, FS_qItself, {"robot0"}, OT_eq, {10},{},1);

    // Regularization: go slow at beginning and end
    // komo.addObjective({0,.1}, FS_qItself, {"robot0"}, OT_sos, {},{},1);
    // komo.addObjective({.9,1}, FS_qItself, {"robot0"}, OT_sos, {},{},1);
  }

  for (auto &obs : obstacles) {
    komo.addObjective({}, FS_distance, {"robot0", obs}, OT_ineq, {1e2});
  }

  komo.run_prepare(0.1); // TODO: is this necessary?
  komo.checkGradients();
  komo.initWithWaypoints(waypoints({1,-1}), N);

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
  return 5;

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

  arrA results = getPath_qAll_with_prefix(komo,order); 

  std::cout << "results: " << std::endl;
  std::cout << results << std::endl;
  std::cout << "(N,T): " << results.N << " " << komo.T << std::endl;

  if (ineq > 0.01 || eq > 0.01) {
    // Optimization failed (constraint violations)
    return 1;
  }

  // write the results.
  std::ofstream out(out_file);
  out << "result:" << std::endl;
  if (car_order == ZERO) {
    out << "  - states:" << std::endl;
    for (size_t t = order - 1; t < results.N; ++t) {
      auto&v = results(t);
      out << "      - [" << v(0) << "," << v(1) << ","
          << std::remainder(v(2), 2 * M_PI) << "]" << std::endl;
    }
  }
  else if (car_order == ONE) {
    out << "  - states:" << std::endl;
    for (size_t t = order - 1; t < results.N; ++t) {
      auto&v = results(t);
      out << "      - [" << v(0) << "," << v(1) << ","
          << std::remainder(v(2), 2 * M_PI) << "]" << std::endl;
    }
    out << "    actions:" << std::endl;
    for (size_t t = order; t < results.N; ++t) {
      out << "      - [" << velocity(results, t, dt) << "," 
          << angularVelocity(results, t, dt) << "]"
          << std::endl;
    }
  } else if (car_order == TWO) {
    out << "  - states:" << std::endl;
    for (size_t t = order-1; t < results.N; ++t)
    {
      const auto& v = results(t);
      out << "      - [" << v(0) << "," << v(1) << ","
          << std::remainder(v(2), 2 * M_PI) << ","
          << velocity(results, t, dt) << "," << angularVelocity(results, t, dt)
          << "]" << std::endl;
    }
    out << "    actions:" << std::endl;
    for (size_t t = order; t < results.N; ++t)
    {
      out << "      - [" << acceleration(results, t, dt) << "," 
          << angularAcceleration(results, t, dt) << "]"
          << std::endl;
    }
  }

  return 0;
}
