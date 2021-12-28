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
  double velocity = std::sqrt(v(0) * v(0) + v(1) * v(1)); // velocity

  // feature is y
  y.resize(2);
  y(0) = cos(theta) * velocity - v(0); // cos V - xdot = 0
  y(1) = sin(theta) * velocity - v(1); // sin V - ydot = 0

  // compute Jacobian
  if (!!J) {
    double tol = 1e-6; // tolerance non differentiable point of sqrt()
    arr Jl;
    Jl.resize(2, 6); // ROWS = 2 equations ; COLUMNS= 3 position * 3 velocities
    Jl.setZero();
    if (velocity > tol) {
      // w.r.t theta
      Jl(0, 2) = -std::sin(theta) * velocity;
      Jl(1, 2) = std::cos(theta) * velocity;
      // w.r.t vx
      Jl(0, 3) = std::cos(theta) / velocity * v(0) - 1;
      Jl(1, 3) = std::sin(theta) / velocity * v(0);
      // w.r.t vy
      Jl(0, 4) = std::cos(theta) / velocity * v(1);
      Jl(1, 4) = std::sin(theta) / velocity * v(1) - 1;
    } else {
      Jl(0, 2) = 0.0;
      Jl(1, 2) = 0.0;

      // in non differentiable point, I take gradient of v > 0
      Jl(0, 3) = std::cos(theta) - 1;
      Jl(1, 3) = std::sin(theta);

      Jl(0, 4) = std::cos(theta);
      Jl(1, 4) = std::sin(theta) - 1;
    }

    arr JBlock;
    JBlock.setBlockMatrix(Jp, Jv);
    J = Jl * JBlock;
  }
}

struct CarFirstOrderVelocity : Feature
{
  uint dim_phi2(const FrameL &)
  {
    return 1;
  }

  void phi2(arr &y, arr &J, const FrameL &F)
  {
    // implementation only for rai::JT_transXYPhi!
    for (auto &f : F) {
      assert(f->joint->type == rai::JT_transXYPhi);
    }

    // p: position = [x,y,theta]
    // v: velocity = [vx,vy,vtheta]
    arr p, v, Jp, Jv;
    F_qItself().setOrder(0).eval(p, Jp, F[1].reshape(1, -1));
    F_qItself().setOrder(1).eval(v, Jv, F);
    double velocity = std::sqrt(v(0) * v(0) + v(1) * v(1)); // velocity

    // feature is y
    y.resize(1);
    y(0) = velocity;

    // compute Jacobian
    if (!!J) {
      double tol = 1e-6; // tolerance non differentiable point of sqrt()
      arr Jl;
      Jl.resize(1, 6); // ROWS = 1 equations ; COLUMNS= 3 position + 3 velocities
      Jl.setZero();
      if (velocity > tol)
      {
        // w.r.t vx
        Jl(0, 3) = v(0) / velocity;
        // w.r.t vy
        Jl(0, 4) = v(1) / velocity;
      }

      arr JBlock;
      JBlock.setBlockMatrix(Jp, Jv);
      J = Jl * JBlock;
    }
  }
};

arrA load_waypoints(const char *filename) {

  // load initial guess
  YAML::Node env = YAML::LoadFile(filename);

  // load states
  arrA waypoints;
  for (const auto &state : env["result"][0]["states"])
  {
    arr stateArray;
    for (const auto& elem : state) {
      stateArray.append(elem.as<double>());
    }
    waypoints.append(stateArray);
  }
  return waypoints;
}

// usage:
// EXECUTABLE -model FILE_G -waypoints FILE_WAY -one_every ONE_EVERY_N -display
// {0,1} -out OUT_FILE OUT_FILE -animate  {0,1,2}

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

  // create optimization problem
  KOMO komo;
  komo.setModel(C, true);
  double duration_phase = waypoints.N * 0.1; // dt is 0.1 s
  komo.setTiming(1, waypoints.N, duration_phase, 2);

  komo.add_qControlObjective({}, 2, .1);
  komo.add_qControlObjective({}, 1, .1);
  // I assume names robot0 and goal0 in the .g file
  komo.addObjective({1., 1.}, FS_poseDiff, {"robot0", "goal0"}, OT_eq, {1e2});

  // robot dynamics
  komo.addObjective({}, make_shared<Dubins2>(), {"robot0"}, OT_eq, {1e1}, {0},
                    1);
  // angular velocity limit
  komo.addObjective({}, FS_qItself, {"robot0"}, OT_ineq, {0, 0, 1}, {0, 0, 0.5 /*rad/s*/}, 1);
  komo.addObjective({}, FS_qItself, {"robot0"}, OT_ineq, {0, 0, -1}, {0, 0, -0.5 /*rad/s*/}, 1);
  // velocity limit
  komo.addObjective({}, make_shared<CarFirstOrderVelocity>(), {"robot0"}, OT_ineq, {1}, {0.5 /*m/s*/}, 1);
  komo.addObjective({}, make_shared<CarFirstOrderVelocity>(), {"robot0"}, OT_ineq, {-1}, {-0.5 /*m/s*/}, 1);

  for (auto &obs : obstacles) {
    komo.addObjective({}, FS_distance, {"robot0", obs}, OT_ineq, {1e2});
  }

  komo.run_prepare(0.1); // TODO: is this necessary?
  komo.initWithWaypoints(waypoints, waypoints.N);

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

  bool check_gradients = false;
  if (check_gradients) {
    std::cout << "checking gradients" << std::endl;
    komo.checkGradients();
    std::cout << "done " << std::endl;
  }

  if (animate) {
    komo.opt.animateOptimization = animate;
  }

  komo.optimize(0.1);

  std::cout << "report after solve" << std::endl;
  komo.reportProblem();

  auto report = komo.getReport(display, 0, std::cout);
  std::cout << "report " << report << std::endl;
  // std::cout << "ineq: " << report.getValuesOfType<double>("ineq") << std::endl;
  double ineq = report.get<double>("ineq");
  double eq = report.get<double>("eq");

  if (display) {
    // komo.view(true);
    komo.view_play(true);
    // komo.view_play(false, .3, "z.vid/");
  }

  if (ineq > 0.1 || eq > 0.1) {
    // Optimization failed (constraint violations)
    return 1;
  }

  // write the results.
  arrA results = komo.getPath_qAll();
  std::ofstream out(out_file);
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (auto &v : results) {
    out << "      - [" << v(0) << "," << v(1) << "," << std::remainder(v(2), 2 * M_PI) << "]" << std::endl;
  }
  out << "    actions:" << std::endl;
  for (size_t t = 1; t < komo.T; ++t) {
    arr delta = results(t) - results(t-1);
    double distance = std::sqrt(delta(0) * delta(0) + delta(1) * delta(1)); // velocity
    double velocity = distance / 0.1; // m/s
    double angular_change = atan2(sin(delta(2)), cos(delta(2)));
    double angular_velocity = angular_change / 0.1; // rad/s
    out << "      - [" << velocity << "," << angular_velocity << "]" << std::endl;
  }

  return 0;
}
