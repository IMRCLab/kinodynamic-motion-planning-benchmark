#include "car_utils.h"
#include "Core/util.h"
#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include <Kin/kin.h>
#include <cassert>
#include <cmath>
#include <iostream>

//#include <iomanip>

#include <yaml-cpp/yaml.h>

void get_speed(arr &y, arr &J, const FrameL &F) {

  const double tol = 0.1; // tolerance to avoid division by zero

  for (auto &f : F)
    assert(f->joint->type == rai::JT_transXYPhi);

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

/* xdot = V cos ( theta ) */
/* ydot = V sin ( theta ) */
/* theta_dot = u */

/* V is car speed */
/* u is change of angular rate */

// Model dubins with 2 non linear equations:
// V cos(theta) - xdot = 0
// V sin(theta) - ydot = 0
void UnicycleDynamics::phi2(arr &y, arr &J, const FrameL &F) {

  const double tol = 0.1; // tolerance to avoid division by zero

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

  arr speed, Jspeed;
  get_speed(speed, Jspeed, F);

  // std::cout << speed << std::endl;
  // std::cout << v << std::endl;
  y.resize(2);
  y(0) = c_theta * speed(0) - v(0); // cos V - xdot = 0
  y(1) = s_theta * speed(0) - v(1); // sin V - ydot = 0

  if (!!J) {
    arr Jl;
    Jl.resize(2, 7); // ROWS = 2 equations ; COLUMNS= 3 position +  3
                     // velocities + 1 speed
    Jl.setZero();

    // w.r.t theta
    Jl(0, 2) = -s_theta * speed(0);
    Jl(1, 2) = c_theta * speed(0);

    // w.r.t sv
    Jl(0, 3) = -1;
    Jl(1, 4) = -1;

    // w.r.t speed
    Jl(0, 6) = c_theta;
    Jl(1, 6) = s_theta;

    arr JBlock;
    JBlock.setBlockMatrix(Jp, Jv);
    arr out;
    out.setBlockMatrix(JBlock, Jspeed);

    J = Jl * out;
  }
}

void UnicycleVelocity::phi2(arr &y, arr &J, const FrameL &F) {

  // implementation only for rai::JT_transXYPhi!
  for (auto &f : F) {
    assert(f->joint->type == rai::JT_transXYPhi);
  }

  get_speed(y, J, F);
};

void UnicycleAcceleration::phi2(arr &y, arr &J, const FrameL &F) {

  if (order != 2)
    throw std::runtime_error("error");

  for (auto &f : F) {
    assert(f->joint->type == rai::JT_transXYPhi);
  }

  arr p, v, Jp, Jv;
  arr pprev, vprev, Jpprev, Jvprev;

  arr speedPrev, speed;
  arr JspeedPrev, Jspeed;
  get_speed(speedPrev, JspeedPrev, F({0, 1}));
  get_speed(speed, Jspeed, F({1, 2}));

  F_qItself().setOrder(0).eval(p, Jp, F[2].reshape(1, -1));
  F_qItself().setOrder(1).eval(v, Jv, F({1, 2}));
  F_qItself().setOrder(0).eval(pprev, Jpprev, F[1].reshape(1, -1));
  F_qItself().setOrder(1).eval(vprev, Jvprev, F({0, 1}));

  y.resize(1);
  y(0) = speed(0) - speedPrev(0);

  if (!!J) {
    J = Jspeed - JspeedPrev;
  }
};

void UnicycleAngularVelocity::phi2(arr &y, arr &J, const FrameL &F) {
  const double dt = 0.1;

  // implementation only for rai::JT_transXYPhi!
  for (auto &f : F) {
    assert(f->joint->type == rai::JT_transXYPhi);
  }

  // p: position = [x,y,theta]
  // v: velocity = [vx,vy,vtheta]
  arr p, v, Jp, Jv;
  F_qItself().setOrder(0).eval(p, Jp, F[1].reshape(1, -1));
  F_qItself().setOrder(1).eval(v, Jv, F);

  double angular_change = atan2(sin(v(2) * dt), cos(v(2) * dt));
  double angular_velocity = angular_change / dt; // rad/s

  // feature is y
  y.resize(1);
  y(0) = angular_velocity;

  // std::cout << "v " << speed << std::endl;

  // compute Jacobian
  if (!!J) {
    arr Jl;
    // ROWS = 1 equations ; COLUMNS= 3 position + 3 velocities
    Jl.resize(1, 6);
    Jl.setZero();
    // w.r.t theta_dot
    Jl(0, 5) = 1;

    arr JBlock;
    JBlock.setBlockMatrix(Jp, Jv);
    J = Jl * JBlock;
  }
}

void UnicycleAngularAcceleration::phi2(arr &y, arr &J, const FrameL &F) {
  const double dt = 0.1;

  assert(order == 2);

  for (auto &f : F) {
    assert(f->joint->type == rai::JT_transXYPhi);
  }

  arr p, v, Jp, Jv;
  arr pprev, vprev, Jpprev, Jvprev;

  F_qItself().setOrder(0).eval(p, Jp, F[2].reshape(1, -1));
  F_qItself().setOrder(1).eval(v, Jv, F({1, 2}));
  F_qItself().setOrder(0).eval(pprev, Jpprev, F[1].reshape(1, -1));
  F_qItself().setOrder(1).eval(vprev, Jvprev, F({0, 1}));

  double angular_change = atan2(sin(v(2) * dt), cos(v(2) * dt));
  double angular_velocity = angular_change / dt; // rad/s

  double angular_change_prev = atan2(sin(vprev(2) * dt), cos(vprev(2) * dt));
  double angular_velocity_prev = angular_change_prev / dt; // rad/s

  y.resize(1);
  y(0) = (angular_velocity - angular_velocity_prev) / dt;

  if (!!J) {
    arr Jl;
    // ROWS = 1 equations ; COLUMNS= 3 p + 3 v + 3 pprev + 3 vprev
    Jl.resize(1, 12);
    Jl.setZero();
    // w.r.t. theta_dot
    Jl(0, 5) = 1 / dt;
    // w.r.t. theta_dot_prev
    Jl(0, 11) = -1 / dt;

    arr JBlock_a;
    JBlock_a.setBlockMatrix(Jp, Jv);
    arr JBlock_b;
    JBlock_b.setBlockMatrix(Jpprev, Jvprev);
    arr JBlock;
    JBlock.setBlockMatrix(JBlock_a, JBlock_b);
    J = Jl * JBlock;
  }
}

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
