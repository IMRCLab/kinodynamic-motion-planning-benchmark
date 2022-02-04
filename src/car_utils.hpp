#pragma once
#include "Core/util.h"
#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include <Kin/kin.h>
#include <cassert>
#include <cmath>
#include <iostream>
//#include <iomanip>

#include <yaml-cpp/yaml.h>

void get_speed(arr &y, arr &J, const FrameL &F);

/* xdot = V cos ( theta ) */
/* ydot = V sin ( theta ) */
/* theta_dot = u */

/* V is car speed */
/* u is change of angular rate */

struct UnicycleDynamics : Feature {
  // Model dubins with 2 non linear equations:
  // V cos(theta) - xdot = 0
  // V sin(theta) - ydot = 0
  void phi2(arr &y, arr &J, const FrameL &F);
  uint inline dim_phi2(const FrameL &F) {
    (void)F;
    return 2;
  }
};

struct UnicycleVelocity : Feature {
  inline uint dim_phi2(const FrameL &) { return 1; }

  void phi2(arr &y, arr &J, const FrameL &F);
};

struct UnicycleAcceleration : Feature {
  inline uint dim_phi2(const FrameL &) { return 1; }

  void phi2(arr &y, arr &J, const FrameL &F);
};

struct UnicycleAngularVelocity : Feature {
  inline uint dim_phi2(const FrameL &) { return 1; }

  void phi2(arr &y, arr &J, const FrameL &F);
};

struct UnicycleAngularAcceleration : Feature {
  inline uint dim_phi2(const FrameL &) { return 1; }
  void phi2(arr &y, arr &J, const FrameL &F);
};

arrA load_waypoints(const char *filename);
