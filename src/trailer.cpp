
#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include "car_utils.hpp"
#include <Kin/kin.h>
#include <cassert>

const double L = .2;   // distance  rear-front wheels
const double d1 = .35; // distance between car centers

#if 0
// feature
struct CarDynamics : Feature {
  // xdot = s cos ( theta )
  // ydot = s sin ( theta )
  // thetadot = s / L tan ( phi )
  void phi2(arr &y, arr &J, const FrameL &F) {

    CHECK_EQ(F.nd, 2, "need two frames");

    CHECK_EQ(F(0, 0)->joint->type, rai::JT_transXYPhi, "");
    CHECK_EQ(F(0, 1)->joint->type, rai::JT_hingeZ, "");
    CHECK_EQ(F(1, 0)->joint->type, rai::JT_transXYPhi, "");
    CHECK_EQ(F(1, 1)->joint->type, rai::JT_hingeZ, "");

    std::cout << "frames " << std::endl;
    std::cout << F.nd << std::endl;
    std::cout << F(0, 0)->name << std::endl; // robot
    std::cout << F(0, 1)->name << std::endl; // front wheel
    for (auto &f : F) {
      std::cout << f->name << std::endl;
    }

    std::cout << "**** " << std::endl;

    // robot
    arr r, Jr;
    arr rdot, Jdot;
    F_qItself().setOrder(0).eval(r, Jr, FrameL{F(1, 0)}.reshape(1, -1));
    F_qItself().setOrder(1).eval(rdot, Jdot,
                                 FrameL{F(0, 0), F(1, 0)}.reshape(-1, 1));

    // phi
    arr phi;
    arr Jphi;
    F_qItself().setOrder(0).eval(phi, Jphi, F(1, {1, 1}).reshape(1, -1));

    std::cout << "r " << r << std::endl;
    std::cout << "Jr " << Jr << std::endl;
    std::cout << "rdot " << rdot << std::endl;
    std::cout << "Jdot " << Jdot << std::endl;
    std::cout << "phi " << phi << std::endl;
    std::cout << "Jphi " << Jphi << std::endl;

    // y(0,1) are the typical velocity stuff...
    // y(2):   dot_theta - 1 / L * vel * tan ( phi ) = 0

    y = zeros(3);
    if (!!J) {
      // ROWS = 3 equations ; COLUMNS= 3 position + 3 velocities + phi
      J.resize(3, 7);

      // Jac of y(2)
      // Jdot_theta-1/L*vel*1/cos(phi)^2*Jphi-1/L*tan(phi)*Jvel
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    return 3;
  }
};
#endif

struct Trailer : Feature {
  // See Quim's Drawing:
  // theta1DOT - s / d1 * cos (theta) = 0
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
    arr rdot, Jdot;
    F_qItself().setOrder(0).eval(r, Jr, FrameL{F(1, 0)}.reshape(1, -1));
    F_qItself().setOrder(1).eval(rdot, Jdot,
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
    y(0) = tdot(0) - vel(0) / d1 * ct;

    std::cout << "y out " << y.N << y << std::endl;
    if (!!J) {
      // ROWS = 1 equations ; COLUMNS= 1 t +  1 tdot + 1 vel
      arr Jl;
      Jl.resize(1, 3);
      Jl(0, 0) = vel(0) / d1 * st;
      Jl(0, 1) = 1;
      Jl(0, 2) = -1 / d1 * ct;

      arr block_, block;
      block_.setBlockMatrix(Jt, Jtdot);
      block.setBlockMatrix(block_, Jvel);
      J = Jl * block;
      std::cout << "returning Jac " << std::endl;
      std::cout << J << std::endl;
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    std::cout << "dim is " << 1 << std::endl;
    return 1;
  }
};

struct FirstCarRotation : Feature {

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

int main() {

  auto filename = "../src/car_with_trailer.g";
  rai::Configuration C;
  C.addFile(filename);

  KOMO komo;
  komo.setModel(C, true);

  komo.setTiming(1, 20, 5, 2);
  // komo.add_qControlObjective({},2,.1);
  komo.add_qControlObjective({}, 2, 1);
  komo.add_qControlObjective({}, 1, 1);

  // Position First Car
  // V cos(theta) - xdot = 0
  // V sin(theta) - ydot = 0
  komo.addObjective({}, make_shared<UnicycleDynamics>(), {"robot"}, OT_eq,
                    {1e1}, {0}, 1);

  // Rotation First Car
  // theta_dot = s / L * tan(phi)
  komo.addObjective({}, make_shared<FirstCarRotation>(),
                    {"robot", "front_wheel"}, OT_eq, {1e1}, {0}, 1);

  // Rotation Trailer
  komo.addObjective({}, make_shared<Trailer>(), {"robot", "arm"}, OT_eq, {1e1},
                    {0}, 1);

  // komo.addObjective({1., 1.}, FS_poseDiff, {"robot", "goal_forward"}, OT_eq,
  //                   {1e2}); SEEMS OK

  komo.addObjective({1., 1.}, FS_poseDiff, {"robot", "goal_curve"}, OT_eq,
                    {1e2});

  komo.run_prepare(0);
  komo.reportProblem();
  // komo.checkGradients(); // it breaks here, not giving the jacobian
  //
  //

  // komo.initWithWaypoints( {

  // goal_forward{ shape:marker, size:[.3], X:<t(2 .2 0) d(0 0 0 1)> }

  komo.run();

  komo.reportProblem();

  komo.view(true);
  komo.view_play(true);

  komo.plotTrajectory();

  do {
    cout << '\n' << "Press a key to continue...";
  } while (std::cin.get() != '\n');
}
