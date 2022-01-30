
#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include <Kin/kin.h>
#include <cassert>

const double L = .2;

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

struct Trailer : Feature {
  void phi2(arr &y, arr &J, const FrameL &F) {
    // equations are:
    // book says:
    // DOTtheta_1 = s / d1 * sin (theta_0 - theta_1)
    // But in our implementation, theta1 is a relative angle.
    // theta_0 + OurTheta_1 = theta_1 (we should check symbol in drawing)
    // DOTtheta_0 + DOTOurTheta_1 = DOTtheta_1
    // DOTtheta_0 + DOTOurTheta_1  - s / d1 * sin ( - OurTheta1 )  = 0

    CHECK_EQ(F.nd, 2, "need two frames");

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
    F_qItself().setOrder(1).eval(tdot, Jtdot,
                                 FrameL{F(1, 0), F(1, 1)}.reshape(-1, 1));

    if (!!J) {
    }
  }

  uint dim_phi2(const FrameL &F) {
    (void)F;
    return 1;
  }
};

int main() {

  auto filename =
      "/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/src/"
      "car_with_trailer.g";
  rai::Configuration C;
  C.addFile(filename);

  auto car_name = "robot";
  auto car_wheel_name = "front_wheel";
  auto second_car = "arm";

  KOMO komo;
  komo.setModel(C, true);

  komo.setTiming(1, 10, 5, 1);
  komo.addObjective({}, make_shared<CarDynamics>(), {car_name, car_wheel_name},
                    OT_eq, {1e1}, {0}, 1);

  komo.addObjective({}, make_shared<Trailer>(), {car_name, second_car}, OT_eq,
                    {1e1}, {0}, 1);

  komo.reportProblem();
  komo.run_prepare(.1);

  komo.checkGradients(); // it breaks here, not giving the jacobian
}
