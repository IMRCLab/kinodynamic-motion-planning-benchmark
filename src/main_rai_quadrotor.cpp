#include <KOMO/komo.h>
#include <Kin/TM_default.h>
#include <Kin/F_collisions.h>
#include <Kin/F_qFeatures.h>
#include <Kin/viewer.h>
#include <Kin/F_pose.h>
#include <Kin/F_forces.h>
#include <Kin/forceExchange.h>

#include <yaml-cpp/yaml.h>


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
  // rai::String waypoints_file =
  //     rai::getParameter<rai::String>("waypoints", STRING("none"));

  // bool display = rai::getParameter<bool>("display", false);
  // int animate = rai::getParameter<int>("animate", 0);
  // rai::String out_file =
  //     rai::getParameter<rai::String>("out", STRING("out.yaml"));

  
  // rai::String env_file =
  //     rai::getParameter<rai::String>("env", STRING("none"));

  // arrA waypoints = load_waypoints(waypoints_file);

  // load G file
  rai::Configuration C;
  C.addFile(model_file);

  // Update inertia to match Crazyflie model
  auto& inertia = C["drone"]->getInertia();
  inertia.setZero();
  inertia.mass = 0.034;
  inertia.matrix.setZero();
  inertia.matrix.m00 = 16.571710e-6;
  inertia.matrix.m11 = 16.655602e-6;
  inertia.matrix.m22 = 29.261652e-6;

  const float force_to_torque = 0.006; // force-to-torque ratio
  const float max_force_per_motor = 12. / 1000. * 9.81;

  KOMO komo;
  komo.setTiming(1., 50, 0.5, 2);

  rai::Frame *world = C["world"];
  auto f1 = new rai::ForceExchange(*world, *C["m1"], rai::FXT_forceZ);
  f1->limits = {-max_force_per_motor * komo.tau, 0.};
  f1->force_to_torque = -force_to_torque;

  auto f2 = new rai::ForceExchange(*world, *C["m2"], rai::FXT_forceZ);
  f2->limits = {-max_force_per_motor * komo.tau, 0.};
  f2->force_to_torque = +force_to_torque;

  auto f3 = new rai::ForceExchange(*world, *C["m3"], rai::FXT_forceZ);
  f3->limits = {-max_force_per_motor * komo.tau, 0.};
  f3->force_to_torque = -force_to_torque;

  auto f4 = new rai::ForceExchange(*world, *C["m4"], rai::FXT_forceZ);
  f4->limits = {-max_force_per_motor * komo.tau, 0.};
  f4->force_to_torque = +force_to_torque;

  //  cout <<C.getLimits() <<endl;
  //  return;

  uint qDim = C.getJointStateDimension();
  CHECK_EQ(qDim, 7 + 4, "dofs don't add up");

  komo.setModel(C);
  komo.addSquaredQuaternionNorms();

  //target
  komo.addObjective({.9,1.}, FS_positionDiff, {"drone", "target"}, OT_eq, {1e2});
  komo.addObjective({1.}, FS_qItself, {"drone"}, OT_eq, {1e2}, {}, 1);

  //NE & starting smoothly
  komo.addObjective({0.}, FS_pose, {"drone"}, OT_eq, {1e2}, {}, 1, +0, +1);
//  komo.addObjective({0.}, make_shared<F_LinAngVel>(), {"drone"}, OT_eq, {1e2}, {}, 1, +0, +1);
  komo.addObjective({}, make_shared<F_NewtonEuler>(true), {"drone"}, OT_eq, {1e0}, {}, 2, +2, -1);

  //force z-aligned
  //obsolete by construction

  // force and torque aligned
  //obsolete by construction

  //force costs
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m1"}, OT_sos, {1e-1}, {}, 1, +2);
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m2"}, OT_sos, {1e-1}, {}, 1, +2);
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m3"}, OT_sos, {1e-1}, {}, 1, +2);
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m4"}, OT_sos, {1e-1}, {}, 1, +2);

  //smoothness in control - helps condition the problem a lot!
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m1"}, OT_sos, {1e-2}, {}, 2, +2);
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m2"}, OT_sos, {1e-2}, {}, 2, +2);
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m3"}, OT_sos, {1e-2}, {}, 2, +2);
  komo.addObjective({}, make_shared<F_fex_Force>(), {"world", "m4"}, OT_sos, {1e-2}, {}, 2, +2);

  //limits using lagrange terms instead of bounds (is softer, somtimes converges better)
  komo.addObjective({}, make_shared<F_qLimits>(), {"world"}, OT_ineq, {1e3});

  //force inequalities
  //obsolete by limits

  // bounded control effort
  //obsolete by limits

  komo.reportProblem();

//  komo.animateOptimization=1;
  komo.optimize(0.);
//  komo.plotTrajectory();
  cout <<komo.getReport(true);
//  komo.checkGradients();


  //pick a 'drone' frame-triplet in the middle of the path
  // evaluate NE - and debug into it to look into numbers...
  uint t=komo.T/2;
  uint droneId = C["drone"]->ID;
  FrameL drones = komo.timeSlices.sub(komo.k_order + t-2, komo.k_order+t, droneId, droneId);
  Value ne = F_NewtonEuler(true).eval(drones);
  cout <<ne.y <<endl;

  arr allDofs = komo.x;
  allDofs.reshape(komo.T, 4+7);

  // write output csv file
  ofstream csv("data.csv");
  csv << "f1, f2, f3, f4, x, y, z, qw, qx, qy, qz\n";
  allDofs.write(csv, ", ", "\n", "");
  csv.close();

  gnuplot("load 'plt'");

  komo.view(true);
  while(komo.view_play(true, 1.));
  komo.view_play(false,.2,"z.vid/");

  return 0;
}
