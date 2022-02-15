#include "Core/util.h"
#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include "car_utils.h"
#include <Kin/F_pose.h>
#include <Kin/kin.h>
#include <cassert>

#include "solvers.hpp"
#include <map>
#include <numeric>
#include <vector>

// Some code i found on the internet. seems to be OK
/**
 * Provides a basic interpolation mechanism in C++ using the STL.
 * Maybe not the fastest or most elegant method, but it works (for
 * linear interpolation!!), and is fast enough for a great deal of
 * purposes. It's also super easy to use, so that's a bonus.
 */

double dt = 0.1;
int order = 1;

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
    y(0) = tdot(0) - ang_vel(2) - vel(0) / d1 * ct;

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
      Jl(0, 2) = -1;            // w.r.t angvel(2)
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

struct TrailerOpt {
  bool goal_eq = true;
  bool velocity_bound = true;
  bool regularize_traj = false;
  int order = 1;
  arrA waypoints;
};

// NOTE: KOMO komo; komo.setModel( ); komo.setTiming should be called before
void create_komo_trailer(KOMO &komo, const TrailerOpt &opt) {

  double action_factor = rai::getParameter<double>("action_factor", 1.0);
  const double L = .4;  // distance  rear-front wheels
  const double d1 = .5; // distance between car centers
  const double max_velocity = 0.5 * action_factor - 0.01; // m/s
  const double max_phi = M_PI / 3;

  auto robot_collision = "R_robot_shape";
  auto car_name = "R_robot";
  auto goal_name = "GOAL_robot";
  auto arm_name = "R_arm";
  auto wheel_name = "R_front_wheel";
  auto trailer_name = "R_trailer";
  auto trailer_goal = "GOAL_trailer";

#if 0
  if (opt.regularize_traj && opt.waypoints.N) {
    double scale_regularization = .1; // try different scales
    int it = 1;
    // ways -> N+1
    // N
    for (const arr &a : opt.waypoints) // i take from index=1 because we
                                       // are ignoring the first waypoint.
    {
      komo.addObjective(double(it) *
                            arr{1. / opt.waypoints.N, 1. / opt.waypoints.N},
                        FS_qItself, {}, OT_sos, {scale_regularization}, a, 0);
      it++;
    }
  }
#endif

  if (opt.regularize_traj) {
    double scale_regularization = 2; // try different scales
    // int it = 1;
    // ways -> N+1
    // N
    for (size_t i = 1; i < komo.T; i++) {
      komo.addObjective(double(i) / komo.T * arr{1., 1.}, FS_poseDiff,
                        {car_name, "REG_robot"}, OT_sos,
                        {scale_regularization});
      komo.addObjective(double(i) / komo.T * arr{1., 1.}, FS_poseDiff,
                        {trailer_name, "REG_trailer"}, OT_sos,
                        {scale_regularization});
    }

    // for (const arr &a : opt.waypoints) // i take from index=1 because we
    //                                    // are ignoring the first waypoint.
    // {
    //   komo.addObjective(double(it) *
    //                         arr{1. / opt.waypoints.N, 1. / opt.waypoints.N},
    //                     FS_qItself, {}, OT_sos, {scale_regularization}, a,
    //                     0);
    //   it++;
    // }
  }

  komo.addObjective({}, FS_qItself, {arm_name}, OT_sos, {.1}, {}, 1);
  komo.addObjective({}, FS_qItself, {wheel_name}, OT_sos, {.1}, {}, 1);
  komo.addObjective({}, make_shared<F_LinAngVel>(), {car_name}, OT_sos, {.1},
                    {}, 1);

  // add the goal
  if (!opt.goal_eq) {
    komo.addObjective({1., 1.}, FS_poseDiff, {car_name, goal_name}, OT_sos,
                      {10});
    komo.addObjective({1., 1.}, FS_poseDiff, {trailer_name, trailer_goal},
                      OT_sos, {10});
  } else {
    komo.addObjective({1., 1.}, FS_poseDiff, {car_name, goal_name}, OT_eq,
                      {1e2});
    komo.addObjective({1., 1.}, FS_poseDiff, {trailer_name, trailer_goal},
                      OT_eq, {1e2});
  }

  // add collisions
  StringA obstacles;
  for (auto &frame : komo.world.frames) {
    std::cout << *frame << std::endl;
    if (frame->shape && frame->name.startsWith("obs")) {
      obstacles.append(frame->name);
    }
  }

  for (auto &obs : obstacles) {
    komo.addObjective({}, FS_distance, {robot_collision, obs}, OT_ineq, {1e2});
    komo.addObjective({}, FS_distance, {trailer_name, obs}, OT_ineq, {1e2});
  }

  if (opt.order == 1) {

    // Linear velocity First Car
    komo.addObjective({}, make_shared<UnicycleDynamics>(), {car_name}, OT_eq,
                      {1e1}, {0}, 1);

    // Rotation First Car
    komo.addObjective({}, make_shared<FirstCarRotation>(L),
                      {car_name, wheel_name}, OT_eq, {1e1}, {0}, 1);

    // Rotation Trailer
    komo.addObjective({}, make_shared<Trailer>(d1), {car_name, arm_name}, OT_eq,
                      {1e1}, {0}, 1);

    if (opt.velocity_bound) {
      komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name},
                        OT_ineq, {1}, {max_velocity}, 1);

      komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name},
                        OT_ineq, {-1}, {-max_velocity}, 1);
    } else {
      komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name}, OT_sos,
                        {1}, {}, 1);
    }
    komo.addObjective({}, FS_qItself, {wheel_name}, OT_ineq, {1}, {max_phi},
                      -1);

    komo.addObjective({}, FS_qItself, {wheel_name}, OT_ineq, {-1}, {-max_phi},
                      -1);

  } else {
    NIY;
  }
};

// running parallel park with
// quim@fourier ~/s/w/k/b/debug.10-02-2022--11-18-40 (time_sos)
// ../main_rai -model env.g -waypoints init.yaml -N -1 -display 1
//  -animate 2 -order 1 -robot car_first_order_with_1_trailers_0
// -cfg "rai.cfg" -env \"../benchmark/car_first_order_with_1_trai
// lers_0/parallelpark_0.yaml\" -out out.yaml

// returns pair { feasible, waypoints }
// feasible is  true or false (Full trajectory feasible).
// waypoints: shortest solution

// returns pair { feasible, waypoints }
// waypoints: biggest feasible starting trajectory

void write_results_tailer_1(const arrA &results, const char *out_file) {

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
  std::vector<std::vector<double>> actions;
  for (size_t t = order; t < results.N; ++t) {
    auto &v = results(t);
    actions.push_back({velocity(results, t, dt), v(3)});
    out << "      - [" << velocity(results, t, dt) << "," << v(3) << "]"
        << std::endl;
  }
};

int main_trailer() {

  arrA results = {};
  bool feasible = false;

  rnd.clockSeed();
  double action_factor = rai::getParameter<double>("action_factor", 1.0);
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

  auto car_name = "R_robot";
  auto trailer_name = "R_trailer";

  rai::Configuration C;
  C.addFile(model_file);
  C["GOAL_robot_shape"]->setColor({1, 0, 0, 0.8});

  rai::Frame *f = C["REG_robot"];

  if (f == nullptr)
    f = C.addFrame("REG_robot");
  f->setShape(rai::ST_marker, {.5}).setColor({1., 1., 0, .5});

  f->set_X() = C[car_name]->ensure_X();
  rai::Frame *f2 = C["REG_trailer"];
  if (f2 == nullptr)
    f2 = C.addFrame("REG_trailer");
  f2->setShape(rai::ST_marker, {.5}).setColor({1., 0., 1, .5});
  f2->set_X() = C[trailer_name]->ensure_X();

  // (robot){ shape:ssBox, Q:<t(.2 0 0 )> , size:[.5 .25 .5 .005],color:[.1 .1
  // .2 .4]   }

  // int order = rai::getParameter<int>("order", 1);
  std::cout << "Car order: " << order << std::endl;

  arrA waypoints;
  if (waypoints_file != "none") {
    // load initial guess
    YAML::Node env = YAML::LoadFile((const char *)waypoints_file);
    const auto &node = env["result"][0];
    size_t num_states = node["states"].size();

    // the initial guess has states: x,y,theta0,theta1v and actions: v, phi
    // KOMO uses 5 states: x,y,theta0,phi,theta1q

    double latest_theta = node["states"][1][2].as<double>();
    for (size_t i = 1; i < num_states; ++i) {
      const auto &state = node["states"][i];

      auto x = state[0].as<double>();
      auto y = state[1].as<double>();
      auto theta0 = state[2].as<double>();

      //
      double theta0_plus = theta0 + 2. * M_PI;
      double theta0_minus = theta0 - 2. * M_PI;
      // check the difference in abs value
      double dif = std::abs(latest_theta - theta0);
      double dif_plus = std::abs(latest_theta - theta0_plus);
      double dif_minus = std::abs(latest_theta - theta0_minus);

      if (dif_plus < dif) {
        theta0 = theta0_plus;
        dif = dif_plus;
      }
      if (dif_minus < dif) {
        theta0 = theta0_minus;
        dif = dif_minus;
      }

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

  std::string mode = rai::getParameter<rai::String>("mode", "normal").p;

  bool normal_mode = false;
  bool time_trick = false;
  bool binary_search_time = false;
  bool horizon_mode = false;
  bool horizon_with_binary_search = false;

  if (mode == "normal") {
    normal_mode = true;
  } else if (mode == "time_trick") {
    time_trick = true;
  } else if (mode == "search_time") {
    binary_search_time = true;
  } else if (mode == "horizon") {
    horizon_mode = true;
  } else if (mode == "horizon_binary_search") {
    horizon_with_binary_search = true;
  }

  if (normal_mode) {
    KOMO komo;
    komo.setModel(C, true);

    double duration_phase = N * dt;
    komo.setTiming(1, N, duration_phase, order);

    TrailerOpt opt;
    opt.goal_eq = true;
    opt.velocity_bound = true;
    create_komo_trailer(komo, opt);

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

    komo.run_prepare(.1);
    komo.run();

    komo.reportProblem();
    arrA newwaypoints = komo.getPath_qAll();

    if (display) {
      komo.view(true);
      komo.view_play(true);
      // komo.view_play(true, 1,"vid/car");
      komo.plotTrajectory();

      do {
        cout << '\n' << "Press a key to continue...";
      } while (std::cin.get() != '\n');
    }

    feasible = is_feasible(komo);
    results = getPath_qAll_with_prefix(komo, order);

    std::cout << "results: " << std::endl;
    std::cout << C.getFrameNames() << std::endl;
    std::cout << results << std::endl;
    std::cout << "(N,T): " << results.N << " " << komo.T << std::endl;
  }

  if (time_trick) {

    auto compute_time_rescaling = [](const arrA &results) {
      // NOTE: results = getPath_qAll_with_prefix(komo, order);
      std::vector<std::vector<double>> actions;
      arrA waypoints = results({1, -1});
      for (size_t t = order; t < results.N; ++t) {
        actions.push_back({velocity(results, t, dt), results(t)(3)});
      }

      // factor between velocity and max velocity at each time step
      double action_factor = rai::getParameter<double>("action_factor", 1.0);
      const double max_velocity = 0.5 * action_factor - 0.01; // m/s
      std::vector<double> factors_vel;
      std::transform(actions.begin(), actions.end(),
                     std::back_inserter(factors_vel),
                     [&](auto &a) { return std::abs(a.at(0)) / max_velocity; });

      // smoothed factors (kind of moving average)
      std::vector<double> factors_vel_smoothed(factors_vel.size());
      {
        std::function<double(const std::vector<double> &)> reduce =
            reduce_max<double>;

        for (size_t i = 0; i < factors_vel.size(); i++) {
          if (i == 0) {
            factors_vel_smoothed[i] =
                reduce({factors_vel[i], factors_vel[i + 1]});
          } else if (i == factors_vel.size() - 1) {
            factors_vel_smoothed[i] =
                reduce({factors_vel[i - 1], factors_vel[i]});
          } else {
            factors_vel_smoothed[i] = reduce(
                {factors_vel[i], factors_vel[i + 1], factors_vel[i - 1]});
          }
        }
      }

      // new time stamps
      std::vector<double> times(factors_vel_smoothed.size());
      {
        std::vector<double> times_(factors_vel_smoothed.size());
        std::partial_sum(factors_vel_smoothed.begin(),
                         factors_vel_smoothed.end(), times_.begin());
        std::cout << "Partial  sum" << std::endl;
        for (auto &s : times_) {
          std::cout << s << " " << std::endl;
        }
        std::transform(times_.begin(), times_.end(), times.begin(),
                       [&](auto &s) { return dt * s; });
      }

      // interpolate the new waypoints
      LinearInterpolator<arr> lerp;
      CHECK_EQ(times.size(), waypoints.N, "");
      lerp.addDataset(times.data(), waypoints.p, times.size());

      int num_steps = std::ceil(times.back() / dt);
      arrA new_waypoints_scaled;
      for (size_t i = 0; i < num_steps; i++) {
        new_waypoints_scaled.append(lerp.interpolate(i * dt));
      }

      std::cout << "new waypoints scaled" << std::endl;
      std::cout << new_waypoints_scaled << std::endl;

      return new_waypoints_scaled;
    };

    auto set_komo_without_vel = [&](KOMO &komo) {
      TrailerOpt opt;
      opt.goal_eq = true;
      opt.velocity_bound = false;
      opt.regularize_traj = false;
      create_komo_trailer(komo, opt);
    };

    auto set_komo_with_vel = [&](KOMO &komo) {
      TrailerOpt opt2;
      opt2.goal_eq = true;
      opt2.velocity_bound = true;
      opt2.regularize_traj = false;
      // TODO: solve the waypoint stuff
      // opt2.waypoints = new_waypoints_scaled;
      create_komo_trailer(komo, opt2);
    };

    solve_with_time_trick(waypoints, C, dt, order, set_komo_without_vel,
                          compute_time_rescaling, set_komo_with_vel);

    // const double max_velocity = 0.5 * action_factor - 0.01; // m/s

    // can I solve this with more time?
  }

  if (binary_search_time) {

    int min_waypoints = waypoints.N / 2;
    int max_waypoints = waypoints.N * 5;
    int increment = 1;

    auto out = komo_binary_search_time(waypoints, min_waypoints, max_waypoints,
                                       increment, dt, C, [&](KOMO &komo) {
                                         TrailerOpt opt;
                                         opt.goal_eq = true;
                                         opt.velocity_bound = true;
                                         opt.regularize_traj = false;
                                         create_komo_trailer(komo, opt);
                                       });
    std::cout << "OUT" << std::endl;
    std::cout << out.first << std::endl;
    std::cout << out.second.N << std::endl;
  }

  int horizon = rai::getParameter<int>("num_h", 100);
  if (horizon_mode) {
    // TODO: last waypoint should be goal, right?
    arr start;
    KOMO komoh, komo_hard;
    arr prefix = C.getJointState();
    // get the waypint
    // TODO: Get the waypoint from somewhere. Important,
    {
      komoh.setModel(C, true);
      komoh.setTiming(1, horizon, dt * horizon, 1);
      TrailerOpt opth;
      opth.goal_eq = false;
      opth.velocity_bound = true;
      opth.regularize_traj = false;
      create_komo_trailer(komoh, opth);
    }

    {
      komo_hard.setModel(C, true);
      komo_hard.setTiming(1, horizon, dt * horizon, 1);
      TrailerOpt opt_hard;
      opt_hard.goal_eq = true;
      opt_hard.velocity_bound = true;
      create_komo_trailer(komo_hard, opt_hard);
    }

    start = komoh.getConfiguration_qAll(-1);

    std::cout << "start " << start << std::endl;
    std::cout << "calling solver" << std::endl;
    komoh.run_prepare(0);
    // komoh.view(true);
    auto out_iterative = iterative_komo_solver(
        waypoints, horizon, komoh, komo_hard, start, set_start,
        [&](auto &komo, const auto &arr, bool true_goal) {
          return set_goal(C, komo, arr, horizon, true_goal);
        });
    feasible = out_iterative.first;
    std::cout << "feasible " << feasible << std::endl;
    std::cout << "NUM waypoints " << out_iterative.second.N << std::endl;
    std::cout << "result " << out_iterative.second << std::endl;
    for (auto &a : out_iterative.second) {
      std::cout << a << std::endl;
    }

    bool double_check = true;
    if (feasible && double_check) {
      {
        std::cout << "DOUBLE CHECK SOLUTION OF ITERATIVE" << std::endl;
        auto ways = out_iterative.second;
        KOMO komoh;
        C.setJointState(start);
        komoh.setModel(C, true);
        komoh.setTiming(1, ways.N, dt * ways.N, 1);
        TrailerOpt opth;
        opth.goal_eq = true;
        opth.velocity_bound = true;
        create_komo_trailer(komoh, opth);
        komoh.initWithWaypoints(ways, ways.N);
        komoh.run_prepare(0);
        komoh.view(true);
        komoh.view_play(true);
        komoh.plotTrajectory();

        do {
          cout << '\n' << "Press a key to continue...";
        } while (std::cin.get() != '\n');
      }

      auto sparse = komoh.nlp_SparseNonFactored();
      arr phi;
      sparse->evaluate(phi, NoArr, komoh.x);
      CHECK(is_feasible(komoh), "");
    }

    results.append(prefix);
    results.append(out_iterative.second);
  }

  // this will do horizon, and then a binary search with the last N steps.
  if (horizon_with_binary_search) {

    arr start;
    KOMO komoh, komo_hard;

    {
      komoh.setModel(C, true);
      komoh.setTiming(1, horizon, dt * horizon, 1);
      TrailerOpt opth;
      opth.goal_eq = false;
      opth.velocity_bound = true;
      opth.regularize_traj = false;
      create_komo_trailer(komoh, opth);
    }

    {
      komo_hard.setModel(C, true);
      komo_hard.setTiming(1, horizon, dt * horizon, 1);
      TrailerOpt opt_hard;
      opt_hard.goal_eq = true;
      opt_hard.velocity_bound = true;
      create_komo_trailer(komo_hard, opt_hard);
    }

    start = komoh.getConfiguration_qAll(-1);

    std::cout << "start " << start << std::endl;
    std::cout << "calling solver" << std::endl;
    komoh.run_prepare(0);
    if (display) {
      komoh.view(true);
    }
    auto out_iterative =
        iterative_komo_solver(waypoints, horizon, komoh, komo_hard, start,
                              set_start, [&](auto &komo, const auto &arr, bool true_goal) {
                                return set_goal(C, komo, arr, horizon, true_goal);
                              });
    std::cout << "feasible " << out_iterative.first << std::endl;
    std::cout << "NUM waypoints " << out_iterative.second.N << std::endl;
    std::cout << "result " << out_iterative.second << std::endl;
    arrA waypoints_first = out_iterative.second;
    for (auto &a : out_iterative.second) {
      std::cout << a << std::endl;
    }

    if (display) {
      for (auto &a : out_iterative.second) {
        C.setJointState(a);
        C.watch(true);
      }
    }
    // throw -1;

    // Lets optimize from

    int waypoint_index = -1; // -1 is last

    // lets use the lastest horizon waypoints.
    arrA waypoints_bin = waypoints({-horizon + waypoint_index, -1});

    int min_waypoints = waypoints_bin.N / 2;
    int max_waypoints = waypoints_bin.N * 5;
    int increment = 1;

    // change the starting point:
    C.setJointState(out_iterative.second(waypoint_index));

    // which are the waypoints?
    auto out =
        komo_binary_search_time(waypoints_bin, min_waypoints, max_waypoints,
                                increment, dt, C, [&](KOMO &komo) {
                                  TrailerOpt opt;
                                  opt.goal_eq = true;
                                  opt.velocity_bound = true;
                                  create_komo_trailer(komo, opt);
                                });
    feasible = out.first;
    std::cout << "out.first " << feasible << std::endl;
    std::cout << "out.second.N " << out.second.N << std::endl;

    // lets glue the trajectory

    if (display) {
      for (auto &a : out.second) {
        C.setJointState(a);
        C.watch(true);
      }

      waypoints_first.append(out.second);
      for (auto &a : waypoints_first) {
        C.setJointState(a);
        C.watch(true);
      }
    }
  }

  if (feasible) {
    write_results_tailer_1(results, out_file);
    return 0;
  } else {
    return 1;
  }

  // TODO: test everything here and in bugtrap.
  // Binary Search
  // Time trick
  // Receding horizon
}
