#include "KOMO/komo.h"
#include "Kin/F_qFeatures.h"
#include "car_utils.h"
#include <Kin/F_pose.h>
#include <Kin/kin.h>
#include <cassert>

#include <map>
#include <numeric>
#include <vector>

// Some code i found on the internet. Super easy,
// seems to be OK
/**
 * Provides a basic interpolation mechanism in C++ using the STL.
 * Maybe not the fastest or most elegant method, but it works (for
 * linear interpolation!!), and is fast enough for a great deal of
 * purposes. It's also super easy to use, so that's a bonus.
 */

template <typename Point> class LinearInterpolator {
public:
  using Map = std::map<double, Point>;
  LinearInterpolator() {}

  void addDataPoint(double x, const Point &d) { data[x] = d; }

  // template <typename AddOperator> Point interpolate(double x, AddOperator add)
  Point interpolate(double x) {
    // loop through all the keys in the map
    // to find one that is greater than our intended value
    typename Map::iterator it = data.begin();
    bool found = false;
    while (it != data.end() && !found) {
      if (it->first >= x) {
        found = true;
        break;
      }

      // advance the iterator
      it++;
    }

    // check to see if we're outside the data range
    if (it == data.begin()) {
      return data.begin()->second;
    } else if (it == data.end()) {
      // move the point back one, as end() points past the list
      it--;
      return it->second;
    }
    // check to see if we landed on a given point
    else if (it->first == x) {
      return it->second;
    }

    // nope, we're in the range somewhere
    // collect some values
    double xb = it->first;
    Point yb = it->second;
    it--;
    double xa = it->first;
    Point ya = it->second;

    // and calculate the result!
    // formula from Wikipedia
    double r = (x - xa) / (xb - xa);
    return r * yb + (1. - r) * ya;
  }

  Map data;
};

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

static arrA getPath_qAll_with_prefix(KOMO &komo, int order) {
  arrA q(komo.T + order);
  for (int t = -order; t < int(komo.T); t++) {
    q(t + order) = komo.getConfiguration_qAll(t);
  }
  return q;
}

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
  bool regularize_traj = true;
  int order = 1;
  arrA waypoints;
};

// note: KOMO komo; komo.setModel( ); komo.setTiming should be done before
void create_komo(KOMO &komo, const TrailerOpt &opt) {

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

  bool regularize_traj = true;
  if (regularize_traj && opt.waypoints.N) {
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

    if (opt.velocity_bound)
    // Bound Linear Velocity
    {
      komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name},
                        OT_ineq, {1}, {max_velocity}, 1);

      komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name},
                        OT_ineq, {-1}, {-max_velocity}, 1);
      // Bound angle on wheel
    } else {
      komo.addObjective({}, make_shared<UnicycleVelocity>(), {car_name}, OT_sos,
                        {100}, {}, 1);
      komo.addObjective({}, FS_qItself, {wheel_name}, OT_sos, {100}, {}, 1);
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

int main_trailer() {

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

  rai::Configuration C;
  C.addFile(model_file);

  int order = rai::getParameter<int>("order", 1);
  std::cout << "Car order: " << order << std::endl;

  KOMO komo;
  komo.setModel(C, true);

  arrA waypoints;
  if (waypoints_file != "none") {
    // load initial guess
    YAML::Node env = YAML::LoadFile((const char *)waypoints_file);
    const auto &node = env["result"][0];
    size_t num_states = node["states"].size();

    // the initial guess has states: x,y,theta0,theta1v and actions: v, phi
    // KOMO uses 5 states: x,y,theta0,phi,theta1q

    double latest_theta = 3.14;
    for (size_t i = 1; i < num_states; ++i) {
      const auto &state = node["states"][i];

      auto x = state[0].as<double>();
      auto y = state[1].as<double>();
      auto theta0 = state[2].as<double>();

      //
      double theta0_plus = theta0 + 2. * M_PI;
      double theta0_minus = theta0 - 2. * M_PI;
      // check the difference in abs value
      // double dif = std::abs(latest_theta - theta0);
      // double dif_plus = std::abs(latest_theta - theta0_plus);
      // double dif_minus = std::abs(latest_theta - theta0_minus);

      // if (dif_plus < dif) {
      //   theta0 = theta0_plus;
      //   dif = dif_plus;
      // }
      // if (dif_minus < dif) {
      //   theta0 = theta0_minus;
      //   dif = dif_minus;
      // }

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

  double dt = 0.1;
  double duration_phase = N * dt;
  komo.setTiming(1, N, duration_phase, order);

  TrailerOpt opt;
  opt.goal_eq = true;
  opt.velocity_bound = false;
  create_komo(komo, opt);

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
  auto it_v0 =
      std::max_element(actions.begin(), actions.end(), [](auto &v, auto &v2) {
        return std::abs(v.at(0)) < std::abs(v2.at(0));
      });

  const double max_velocity = 0.5 * action_factor - 0.01; // m/s
  std::vector<double> factors_vel;
  std::transform(actions.begin(), actions.end(),
                 std::back_inserter(factors_vel),
                 [&](auto &a) { return std::abs(a.at(0)) / max_velocity; });

  std::cout << "Factors" << std::endl;
  for (auto &s : factors_vel) {
    std::cout << s << " " << std::endl;
  }

  std::vector<double> factors_vel_smoothed(factors_vel.size());

  for (size_t i = 0; i < factors_vel.size(); i++) {

    if (i == 0) {
      // factors_vel_smoothed[i] = .5 * (factors_vel[i] + factors_vel[i + 1]);
      factors_vel_smoothed[i] = std::max(factors_vel[i], factors_vel[i + 1]);

    } else if (i == factors_vel.size() - 1) {
      factors_vel_smoothed[i] = std::max(factors_vel[i - 1], factors_vel[i]);
      // factors_vel_smoothed[i] = .5 * (factors_vel[i - 1] + factors_vel[i]);
    } else {
      // factors_vel_smoothed[i] =
      //     (factors_vel[i] + factors_vel[i + 1] + factors_vel[i - 1]) / 3.0;
      //
      factors_vel_smoothed[i] = std::max(
          factors_vel[i], std::max(factors_vel[i + 1], factors_vel[i - 1]));
    }
  }
  std::cout << "Factors smoothed" << std::endl;
  for (auto &s : factors_vel_smoothed) {
    std::cout << s << " " << std::endl;
  }

  double total_time = std::accumulate(factors_vel_smoothed.begin(),
                                      factors_vel_smoothed.end(), 0);
  std::cout << "total time " << total_time << std::endl;

  std::cout << "Convert to time indices" << std::endl;

  std::vector<double> times_(factors_vel_smoothed.size());
  std::vector<double> times(factors_vel_smoothed.size());
  std::partial_sum(factors_vel_smoothed.begin(), factors_vel_smoothed.end(),
                   times_.begin());
  std::cout << "Partial  sum" << std::endl;
  for (auto &s : times_) {
    std::cout << s << " " << std::endl;
  }

  std::transform(times_.begin(), times_.end(), times.begin(),
                 [&](auto &s) { return dt * s; });

  std::cout << "Partial  sum 2" << std::endl;
  for (auto &s : times) {
    std::cout << s << " " << std::endl;
  }

  LinearInterpolator<arr> lerp;

  for (size_t i = 0; i < times.size(); i++) {
    lerp.addDataPoint(times.at(i), newwaypoints(i));
  }
  // TAKE the full time:
  // interpolate the numbers.

  int num_steps = std::ceil(times_.back());
  arrA new_waypoints_scaled;
  for (size_t i = 0; i < num_steps; i++) {
    // int nx = waypoints(0).N;
    // arr point(nx);
    // for (size_t j = 0; j < nx; j++)
    //   point(j) = lerp.interpolate(i * dt, j);
    new_waypoints_scaled.append(lerp.interpolate(i * dt));
  }

  std::cout << "new waypoints scaled" << std::endl;
  std::cout << new_waypoints_scaled << std::endl;

  // return 0;
  double max_vel = it_v0->at(0);
  std::cout << "max vel is " << max_vel << std::endl;
  std::cout << "bound vel is  " << max_velocity << std::endl;
  double factor = std::abs(max_vel) / max_velocity;
  std::cout << "Factor is " << factor << std::endl;
  std::cout << "Estimated time " << komo.T * dt * factor << std::endl;
  std::cout << "Repeat and make sure that it is solvable " << std::endl;
  // const double max_velocity = 0.5 * action_factor - 0.01; // m/s

  // can I solve this with more time?

  KOMO komo2;
  komo2.setModel(C, true);

#if 0
  double duration_phase2 = 5 * N * dt;
  komo2.setTiming(1, 5 * N, duration_phase2, order);
#endif

  double duration_phase2 = new_waypoints_scaled.N * dt;
  komo2.setTiming(1, new_waypoints_scaled.N, duration_phase2, order);

  TrailerOpt opt2;
  opt2.goal_eq = true;
  opt2.velocity_bound = true;
  // opt2.regularize_traj = fa;
  opt2.regularize_traj = true;
  opt2.waypoints = new_waypoints_scaled;
  create_komo(komo2, opt2);
  komo2.initWithWaypoints(new_waypoints_scaled, new_waypoints_scaled.N);

  std::cout << "before second optimization" << std::endl;
  if (display) {
    komo2.view(true);
    komo2.view_play(true);
    komo2.plotTrajectory();

    do {
      cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
  }

  komo2.run_prepare(.05);
  komo2.run();

  if (display) {
    komo2.view(true);
    komo2.view_play(true);
    // komo2.view_play(true, 1,"vid/car");
    komo2.plotTrajectory();

    do {
      cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
  }

  komo2.reportProblem();

  if (display) {
    komo2.view(true);
    komo2.view_play(true);
    // komo2.view_play(true, 1,"vid/car");
    komo2.plotTrajectory();

    do {
      cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
  }

  auto report2 = komo2.getReport(display, 0, std::cout);
  std::cout << "report " << report2 << std::endl;
  ineq = report2.get<double>("ineq") / komo.T;
  eq = report2.get<double>("eq") / komo.T;
  if (ineq > 0.01 || eq > 0.01) {
    // Optimization failed (constraint violations)
    std::cout << "Optimization failed (constraint violation)!" << std::endl;
    return 1;
  }

  arrA results2 = getPath_qAll_with_prefix(komo2, order);
  std::cout << "(N,T): " << results2.N << " " << komo2.T << std::endl;

  std::vector<std::vector<double>> actions2;
  for (size_t t = order; t < results2.N; ++t) {
    auto &v = results2(t);
    actions2.push_back({velocity(results2, t, dt), v(3)});
    out << "      - [" << velocity(results2, t, dt) << "," << v(3) << "]"
        << std::endl;
  }
  auto it_v02 =
      std::max_element(actions2.begin(), actions2.end(), [](auto &v, auto &v2) {
        return std::abs(v.at(0)) < std::abs(v2.at(0));
      });
  double max_vel2 = it_v02->at(0);
  std::cout << "max vel is " << max_vel2 << std::endl;
  std::cout << "bound vel is  " << max_velocity << std::endl;
  double factor2 = std::abs(max_vel2) / max_velocity;
  std::cout << "Factor is " << factor2 << std::endl;
  std::cout << "Estimated time " << komo2.T * dt * factor2 << std::endl;
  std::cout << "Repeat and make sure that it is solvable " << std::endl;

  // Hey, I could reescale better right?
  // I only need to enlarge a local path?

  // IDEA 1: Better rescaling of dt.
  // create array of velocity factors. Compute the new time from that .
  // Mapping to generate initial guess.

  // IDEA 2: lets try the reciding horizon. How long?
  // Lets say 1 second.


  // lets try ideas about the receding horizon.


  return 0;
}
