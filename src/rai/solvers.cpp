#include "solvers.hpp"

arrA getPath_qAll_with_prefix(KOMO &komo, int order) {
  arrA q(komo.T + order);
  for (int t = -order; t < int(komo.T); t++) {
    q(t + order) = komo.getConfiguration_qAll(t);
  }
  return q;
}

void komo_setConfiguration_X_name(KOMO &komo, int t, const arr &X,
                                  const char *name) {

  auto frames = komo.timeSlices[komo.k_order + t];
  auto it = std::find_if(frames.begin(), frames.end(),
                         [&](auto &f) { return strcmp(name, f->name.p) == 0; });
  CHECK(it != frames.end(), "frame is not found");
  std::cout << (*it)->name() << std::endl;
  komo.pathConfig.setFrameState(X, {*it});
}

arr komo_getConfiguration_X_name(KOMO &komo, int t, const char *name) {
  auto frames = komo.timeSlices[komo.k_order + t];
  auto it = std::find_if(frames.begin(), frames.end(),
                         [&](auto &f) { return strcmp(name, f->name.p) == 0; });
  CHECK(it != frames.end(), "frame is not found");
  return komo.pathConfig.getFrameState({*it});
}

bool is_feasible(KOMO &komo) {
  bool feasible;
  auto report = komo.getReport(false, 0, std::cout);
  std::cout << "report " << report << std::endl;
  double ineq = report.get<double>("ineq") / komo.T;
  double eq = report.get<double>("eq") / komo.T;
  if (ineq > 0.01 || eq > 0.01) {
    // Optimization failed (constraint violations)
    std::cout << "Optimization failed (constraint violation)!" << std::endl;
    feasible = false;
  } else {
    feasible = true;
  }
  return feasible;
}

// returns {feasible,waypoints}
std::pair<bool, arrA>
solve_with_time_trick(const arrA &waypoints, rai::Configuration &C, double dt,
                      int order,
                      std::function<void(KOMO &)> set_komo_without_vel,
                      std::function<arrA(const arrA &)> compute_time_rescaling,
                      std::function<void(KOMO &)> set_komo_with_vel) {
  CHECK_EQ(order, 1, "");

  bool display = true;
  // solve first without time constraints

  KOMO komo;
  komo.setModel(C, true);
  komo.setTiming(1, waypoints.N, waypoints.N * dt, order);
  set_komo_without_vel(komo);
  komo.initWithWaypoints(waypoints, waypoints.N);
  // update the regularization

  komo.view(true);
  update_reg(komo);
  // komo.initWithConstant(waypoints(0));
  komo.run_prepare(0.05);
  // double check

  std::cout << "check before" << std::endl;
  auto sparse = komo.nlp_SparseNonFactored();
  arr phi;
  sparse->evaluate(phi, NoArr, komo.x);

  komo.reportProblem();
  rai::Graph report = komo.getReport(display, 0, std::cout);
  std::cout << "report " << report << std::endl;

  komo.run();
  bool feasible = is_feasible(komo);
  if (!feasible) {
    std::cout << "Warning: Not feasible without velocity constraints"
              << std::endl;
    return {false, waypoints};
  }

  // arrA sol_ = komo.getPath_qAll();
  arrA sol_with_prefix = getPath_qAll_with_prefix(komo, order);
  arrA init_reescaled = compute_time_rescaling(sol_with_prefix);

  KOMO komo2;
  komo2.setModel(C, true);
  komo2.setTiming(1, init_reescaled.N, init_reescaled.N * dt, order);
  set_komo_with_vel(komo2);
  komo2.initWithWaypoints(init_reescaled, init_reescaled.N);
  update_reg(komo2);
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
    komo2.plotTrajectory();
    do {
      cout << '\n' << "Press a key to continue...";
    } while (std::cin.get() != '\n');
  }

  feasible = is_feasible(komo2);
  if (feasible) {
    std::cout << "Feasible after time reescaling" << std::endl;
    std::cout << "num waypoints " << komo2.getPath_qAll().N << std::endl;
    return {feasible, komo2.getPath_qAll()};
  } else {
    std::cout << "Warning: Infeasible after time reescaling" << std::endl;
    return {feasible, {}};
  }
}

void set_start(KOMO &komo, const arr &s) {
  std::cout << "setting start" << std::endl;
  std::cout << "s " << s << std::endl;
  komo.setConfiguration_qAll(-1, s);
  // komo.initWithConstant(s);
  // std::cout << "viewing" << std::endl;
  // komo.view(true);
  // std::cout << "done" << std::endl;
}

// run after the waypoints
void update_reg(KOMO &komo) {

  // compute the configuration of
  std::vector<std::pair<std::string, std::string>> pairs{
      {"REG_robot", "R_robot"}, {"REG_trailer", "R_trailer"}};

  for (auto &pp : pairs) {
    for (uint i = 0; i < komo.T; i++) {
      auto X = komo_getConfiguration_X_name(komo, i, pp.second.c_str());
      komo_setConfiguration_X_name(komo, i, X, pp.first.c_str());
    }
    // komo.view(true);
  }

  std::cout << "view after update reg" << std::endl;
  komo.view(true);

  // for (int i = -1; i < horizon; i++) {
  //   komo_setConfiguration_X_name(komo, i, Xtrailer.reshape(1, -1),
  //                                trailer_goal);
  // }
  // komo.view(true);
}

void set_goal(rai::Configuration &Cref, KOMO &komo, const arr &s, int horizon) {

  // compute the configuration of

  auto robot_collision = "R_robot_shape";
  auto car_name = "R_robot";
  auto goal_name = "GOAL_robot";
  auto arm_name = "R_arm";
  auto arm_goal = "GOAL_arm";
  auto wheel_name = "R_front_wheel";
  auto trailer_name = "R_trailer";
  auto trailer_goal = "GOAL_trailer";
  std::cout << "goal " << std::endl;
  std::cout << s << std::endl;
  std::cout << "seeing goal" << std::endl;
  Cref.setJointState(s);
  // Cref.watch(true);
  std::cout << "done" << std::endl;
  Cref.getFrameState();
  arr Xcar = Cref[car_name]->get_X().getArr7d();
  arr Xtrailer = Cref[trailer_name]->get_X().getArr7d();
  arr Xarm = Cref[arm_name]->get_X().getArr7d();
  std::cout << "Xcar " << Xcar << std::endl;
  std::cout << "Xtrailer " << Xtrailer << std::endl;

  for (int i = -1; i < horizon; i++) {
    komo_setConfiguration_X_name(komo, i, Xcar.reshape(1, -1), goal_name);
  }
  // komo.view(true);

  for (int i = -1; i < horizon; i++) {
    komo_setConfiguration_X_name(komo, i, Xarm.reshape(1, -1), arm_goal);
  }

  // komo.view(true);

  // for (int i = -1; i < horizon; i++) {
  //   komo_setConfiguration_X_name(komo, i, Xtrailer.reshape(1, -1),
  //                                trailer_goal);
  // }
  // komo.view(true);

}

std::pair<bool, arrA> komo_binary_search_time(
    const arrA &waypoints, int min_waypoints, int max_waypoints, int increment,
    double dt, rai::Configuration C, std::function<void(KOMO &)> set_komo) {

  int num_elements = (max_waypoints - min_waypoints) / increment + 1;
  std::vector<int> num_waypoints_vec(num_elements);
  {
    int counter = 0;
    std::generate(num_waypoints_vec.begin(), num_waypoints_vec.end(), [&]() {
      counter += increment;
      return counter;
    });
  }
  std::map<int, arrA> feasible_sols;
  for (auto &s : num_waypoints_vec) {
    std::cout << s << std::endl;
  }
  bool visualize = rai::getParameter<bool>("display", false);
  auto it = std::lower_bound(
      num_waypoints_vec.begin(), num_waypoints_vec.end(), true,
      [&](auto &i, auto &__notused) {
        (void)__notused;
        KOMO komo;
        komo.setModel(C);
        komo.setTiming(1, i, i * dt, 1);
        set_komo(komo);
        if (feasible_sols.size() == 0)
          komo.initWithWaypoints(waypoints, waypoints.N);
        else {
          auto &first = *feasible_sols.begin();
          // TODO: Check if it is okay if I have more waypoints
          // than time steps?
          komo.initWithWaypoints(first.second, first.second.N);
        }

        komo.run_prepare(0);
        if (visualize) {
          komo.view(true);
          // komo.view_play(true);
          komo.plotTrajectory();
          do {
            cout << '\n' << "Press a key to continue...";
          } while (std::cin.get() != '\n');
        }

        komo.run_prepare(.001);
        komo.run();
        if (visualize) {
          komo.view(true);
          // komo.view_play(true);
        }
        bool feasible = is_feasible(komo);
        std::cout << "NUM " << i << " IS FEASIBLE " << feasible << std::endl;
        if (feasible) {
          feasible_sols.insert({i, komo.getPath_qAll()});
        }
        return !feasible; // I will find the first feasible=TRUE
      });

  if (it == num_waypoints_vec.end()) {
    std::cout << "WARNING: Problem is infeasible " << std::endl;
    return {false, {}};
  } else {
    std::cout << "Problem is feasible " << std::endl;
    arrA sol = feasible_sols[*it];
    return {true, sol};
  }
}

std::pair<bool, arrA> iterative_komo_solver(
    const arrA &waypoints, int horizon, KOMO &komo, KOMO &komo_hard,
    const arr &start, std::function<void(KOMO &, const arr &)> set_start,
    std::function<void(KOMO &, const arr &)> set_goal,
    std::function<double(const arr &, const arr &)> distance_fun) {

  arrA results_out;

  int it = 0;
  int it_max = 1000;
  std::cout << "NUM waypoints " << waypoints.N << std::endl;

  arr pos_start = start;
  bool feasible = true;
  bool visualize = rai::getParameter<bool>("display", false);
  bool project = false;
  int ub, lb;
  arr pos_goal;
  bool finished = false;
  // 0 1 2 || 3 4 5 || 6 7 8
  // lb = 0 ; ub = 2;
  // lb = 3 = ub = 5
  while (!finished && horizon * it < int(waypoints.N) && it < it_max &&
         feasible) {
    if (!project) {
      lb = it * horizon;
    } else {
      auto it = std::min_element(
          waypoints.begin(), waypoints.end(), [&](auto &a, auto &b) {
            return distance_fun(a, pos_start) < distance_fun(b, pos_start);
          });
      lb = std::distance(waypoints.begin(), it) + 1;
    }
    ub = lb + horizon - 1;
    if (ub > int(waypoints.N - 1)) {
      ub = int(waypoints.N - 1);
      finished = true;
    }

    arr pos_goal = waypoints(ub);

    KOMO *kptr;
    if (!finished) {
      kptr = &komo;
    } else {
      kptr = &komo_hard;
    }

    arrA ways = waypoints({lb, ub});
    if (finished) {
      std::cout << "waypoints" << std::endl;
      for (auto &w : ways) {
        std::cout << w << std::endl;
      }
    }

    set_start(*kptr, pos_start);
    set_goal(*kptr, pos_goal);

    std::cout << "lb, ub " << lb << " " << ub << std::endl;

    kptr->initWithConstant(pos_start);
    kptr->initWithWaypoints(ways, ways.N);
    // kptr->initWithConstant(pos_start);
    // kAWaypoints(ways, ways.N);
    kptr->run_prepare(.001);
    if (visualize) {
      std::cout << "before opti" << std::endl;
      kptr->view(true);
      kptr->view_play(true);
      kptr->plotTrajectory();
      arrA init = getPath_qAll_with_prefix(*kptr, 1);
      for (auto &i : init) {
        std::cout << i << std::endl;
      }
      do {
        cout << '\n' << "Press a key to continue...";
      } while (std::cin.get() != '\n');
    }

    kptr->run();
    if (visualize) {
      std::cout << "after opti" << std::endl;
      kptr->view(true);
      kptr->view_play(true);
    }

    auto report = kptr->getReport(false, 0, std::cout);
    std::cout << "report " << report << std::endl;
    double ineq = report.get<double>("ineq") / kptr->T;
    double eq = report.get<double>("eq") / kptr->T;
    if (ineq > 0.01 || eq > 0.01) {
      // Optimization failed (constraint violations)
      std::cout << "Optimization failed (constraint violation)!" << std::endl;
      feasible = false;
    } else {
      arrA sol = kptr->getPath_qAll();
      results_out.append(sol);
      pos_start = sol(-1);
    }

    it++;
  }
  return {feasible, results_out};
}
