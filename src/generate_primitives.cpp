#include "generate_primitives.hpp"
#include "croco_macros.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "ocp.hpp"

void sort_motion_primitives(
    const Trajectories &trajs, Trajectories &trajs_out,
    std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)>
        distance_fun,
    int top_k) {

  if (top_k == -1 || top_k > static_cast<int>(trajs.data.size())) {
    top_k = trajs.data.size();
  }

  for (const auto &traj : trajs.data) {
    // CSTR_V(traj.states.front());
    // CSTR_V(traj.start);
    CHECK_LEQ((traj.states.front() - traj.start).norm(), 1e-8, AT);
    CHECK_LEQ((traj.states.back() - traj.goal).norm(), 1e-8, AT);
  }

  auto goal_dist = [&](const Trajectory &a, const Trajectory &b) {
    return distance_fun(a.goal, b.goal);
  };
  auto start_dist = [&](const Trajectory &a, const Trajectory &b) {
    return distance_fun(a.start, b.start);
  };

  std::vector<std::pair<double, double>> distance_map(trajs.data.size());
  std::vector<size_t> used_motions;
  std::set<size_t> unused_motions;
  for (size_t i = 0; i < trajs.data.size(); i++) {
    unused_motions.insert(i);
  }

  // use as first/seed motion the one that moves furthest
  size_t next_best_motion = 0;
  double largest_d = 0;
  for (size_t i = 0; i < trajs.data.size(); i++) {
    auto &traj = trajs.data.at(i);
    double d = distance_fun(traj.start, traj.goal);
    if (d > largest_d) {
      largest_d = d;
      next_best_motion = i;
    }
  }
  used_motions.push_back(next_best_motion);
  unused_motions.erase(next_best_motion);

  for (auto &mi : unused_motions) {
    auto &m = trajs.data.at(mi);
    CHECK(used_motions.size(), AT);
    distance_map.at(mi).first =
        start_dist(m, trajs.data.at(used_motions.at(0)));
    distance_map.at(mi).second =
        goal_dist(m, trajs.data.at(used_motions.at(0)));
  }

  // TODO: evaluate if I should use a joint space!
  //

  CSTR_(top_k);
  for (size_t k = 1; k < top_k; ++k) {
    if (k % 1000 == 0) {
      CSTR_(k);
    }
    auto it = std::max_element(
        unused_motions.begin(), unused_motions.end(), [&](auto &a, auto &b) {
          return distance_map.at(a).first + distance_map.at(a).second <
                 distance_map.at(b).first + distance_map.at(b).second;
        });

    next_best_motion = *it;
    used_motions.push_back(*it);
    unused_motions.erase(*it);

    // update
    std::for_each(unused_motions.begin(), unused_motions.end(), [&](auto &mi) {
      distance_map.at(mi).first = std::min(
          distance_map.at(mi).first,
          start_dist(trajs.data.at(mi), trajs.data.at(next_best_motion)));

      distance_map.at(mi).second = std::min(
          distance_map.at(mi).second,
          goal_dist(trajs.data.at(mi), trajs.data.at(next_best_motion)));
    });
  }

  for (auto &i : used_motions) {
    trajs_out.data.push_back(trajs.data.at(i));
  }
}

void split_motion_primitives(const Trajectories &in, size_t num_translation,
                             Trajectories &out,
                             const Options_primitives &options_primitives) {

  std::random_device rd;  // a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()

  for (size_t i = 0; i < in.data.size(); i++) {
    const auto &traj = in.data.at(i);
    const size_t T = traj.actions.size();
    std::uniform_int_distribution<> distrib_start(
        0, T - options_primitives.min_length_cut);
    std::uniform_int_distribution<> distrib_length(
        options_primitives.min_length_cut,
        std::min(options_primitives.max_length_cut, T));

    size_t num_splits =
        std::min(std::max(int(T / options_primitives.min_length_cut), 0),
                 static_cast<int>(options_primitives.max_splits));

    for (size_t j = 0; j < num_splits; j++) {
      size_t start = distrib_start(gen);
      size_t length = distrib_length(gen);

      if (j == 0) {
        // I always generate one primitive with the original start!
        start = 0;
      }
      if (start + length > T) {
        start = T - length;
      }

      Trajectory new_traj;
      new_traj.states = std::vector<Eigen::VectorXd>{
          traj.states.begin() + start,
          traj.states.begin() + start + length + 1};
      new_traj.actions = std::vector<Eigen::VectorXd>{
          traj.actions.begin() + start, traj.actions.begin() + start + length};
      //
      Eigen::VectorXd first_state = new_traj.states.front();

      for (auto &state : new_traj.states) {
        state.head(num_translation) -= first_state.head(num_translation);
      }

      new_traj.start = new_traj.states.front();
      new_traj.goal = new_traj.states.back();
      // TODO: add cost!!

      out.data.push_back(new_traj);
    }
  }
}

void improve_motion_primitives(const Options_trajopt &options_trajopt,
                               const Trajectories &trajs_in,
                               const std::string &dynamics,
                               Trajectories &trajs_out) {

  for (size_t i = 0; i < trajs_in.data.size(); i++) {

    auto &traj = trajs_in.data.at(i);

    Problem problem;
    Trajectory traj_out;
    CHECK(traj.states.size(), AT);
    CHECK(traj.actions.size(), AT);
    problem.goal = traj.states.at(0);
    ;
    problem.start = traj.states.back();
    problem.robotType = dynamics;

    Result_opti opti_out;
    trajectory_optimization(problem, traj, options_trajopt, traj_out, opti_out);

    CHECK_EQ(opti_out.feasible, traj_out.feasible, AT);

    CSTR_(traj_out.feasible);
    if (traj_out.feasible && traj_out.cost <= traj.cost) {
      std::cout << "we have a better trajectory!" << std::endl;
      CSTR_(traj_out.cost);
      CSTR_(traj.cost);
      trajs_out.data.push_back(traj_out);
    } else {
      trajs_out.data.push_back(traj);
    }
  }
}

void generate_primitives(const Options_trajopt &options_trajopt,
                         const Options_primitives &options_primitives,
                         Trajectories &trajectories) {

  // generate an empty problem
  //

  bool finished = false;

  int num_primitives = 0;

  auto robot_model =
      robot_factory(robot_type_to_path(options_primitives.dynamics).c_str());

  size_t num_translation = robot_model->get_translation_invariance();
  Eigen::VectorXd p_lb(num_translation);
  Eigen::VectorXd p_ub(num_translation);

  p_lb.setOnes();
  p_lb *= -1;

  p_ub.setOnes();

  robot_model->setPositionBounds(p_lb, p_ub);

  auto time_start = std::chrono::steady_clock::now();
  size_t attempts = 0;
  while (!finished) {

    Eigen::VectorXd start(robot_model->nx);
    Eigen::VectorXd goal(robot_model->nx);

    robot_model->sample_uniform(start);
    robot_model->sample_uniform(goal);

    if (num_translation) {
      goal.head(num_translation) -= start.head(num_translation);
      start.head(num_translation).setZero();
    }

    std::cout << "Trying to Generate a path betweeen " << std::endl;

    // start.setZero();
    // start(6) = 1.;

    CSTR_V(start);
    CSTR_V(goal);

    Problem problem;
    problem.goal = goal;
    problem.start = start;
    problem.robotType = options_primitives.dynamics;

    // double try

    std::vector<double> try_rates{.5, 1., 2.};

    for (const auto &try_rate : try_rates) {
      Trajectory init_guess;
      init_guess.num_time_steps =
          int(try_rate * options_primitives.ref_time_steps);

      Trajectory traj;
      Result_opti opti_out;

      trajectory_optimization(problem, init_guess, options_trajopt, traj,
                              opti_out);

      if (opti_out.feasible) {
        CHECK(traj.states.size(), AT);
        traj.start = traj.states.front();
        traj.goal = traj.states.back();
        trajectories.data.push_back(traj);
        break;
      } else {
        if (options_primitives.adapt_infeas_primitives) {
          ERROR_WITH_INFO("not implemented");
        }
      }
    }

    attempts++;

    if (attempts >= options_primitives.max_attempts) {
      finished = true;
    }

    if (trajectories.data.size() >= options_primitives.max_num_primitives) {
      finished = true;
    }

    if (get_time_stamp_ms(time_start) / 1000. >=
        options_primitives.time_limit) {
      finished = true;
    }
  }

  CSTR_(attempts);
  CSTR_(trajectories.data.size());
  double success_rate = double(attempts) / trajectories.data.size();
  CSTR_(success_rate);
}
