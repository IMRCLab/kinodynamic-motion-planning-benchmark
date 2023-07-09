#include "dbrrt.hpp"
#include <boost/graph/graphviz.hpp>

#include <flann/flann.hpp>
#include <msgpack.hpp>
#include <ompl/base/spaces/SE2StateSpace.h>
#include <yaml-cpp/yaml.h>

// #include <boost/functional/hash.hpp>
#include <boost/heap/d_ary_heap.hpp>
#include <boost/program_options.hpp>

// OMPL headers
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/control/SpaceInformation.h>
#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/datastructures/NearestNeighborsFLANN.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>
#include <ompl/datastructures/NearestNeighborsSqrtApprox.h>

#include "dbastar.hpp"
#include "motions.hpp"
#include "ompl/base/Path.h"
#include "ompl/base/ScopedState.h"
#include "robot_models.hpp"
#include "robots.h"

// boost stuff for the graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/property_map/property_map.hpp>

#include "general_utils.hpp"

#include "nigh_custom_spaces.hpp"

void dbrrt(const Problem &problem, const Options_dbrrt &options_dbrrt,
           Trajectory &traj_out, Out_info_db &out_info_db) {

  std::cout << "options dbrrt" << std::endl;
  options_dbrrt.print(std::cout);
  std::cout << "***" << std::endl;

  std::shared_ptr<RobotOmpl> robot = robot_factory_ompl(problem);

  auto si = robot->getSpaceInformation();
  std::cout << "Space information" << std::endl;
  si->printSettings(std::cout);
  std::cout << "***" << std::endl;

  std::vector<Motion> motions;
  if (options_dbrrt.motions_ptr) {
    std::cout << "motions have alredy loaded " << std::endl;
    motions = *options_dbrrt.motions_ptr;
    CSTR_V(motions.at(0).traj.states.at(0));

    if (options_dbrrt.max_motions < motions.size()) {
      motions.resize(2 * options_dbrrt.max_motions);
      CSTR_V(motions.at(0).traj.states.at(0));
    }

  } else {
    std::cout << "loading motions ... " << std::endl;
    load_motion_primitives_new(
        options_dbrrt.motionsFile, *robot_factory_ompl(problem), motions,
        options_dbrrt.max_motions * 2, options_dbrrt.cut_actions, false,
        options_dbrrt.check_cols);
  }

  Time_benchmark time_bench;
  ompl::NearestNeighbors<Motion *> *T_m;
  if (options_dbrrt.use_nigh_nn) {
    T_m = nigh_factory<Motion *>(problem.robotType, robot);
  } else if (si->getStateSpace()->isMetricSpace()) {
    T_m = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    T_m->setDistanceFunction([si, motions](const Motion *a, const Motion *b) {
      return si->distance(a->states[0], b->states[0]);
    });

  } else {
    T_m = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    T_m->setDistanceFunction([si, motions](const Motion *a, const Motion *b) {
      return si->distance(a->states[0], b->states[0]);
    });
  }

  std::vector<Motion *> motions_ptr(motions.size());
  std::transform(motions.begin(), motions.end(), motions_ptr.begin(),
                 [](auto &s) { return &s; });
  {
    Stopwatch s;
    T_m->add(motions_ptr);
    time_bench.time_nearestMotion += s.elapsed_ms();
  }

  ompl::NearestNeighbors<AStarNode *> *T_n;

  if (options_dbrrt.use_nigh_nn) {
    T_n = nigh_factory<AStarNode *>(problem.robotType, robot);
  } else {
    NOT_IMPLEMENTED;
  }

  Terminate_status status = Terminate_status::UNKNOWN;

  double radius = options_dbrrt.delta;

  auto nearest_state_timed = [&](auto &query_n, auto &neighbour) {
    auto _out = timed_fun([&] {
      neighbour = T_n->nearest(query_n);
      return 0;
    });
    time_bench.num_nn_states++;
    time_bench.time_nearestNode += _out.second;
    time_bench.time_nearestNode_search += _out.second;
  };

  auto is_motion_valid_timed = [&](auto &motion, auto &offset,
                                   bool &motionValid, Trajectory &traj) {
    if (options_dbrrt.check_cols) {
      if (options_dbrrt.use_collision_shape) {

        auto out = timed_fun([&] {
          motion->collision_manager->shift(offset);
          fcl::DefaultCollisionData<double> collision_data;
          motion->collision_manager->collide(
              robot->diff_model->env.get(), &collision_data,
              fcl::DefaultCollisionFunction<double>);
          bool motionValid = !collision_data.result.isCollision();
          motion->collision_manager->shift(-offset);
          return motionValid;
        });

        motionValid = out.first;
        time_bench.time_collisions += out.second;
        time_bench.num_col_motions++;
      } else {
        // check all the configuration, starting by the middle

        CHECK(traj.states.size(), AT);

        Stopwatch watch;

        size_t index_start = 0;
        size_t index_last = motion->states.size() - 1;

        // check the first and last state

        size_t nx = robot->diff_model->nx;
        Eigen::VectorXd x(nx);
        x = traj.states.front();

        // robot->toEigen(motion->states.front(), x);

        bool start_good = false;
        bool goal_good = false;
        if (robot->diff_model->collision_check(x)) {
          start_good = true;
        }
        // robot->toEigen(motion->states.back(), x);
        x = traj.states.back();
        if (robot->diff_model->collision_check(x)) {
          goal_good = true;
        }

        if (start_good && goal_good) {

          using Segment = std::pair<size_t, size_t>;
          std::queue<Segment> queue;

          queue.push(Segment{index_start, index_last});

          size_t index_resolution = 1;

          if (robot->diff_model->ref_dt < .05) {
            // TODO: which number to put here?
            index_resolution = 5;
          }

          // I could use a spatial resolution also...

          motionValid = true;
          while (!queue.empty()) {

            auto [si, gi] = queue.front();
            queue.pop();

            if (gi - si > index_resolution) {

              // check if they are very close -> HOW exactly?
              // auto &gix = traj.states.at(gi);
              // auto &six = traj.states.at(si);

              size_t ii = int((si + gi) / 2);

              if (ii == si || ii == gi) {
                continue;
              }
              // robot->toEigen(motion->states.at(ii), x);
              x = traj.states.at(ii);
              if (robot->diff_model->collision_check(x)) {
                if (ii != si)
                  queue.push(Segment{ii, gi});
                if (ii != gi)
                  queue.push(Segment{si, ii});
              } else {
                motionValid = false;
                break;
              }
            }
          }
        } else {
          motionValid = false;
        }

        time_bench.time_collisions += watch.elapsed_ms();
        time_bench.num_col_motions++;
      }
    } else {
      motionValid = true;
    }
  };

  auto add_state_timed = [&](auto &node) {
    auto out = timed_fun([&] {
      T_n->add(node);
      return 0;
    });
    time_bench.time_nearestNode += out.second;
    time_bench.time_nearestNode_add += out.second;
  };

  auto start_node = new AStarNode;
  start_node->gScore = 0;
  start_node->state = robot->startState;
  start_node->came_from = nullptr;

  add_state_timed(start_node);

  //

  auto start = std::chrono::steady_clock::now();
  auto get_time_stamp_ms = [&] {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };

  Eigen::VectorXd x(robot->diff_model->nx);
  Eigen::VectorXd goalState_eig(robot->diff_model->nx);

  robot->toEigen(robot->goalState, goalState_eig);

  Eigen::VectorXd __current_state(robot->diff_model->nx);
  Eigen::VectorXd __canonical_state(robot->diff_model->nx);
  Eigen::VectorXd offsete(robot->diff_model->get_offset_dim());
  Eigen::VectorXd offseteX(robot->diff_model->get_offset_dim());
  Eigen::VectorXd ___current_state(robot->diff_model->nx);

  CSTR_V(robot->diff_model->x_lb);
  CSTR_V(robot->diff_model->x_ub);

  AStarNode *current;
  AStarNode *tmp;

  Motion fakeMotion;
  fakeMotion.idx = -1;
  fakeMotion.states.push_back(si->allocState());

  std::vector<Motion *> neighbors_m;

  auto nearest_motion_timed = [&](auto &fakeMotion, auto &neighbors_m) {
    std::vector<double> real;

    bool verbose = true;

    if (options_dbrrt.debug) {
      si->getStateSpace()->copyToReals(real, fakeMotion.getState());
      // data_out_query_Tm.push_back(real);
    }

    auto out = timed_fun([&] {
      T_m->nearestR(&fakeMotion, options_dbrrt.delta, neighbors_m);

      // num_nn_motions.push_back(neighbors_m.size());
      // if (neighbors_m.size() > max_num_nn_motions) {
      //   max_num_nn_motions = neighbors_m.size();
      //   si->getStateSpace()->copyState(state_more_nn, fakeMotion.getState());
      //   nn_of_best = neighbors_m;
      // }

      if (!neighbors_m.size() && true) {

        std::cout << "no neighours for state " << std::endl;
        si->printState(fakeMotion.getState(), std::cout);

        std::cout << "close state is  " << std::endl;
        auto close_motion = T_m->nearest(&fakeMotion);
        si->printState(close_motion->getState(), std::cout);
        std::cout << std::endl;

        std::cout << "close distance is:  "
                  << si->distance(close_motion->getState(),
                                  fakeMotion.getState())
                  << std::endl;
        std::cout << "R is " << options_dbrrt.delta << std::endl;
      }

      return 0;
    });
    time_bench.time_nearestMotion += out.second;
    time_bench.num_nn_motions++;
  };

  ob::State *tmpState = si->allocState();
  ob::State *tmpState2 = si->allocState();
  AStarNode *query = new AStarNode;
  AStarNode *solution = nullptr;
  Trajectory tmp_traj;

  double best_distance_to_goal =
      si->distance(start_node->state, robot->goalState);
  AStarNode *best_node;

  std::vector<Eigen::VectorXd> query_nodes;
  std::vector<Eigen::VectorXd> current_nodes;
  std::vector<Trajectory> trajs;
  std::vector<Trajectory> chosen_trajs;

  std::random_device rd;
  std::mt19937 g(rd());
  while (true) {

    if (static_cast<size_t>(time_bench.expands) >= options_dbrrt.max_expands) {
      status = Terminate_status::MAX_EXPANDS;
      std::cout << "BREAK search:"
                << "MAX_EXPANDS" << std::endl;
      break;
    }

    if (get_time_stamp_ms() > options_dbrrt.search_timelimit) {
      status = Terminate_status::MAX_TIME;
      std::cout << "BREAK search:"
                << "MAX_TIME" << std::endl;
      break;
    }

    // sample a configuration at random

    if (((double)rand()) / RAND_MAX < options_dbrrt.goal_bias) {
      x = goalState_eig;
    } else {
      robot->diff_model->sample_uniform(x);
    }
    query_nodes.push_back(x);

    time_bench.expands++;

    if (time_bench.expands % 100 == 0) {
      std::cout << "expands: " << time_bench.expands
                << " best distance: " << best_distance_to_goal << std::endl;
    }

    robot->fromEigen(tmpState, x);
    query->state = tmpState;
    nearest_state_timed(query, current);
    Eigen::VectorXd current_eig(robot->diff_model->nx);
    robot->toEigen(current->state, current_eig);
    current_nodes.push_back(current_eig);

    std::cout << "query " << std::endl;
    si->printState(query->state, std::cout);
    std::cout << "current " << std::endl;
    si->printState(current->state, std::cout);

    // check which motions are applicable

    si->copyState(fakeMotion.states[0], current->state);

    // CHANGE THIS
    if (!options_dbrrt.new_invariance) {
      if (robot->isTranslationInvariant())
        robot->setPosition(fakeMotion.states[0], fcl::Vector3d(0, 0, 0));
    } else {
      // new
      robot->toEigen(current->state, __current_state);
      robot->diff_model->canonical_state(__current_state, __canonical_state);
      robot->fromEigen(fakeMotion.states[0], __canonical_state);
    }

    std::vector<Motion *> neighbors_m;
    nearest_motion_timed(fakeMotion, neighbors_m);

    // limit the step size?

    Eigen::VectorXd target_state = x;

    double min_distance = std::numeric_limits<double>::max();
    int best_index = -1;
    Trajectory chosen_traj;

    std::shuffle(neighbors_m.begin(), neighbors_m.end(), g);

    for (size_t i = 0; i < neighbors_m.size(); i++) {

      auto &motion = neighbors_m[i];
      fcl::Vector3d offset(0., 0., 0.);
      fcl::Vector3d computed_offset(0., 0., 0.);
      fcl::Vector3d current_pos(0., 0., 0.);

      // if (!options_dbrrt.new_invariance) {
      //   if (robot->isTranslationInvariant()) {
      //     current_pos = robot->getTransform(current->state).translation();
      //     offset = current_pos + computed_offset;
      //     const auto relative_pos =
      //     robot->getTransform(tmpState).translation();
      //     robot->setPosition(tmpState, offset + relative_pos);
      //     robot->setPosition(tmpState2, current_pos);
      //   }
      // } else {

      robot->toEigen(current->state, ___current_state);
      robot->diff_model->offset(___current_state, offsete);
      robot->diff_model->transform_primitive(offsete, motion->traj.states,
                                             motion->traj.actions,
                                             tmp_traj.states, tmp_traj.actions);

      trajs.push_back(tmp_traj);

      robot->fromEigen(tmpState, tmp_traj.states.back());
      robot->fromEigen(tmpState2, tmp_traj.states.front());

      // which is the current offset?
      // robot->diff_model->offset(tmp_traj.states.front(), offseteX);
      // offset.head(robot->diff_model->translation_invariance) =
      //     offseteX.head(robot->diff_model->translation_invariance);
      // }

      bool invalid = false;
      for (auto &state : tmp_traj.states) {
        if (!robot->diff_model->is_state_valid(state)) {
          invalid = true;
          break;
        }
      }
      if (invalid)
        continue;

      bool motionValid;
      is_motion_valid_timed(motion, offset, motionValid, tmp_traj);

      if (!motionValid)
        continue;

      double d =
          robot->diff_model->distance(tmp_traj.states.back(), target_state);

      if (d < min_distance) {
        min_distance = d;
        best_index = i;
        chosen_traj = tmp_traj;

        if (options_dbrrt.choose_first_motion_valid)
          break;
      }
    }
    // add a new state to the tree!

    if (best_index != -1) {
      chosen_trajs.push_back(chosen_traj);
      auto last_state = robot->si_->allocState();
      robot->fromEigen(last_state, chosen_traj.states.back());
      auto new_node = new AStarNode();
      new_node->state = si->cloneState(last_state);
      new_node->came_from = current;
      new_node->used_motion = neighbors_m[best_index]->idx;

      // check that the node is not alreay in the tree!
      nearest_state_timed(new_node, tmp);

      if (si->distance(tmp->state, new_node->state) <
          options_dbrrt.delta / 2.) {
        std::cout << "warning, node already in the tree" << std::endl;
        continue;
      }

      add_state_timed(new_node);

      if (si->distance(new_node->state, robot->goalState) <
          best_distance_to_goal) {
        best_distance_to_goal = si->distance(new_node->state, robot->goalState);
        best_node = new_node;
      }

      if (robot->si_->distance(new_node->state, robot->goalState) <
          options_dbrrt.goal_region) {
        status = Terminate_status::SOLVED;
        si->printState(new_node->state, std::cout);
        std::cout << "BREAK search:"
                  << "GOAL_REACHED" << std::endl;
        solution = new_node;
        break;
      }
    } else {
      std::cout << "warning, all expansions failed" << std::endl;
    }
  }

  std::ofstream debug_file("debug.yaml");

  debug_file << "query_nodes:" << std::endl;
  for (auto &q : query_nodes) {
    debug_file << "  - [";
    for (size_t i = 0; i < q.size(); i++) {
      debug_file << q[i];
      if (i < q.size() - 1)
        debug_file << ", ";
    }
    debug_file << "]" << std::endl;
  }

  debug_file << "current_nodes:" << std::endl;
  for (auto &q : current_nodes) {
    debug_file << "  - [";
    for (size_t i = 0; i < q.size(); i++) {
      debug_file << q[i];
      if (i < q.size() - 1)
        debug_file << ", ";
    }
    debug_file << "]" << std::endl;
  }

  debug_file << "trajs:" << std::endl;
  for (auto &traj : trajs) {
    debug_file << "  - " << std::endl;
    traj.to_yaml_format(debug_file, "    ");
  }

  debug_file << "chosen_trajs:" << std::endl;
  for (auto &traj : chosen_trajs) {
    debug_file << "  - " << std::endl;
    traj.to_yaml_format(debug_file, "    ");
  }

  auto startState = robot->startState;
  auto goalState = robot->goalState;

  std::ofstream out("out_dbrrt.yaml");

  auto tmpStateq = si->allocState();
  if (solution) {
    state_to_eigen(traj_out.start, si, startState);
    state_to_eigen(traj_out.goal, si, goalState);

    std::vector<const AStarNode *> result;

    const AStarNode *n = solution;
    while (n != nullptr) {
      result.push_back(n);
      // std::cout << n->used_motion << std::endl;
      // si->printState(n->state);
      n = n->came_from;
    }
    std::reverse(result.begin(), result.end());

    std::cout << "result size " << result.size() << std::endl;

    out << "result:" << std::endl;
    out << "  - states:" << std::endl;

    si->copyState(tmpStateq, startState);

    auto space6 = std::string(6, ' ');
#if 1

    auto &mm = robot->diff_model;

    Eigen::VectorXd __tmp(robot->diff_model->nx);
    Eigen::VectorXd __offset(robot->diff_model->get_offset_dim());
    for (size_t i = 0; i < result.size() - 1; ++i) {
      const auto node_state = result[i]->state;
      const auto &motion = motions.at(result[i + 1]->used_motion);
      int take_until = result[i + 1]->intermediate_state;
      if (take_until != -1) {
        out << space6 + "# (note: we have stopped at intermediate state) "
            << std::endl;
      }
      out << space6 + "# (node_state) ";
      printState(out, si, node_state); // debug
      out << std::endl;
      out << space6 + "# motion " << motion.idx << " with cost " << motion.cost
          << std::endl; // debug
      out << space6 + "# motion first state "
          << motion.traj.states.front().format(FMT) << std::endl;
      out << space6 + "# motion last state "
          << motion.traj.states.back().format(FMT) << std::endl;
      //
      //
      //
      //
      // transform the motion to match the state

      // get the motion
      robot->toEigen(node_state, __tmp);
      robot->diff_model->offset(__tmp, __offset);
      out << space6 + "# (tmp) " << __tmp.format(FMT) << std::endl;
      out << space6 + "# (offset) " << __offset.format(FMT) << std::endl;
      ;

      std::vector<Eigen::VectorXd> xs;
      std::vector<Eigen::VectorXd> us;
      auto &traj = motion.traj;
      robot->diff_model->transform_primitive(__offset, traj.states,
                                             traj.actions, xs, us);
      // TODO: missing additional offset, if any

      out << space6 + "# (traj.states.front) "
          << traj.states.front().format(FMT) << std::endl;
      out << space6 + "# (xs.front) " << xs.front().format(FMT) << std::endl;

      size_t take_num_states = xs.size();
      if (take_until != -1)
        take_num_states = take_until + 1;

      for (size_t k = 0; k < take_num_states; ++k) {
        if (k < take_num_states - 1) {
          // print the state
          out << space6 << "- ";
          traj_out.states.push_back(xs.at(k));
        } else if (i == result.size() - 2) {
          out << space6 << "- ";
          traj_out.states.push_back(xs.at(k));
        } else {
          out << space6 << "# (last state) ";
        }
        out << xs.at(k).format(FMT) << std::endl;
      }

      // Continue here!!
      // Just get state + motion
      // skip last, then state... and so on!!!
    }
    out << space6 << "# goal state is " << goalState_eig.format(FMT)
        << std::endl;
#endif
    out << "    actions:" << std::endl;

    int action_counter = 0;
    for (size_t i = 0; i < result.size() - 1; ++i) {
      const auto &motion = motions.at(result.at(i + 1)->used_motion);
      int take_until = result.at(i + 1)->intermediate_state;
      if (take_until != -1) {
        out << space6 + "# (note: we have stop at intermediate state) "
            << std::endl;
      }

      out << space6 + "# motion " << motion.idx << " with cost " << motion.cost
          << std::endl; // debug
      //
      //
      //

      size_t take_num_actions = motion.actions.size();

      if (take_until != -1) {
        take_num_actions = take_until;
      }
      CHECK_LEQ(take_num_actions, motion.actions.size(), AT);
      out << space6 + "# "
          << "take_num_actions " << take_num_actions << std::endl;

      for (size_t k = 0; k < take_num_actions; ++k) {
        const auto &action = motion.actions[k];
        out << space6 + "- ";
        action_counter += 1;
        printAction(out, si, action);
        Eigen::VectorXd x;
        control_to_eigen(x, si, action);
        traj_out.actions.push_back(x);
        out << std::endl;
      }
      out << std::endl;
    }
  }

  out_info_db.solved = solution;
  if (out_info_db.solved)
    out_info_db.cost = traj_out.actions.size() * robot->diff_model->ref_dt;
}
