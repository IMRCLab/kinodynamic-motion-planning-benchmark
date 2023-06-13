
#include "idbastar.hpp"

// int main(int argc, char *argv[]) {
//     srand((unsigned)time(NULL) * getpid());
//     std::cout << gen_random(12) << "\n";
//     return 0;
// }

// TODO
// give options to load primitives and heuristic map only once.
// cli to create and store a heuristic map for a robot in an environment.

void idbA(const Problem &problem, const Options_idbAStar &options_idbas,
          const Options_dbastar &options_dbastar,
          const Options_trajopt &options_trajopt, Trajectory &traj_out,
          Info_out_idbastar &info_out_idbastar) {

  bool finished = false;

  Options_dbastar options_dbastar_local = options_dbastar;

  size_t it = 0;

  std::vector<Motion> motions;

  auto robot = robot_factory_ompl(problem);
  std::cout << "Loading motion primitives " << std::endl;
  if (!options_dbastar_local.primitives_new_format) {
    CHECK(robot, AT);
    load_motion_primitives(options_dbastar_local.motionsFile, *robot, motions,
                           options_idbas.max_motions_primitives,
                           options_dbastar_local.cut_actions, true);
  } else {
    CHECK(robot, AT);
    load_motion_primitives_new(options_dbastar_local.motionsFile, *robot,
                               motions, options_idbas.max_motions_primitives,
                               options_dbastar_local.cut_actions, true,
                               options_dbastar_local.check_cols);
  }
  options_dbastar_local.motions_ptr = &motions;
  std::cout << "Loading motion primitives -- DONE " << std::endl;

  std::vector<Heuristic_node> heu_map;
  if (options_dbastar.heuristic == 1) {

    if (options_dbastar.heu_map_file.size()) {

      load_heu_map(options_dbastar_local.heu_map_file.c_str(), heu_map);

    } else {
      std::cout << "not heu map provided. Computing one .... " << std::endl;
      // there is not
      generate_heuristic_map(problem, robot, options_dbastar_local, heu_map);
      std::cout << "writing heu map " << std::endl;
      write_heu_map(heu_map, "tmp_heu_map.yaml");
    }
    options_dbastar_local.heu_map_ptr = &heu_map;
  }

  auto start = std::chrono::steady_clock::now();

  auto get_time_stamp_ms = [&] {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };

  // TODO: load heurisitic Map if necessary

  info_out_idbastar.cost = 1e8;

  size_t num_solutions = 0;

  double non_counter_time = 0;

  while (!finished) {

    // TODO: implement also the original schedule of

    if (options_idbas.new_schedule) {
      if (it == 0) {
        options_dbastar_local.delta = options_idbas.delta_0;
        options_dbastar_local.max_motions = options_idbas.num_primitives_0;
      } else {
        options_dbastar_local.delta *= options_idbas.delta_rate;
        options_dbastar_local.max_motions *= options_idbas.num_primitives_rate;
      }
    } else {
      NOT_IMPLEMENTED;
    }

    // options_dbastar_local.maxCost = info_out_idbastar.cost;
    double delta_cost = 1.2;
    CSTR_(delta_cost);
    options_dbastar_local.maxCost = info_out_idbastar.cost * delta_cost;

    std::cout << "*** Running DB-astar ***" << std::endl;

    Trajectory traj_db, traj;
    Out_info_db out_info_db;
    double __pre_t = get_time_stamp_ms();
    dbastar(problem, options_dbastar_local, traj_db, out_info_db);
    std::cout << "warning: using as time only the search!" << std::endl;
    double __after_t = get_time_stamp_ms();
    non_counter_time += __after_t - __pre_t - out_info_db.time_search;
    CSTR_(non_counter_time);
    traj_db.time_stamp = get_time_stamp_ms() - non_counter_time;
    info_out_idbastar.trajs_raw.push_back(traj_db);
    info_out_idbastar.infos_raw.push_back(out_info_db.data);

    if (out_info_db.solved) {
      {
        std::string filename = "trajdb_" + gen_random(6) + ".yaml";
        std::cout << "saving traj db file: " << filename << std::endl;
        std::ofstream traj_db_out(filename);
        traj_db.to_yaml_format(traj_db_out);
      }

      Result_opti result;
      std::cout << "***Trajectory Optimization -- START ***" << std::endl;
      trajectory_optimization(problem, traj_db, options_trajopt, traj, result);
      std::cout << "***Trajectory Optimization -- DONE ***" << std::endl;

      traj.time_stamp = get_time_stamp_ms() - non_counter_time;
      info_out_idbastar.trajs_opt.push_back(traj);
      info_out_idbastar.infos_opt.push_back(result.data);

      if (traj.feasible) {
        num_solutions++;
        info_out_idbastar.solved = true;

        std::cout << "we have a feasible solution! Cost: " << traj.cost
                  << std::endl;

        if (traj.cost < info_out_idbastar.cost) {
          info_out_idbastar.cost = traj.cost;
          traj_out = traj;
        }

        // add primitives
        if (options_idbas.add_primitives_opt) {

          // lets generate primitives
          size_t number_of_cuts = 5;

          Trajectories new_trajectories =
              cut_trajectory(traj, number_of_cuts, robot->diff_model);

          auto &rr = robot->diff_model;

          Trajectories trajs_canonical;
          for (const auto &traj : new_trajectories.data) {

            Eigen::VectorXd x0(traj.states.front().size());
            rr->canonical_state(traj.states.front(), x0);
            std::vector<Eigen::VectorXd> xx = traj.states;
            rr->rollout(x0, traj.actions, xx);

            Trajectory traj_out;
            traj_out.actions = traj.actions;
            traj_out.states = xx;
            traj_out.goal = traj_out.states.back();
            traj_out.start = traj_out.states.front();
            traj_out.cost = traj.cost;

            {
              traj_out.check(rr, true);
              traj_out.update_feasibility();
              // CHECK(traj_out.feasible, AT); // NOTE: some can go out of
              // bounds in the canonical form.
            }

            trajs_canonical.data.push_back(traj_out);
          }

          {
            std::string filename =
                "trajs_cuts_canonical_" + gen_random(6) + ".yaml";
            std::cout << "saving traj file: " << filename << std::endl;
            trajs_canonical.save_file_yaml(filename.c_str());
          }

          std::vector<Motion> motions_out;
          for (const auto &traj : trajs_canonical.data) {
            Motion motion_out;
            CHECK(robot, AT)
            traj_to_motion(traj, *robot, motion_out, true);
            motions_out.push_back(motion_out);
          }

          const bool debug_primitves_extraction = false;
          CSTR_(debug_primitves_extraction);

          if (debug_primitves_extraction) {
            CSTR_(motions_out.size());
            motions.insert(motions.begin(), motions_out.begin(),
                           motions_out.end());

            std::cout << "Afer insert " << motions.size() << std::endl;
            std::cout << "Warning: "
                      << "I am inserting at the beginning" << std::endl;

          } else {
            motions = motions_out;
          }
        }
      }
    }

    it++;

    double time_stamp_ms = static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    if (it >= options_idbas.max_it) {
      finished = true;
      info_out_idbastar.exit_criteria = EXIT_CRITERIA::max_it;
    }

    if (num_solutions >= options_idbas.max_num_sol) {
      finished = true;
      info_out_idbastar.exit_criteria = EXIT_CRITERIA::max_solution;
    }

    if (time_stamp_ms / 1000. > options_idbas.timelimit) {
      finished = true;
      info_out_idbastar.exit_criteria = EXIT_CRITERIA::time_limit;
    }
  }

  std::cout << "exit criteria"
            << static_cast<int>(info_out_idbastar.exit_criteria) << std::endl;
}

void write_results_idbastar(const char *results_file,
                            const Options_idbAStar &options_idbastar,
                            const Options_dbastar &options_dbastar,
                            const Options_trajopt &options_trajopt,
                            const Info_out_idbastar &info_out_idbastar) {

  std::ofstream results(results_file);

  results << "alg: idbastar" << std::endl;
  results << "time_stamp: " << get_time_stamp() << std::endl;

  results << "options_idbastar:" << std::endl;
  options_idbastar.print(results, "  ");

  results << "options_dbastar:" << std::endl;
  options_dbastar.print(results, "  ");

  results << "options trajopt:" << std::endl;
  options_trajopt.print(results, "  ");

  info_out_idbastar.to_yaml(results);
}
