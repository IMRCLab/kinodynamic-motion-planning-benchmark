
#include "idbastar.hpp"

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

  std::cout << "Loading motion primitives " << std::endl;
  if (!options_dbastar_local.primitives_new_format) {
    load_motion_primitives(options_dbastar_local.motionsFile,
                           *robot_factory_ompl(problem), motions,
                           options_idbas.max_motions_primitives,
                           options_dbastar_local.cut_actions, true);
  } else {
    load_motion_primitives_new(options_dbastar_local.motionsFile,
                               *robot_factory_ompl(problem), motions,
                               options_idbas.max_motions_primitives,
                               options_dbastar_local.cut_actions, true);
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
      generate_heuristic_map(problem, robot_factory_ompl(problem),
                             options_dbastar_local, heu_map);
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

  while (!finished) {

    // TODO: implemente the original schedule of

    if (options_idbas.new_schedule) {
      if (it == 0) {
        options_dbastar_local.delta = options_idbas.delta_0;
        options_dbastar_local.max_motions = options_idbas.num_primitives_0;
      } else {
        options_dbastar_local.delta *= options_idbas.delta_rate;
        options_dbastar_local.max_motions *= options_idbas.num_primitives_rate;
      }
    } else {
      ERROR_WITH_INFO("not implemented");
    }

    options_dbastar_local.maxCost = info_out_idbastar.cost;

    std::cout << "*** Running DB-astar ***" << std::endl;

    Trajectory traj_db, traj;
    Out_info_db out_info_db;
    dbastar(problem, options_dbastar_local, traj_db, out_info_db);
    traj_db.time_stamp = get_time_stamp_ms();
    info_out_idbastar.trajs_raw.push_back(traj_db);

    if (out_info_db.solved) {

      Result_opti result;
      std::cout << "***Trajectory Optimization -- START ***" << std::endl;
      trajectory_optimization(problem, traj_db, options_trajopt, traj, result);
      std::cout << "***Trajectory Optimization -- DONE ***" << std::endl;

      traj.time_stamp = get_time_stamp_ms();
      info_out_idbastar.trajs_opt.push_back(traj);

      if (traj.feasible) {
        num_solutions++;
        info_out_idbastar.solved = true;

        std::cout << "we have a feasible solution! Cost: " << traj.cost
                  << std::endl;
        std::cout << "best cost " << info_out_idbastar.cost << std::endl;

        if (traj.cost < info_out_idbastar.cost) {
          info_out_idbastar.cost = traj.cost;
          traj_out = traj;
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
