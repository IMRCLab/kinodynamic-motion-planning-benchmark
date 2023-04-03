#include "ompl_geo.hpp"
#include "ocp.hpp"

namespace ob = ompl::base;
namespace og = ompl::geometric;

void solve_ompl_geometric(const Problem &problem,
                          const Options_geo &options_geo,
                          const Options_trajopt &options_trajopt,
                          Trajectory &traj_out, Info_out &info_out_omplgeo) {

  std::shared_ptr<RobotOmpl> robot = robot_factory_ompl(problem);

  // create and set a start state
  auto startState = robot->startState;
  auto goalState = robot->goalState;
  auto si = robot->getSpaceInformation();

  // create a problem instance
  auto pdef(std::make_shared<ob::ProblemDefinition>(si));

  pdef->addStartState(startState);
  pdef->setGoalState(goalState, options_geo.goalregion);
  // si->freeState(startState); @Wolfgang WHY??
  // si->freeState(goalState); @Wolfgang WHY??

  // create a planner for the defined space
  std::shared_ptr<ob::Planner> planner;
  if (options_geo.planner == "rrt*") {
    planner.reset(new og::RRTstar(si));
  } else if (options_geo.planner == "sst") {
    planner.reset(new og::SST(si));
  }
  // rrt->setGoalBias(params["goalBias"].as<float>());
  // auto planner(rrt);

  auto start = std::chrono::steady_clock::now();
  bool has_solution = false;
  std::chrono::steady_clock::time_point previous_solution;

  // std::vector<Trajectory> trajectories;

  bool traj_opti = true;

  auto get_time_stamp_ms = [&] {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };

  pdef->setIntermediateSolutionCallback(
      [&](const ob::Planner *, const std::vector<const ob::State *> &states,
          const ob::Cost cost) {
        double t = get_time_stamp_ms();

        Trajectory traj;
        Trajectory traj_geo;
        traj_geo.time_stamp = t;
        state_to_eigen(traj_geo.start, si, startState);
        state_to_eigen(traj_geo.goal, si, goalState);
        traj_geo.states.push_back(traj_geo.start);

        auto __states = states;
        std::reverse(__states.begin(), __states.end());

        for (auto &s : __states) {
          Eigen::VectorXd x;
          state_to_eigen(x, si, s);
          traj_geo.states.push_back(x);
        }
        traj_geo.states.push_back(traj_geo.goal);
        traj_geo.feasible = false;
        double time = 0;

        traj_geo.times.resize(traj_geo.states.size());
        traj_geo.times(0) = 0.;

        // compute the times.
        for (size_t i = 0; i < traj_geo.states.size() - 1; i++) {
          auto &x = traj_geo.states.at(i);
          auto &y = traj_geo.states.at(i + 1);
          time += robot->diff_model->lower_bound_time(x, y);
          traj_geo.times(i + 1) = time;
        }

        traj_geo.cost = traj_geo.times.tail(1)(0);

        // add the control

        for (size_t i = 0; i < traj_geo.states.size() - 1; i++) {
          traj_geo.actions.push_back(robot->diff_model->u_ref);
        }

        // Path path;
        // path.time_stamp = t;

        // double t =
        // std::chrono::duration_cast<std::chrono::milliseconds>(now
        // - start).count(); double dt =
        // std::chrono::duration_cast<std::chrono::milliseconds>(now -
        // previous_solution).count();
        std::cout << "Intermediate geometric solution! " << cost.value()
                  << std::endl;
        has_solution = true;
        // last_solution_in_sec = dt / 1000.0f;
        std::cout << "printing a new path" << std::endl;
        for (auto &state : states)
          si->printState(state, std::cout);
        std::cout << "printing a new path -- DONE" << std::endl;

        info_out_omplgeo.trajs_raw.push_back(traj_geo);

        std::cout << "traj geo" << std::endl;
        traj_geo.to_yaml_format(std::cout);

        if (traj_opti) {

          // Options_trajopt opti;
          // opti.solver_id = 0;
          // opti.control_bounds = 1;
          // opti.use_warmstart = 1;
          // opti.weight_goal = 100;
          // opti.max_iter = 100;
          // opti.noise_level = 1e-4;

          Result_opti result;

          // convert path to eigen
          // file_inout.xs = traj.states;
          // file_inout.us = traj.actions;
          // file_inout.ts = traj.times;

          // compute approximate time.

          std::cout << "*** Sart optimization ***" << std::endl;
          trajectory_optimization(problem, traj_geo, options_trajopt, traj,
                                  result);

          std::cout << "*** Optimization done ***" << std::endl;

          std::cout << "traj opt" << std::endl;
          traj.to_yaml_format(std::cout);

          if (traj.feasible) {
            info_out_omplgeo.solved = true;

            if (traj.cost < info_out_omplgeo.cost) {
              info_out_omplgeo.cost = traj.cost;
              traj_out = traj;
            }
          }

          traj.time_stamp = get_time_stamp_ms();
          info_out_omplgeo.trajs_opt.push_back(traj);
        }

        previous_solution = std::chrono::steady_clock::now();
      });

  // set the problem we are trying to solve for the planner
  planner->setProblemDefinition(pdef);

  // perform setup steps f  r the planner
  // (this will set the optimization objective)
  planner->setup();

  // set a really high cost threshold, so that planner stops after first
  // solution was found
  // pdef->getOptimizationObjective()->setCostThreshold(ob::Cost(1e6));

  // print the settings for this space
  si->printSettings(std::cout);

  // print the problem settings
  pdef->print(std::cout);

  // attempt to solve the problem within timelimit
  ob::PlannerStatus solved;

  // solved = planner->ob::Planner::solve(timelimit);
  // terminate if no better solution is found within the timelimit
  // WHY?

  solved = planner->solve(
      ob::PlannerTerminationCondition([&previous_solution, &has_solution] {
        if (!has_solution) {
          return false;
        }
        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - previous_solution)
                        .count() /
                    1000.0f;
        return dt > 1.0;
      }));
  std::cout << solved << std::endl;

  // lets print all the paths

  info_out_omplgeo.solved = solved;

  if (solved) {
    std::cout << "Found solution:" << std::endl;
  } else {
    std::cout << "No solution found" << std::endl;
  }
}
