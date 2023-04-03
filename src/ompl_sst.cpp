

#include "ompl_sst.hpp"
#include "ocp.hpp"

namespace ob = ompl::base;
namespace oc = ompl::control;

struct SST_public_interface : public oc::SST {

  SST_public_interface(const oc::SpaceInformationPtr &si) : oc::SST(si) {}

  ~SST_public_interface() = default;

  std::vector<ompl::base::State *> get_prevSolution_() const {
    return prevSolution_;
  };

  std::vector<oc::Control *> get_prevSolutionControls_() const {
    return prevSolutionControls_;
  };

  std::vector<unsigned> get_prevSolutionSteps_() const {
    return prevSolutionSteps_;
  };

  ompl::base::Cost get_prevSolutionCost_() const { return prevSolutionCost_; }
};

void solve_sst(const Problem &problem, const Options_sst &options_ompl_sst,
              const  Options_trajopt &options_trajopt, Trajectory &traj_out,
               Info_out &info_out_omplsst) {

  auto robot = robot_factory_ompl(problem);

  auto si = robot->getSpaceInformation();

  si->setPropagationStepSize(options_ompl_sst.propagation_step_size);
  si->setMinMaxControlDuration(options_ompl_sst.min_control_duration,
                               options_ompl_sst.max_control_duration);
  si->setup(); // does it work?

  std::cout << "after second setup" << std::endl;
  si->printSettings(std::cout);

  //
  //

  // set number of control steps (use 0.1s as increment -> 0.1 to 1s per Steer
  // function)

  // set state validity checking for this space
  // auto stateValidityChecker(std::make_shared<fclStateValidityChecker>(si,
  // bpcm_env, robot)); si->setStateValidityChecker(stateValidityChecker);
  //
  // // set the state propagator
  // std::shared_ptr<oc::StatePropagator> statePropagator(new
  // RobotOmplStatePropagator(si, robot));
  // si->setStatePropagator(statePropagator);
  //
  // si->setup();

  // create a problem instance
  auto pdef(std::make_shared<ob::ProblemDefinition>(si));
  auto startState = robot->startState;
  auto goalState = robot->goalState;

  // create and set a start state
  pdef->addStartState(startState);
  // si->freeState(startState);

  // set goal state
  pdef->setGoalState(goalState, options_ompl_sst.goal_epsilon);

  // si->freeState(goalState);

  // create a planner for the defined space
  std::shared_ptr<ob::Planner> planner;
  if (options_ompl_sst.planner == "rrt") {
    auto rrt = new oc::RRT(si);
    rrt->setGoalBias(options_ompl_sst.goal_bias);
    planner.reset(rrt);
  } else if (options_ompl_sst.planner == "sst") {
    // auto sst = new oc::SST(si);
    auto sst = new SST_public_interface(si);
    sst->setGoalBias(options_ompl_sst.goal_bias);

    sst->setSelectionRadius(options_ompl_sst.selection_radius);
    sst->setPruningRadius(options_ompl_sst.pruning_radius);
    planner.reset(sst);
  }
  // rrt->setGoalBias(params["goalBias"].as<float>());
  // auto planner(rrt);

  pdef->setOptimizationObjective(
      std::make_shared<ob::ControlDurationObjective>(si));

  // empty stats file
  // std::ofstream stats(inout.statsFile);
  // stats << "stats:" << std::endl;

  auto start = std::chrono::steady_clock::now();

  double best_sol = -1;

  // TODO
  // how to get the cost of the solution?
  // Either I am stupid or the interface in OMPL is not good -- probably both.
  // @Wolfang Help

  Eigen::VectorXd start_eigen;
  Eigen::VectorXd goal_eigen;
  state_to_eigen(start_eigen, si, startState);
  state_to_eigen(goal_eigen, si, goalState);

  auto get_time_stamp_ms = [&] {
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
  };

  pdef->setIntermediateSolutionCallback(
      [&](const ob::Planner *pp, const std::vector<const ob::State *> &states,
          const ob::Cost cost) {
        // i cast hte planeer

        (void)states;
        std::cout << "hello world " << std::endl;
        double tt = get_time_stamp_ms();
        {

          Trajectory traj_sst;
          traj_sst.start = start_eigen;
          traj_sst.goal = goal_eigen;
          traj_sst.time_stamp = tt;

          std::cout << "hello world 22 " << std::endl;
          const SST_public_interface *planner =
              dynamic_cast<const SST_public_interface *>(pp);
          auto states = planner->get_prevSolution_();
          auto steps = planner->get_prevSolutionSteps_();
          auto controls = planner->get_prevSolutionControls_();

          auto path(std::make_shared<ompl::control::PathControl>(si));
          for (int i = states.size() - 1; i >= 1; --i)
            path->append(states[i], controls[i - 1],
                         steps[i - 1] * si->getPropagationStepSize());
          path->append(states[0]);

          path->interpolate(); // normalize to a single control step
                               //
                               //
                               //
                               //
          Eigen::VectorXd x;
          for (size_t i = 0; i < path->getStateCount(); ++i) {
            const auto state = path->getState(i);
            state_to_eigen(x, si, state);
            traj_sst.states.push_back(x);
          }

          Eigen::VectorXd u;
          for (size_t i = 0; i < path->getControlCount(); ++i) {
            const auto action = path->getControl(i);
            control_to_eigen(u, si, action);
            traj_sst.actions.push_back(u);
          }
          traj_sst.cost = robot->diff_model->ref_dt * traj_sst.actions.size();

          traj_sst.check(robot->diff_model);
          info_out_omplsst.trajs_raw.push_back(traj_sst);

#if 0
          auto cc = planner->get_prevSolutionCost_();
          CSTR_(cc.value());
          auto __states = states;
          std::reverse(__states.begin(), __states.end());
          CHECK_EQ(states.size(), controls.size() + 1, AT);
          CHECK_EQ(steps.size(), controls.size(), AT);

          for (auto &s : __states) {
            Eigen::VectorXd x;
            state_to_eigen(x, si, s);
            traj.states.push_back(x);
            CSTR_V(x);
          }

          std::cout << "printing steps" << std::endl;
          for (auto &s : steps) {
            std::cout << s << " ";
          }
          std::cout << std::endl;

          std::cout << "printing controls" << std::endl;

          CSTR_(controls.size());

          auto __controls = controls;
          std::reverse(__controls.begin(), __controls.end());

          auto __steps = steps;
          std::reverse(__steps.begin(), __steps.end());

          for (auto &c : __controls) {
            std::vector<double> cc(robot->diff_model->nu);
            copyToRealsControl(si, c, cc);
            print_vec(cc.data(), cc.size());
            Eigen::VectorXd out;
            control_to_eigen(out, si, c);
            traj.actions.push_back(out);
          }

          traj.times.resize(controls.size());
          double dt = robot->diff_model->ref_dt;
          for (size_t i = 0; i < __steps.size(); i++) {
            traj.times(i) = dt * __steps.at(i);
          }
#endif

          if (options_ompl_sst.reach_goal_with_opt) {
            Result_opti result;
            Trajectory traj_opt;
            std::cout << "*** Sart optimization ***" << std::endl;
            trajectory_optimization(problem, traj_sst, options_trajopt,
                                    traj_opt, result);

            std::cout << "*** Optimization done ***" << std::endl;
            traj_opt.time_stamp = get_time_stamp_ms();
            info_out_omplsst.trajs_opt.push_back(traj_opt);

            if (traj_opt.feasible) {
              std::cout << "optimization is feasible " << std::endl;
              info_out_omplsst.solved = true;
              if (traj_opt.cost < info_out_omplsst.cost) {
                std::cout << "updating best cost to " << traj_opt.cost
                          << std::endl;

                info_out_omplsst.cost = traj_opt.cost;
                traj_out = traj_opt;
              }
            }
          } else {
            info_out_omplsst.solved = true;
            std::cout << "Warning: we are not doing OPT -- only reaching the "
                         "goal approximately "
                      << std::endl;
            CHECK_LEQ(traj_sst.cost, info_out_omplsst.cost, AT);
            info_out_omplsst.cost = traj_sst.cost;
            traj_out = traj_sst;
          }
        }

        // stats << "  - t: " << t / 1000.0f << std::endl;
        // stats << "    cost: " << cost.value() << std::endl;
        std::cout << "Intermediate solution! Cost:" << cost.value()
                  << " at time: " << tt / 1000.0f << std::endl;
        best_sol = cost.value();
      });

  // set the problem we are trying to solve for the planner
  planner->setProblemDefinition(pdef);

  // perform setup steps for the planner
  planner->setup();

  // print the settings for this space
  si->printSettings(std::cout);

  // print the problem settings
  pdef->print(std::cout);

  // attempt to solve the problem within timelimit
  ob::PlannerStatus solved;

  // for (int i = 0; i < 3; ++i) {
  solved = planner->ob::Planner::solve(options_ompl_sst.timelimit);
  CSTR_(solved);

  // This does not work because it is protected member
  // oc::SST *ppp = static_cast<oc::SST *>(planner.get());
  // ppp->prevSolution_ ;

  // this does not work for kinodynamic planning XD
  // std::cout
  //     << "cost of solution is "
  //     <<
  //     pdef->getSolutionPath()->cost(pdef->getOptimizationObjective()).value()
  //     << std::endl;

  if (info_out_omplsst.solved) {
    std::cout << "Found solution:" << std::endl;
  } else {
    std::cout << "No solution found" << std::endl;
  }
}
