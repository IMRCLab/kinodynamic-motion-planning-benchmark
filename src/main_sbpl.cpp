#include <fstream>
#include <iostream>
#include <algorithm>

#include <yaml-cpp/yaml.h>

#include <boost/program_options.hpp>

#include <fcl/fcl.h>

// using namespace std;

#include <sbpl/headers.h>

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string primitivesFile;
  std::string statsFile;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
    ("primitives,p", po::value<std::string>(&primitivesFile)->required(), "primitive file (prim)")
    ("stats", po::value<std::string>(&statsFile)->default_value("stats.yaml"), "output file (yaml)")
    ("output,o", po::value<std::string>(&outputFile)->required(), "output file (yaml)");

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error& e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  // load problem description
  YAML::Node env = YAML::LoadFile(inputFile);

  // load obstacles
  std::vector<fcl::CollisionObjectf *> obstacles;
  for (const auto &obs : env["environment"]["obstacles"])
  {
    if (obs["type"].as<std::string>() == "box")
    {
      const auto &size = obs["size"];
      std::shared_ptr<fcl::CollisionGeometryf> geom;
      geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0));
      const auto &center = obs["center"];
      auto co = new fcl::CollisionObjectf(geom);
      co->setTranslation(fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
      co->computeAABB();
      obstacles.push_back(co);
    }
    else
    {
      throw std::runtime_error("Unknown obstacle type!");
    }
  }
  fcl::BroadPhaseCollisionManagerf *bpcm_env = new fcl::DynamicAABBTreeCollisionManagerf();
  bpcm_env->registerObjects(obstacles);
  bpcm_env->setup();

  // logic based on planxythetamlevlat in the offical SBPL test application

  //set the perimeter of the robot (it is given with 0,0,0 robot ref. point for which planning is done)
  //this is for the default level - base level
  std::vector<sbpl_2Dpt_t> perimeterptsV;
  sbpl_2Dpt_t pt_m;
  double halfwidth = 0.5 / 2;
  double halflength = 0.25 / 2;
  pt_m.x = -halflength;
  pt_m.y = -halfwidth;
  perimeterptsV.push_back(pt_m);
  pt_m.x = halflength;
  pt_m.y = -halfwidth;
  perimeterptsV.push_back(pt_m);
  pt_m.x = halflength;
  pt_m.y = halfwidth;
  perimeterptsV.push_back(pt_m);
  pt_m.x = -halflength;
  pt_m.y = halfwidth;
  perimeterptsV.push_back(pt_m);

  //Initialize Environment (should be called before initializing anything else)
  EnvironmentNAVXYTHETAMLEVLAT environment_navxythetalat;

  // NOTE: This assumes that "min" is zero!
  const auto &dims = env["environment"]["max"];
  double cellsize_m = 0.025;// needs to match mprim file
  int width = dims[0].as<double>() / cellsize_m;
  int height = dims[1].as<double>() / cellsize_m;
  std::vector<unsigned char> mapdata(width*height, 0);

  // Use FCL to initialize the discrete map (robot = 1 grid cell)
  std::shared_ptr<fcl::CollisionObjectf> co_robot;
  std::shared_ptr<fcl::CollisionGeometryf> geom;
  geom.reset(new fcl::Boxf(cellsize_m, cellsize_m, 1.0));
  co_robot.reset(new fcl::CollisionObjectf(geom));

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      float center_x = x * cellsize_m + cellsize_m / 2;
      float center_y = y * cellsize_m + cellsize_m / 2;
      co_robot->setTranslation(fcl::Vector3f(center_x, center_y, 0));
      co_robot->computeAABB();
      fcl::DefaultCollisionData<float> collision_data;
      bpcm_env->collide(co_robot.get(), &collision_data, fcl::DefaultCollisionFunction<float>);
      if (collision_data.result.isCollision()) {
        mapdata[x + y * width] = 255;
      }
    }
  }

  const auto &robot_node = env["robots"][0];
  double startx = robot_node["start"][0].as<double>();
  double starty = robot_node["start"][1].as<double>();
  double starttheta = robot_node["start"][2].as<double>();
  double goalx = robot_node["goal"][0].as<double>();
  double goaly = robot_node["goal"][1].as<double>();
  double goaltheta = robot_node["goal"][2].as<double>();

  bool r = environment_navxythetalat.InitializeEnv(
      width,
      height,
      mapdata.data(),
      startx, starty, starttheta,
      goalx, goaly, goaltheta,
      /*goaltol_x*/ 0, /*goaltol_y*/ 0, /*goaltol_theta*/ 0,
      perimeterptsV,
      cellsize_m,
      /*nominalvel_mpersecs*/ sqrt(0.5),
      /*timetoturn45degsinplace_secs*/ sqrt(0.5 * M_PI_2),
      /*obsthresh*/ 100,
      primitivesFile.c_str());

  if (!r)
  {
    throw SBPL_Exception("ERROR: InitializeEnv failed");
  }

  ARAPlanner *planner = new ARAPlanner(&environment_navxythetalat, /*bforwardsearch*/ true);

  // set planner properties
  MDPConfig MDPCfg;
  // Initialize MDP Info
  if (!environment_navxythetalat.InitializeMDPCfg(&MDPCfg))
  {
    throw SBPL_Exception("ERROR: InitializeMDPCfg failed");
  }

  if (planner->set_start(MDPCfg.startstateid) == 0)
  {
    printf("ERROR: failed to set start state\n");
    throw SBPL_Exception("ERROR: failed to set start state");
  }
  if (planner->set_goal(MDPCfg.goalstateid) == 0)
  {
    printf("ERROR: failed to set goal state\n");
    throw SBPL_Exception("ERROR: failed to set goal state");
  }

  // plan
  std::vector<int> solution_stateIDs_V;
  int cost;
  // ReplanParams params(allocated_time_secs);
  // int bRet = planner->replan(&solution_stateIDs_V, params, &cost);
  double initialEpsilon = 2.0;
  bool bsearchuntilfirstsolution = false;
  double allocated_time_secs = 10.0; // in seconds
  planner->set_initialsolution_eps(initialEpsilon);
  planner->set_search_mode(bsearchuntilfirstsolution);
  int bRet = planner->replan(allocated_time_secs, &solution_stateIDs_V, &cost);
  printf("done planning with cost %d (suboptimality is: %f, desired: %f)\n", 
    cost,
    planner->compute_suboptimality(),
    planner->get_solution_eps());
  printf("size of solution (states)=%d\n", (unsigned int)solution_stateIDs_V.size());
// }

  // write the continuous solution to file
  std::vector<sbpl_xy_theta_pt_t> xythetaPath;
  environment_navxythetalat.ConvertStateIDPathintoXYThetaPath(&solution_stateIDs_V, &xythetaPath);
  printf("size of solution (cont. states)=%d\n", (unsigned int)xythetaPath.size());

  // this function is buggy and misses the goal state
  int x_d, y_d, theta_d;
  int sourceID = solution_stateIDs_V.back();
  environment_navxythetalat.GetCoordFromState(sourceID, x_d, y_d, theta_d);
  double x_c, y_c, theta_c;
  environment_navxythetalat.PoseDiscToCont(x_d, y_d, theta_d, x_c, y_c, theta_c);
  sbpl_xy_theta_pt_t pt(x_c, y_c, theta_c);
  xythetaPath.push_back(pt);

  std::vector<EnvNAVXYTHETALATAction_t> actions;
  environment_navxythetalat.GetActionsFromStateIDPath(&solution_stateIDs_V, &actions);

  std::ofstream out(outputFile);
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (const auto &state : xythetaPath)
  {
    out << "      - [" << state.x << "," << state.y << "," << std::remainder(state.theta, 2 * M_PI) << "]" << std::endl;
  }
  out << "    actions_mprim:" << std::endl;
  for (const auto &action : actions)
  {
    out << "      - [" << (int)action.starttheta << "," << (int)action.dX << "," << (int)action.dY << "," << (int)action.endtheta << "]" << std::endl;
  }

  // create stats file
  std::ofstream stats(statsFile);
  stats << "stats:" << std::endl;

  std::vector<PlannerStats> planner_stats;
  planner->get_search_stats(&planner_stats);
  double total_time = 0;
  for (const auto& s : planner_stats) {
    total_time += s.time;
    stats << "  - t: " << total_time << std::endl;
    stats << "    cost: " << s.cost / 10.0f << std::endl;
    stats << "    eps: " << s.eps << std::endl;
    stats << "    expands: " << s.expands << std::endl;
  }

  return 0;
}
