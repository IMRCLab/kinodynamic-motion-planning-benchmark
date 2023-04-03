#include "ocp.hpp"
#include "ompl_geo.hpp"

int main(int argc, char *argv[]) {

  po::options_description desc("Allowed options");
  Options_geo options_geo;
  Options_trajopt options_trajopt;
  options_geo.add_options(desc);
  options_trajopt.add_options(desc);

  std::string env_file;
  std::string results_file = "";
  std::string cfg_file;
  set_from_boostop(desc, VAR_WITH_NAME(env_file));
  set_from_boostop(desc, VAR_WITH_NAME(results_file));
  set_from_boostop(desc, VAR_WITH_NAME(cfg_file));

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 0;
    }
  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  if (cfg_file != "") {
    // TODO: allow for hierarchial yaml
    options_geo.read_from_yaml(cfg_file.c_str());
    options_trajopt.read_from_yaml(cfg_file.c_str());
  }

  std::cout << "*** options_geo ***" << std::endl;
  options_geo.print(std::cout);
  std::cout << "***" << std::endl;

  std::cout << "*** options_traj_opt ***" << std::endl;
  options_trajopt.print(std::cout);
  std::cout << "***" << std::endl;

  Problem problem(env_file.c_str());

  Info_out info_out_omplgeo;
  Trajectory traj_out;

  solve_ompl_geometric(problem, options_geo, options_trajopt, traj_out,
                       info_out_omplgeo);

  std::cout << "*** info_out_omplgeo   *** " << std::endl;
  info_out_omplgeo.print(std::cout);
  std::cout << "***" << std::endl;

  std::ofstream results(results_file);

  results << "alg: ompl_geo" << std::endl;
  results << "time_stamp: " << get_time_stamp() << std::endl;

  results << "options_geo:" << std::endl;
  options_geo.print(results, "  ");

  results << "options_trajopt:" << std::endl;
  options_trajopt.print(results, "  ");

  info_out_omplgeo.to_yaml(results);
}
