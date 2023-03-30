#include "ocp.hpp"
#include "ompl_geo.hpp"

int main(int argc, char *argv[]) {

  Options_ompl_geo options_ompl_geo;
  Options_trajopt options_trajopt;

  std::string env_file;
  po::options_description desc("Allowed options");
  options_ompl_geo.add_options(desc);
  options_trajopt.add_options(desc);
  set_from_boostop(desc, VAR_WITH_NAME(env_file));

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

  // if (cfg_file != "") {
  //   options_db.read_from_yaml(cfg_file.c_str());
  // }
  //
  // if (data_file != "") {
  //   inout_db.read_from_yaml(data_file.c_str());
  // }

  std::cout << "*** options_ompl_geo ***" << std::endl;
  options_ompl_geo.print(std::cout);
  std::cout << "***" << std::endl;

  std::cout << "*** options_traj_opt ***" << std::endl;
  options_trajopt.print(std::cout);
  std::cout << "***" << std::endl;

  Problem problem(env_file.c_str());

  Info_out_omplgeo info_out_omplgeo;
  Trajectory traj_out;

  solve_ompl_geometric(problem, options_ompl_geo, options_trajopt, traj_out,
                       info_out_omplgeo);

  std::cout << "*** info_out_omplgeo   *** " << std::endl;
  info_out_omplgeo.print(std::cout);
  std::cout << "***" << std::endl;

  if (info_out_omplgeo.solved) {
    return 0;
  } else {
    return 1;
  }
}
