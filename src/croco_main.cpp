#include "ocp.hpp"

int main(int argc, const char *argv[]) {

  // auto argv = boost::unit_test::framework::master_test_suite().argv;
  // auto argc = boost::unit_test::framework::master_test_suite().argc;

  File_parser_inout file_inout;

  std::string config_file = "";
  std::string out = "out.yaml";
  std::string out_bench = "out_bench.yaml";
  std::string yaml_solver_file = "";
  std::string yaml_problem_file = "";

  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");
  opti_params.add_options(desc);
  file_inout.add_options(desc);

  set_from_boostop(desc, VAR_WITH_NAME(out));
  set_from_boostop(desc, VAR_WITH_NAME(out_bench));
  set_from_boostop(desc, VAR_WITH_NAME(config_file));
  set_from_boostop(desc, VAR_WITH_NAME(yaml_solver_file));
  set_from_boostop(desc, VAR_WITH_NAME(yaml_problem_file));

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return 1;
    }

    if (config_file != "") {
      std::ifstream ifs{config_file};
      if (ifs)
        store(parse_config_file(ifs, desc), vm);
      else {
        std::cout << "cannont open config file: " << config_file << std::endl;
      }
      notify(vm);
    }

    if (yaml_solver_file != "") {
      opti_params.read_from_yaml(yaml_solver_file.c_str());
    }
    if (yaml_problem_file != "") {
      file_inout.read_from_yaml(yaml_problem_file.c_str());
    }

    std::cout << "***" << std::endl;
    std::cout << "VARIABLE MAP" << std::endl;
    PrintVariableMap(vm, std::cout);
    std::cout << "***" << std::endl;

    std::cout << "***" << std::endl;
    std::cout << "OPTI PARAMS" << std::endl;
    opti_params.print(std::cout);
    std::cout << "***" << std::endl;

    std::cout << "***" << std::endl;
    std::cout << "FILE INOUT" << std::endl;
    file_inout.print(std::cout);
    std::cout << "***" << std::endl;

  } catch (po::error &e) {
    std::cerr << e.what() << std::endl << std::endl;
    std::cerr << desc << std::endl;
    return 1;
  }

  Result_opti result;
  crocoddyl::Timer timer;
  accumulated_time = 0.;
  compound_solvers(file_inout, result);
  double d = timer.get_duration();
  std::cout << STR_(accumulated_time) << std::endl;

  std::cout << "writing results original " << std::endl;
  std::ofstream file_out(out);
  result.write_yaml_db(file_out);

  std::cout << "writing results benchmark " << std::endl;
  std::ofstream ff(out_bench);
  std::cout << "writing to " << out << std::endl;
  ff << "solver_name: " << opti_params.solver_name << std::endl;
  ff << "problem_name: " << file_inout.problem_name << std::endl;
  ff << "feasible: " << result.feasible << std::endl;
  ff << "cost: " << result.cost << std::endl;
  ff << "time_solve: " << accumulated_time << std::endl;
  ff << "time_total: " << d << std::endl;
  ff << "time_stamp: " << get_time_stamp() << std::endl;
  ff << "solver_file: " << yaml_solver_file << std::endl;
  ff << "problem_file: " << yaml_problem_file << std::endl;
}
