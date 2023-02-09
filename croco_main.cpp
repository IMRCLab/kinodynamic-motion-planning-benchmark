




BOOST_AUTO_TEST_CASE(quim) {

  auto argv = boost::unit_test::framework::master_test_suite().argv;
  auto argc = boost::unit_test::framework::master_test_suite().argc;

  int verbose = 0;

  File_parser_inout file_inout;
  std::string config_file = "";
  std::string out;

  po::options_description desc("Allowed options");

  desc.add_options()("help", "produce help message");
  opti_params.add_options(desc);
  file_inout.add_options(desc);

  desc.add_options()("out", po::value<std::string>(&out)->required())(
      "config_file",
      po::value<std::string>(&config_file)->default_value(config_file));

  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help") != 0u) {
      std::cout << desc << "\n";
      return;
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
    return;
  }

  Result_opti result;
  compound_solvers(file_inout, result);

  std::cout << "writing results " << std::endl;

  std::ofstream file_out(out);
  result.write_yaml_db(file_out);

  std::cout << "accumulated time in solve is " << accumulated_time << std::endl;
}
