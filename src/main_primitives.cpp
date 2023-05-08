

enum class PRIMITIVE_MODE {
  generate = 0,
  improve = 1,
  split = 2,
  sort = 3,
  merge = 4,
  check = 5,
  stats = 6,
  cut = 7,
  shuffle = 8,
};

#include "generate_primitives.hpp"

int main(int argc, const char *argv[]) {

  std::cout << "seed with time " << std::endl;
  srand((unsigned int)time(0));

  CSTR_(__FILE__);
  CSTR_(argc);

  std::cout << "argv: " << std::endl;
  for (int i = 0; i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }

  std::string in_file;
  std::string out_file = "auto";
  int mode_gen_id = 0;
  std::string cfg_file = "";
  Options_trajopt options_trajopt;
  Options_primitives options_primitives;

  po::options_description desc("Allowed options");
  desc.add_options()("help", "produce help message");
  set_from_boostop(desc, VAR_WITH_NAME(in_file));
  set_from_boostop(desc, VAR_WITH_NAME(out_file));
  set_from_boostop(desc, VAR_WITH_NAME(mode_gen_id));
  set_from_boostop(desc, VAR_WITH_NAME(cfg_file));
  options_primitives.add_options(desc);
  options_trajopt.add_options(desc);

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
    options_trajopt.read_from_yaml(cfg_file.c_str());
  }

  std::cout << " *** options_primitives *** " << std::endl;
  options_primitives.print(std::cout);
  std::cout << " *** " << std::endl;
  std::cout << " *** options_trajopt *** " << std::endl;
  options_trajopt.print(std::cout);
  std::cout << " *** " << std::endl;

  PRIMITIVE_MODE mode = static_cast<PRIMITIVE_MODE>(mode_gen_id);

  std::shared_ptr<Model_robot> robot_model =
      robot_factory(robot_type_to_path(options_primitives.dynamics).c_str());

  if (mode == PRIMITIVE_MODE::generate) {

    Trajectories trajectories;

    generate_primitives(options_trajopt, options_primitives, trajectories);

    trajectories.save_file_boost(out_file.c_str());
    trajectories.save_file_yaml((out_file + ".yaml").c_str(), 1000);
    trajectories.compute_stats("tmp_stats.yaml");
  }

  if (mode == PRIMITIVE_MODE::improve) {
    Trajectories trajectories, trajectories_out;

    trajectories.load_file_boost(in_file.c_str());

    if (options_primitives.max_num_primitives > 0 &&
        static_cast<size_t>(options_primitives.max_num_primitives) <
            trajectories.data.size()) {
      trajectories.data.resize(options_primitives.max_num_primitives);
    }

    // Options_trajopt options_trajopt;
    // options_trajopt.solver_id =
    //     static_cast<int>(SOLVER::traj_opt_free_time_linear);

    improve_motion_primitives(options_trajopt, trajectories,
                              options_primitives.dynamics, trajectories_out);

    trajectories_out.save_file_boost(out_file.c_str());
    trajectories_out.save_file_yaml((out_file + ".yaml").c_str(), 1000);

    trajectories_out.compute_stats("tmp_stats.yaml");
  }

  if (mode == PRIMITIVE_MODE::split) {

    Trajectories trajectories, trajectories_out;
    trajectories.load_file_boost(in_file.c_str());

    if (startsWith(options_primitives.dynamics, "quad3d")) {

      for (auto &t : trajectories.data) {

        for (auto &s : t.states) {
          s.segment<4>(3).normalize();
        }
      }
    }

    split_motion_primitives(trajectories, options_primitives.dynamics,
                            trajectories_out, options_primitives);

    // std::vector<bool> valid(trajectories_out.data.size(), true);
    Trajectories __trajectories_out;
    for (size_t i = 0; i < trajectories_out.data.size(); i++) {
      auto &traj = trajectories_out.data.at(i);
      // traj.check(robot_model, true);
      traj.check(robot_model);
      traj.update_feasibility();
      if (traj.feasible) {
        __trajectories_out.data.push_back(traj);
      }
    }

    trajectories_out = __trajectories_out;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(trajectories_out.data.begin(), trajectories_out.data.end(), g);
    if (out_file == "auto") {
      out_file = in_file + ".sp.bin";
    }

    trajectories_out.save_file_boost(out_file.c_str());
    trajectories_out.save_file_yaml((out_file + ".yaml").c_str(), 1000);
    trajectories_out.compute_stats("tmp_stats.yaml");
  }

  if (mode == PRIMITIVE_MODE::sort) {

    Trajectories trajectories, trajectories_out;
    trajectories.load_file_boost(in_file.c_str());
    CSTR_(trajectories.data.size());

    sort_motion_primitives(
        trajectories, trajectories_out,
        [&](const auto &x, const auto &y) {
          return robot_model->distance(x, y);
        },
        options_primitives.max_num_primitives);

    // check that they are valid...

    for (auto &traj : trajectories_out.data) {
      traj.check(robot_model);
      traj.update_feasibility();
      CHECK(traj.feasible, AT);
    }

    if (out_file == "auto") {
      out_file = in_file + ".so.bin";
    }

    CSTR_(trajectories_out.data.size());
    trajectories_out.save_file_boost(out_file.c_str());
    trajectories_out.save_file_yaml((out_file + ".yaml").c_str(), 1000);

    trajectories_out.compute_stats("tmp_stats.yaml");
  }

  if (mode == PRIMITIVE_MODE::check) {

    Trajectories trajectories;
    trajectories.load_file_boost(in_file.c_str());
    CSTR_(trajectories.data.size());

    for (auto &traj : trajectories.data) {
      traj.check(robot_model);
      traj.update_feasibility();
      CHECK(traj.feasible, AT);
    }

    trajectories.compute_stats("tmp_stats.yaml");
  }

  if (mode == PRIMITIVE_MODE::cut) {

    Trajectories trajectories;
    trajectories.load_file_boost(in_file.c_str());
    CSTR_(trajectories.data.size());

    trajectories.data.resize(options_primitives.max_num_primitives);

    if (out_file == "auto") {
      out_file = in_file + ".c" +
                 std::to_string(options_primitives.max_num_primitives) + ".bin";
    }

    trajectories.save_file_boost(out_file.c_str());
    trajectories.compute_stats("tmp_stats.yaml");
  }

  if (mode == PRIMITIVE_MODE::stats) {
    Trajectories trajectories;
    trajectories.load_file_boost(in_file.c_str());
    if (options_primitives.max_num_primitives > 0 &&
        options_primitives.max_num_primitives < trajectories.data.size())
      trajectories.data.resize(options_primitives.max_num_primitives);
    if (out_file == "auto") {
      out_file = in_file + ".stats.yaml";
    }
    CSTR_(trajectories.data.size());
    trajectories.compute_stats(out_file.c_str());
  }

  if (mode == PRIMITIVE_MODE::shuffle) {
    Trajectories trajectories;
    trajectories.load_file_boost(in_file.c_str());
    if (options_primitives.max_num_primitives > 0 &&
        options_primitives.max_num_primitives < trajectories.data.size())
      trajectories.data.resize(options_primitives.max_num_primitives);
    CSTR_(trajectories.data.size());

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(trajectories.data.begin(), trajectories.data.end(), g);
    trajectories.save_file_boost(in_file.c_str());
    trajectories.compute_stats("tmp_stats.yaml");
  }
}
