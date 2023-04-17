

#include "motions.hpp"
#include <filesystem>

int main(int argc, const char *argv[]) {

  std::string in_folder;
  std::string out_file;

  po::options_description desc("Allowed options");
  set_from_boostop(desc, VAR_WITH_NAME(in_folder));
  set_from_boostop(desc, VAR_WITH_NAME(out_file));

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

  CSTR_(in_folder);
  CSTR_(out_file);

  std::vector<std::string> files;
  for (const auto &entry : std::filesystem::directory_iterator(in_folder)) {
    files.push_back(entry.path());
  }

  Trajectories trajectories_all;
  for (auto &file : files) {
    Trajectories trajectories;
    trajectories.load_file_boost(file.c_str());
    CSTR_(trajectories.data.size());
    trajectories_all.data.insert(trajectories_all.data.end(),
                                 trajectories.data.begin(),
                                 trajectories.data.end());
    CSTR_(trajectories_all.data.size());
  }
  trajectories_all.save_file_boost(out_file.c_str());
}
