#include <fstream>
#include <iostream>
#include <algorithm>

#include <yaml-cpp/yaml.h>

#include <boost/program_options.hpp>

// #include <fcl/collision.h>
// #include <fcl/collision_object.h>
// #include <fcl/broadphase/broadphase.h>
#include <fcl/fcl.h>

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;
  // Declare the supported options.
  po::options_description desc("Allowed options");
  std::string inputFile;
  std::string outputFile;
  desc.add_options()
    ("help", "produce help message")
    ("input,i", po::value<std::string>(&inputFile)->required(), "input file (yaml)")
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

  YAML::Node env = YAML::LoadFile(inputFile);

  // load obstacles
  std::vector<fcl::CollisionObjectf*> obstacles;
  for (const auto& obs : env["environment"]["obstacles"]) {
    if (obs["type"].as<std::string>() == "box") {
      const auto& size = obs["size"];
      std::shared_ptr<fcl::CollisionGeometryf> geom;
      geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0));
      const auto& center = obs["center"];
      auto co = new fcl::CollisionObjectf(geom);
      co->setTranslation(fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
      obstacles.push_back(co);
    } else {
      throw std::runtime_error("Unknown obstacle type!");
    }
  }
  fcl::BroadPhaseCollisionManagerf* bpcm_env = new fcl::DynamicAABBTreeCollisionManagerf();
  bpcm_env->registerObjects(obstacles);
  bpcm_env->setup();

  // load robot
  fcl::BroadPhaseCollisionManagerf *bpcm_robot = new fcl::DynamicAABBTreeCollisionManagerf();

  const auto& robot = env["robots"][0];
  if (robot["type"].as<std::string>() == "dubins_0") {
    std::shared_ptr<fcl::CollisionGeometryf> geom;
    geom.reset(new fcl::Boxf(0.5, 0.25, 1.0));
    auto co = new fcl::CollisionObjectf(geom);
    co->setTranslation(fcl::Vector3f(3, 3.5, 0));
    bpcm_robot->registerObject(co);
  } else {
    throw std::runtime_error("Unknown robot type!");
  }
  bpcm_robot->setup();

  fcl::DefaultCollisionData<float> collision_data;
  bpcm_env->collide(bpcm_robot, &collision_data, fcl::DefaultCollisionFunction<float>);

  std::cout << collision_data.result.isCollision() << std::endl;

  return 0;
}
