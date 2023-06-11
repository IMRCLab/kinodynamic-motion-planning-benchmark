#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <nigh/kdtree_batch.hpp>
#include <nigh/se3_space.hpp>
#include <nigh/so3_space.hpp>
#pragma GCC diagnostic pop

// qw, qx, qy, qz, x, y, z
static constexpr std::array poses{
    std::array{0.662702f, 0.219097f, 0.211232f, 0.684254f, 0.0294779f,
               0.0123540f, -0.0139259f}, // 0
    std::array{0.658478f, 0.256881f, 0.256784f, 0.659152f, 0.0228988f,
               -0.0116276f, -0.0211147f}, // 1
    std::array{0.658478f, 0.256881f, 0.256784f, 0.659152f, 0.0193635f,
               -0.0116276f, -0.0175792f}, // 2

    std::array{0.641592f, -0.290111f, 0.286126f, -0.649867f, -0.0893326f,
               0.0113263f, 0.0924346f}, // 3

    std::array{0.644836f, -0.274049f, 0.267667f, -0.661391f, -0.0922808f,
               -0.0118977f, 0.0894458f}, // 4
    std::array{0.642503f, -0.279932f, 0.273478f, -0.658816f, -0.0919003f,
               -0.0130335f, 0.0910987f}, // 5
    std::array{0.642491f, -0.290219f, 0.287658f, -0.648252f, -0.0937908f,
               -0.0273179f, 0.0970888f}, // 6
    std::array{0.644836f, -0.274050f, 0.267667f, -0.661391f, -0.0958166f,
               -0.0118978f, 0.0929817f}, // 7
    std::array{0.642443f, -0.290327f, 0.287766f, -0.648203f, -0.0937701f,
               -0.0264953f, 0.0971051f} // 8
};

void test_ori() {
  std::cout << "TEST ORI\n";

  for (std::size_t eliminated = 0; eliminated < poses.size() + 1;
       ++eliminated) {
    if (eliminated < poses.size()) {
      std::cout << "eliminated " << eliminated << std::endl;
    } else {
      std::cout << "eliminated none" << std::endl;
    }
    namespace nigh = unc::robotics::nigh;
    using Space = nigh::SO3Space<float>;

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> d{0, 1};
    std::vector<Space::Type> ps;
    ps.reserve(poses.size());
    for (auto [qw, qx, qy, qz, x, y, z] : poses) {
      ps.emplace_back(qw, qx, qy, qz);
    }

    const auto key = [&ps](std::size_t i) -> const Space::Type {
      return ps.at(i);
    };

    using Type = std::size_t;
    using Key = decltype(key);
    using Tree =
        nigh::Nigh<Type, Space, Key, nigh::NoThreadSafety, nigh::KDTreeBatch<>>;

    Tree tree{Space{}, key};
    for (std::size_t i = 0; i < ps.size(); ++i) {
      if (i == eliminated) {
        continue;
      }
      const auto &q = ps.at(i);
      std::cout << q.w() << "f, " << q.x() << "f, " << q.y() << "f, " << q.z()
                << 'f' << std::endl;
      tree.insert(i);
    }
  }
  std::cout << "TEST ORI DONE\n";
}

void test_pose() {
  std::cout << "TEST\n";
  for (std::size_t eliminated = 0; eliminated < poses.size() + 1;
       ++eliminated) {
    if (eliminated < poses.size()) {
      std::cout << "eliminated " << eliminated << std::endl;
    } else {
      std::cout << "eliminated none" << std::endl;
    }
    namespace nigh = unc::robotics::nigh;
    using Space = nigh::SE3Space<float>;

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<float> d{0, 1};
    std::vector<Space::Type> ps;
    ps.reserve(poses.size());
    for (auto [qw, qx, qy, qz, x, y, z] : poses) {
      ps.emplace_back(std::make_tuple(Eigen::Quaternionf{qw, qx, qy, qz},
                                      Eigen::Vector3f{d(gen), d(gen), d(gen)}));
    }

    const auto key = [&ps](std::size_t i) -> const Space::Type {
      return ps.at(i);
    };

    using Type = std::size_t;
    using Key = decltype(key);
    using Tree =
        nigh::Nigh<Type, Space, Key, nigh::NoThreadSafety, nigh::KDTreeBatch<>>;

    Tree tree{Space{}, key};
    for (std::size_t i = 0; i < ps.size(); ++i) {
      if (i == eliminated) {
        continue;
      }
      const auto &[q, v] = ps.at(i);
      std::cout << q.w() << "f, " << q.x() << "f, " << q.y() << "f, " << q.z()
                << "f, " << v(0) << "f, " << v(1) << "f, " << v(2) << 'f'
                << std::endl;
      tree.insert(i);
    }
  }
  std::cout << "TEST DONE\n";
}
int main() {
  test_ori();
  test_pose();
  return 0;
}
