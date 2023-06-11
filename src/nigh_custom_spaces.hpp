#include <nigh/kdtree_batch.hpp>
#include <nigh/kdtree_median.hpp>
#include <nigh/se3_space.hpp>
#include <nigh/so2_space.hpp>
#include <nigh/so3_space.hpp>

namespace nigh = unc::robotics::nigh;

using Space = nigh::CartesianSpace<
    nigh::L2Space<double, 2>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>>;

using __Space =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>>;

using SpaceUni2 = nigh::CartesianSpace<
    nigh::L2Space<double, 2>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::L2Space<double, 1>, std::ratio<1, 4>>,
    nigh::ScaledSpace<nigh::L2Space<double, 1>, std::ratio<1, 4>>>;

using __SpaceUni2 =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 1>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 1>>>;

// x y theta  vx  vw
using SpaceQuad2d = nigh::CartesianSpace<
    nigh::L2Space<double, 2>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::L2Space<double, 2>, std::ratio<1, 5>>,
    nigh::ScaledSpace<nigh::L2Space<double, 1>, std::ratio<1, 10>>>;

using __SpaceQuad2dPole =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 1>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 1>>>;

using __SpaceQuad2d =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 1>>>;

using SpaceAcrobot = nigh::CartesianSpace<
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::L2Space<double, 2>, std::ratio<1, 5>>>;

using __SpaceAcrobot =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 2>>>;

using SpaceQuad3d = nigh::CartesianSpace<
    nigh::L2Space<double, 3>,
    nigh::ScaledSpace<nigh::SO3Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::L2Space<double, 3>, std::ratio<1, 10>>,
    nigh::ScaledSpace<nigh::L2Space<double, 3>, std::ratio<1, 20>>>;

using __SpaceQuad3d =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::L2Space<double, 3>>,
                         nigh::ScaledSpace<nigh::SO3Space<double>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 3>>,
                         nigh::ScaledSpace<nigh::L2Space<double, 3>>>;

using SpaceCar1 = nigh::CartesianSpace<
    nigh::L2Space<double, 2>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>,
    nigh::ScaledSpace<nigh::SO2Space<double>, std::ratio<1, 2>>>;

using __SpaceCar1 =
    nigh::CartesianSpace<nigh::ScaledSpace<nigh::L2Space<double, 2>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>,
                         nigh::ScaledSpace<nigh::SO2Space<double>>>;
