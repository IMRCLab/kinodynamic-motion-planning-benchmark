
#include <memory>
#include <mutex>

#include <cassert> // bug in /home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/nigh/src/nigh/impl/non_atomic.hpp:88:19: erro
//

#include "croco_macros.hpp"
#include "nigh/kdtree_batch.hpp"
#include "nigh/kdtree_median.hpp"
#include "nigh/lp_space.hpp"
#include "nigh/so3_space.hpp"
#include <nigh/cartesian_space.hpp>
#include <nigh/scaled_space.hpp>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

// FCL
#include <fcl/fcl.h>

// YAML
#include <yaml-cpp/yaml.h>

// OMPL
#include "ocp.hpp"
#include <ompl/datastructures/NearestNeighbors.h>
#include <ompl/datastructures/NearestNeighborsGNATNoThreadSafety.h>
#include <ompl/datastructures/NearestNeighborsSqrtApprox.h>

// local
// #include "collision_checker.hpp"
#include "robots.h"

using namespace pybind11::literals;

struct Key {
  template <class T> const T &operator()(const T &t) const { return t; }
};

struct Key2 {
  template <class T> T *operator()(T *t) const { return t; }
};

// class RobotHelper {
// public:
//   RobotHelper(const std::string &robotType, float pos_limit = 2) {
//     size_t dim = 2;
//     if (robotType == "quadrotor_0") {
//       dim = 3;
//     }
//
//     ob::RealVectorBounds position_bounds(dim);
//     position_bounds.setLow(-pos_limit);
//     position_bounds.setHigh(pos_limit);
//     robot_ = create_robot_ompl(robotType, position_bounds);
//
//     auto si = robot_->getSpaceInformation();
//     si->getStateSpace()->setup();
//     state_sampler_ = si->allocStateSampler();
//     control_sampler_ = si->allocControlSampler();
//
//     tmp_state_a_ = si->allocState();
//     tmp_state_b_ = si->allocState();
//     tmp_control_ = si->allocControl();
//   }
//
//   ~RobotHelper() {
//     auto si = robot_->getSpaceInformation();
//     si->freeState(tmp_state_a_);
//     si->freeState(tmp_state_b_);
//     si->freeControl(tmp_control_);
//   }
//
//   float distance(const std::vector<double> &stateA,
//                  const std::vector<double> &stateB) {
//     auto si = robot_->getSpaceInformation();
//     si->getStateSpace()->copyFromReals(tmp_state_a_, stateA);
//     si->getStateSpace()->copyFromReals(tmp_state_b_, stateB);
//     return si->distance(tmp_state_a_, tmp_state_b_);
//   }
//
//   std::vector<double> sampleStateUniform() {
//     auto si = robot_->getSpaceInformation();
//     do {
//       state_sampler_->sampleUniform(tmp_state_a_);
//     } while (!si->satisfiesBounds(tmp_state_a_));
//     std::vector<double> reals;
//     si->getStateSpace()->copyToReals(reals, tmp_state_a_);
//     return reals;
//   }
//
//   std::vector<double> sampleControlUniform() {
//     control_sampler_->sample(tmp_control_);
//     auto si = robot_->getSpaceInformation();
//     const size_t dim = si->getControlSpace()->getDimension();
//     std::vector<double> reals(dim);
//     for (size_t d = 0; d < dim; ++d) {
//       double *address =
//           si->getControlSpace()->getValueAddressAtIndex(tmp_control_, d);
//       reals[d] = *address;
//     }
//     return reals;
//   }
//
//   std::vector<double> step(const std::vector<double> &state,
//                            const std::vector<double> &action, double
//                            duration) {
//     auto si = robot_->getSpaceInformation();
//     si->getStateSpace()->copyFromReals(tmp_state_a_, state);
//
//     const size_t dim = si->getControlSpace()->getDimension();
//     assert(dim == action.size());
//     for (size_t d = 0; d < dim; ++d) {
//       double *address =
//           si->getControlSpace()->getValueAddressAtIndex(tmp_control_, d);
//       *address = action[d];
//     }
//     robot_->propagate(tmp_state_a_, tmp_control_, duration, tmp_state_b_);
//
//     std::vector<double> reals;
//     si->getStateSpace()->copyToReals(reals, tmp_state_b_);
//     return reals;
//   }
//
//   std::vector<double> interpolate(const std::vector<double> &stateFrom,
//                                   const std::vector<double> &stateTo,
//                                   double t) {
//     auto si = robot_->getSpaceInformation();
//     si->getStateSpace()->copyFromReals(tmp_state_a_, stateFrom);
//     si->getStateSpace()->copyFromReals(tmp_state_b_, stateTo);
//
//     si->getStateSpace()->interpolate(tmp_state_a_, tmp_state_b_, t,
//                                      tmp_state_a_);
//
//     std::vector<double> reals;
//     si->getStateSpace()->copyToReals(reals, tmp_state_a_);
//     return reals;
//   }
//
//   bool is2D() const { return robot_->is2D(); }
//   bool isTranslationInvariant() const {
//     return robot_->isTranslationInvariant();
//   }
//
//
// private:
//   std::shared_ptr<RobotOmpl> robot_;
//   ob::StateSamplerPtr state_sampler_;
//   oc::ControlSamplerPtr control_sampler_;
//   ob::State *tmp_state_a_;
//   ob::State *tmp_state_b_;
//   oc::Control *tmp_control_;
// };
//
PYBIND11_MODULE(motionplanningutils, m) {
  //   pybind11::class_<CollisionChecker>(m, "CollisionChecker")
  //       .def(pybind11::init())
  //       .def("load", &CollisionChecker::load)
  //       .def("distance", &CollisionChecker::distance)
  //       .def("distanceWithFDiffGradient",
  //            &CollisionChecker::distanceWithFDiffGradient);
  //
  //   pybind11::class_<RobotHelper>(m, "RobotHelper")
  //       .def(pybind11::init<const std::string &, float>(),
  //       py::arg("robot_type"),
  //            py::arg("pos_limit") = 2)
  //       .def("distance", &RobotHelper::distance)
  //       .def("sampleUniform", &RobotHelper::sampleStateUniform)
  //       .def("sampleControlUniform", &RobotHelper::sampleControlUniform)
  //       .def("step", &RobotHelper::step)
  //       .def("interpolate", &RobotHelper::interpolate)
  //       .def("is2D", &RobotHelper::is2D)
  //       .def("isTranslationInvariant", &RobotHelper::isTranslationInvariant)
  //       .def("sortMotions", &RobotHelper::sortMotions3);

  pybind11::class_<Model_robot>(m, "Model_robot")
      .def(pybind11::init())
      .def("setPositionBounds", &Model_robot::setPositionBounds)
      .def("get_translation_invariance",
           &Model_robot::get_translation_invariance)
      .def("get_x_ub", &Model_robot::get_x_ub)
      .def("set_position_ub", &Model_robot::set_position_ub)
      .def("set_position_lb", &Model_robot::set_position_lb)
      .def("get_x_lb", &Model_robot::get_x_lb)
      .def("get_nx", &Model_robot::get_nx)
      .def("get_nu", &Model_robot::get_nu)

      .def("get_u_ub", &Model_robot::get_u_ub)
      .def("get_u_lb", &Model_robot::get_u_lb)
      .def("get_x_desc", &Model_robot::get_x_desc)
      .def("get_u_desc", &Model_robot::get_u_desc)
      .def("get_u_ref", &Model_robot::get_u_ref)
      .def("stepDiff", &Model_robot::stepDiff)
      // .def("stepDiffdt", &Model_robot::stepDiffdt)
      .def("calcDiffV", &Model_robot::calcDiffV)
      .def("calcV", &Model_robot::calcV)
      .def("step", &Model_robot::step)
      .def("stepR4", &Model_robot::stepR4)
      .def("distance", &Model_robot::distance)
      .def("sample_uniform", &Model_robot::sample_uniform)
      .def("interpolate", &Model_robot::interpolate)
      .def("lower_bound_time", &Model_robot::lower_bound_time)
      .def("collision_distance", &Model_robot::collision_distance)
      .def("collision_distance_diff", &Model_robot::collision_distance_diff)
      .def("transformation_collision_geometries",
           &Model_robot::transformation_collision_geometries);
  // .def("load_env_quim", &Model_robot::load_env_quim);

  m.def("robot_factory", robot_factory);

  // pybind11::class_<Model_unicycle1>(m, "Model_unicycle1")
  //     .def(pybind11::init())
  //     .def("calcV", &Model_unicycle1::calcV)
  //     .def("calcVDiff", &Model_unicycle1::calcDiffV)
  //     .def("step", &Model_unicycle1::step)
  //     .def("interpolate", &Model_unicycle1::interpolate)
  //     .def("lower_bound_time", &Model_unicycle1::lower_bound_time)
  //     .def("distance", &Model_unicycle1::distance)
  //     .def_readonly("u_ub", &Model_unicycle1::u_ub)
  //     .def_readonly("u_lb", &Model_unicycle1::u_lb)
  //     .def_readonly("x_ub", &Model_unicycle1::x_ub)
  //     .def_readonly("x_lb", &Model_unicycle1::x_lb)
  //     .def_readonly("x_desc", &Model_unicycle1::x_desc)
  //     .def_readonly("u_desc", &Model_unicycle1::u_desc);
}
