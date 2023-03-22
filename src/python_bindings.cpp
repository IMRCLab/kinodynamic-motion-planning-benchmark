

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

template <typename Fun, typename... Args>
auto timed_fun(Fun fun, Args &&...args) {
  auto tic = std::chrono::high_resolution_clock::now();
  auto out = fun(std::forward<Args>(args)...);
  auto tac = std::chrono::high_resolution_clock::now();
  return std::make_pair(
      out, std::chrono::duration<double, std::milli>(tac - tic).count());
}

void print_matrix(std::ostream &out,
                  const std::vector<std::vector<double>> &data) {

  for (auto &v : data) {
    for (auto &e : v)
      out << e << " ";
    out << std::endl;
  }
}

// local
#include "collision_checker.hpp"
#include "robotStatePropagator.hpp"
#include "robots.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace ob = ompl::base;
namespace oc = ompl::control;

struct Key {
  template <class T> const T &operator()(const T &t) const { return t; }
};

struct Key2 {
  template <class T> T *operator()(T *t) const { return t; }
};

class RobotHelper {
public:
  RobotHelper(const std::string &robotType, float pos_limit = 2) {
    size_t dim = 2;
    if (robotType == "quadrotor_0") {
      dim = 3;
    }

    ob::RealVectorBounds position_bounds(dim);
    position_bounds.setLow(-pos_limit);
    position_bounds.setHigh(pos_limit);
    robot_ = create_robot_ompl(robotType, position_bounds);

    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->setup();
    state_sampler_ = si->allocStateSampler();
    control_sampler_ = si->allocControlSampler();

    tmp_state_a_ = si->allocState();
    tmp_state_b_ = si->allocState();
    tmp_control_ = si->allocControl();
  }

  ~RobotHelper() {
    auto si = robot_->getSpaceInformation();
    si->freeState(tmp_state_a_);
    si->freeState(tmp_state_b_);
    si->freeControl(tmp_control_);
  }

  float distance(const std::vector<double> &stateA,
                 const std::vector<double> &stateB) {
    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_a_, stateA);
    si->getStateSpace()->copyFromReals(tmp_state_b_, stateB);
    return si->distance(tmp_state_a_, tmp_state_b_);
  }

  std::vector<double> sampleStateUniform() {
    auto si = robot_->getSpaceInformation();
    do {
      state_sampler_->sampleUniform(tmp_state_a_);
    } while (!si->satisfiesBounds(tmp_state_a_));
    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, tmp_state_a_);
    return reals;
  }

  std::vector<double> sampleControlUniform() {
    control_sampler_->sample(tmp_control_);
    auto si = robot_->getSpaceInformation();
    const size_t dim = si->getControlSpace()->getDimension();
    std::vector<double> reals(dim);
    for (size_t d = 0; d < dim; ++d) {
      double *address =
          si->getControlSpace()->getValueAddressAtIndex(tmp_control_, d);
      reals[d] = *address;
    }
    return reals;
  }

  std::vector<double> step(const std::vector<double> &state,
                           const std::vector<double> &action, double duration) {
    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_a_, state);

    const size_t dim = si->getControlSpace()->getDimension();
    assert(dim == action.size());
    for (size_t d = 0; d < dim; ++d) {
      double *address =
          si->getControlSpace()->getValueAddressAtIndex(tmp_control_, d);
      *address = action[d];
    }
    robot_->propagate(tmp_state_a_, tmp_control_, duration, tmp_state_b_);

    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, tmp_state_b_);
    return reals;
  }

  std::vector<double> interpolate(const std::vector<double> &stateFrom,
                                  const std::vector<double> &stateTo,
                                  double t) {
    auto si = robot_->getSpaceInformation();
    si->getStateSpace()->copyFromReals(tmp_state_a_, stateFrom);
    si->getStateSpace()->copyFromReals(tmp_state_b_, stateTo);

    si->getStateSpace()->interpolate(tmp_state_a_, tmp_state_b_, t,
                                     tmp_state_a_);

    std::vector<double> reals;
    si->getStateSpace()->copyToReals(reals, tmp_state_a_);
    return reals;
  }

  bool is2D() const { return robot_->is2D(); }
  bool isTranslationInvariant() const {
    return robot_->isTranslationInvariant();
  }

  std::vector<size_t> sortMotions2(const std::vector<std::vector<double>> &x0s,
                                   const std::vector<std::vector<double>> &xfs,
                                   size_t top_k) {
    assert(x0s.size() == xfs.size());
    assert(x0s.size() > 0);
    auto si = robot_->getSpaceInformation();

    struct Motion {
      ob::State *x0;
      ob::State *xf;
      size_t idx;
    };
    // create vector of motions
    std::vector<Motion> motions;
    for (size_t i = 0; i < x0s.size(); ++i) {
      Motion m;
      m.x0 = si->allocState();
      si->getStateSpace()->copyFromReals(m.x0, x0s[i]);
      si->enforceBounds(m.x0);
      m.xf = si->allocState();
      si->getStateSpace()->copyFromReals(m.xf, xfs[i]);
      si->enforceBounds(m.xf);
      m.idx = i;
      motions.push_back(m);
    }

    {
      using namespace unc::robotics::nigh;

      using Space = CartesianSpace<
          ScaledSpace<L2Space<double, 3>, std::ratio<1, 1>>,  // x
          ScaledSpace<SO3Space<double>, std::ratio<1, 1>>,    // quat
          ScaledSpace<L2Space<double, 3>, std::ratio<1, 20>>, // v
          ScaledSpace<L2Space<double, 3>, std::ratio<1, 20>>  // w
          >;

      // Space space(weight_p, weight_q, weight_v, weight_w);

      // Nigh<..., Space, ...> nn(space);

      using State = std::tuple<Eigen::Vector3d, Eigen::Quaterniond,
                               Eigen::Vector3d, Eigen::Vector3d>;

      const Eigen::IOFormat FMT(6, Eigen::DontAlignCols, ",", ",", "", "", "[",
                                "]");
      auto print_state = [&](const auto &state) {
        std::cout << "0: " << std::get<0>(state).format(FMT) << std::endl;
        std::cout << "1: " << std::get<1>(state).coeffs().format(FMT)
                  << std::endl;
        std::cout << "2: " << std::get<2>(state).format(FMT) << std::endl;
        std::cout << "3: " << std::get<3>(state).format(FMT) << std::endl;
      };

      struct Qstate {
        State *state;
      };

      struct Key3 {
        const State &operator()(const Qstate &node) const {
          return *node.state;
        }
      };

      Nigh<Qstate, Space, Key3, NoThreadSafety, KDTreeMedian<>> nn;

      std::vector<State> states(xfs.size());
      std::transform(xfs.begin(), xfs.end(), states.begin(), [](const auto &s) {
        State ss;
        std::get<0>(ss) = {s[0], s[1], s[2]};
        std::get<1>(ss) = {s[3], s[4], s[5], s[6]};
        std::get<1>(ss).normalize();
        std::get<2>(ss) = {s[7], s[8], s[9]};
        std::get<3>(ss) = {s[10], s[11], s[12]};
        return ss;
      });

      std::ofstream out_x0("xfs.txt");
      print_matrix(out_x0, xfs);

      for (std::size_t i = 0; i < states.size(); ++i) {
        Qstate q = Qstate{&states[i]};
        nn.insert(q);
      }

      // first check
      for (std::size_t i = 0; i < states.size(); ++i) {
        std::optional<std::pair<Qstate, double>> pt = nn.nearest(states[i]);
        std::cout << "state  " << i << std::endl;
        print_state(states[i]);
        std::cout << "nn  " << std::endl;
        print_state(*pt->first.state);
        CHECK(pt.has_value(), AT);
        CHECK_GEQ(1e-7, pt->second, AT);
      }

      auto out = timed_fun([&]() {
        std::cout << nn.size() << " " << states.size() << std::endl;
        for (std::size_t i = 0; i < states.size(); ++i) {
          std::vector<std::pair<Qstate, double>> nbh;
          // nn.nearest(nbh, states[i], 5,1e10);
          nn.nearest(nbh, states[i], states.size(), 1.);
        }
        return true;
      });

      std::cout << "nigh " << out.second << std::endl;

      ompl::NearestNeighbors<Motion *> *Txf;
      if (si->getStateSpace()->isMetricSpace()) {
        Txf = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
      } else {
        Txf = new ompl::NearestNeighborsSqrtApprox<Motion *>();
      }
      Txf->setDistanceFunction([si](const Motion *a, const Motion *b) {
        return si->distance(a->xf, b->xf);
      });

      for (auto &motion : motions) {
        Txf->add(&motion);
      }

      for (auto &motion : motions) {
        auto p = Txf->nearest(&motion);
        CHECK_SEQ(Txf->getDistanceFunction()(p, &motion), 1e-10, AT);
      }

      auto out2 = timed_fun([&]() {
        std::cout << Txf->size() << " " << motions.size() << std::endl;
        for (std::size_t i = 0; i < motions.size(); ++i) {
          std::vector<Motion *> nbh;
          // Tx0->nearestK(&motions[i], 5, nbh);
          Txf->nearestR(&motions[i], 1, nbh);
        }
        return true;
      });

      std::cout << "ompl " << out2.second << std::endl;

      throw -1;
    }

    // build kd-tree for Tx0
    ompl::NearestNeighbors<Motion *> *Tx0;
    if (si->getStateSpace()->isMetricSpace()) {
      Tx0 = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    } else {
      Tx0 = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    }
    Tx0->setDistanceFunction([si](const Motion *a, const Motion *b) {
      return si->distance(a->x0, b->x0);
    });

    // build kd-tree for Txf
    ompl::NearestNeighbors<Motion *> *Txf;
    if (si->getStateSpace()->isMetricSpace()) {
      Txf = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    } else {
      Txf = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    }
    Txf->setDistanceFunction([si](const Motion *a, const Motion *b) {
      return si->distance(a->xf, b->xf);
    });

    // use as first/seed motion the one that moves furthest
    std::vector<size_t> used_motions;
    size_t best_motion = 0;
    double largest_d = 0;
    for (const auto &m : motions) {
      double d = si->distance(m.x0, m.xf);
      if (d > largest_d) {
        largest_d = d;
        best_motion = m.idx;
      }
    }
    used_motions.push_back(best_motion);
    Tx0->add(&motions[best_motion]);
    Txf->add(&motions[best_motion]);

    std::set<size_t> unused_motions;
    for (const auto &m : motions) {
      unused_motions.insert(m.idx);
    }
    unused_motions.erase(best_motion);
    // TODO: i should use a joint space!

    for (size_t k = 1; k < top_k; ++k) {
      std::cout << "sorting " << k << std::endl;
      double best_d = -1;
      size_t best_motion = -1;
      for (size_t m1 : unused_motions) {
        // find smallest distance to existing neighbors
        auto m2 = Tx0->nearest(&motions[m1]);
        double smallest_d_x0 = si->distance(motions[m1].x0, m2->x0);

        m2 = Txf->nearest(&motions[m1]);
        double smallest_d_xf = si->distance(motions[m1].xf, m2->xf);

        double smallest_d = smallest_d_x0 + smallest_d_xf;
        if (smallest_d > best_d) {
          best_motion = m1;
          best_d = smallest_d;
        }
      }
      used_motions.push_back(best_motion);
      unused_motions.erase(best_motion);
      Tx0->add(&motions[best_motion]);
      Txf->add(&motions[best_motion]);

      if (Tx0->size() && Tx0->size() % 200 == 0) {
        auto p_derived =
            static_cast<ompl::NearestNeighborsGNATNoThreadSafety<Motion *> *>(
                Tx0);
        p_derived->rebuildDataStructure();
        p_derived->rebuildSize_ = 1e8;
      }

      if (Txf->size() && Txf->size() % 200 == 0) {
        auto p_derived =
            static_cast<ompl::NearestNeighborsGNATNoThreadSafety<Motion *> *>(
                Txf);
        p_derived->rebuildDataStructure();
        p_derived->rebuildSize_ = 1e8;
      }
    }

    // clean-up memory
    for (const auto &m : motions) {
      si->freeState(m.x0);
      si->freeState(m.xf);
    }

    return used_motions;
  }

  std::vector<size_t> sortMotions(const std::vector<std::vector<double>> &x0s,
                                  const std::vector<std::vector<double>> &xfs,
                                  size_t top_k) {
    assert(x0s.size() == xfs.size());
    assert(x0s.size() > 0);
    auto si = robot_->getSpaceInformation();
    struct Motion {
      ob::State *x0;
      ob::State *xf;
      size_t idx;
    };
    // create vector of motions
    std::vector<Motion> motions;
    for (size_t i = 0; i < x0s.size(); ++i) {
      Motion m;
      m.x0 = si->allocState();
      si->getStateSpace()->copyFromReals(m.x0, x0s[i]);
      si->enforceBounds(m.x0);
      m.xf = si->allocState();
      si->getStateSpace()->copyFromReals(m.xf, xfs[i]);
      si->enforceBounds(m.xf);
      m.idx = i;
      motions.push_back(m);
    }

    // build kd-tree for Tx0
    ompl::NearestNeighbors<Motion *> *Tx0;
    if (si->getStateSpace()->isMetricSpace()) {
      Tx0 = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    } else {
      Tx0 = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    }
    Tx0->setDistanceFunction([si](const Motion *a, const Motion *b) {
      return si->distance(a->x0, b->x0);
    });

    // build kd-tree for Txf
    ompl::NearestNeighbors<Motion *> *Txf;
    if (si->getStateSpace()->isMetricSpace()) {
      Txf = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    } else {
      Txf = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    }
    Txf->setDistanceFunction([si](const Motion *a, const Motion *b) {
      return si->distance(a->xf, b->xf);
    });

    // use as first/seed motion the one that moves furthest
    std::vector<size_t> used_motions;
    size_t best_motion = 0;
    double largest_d = 0;
    for (const auto &m : motions) {
      double d = si->distance(m.x0, m.xf);
      if (d > largest_d) {
        largest_d = d;
        best_motion = m.idx;
      }
    }
    used_motions.push_back(best_motion);
    Tx0->add(&motions[best_motion]);
    Txf->add(&motions[best_motion]);

    std::set<size_t> unused_motions;
    for (const auto &m : motions) {
      unused_motions.insert(m.idx);
    }
    unused_motions.erase(best_motion);
    // TODO: i should use a joint space!

    for (size_t k = 1; k < top_k; ++k) {
      std::cout << "sorting " << k << std::endl;
      double best_d = -1;
      size_t best_motion = -1;
      for (size_t m1 : unused_motions) {
        // find smallest distance to existing neighbors
        auto m2 = Tx0->nearest(&motions[m1]);
        double smallest_d_x0 = si->distance(motions[m1].x0, m2->x0);

        m2 = Txf->nearest(&motions[m1]);
        double smallest_d_xf = si->distance(motions[m1].xf, m2->xf);

        double smallest_d = smallest_d_x0 + smallest_d_xf;
        if (smallest_d > best_d) {
          best_motion = m1;
          best_d = smallest_d;
        }
      }
      used_motions.push_back(best_motion);
      unused_motions.erase(best_motion);
      Tx0->add(&motions[best_motion]);
      Txf->add(&motions[best_motion]);

      if (Tx0->size() && Tx0->size() % 200 == 0) {
        auto p_derived =
            static_cast<ompl::NearestNeighborsGNATNoThreadSafety<Motion *> *>(
                Tx0);
        p_derived->rebuildDataStructure();
        p_derived->rebuildSize_ = 1e8;
      }

      if (Txf->size() && Txf->size() % 200 == 0) {
        auto p_derived =
            static_cast<ompl::NearestNeighborsGNATNoThreadSafety<Motion *> *>(
                Txf);
        p_derived->rebuildDataStructure();
        p_derived->rebuildSize_ = 1e8;
      }
    }

    // clean-up memory
    for (const auto &m : motions) {
      si->freeState(m.x0);
      si->freeState(m.xf);
    }

#if 1
    std::cout << "checking against motions 3 " << std::endl;
    auto index_outs = sortMotions3(x0s, xfs, top_k);
    //

    CHECK_EQ(used_motions.size(), index_outs.size(), AT);

    for (size_t i = 0; i < used_motions.size(); i++) {
      CHECK_EQ(index_outs.at(i), used_motions.at(i), AT);
    }

#endif

    return used_motions;
  }

  std::vector<size_t> sortMotions3(const std::vector<std::vector<double>> &x0s,
                                   const std::vector<std::vector<double>> &xfs,
                                   size_t top_k) {
    assert(x0s.size() == xfs.size());
    assert(x0s.size() > 0);
    auto si = robot_->getSpaceInformation();
    struct Motion {
      ob::State *x0;
      ob::State *xf;
      size_t idx;
      double min_d_to_x0;
      double min_d_to_xf;
    };
    // create vector of motions
    std::vector<Motion> motions;
    for (size_t i = 0; i < x0s.size(); ++i) {
      Motion m;
      m.x0 = si->allocState();
      si->getStateSpace()->copyFromReals(m.x0, x0s[i]);
      si->enforceBounds(m.x0);
      m.xf = si->allocState();
      si->getStateSpace()->copyFromReals(m.xf, xfs[i]);
      si->enforceBounds(m.xf);
      m.idx = i;
      motions.push_back(m);
    }

    // build kd-tree for Tx0
    // ompl::NearestNeighbors<Motion *> *Tx0;
    // if (si->getStateSpace()->isMetricSpace()) {
    //   Tx0 = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    // } else {
    //   Tx0 = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    // }
    // Tx0->setDistanceFunction([si](const Motion *a, const Motion *b) {
    //   return si->distance(a->x0, b->x0);
    // });
    //
    // // build kd-tree for Txf
    // ompl::NearestNeighbors<Motion *> *Txf;
    // if (si->getStateSpace()->isMetricSpace()) {
    //   Txf = new ompl::NearestNeighborsGNATNoThreadSafety<Motion *>();
    // } else {
    //   Txf = new ompl::NearestNeighborsSqrtApprox<Motion *>();
    // }
    // Txf->setDistanceFunction([si](const Motion *a, const Motion *b) {
    //   return si->distance(a->xf, b->xf);
    // });

    // use as first/seed motion the one that moves furthest
    std::vector<size_t> used_motions;
    size_t best_motion = 0;
    double largest_d = 0;
    for (const auto &m : motions) {
      double d = si->distance(m.x0, m.xf);
      if (d > largest_d) {
        largest_d = d;
        best_motion = m.idx;
      }
    }

    // compute the original min distances

    auto xf_dist = [&](const Motion *a, const Motion *b) {
      return si->distance(a->xf, b->xf);
    };

    auto x0_dist = [&](const Motion *a, const Motion *b) {
      return si->distance(a->x0, b->x0);
    };

    used_motions.push_back(best_motion);
    // Tx0->add(&motions[best_motion]);
    // Txf->add(&motions[best_motion]);

    std::set<size_t> unused_motions;
    for (const auto &m : motions) {
      unused_motions.insert(m.idx);
    }
    unused_motions.erase(best_motion);

    for (auto &mi : unused_motions) {
      // unused_motions.insert(m.idx);
      auto &m = motions.at(mi);
      CHECK(used_motions.size(), AT);
      m.min_d_to_x0 = x0_dist(&m, &motions.at(used_motions.at(0)));
      m.min_d_to_xf = xf_dist(&m, &motions.at(used_motions.at(0)));
    }

    // TODO: i should use a joint space!

    for (size_t k = 1; k < top_k; ++k) {
      std::cout << "sorting " << k << std::endl;
      auto it = std::max_element(unused_motions.begin(), unused_motions.end(),
                                 [&](auto &a, auto &b) {
                                   auto ma = motions.at(a);
                                   auto mb = motions.at(b);
                                   return ma.min_d_to_x0 + ma.min_d_to_xf <
                                          mb.min_d_to_x0 + mb.min_d_to_xf;
                                 });

      // size_t index = std::distance(unused_motions.begin(), it);
      auto best_m = motions.at(*it);
      // unused_motions.at(index));
      used_motions.push_back(best_m.idx);
      unused_motions.erase(best_m.idx);

      // update
      std::for_each(
          unused_motions.begin(), unused_motions.end(), [&](auto &mi) {
            auto &m = motions.at(mi);
            m.min_d_to_x0 = std::min(m.min_d_to_x0, x0_dist(&m, &best_m));
            m.min_d_to_xf = std::min(m.min_d_to_xf, xf_dist(&m, &best_m));
          });
    }

    // clean-up memory
    for (const auto &m : motions) {
      si->freeState(m.x0);
      si->freeState(m.xf);
    }

    return used_motions;
  }

private:
  std::shared_ptr<RobotOmpl> robot_;
  ob::StateSamplerPtr state_sampler_;
  oc::ControlSamplerPtr control_sampler_;
  ob::State *tmp_state_a_;
  ob::State *tmp_state_b_;
  oc::Control *tmp_control_;
};

PYBIND11_MODULE(motionplanningutils, m) {
  pybind11::class_<CollisionChecker>(m, "CollisionChecker")
      .def(pybind11::init())
      .def("load", &CollisionChecker::load)
      .def("distance", &CollisionChecker::distance)
      .def("distanceWithFDiffGradient",
           &CollisionChecker::distanceWithFDiffGradient);

  pybind11::class_<RobotHelper>(m, "RobotHelper")
      .def(pybind11::init<const std::string &, float>(), py::arg("robot_type"),
           py::arg("pos_limit") = 2)
      .def("distance", &RobotHelper::distance)
      .def("sampleUniform", &RobotHelper::sampleStateUniform)
      .def("sampleControlUniform", &RobotHelper::sampleControlUniform)
      .def("step", &RobotHelper::step)
      .def("interpolate", &RobotHelper::interpolate)
      .def("is2D", &RobotHelper::is2D)
      .def("isTranslationInvariant", &RobotHelper::isTranslationInvariant)
      .def("sortMotions", &RobotHelper::sortMotions3);

  pybind11::class_<Model_robot>(m, "Model_robot")
      .def(pybind11::init())
      .def("setPositionBounds", &Model_robot::setPositionBounds)
      .def("get_translation_invariance", &Model_robot::get_translation_invariance)
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
      .def("stepDiffdt", &Model_robot::stepDiffdt)
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
           &Model_robot::transformation_collision_geometries)
      .def("load_env_quim", &Model_robot::load_env_quim);

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
