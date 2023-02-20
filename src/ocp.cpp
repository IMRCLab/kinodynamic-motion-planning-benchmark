#include "ocp.hpp"
#include "croco_macros.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <boost/test/tools/interface.hpp>

Opti_params opti_params;
double accumulated_time;

using vstr = std::vector<std::string>;
using V2d = Eigen::Vector2d;
using V3d = Eigen::Vector3d;
using V4d = Eigen::Vector4d;
using Vxd = Eigen::VectorXd;

void check_input_calc(Eigen::Ref<Eigen::VectorXd> xnext,
                      const Eigen::Ref<const Vxd> &x,
                      const Eigen::Ref<const Vxd> &u, size_t nx, size_t nu) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }

  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  if (static_cast<std::size_t>(xnext.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "xnext has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
};

void normalize(const Eigen::Ref<const Eigen::Vector4d> &q,
               Eigen::Ref<Eigen::Vector4d> y, Eigen::Ref<Eigen::Matrix4d> J) {
  double norm = q.norm();
  y = q / norm;
  Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
  J.noalias() = I4 / norm - q * q.transpose() / (std::pow(norm, 3));
}

void rotate_with_q(const Eigen::Ref<const Eigen::Vector4d> &x,
                   const Eigen::Ref<const Eigen::Vector3d> &a,
                   Eigen::Ref<Eigen::Vector3d> y, Eigen::Ref<Matrix34> Jx,
                   Eigen::Ref<Eigen::Matrix3d> Ja) {

  Eigen::Vector4d q;
  Eigen::Matrix4d Jnorm;
  Matrix34 Jq;

  normalize(x, q, Jnorm);

  double w = q(3);
  Eigen::Vector3d v = q.block<3, 1>(0, 0);
  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  Eigen::Quaterniond quat = Eigen::Quaterniond(q);

  Eigen::Matrix3d R = quat.toRotationMatrix();
  // std::cout << "R\n" << R << std::endl;
  // std::cout << "a\n" << a << std::endl;

  y = R * a;
  // std::cout << "y\n" << y << std::endl;

  if (Jq.cols() && Ja.cols()) {
    assert(Jq.cols() == 4);
    assert(Jq.rows() == 3);

    assert(Ja.cols() == 3);
    assert(Ja.rows() == 3);

    Eigen::Vector3d Jq_1 = 2 * (w * a + v.cross(a));
    // std::cout << "Jq_1\n" << Jq_1 << std::endl;
    Eigen::Matrix3d Jq_2 = 2 * (v.dot(a) * I3 + v * a.transpose() -
                                a * v.transpose() - w * Skew(a));
    // std::cout << "Jq_2\n" << Jq_2 << std::endl;
    Jq.col(3) = Jq_1;
    Jq.block<3, 3>(0, 0) = Jq_2;
    // std::cout << "Jq_q\n" << Jq << std::endl;

    Jx.noalias() = Jq * Jnorm;

    // std::cout << "Jx\n" << Jx << std::endl;

    Ja = R;
  }
}

void check_input_calcdiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                          Eigen::Ref<Eigen::MatrixXd> Fu,
                          const Eigen::Ref<const Vxd> &x,
                          const Eigen::Ref<const Vxd> &u, size_t nx,
                          size_t nu) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  if (static_cast<std::size_t>(Fx.cols()) != nx) {
    throw_pretty("Invalid argument: "
                 << "Fx has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }

  if (static_cast<std::size_t>(Fx.rows()) != nx) {
    throw_pretty("Invalid argument: "
                 << "Fx has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }

  if (static_cast<std::size_t>(Fu.cols()) != nu) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }

  if (static_cast<std::size_t>(Fu.rows()) != nx) {
    throw_pretty("Invalid argument: "
                 << "Fu has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
}

void Opti_params::add_options(po::options_description &desc) {

  set_from_boostop(desc, VAR_WITH_NAME(collision_weight));
  set_from_boostop(desc, VAR_WITH_NAME(th_acceptnegstep));
  set_from_boostop(desc, VAR_WITH_NAME(states_reg));
  set_from_boostop(desc, VAR_WITH_NAME(init_reg));
  set_from_boostop(desc, VAR_WITH_NAME(control_bounds));
  set_from_boostop(desc, VAR_WITH_NAME(max_iter));
  set_from_boostop(desc, VAR_WITH_NAME(window_optimize));
  set_from_boostop(desc, VAR_WITH_NAME(window_shift));
  set_from_boostop(desc, VAR_WITH_NAME(solver_id));
  set_from_boostop(desc, VAR_WITH_NAME(use_warmstart));
  set_from_boostop(desc, VAR_WITH_NAME(use_finite_diff));
  set_from_boostop(desc, VAR_WITH_NAME(k_linear));
  set_from_boostop(desc, VAR_WITH_NAME(noise_level));
  set_from_boostop(desc, VAR_WITH_NAME(k_contour));
  set_from_boostop(desc, VAR_WITH_NAME(smooth_traj));
  set_from_boostop(desc, VAR_WITH_NAME(weight_goal));
  set_from_boostop(desc, VAR_WITH_NAME(shift_repeat));
  set_from_boostop(desc, VAR_WITH_NAME(solver_name));
  set_from_boostop(desc, VAR_WITH_NAME(tsearch_max_rate));
  set_from_boostop(desc, VAR_WITH_NAME(tsearch_min_rate));
  set_from_boostop(desc, VAR_WITH_NAME(tsearch_num_check));
}

void Opti_params::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void Opti_params::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(collision_weight));
  set_from_yaml(node, VAR_WITH_NAME(th_acceptnegstep));
  set_from_yaml(node, VAR_WITH_NAME(states_reg));
  set_from_yaml(node, VAR_WITH_NAME(init_reg));
  set_from_yaml(node, VAR_WITH_NAME(solver_name));
  set_from_yaml(node, VAR_WITH_NAME(solver_id));
  set_from_yaml(node, VAR_WITH_NAME(use_warmstart));
  set_from_yaml(node, VAR_WITH_NAME(control_bounds));
  set_from_yaml(node, VAR_WITH_NAME(k_linear));
  set_from_yaml(node, VAR_WITH_NAME(k_contour));
  set_from_yaml(node, VAR_WITH_NAME(max_iter));
  set_from_yaml(node, VAR_WITH_NAME(window_optimize));
  set_from_yaml(node, VAR_WITH_NAME(window_shift));
  set_from_yaml(node, VAR_WITH_NAME(max_mpc_iterations));
  set_from_yaml(node, VAR_WITH_NAME(debug_file_name));
  set_from_yaml(node, VAR_WITH_NAME(weight_goal));
  set_from_yaml(node, VAR_WITH_NAME(collision_weight));
  set_from_yaml(node, VAR_WITH_NAME(smooth_traj));
  set_from_yaml(node, VAR_WITH_NAME(shift_repeat));
  set_from_yaml(node, VAR_WITH_NAME(tsearch_max_rate));
  set_from_yaml(node, VAR_WITH_NAME(tsearch_min_rate));
  set_from_yaml(node, VAR_WITH_NAME(tsearch_num_check));
}

void Opti_params::print(std::ostream &out) {

  std::string be = "";
  std::string af = ": ";

  out << be << STR(th_acceptnegstep, af) << std::endl;
  out << be << STR(states_reg, af) << std::endl;
  out << be << STR(solver_name, af) << std::endl;
  out << be << STR(CALLBACKS, af) << std::endl;
  out << be << STR(solver_id, af) << std::endl;
  out << be << STR(use_finite_diff, af) << std::endl;
  out << be << STR(use_warmstart, af) << std::endl;
  out << be << STR(repair_init_guess, af) << std::endl;
  out << be << STR(control_bounds, af) << std::endl;
  out << be << STR(th_stop, af) << std::endl;
  out << be << STR(init_reg, af) << std::endl;
  out << be << STR(th_acceptnegstep, af) << std::endl;
  out << be << STR(noise_level, af) << std::endl;
  out << be << STR(max_iter, af) << std::endl;
  out << be << STR(window_optimize, af) << std::endl;
  out << be << STR(window_shift, af) << std::endl;
  out << be << STR(max_mpc_iterations, af) << std::endl;
  out << be << STR(debug_file_name, af) << std::endl;
  out << be << STR(k_linear, af) << std::endl;
  out << be << STR(k_contour, af) << std::endl;
  out << be << STR(weight_goal, af) << std::endl;
  out << be << STR(collision_weight, af) << std::endl;
  out << be << STR(smooth_traj, af) << std::endl;

  out << be << STR(tsearch_max_rate, af) << std::endl;
  out << be << STR(tsearch_min_rate, af) << std::endl;
  out << be << STR(tsearch_num_check, af) << std::endl;
}

const char *SOLVER_txt[] = {"traj_opt",
                            "traj_opt_free_time",
                            "traj_opt_smooth_then_free_time",
                            "mpc",
                            "mpcc",
                            "mpcc2",
                            "traj_opt_mpcc",
                            "mpc_nobound_mpcc",
                            "mpcc_linear",
                            "time_search_traj_opt",
                            "mpc_adaptative",
                            "traj_opt_free_time_proxi",
                            "none"};

void linearInterpolation(const Vxd &times, const std::vector<Vxd> &x,
                         double t_query, Eigen::Ref<Vxd> out,
                         Eigen::Ref<Vxd> Jx) {

  CHECK(x.size(), AT);
  CHECK_EQ(x.front().size(), out.size(), AT);

  double num_tolerance = 1e-8;
  // CHECK_GEQ(t_query + num_tolerance, times.head(1)(0), AT);
  assert(times.size() == x.size());

  size_t index = 0;
  if (t_query < times(0)) {
    std::cout << "WARNING: " << AT << std::endl;
    std::cout << "EXTRAPOLATION: " << t_query << " " << times(0) << "  "
              << t_query - times(0) << std::endl;
    index = 1;
  } else if (t_query >= times(times.size() - 1)) {
    std::cout << "WARNING: " << AT << std::endl;
    std::cout << "EXTRAPOLATION: " << t_query << " " << times(times.size() - 1)
              << "  " << t_query - times(times.size() - 1) << std::endl;
    index = times.size() - 1;
  } else {

    auto it = std::lower_bound(
        times.data(), times.data() + times.size(), t_query,
        [](const auto &it, const auto &value) { return it <= value; });

    index = std::distance(times.data(), it);

    const bool debug_bs = false;
    if (debug_bs) {
      size_t index2 = 0;
      for (size_t i = 0; i < times.size(); i++) {
        if (t_query < times(i)) {
          index2 = i;
          break;
        }
      }
      CHECK_EQ(index, index2, AT);
    }
  }

  double factor =
      (t_query - times(index - 1)) / (times(index) - times(index - 1));

  // std::cout << "index is " << index << std::endl;
  // std::cout << "size " << times.size() << std::endl;
  // std::cout << "factor " << factor << std::endl;

  out = x.at(index - 1) + factor * (x.at(index) - x.at(index - 1));
  Jx = (x.at(index) - x.at(index - 1)) / (times(index) - times(index - 1));
}

Dynamics_contour::Dynamics_contour(ptr<Dynamics> dyn, bool accumulate)
    : dyn(dyn), accumulate(accumulate) {
  CHECK(accumulate, AT);
  nx = dyn->nx + 1;
  nu = dyn->nu + 1;
}

void Dynamics_contour::calc(Eigen::Ref<Vxd> xnext,
                            const Eigen::Ref<const VectorXs> &x,
                            const Eigen::Ref<const VectorXs> &u) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  dyn->calc(xnext.head(dyn->nx), x.head(dyn->nx), u.head(dyn->nu));

  if (accumulate)
    xnext(nx - 1) = x(nx - 1) + u(nu - 1); // linear model
  else
    xnext(nx - 1) = u(nu - 1); // linear model
};

void Dynamics_contour::calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                Eigen::Ref<Eigen::MatrixXd> Fu,
                                const Eigen::Ref<const VectorXs> &x,
                                const Eigen::Ref<const VectorXs> &u) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  dyn->calcDiff(Fx.block(0, 0, dyn->nx, dyn->nx),
                Fu.block(0, 0, dyn->nx, dyn->nu), x.head(dyn->nx),
                u.head(dyn->nu));

  if (accumulate) {
    Fx(nx - 1, nx - 1) = 1.0;
    Fu(nx - 1, nu - 1) = 1.0;
  } else {
    Fu(nx - 1, nu - 1) = 1.0;
  }
}

Dynamics_unicycle2::Dynamics_unicycle2(bool free_time) : free_time(free_time) {
  nx = 5;
  nu = 2;
  if (free_time)
    nu += 1;

  uref = Vxd::Zero(nx);
}

void Dynamics_unicycle2::calc(Eigen::Ref<Vxd> xnext,
                              const Eigen::Ref<const VectorXs> &x,
                              const Eigen::Ref<const VectorXs> &u) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  const double xx = x[0];
  const double y = x[1];
  const double yaw = x[2];
  const double v = x[3];
  const double w = x[4];

  const double c = cos(yaw);
  const double s = sin(yaw);

  const double a = u[0];
  const double w_dot = u[1];

  double dt_ = dt;
  if (free_time)
    dt_ *= u[2];

  const double v_next = v + a * dt_;
  const double w_next = w + w_dot * dt_;
  const double yaw_next = yaw + w * dt_;
  const double x_next = xx + v * c * dt_;
  const double y_next = y + v * s * dt_;

  xnext << x_next, y_next, yaw_next, v_next, w_next;
};

void Dynamics_unicycle2::calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                  Eigen::Ref<Eigen::MatrixXd> Fu,
                                  const Eigen::Ref<const VectorXs> &x,
                                  const Eigen::Ref<const VectorXs> &u) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  const double c = cos(x[2]);
  const double s = sin(x[2]);

  const double v = x[3];
  Fx.setZero();
  Fu.setZero();

  double dt_ = dt;
  if (free_time)
    dt_ *= u[2];

  Fx(0, 0) = 1;
  Fx(1, 1) = 1;
  Fx(2, 2) = 1;
  Fx(3, 3) = 1;
  Fx(4, 4) = 1;

  Fx(0, 2) = -s * v * dt_;
  Fx(1, 2) = c * v * dt_;

  Fx(0, 3) = c * dt_;
  Fx(1, 3) = s * dt_;
  Fx(2, 4) = dt_;

  Fu(3, 0) = dt_;
  Fu(4, 1) = dt_;

  if (free_time) {

    const double a = u[0];
    const double w_dot = u[1];
    const double w = x[4];

    Fu(0, 2) = v * c * dt;
    Fu(1, 2) = v * s * dt;
    Fu(2, 2) = w * dt;
    Fu(3, 2) = a * dt;
    Fu(4, 2) = w_dot * dt;
  }
}

Dynamics_unicycle::Dynamics_unicycle(bool free_time) : free_time(free_time) {
  nx = 3;
  if (free_time)
    nu = 3;
  else
    nu = 2;
  uref = Vxd::Zero(2);
}

void Dynamics_unicycle::calc(Eigen::Ref<Vxd> xnext,
                             const Eigen::Ref<const VectorXs> &x,
                             const Eigen::Ref<const VectorXs> &u) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  const double c = cos(x[2]);
  const double s = sin(x[2]);

  double dt_ = dt;
  if (free_time)
    dt_ *= u[2];

  xnext << x[0] + c * u[0] * dt_, x[1] + s * u[0] * dt_, x[2] + u[1] * dt_;
};

void Dynamics_unicycle::calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                 Eigen::Ref<Eigen::MatrixXd> Fu,
                                 const Eigen::Ref<const VectorXs> &x,
                                 const Eigen::Ref<const VectorXs> &u) {

  if (static_cast<std::size_t>(x.size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " +
                        std::to_string(nx) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu) + ")");
  }

  const double c = cos(x[2]);
  const double s = sin(x[2]);

  double dt_ = dt;
  if (free_time)
    dt_ *= u[2];

  Fx.setZero();
  Fu.setZero();

  Fx(0, 0) = 1;
  Fx(1, 1) = 1;
  Fx(2, 2) = 1;
  Fx(0, 2) = -s * u[0] * dt_;
  Fx(1, 2) = c * u[0] * dt_;
  Fu(0, 0) = c * dt_;
  Fu(1, 0) = s * dt_;
  Fu(2, 1) = dt_;

  if (free_time) {
    Fu(0, 2) = c * u[0] * dt;
    Fu(1, 2) = s * u[0] * dt;
    Fu(2, 2) = u[1] * dt;
  }
}

Contour_cost_alpha_x::Contour_cost_alpha_x(size_t nx, size_t nu)
    : Cost(nx, nu, 1) {
  name = "contour-cost-alpha-x";
  cost_type = CostTYPE::linear;
}

void Contour_cost_alpha_x::calc(Eigen::Ref<Vxd> r,
                                const Eigen::Ref<const Vxd> &x,
                                const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);
  assert(k > 0);

  r(0) = -k * x(nx - 1);
}

void Contour_cost_alpha_x::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                                    Eigen::Ref<Eigen::MatrixXd> Ju,
                                    const Eigen::Ref<const Vxd> &x,
                                    const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  Jx(0, nx - 1) = -k;
}

Contour_cost_alpha_u::Contour_cost_alpha_u(size_t nx, size_t nu)
    : Cost(nx, nu, 1) {
  name = "contour-cost-alpha-u";
  cost_type = CostTYPE::linear;
}

void Contour_cost_alpha_u::calc(Eigen::Ref<Vxd> r,
                                const Eigen::Ref<const Vxd> &x,
                                const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);
  assert(k > 0);

  r(0) = -k * u(nu - 1);
}

void Contour_cost_alpha_u::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                                    Eigen::Ref<Eigen::MatrixXd> Ju,
                                    const Eigen::Ref<const Vxd> &x,
                                    const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  Ju(0, nu - 1) = -k;
};

void finite_diff_cost(ptr<Cost> cost, Eigen::Ref<Eigen::MatrixXd> Jx,
                      Eigen::Ref<Eigen::MatrixXd> Ju, const Vxd &x,
                      const Vxd &u, const int nr) {

  Vxd r_ref(nr);
  cost->calc(r_ref, x, u);
  int nu = u.size();
  int nx = x.size();

  Ju.setZero();

  double eps = 1e-5;
  for (size_t i = 0; i < nu; i++) {
    Eigen::MatrixXd ue;
    ue = u;
    ue(i) += eps;
    Vxd r_e(nr);
    r_e.setZero();
    cost->calc(r_e, x, ue);
    auto df = (r_e - r_ref) / eps;
    Ju.col(i) = df;
  }

  Jx.setZero();
  for (size_t i = 0; i < nx; i++) {
    Eigen::MatrixXd xe;
    xe = x;
    xe(i) += eps;
    Vxd r_e(nr);
    r_e.setZero();
    cost->calc(r_e, xe, u);
    auto df = (r_e - r_ref) / eps;
    Jx.col(i) = df;
  }
}

Contour_cost_x::Contour_cost_x(size_t nx, size_t nu, ptr<Interpolator> path)
    : Cost(nx, nu, nx - 1), path(path), last_query(-1.), last_out(nx - 1),
      last_J(nx - 1) {
  name = "contour-cost-x";
}
void Contour_cost_x::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                          const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);

  double alpha = x(nx - 1);

  last_query = alpha;
  assert(path);
  assert(weight > 0);
  path->interpolate(alpha, last_out, last_J);

  r = weight * (last_out - x.head(nx - 1));
}

void Contour_cost_x::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {
  calc(r, x, zero_u);
}

void Contour_cost_x::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                              Eigen::Ref<Eigen::MatrixXd> Ju,
                              const Eigen::Ref<const Vxd> &x,
                              const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  // lets use finite diff
  if (use_finite_diff) {
  }

  else {
    double alpha = x(nx - 1);
    if (std::fabs(alpha - last_query) > 1e-8) {
      last_query = alpha;
      assert(path);
      path->interpolate(alpha, last_out, last_J);
    }

    assert(weight > 0);

    Jx.block(0, 0, nx - 1, nx - 1).diagonal() = -weight * Vxd::Ones(nx - 1);
    Jx.block(0, nx - 1, nx - 1, 1) = weight * last_J;
  }
};

void Contour_cost_x::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                              const Eigen::Ref<const Vxd> &x) {

  calcDiff(Jx, zero_Ju, x, zero_u);
}

Contour_cost::Contour_cost(size_t nx, size_t nu, ptr<Interpolator> path)
    : Cost(nx, nu, nx + 1), path(path), last_query(-1.), last_out(nx - 1),
      last_J(nx - 1) {
  name = "contour";
}

void Contour_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                        const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);

  double alpha = x(nx - 1);

  last_query = alpha;
  assert(path);
  path->interpolate(alpha, last_out, last_J);

  r.head(nx - 1) = weight_contour * weight_diff * (last_out - x.head(nx - 1));
  r(nx - 1) = weight_contour * weight_alpha * (alpha - ref_alpha);
  r(nx) = weight_virtual_control * u(nu - 1);
  // std::cout << "calc in contour " << r.transpose() << std::endl;
}

void Contour_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {
  calc(r, x, zero_u);
}

void Contour_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            Eigen::Ref<Eigen::MatrixXd> Ju,
                            const Eigen::Ref<const Vxd> &x,
                            const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);

  // lets use finite diff
  if (use_finite_diff) {
    Vxd r_ref(nr);
    calc(r_ref, x, u);

    Ju.setZero();

    double eps = 1e-5;
    for (size_t i = 0; i < nu; i++) {
      Eigen::MatrixXd ue;
      ue = u;
      ue(i) += eps;
      Vxd r_e(nr);
      r_e.setZero();
      calc(r_e, x, ue);
      auto df = (r_e - r_ref) / eps;
      Ju.col(i) = df;
    }

    Jx.setZero();
    for (size_t i = 0; i < nx; i++) {
      Eigen::MatrixXd xe;
      xe = x;
      xe(i) += eps;
      Vxd r_e(nr);
      r_e.setZero();
      calc(r_e, xe, u);
      auto df = (r_e - r_ref) / eps;
      Jx.col(i) = df;
    }
  }

  else {
    double alpha = x(nx - 1);
    if (std::fabs(alpha - last_query) > 1e-8) {
      last_query = alpha;
      assert(path);
      path->interpolate(alpha, last_out, last_J);
    }

    Jx.block(0, 0, nx - 1, nx - 1).diagonal() =
        -weight_contour * weight_diff * Vxd::Ones(nx - 1);
    Jx.block(0, nx - 1, nx - 1, 1) = weight_contour * weight_diff * last_J;
    Jx(nx - 1, nx - 1) = weight_contour * weight_alpha;
    Ju(nx, nu - 1) = weight_virtual_control;
  }
};

void Contour_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            const Eigen::Ref<const Vxd> &x) {

  calcDiff(Jx, zero_Ju, x, zero_u);
}

Col_cost::Col_cost(size_t nx, size_t nu, size_t nr,
                   boost::shared_ptr<CollisionChecker> cl)
    : Cost(nx, nu, nr), cl(cl) {
  last_x = Vxd::Zero(nx);
  name = "collision";
  nx_effective = nx;
}

void Col_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                    const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  // std::vector<double> query{x.data(), x.data() + x.size()};
  std::vector<double> query{x.data(), x.data() + nx_effective};

  double raw_d;

  bool check_one = (x - last_x).squaredNorm() < 1e-8;
  bool check_two = (last_raw_d - margin) > 0 &&
                   (x - last_x).norm() < sec_factor * (last_raw_d - margin);

  // std::cout << "checking collisions " << std::endl;

  if (check_one || check_two) {
    raw_d = last_raw_d;
  } else {
    raw_d = std::get<0>(cl->distance(query));
    last_x = x;
    last_raw_d = raw_d;
  }
  double d = opti_params.collision_weight * (raw_d - margin);
  auto out = Eigen::Matrix<double, 1, 1>(std::min(d, 0.));
  r = out;
}

void Col_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {
  calc(r, x, Vxd(nu));
}

void Col_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Vxd> &x,
                        const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  std::vector<double> query{x.data(), x.data() + nx_effective};
  double raw_d, d;
  Vxd v(nx);
  bool check_one =
      (x - last_x).squaredNorm() < 1e-8 && (last_raw_d - margin) > 0;
  bool check_two = (last_raw_d - margin) > 0 &&
                   (x - last_x).norm() < sec_factor * (last_raw_d - margin);

  if (check_one || check_two) {
    Jx.setZero();
  } else {
    auto out = cl->distanceWithFDiffGradient(
        query, faraway_zero_gradient_bound, epsilon,
        non_zero_flags.size() ? &non_zero_flags : nullptr);
    raw_d = std::get<0>(out);
    last_x = x;
    last_raw_d = raw_d;
    d = opti_params.collision_weight * (raw_d - margin);
    auto grad = std::get<1>(out);
    v = opti_params.collision_weight * Vxd::Map(grad.data(), grad.size());
    if (d <= 0) {
      Jx.block(0, 0, 1, nx_effective) = v.transpose();
    } else {
      Jx.setZero();
    }
  }
  Ju.setZero();
};

void Col_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Vxd> &x) {

  auto Ju = Eigen::MatrixXd(1, 1);
  auto u = Vxd(1);
  calcDiff(Jx, Ju, x, u);
}

Control_cost::Control_cost(size_t nx, size_t nu, size_t nr, const Vxd &u_weight,
                           const Vxd &u_ref)
    : Cost(nx, nu, nr), u_weight(u_weight), u_ref(u_ref) {
  CHECK_EQ(u_weight.size(), nu, AT);
  CHECK_EQ(u_ref.size(), nu, AT);
  CHECK_EQ(nu, nr, AT);
  name = "control";
}

void Control_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                        const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);
  r = (u - u_ref).cwiseProduct(u_weight);
}

void Control_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {

  auto u = Vxd::Zero(nu);
  calc(r, x, u);
}

void Control_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            Eigen::Ref<Eigen::MatrixXd> Ju,
                            const Eigen::Ref<const Vxd> &x,
                            const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);
  Ju = u_weight.asDiagonal();
  Jx.setZero();
}

void Control_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            const Eigen::Ref<const Vxd> &x) {

  Vxd u(0);
  Eigen::MatrixXd Ju(0, 0);
  calcDiff(Jx, Ju, x, u);
}

// weight * ( x - ub ) <= 0
//
// NOTE:
// you can implement lower bounds
// weight = -w
// ub = lb
// DEMO:
// -w ( x - lb ) <= 0
// w ( x - lb ) >= 0
// w x >= w lb

State_bounds::State_bounds(size_t nx, size_t nu, size_t nr, const Vxd &ub,
                           const Vxd &weight)
    : Cost(nx, nu, nr), ub(ub), weight(weight) {
  name = "xbound";
  CHECK_EQ(weight.size(), ub.size(), AT);
  CHECK_EQ(nx, nr, AT);
  CHECK_EQ(weight.size(), nx, AT);
}

void State_bounds::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                        const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  calc(r, x);
}

void State_bounds::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  r = ((x - ub).cwiseProduct(weight)).cwiseMax(0.);
}

void State_bounds::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            Eigen::Ref<Eigen::MatrixXd> Ju,
                            const Eigen::Ref<const Vxd> &x,
                            const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  calcDiff(Jx, x);
}

void State_bounds::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            const Eigen::Ref<const Vxd> &x) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);

  Eigen::Matrix<bool, Eigen::Dynamic, 1> result =
      (x - ub).cwiseProduct(weight).array() >= 0;
  // std::cout << " x " << x.format(FMT) << std::endl;
  // std::cout << " ub " << ub.format(FMT) << std::endl;
  // std::cout << " result " << result.cast<double>().format(FMT) << std::endl;
  Jx.diagonal() = (result.cast<double>()).cwiseProduct(weight);
}

State_cost::State_cost(size_t nx, size_t nu, size_t nr, const Vxd &x_weight,
                       const Vxd &ref)
    : Cost(nx, nu, nr), x_weight(x_weight), ref(ref) {
  name = "state";
  assert(static_cast<std::size_t>(x_weight.size()) == nx);
  assert(static_cast<std::size_t>(ref.size()) == nx);
}

void State_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                      const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  r = (x - ref).cwiseProduct(x_weight);
}

void State_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  r = (x - ref).cwiseProduct(x_weight);
}

void State_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                          Eigen::Ref<Eigen::MatrixXd> Ju,
                          const Eigen::Ref<const Vxd> &x,
                          const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  Jx = x_weight.asDiagonal();
  Ju.setZero();
}

void State_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                          const Eigen::Ref<const Vxd> &x) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);

  Jx = x_weight.asDiagonal();
}

All_cost::All_cost(size_t nx, size_t nu, size_t nr,
                   const std::vector<boost::shared_ptr<Cost>> &costs)
    : Cost(nx, nu, nr), costs(costs) {}

void All_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x,
                    const Eigen::Ref<const Vxd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  int index = 0;
  for (size_t i = 0; i < costs.size(); i++) {
    // fill the matrix
    auto &feat = costs.at(i);
    int _nr = feat->nr;
    feat->calc(r.segment(index, _nr), x, u);
    index += _nr;
  }
}

void All_cost::calc(Eigen::Ref<Vxd> r, const Eigen::Ref<const Vxd> &x) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);

  int index = 0;
  for (size_t i = 0; i < costs.size(); i++) {
    auto &feat = costs.at(i);
    int _nr = feat->nr;
    feat->calc(r.segment(index, _nr), x);
    index += _nr;
  }
}

void All_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Vxd> &x,
                        const Eigen::Ref<const Vxd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  // TODO: I shoudl only give the residuals...
  int index = 0;
  for (size_t i = 0; i < costs.size(); i++) {
    auto &feat = costs.at(i);
    int _nr = feat->nr;
    feat->calcDiff(Jx.block(index, 0, _nr, nx), Ju.block(index, 0, _nr, nu), x,
                   u);
    index += _nr;
  }
}

void All_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Vxd> &x) {

  assert(static_cast<std::size_t>(x.size()) == nx);

  // TODO: I shoudl only give the residuals...
  int index = 0;
  for (size_t i = 0; i < costs.size(); i++) {
    auto &feat = costs.at(i);
    int _nr = costs.at(i)->nr;
    feat->calcDiff(Jx.block(index, 0, _nr, nx), x);
    index += _nr;
  }
}

size_t
get_total_num_features(const std::vector<boost::shared_ptr<Cost>> &features) {

  return std::accumulate(features.begin(), features.end(), 0,
                         [](auto &a, auto &b) { return a + b->nr; });
}

ActionModelQ::ActionModelQ(ptr<Dynamics> dynamics,
                           const std::vector<ptr<Cost>> &features)
    : Base(boost::make_shared<StateVectorTpl<Scalar>>(dynamics->nx),
           dynamics->nu, get_total_num_features(features)),
      dynamics(dynamics), features(features), nx(dynamics->nx),
      nu(dynamics->nu), nr(get_total_num_features(features)), Jx(nr, nx),
      Ju(nr, nu) {
  Jx.setZero();
  Ju.setZero();
}

ActionModelQ::~ActionModelQ(){};

void ActionModelQ::calc(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) {
  Data *d = static_cast<Data *>(data.get());
  d->xnext.setZero();
  dynamics->calc(d->xnext, x, u);

  int index = 0;

  d->cost = 0;
  d->r.setZero();

  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;
    feat->calc(d->r.segment(index, _nr), x, u);

    if (feat->cost_type == CostTYPE::least_squares) {
      d->cost +=
          Scalar(0.5) * d->r.segment(index, _nr).dot(d->r.segment(index, _nr));
    } else if (feat->cost_type == CostTYPE::linear) {
      d->cost += d->r.segment(index, _nr).sum();
    }
    index += _nr;
  }
}

void ActionModelQ::calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                            const Eigen::Ref<const VectorXs> &x,
                            const Eigen::Ref<const VectorXs> &u) {

  Data *d = static_cast<Data *>(data.get());
  d->Fx.setZero();
  d->Fu.setZero();
  dynamics->calcDiff(d->Fx, d->Fu, x, u);

  // create a matrix for the Jacobians
  d->Lx.setZero();
  d->Lu.setZero();
  d->Lxx.setZero();
  d->Luu.setZero();
  d->Lxu.setZero();

  Jx.setZero();
  Ju.setZero();
  size_t index = 0;
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;
    // std::cout << feat->get_name() << std::endl;

    Eigen::Ref<Eigen::VectorXd> r = d->r.segment(index, _nr);
    Eigen::Ref<Eigen::MatrixXd> jx = Jx.block(index, 0, _nr, nx);
    Eigen::Ref<Eigen::MatrixXd> ju = Ju.block(index, 0, _nr, nu);

    feat->calcDiff(jx, ju, x, u);
    if (feat->cost_type == CostTYPE::least_squares) {
      d->Lx.noalias() += r.transpose() * jx;
      d->Lu.noalias() += r.transpose() * ju;
      d->Lxx.noalias() += jx.transpose() * jx;
      d->Luu.noalias() += ju.transpose() * ju;
      d->Lxu.noalias() += jx.transpose() * ju;
    } else if (feat->cost_type == CostTYPE::linear) {
      d->Lx.noalias() += jx.colwise().sum();
      d->Lu.noalias() += ju.colwise().sum();
    }
    index += _nr;
  }
}

void ActionModelQ::calc(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());
  d->r.setZero();

  int index = 0;

  d->cost = 0;
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;
    Eigen::Ref<Eigen::VectorXd> r = d->r.segment(index, _nr);
    feat->calc(r, x);
    if (feat->cost_type == CostTYPE::least_squares) {
      d->cost += Scalar(0.5) * r.dot(r);
    } else if (feat->cost_type == CostTYPE::linear) {
      d->cost += r.sum();
    }
    index += _nr;
  }
}

void ActionModelQ::calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                            const Eigen::Ref<const VectorXs> &x) {

  Data *d = static_cast<Data *>(data.get());

  d->Lx.setZero();
  d->Lxx.setZero();

  size_t index = 0;
  Jx.setZero();
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;

    Eigen::Ref<Vxd> r = d->r.segment(index, _nr);
    Eigen::Ref<Eigen::MatrixXd> jx = Jx.block(index, 0, _nr, nx);

    feat->calcDiff(jx, x);

    if (feat->cost_type == CostTYPE::least_squares) {
      data->Lx.noalias() += r.transpose() * jx;
      data->Lxx.noalias() += jx.transpose() * jx;
    } else if (feat->cost_type == CostTYPE::linear) {
      data->Lx.noalias() += jx.colwise().sum();
    }
    index += _nr;
  }
}

boost::shared_ptr<ActionDataAbstract> ActionModelQ::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}
bool ActionModelQ::checkData(
    const boost::shared_ptr<ActionDataAbstract> &data) {

  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

void ActionModelQ::print(std::ostream &os) const {
  os << "wrapper" << std::endl;
}

void PrintVariableMap(const boost::program_options::variables_map &vm,
                      std::ostream &out) {
  for (po::variables_map::const_iterator it = vm.cbegin(); it != vm.cend();
       it++) {
    out << "> " << it->first;
    if (((boost::any)it->second.value()).empty()) {
      out << "(empty)";
    }
    if (vm[it->first].defaulted() || it->second.defaulted()) {
      out << "(default)";
    }
    out << "=";

    bool is_char;
    try {
      boost::any_cast<const char *>(it->second.value());
      is_char = true;
    } catch (const boost::bad_any_cast &) {
      is_char = false;
    }
    bool is_str;
    try {
      boost::any_cast<std::string>(it->second.value());
      is_str = true;
    } catch (const boost::bad_any_cast &) {
      is_str = false;
    }

    auto &type = ((boost::any)it->second.value()).type();

    if (type == typeid(int)) {
      out << vm[it->first].as<int>() << std::endl;
    } else if (type == typeid(size_t)) {
      out << vm[it->first].as<size_t>() << std::endl;
    } else if (type == typeid(bool)) {
      out << vm[it->first].as<bool>() << std::endl;
    } else if (type == typeid(double)) {
      out << vm[it->first].as<double>() << std::endl;
    } else if (is_char) {
      out << vm[it->first].as<const char *>() << std::endl;
    } else if (is_str) {
      std::string temp = vm[it->first].as<std::string>();
      if (temp.size()) {
        out << temp << std::endl;
      } else {
        out << "true" << std::endl;
      }
    } else { // Assumes that the only remainder is vector<string>
      try {
        std::vector<std::string> vect =
            vm[it->first].as<std::vector<std::string>>();
        uint i = 0;
        for (std::vector<std::string>::iterator oit = vect.begin();
             oit != vect.end(); oit++, ++i) {
          out << "\r> " << it->first << "[" << i << "]=" << (*oit) << std::endl;
        }
      } catch (const boost::bad_any_cast &) {
        out << "UnknownType(" << ((boost::any)it->second.value()).type().name()
            << ")" << std::endl;
        assert(false);
      }
    }
  }
};

void print_data(boost::shared_ptr<ActionDataAbstractTpl<double>> data) {
  std::cout << "***\n";
  std::cout << "xnext\n" << data->xnext << std::endl;
  std::cout << "Fx:\n" << data->Fx << std::endl;
  std::cout << "Fu:\n" << data->Fu << std::endl;
  std::cout << "r:" << data->r.transpose() << std::endl;
  std::cout << "cost:" << data->cost << std::endl;
  std::cout << "Lx:" << data->Lx.transpose() << std::endl;
  std::cout << "Lu:" << data->Lu.transpose() << std::endl;
  std::cout << "Lxx:\n" << data->Lxx << std::endl;
  std::cout << "Luu:\n" << data->Luu << std::endl;
  std::cout << "Lxu:\n" << data->Lxu << std::endl;
  std::cout << "***\n";
}

void check_dyn(boost::shared_ptr<Dynamics> dyn, double eps) {

  int nx = dyn->nx;
  int nu = dyn->nu;

  Eigen::MatrixXd Fx(nx, nx);
  Eigen::MatrixXd Fu(nx, nu);
  Fx.setZero();
  Fu.setZero();

  Vxd x(nx);
  Vxd u(nu);
  x.setRandom();
  // x.segment(3,4).normalize();
  // x.segment(3, 4) << 0, 0, 0, 1;
  u.setRandom();

  dyn->calcDiff(Fx, Fu, x, u);

  // compute the same using finite diff

  Eigen::MatrixXd FxD(nx, nx);
  FxD.setZero();
  Eigen::MatrixXd FuD(nx, nu);
  FuD.setZero();

  Vxd xnext(nx);
  xnext.setZero();
  dyn->calc(xnext, x, u);
  std::cout << "xnext\n" << std::endl;
  std::cout << xnext.format(FMT) << std::endl;
  for (size_t i = 0; i < nx; i++) {
    Eigen::MatrixXd xe;
    xe = x;
    xe(i) += eps;
    Vxd xnexte(nx);
    xnexte.setZero();
    dyn->calc(xnexte, xe, u);
    auto df = (xnexte - xnext) / eps;
    FxD.col(i) = df;
  }

  for (size_t i = 0; i < nu; i++) {
    Eigen::MatrixXd ue;
    ue = u;
    ue(i) += eps;
    Vxd xnexte(nx);
    xnexte.setZero();
    dyn->calc(xnexte, x, ue);
    auto df = (xnexte - xnext) / eps;
    FuD.col(i) = df;
  }
  bool verbose = false;
  if (verbose) {
    std::cout << "Analytical " << std::endl;
    std::cout << "Fx\n" << Fx << std::endl;
    std::cout << "Fu\n" << Fu << std::endl;
    std::cout << "Finite Diff " << std::endl;
    std::cout << "Fx\n" << FxD << std::endl;
    std::cout << "Fu\n" << FuD << std::endl;
  }

  bool check1 = (Fx - FxD).cwiseAbs().maxCoeff() < 10 * eps;
  bool check2 = (Fu - FuD).cwiseAbs().maxCoeff() < 10 * eps;

  std::cout << "Fx\n" << std::endl;
  std::cout << Fx << std::endl;
  std::cout << "Fu\n" << std::endl;
  std::cout << Fu << std::endl;
  if (!check1) {
    std::cout << "Fx" << std::endl;
    std::cout << Fx << std::endl;
    std::cout << "FxD" << std::endl;
    std::cout << FxD << std::endl;
    std::cout << "Fx - FxD" << std::endl;
    std::cout << Fx - FxD << std::endl;
    CHECK(((Fx - FxD).cwiseAbs().maxCoeff() < 10 * eps), AT);
  }

  if (!check2) {
    std::cout << "Fu" << std::endl;
    std::cout << Fu << std::endl;
    std::cout << "FuD" << std::endl;
    std::cout << FuD << std::endl;
    std::cout << "Fu - FuD" << std::endl;
    std::cout << Fu - FuD << std::endl;
    CHECK(((Fu - FuD).cwiseAbs().maxCoeff() < 10 * eps), AT);
  }
}

void Generate_params::print(std::ostream &out) const {
  auto pre = "";
  auto after = ": ";
  out << pre << STR(collisions, after) << std::endl;
  out << pre << STR(free_time, after) << std::endl;
  out << pre << STR(name, after) << std::endl;
  out << pre << STR(N, after) << std::endl;
  out << pre << STR(cl, after) << std::endl;
  out << pre << STR(contour_control, after) << std::endl;
  out << pre << STR(max_alpha, after) << std::endl;
  out << STR(goal_cost, after) << std::endl;

  out << pre << "goal" << after << goal.transpose() << std::endl;
  out << pre << "start" << after << start.transpose() << std::endl;
  out << pre << "states" << std::endl;
  for (const auto &s : states)
    out << "  - " << s.format(FMT) << std::endl;
  out << pre << "states_weights" << std::endl;
  for (const auto &s : states_weights)
    out << "  - " << s.format(FMT) << std::endl;
  out << pre << "actions" << std::endl;
  for (const auto &s : actions)
    out << "  - " << s.format(FMT) << std::endl;
}

double max_rollout_error(ptr<Dynamics> dyn, const std::vector<Vxd> &xs,
                         const std::vector<Vxd> &us) {

  assert(xs.size() == us.size() + 1);

  size_t N = us.size();

  size_t nx = xs.front().size();

  Vxd xnext(nx);
  double max_error = 0;

  for (size_t i = 0; i < N; i++) {

    dyn->calc(xnext, xs.at(i), us.at(i));
    double d = (xnext - xs.at(i + 1)).norm();

    if (d > max_error) {
      max_error = d;
    }
  }
  return max_error;
}

bool check_feas(ptr<Col_cost> feat_col, const std::vector<Vxd> &xs,
                const std::vector<Vxd> &us, const Vxd &goal) {

  double accumulated_c = 0;
  double max_c = 0;
  if (feat_col->cl) {
    for (auto &x : xs) {
      Vxd out(1);
      feat_col->calc(out, x);
      accumulated_c += std::abs(out(0));

      if (std::abs(out(0)) > max_c) {
        max_c = std::abs(out(0));
      }
    }
  }
  double dist_to_goal = (xs.back() - goal).norm();

  std::cout << "CROCO DONE " << std::endl;
  std::cout << "accumulated_c is " << accumulated_c << std::endl;
  std::cout << "max_c is " << max_c << std::endl;
  std::cout << "distance to goal " << dist_to_goal << std::endl;

  double threshold_feas = .1;

  // Dist to goal: 1 cm is OK
  // Accumulated_c: 1 cm  ACCUM is OK
  bool feasible =
      10 * (accumulated_c / opti_params.collision_weight) + 10 * dist_to_goal <
      threshold_feas;
  return feasible;
};

void modify_x_bound_for_contour(const Vxd &__x_lb, const Vxd &__x_ub,
                                const Vxd &__xb__weight, Eigen::Ref<Vxd> x_lb,
                                Eigen::Ref<Vxd> x_ub, Eigen::Ref<Vxd> xb_weight,
                                double max_alpha) {

  CHECK_EQ(__x_lb.size(), __x_ub.size(), AT);
  CHECK_EQ(__xb__weight.size(), __x_ub.size(), AT);

  size_t nx = __x_lb.size() + 1;

  xb_weight = Vxd(nx);
  x_lb = Vxd(nx);
  x_ub = Vxd(nx);

  x_lb << __x_lb, -10.;
  x_ub << __x_ub, max_alpha;
  xb_weight << __xb__weight, 200.;
}

void modify_u_bound_for_contour(const Vxd &__u_lb, const Vxd &__u_ub,
                                const Vxd &__u__weight, const Vxd &__u__ref,
                                Eigen::Ref<Vxd> u_lb, Eigen::Ref<Vxd> u_ub,
                                Eigen::Ref<Vxd> u_weight,
                                Eigen::Ref<Vxd> u_ref) {

  CHECK_EQ(__u_lb.size(), __u_ub.size(), AT);
  CHECK_EQ(__u__weight.size(), __u_ub.size(), AT);
  CHECK_EQ(__u__ref.size(), __u_ub.size(), AT);

  size_t nu = __u_lb.size() + 1;

  u_weight = Vxd(nu);
  u_lb = Vxd(nu);
  u_ub = Vxd(nu);
  u_ref = Vxd(nu);

  u_lb << __u_lb, -10.;
  u_ub << __u_ub, 10.;
  u_ref << __u__ref, 0.;
  u_weight << __u__weight, .1;
}

void modify_u_bound_for_free_time(const Vxd &__u_lb, const Vxd &__u_ub,
                                  const Vxd &__u__weight, const Vxd &__u__ref,
                                  Eigen::Ref<Vxd> u_lb, Eigen::Ref<Vxd> u_ub,
                                  Eigen::Ref<Vxd> u_weight,
                                  Eigen::Ref<Vxd> u_ref) {

  CHECK_EQ(__u_lb.size(), __u_ub.size(), AT);
  CHECK_EQ(__u__weight.size(), __u_ub.size(), AT);
  CHECK_EQ(__u__ref.size(), __u_ub.size(), AT);

  size_t nu = __u_lb.size() + 1;

  u_weight = Vxd(nu);
  u_lb = Vxd(nu);
  u_ub = Vxd(nu);
  u_ref = Vxd(nu);

  u_lb << __u_lb, .4;
  u_ub << __u_ub, 1.5;
  u_ref << __u__ref, .5;
  u_weight << __u__weight, .7;
}

ptr<crocoddyl::ShootingProblem>
generate_problem(const Generate_params &gen_args, size_t &nx, size_t &nu) {

  std::cout << "**\nGENERATING PROBLEM\n**\n" << std::endl;
  gen_args.print(std::cout);
  std::cout << "**\n" << std::endl;

  std::vector<ptr<Cost>> feats_terminal;
  ptr<crocoddyl::ActionModelAbstract> am_terminal;
  ptr<Dynamics> dyn;

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> amq_runs;
  Vxd goal_v = gen_args.goal;

  if (gen_args.free_time && gen_args.contour_control) {
    CHECK(false, AT);
  }

  Vxd x_ub, x_lb, u_ub, u_lb, u_ref;
  Vxd weight_b;
  Vxd u_weight;

  double dt = .0;

  if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0"},
           gen_args.name))
    dt = .1;
  else if (__in(vstr{"quad2d", "quadrotor_0"}, gen_args.name))
    dt = .01;
  else
    CHECK(false, AT);

  double max_ = std::numeric_limits<double>::max();
  double low_ = std::numeric_limits<double>::lowest();
  if (gen_args.name == "unicycle_first_order_0") {
    dyn = mk<Dynamics_unicycle>(gen_args.free_time);
    if (gen_args.free_time) {
      u_weight = V3d(.2, .2, 1.);
      u_ref = V3d(0., 0., .5);
      u_lb = V3d(-.5, -.5, .4);
      u_ub = V3d(.5, .5, 1.5);
    } else if (gen_args.contour_control) {
      u_weight = V3d(.5, .5, .1);
      u_ref = V3d(0, 0, dt);
      u_lb = V3d(-.5, -.5, -10.);
      u_ub = V3d(.5, .5, 10);
      x_ub = max_ * Vxd::Ones(4);
      x_ub(3) = gen_args.max_alpha;
      weight_b = Vxd(4);
      weight_b << 0, 0, 0, 200.;
    } else {
      u_weight = V2d(.5, .5);
      u_ref = V2d::Zero();
      u_lb = V2d(-.5, -.5);
      u_ub = V2d(.5, .5);
    }
  } else if (gen_args.name == "unicycle_second_order_0") {
    dyn = mk<Dynamics_unicycle2>(gen_args.free_time);

    weight_b = Eigen::VectorXd(5);
    x_ub = Eigen::VectorXd(5);
    x_lb = Eigen::VectorXd(5);

    weight_b << 0., 0., 0., 20., 20.;
    x_ub << max_, max_, max_, .5, .5;
    x_lb << low_, low_, low_, -.5, -.5;

    if (gen_args.free_time) {
      u_weight = V3d(.2, .2, 1.);
      u_ref = V3d(0, 0, .5);
      u_lb = V3d(-.25, -.25, .4);
      u_ub = V3d(.25, .25, 1.5);
    } else if (gen_args.contour_control) {
      u_weight = V3d(.5, .5, .1);
      u_ref = V3d(.0, .0, dt);
      u_lb = V3d(-.25, -.25, -10);
      u_ub = V3d(.25, .25, 10);
      // i need new bounds for alpha
      x_ub.resize(6);
      x_lb.resize(6);
      weight_b.resize(6);
      weight_b << 0., 0., 0., 20., 20., 100;
      x_ub << max_, max_, max_, .5, .5, gen_args.max_alpha;
      x_lb << low_, low_, low_, -.5, -.5, -10;
    } else {
      u_weight = V2d(.5, .5);
      u_ref = V2d::Zero();
      u_lb = V2d(-.25, -.25);
      u_ub = V2d(.25, .25);
    }
  } else if (gen_args.name == "car_first_order_with_1_trailers_0") {

    std::cout << "missing the term to constrain the angle diff! " << std::endl;

    Vxd l(1);
    l << .5;
    dyn = mk<Dynamics_car_with_trailers>(l, gen_args.free_time);

    V2d __u_lb = V2d(-.1, -M_PI / 3.);
    V2d __u_ub = V2d(.5, M_PI / 3.);

    if (gen_args.free_time) {
      u_weight = V3d(.2, .2, 1.);
      u_lb = V3d();
      u_ub = V3d();
      u_lb << __u_lb, .4;
      u_ub << __u_ub, 1.5;
      u_ref = V3d(0, 0, .5);
    } else if (gen_args.contour_control) {
      u_weight = V3d(.5, .5, .1);
      u_lb = V3d();
      u_ub = V3d();
      u_lb << __u_lb, -10.;
      u_ub << __u_ub, 10.;
      x_ub = max_ * Vxd::Ones(5);
      x_ub(4) = gen_args.max_alpha;
      weight_b = Vxd(5);
      weight_b << 0, 0, 0, 0, 200.;
      u_ref = V3d::Zero();
    } else {
      u_weight = V2d(.2, .2);
      u_lb = __u_lb;
      u_ub = __u_ub;
      u_ref = V2d::Zero();
    }
  } else if (gen_args.name == "quad2d") {

    double max_v = 10;
    double max_omega = 10;
    Vxd __x_lb = Vxd(6);
    Vxd __x_ub = Vxd(6);
    Vxd __weight_xb = 10. * Vxd::Ones(6);
    // Vxd __weight_xb = 100 * Vxd::Ones(6);

    __x_lb << low_, low_, low_, -max_v, -max_v, -max_omega;
    __x_ub << max_, max_, max_, max_v, max_v, max_omega;

    double max_f = 2.;
    V2d __u_lb = V2d(0, 0);
    V2d __u_ub = V2d(max_f, max_f);
    dyn = mk<Dynamics_quadcopter2d>(gen_args.free_time);

    if (gen_args.free_time) {
      u_weight = .5 * V3d(.5, .5, 1.);
      u_lb = V3d();
      u_ub = V3d();
      u_lb << __u_lb, .4;
      u_ub << __u_ub, 1.5;
      u_ref = V3d(0, 0, .5);

      x_lb = __x_lb;
      x_ub = __x_ub;
      weight_b = __weight_xb;

    } else if (gen_args.contour_control) {
      u_weight = .5 * V3d(.5, .5, .1);
      u_lb = V3d();
      u_ub = V3d();
      u_lb << __u_lb, -10.;
      u_ub << __u_ub, 10.;
      u_ref = V3d::Zero();

      x_lb = Vxd(7);
      x_ub = Vxd(7);
      weight_b = Vxd(7);

      x_lb << __x_lb, -10;
      x_ub << __x_ub, gen_args.max_alpha;
      weight_b << __weight_xb, 200;

    }

    else {
      u_weight = .5 * V2d(.5, .5);
      u_lb = __u_lb;
      u_ub = __u_ub;
      u_ref = V2d::Zero();

      x_lb = __x_lb;
      x_ub = __x_ub;
      weight_b = __weight_xb;
    }
  }

  else if (gen_args.name == "quadrotor_0") {

    double max_v = 10;
    double max_omega = 10;
    nx = 13;
    Vxd __x_lb = Vxd(nx);
    Vxd __x_ub = Vxd(nx);
    Vxd __weight_xb = 10. * Vxd::Ones(nx);

    __x_lb.segment(0, 7) << low_, low_, low_, low_, low_, low_, low_;
    __x_lb.segment(7, 3) << -max_v, -max_v, -max_v;
    __x_lb.segment(10, 3) << -max_omega, -max_omega, -max_omega;

    __x_ub.segment(0, 7) << max_, max_, max_, max_, max_, max_, max_;
    __x_ub.segment(7, 3) << max_v, max_v, max_v;
    __x_ub.segment(10, 3) << max_omega, max_omega, max_omega;

    double max_f = 2.;
    V4d __u_lb = V4d(0, 0, 0, 0);
    V4d __u_ref = V4d(0, 0, 0, 0);
    V4d __u_ub = V4d(max_f, max_f, max_f, max_f);
    V4d __u__weight = .5 * V4d::Ones();

    dyn = mk<Dynamics_quadcopter3d>(gen_args.free_time);

    if (gen_args.free_time) {

      modify_u_bound_for_free_time(__u_lb, __u_ub, __u__weight, __u_ref, u_lb,
                                   u_ub, u_weight, u_ref);

      x_lb = __x_lb;
      x_ub = __x_ub;
      weight_b = __weight_xb;

    } else if (gen_args.contour_control) {

      modify_u_bound_for_contour(__u_lb, __u_ub, __u__weight, __u_ref, u_lb,
                                 u_ub, u_weight, u_ref);

      modify_x_bound_for_contour(__x_lb, __x_ub, __weight_xb, x_lb, x_ub,
                                 weight_b, gen_args.max_alpha);

    } else {
      u_lb = __u_lb;
      u_ub = __u_ub;
      u_ref = __u_ref;
      u_weight = __u__weight;

      x_lb = __x_lb;
      x_ub = __x_ub;
      weight_b = __weight_xb;
    }
  } else {
    CHECK(false, AT);
  }

  std::cout << STR_V(x_ub) << std::endl;
  std::cout << STR_V(x_lb) << std::endl;
  std::cout << STR_V(u_ub) << std::endl;
  std::cout << STR_V(u_lb) << std::endl;
  std::cout << STR_V(u_ref) << std::endl;
  std::cout << STR_V(weight_b) << std::endl;
  std::cout << STR_V(u_weight) << std::endl;

  if (gen_args.contour_control) {
    dyn = mk<Dynamics_contour>(dyn, true);
  }

  CHECK(dyn, AT);
  nx = dyn->nx;
  nu = dyn->nu;

  ptr<Cost> control_feature = mk<Control_cost>(nx, nu, nu, u_weight, u_ref);

  for (size_t t = 0; t < gen_args.N; t++) {

    std::vector<ptr<Cost>> feats_run;
    feats_run.push_back(control_feature);

    ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, gen_args.cl);

    if (gen_args.name == "unicycle_first_order_0") {
      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true, true, true};
    } else if (gen_args.name == "unicycle_second_order_0") {
      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true, true, true, false, false};
    } else if (gen_args.name == "car_first_order_with_1_trailers_0") {
      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true, true, true, true};
    } else if (gen_args.name == "quad2d") {
      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true, true, true, false, false, false};
    } else if (gen_args.name == "quadrotor_0") {
      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true,  true,  true,  true,  true,  true, true,
          false, false, false, false, false, false};
    } else {
      CHECK(false, AT);
    }

    if (gen_args.cl && gen_args.collisions)
      feats_run.push_back(cl_feature);
    else {
      std::cout << "not adding collision feature -- it is a nullptr or "
                   "collisions is false"
                << std::endl;
    }
    //
    if (gen_args.name == "quad2d") {
      std::cout << "adding regularization on w and v" << std::endl;

      Vxd state_weights(6);
      state_weights << .0, .0, .0, .2, .2, .2;
      Vxd state_ref = Vxd::Zero(6);

      ptr<Cost> state_feature =
          mk<State_cost>(nx, nu, nx, state_weights, state_ref);
      feats_run.push_back(state_feature);
    }
    if (gen_args.name == "quadrotor_0") {
      std::cout << "adding regularization on q, w and v" << std::endl;
      Vxd state_weights(13);
      state_weights.setOnes();
      state_weights *= .2;
      state_weights.segment(0, 3).setZero();
      Vxd state_ref = Vxd::Zero(13);
      state_ref(6) = 1.;

      ptr<Cost> state_feature =
          mk<State_cost>(nx, nu, nx, state_weights, state_ref);
      feats_run.push_back(state_feature);

      std::cout << "adding regularization on quaternion " << std::endl;

      ptr<Cost> quat_feature = mk<Quaternion_cost>(nx, nu);
      feats_run.push_back(quat_feature);
    }

    if (gen_args.states_weights.size() && gen_args.states.size()) {

      assert(gen_args.states_weights.size() == gen_args.states.size());
      assert(gen_args.states_weights.size() == gen_args.N);

      ptr<Cost> state_feature = mk<State_cost>(
          nx, nu, nx, gen_args.states_weights.at(t), gen_args.states.at(t));
      feats_run.push_back(state_feature);
    }

    if (x_lb.size())
      feats_run.push_back(mk<State_bounds>(nx, nu, nx, x_lb, -weight_b));

    if (x_ub.size())
      feats_run.push_back(mk<State_bounds>(nx, nu, nx, x_ub, weight_b));

    if (gen_args.contour_control && gen_args.cl)
      boost::static_pointer_cast<Col_cost>(cl_feature)->nx_effective = nx - 1;

    if (gen_args.contour_control) {

      CHECK(gen_args.linear_contour, AT);

      ptr<Contour_cost_alpha_u> contour_alpha_u =
          mk<Contour_cost_alpha_u>(nx, nu);
      contour_alpha_u->k = opti_params.k_linear;

      feats_run.push_back(contour_alpha_u);

      std::cout << "warning, no contour in non-terminal states" << std::endl;
      // ptr<Contour_cost_x> contour_x =
      //     mk<Contour_cost_x>(nx, nu, gen_args.interpolator);
      // contour_x->weight = opti_params.k_contour;

      std::cout << "warning, no cost on alpha in non-terminal states"
                << std::endl;
      // idea: use this only if it is close
      // ptr<Contour_cost_alpha_x> contour_alpha_x =
      //     mk<Contour_cost_alpha_x>(nx, nu);
      // contour_alpha_x->k = .1 * opti_params.k_linear;

      // feats_run.push_back(contour_x);
      // feats_run.push_back(contour_alpha_x);
    }

    auto am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));

    if (opti_params.control_bounds) {
      am_run->set_u_lb(u_lb);
      am_run->set_u_ub(u_ub);
    }
    amq_runs.push_back(am_run);
  }

  // Terminal

  if (gen_args.contour_control) {

    CHECK(gen_args.linear_contour, AT);
    ptr<Cost> state_bounds = mk<State_bounds>(nx, nu, nx, x_ub, weight_b);
    ptr<Contour_cost_x> contour_x =
        mk<Contour_cost_x>(nx, nu, gen_args.interpolator);
    contour_x->weight = opti_params.k_contour;

    // continue here: i have to add a weight!!

    feats_terminal.push_back(contour_x);
    feats_terminal.push_back(state_bounds);
  }

  if (gen_args.goal_cost) {
    ptr<Cost> state_feature = mk<State_cost>(
        nx, nu, nx, opti_params.weight_goal * Vxd::Ones(nx), gen_args.goal);
    feats_terminal.push_back(state_feature);
  }
  am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

  if (opti_params.use_finite_diff) {
    std::cout << "using finite diff!" << std::endl;

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
        amq_runs_diff(amq_runs.size());

    // double disturbance = 1e-4; // should be high, becaues I have collisions
    double disturbance = opti_params.disturbance;
    std::transform(
        amq_runs.begin(), amq_runs.end(), amq_runs_diff.begin(),
        [&](const auto &am_run) {
          auto am_rundiff = mk<crocoddyl::ActionModelNumDiff>(am_run, true);
          boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_rundiff)
              ->set_disturbance(disturbance);
          if (opti_params.control_bounds) {
            am_rundiff->set_u_lb(am_run->get_u_lb());
            am_rundiff->set_u_ub(am_run->get_u_ub());
          }
          return am_rundiff;
        });

    amq_runs = amq_runs_diff;

    auto am_terminal_diff =
        mk<crocoddyl::ActionModelNumDiff>(am_terminal, true);
    boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_terminal_diff)
        ->set_disturbance(disturbance);
    am_terminal = am_terminal_diff;
  }

  CHECK(am_terminal, AT);

  for (auto &a : amq_runs)
    CHECK(a, AT);

  ptr<crocoddyl::ShootingProblem> problem =
      mk<crocoddyl::ShootingProblem>(gen_args.start, amq_runs, am_terminal);

  return problem;
};

bool check_dynamics(const std::vector<Vxd> &xs_out,
                    const std::vector<Vxd> &us_out, ptr<Dynamics> dyn) {

  CHECK(xs_out.size(), AT);
  CHECK(us_out.size(), AT);
  CHECK(dyn, AT);
  CHECK_EQ(xs_out.size(), us_out.size() + 1, AT);

  double tolerance = 1e-4;
  size_t N = us_out.size();
  bool feasible = true;

  for (size_t i = 0; i < N; i++) {
    Vxd xnext(dyn->nx);

    auto &x = xs_out.at(i);
    auto &u = us_out.at(i);
    dyn->calc(xnext, x, u);

    if ((xnext - xs_out.at(i + 1)).norm() > tolerance) {
      std::cout << "Infeasible at " << i << std::endl;
      std::cout << xnext.format(FMT) << std::endl;
      std::cout << xs_out.at(i + 1).format(FMT) << std::endl;
      std::cout << (xnext - xs_out.at(i + 1)).format(FMT) << std::endl;
      feasible = false;
      break;
    }
  }

  return feasible;
}

Vxd enforce_bounds(const Vxd &us, const Vxd &lb, const Vxd &ub) {

  CHECK_EQ(us.size(), lb.size(), AT);
  CHECK_EQ(us.size(), ub.size(), AT);
  return us.cwiseMax(lb).cwiseMin(ub);
}

void read_from_file(File_parser_inout &inout) {

  double dt = 0;

  YAML::Node init = YAML::LoadFile(inout.init_guess);
  YAML::Node env = YAML::LoadFile(inout.env_file);

  if (!env["robots"]) {
    CHECK(false, AT);
    // ...
  }

  inout.name = env["robots"][0]["type"].as<std::string>();

  if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0"},
           inout.name))
    dt = .1;
  else if (__in(vstr{"quad2d", "quadrotor_0"}, inout.name))
    dt = .01;
  else
    CHECK(false, AT);

  std::cout << STR_(dt) << std::endl;
  // load the collision checker
  if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0",
                "car_first_order_with_1_trailers_0", "quad2d", "quadrotor_0"},
           inout.name)) {
    inout.cl = mk<CollisionChecker>();
    inout.cl->load(inout.env_file);
  } else {
    std::cout << "this robot doesn't have collision checking " << std::endl;
    inout.cl = nullptr;
  }

  std::vector<std::vector<double>> states;
  // std::vector<Vxd> xs_init;
  // std::vector<Vxd> us_init;

  size_t N;
  std::vector<std::vector<double>> actions;

  if (!inout.new_format) {

    if (!init["result"]) {
      CHECK(false, AT);
      // ...
    }

    for (const auto &state : init["result"][0]["states"]) {
      std::vector<double> p;
      for (const auto &elem : state) {
        p.push_back(elem.as<double>());
      }
      states.push_back(p);
    }

    N = states.size() - 1;

    for (const auto &state : init["result"][0]["actions"]) {
      std::vector<double> p;
      for (const auto &elem : state) {
        p.push_back(elem.as<double>());
      }
      actions.push_back(p);
    }

    inout.xs.resize(states.size());
    inout.us.resize(actions.size());

    std::transform(states.begin(), states.end(), inout.xs.begin(),
                   [](const auto &s) { return Vxd::Map(s.data(), s.size()); });

    std::transform(actions.begin(), actions.end(), inout.us.begin(),
                   [](const auto &s) { return Vxd::Map(s.data(), s.size()); });

  } else {

    if (!init["result2"]) {
      CHECK(false, AT);
    }

    std::cout << "Reading results in the new format " << std::endl;
    for (const auto &state : init["result2"][0]["states"]) {
      std::vector<double> p;
      for (const auto &elem : state) {
        p.push_back(elem.as<double>());
      }
      states.push_back(p);
    }

    N = states.size() - 1;

    for (const auto &action : init["result2"][0]["actions"]) {
      std::vector<double> p;
      for (const auto &elem : action) {
        p.push_back(elem.as<double>());
      }
      actions.push_back(p);
    }

    std::vector<double> _times;

    for (const auto &time : init["result2"][0]["times"]) {
      _times.push_back(time.as<double>());
    }

    // 0 ... 3.5
    // dt is 1.
    // 0 1 2 3 4

    // we use floor in the time to be more agressive
    std::cout << STR_(dt) << std::endl;

    double total_time = _times.back();

    int num_time_steps = std::ceil(total_time / dt);

    Vxd times = Vxd::Map(_times.data(), _times.size());

    std::vector<Vxd> _xs_init(states.size());
    std::vector<Vxd> _us_init(actions.size());

    std::vector<Vxd> xs_init_new;
    std::vector<Vxd> us_init_new;

    int nx = 3;
    int nu = 2;

    std::transform(states.begin(), states.end(), _xs_init.begin(),
                   [](const auto &s) { return Vxd::Map(s.data(), s.size()); });

    std::transform(actions.begin(), actions.end(), _us_init.begin(),
                   [](const auto &s) { return Vxd::Map(s.data(), s.size()); });

    auto ts = Vxd::LinSpaced(num_time_steps + 1, 0, num_time_steps * dt);

    std::cout << "taking samples at " << ts.transpose() << std::endl;

    for (size_t ti = 0; ti < num_time_steps + 1; ti++) {
      Vxd xout(nx);
      Vxd Jout(nx);

      if (ts(ti) > times.tail(1)(0))
        xout = _xs_init.back();
      else
        linearInterpolation(times, _xs_init, ts(ti), xout, Jout);
      xs_init_new.push_back(xout);
    }

    auto times_u = times.head(times.size() - 1);
    for (size_t ti = 0; ti < num_time_steps; ti++) {
      Vxd uout(nu);
      Vxd Jout(nu);
      if (ts(ti) > times_u.tail(1)(0))
        uout = _us_init.back();
      else
        linearInterpolation(times_u, _us_init, ts(ti), uout, Jout);
      us_init_new.push_back(uout);
    }

    N = num_time_steps;

    std::cout << "N: " << N << std::endl;
    std::cout << "us:  " << us_init_new.size() << std::endl;
    std::cout << "xs: " << xs_init_new.size() << std::endl;

    inout.xs = xs_init_new;
    inout.us = us_init_new;

    std::ofstream debug_file("debug.txt");

    for (auto &x : inout.xs) {
      debug_file << x.format(FMT) << std::endl;
    }
    debug_file << "---" << std::endl;

    for (auto &u : inout.us) {
      debug_file << u.format(FMT) << std::endl;
    }
  }

  std::vector<double> _start, _goal;
  for (const auto &e : env["robots"][0]["start"]) {
    _start.push_back(e.as<double>());
  }

  for (const auto &e : env["robots"][0]["goal"]) {
    _goal.push_back(e.as<double>());
  }

  inout.start = Vxd::Map(_start.data(), _start.size());
  inout.goal = Vxd::Map(_goal.data(), _goal.size());

  bool verbose = false;
  if (verbose) {

    std::cout << "states " << std::endl;
    for (auto &x : inout.xs)
      std::cout << x.format(FMT) << std::endl;

    std::cout << "actions " << std::endl;
    for (auto &u : inout.us)
      std::cout << u.format(FMT) << std::endl;
  }
}

void convert_traj_with_variable_time(const std::vector<Vxd> &xs,
                                     const std::vector<Vxd> &us,
                                     std::vector<Vxd> &xs_out,
                                     std::vector<Vxd> &us_out,
                                     const double &dt) {

  CHECK(xs.size(), AT);
  CHECK(us.size(), AT);
  CHECK_EQ(xs.size(), us.size() + 1, AT);

  size_t N = us.size();
  size_t nx = xs.front().size();

  size_t nu = us.front().size();
  double total_time =
      std::accumulate(us.begin(), us.end(), 0., [&dt](auto &a, auto &b) {
        return a + dt * b(b.size() - 1);
      });

  std::cout << "original total time: " << dt * us.size() << std::endl;
  std::cout << "new total_time: " << total_time << std::endl;

  int num_time_steps = std::ceil(total_time / dt);
  std::cout << "number of time steps " << num_time_steps << std::endl;
  std::cout << "new total time " << num_time_steps * dt << std::endl;
  double scaling_factor = num_time_steps * dt / total_time;
  std::cout << "scaling factor " << scaling_factor << std::endl;
  assert(scaling_factor >= 1);
  CHECK_GEQ(scaling_factor, 1., AT);

  // now I have to sample at every dt
  // TODO: lOOK for Better solution than the trick with scaling

  auto times = Vxd(N + 1);
  times.setZero();
  for (size_t i = 1; i < times.size(); i++) {
    times(i) = times(i - 1) + dt * us.at(i - 1)(nu - 1);
  }
  // std::cout << times.transpose() << std::endl;

  // TODO: be careful with SO(2)
  std::vector<Vxd> x_out, u_out;
  for (size_t i = 0; i < num_time_steps + 1; i++) {
    double t = i * dt / scaling_factor;
    Vxd out(nx);
    Vxd Jout(nx);
    linearInterpolation(times, xs, t, out, Jout);
    x_out.push_back(out);
  }

  std::vector<Vxd> u_nx_orig(us.size());
  std::transform(us.begin(), us.end(), u_nx_orig.begin(),
                 [&nu](auto &s) { return s.head(nu - 1); });

  for (size_t i = 0; i < num_time_steps; i++) {
    double t = i * dt / scaling_factor;
    Vxd out(nu - 1);
    // std::cout << " i and time and num_time_steps is " << i << " " << t << "
    // "
    //           << num_time_steps << std::endl;
    Vxd J(nu - 1);
    linearInterpolation(times.head(times.size() - 1), u_nx_orig, t, out, J);
    u_out.push_back(out);
  }

  std::cout << "u out " << u_out.size() << std::endl;
  std::cout << "x out " << x_out.size() << std::endl;

  xs_out = x_out;
  us_out = u_out;
}

int inside_bounds(int i, int lb, int ub) {

  assert(lb <= ub);

  if (i < lb)
    return lb;

  else if (i > ub)
    return ub;
  else
    return i;
}

auto smooth_traj(const std::vector<Vxd> &us_init) {

  size_t n = us_init.front().size();
  std::vector<Vxd> us_out(us_init.size());
  // kernel

  Vxd kernel(5);

  kernel << 1, 2, 3, 2, 1;

  kernel /= kernel.sum();

  for (size_t i = 0; i < us_init.size(); i++) {
    Vxd out = Vxd::Zero(n);
    for (size_t j = 0; j < kernel.size(); j++) {
      out += kernel(j) * us_init.at(inside_bounds(i - kernel.size() / 2 + j, 0,
                                                  us_init.size() - 1));
    }
    us_out.at(i) = out;
  }
  return us_out;
}

struct ReportCost {
  double cost;
  int time;
  std::string name;
  Eigen::VectorXd r;
  CostTYPE type;
};

std::vector<ReportCost>
get_report(ptr<ActionModelQ> p,
           std::function<void(ptr<Cost>, Eigen::Ref<Vxd>)> fun) {

  std::vector<ReportCost> reports;
  for (size_t j = 0; j < p->features.size(); j++) {
    ReportCost report;
    auto &f = p->features.at(j);
    Vxd r(f->nr);
    fun(f, r);
    report.type = CostTYPE::least_squares;
    report.name = f->get_name();
    report.r = r;
    if (f->cost_type == CostTYPE::least_squares) {
      report.cost = .5 * r.dot(r);
    } else if (f->cost_type == CostTYPE::linear) {
      report.cost = r.sum();
    }
    reports.push_back(report);
  }
  return reports;
}

std::vector<ReportCost> report_problem(ptr<crocoddyl::ShootingProblem> problem,
                                       const std::vector<Vxd> &xs,
                                       const std::vector<Vxd> &us,
                                       const char *file_name) {
  std::vector<ReportCost> reports;

  for (size_t i = 0; i < problem->get_runningModels().size(); i++) {
    auto &x = xs.at(i);
    auto &u = us.at(i);
    auto p = boost::static_pointer_cast<ActionModelQ>(
        problem->get_runningModels().at(i));
    std::vector<ReportCost> reports_i = get_report(
        p, [&](ptr<Cost> f, Eigen::Ref<Vxd> r) { f->calc(r, x, u); });

    for (auto &report_ii : reports_i)
      report_ii.time = i;
    reports.insert(reports.end(), reports_i.begin(), reports_i.end());
  }

  auto p =
      boost::static_pointer_cast<ActionModelQ>(problem->get_terminalModel());
  std::vector<ReportCost> reports_t = get_report(
      p, [&](ptr<Cost> f, Eigen::Ref<Vxd> r) { f->calc(r, xs.back()); });

  for (auto &report_ti : reports_t)
    report_ti.time = xs.size() - 1;
  ;

  reports.insert(reports.begin(), reports_t.begin(), reports_t.end());

  // write down the reports.
  //

  std::string one_space = " ";
  std::string two_space = "  ";
  std::string four_space = "    ";
  std::ofstream reports_file(file_name);
  for (auto &report : reports) {
    reports_file << "-" << one_space << "name: " << report.name << std::endl;
    reports_file << two_space << "time: " << report.time << std::endl;
    reports_file << two_space << "cost: " << report.cost << std::endl;
    reports_file << two_space << "type: " << static_cast<int>(report.type)
                 << std::endl;
    if (report.r.size()) {
      reports_file << two_space << "r: " << report.r.format(FMT) << std::endl;
    }
  }

  return reports;
}

void solve_with_custom_solver(File_parser_inout &file_inout,
                              Result_opti &opti_out) {

  // list of single solver

  std::vector<SOLVER> solvers{
      SOLVER::traj_opt,      SOLVER::traj_opt_free_time_proxi,
      SOLVER::mpc,           SOLVER::mpcc,
      SOLVER::mpcc2,         SOLVER::mpcc_linear,
      SOLVER::mpc_adaptative};

  assert(__in_if(solvers, [](const SOLVER &s) {
    return s == static_cast<SOLVER>(opti_params.solver_id);
  }));

  bool verbose = false;
  auto cl = file_inout.cl;
  auto xs_init = file_inout.xs;
  auto us_init = file_inout.us;
  size_t N = us_init.size();
  auto goal = file_inout.goal;
  auto start = file_inout.start;
  auto name = file_inout.name;
  double dt = 0;

  if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0"}, name))
    dt = .1;
  else if (__in(vstr{"quad2d", "quadrotor_0"}, name))
    dt = .01;
  else
    CHECK(false, AT);

  std::cout << STR_(dt) << std::endl;

  CHECK(us_init.size(), AT);
  CHECK(xs_init.size(), AT);
  CHECK_EQ(xs_init.size(), us_init.size() + 1, AT);
  const size_t _nx = xs_init.front().size();
  const size_t _nu = us_init.front().size();

  SOLVER solver = static_cast<SOLVER>(opti_params.solver_id);

  if (opti_params.repair_init_guess) {
    if (name == "unicycle_first_order_0" || name == "unicycle_second_order_0") {
      std::cout << "WARNING: reparing init guess, annoying SO2" << std::endl;
      for (size_t i = 1; i < N + 1; i++) {
        xs_init.at(i)(2) = xs_init.at(i - 1)(2) +
                           diff_angle(xs_init.at(i)(2), xs_init.at(i - 1)(2));
      }
      goal(2) = xs_init.at(N)(2) + diff_angle(goal(2), xs_init.at(N)(2));

      std::cout << "goal is now (maybe updated) " << goal.transpose()
                << std::endl;
    } else if (name == "car_first_order_with_1_trailers_0") {

      std::cout << "WARNING: reparing init guess, annoying SO2" << std::endl;
      for (size_t i = 1; i < N + 1; i++) {
        xs_init.at(i)(2) = xs_init.at(i - 1)(2) +
                           diff_angle(xs_init.at(i)(2), xs_init.at(i - 1)(2));

        xs_init.at(i)(3) = xs_init.at(i - 1)(3) +
                           diff_angle(xs_init.at(i)(3), xs_init.at(i - 1)(3));
      }
      goal(2) = xs_init.at(N)(2) + diff_angle(goal(2), xs_init.at(N)(2));
      goal(3) = xs_init.at(N)(3) + diff_angle(goal(3), xs_init.at(N)(3));

      std::cout << "goal is now (maybe updated) " << goal.transpose()
                << std::endl;
    }
  }

  if (opti_params.smooth_traj) {
    for (size_t i = 0; i < 10; i++) {
      xs_init = smooth_traj(xs_init);
      xs_init.at(0) = start;
      xs_init.back() = goal;
    }

    for (size_t i = 0; i < 10; i++) {
      us_init = smooth_traj(us_init);
      // xs_init.at(0) = start;
      // xs_init.back() = goal;
    }

    // xs_init = smooth_traj(xs_init);
    // xs_init.at(0) = start;
    // xs_init.back() = goal;
    //
    //
    // xs_init = smooth_traj(xs_init);
    // xs_init.at(0) = start;
    // xs_init.back() = goal;
    //
    //
    // xs_init = smooth_traj(xs_init);
    // xs_init.at(0) = start;
    // xs_init.back() = goal;
    //
    // // force start and goal.
    //
    //
    // us_init = smooth_traj(us_init);
  }

  // if (free_time) {
  //   for (size_t i = 0; i < N; i++) {
  //     std::vector<double> new_u = actions.at(i);
  //     new_u.push_back(1.);
  //     actions.at(i) = new_u;
  //   }
  // }

  if (verbose) {
    std::cout << "states " << std::endl;
    for (auto &x : xs_init)
      std::cout << x.format(FMT) << std::endl;
  }

  bool feasible;
  std::vector<Vxd> xs_out;
  std::vector<Vxd> us_out;

  std::cout << "WARNING: "
            << "i modify last state to match goal" << std::endl;
  xs_init.back() = goal;

  std::ofstream debug_file_yaml(opti_params.debug_file_name);
  debug_file_yaml << "name: " << name << std::endl;
  debug_file_yaml << "N: " << N << std::endl;
  debug_file_yaml << "start: " << start.format(FMT) << std::endl;
  debug_file_yaml << "goal: " << goal.format(FMT) << std::endl;
  debug_file_yaml << "xs0: " << std::endl;
  for (auto &x : xs_init)
    debug_file_yaml << "  - " << x.format(FMT) << std::endl;

  debug_file_yaml << "us0: " << std::endl;
  for (auto &x : us_init)
    debug_file_yaml << "  - " << x.format(FMT) << std::endl;

  if (solver == SOLVER::mpc || solver == SOLVER::mpcc ||
      solver == SOLVER::mpcc_linear || solver == SOLVER::mpc_adaptative) {
    // i could not stop when I reach the goal, only stop when I reach
    // it with the step move. Then, I would do the last at full speed?
    // ( I hope :) ) Anyway, now is just fine

    CHECK_GEQ(opti_params.window_optimize, opti_params.window_shift, AT);

    bool finished = false;

    std::vector<Vxd> xs_opt;
    std::vector<Vxd> us_opt;

    std::vector<Vxd> xs_init_rewrite = xs_init;
    std::vector<Vxd> us_init_rewrite = us_init;

    std::vector<Vxd> xs_warmstart_mpcc;
    std::vector<Vxd> us_warmstart_mpcc;

    std::vector<Vxd> xs_warmstart_adptative;
    std::vector<Vxd> us_warmstart_adptative;

    xs_opt.push_back(start);
    xs_init_rewrite.at(0) = start;

    debug_file_yaml << "opti:" << std::endl;

    auto times = Vxd::LinSpaced(xs_init.size(), 0, (xs_init.size() - 1) * dt);

    double max_alpha = times(times.size() - 1);

    ptr<Interpolator> path = mk<Interpolator>(times, xs_init);
    ptr<Interpolator> path_u =
        mk<Interpolator>(times.head(times.size() - 1), us_init);

    std::vector<Vxd> xs;
    std::vector<Vxd> us;

    double previous_alpha;

    Vxd previous_state = start;
    ptr<crocoddyl::ShootingProblem> problem;

    Vxd goal_mpc(_nx);

    bool is_last = false;

    double total_time = 0;
    size_t counter = 0;
    size_t total_iterations = 0;
    size_t window_optimize_i = 0;

    bool close_to_goal = false;

    Vxd goal_with_alpha(_nx + 1);
    goal_with_alpha.head(_nx) = goal;
    goal_with_alpha(_nx) = max_alpha;

    bool last_reaches_ = false;
    size_t index_first_goal = 0;
    while (!finished) {

      if (solver == SOLVER::mpc || solver == SOLVER::mpc_adaptative) {

        auto start_i = previous_state;
        if (solver == SOLVER::mpc) {
          assert(N - counter * opti_params.window_shift >= 0);
          size_t remaining_steps = N - counter * opti_params.window_shift;

          window_optimize_i =
              std::min(opti_params.window_optimize, remaining_steps);

          int subgoal_index =
              counter * opti_params.window_shift + window_optimize_i;

          is_last = opti_params.window_optimize > remaining_steps;

          if (is_last)
            goal_mpc = goal;
          else
            goal_mpc = xs_init.at(subgoal_index);
        } else if (solver == SOLVER::mpc_adaptative) {

          std::cout << "previous state is " << previous_state.format(FMT)
                    << std::endl;
          auto it =
              std::min_element(path->x.begin(), path->x.end(),
                               [&](const auto &a, const auto &b) {
                                 return (a - previous_state).squaredNorm() <=
                                        (b - previous_state).squaredNorm();
                               });

          size_t index = std::distance(path->x.begin(), it);
          std::cout << "starting with approx index " << index << std::endl;
          std::cout << "Non adaptative index would be: "
                    << counter * opti_params.window_shift << std::endl;

          window_optimize_i = opti_params.window_optimize;
          // next goal:
          size_t goal_index = index + window_optimize_i;
          if (goal_index > xs_init.size() - 1) {
            std::cout << "trying to reach the goal " << std::endl;
            goal_mpc = goal;
          } else {
            goal_mpc = xs_init.at(goal_index);
          }
        }

        std::cout << "is_last:" << is_last << std::endl;
        std::cout << "counter:" << counter << std::endl;
        std::cout << "goal_i:" << goal_mpc.transpose() << std::endl;
        std::cout << "start_i:" << start_i.transpose() << std::endl;

        // i nd

        Generate_params gen_args{.free_time = false,
                                 .name = name,
                                 .N = window_optimize_i,
                                 .goal = goal_mpc,
                                 .start = start_i,
                                 .cl = cl,
                                 .states = {},
                                 .actions = {},
                                 .collisions =
                                     opti_params.collision_weight > 1e-3};

        size_t nx, nu;

        problem = generate_problem(gen_args, nx, nu);

        // report problem

        if (opti_params.use_warmstart) {

          if (solver == SOLVER::mpc) {
            xs = std::vector<Vxd>(
                xs_init_rewrite.begin() + counter * opti_params.window_shift,
                xs_init_rewrite.begin() + counter * opti_params.window_shift +
                    window_optimize_i + 1);

            us = std::vector<Vxd>(
                us_init_rewrite.begin() + counter * opti_params.window_shift,
                us_init_rewrite.begin() + counter * opti_params.window_shift +
                    window_optimize_i);

          } else {

            if (counter) {
              std::cout << "new warmstart" << std::endl;
              xs = xs_warmstart_adptative;
              us = us_warmstart_adptative;

              size_t missing_steps = window_optimize_i - us.size();

              Vxd u_last = Vxd::Zero(nu);
              Vxd x_last = xs.back();

              // TODO: Sample the interpolator to get new init guess.

              if (opti_params.shift_repeat) {
                for (size_t i = 0; i < missing_steps; i++) {
                  us.push_back(u_last);
                  xs.push_back(x_last);
                }
              } else {

                std::cout << "filling window by sampling the trajectory"
                          << std::endl;
                Vxd last = xs_warmstart_adptative.back().head(_nx);

                auto it = std::min_element(path->x.begin(), path->x.end(),
                                           [&](const auto &a, const auto &b) {
                                             return (a - last).squaredNorm() <=
                                                    (b - last).squaredNorm();
                                           });

                size_t last_index = std::distance(path->x.begin(), it);
                double alpha_of_last = path->times(last_index);
                std::cout << STR_(last_index) << std::endl;
                // now I

                Vxd out(_nx);
                Vxd J(_nx);

                Vxd out_u(_nu);
                Vxd J_u(_nu);

                for (size_t i = 0; i < missing_steps; i++) {
                  {
                    path->interpolate(
                        std::min(alpha_of_last + (i + 1) * dt, max_alpha), out,
                        J);
                    xs.push_back(out);
                  }

                  {
                    path_u->interpolate(
                        std::min(alpha_of_last + i * dt, max_alpha - dt), out_u,
                        J_u);
                    us.push_back(out_u);
                  }
                }
              }

            } else {
              std::cout << "first iteration -- using first" << std::endl;

              if (window_optimize_i + 1 < xs_init.size()) {
                xs = std::vector<Vxd>(xs_init.begin(),
                                      xs_init.begin() + window_optimize_i + 1);
                us = std::vector<Vxd>(us_init.begin(),
                                      us_init.begin() + window_optimize_i);
              } else {
                std::cout << "Optimizing more steps than required" << std::endl;
                xs = xs_init;
                us = us_init;

                size_t missing_steps = window_optimize_i - us.size();
                Vxd u_last = Vxd::Zero(nu);
                Vxd x_last = xs.back();

                // TODO: Sample the interpolator to get new init guess.

                for (size_t i = 0; i < missing_steps; i++) {
                  us.push_back(u_last);
                  xs.push_back(x_last);
                }
              }

              // xs = std::vector<Vxd>(window_optimize_i + 1,
              // gen_args.start); us =
              // std::vector<Vxd>(window_optimize_i, Vxd::Zero(nu));
            }
          }
          CHECK_EQ(xs.size(), window_optimize_i + 1, AT);
          CHECK_EQ(us.size(), window_optimize_i, AT);

          CHECK_EQ(xs.size(), us.size() + 1, AT);
          CHECK_EQ(us.size(), window_optimize_i, AT);

        } else {
          xs = std::vector<Vxd>(window_optimize_i + 1, gen_args.start);
          us = std::vector<Vxd>(window_optimize_i, Vxd::Zero(nu));
        }
      } else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        window_optimize_i =
            std::min(opti_params.window_optimize, us_init.size());

        std::cout << "previous state " << previous_state.format(FMT)
                  << std::endl;
        // approx alpha of first state
        auto it = std::min_element(
            path->x.begin(), path->x.end(), [&](const auto &a, const auto &b) {
              return (a - previous_state).squaredNorm() <=
                     (b - previous_state).squaredNorm();
            });
        size_t first_index = std::distance(path->x.begin(), it);
        double alpha_of_first = path->times(first_index);

        size_t expected_last_index = first_index + window_optimize_i;

        // std::cout << "starting with approx first_index " <<
        // first_index << std::endl; std::cout << "alpha of first " <<
        // alpha_of_first << std::endl;

        Vxd alpha_refs =
            Vxd::LinSpaced(window_optimize_i + 1, alpha_of_first,
                           alpha_of_first + window_optimize_i * dt);

        double expected_final_alpha =
            std::min(alpha_of_first + window_optimize_i * dt, max_alpha);

        std::cout << STR(first_index, ":") << std::endl;
        std::cout << STR(expected_final_alpha, ":") << std::endl;
        std::cout << STR(expected_last_index, ":") << std::endl;
        std::cout << STR(alpha_of_first, ":") << std::endl;
        std::cout << STR(max_alpha, ":") << std::endl;

        size_t nx, nu;

        Vxd start_ic(_nx + 1);
        start_ic.head(_nx) = previous_state.head(_nx);
        start_ic(_nx) = alpha_of_first;

        bool goal_cost = false;

        if (expected_final_alpha > max_alpha - 1e-3 || close_to_goal) {
          std::cout << "alpha_refs > max_alpha || close to goal" << std::endl;
          goal_cost = true; // new
        }

        std::cout << "goal " << goal_with_alpha.format(FMT) << std::endl;
        std::cout << STR_(goal_cost) << std::endl;

        std::vector<Vxd> state_weights;
        std::vector<Vxd> _states(window_optimize_i);

        int try_faster = 5;
        if (last_reaches_) {
          std::cout << "last_reaches_ adds goal cost special" << std::endl;
          std::cout << "try_faster: " << try_faster << std::endl;

          state_weights.resize(window_optimize_i);
          _states.resize(window_optimize_i);

          for (size_t t = 0; t < window_optimize_i; t++) {
            if (t > index_first_goal - opti_params.window_shift - try_faster)
              state_weights.at(t) = 1. * Vxd::Ones(_nx + 1);
            else
              state_weights.at(t) = Vxd::Zero(_nx + 1);
          }

          for (size_t t = 0; t < window_optimize_i; t++) {
            _states.at(t) = goal_with_alpha;
          }
        }

        Generate_params gen_args{
            .free_time = false,
            .name = name,
            .N = window_optimize_i,
            .goal = goal_with_alpha,
            .start = start_ic,
            .cl = cl,
            .states = _states,
            .states_weights = state_weights,
            .actions = {},
            .contour_control = true,
            .interpolator = path,
            .max_alpha = max_alpha,
            .linear_contour = solver == SOLVER::mpcc_linear,
            .goal_cost = goal_cost,
            .collisions = opti_params.collision_weight > 1e-3

        };

        problem = generate_problem(gen_args, nx, nu);

        if (opti_params.use_warmstart) {

          // TODO: I need a more clever initialization. For example,
          // using the ones missing from last time, and then the
          // default?

          std::cout << "warmstarting " << std::endl;

          std::vector<Vxd> xs_i;
          std::vector<Vxd> us_i;

          std::cout << STR(counter, ":") << std::endl;

          if (counter) {
            std::cout << "reusing solution from last iteration "
                         "(window swift)"
                      << std::endl;
            xs_i = xs_warmstart_mpcc;
            us_i = us_warmstart_mpcc;

            size_t missing_steps = window_optimize_i - us_i.size();

            Vxd u_last = Vxd::Zero(_nu + 1);
            Vxd x_last = xs_i.back();

            // TODO: Sample the interpolator to get new init guess.

            if (opti_params.shift_repeat) {
              std::cout << "filling window with last solution " << std::endl;
              for (size_t i = 0; i < missing_steps; i++) {
                us_i.push_back(u_last);
                xs_i.push_back(x_last);
              }
            } else {
              // get the alpha  of the last one.
              std::cout << "filling window by sampling the trajectory"
                        << std::endl;
              Vxd last = xs_warmstart_mpcc.back().head(_nx);

              auto it = std::min_element(path->x.begin(), path->x.end(),
                                         [&](const auto &a, const auto &b) {
                                           return (a - last).squaredNorm() <=
                                                  (b - last).squaredNorm();
                                         });

              size_t last_index = std::distance(path->x.begin(), it);
              double alpha_of_last = path->times(last_index);
              std::cout << STR_(last_index) << std::endl;
              // now I

              Vxd out(_nx);
              Vxd J(_nx);

              Vxd out_u(_nu);
              Vxd J_u(_nu);

              Vxd uu(_nu + 1);
              Vxd xx(_nx + 1);
              for (size_t i = 0; i < missing_steps; i++) {
                {
                  path->interpolate(
                      std::min(alpha_of_last + (i + 1) * dt, max_alpha), out,
                      J);
                  xx.head(_nx) = out;
                  xx(_nx) = alpha_of_last + i * dt;
                  xs_i.push_back(xx);
                }

                {
                  path_u->interpolate(
                      std::min(alpha_of_last + i * dt, max_alpha - dt), out_u,
                      J_u);
                  uu.head(_nu) = out_u;
                  uu(_nu) = dt;
                  us_i.push_back(uu);
                }
              }
            }
          } else {
            std::cout << "first iteration, using initial guess trajectory"
                      << std::endl;

            Vxd x(_nx + 1);
            Vxd u(_nu + 1);
            for (size_t i = 0; i < window_optimize_i + 1; i++) {

              x.head(_nx) = xs_init.at(i);
              x(_nx) = alpha_of_first + dt * i;
              xs_i.push_back(x);

              if (i < window_optimize_i) {
                u.head(_nu) = us_init.at(i);
                u(_nu) = dt;
                us_i.push_back(u);
              }
            }
          }
          xs = xs_i;
          us = us_i;

        } else {
          std::cout << "no warmstart " << std::endl;
          // no warmstart
          Vxd u0c(_nu + 1);
          u0c.head(_nu).setZero();
          u0c(_nu) = dt;
          xs = std::vector<Vxd>(window_optimize_i + 1, gen_args.start);
          us = std::vector<Vxd>(window_optimize_i, u0c);
        }
        CHECK_EQ(xs.size(), window_optimize_i + 1, AT);
        CHECK_EQ(us.size(), window_optimize_i, AT);
      }
      // report problem

      // auto models = problem->get_runningModels();

      crocoddyl::SolverBoxFDDP ddp(problem);
      ddp.set_th_stop(opti_params.th_stop);
      ddp.set_th_acceptnegstep(opti_params.th_acceptnegstep);

      if (opti_params.CALLBACKS) {
        std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
        cbs.push_back(mk<crocoddyl::CallbackVerbose>());
        ddp.setCallbacks(cbs);
      }

      // ENFORCING BOUNDS

      Vxd x_lb, x_ub, u_lb, u_ub;

      double inf = std::numeric_limits<double>::max();
      if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        x_lb = -inf * Vxd::Ones(_nx + 1);
        x_ub = inf * Vxd::Ones(_nx + 1);

        u_lb = -inf * Vxd::Ones(_nu + 1);
        u_ub = inf * Vxd::Ones(_nu + 1);

        x_lb(x_lb.size() - 1) = 0;
        x_ub(x_ub.size() - 1) = max_alpha;
        u_lb(u_lb.size() - 1) = 0;

      } else {
        std::cout << "_nx" << _nx << std::endl;
        x_lb = -inf * Vxd::Ones(_nx);
        x_ub = inf * Vxd::Ones(_nx);

        u_lb = -inf * Vxd::Ones(_nu);
        u_ub = inf * Vxd::Ones(_nu);
      }

      if (opti_params.noise_level > 1e-8) {
        for (size_t i = 0; i < xs.size(); i++) {
          std::cout << "i " << i << " " << xs.at(i).size() << std::endl;
          xs.at(i) += opti_params.noise_level * Vxd::Random(xs.front().size());
          xs.at(i) = enforce_bounds(xs.at(i), x_lb, x_ub);
        }

        for (size_t i = 0; i < us.size(); i++) {
          us.at(i) += opti_params.noise_level * Vxd::Random(us.front().size());
          us.at(i) = enforce_bounds(us.at(i), u_lb, u_ub);
        }
      }

      crocoddyl::Timer timer;

      if (!opti_params.use_finite_diff)
        report_problem(problem, xs, us, "report-0.yaml");
      ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
      double time_i = timer.get_duration();
      size_t iterations_i = ddp.get_iter();
      if (!opti_params.use_finite_diff)
        report_problem(problem, ddp.get_xs(), ddp.get_us(), "report-1.yaml");
      accumulated_time += time_i;
      // solver == SOLVER::traj_opt_smooth_then_free_time) {

      std::cout << "time: " << time_i << std::endl;
      std::cout << "iterations: " << iterations_i << std::endl;
      total_time += time_i;
      total_iterations += iterations_i;
      std::vector<Vxd> xs_i_sol = ddp.get_xs();
      std::vector<Vxd> us_i_sol = ddp.get_us();

      previous_state = xs_i_sol.at(opti_params.window_shift).head(_nx);

      size_t copy_steps = 0;

      auto fun_is_goal = [&](const auto &x) {
        return (x.head(_nx) - goal).norm() < 1e-2;
      };

      if (solver == SOLVER::mpc_adaptative) {
        // check if I reach the goal.

        size_t final_index = window_optimize_i;
        Vxd x_last = ddp.get_xs().at(final_index);
        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "true last state: " << ddp.get_xs().back().format(FMT)
                  << std::endl;
        std::cout << "distance to goal: " << (x_last.head(_nx) - goal).norm()
                  << std::endl;

        if (fun_is_goal(x_last)) {
          std::cout << " x last " << x_last.format(FMT) << "reaches the goal"
                    << std::endl;

          std::cout << "setting x last to true " << std::endl;
          is_last = true;
        }

      }

      else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        std::cout << "if final reaches the goal, i stop" << std::endl;
        std::cout << "ideally, I should check if I can check the goal "
                     "faster, "
                     "with a small linear search "
                  << std::endl;
        // Vxd x_last = ddp.get_xs().back();
        size_t final_index = window_optimize_i;
        // size_t final_index = opti_params.window_shift;
        std::cout << "final index is " << final_index << std::endl;

        double alpha_mpcc = ddp.get_xs().at(final_index)(_nx);
        Vxd x_last = ddp.get_xs().at(final_index);
        last_reaches_ = fun_is_goal(ddp.get_xs().back());

        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "alpha_mpcc:" << alpha_mpcc << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "true last state: " << ddp.get_xs().back().format(FMT)
                  << std::endl;
        std::cout << "distance to goal: " << (x_last.head(_nx) - goal).norm()
                  << std::endl;
        std::cout << "last_reaches_: " << last_reaches_ << std::endl;
        std::cout << "\n**\n";

        if (last_reaches_) {

          auto it = std::find_if(ddp.get_xs().begin(), ddp.get_xs().end(),
                                 [&](const auto &x) { return fun_is_goal(x); });

          assert(it != ddp.get_xs().end());

          index_first_goal = std::distance(ddp.get_xs().begin(), it);
          std::cout << "index first goal " << index_first_goal << std::endl;
        }

        if (std::fabs(alpha_mpcc - times(times.size() - 1)) < 1. &&
            fun_is_goal(x_last)) {

          is_last = true;
          // check which is the first state that is close to goal

          std::cout << "checking first state that reaches the goal "
                    << std::endl;

          auto it = std::find_if(ddp.get_xs().begin(), ddp.get_xs().end(),
                                 [&](const auto &x) { return fun_is_goal(x); });

          assert(it != ddp.get_xs().end());

          window_optimize_i = std::distance(ddp.get_xs().begin(), it);
          std::cout << "changing the number of steps to optimize(copy) to "
                    << window_optimize_i << std::endl;
        }

        std::cout << "checking if i am close to the goal " << std::endl;

        // check 1

        for (size_t i = 0; i < ddp.get_xs().size(); i++) {
          auto &x = ddp.get_xs().at(i);
          if ((x.head(_nx) - goal).norm() < 1e-1) {
            std::cout << "one state is close to goal! " << std::endl;
            close_to_goal = true;
          }

          if (std::fabs(x(_nx) - max_alpha) < 1e-1) {
            std::cout << "alpha is close to final " << std::endl;
            close_to_goal = true;
          }
        }

        std::cout << "done" << std::endl;
      }

      if (is_last)
        copy_steps = window_optimize_i;
      else
        copy_steps = opti_params.window_shift;

      for (size_t i = 0; i < copy_steps; i++)
        xs_opt.push_back(xs_i_sol.at(i + 1).head(_nx));

      for (size_t i = 0; i < copy_steps; i++)
        us_opt.push_back(us_i_sol.at(i).head(2));

      if (solver == SOLVER::mpc) {
        for (size_t i = 0; i < window_optimize_i; i++) {
          xs_init_rewrite.at(1 + counter * opti_params.window_shift + i) =
              xs_i_sol.at(i + 1).head(_nx);

          us_init_rewrite.at(counter * opti_params.window_shift + i) =
              us_i_sol.at(i).head(_nu);
        }
      } else if (solver == SOLVER::mpc_adaptative) {

        xs_warmstart_adptative.clear();
        us_warmstart_adptative.clear();

        for (size_t i = copy_steps; i < window_optimize_i; i++) {
          xs_warmstart_adptative.push_back(xs_i_sol.at(i));
          us_warmstart_adptative.push_back(us_i_sol.at(i));
        }
        xs_warmstart_adptative.push_back(xs_i_sol.back());

      }

      else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        xs_warmstart_mpcc.clear();
        us_warmstart_mpcc.clear();

        for (size_t i = copy_steps; i < window_optimize_i; i++) {
          xs_warmstart_mpcc.push_back(xs_i_sol.at(i));
          us_warmstart_mpcc.push_back(us_i_sol.at(i));
        }
        xs_warmstart_mpcc.push_back(xs_i_sol.back());
      }

      debug_file_yaml << "  - xs0:" << std::endl;
      for (auto &x : xs)
        debug_file_yaml << "    - " << x.format(FMT) << std::endl;

      debug_file_yaml << "    us0:" << std::endl;
      for (auto &u : us)
        debug_file_yaml << "    - " << u.format(FMT) << std::endl;

      debug_file_yaml << "    xsOPT:" << std::endl;
      for (auto &x : xs_i_sol)
        debug_file_yaml << "    - " << x.format(FMT) << std::endl;

      debug_file_yaml << "    usOPT:" << std::endl;
      for (auto &u : us_i_sol)
        debug_file_yaml << "    - " << u.format(FMT) << std::endl;

      debug_file_yaml << "    start: " << xs.front().format(FMT) << std::endl;

      if (solver == SOLVER::mpc || solver == SOLVER::mpc_adaptative) {
        debug_file_yaml << "    goal: " << goal_mpc.format(FMT) << std::endl;
      } else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {
        double alpha_mpcc = ddp.get_xs().back()(_nx);
        Vxd out(_nx);
        Vxd Jout(_nx);
        path->interpolate(alpha_mpcc, out, Jout);
        debug_file_yaml << "    alpha: " << alpha_mpcc << std::endl;
        debug_file_yaml << "    state_alpha: " << out.format(FMT) << std::endl;
      }

      CHECK_EQ(us_i_sol.size() + 1, xs_i_sol.size(), AT);

      // copy results

      if (is_last) {
        finished = true;
        std::cout << "finished: "
                  << "is_last" << std::endl;
      }

      counter++;

      if (counter > opti_params.max_mpc_iterations) {
        finished = true;
        std::cout << "finished: "
                  << "max mpc iterations" << std::endl;
      }
    }
    std::cout << "Total TIME: " << total_time << std::endl;
    std::cout << "Total Iterations: " << total_iterations << std::endl;

    xs_out = xs_opt;
    us_out = us_opt;

    debug_file_yaml << "xsOPT: " << std::endl;
    for (auto &x : xs_out)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    debug_file_yaml << "usOPT: " << std::endl;
    for (auto &u : us_out)
      debug_file_yaml << "  - " << u.format(FMT) << std::endl;

    // checking feasibility

    std::cout << "checking feasibility" << std::endl;
    ptr<Col_cost> feat_col = mk<Col_cost>(_nx, _nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(feat_col)->margin = 0.;
    Vxd goal_last = Vxd::Map(goal.data(), goal.size());

    bool feasible_ = check_feas(feat_col, xs_out, us_out, goal_last);

    ptr<Dynamics> dyn;

    if (name == "unicycle_first_order_0")
      dyn = mk<Dynamics_unicycle>(false);
    else if (name == "unicycle_second_order_0")
      dyn = mk<Dynamics_unicycle2>(false);
    else if (name == "car_first_order_with_1_trailers_0") {
      Vxd l(1);
      l << .5;
      dyn = mk<Dynamics_car_with_trailers>(l, false);
    } else if (name == "quad2d")
      dyn = mk<Dynamics_quadcopter2d>(false);
    else if (name == "quadrotor_0")
      dyn = mk<Dynamics_quadcopter3d>(false);
    else
      CHECK(false, AT);

    bool dynamics_feas = check_dynamics(xs_out, us_out, dyn);

    feasible = feasible_ && dynamics_feas;

    std::cout << "dynamics_feas: " << dynamics_feas << std::endl;

    std::cout << "feasible_ is " << feasible_ << std::endl;
    std::cout << "feasible is " << feasible << std::endl;
  } else if (solver == SOLVER::traj_opt ||
             solver == SOLVER::traj_opt_free_time_proxi) {
    if (solver == SOLVER::traj_opt_free_time_proxi) {
      std::vector<Vxd> us_init_time(us_init.size());
      size_t nu = us_init.front().size();
      for (size_t i = 0; i < N; i++) {
        Vxd u(nu + 1);
        u.head(nu) = us_init.at(i);
        u(nu) = 1.;
        us_init_time.at(i) = u;
      }
      us_init = us_init_time;
    }

    // if reg

    std::vector<Vxd> regs;
    if (opti_params.states_reg && solver == SOLVER::traj_opt) {
      double state_reg_weight = 100.;
      regs = std::vector<Vxd>(xs_init.size() - 1,
                              state_reg_weight * Vxd::Ones(_nx));
    }

    Generate_params gen_args{.free_time =
                                 solver == SOLVER::traj_opt_free_time_proxi,
                             .name = name,
                             .N = N,
                             .goal = goal,
                             .start = start,
                             .cl = cl,
                             .states = {xs_init.begin(), xs_init.end() - 1},
                             .states_weights = regs,
                             .actions = us_init,
                             .collisions = opti_params.collision_weight > 1e-3

    };

    size_t nx, nu;

    ptr<crocoddyl::ShootingProblem> problem =
        generate_problem(gen_args, nx, nu);
    std::vector<Vxd> xs(N + 1, gen_args.start);
    std::vector<Vxd> us(N, Vxd::Zero(nu));

    if (opti_params.use_warmstart) {
      xs = xs_init;
      us = us_init;
    }

    if (opti_params.noise_level > 0.) {
      for (size_t i = 0; i < xs.size(); i++) {
        assert(xs.at(i).size() == nx);
        xs.at(i) += opti_params.noise_level * Vxd::Random(nx);
      }

      for (size_t i = 0; i < us.size(); i++) {
        assert(us.at(i).size() == nu);
        us.at(i) += opti_params.noise_level * Vxd::Random(nu);
      }
    }

    crocoddyl::SolverBoxFDDP ddp(problem);
    ddp.set_th_stop(opti_params.th_stop);
    ddp.set_th_acceptnegstep(opti_params.th_acceptnegstep);

    if (opti_params.CALLBACKS) {
      std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
      cbs.push_back(mk<crocoddyl::CallbackVerbose>());
      ddp.setCallbacks(cbs);
    }

    crocoddyl::Timer timer;

    if (!opti_params.use_finite_diff)
      report_problem(problem, xs, us, "report-0.yaml");
    ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
    std::cout << "time: " << timer.get_duration() << std::endl;
    accumulated_time += timer.get_duration();
    if (!opti_params.use_finite_diff)
      report_problem(problem, ddp.get_xs(), ddp.get_us(), "report-1.yaml");

    // check the distance to the goal:
    ptr<Col_cost> feat_col = mk<Col_cost>(nx, nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(feat_col)->margin = 0.;

    feasible = check_feas(feat_col, ddp.get_xs(), ddp.get_us(), gen_args.goal);
    std::cout << "feasible is " << feasible << std::endl;

    std::cout << "solution" << std::endl;
    std::cout << problem->calc(xs, us) << std::endl;

    if (solver == SOLVER::traj_opt_free_time_proxi) {
      std::vector<Vxd> _xs;
      std::vector<Vxd> _us;

      ptr<Dynamics> dyn;
      if (name == "unicycle_first_order_0")
        dyn = mk<Dynamics_unicycle>(true);
      else if (name == "unicycle_second_order_0")
        dyn = mk<Dynamics_unicycle2>(true);
      else if (name == "car_first_order_with_1_trailers_0") {
        Vxd l(1);
        l << .5;
        dyn = mk<Dynamics_car_with_trailers>(l, true);
      } else if (name == "quad2d") {
        dyn = mk<Dynamics_quadcopter2d>(true);
      } else if (name == "quadrotor_0") {
        dyn = mk<Dynamics_quadcopter3d>(true);
      }

      else
        CHECK(false, AT);

      std::cout << "max error before "
                << max_rollout_error(dyn, ddp.get_xs(), ddp.get_us())
                << std::endl;

      convert_traj_with_variable_time(ddp.get_xs(), ddp.get_us(), _xs, _us, dt);
      xs_out = _xs;
      us_out = _us;

      if (name == "unicycle_first_order_0")
        dyn = mk<Dynamics_unicycle>(false);
      else if (name == "unicycle_second_order_0")
        dyn = mk<Dynamics_unicycle2>(false);
      else if (name == "car_first_order_with_1_trailers_0") {
        Vxd l(1);
        l << .5;
        dyn = mk<Dynamics_car_with_trailers>(l, false);
      } else if (name == "quad2d") {
        dyn = mk<Dynamics_quadcopter2d>(false);
      } else if (name == "quadrotor_0") {
        dyn = mk<Dynamics_quadcopter3d>(false);
      } else
        CHECK(false, AT);

      std::cout << "max error after " << max_rollout_error(dyn, xs_out, us_out)
                << std::endl;

    } else {
      xs_out = ddp.get_xs();
      us_out = ddp.get_us();
    }

    debug_file_yaml << "xsOPT: " << std::endl;
    for (auto &x : xs_out)
      debug_file_yaml << "  - " << x.format(FMT) << std::endl;

    debug_file_yaml << "usOPT: " << std::endl;
    for (auto &u : us_out)
      debug_file_yaml << "  - " << u.format(FMT) << std::endl;
  }

  std::ofstream results_txt("out.txt");

  for (auto &x : xs_out)
    results_txt << x.transpose().format(FMT) << std::endl;

  results_txt << "---" << std::endl;
  for (auto &u : us_out)
    results_txt << u.transpose().format(FMT) << std::endl;

  // store in the good format
  opti_out.feasible = feasible;
  opti_out.xs_out = xs_out;
  opti_out.us_out = us_out;
  opti_out.cost = us_out.size() * dt;
}

void compound_solvers(File_parser_inout file_inout, Result_opti &opti_out) {
  read_from_file(file_inout);
  opti_out.name = file_inout.name;

  std::cout << "file_inout parsed " << std::endl;
  file_inout.print(std::cout);

  CHECK(file_inout.us.size(), AT);
  CHECK(file_inout.xs.size(), AT);
  CHECK_EQ(file_inout.xs.size(), file_inout.us.size() + 1, AT);

  size_t _nx = file_inout.xs.front().size();
  size_t _nu = file_inout.us.front().size();

  switch (static_cast<SOLVER>(opti_params.solver_id)) {

  case SOLVER::mpc_nobound_mpcc: {

    size_t num_solve_mpc_no_bounds = 3;
    size_t num_solve_mpcc_with_bounds = 3;

    bool do_mpcc = true;
    std::cout << "MPC" << std::endl;

    for (size_t i = 0; i < num_solve_mpc_no_bounds; i++) {
      std::cout << "iteration " << i << std::endl;

      opti_params.solver_id = static_cast<int>(SOLVER::mpc);
      opti_params.control_bounds = 0;
      opti_params.debug_file_name =
          "debug_file_mpc_" + std::to_string(i) + ".yaml";

      std::cout << "**\nopti params is " << std::endl;

      opti_params.print(std::cout);

      if (i > 0) {
        file_inout.xs = opti_out.xs_out;
        file_inout.us = opti_out.us_out;
      }

      solve_with_custom_solver(file_inout, opti_out);

      if (!opti_out.feasible) {
        std::cout << "warning"
                  << " "
                  << "infeasible" << std::endl;
        do_mpcc = false;
        break;
      }
    }

    std::cout << "MPCC" << std::endl;
    if (do_mpcc) {
      for (size_t i = 0; i < num_solve_mpcc_with_bounds; i++) {

        std::cout << "iteration " << i << std::endl;
        opti_params.solver_id = static_cast<int>(SOLVER::mpcc);
        opti_params.control_bounds = 1;
        opti_params.debug_file_name =
            "debug_file_mpcc_" + std::to_string(i) + ".yaml";

        file_inout.xs = opti_out.xs_out;
        file_inout.us = opti_out.us_out;

        solve_with_custom_solver(file_inout, opti_out);

        if (!opti_out.feasible) {
          std::cout << "warning"
                    << " "
                    << "infeasible" << std::endl;
          break;
        }
      }
    }
  } break;

  case SOLVER::traj_opt_smooth_then_free_time: {
    // continue here

    bool do_free_time = true;
    opti_params.control_bounds = false;
    opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
    opti_params.control_bounds = 0;
    opti_params.debug_file_name = "debug_file_trajopt.yaml";
    std::cout << "**\nopti params is " << std::endl;
    opti_params.print(std::cout);

    solve_with_custom_solver(file_inout, opti_out);

    if (!opti_out.feasible) {
      std::cout << "warning"
                << " "
                << "infeasible" << std::endl;
      do_free_time = false;
    }

    if (do_free_time) {

      opti_params.control_bounds = true;
      opti_params.solver_id = static_cast<int>(SOLVER::traj_opt_free_time);
      opti_params.control_bounds = 1;
      opti_params.debug_file_name = "debug_file_trajopt_freetime.yaml";

      file_inout.xs = opti_out.xs_out;
      file_inout.us = opti_out.us_out;

      solve_with_custom_solver(file_inout, opti_out);
    }

  } break;

  case SOLVER::time_search_traj_opt: {

    auto check_with_rate = [&](const File_parser_inout &file_inout, double rate,
                               Result_opti &opti_out) {
      double dt = 0.;
      auto name = file_inout.name;

      if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0"}, name))
        dt = .1;
      else if (__in(vstr{"quad2d", "quadrotor_0"}, name))
        dt = .01;
      else
        CHECK(false, AT);

      std::vector<Vxd> us_init = file_inout.us;
      std::vector<Vxd> xs_init = file_inout.xs;

      Vxd times = Vxd::LinSpaced(us_init.size() + 1, 0, us_init.size() * dt);

      // resample a trajectory
      size_t original_n = us_init.size();
      Vxd times_2 = rate * times;

      // create an interpolator
      Interpolator interp_x(times_2, xs_init);
      Interpolator interp_u(times_2.head(us_init.size()), us_init);

      int new_n = std::ceil(rate * original_n);

      Vxd new_times = Vxd::LinSpaced(new_n + 1, 0, new_n * dt);
      std::vector<Vxd> new_xs(new_n + 1);
      std::vector<Vxd> new_us(new_n);

      Vxd x(_nx);
      Vxd Jx(_nx);
      Vxd u(_nu);
      Vxd Ju(_nu);

      for (size_t i = 0; i < new_times.size(); i++) {
        interp_x.interpolate(new_times(i), x, Jx);
        new_xs.at(i) = x;

        if (i < new_times.size() - 1) {
          interp_u.interpolate(new_times(i), u, Ju);
          new_us.at(i) = u;
        }
      }

      File_parser_inout file_next = file_inout;
      file_next.xs = new_xs;
      file_next.us = new_us;

      opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
      solve_with_custom_solver(file_next, opti_out);
    };

    Vxd rates = Vxd::LinSpaced(opti_params.tsearch_num_check,
                               opti_params.tsearch_min_rate,
                               opti_params.tsearch_max_rate);

    Result_opti opti_out_local;
    size_t counter = 0;

    // linear search
    // auto it =
    //     std::find_if(rates.data(), rates.data() + rates.size(), [&](auto
    //     rate) {
    //       std::cout << "checking rate " << rate << std::endl;
    //       opti_params.debug_file_name =
    //           "debug_file_trajopt_" + std::to_string(counter++) + ".yaml";
    //       check_with_rate(file_inout, rate, opti_out_local);
    //       return opti_out_local.feasible;
    //     });

    //  binary search

    Result_opti best;
    best.cost = std::numeric_limits<double>::max();

    auto it = std::lower_bound(
        rates.data(), rates.data() + rates.size(), true,
        [&](auto rate, auto val) {
          std::cout << "checking rate " << rate << std::endl;
          opti_params.debug_file_name =
              "debug_file_trajopt_" + std::to_string(counter++) + ".yaml";
          check_with_rate(file_inout, rate, opti_out_local);
          if (opti_out_local.feasible) {
            assert(opti_out_local.cost <= best.cost);
            best = opti_out_local;
          }
          return !opti_out_local.feasible;
        });

    if (it == rates.data() + rates.size()) {
      std::cout << "all rates are infeasible " << std::endl;
      opti_out.feasible = false;
    } else {
      size_t index = std::distance(rates.data(), it);
      std::cout << "first valid is index " << index << " rate " << rates(index)
                << std::endl;
      opti_out = best;
    }
  } break;

  case SOLVER::traj_opt_free_time: {

    bool do_final_repair_step = true;
    opti_params.control_bounds = true;
    opti_params.solver_id = static_cast<int>(SOLVER::traj_opt_free_time_proxi);
    opti_params.control_bounds = 1;
    opti_params.debug_file_name = "debug_file_trajopt_freetime_proxi.yaml";
    std::cout << "**\nopti params is " << std::endl;
    opti_params.print(std::cout);

    solve_with_custom_solver(file_inout, opti_out);

    if (!opti_out.feasible) {
      std::cout << "warning"
                << " "
                << "infeasible" << std::endl;
      do_final_repair_step = false;
    }

    if (do_final_repair_step) {

      opti_params.control_bounds = true;
      opti_params.solver_id = static_cast<int>(SOLVER::traj_opt);
      opti_params.control_bounds = 1;
      opti_params.debug_file_name =
          "debug_file_trajopt_after_freetime_proxi.yaml";

      file_inout.xs = opti_out.xs_out;
      file_inout.us = opti_out.us_out;

      solve_with_custom_solver(file_inout, opti_out);
    }
  }

  break;

  default: {
    solve_with_custom_solver(file_inout, opti_out);
  }
  }
}

;

void Result_opti::write_yaml(std::ostream &out) {
  out << "feasible: " << feasible << std::endl;
  out << "cost: " << cost << std::endl;

  out << "xs_out: " << std::endl;
  for (auto &x : xs_out)
    out << "  - " << x.format(FMT) << std::endl;

  out << "us_out: " << std::endl;
  for (auto &u : us_out)
    out << "  - " << u.format(FMT) << std::endl;
}

void Result_opti::write_yaml_db(std::ostream &out) {
  CHECK((name != ""), AT);
  out << "feasible: " << feasible << std::endl;
  out << "cost: " << cost << std::endl;
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (auto &x : xs_out) {
    if (__in(vstr{"unicycle_first_order_0", "unicycle_second_order_0",
                  "car_first_order_with_1_trailers_0"},
             name)) {
      x(2) = std::remainder(x(2), 2 * M_PI);
    }
    out << "      - " << x.format(FMT) << std::endl;
  }

  out << "    actions:" << std::endl;
  for (auto &u : us_out) {
    out << "      - " << u.format(FMT) << std::endl;
  }
};

void File_parser_inout::add_options(po::options_description &desc) {
  // desc.add_options()("env", po::value<std::string>(&env_file)->required())(
  //     "waypoints", po::value<std::string>(&init_guess)->required())(
  //     "new_format",
  //     po::value<bool>(&new_format)->default_value(new_format));
  set_from_boostop(desc, VAR_WITH_NAME(init_guess));
  set_from_boostop(desc, VAR_WITH_NAME(env_file));
  set_from_boostop(desc, VAR_WITH_NAME(new_format));
  set_from_boostop(desc, VAR_WITH_NAME(problem_name));
}

void File_parser_inout::read_from_yaml(const char *file) {
  std::cout << "loading file: " << file << std::endl;
  YAML::Node node = YAML::LoadFile(file);
  read_from_yaml(node);
}

void File_parser_inout::read_from_yaml(YAML::Node &node) {
  set_from_yaml(node, VAR_WITH_NAME(env_file));
  set_from_yaml(node, VAR_WITH_NAME(init_guess));
  set_from_yaml(node, VAR_WITH_NAME(new_format));
  set_from_yaml(node, VAR_WITH_NAME(problem_name));
}

void File_parser_inout::print(std::ostream &out) {
  std::string be = "";
  std::string af = ": ";
  out << be << STR(init_guess, af) << std::endl;
  out << be << STR(problem_name, af) << std::endl;
  out << be << STR(env_file, af) << std::endl;
  out << be << STR(new_format, af) << std::endl;
  out << be << STR(name, af) << std::endl;
  out << be << STR(cl, af) << std::endl;

  out << be << "xs" << std::endl;
  for (const auto &s : xs)
    out << "  - " << s.format(FMT) << std::endl;

  out << be << "us" << std::endl;
  for (const auto &s : us)
    out << "  - " << s.format(FMT) << std::endl;

  out << be << "start" << af << start.format(FMT) << std::endl;
  out << be << "goal" << af << start.format(FMT) << std::endl;
}

Dynamics_car_with_trailers::Dynamics_car_with_trailers(const Vxd &hitch_lengths,
                                                       bool free_time)
    : num_trailers(hitch_lengths.size()), hitch_lengths(hitch_lengths),
      free_time(free_time) {
  nx = 3 + num_trailers;
  nu = 2;

  if (free_time)
    nu += 1;
}

void Dynamics_car_with_trailers::calc(Eigen::Ref<Eigen::VectorXd> xnext,
                                      const Eigen::Ref<const VectorXs> &x,
                                      const Eigen::Ref<const VectorXs> &u) {
  check_input_calc(xnext, x, u, nx, nu);
  double dt_ = dt;
  if (free_time)
    dt_ *= u[2];

  const double &v = u(0);
  const double &phi = u(1);
  const double &yaw = x(2);

  const double &c = std::cos(yaw);
  const double &s = std::sin(yaw);

  xnext(0) = x(0) + dt_ * v * c;
  xnext(1) = x(1) + dt_ * v * s;
  xnext(2) = x(2) + dt_ * v / l * std::tan(phi);

  if (num_trailers) {
    CHECK_EQ(num_trailers, 1, AT);
    double d = hitch_lengths(0);
    double theta_dot = v / d;
    theta_dot *= std::sin(x(2) - x(3));
    xnext(3) = x(3) + theta_dot * dt_;
  }
}

void Dynamics_car_with_trailers::calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                          Eigen::Ref<Eigen::MatrixXd> Fu,
                                          const Eigen::Ref<const VectorXs> &x,
                                          const Eigen::Ref<const VectorXs> &u) {
  check_input_calcdiff(Fx, Fu, x, u, nx, nu);
  // CHECK_EQ(free_time, false, AT);

  const double &v = u(0);
  const double &phi = u(1);
  const double &yaw = x(2);

  const double &c = std::cos(yaw);
  const double &s = std::sin(yaw);

  double dt_ = dt;
  if (free_time)
    dt_ *= u[2];

  Fx.setIdentity();
  Fx(0, 2) = -dt_ * v * s;
  Fx(1, 2) = dt_ * v * c;

  Fu(0, 0) = dt_ * c;
  Fu(1, 0) = dt_ * s;
  Fu(2, 0) = dt_ / l * std::tan(phi);
  Fu(2, 1) = dt_ * v / l / (std::cos(phi) * std::cos(phi));

  if (free_time) {
    Fu(0, 2) = dt * v * c;
    Fu(1, 2) = dt * v * s;
    Fu(2, 2) = dt * v / l * std::tan(phi);
  }

  if (num_trailers) {
    CHECK_EQ(num_trailers, 1, AT);
    double d = hitch_lengths(0);
    // double theta_dot = v / d;
    // double theta_dot =  v / d * std::sin(x(2) - x(3));
    // xnext(3) = x(3) + theta_dot * dt_;
    Fx(3, 2) = dt_ * v / d * std::cos(x(2) - x(3));
    Fx(3, 3) -= dt_ * v / d * std::cos(x(2) - x(3));
    Fu(3, 0) = dt_ * std::sin(x(2) - x(3)) / d;
    if (free_time) {
      Fu(3, 2) = dt * v / d * std::sin(x(2) - x(3));
    }
  }
}

Dynamics_quadcopter2d::Dynamics_quadcopter2d(bool free_time)
    : free_time(free_time) {
  nx = 6;
  nu = 2;
  if (free_time)
    nu += 1;
}

void Dynamics_quadcopter2d::compute_acc(const Eigen::Ref<const VectorXs> &x,
                                        const Eigen::Ref<const VectorXs> &u) {

  const double &f1 = u_nominal * u(0);
  const double &f2 = u_nominal * u(1);
  const double &c = std::cos(x(2));
  const double &s = std::sin(x(2));

  const double &xdot = x(3);
  const double &ydot = x(4);
  const double &thetadot = x(5);

  data.xdotdot = -m_inv * (f1 + f2) * s;
  data.ydotdot = m_inv * (f1 + f2) * c - g;
  data.thetadotdot = l * I_inv * (f1 - f2);

  if (drag_against_vel) {
    data.xdotdot -= m_inv * k_drag_linear * xdot;
    data.ydotdot -= m_inv * k_drag_linear * ydot;
    data.thetadotdot -= I_inv * k_drag_angular * thetadot;
  }
}

void Dynamics_quadcopter2d::calc(Eigen::Ref<Eigen::VectorXd> xnext,
                                 const Eigen::Ref<const VectorXs> &x,
                                 const Eigen::Ref<const VectorXs> &u) {
  check_input_calc(xnext, x, u, nx, nu);

  compute_acc(x, u);

  double dt_ = dt;
  if (free_time)
    dt_ *= u(2);

  xnext.head(3) = x.head(3) + dt_ * x.segment(3, 3);
  xnext.segment(3, 3) << x(3) + dt_ * data.xdotdot, x(4) + dt_ * data.ydotdot,
      x(5) + dt_ * data.thetadotdot;
}

void Dynamics_quadcopter2d::calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                     Eigen::Ref<Eigen::MatrixXd> Fu,
                                     const Eigen::Ref<const Vxd> &x,
                                     const Eigen::Ref<const Vxd> &u) {
  check_input_calcdiff(Fx, Fu, x, u, nx, nu);
  compute_acc(x, u);

  const double &f1 = u_nominal * u(0);
  const double &f2 = u_nominal * u(1);
  const double &c = std::cos(x(2));
  const double &s = std::sin(x(2));

  double dt_ = dt;
  if (free_time)
    dt_ *= u(2);

  Fx.setIdentity();
  Fx.block(0, 3, 3, 3).setIdentity();
  Fx.block(0, 3, 3, 3) *= dt_;

  const double &d_xdotdot_dtheta = -m_inv * (f1 + f2) * c;
  const double &d_ydotdot_dtheta = m_inv * (f1 + f2) * (-s);

  Fx(3, 2) = dt_ * d_xdotdot_dtheta;
  Fx(4, 2) = dt_ * d_ydotdot_dtheta;

  Fu(3, 0) = -dt_ * m_inv * s * u_nominal;
  Fu(3, 1) = -dt_ * m_inv * s * u_nominal;

  Fu(4, 0) = dt_ * m_inv * c * u_nominal;
  Fu(4, 1) = dt_ * m_inv * c * u_nominal;

  Fu(5, 0) = dt_ * l * I_inv * u_nominal;
  Fu(5, 1) = -dt_ * l * I_inv * u_nominal;

  if (drag_against_vel) {
    Fx(3, 3) -= m_inv * dt_ * k_drag_linear;
    Fx(4, 4) -= m_inv * dt_ * k_drag_linear;
    Fx(5, 5) -= I_inv * dt_ * k_drag_angular;
  }

  if (free_time) {
    Fu.col(2) << dt * x.segment(3, 3), dt * data.xdotdot, dt * data.ydotdot,
        dt * data.thetadotdot;
  }
}

void quat_product(const Eigen::Vector4d &p, const Eigen::Vector4d &q,
                  Eigen::Ref<Eigen::VectorXd> out,
                  Eigen::Ref<Eigen::Matrix4d> Jp,
                  Eigen::Ref<Eigen::Matrix4d> Jq) {

  const double px = p(0);
  const double py = p(1);
  const double pz = p(2);
  const double pw = p(3);

  const double qx = q(0);
  const double qy = q(1);
  const double qz = q(2);
  const double qw = q(3);

  out(0) = pw * qx + px * qw + py * qz - pz * qy;
  out(1) = pw * qy - px * qz + py * qw + pz * qx;
  out(2) = pw * qz + px * qy - py * qx + pz * qw;
  out(3) = pw * qw - px * qx - py * qy - pz * qz;

  Eigen::Vector4d dyw_dq{-px, -py, -pz, pw};
  Eigen::Vector4d dyx_dq{pw, -pz, py, px};
  Eigen::Vector4d dyy_dq{pz, pw, -px, py};
  Eigen::Vector4d dyz_dq{-py, px, pw, pz};

  Jq.row(0) = dyx_dq;
  Jq.row(1) = dyy_dq;
  Jq.row(2) = dyz_dq;
  Jq.row(3) = dyw_dq;

  Eigen::Vector4d dyw_dp{-qx, -qy, -qz, qw};
  Eigen::Vector4d dyx_dp{qw, qz, -qy, qx};
  Eigen::Vector4d dyy_dp{-qz, qw, qx, qy};
  Eigen::Vector4d dyz_dp{qy, -qx, qw, qz};

  Jp.row(0) = dyx_dp;
  Jp.row(1) = dyy_dp;
  Jp.row(2) = dyz_dp;
  Jp.row(3) = dyw_dp;
}

Dynamics_quadcopter3d::Dynamics_quadcopter3d(bool free_time)
    : free_time(free_time) {
  nx = 13;
  nu = 4;
  CHECK_EQ(free_time, false, AT);

  B0 << 1, 1, 1, 1, -arm, -arm, arm, arm, -arm, arm, arm, -arm, -t2t, t2t, -t2t,
      t2t;

  Fu_selection.setZero();
  Fu_selection(2, 0) = 1.;

  // [ 0, 0, 0, 0]   [eta(0)]    =
  // [ 0, 0, 0, 0]   [eta(1)]
  // [ 1, 0, 0, 0]   [eta(2)]
  //                 [eta(3)]

  Ftau_selection.setZero();
  Ftau_selection(0, 1) = 1.;
  Ftau_selection(1, 2) = 1.;
  Ftau_selection(2, 3) = 1.;

  // [ 0, 1, 0, 0]   [eta(0)]    =
  // [ 0, 0, 1, 0]   [eta(1)]
  // [ 0, 0, 0, 1]   [eta(2)]
  //                 [eta(3)]

  Fu_selection_B0 = Fu_selection * B0;
  Ftau_selection_B0 = Ftau_selection * B0;
}

// def integrate_quat(self, q, wb, dt):
//     return multiply(q, exp(_promote_vec(wb * dt / 2)))

Eigen::Quaterniond
get_quat_from_ang_vel_time(const Eigen::Vector3d &angular_rotation) {

  Eigen::Quaterniond deltaQ;
  auto theta = angular_rotation * 0.5;
  float thetaMagSq = theta.squaredNorm();
  float s;
  if (thetaMagSq * thetaMagSq / 24.0 < std::numeric_limits<float>::min()) {
    deltaQ.w() = 1.0 - thetaMagSq / 2.0;
    s = 1.0 - thetaMagSq / 6.0;
  } else {
    float thetaMag = std::sqrt(thetaMagSq);
    deltaQ.w() = std::cos(thetaMag);
    // Real part:
    // cos ( thetaMag )   = cos (sqrt(  || omega * dt * .5 ||^2 )  =  cos( ||
    // omega * dt  || * .5)
    s = std::sin(thetaMag) / thetaMag;
  }
  // Imaginary part
  //  omega * dt . 5 /  (||omega * dt || * .5) * sin(  || omega * dt  || * .5
  //  ) = omega * dt  /  ||omega * dt ||  * sin(  || omega * dt  || * .5 )
  deltaQ.x() = theta.x() * s;
  deltaQ.y() = theta.y() * s;
  deltaQ.z() = theta.z() * s;

  return deltaQ;
}

Eigen::Quaterniond qintegrate(const Eigen::Quaterniond &q,
                              const Eigen::Vector3d &omega, float dt) {
  Eigen::Quaterniond deltaQ = get_quat_from_ang_vel_time(omega * dt);
  return q * deltaQ;
};

void Dynamics_quadcopter3d::calc(Eigen::Ref<Eigen::VectorXd> xnext,
                                 const Eigen::Ref<const VectorXs> &x,
                                 const Eigen::Ref<const VectorXs> &u) {

  check_input_calc(xnext, x, u, nx, nu);
  Eigen::Vector4d f = u_nominal * u;
  // const double &f1 = u_nominal * u(0);
  // const double &f2 = u_nominal * u(1);

  CHECK_EQ(free_time, false, AT);
  Eigen::Vector3d f_u;
  Eigen::Vector3d tau_u;

  if (force_control) {
    Eigen::Vector4d eta = B0 * f;
    f_u << 0, 0, eta(0);

    // f_u
    // [ 0, 0, 0, 0]   [eta(0)]    =
    // [ 0, 0, 0, 0]   [eta(1)]
    // [ 1, 0, 0, 0]   [eta(2)]
    //                 [eta(3)]

    // tau_u

    // [ 0, 1, 0, 0]   [eta(0)]    =
    // [ 0, 0, 1, 0]   [eta(1)]
    // [ 0, 0, 0, 1]   [eta(2)]
    //                 [eta(3)]

    tau_u << eta(1), eta(2), eta(3);
  } else {
    CHECK(false, AT);
  }

  Eigen::Vector3d pos = x.head(3).head<3>();
  Eigen::Vector4d q = x.segment(3, 4).head<4>().normalized();
  Eigen::Vector3d vel = x.segment(7, 3).head<3>();
  Eigen::Vector3d w = x.segment(10, 3).head<3>();

  Eigen::Ref<V3d> pos_next = xnext.head(3);
  Eigen::Ref<V4d> q_next = xnext.segment(3, 4);
  Eigen::Ref<V3d> vel_next = xnext.segment(7, 3);
  Eigen::Ref<V3d> w_next = xnext.segment(10, 3);

  auto fa_v = Eigen::Vector3d(0, 0, 0); // drag model

  Eigen::Vector3d a =
      m_inv * (grav_v + Eigen::Quaterniond(q)._transformVector(f_u) + fa_v);
  // easy to get derivatives :)
  // see (174)
  // and 170
  //
  // d a / d fu = m_inv * R

  pos_next = pos + dt * vel;
  vel_next = vel + dt * a;

  // w is in body frame.
  Eigen::Vector4d __q_next = qintegrate(Eigen::Quaterniond(q), w, dt).coeffs();
  q_next = __q_next;

  // derivative w.r.t w?

  w_next =
      w + dt * inverseJ_v.cwiseProduct((J_v.cwiseProduct(w)).cross(w) + tau_u);

  // dw_next / d_tau = dt *  inverseJ_M

  // derivative of cross product

  // d / dx [ a x b ] = da / dx x b + a x db / dx

  // d / dx [ Jw x w ] = J x b + a x db / dx

  // J ( A x B ) = Skew(A) * JB - Skew(B) JA

  // J ( Kw x w ) = Skew(k w) * Id - Skew(w) * K

  // dw_next /  tau = inverseJ_
}

void Dynamics_quadcopter3d::calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                                     Eigen::Ref<Eigen::MatrixXd> Fu,
                                     const Eigen::Ref<const VectorXs> &x,
                                     const Eigen::Ref<const VectorXs> &u) {

  // std::cout << "dif" << std::endl;
  check_input_calcdiff(Fx, Fu, x, u, nx, nu);

  CHECK_EQ(free_time, false, AT);

  Eigen::Vector4d f = u_nominal * u;

  // todo: refactor this
  if (force_control) {
    Eigen::Vector4d eta = B0 * f;
    data.f_u << 0, 0, eta(0);
    data.tau_u << eta(1), eta(2), eta(3);
  } else {
    CHECK(false, AT);
  }

  Eigen::Vector4d q = x.segment(3, 4).head<4>().normalized();

  // lets do some parts analytically.
  Fx.block<3, 3>(0, 0).diagonal() = Eigen::Vector3d::Ones();
  Fx.block<3, 3>(0, 7).diagonal() = dt * Eigen::Vector3d::Ones();
  Fx.block<3, 3>(7, 7).diagonal() = Eigen::Vector3d::Ones();

  // I compute with finite diff only the quaternions, the w and the u's

  // quaternion
  // for (size_t i = 3; i < 7; i++) {
  //   Eigen::MatrixXd xe;
  //   xe = x;
  //   xe(i) += eps;
  //   Eigen::VectorXd xnexte(nx);
  //   xnexte.setZero();
  //   calc(xnexte, xe, u);
  //   auto df = (xnexte - xnext) / eps;
  //   Fx.col(i) = df;
  // }

  // auto &&a =
  //     m_inv * (grav_v + Eigen::Quaterniond(q)._transformVector(f_u) +
  //     fa_v);

  Eigen::Vector3d y;
  const Eigen::Vector4d &xq = x.segment<4>(3);

  rotate_with_q(xq, data.f_u, y, data.Jx, data.Ja);

  Fx.block<3, 4>(7, 3).noalias() = dt * m_inv * data.Jx;

  const Eigen::Vector3d &w = x.segment<3>(10);

  // q_next = qintegrate(Eigen::Quaterniond(q), w, dt).coeffs();
  Eigen::Quaterniond deltaQ = get_quat_from_ang_vel_time(w * dt);
  Eigen::Vector4d xq_normlized;
  Eigen::Matrix4d Jqnorm;
  Eigen::Matrix4d J1;
  Eigen::Matrix4d J2;
  Eigen::Vector4d yy;
  normalize(xq, xq_normlized, Jqnorm);
  quat_product(xq_normlized, deltaQ.coeffs(), yy, J1, J2);

  Fx.block<4, 4>(3, 3).noalias() = J1 * Jqnorm;

  // angular velocity
  // for (size_t i = 10; i < 13; i++) {
  //   Eigen::MatrixXd xe;
  //   xe = x;
  //   xe(i) += eps;
  //   Eigen::VectorXd xnexte(nx);
  //   xnexte.setZero();
  //   calc(xnexte, xe, u);
  //   auto df = (xnexte - xnext) / eps;
  //   Fx.col(i) = df;
  // }

  // w_next =
  //     w + dt * inverseJ_v.cwiseProduct((J_v.cwiseProduct(w)).cross(w) +
  //     tau_u);

  // dw_next / d_tau = dt *  inverseJ_M

  // derivative of cross product

  // d / dx [ a x b ] = da / dx x b + a x db / dx

  // d / dx [ Jw x w ] = J x b + a x db / dx

  // J ( A x B ) = Skew(A) * JB - Skew(B) JA

  // J ( Kw x w ) = Skew(k w) * Id - Skew(w) * K

  finite_diff_jac(
      [&](const Eigen::VectorXd &x, Eigen::Ref<Eigen::VectorXd> y) {
        y = Eigen::Vector4d(qintegrate(Eigen::Quaterniond(q), x, dt).coeffs());
      },
      w, 4, Fx.block<4, 3>(3, 10), 1e-5);

  Fx.block<3, 3>(10, 10).diagonal().setOnes();
  Fx.block<3, 3>(10, 10).noalias() +=
      dt * inverseJ_M * (Skew(J_v.cwiseProduct(w)) - Skew(w) * J_M);

  Eigen::Matrix3d R = Eigen::Quaterniond(q).toRotationMatrix();

  Fu.block<3, 4>(7, 0).noalias() = u_nominal * dt * m_inv * R * Fu_selection_B0;
  Fu.block<3, 4>(10, 0).noalias() =
      u_nominal * dt * inverseJ_M * Ftau_selection_B0;
}

Quaternion_cost::Quaternion_cost(size_t nx, size_t nu) : Cost(nx, nu, 1){};

void Quaternion_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                           const Eigen::Ref<const Eigen::VectorXd> &x) {
  CHECK_GEQ(k_quat, 0., AT);
  Eigen::Vector4d q = x.segment<4>(3);
  r(0) = k_quat * (q.squaredNorm() - 1);
}

void Quaternion_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                           const Eigen::Ref<const Eigen::VectorXd> &x,
                           const Eigen::Ref<const Eigen::VectorXd> &u) {
  calc(r, x);
}

void Quaternion_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                               const Eigen::Ref<const Eigen::VectorXd> &x) {

  Eigen::Vector4d q = x.segment<4>(3);
  Jx.row(0).segment<4>(3) = (2. * k_quat) * q;
}

void Quaternion_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                               Eigen::Ref<Eigen::MatrixXd> Ju,
                               const Eigen::Ref<const Eigen::VectorXd> &x,
                               const Eigen::Ref<const Eigen::VectorXd> &u) {
  calcDiff(Jx, x);
}
