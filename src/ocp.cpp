#include "ocp.hpp"
#include <boost/test/tools/interface.hpp>

Opti_params opti_params;
double accumulated_time;

void Opti_params::add_options(po::options_description &desc) {

  desc.add_options()("free_time",
                     po::value<bool>(&free_time)->default_value(free_time))(
      "control_bounds",
      po::value<bool>(&control_bounds)->default_value(control_bounds))(
      "max_iter", po::value<size_t>(&max_iter)->default_value(max_iter))(
      "step_opt", po::value<size_t>(&num_steps_to_optimize)
                      ->default_value(num_steps_to_optimize))(
      "step_move",
      po::value<size_t>(&num_steps_to_move)->default_value(num_steps_to_move))(
      "solver", po::value<int>(&solver_id)->default_value(solver_id))(
      "use_warmstart",
      po::value<bool>(&use_warmstart)->default_value(use_warmstart))(
      "use_fdiff",
      po::value<bool>(&use_finite_diff)->default_value(use_finite_diff))(
      "alpha_rate", po::value<double>(&alpha_rate)->default_value(alpha_rate))(
      "k_linear", po::value<double>(&k_linear)->default_value(k_linear))(
      "noise_level",
      po::value<double>(&noise_level)->default_value(noise_level))(
      "smooth", po::value<bool>(&smooth_traj)->default_value(smooth_traj))(
      "k_contour", po::value<double>(&k_contour)->default_value(k_contour))(
      "weight_goal",
      po::value<double>(&weight_goal)->default_value(weight_goal))(
      "reg", po::value<bool>(&regularize_wrt_init_guess)
                 ->default_value(regularize_wrt_init_guess));
}

void Opti_params::print(std::ostream &out) {

  std::string be = "";
  std::string af = ": ";
  out << be << "CALLBACKS" << af << CALLBACKS << std::endl;
  out << be << "use_finite_diff" << af << use_finite_diff << std::endl;
  out << be << "use_warmstart" << af << use_warmstart << std::endl;
  out << be << "free_time" << af << free_time << std::endl;
  out << be << "repair_init_guess" << af << repair_init_guess << std::endl;
  out << be << "regularize_wrt_init_guess" << af << regularize_wrt_init_guess
      << std::endl;
  out << be << "control_bounds" << af << control_bounds << std::endl;
  out << be << "adaptative_goal_mpc" << af << adaptative_goal_mpc << std::endl;
  out << be << "th_stop" << af << th_stop << std::endl;
  out << be << "init_reg" << af << init_reg << std::endl;
  out << be << "th_acceptnegstep" << af << th_acceptnegstep << std::endl;
  out << be << "noise_level" << af << noise_level << std::endl;
  out << be << "alpha_rate" << af << alpha_rate << std::endl;
  out << be << "max_iter" << af << max_iter << std::endl;
  out << be << "num_steps_to_optimize" << af << num_steps_to_optimize
      << std::endl;
  out << be << "num_steps_to_move" << af << num_steps_to_move << std::endl;
  out << be << "max_mpc_iterations" << af << max_mpc_iterations << std::endl;
  out << be << "debug_file_name" << af << debug_file_name << std::endl;
  out << be << STR(k_linear, af) << std::endl;
  out << be << STR(k_contour, af) << std::endl;
  out << be << STR(weight_goal, af) << std::endl;
  out << be << STR(collision_weight, af) << std::endl;
  out << be << STR(smooth_traj, af) << std::endl;
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
                            "none"};

void linearInterpolation(const Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &x, double t_query,
                         Eigen::Ref<Eigen::VectorXd> out,
                         Eigen::Ref<Eigen::VectorXd> Jx) {

  double num_tolerance = 1e-8;
  CHECK_GEQ(t_query + num_tolerance, times.head(1)(0), AT);
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

  std::cout << "index is " << index << std::endl;
  std::cout << "size " << times.size() << std::endl;
  // std::cout << "factor " << factor << std::endl;

  out = x.at(index - 1) + factor * (x.at(index) - x.at(index - 1));
  Jx = (x.at(index) - x.at(index - 1)) / (times(index) - times(index - 1));
}

Dynamics_contour::Dynamics_contour(ptr<Dynamics> dyn, bool accumulate)
    : dyn(dyn), accumulate(accumulate) {
  nx = dyn->nx + 1;
  nu = dyn->nu + 1;
}

void Dynamics_contour::calc(Eigen::Ref<Eigen::VectorXd> xnext,
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
}

void Dynamics_unicycle2::calc(Eigen::Ref<Eigen::VectorXd> xnext,
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
}

void Dynamics_unicycle::calc(Eigen::Ref<Eigen::VectorXd> xnext,
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

void Contour_cost_alpha_x::calc(Eigen::Ref<Eigen::VectorXd> r,
                                const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);
  assert(k > 0);

  r(0) = -k * x(nx - 1);
}

void Contour_cost_alpha_x::calcDiff(
    Eigen::Ref<Eigen::MatrixXd> Jx, Eigen::Ref<Eigen::MatrixXd> Ju,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

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

void Contour_cost_alpha_u::calc(Eigen::Ref<Eigen::VectorXd> r,
                                const Eigen::Ref<const Eigen::VectorXd> &x,
                                const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);
  assert(k > 0);

  r(0) = -k * u(nu - 1);
}

void Contour_cost_alpha_u::calcDiff(
    Eigen::Ref<Eigen::MatrixXd> Jx, Eigen::Ref<Eigen::MatrixXd> Ju,
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  Ju(0, nu - 1) = -k;
};

void finite_diff_cost(ptr<Cost> cost, Eigen::Ref<Eigen::MatrixXd> Jx,
                      Eigen::Ref<Eigen::MatrixXd> Ju, const Eigen::VectorXd &x,
                      const Eigen::VectorXd &u, const int nr) {

  Eigen::VectorXd r_ref(nr);
  cost->calc(r_ref, x, u);
  int nu = u.size();
  int nx = x.size();

  Ju.setZero();

  double eps = 1e-5;
  for (size_t i = 0; i < nu; i++) {
    Eigen::MatrixXd ue;
    ue = u;
    ue(i) += eps;
    Eigen::VectorXd r_e(nr);
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
    Eigen::VectorXd r_e(nr);
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
void Contour_cost_x::calc(Eigen::Ref<Eigen::VectorXd> r,
                          const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &u) {
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

void Contour_cost_x::calc(Eigen::Ref<Eigen::VectorXd> r,
                          const Eigen::Ref<const Eigen::VectorXd> &x) {
  calc(r, x, zero_u);
}

void Contour_cost_x::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                              Eigen::Ref<Eigen::MatrixXd> Ju,
                              const Eigen::Ref<const Eigen::VectorXd> &x,
                              const Eigen::Ref<const Eigen::VectorXd> &u) {

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

    Jx.block(0, 0, nx - 1, nx - 1).diagonal() =
        -weight * Eigen::VectorXd::Ones(nx - 1);
    Jx.block(0, nx - 1, nx - 1, 1) = weight * last_J;
  }
};

void Contour_cost_x::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                              const Eigen::Ref<const Eigen::VectorXd> &x) {

  calcDiff(Jx, zero_Ju, x, zero_u);
}

Contour_cost::Contour_cost(size_t nx, size_t nu, ptr<Interpolator> path)
    : Cost(nx, nu, nx + 1), path(path), last_query(-1.), last_out(nx - 1),
      last_J(nx - 1) {
  name = "contour";
}

void Contour_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {
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

void Contour_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {
  calc(r, x, zero_u);
}

void Contour_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            Eigen::Ref<Eigen::MatrixXd> Ju,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);

  // lets use finite diff
  if (use_finite_diff) {
    Eigen::VectorXd r_ref(nr);
    calc(r_ref, x, u);

    Ju.setZero();

    double eps = 1e-5;
    for (size_t i = 0; i < nu; i++) {
      Eigen::MatrixXd ue;
      ue = u;
      ue(i) += eps;
      Eigen::VectorXd r_e(nr);
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
      Eigen::VectorXd r_e(nr);
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
        -weight_contour * weight_diff * Eigen::VectorXd::Ones(nx - 1);
    Jx.block(0, nx - 1, nx - 1, 1) = weight_contour * weight_diff * last_J;
    Jx(nx - 1, nx - 1) = weight_contour * weight_alpha;
    Ju(nx, nu - 1) = weight_virtual_control;
  }
};

void Contour_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            const Eigen::Ref<const Eigen::VectorXd> &x) {

  calcDiff(Jx, zero_Ju, x, zero_u);
}

Col_cost::Col_cost(size_t nx, size_t nu, size_t nr,
                   boost::shared_ptr<CollisionChecker> cl)
    : Cost(nx, nu, nr), cl(cl) {
  last_x = Eigen::VectorXd::Zero(nx);
  name = "collision";
  nx_effective = nx;
}

void Col_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
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

void Col_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {
  calc(r, x, Eigen::VectorXd(nu));
}

void Col_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  std::vector<double> query{x.data(), x.data() + nx_effective};
  double raw_d, d;
  Eigen::VectorXd v(nx);
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
    v = opti_params.collision_weight *
        Eigen::VectorXd::Map(grad.data(), grad.size());
    if (d <= 0) {
      Jx.block(0, 0, 1, nx_effective) = v.transpose();
    } else {
      Jx.setZero();
    }
  }
  Ju.setZero();
};

void Col_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

  auto Ju = Eigen::MatrixXd(1, 1);
  auto u = Eigen::VectorXd(1);
  calcDiff(Jx, Ju, x, u);
}

Control_cost::Control_cost(size_t nx, size_t nu, size_t nr,
                           const Eigen::VectorXd &u_weight,
                           const Eigen::VectorXd &u_ref)
    : Cost(nx, nu, nr), u_weight(u_weight), u_ref(u_ref) {
  CHECK_EQ(u_weight.size(), nu, AT);
  CHECK_EQ(u_ref.size(), nu, AT);
  CHECK_EQ(nu, nr, AT);
  name = "control";
}

void Control_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);
  r = (u - u_ref).cwiseProduct(u_weight);
}

void Control_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

  auto u = Eigen::VectorXd::Zero(nu);
  calc(r, x, u);
}

void Control_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            Eigen::Ref<Eigen::MatrixXd> Ju,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(x.size()) == nx);
  assert(static_cast<std::size_t>(u.size()) == nu);

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);
  Ju = u_weight.asDiagonal();
  Jx.setZero();
}

void Control_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            const Eigen::Ref<const Eigen::VectorXd> &x) {

  Eigen::VectorXd u(0);
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

State_bounds::State_bounds(size_t nx, size_t nu, size_t nr,
                           const Eigen::VectorXd &ub,
                           const Eigen::VectorXd &weight)
    : Cost(nx, nu, nr), ub(ub), weight(weight) {
  name = "state";
}

void State_bounds::calc(Eigen::Ref<Eigen::VectorXd> r,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  calc(r, x);
}

void State_bounds::calc(Eigen::Ref<Eigen::VectorXd> r,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  r = (x - ub).cwiseProduct(weight).cwiseMax(0.);
}

void State_bounds::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            Eigen::Ref<Eigen::MatrixXd> Ju,
                            const Eigen::Ref<const Eigen::VectorXd> &x,
                            const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  calcDiff(Jx, x);
}

void State_bounds::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                            const Eigen::Ref<const Eigen::VectorXd> &x) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);

  Eigen::Matrix<bool, Eigen::Dynamic, 1> result = (x - ub).array() >= 0;
  Jx.diagonal() = (result.cast<double>()).cwiseProduct(weight);
}

State_cost::State_cost(size_t nx, size_t nu, size_t nr,
                       const Eigen::VectorXd &x_weight,
                       const Eigen::VectorXd &ref)
    : Cost(nx, nu, nr), x_weight(x_weight), ref(ref) {
  name = "state";
  assert(static_cast<std::size_t>(x_weight.size()) == nx);
  assert(static_cast<std::size_t>(ref.size()) == nx);
}

void State_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                      const Eigen::Ref<const Eigen::VectorXd> &x,
                      const Eigen::Ref<const Eigen::VectorXd> &u) {
  // check that r
  assert(static_cast<std::size_t>(r.size()) == nr);
  r = (x - ref).cwiseProduct(x_weight);
}

void State_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                      const Eigen::Ref<const Eigen::VectorXd> &x) {

  assert(static_cast<std::size_t>(r.size()) == nr);
  r = (x - ref).cwiseProduct(x_weight);
}

void State_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                          Eigen::Ref<Eigen::MatrixXd> Ju,
                          const Eigen::Ref<const Eigen::VectorXd> &x,
                          const Eigen::Ref<const Eigen::VectorXd> &u) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Ju.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);
  assert(static_cast<std::size_t>(Ju.cols()) == nu);

  Jx = x_weight.asDiagonal();
  Ju.setZero();
}

void State_cost::calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                          const Eigen::Ref<const Eigen::VectorXd> &x) {

  assert(static_cast<std::size_t>(Jx.rows()) == nr);
  assert(static_cast<std::size_t>(Jx.cols()) == nx);

  Jx = x_weight.asDiagonal();
}

All_cost::All_cost(size_t nx, size_t nu, size_t nr,
                   const std::vector<boost::shared_ptr<Cost>> &costs)
    : Cost(nx, nu, nr), costs(costs) {}

void All_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {
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

void All_cost::calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {
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
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

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
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

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
  dynamics->calc(d->xnext, x, u);

  int index = 0;

  d->cost = 0;
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;
    feat->calc(data->r.segment(index, _nr), x, u);

    if (feat->cost_type == CostTYPE::least_squares) {
      d->cost += Scalar(0.5) *
                 data->r.segment(index, _nr).dot(data->r.segment(index, _nr));
    } else if (feat->cost_type == CostTYPE::linear) {
      d->cost += data->r.segment(index, _nr).sum();
    }
    index += _nr;
  }
}

void ActionModelQ::calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                            const Eigen::Ref<const VectorXs> &x,
                            const Eigen::Ref<const VectorXs> &u) {

  Data *d = static_cast<Data *>(data.get());
  dynamics->calcDiff(d->Fx, d->Fu, x, u);
  // CHANGE THIS

  // create a matrix for the Jacobians
  data->Lx.setZero();
  data->Lu.setZero();
  data->Lxx.setZero();
  data->Luu.setZero();
  data->Lxu.setZero();

  size_t index = 0;
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;

    auto &&r = data->r.segment(index, _nr);
    auto &&jx = Jx.block(index, 0, _nr, nx);
    auto &&ju = Ju.block(index, 0, _nr, nu);

    feat->calcDiff(jx, ju, x, u);

    if (feat->cost_type == CostTYPE::least_squares) {
      // std::cout << index << " " << _nr << std::endl;
      // std::cout << feat->get_name() << std::endl;
      // std::cout << r << std::endl;
      // std::cout << jx << std::endl;
      data->Lx.noalias() += r.transpose() * jx;
      data->Lu.noalias() += r.transpose() * ju;
      data->Lxx.noalias() += jx.transpose() * jx;
      data->Luu.noalias() += ju.transpose() * ju;
      data->Lxu.noalias() += jx.transpose() * ju;
    } else if (feat->cost_type == CostTYPE::linear) {
      data->Lx.noalias() += jx.colwise().sum();
      data->Lu.noalias() += ju.colwise().sum();
    }
    index += _nr;
  }
}

void ActionModelQ::calc(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x) {
  Data *d = static_cast<Data *>(data.get());

  int index = 0;

  d->cost = 0;
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;
    auto &&r = data->r.segment(index, _nr);
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

  // calcDiff(data, x, zero_u);
  Data *d = static_cast<Data *>(data.get());
  // CHANGE THIS

  data->Lx.setZero();
  data->Lu.setZero();
  data->Lxx.setZero();
  data->Luu.setZero();
  data->Lxu.setZero();

  size_t index = 0;
  for (size_t i = 0; i < features.size(); i++) {
    auto &feat = features.at(i);
    size_t &_nr = feat->nr;

    auto &&r = data->r.segment(index, _nr);
    auto &&jx = Jx.block(index, 0, _nr, nx);

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

  Eigen::VectorXd x(nx);
  Eigen::VectorXd u(nu);
  x.setRandom();
  u.setRandom();

  dyn->calcDiff(Fx, Fu, x, u);

  // compute the same using finite diff

  Eigen::MatrixXd FxD(nx, nx);
  FxD.setZero();
  Eigen::MatrixXd FuD(nx, nu);
  FuD.setZero();

  Eigen::VectorXd xnext(nx);
  xnext.setZero();
  dyn->calc(xnext, x, u);
  for (size_t i = 0; i < nx; i++) {
    Eigen::MatrixXd xe;
    xe = x;
    xe(i) += eps;
    Eigen::VectorXd xnexte(nx);
    xnexte.setZero();
    dyn->calc(xnexte, xe, u);
    auto df = (xnexte - xnext) / eps;
    FxD.col(i) = df;
  }

  for (size_t i = 0; i < nu; i++) {
    Eigen::MatrixXd ue;
    ue = u;
    ue(i) += eps;
    Eigen::VectorXd xnexte(nx);
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

  CHECK(((Fx - FxD).cwiseAbs().maxCoeff() < 10 * eps), AT);
  CHECK(((Fu - FuD).cwiseAbs().maxCoeff() < 10 * eps), AT);
}

void Generate_params::print(std::ostream &out) const {
  auto pre = "";
  auto after = ": ";
  out << pre << "free_time" << after << free_time << std::endl;
  out << pre << "name" << after << name << std::endl;
  out << pre << "N" << after << N << std::endl;
  out << pre << "goal" << after << goal.transpose() << std::endl;
  out << pre << "start" << after << start.transpose() << std::endl;
  out << pre << "cl" << after << cl << std::endl;
  out << pre << "states" << std::endl;
  for (const auto &s : states)
    out << "  - " << s.format(FMT) << std::endl;
  out << pre << "states_weights" << std::endl;
  for (const auto &s : states_weights)
    out << "  - " << s.format(FMT) << std::endl;
  out << pre << "actions" << std::endl;
  for (const auto &s : actions)
    out << "  - " << s.format(FMT) << std::endl;
  out << pre << "alpha_refs" << after << alpha_refs.format(FMT) << std::endl;
  out << pre << "cost_alpha_multis" << after << cost_alpha_multis.format(FMT)
      << std::endl;
  out << pre << "contour_control" << after << contour_control << std::endl;
  out << pre << "ref_alpha" << after << ref_alpha << std::endl;
  out << pre << "max_alpha" << after << max_alpha << std::endl;
  out << pre << "cost_alpha_multi" << after << cost_alpha_multi << std::endl;
  out << pre << "only_contour_last" << after << only_contour_last << std::endl;
  out << pre << "alpha_refs" << after << alpha_refs.transpose() << std::endl;
  out << pre << "cost_alpha_multis" << after << cost_alpha_multis.transpose()
      << std::endl;
  out << STR(goal_cost, after) << std::endl;
}

double max_rollout_error(ptr<Dynamics> dyn,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us) {

  assert(xs.size() == us.size() + 1);

  size_t N = us.size();

  size_t nx = xs.front().size();

  Eigen::VectorXd xnext(nx);
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

bool check_feas(ptr<Cost> feat_col, const std::vector<Eigen::VectorXd> &xs,
                const std::vector<Eigen::VectorXd> &us,
                const Eigen::VectorXd &goal) {

  double accumulated_c = 0;
  double max_c = 0;
  for (auto &x : xs) {
    Eigen::VectorXd out(1);
    feat_col->calc(out, x);
    accumulated_c += std::abs(out(0));

    if (std::abs(out(0)) > max_c) {
      max_c = std::abs(out(0));
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

ptr<crocoddyl::ShootingProblem>
generate_problem(const Generate_params &gen_args, size_t &nx, size_t &nu) {

  std::cout << "**\nGENERATING PROBLEM\n**\n" << std::endl;
  gen_args.print(std::cout);
  std::cout << "**\n" << std::endl;

  std::vector<ptr<Cost>> feats_terminal;
  ptr<crocoddyl::ActionModelAbstract> am_terminal;
  ptr<Dynamics> dyn;

  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> amq_runs;
  Eigen::VectorXd goal_v = gen_args.goal;

  if (opti_params.regularize_wrt_init_guess && gen_args.contour_control) {
    CHECK(false, AT);
  }


  Eigen::VectorXd x_ub , x_lb;

  if (gen_args.name == "unicycle_first_order_0") {

    x_ub =
        std::numeric_limits<double>::max() * Eigen::VectorXd::Ones(4);
    x_ub(3) = gen_args.max_alpha;

    Eigen::VectorXd weight_b(4);
    weight_b << 0, 0, 0, 200.;

    dyn = mk<Dynamics_unicycle>(gen_args.free_time);

    if (gen_args.contour_control) {
      dyn = mk<Dynamics_contour>(dyn, !gen_args.only_contour_last);
      // dyn = mk<Dynamics_Contour>(dyn, false);
    }

    nx = dyn->nx;
    nu = dyn->nu;

    ptr<Cost> control_feature;

    if (gen_args.free_time && !gen_args.contour_control)
      control_feature = mk<Control_cost>(
          nx, nu, nu, Eigen::Vector3d(.2, .2, 1.), Eigen::Vector3d(0., 0., .5));
    else if (!gen_args.free_time && !gen_args.contour_control)
      control_feature = mk<Control_cost>(nx, nu, nu, Eigen::Vector2d(.5, .5),
                                         Eigen::Vector2d::Zero());
    else if (!gen_args.free_time && gen_args.contour_control) {
      control_feature = mk<Control_cost>(
          nx, nu, nu, Eigen::Vector3d(.5, .5, 0.), Eigen::Vector3d::Zero());
    } else {
      CHECK_EQ(true, false, AT);
    }

    for (size_t t = 0; t < gen_args.N; t++) {

      std::vector<ptr<Cost>> feats_run;
      ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, gen_args.cl);
      feats_run = {cl_feature, control_feature};

      if (gen_args.states_weights.size() && gen_args.states.size()) {

        assert(gen_args.states_weights.size() == gen_args.states.size());
        assert(gen_args.states_weights.size() == gen_args.N);

        ptr<Cost> state_feature = mk<State_cost>(
            nx, nu, nx,
            .5 * opti_params.weight_goal * gen_args.states_weights.at(t),
            gen_args.states.at(t));
        feats_run.push_back(state_feature);
      }

      boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
          true, true, true};

      if (gen_args.contour_control)
        boost::static_pointer_cast<Col_cost>(cl_feature)->nx_effective = nx - 1;

      if (gen_args.contour_control) {

        if (!gen_args.linear_contour) {
          ptr<Contour_cost> contour =
              mk<Contour_cost>(nx, nu, gen_args.interpolator);
          contour->ref_alpha = gen_args.alpha_refs(t);
          contour->weight_contour *= gen_args.cost_alpha_multis(t);
          feats_run.push_back(contour);
        } else {
          std::cout << "warning, no contour in the trajectory " << std::endl;
          // ptr<Contour_cost_x> contour_x =
          //     mk<Contour_cost_x>(nx, nu, gen_args.interpolator);
          // contour_x->weight = opti_params.k_contour;

          ptr<Contour_cost_alpha_u> contour_alpha_u =
              mk<Contour_cost_alpha_u>(nx, nu);
          contour_alpha_u->k = opti_params.k_linear;

          // idea: use this only if it is close
          // ptr<Contour_cost_alpha_x> contour_alpha_x =
          //     mk<Contour_cost_alpha_x>(nx, nu);
          // contour_alpha_x->k = .1 * opti_params.k_linear;

          ptr<Cost> state_bounds = mk<State_bounds>(nx, nu, nx, x_ub, weight_b);

          // feats_run.push_back(contour_x);
          feats_run.push_back(contour_alpha_u);
          // feats_run.push_back(contour_alpha_x);
          feats_run.push_back(state_bounds);
        }
      }

      auto am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));

      if (opti_params.control_bounds) {

        if (gen_args.free_time) {
          am_run->set_u_lb(Eigen::Vector3d(-.5, -.5, .4));
          am_run->set_u_ub(Eigen::Vector3d(.5, .5, 1.5));
        } else if (gen_args.contour_control) {
          am_run->set_u_lb(Eigen::Vector3d(-.5, -.5, -10.));
          am_run->set_u_ub(Eigen::Vector3d(.5, .5, 10.));
        } else {
          // am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
          // am_run->set_u_ub(Eigen::Vector2d(.5, .5));
          // am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
          // am_run->set_u_ub(Eigen::Vector2d(.5, .5));

          am_run->set_u_lb(Eigen::Vector2d(-.5, -.5));
          am_run->set_u_ub(Eigen::Vector2d(.5, .5));
        }
      }

      amq_runs.push_back(am_run);
    }

    if (gen_args.contour_control) {

      if (!gen_args.linear_contour) {
        ptr<Contour_cost> Contour =
            mk<Contour_cost>(nx, nu, gen_args.interpolator);
        Contour->ref_alpha = gen_args.alpha_refs(gen_args.N);
        Contour->weight_contour *= gen_args.cost_alpha_multis(gen_args.N);

        feats_terminal.push_back(Contour);
      } else {
        ptr<Cost> state_bounds = mk<State_bounds>(nx, nu, nx, ub, weight_b);
        ptr<Contour_cost_x> contour_x =
            mk<Contour_cost_x>(nx, nu, gen_args.interpolator);
        feats_terminal.push_back(contour_x);
        feats_terminal.push_back(state_bounds);
      }
    }

    // if (gen_args.goal_cost && gen_args.contour_control) {
    //
    //   std::cout << "goal and contour " << std::endl;
    //   Eigen::VectorXd weight = Eigen::VectorXd::Ones(nx);
    //   // weight(nx - 1) = 0.;
    //   std::cout << "weights " << std::endl;
    //   std::cout << weight.format(FMT) << std::endl;
    //   ptr<Cost> state_feature = mk<State_cost>(
    //       nx, nu, nx, opti_params.weight_goal * weight, gen_args.goal);
    //
    //   ptr<Cost> state_bounds = mk<State_bounds>(nx, nu, nx, ub, weight_b);
    //
    //   feats_terminal = {state_feature, state_bounds};
    // }

    if (gen_args.goal_cost) {
      ptr<Cost> state_feature = mk<State_cost>(
          nx, nu, nx, opti_params.weight_goal * Eigen::VectorXd::Ones(nx),
          gen_args.goal);
      feats_terminal.push_back(state_feature);
    }
    am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

  } else if (gen_args.name == "unicycle_second_order_0") {

#if 0 

    dyn = mk<Dynamics_unicycle2>(gen_args.free_time);
    nx = dyn->nx;
    nu = dyn->nu;
    ptr<Cost> cl_feature = mk<Col_cost>(nx, nu, 1, opts.cl);
    boost::static_pointer_cast<Col_cost>(cl_feature)->non_zero_flags = {
        true, true, true, false, false};

    ptr<Cost> control_feature = mk<Control_cost>(
        nx, nu, nu, Eigen::Vector2d(.5, .5), Eigen::Vector2d::Zero());

    ptr<Cost> state_feature = mk<State_cost>(
        nx, nu, nx, 100. * Eigen::VectorXd::Ones(nx),
        Eigen::VectorXd::Map(opts.goal.data(), opts.goal.size()));

    ptr<Cost> feats_run =
        mk<All_cost>(nx, nu, cl_feature->nr + control_feature->nr,
                     std::vector<ptr<Cost>>{cl_feature, control_feature});

    feats_terminal = mk<All_cost>(nx, nu, state_feature->nr,
                                  std::vector<ptr<Cost>>{state_feature});

    am_run = to_am_base(mk<ActionModelQ>(dyn, feats_run));

    if (opts.control_bounds) {
      am_run->set_u_lb(Eigen::Vector2d(-.25, -.25));
      am_run->set_u_ub(Eigen::Vector2d(.25, .25));
    }
    am_terminal = to_am_base(mk<ActionModelQ>(dyn, feats_terminal));

    // TODO
    amq_runs = std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>(
        opts.N, am_run);

    // TODO: option to regularize w.r.t. intial guess.
  //
  //
  //

#endif
  } else {
    throw -1;
  }

  if (opti_params.use_finite_diff) {

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>>
        amq_runs_diff(amq_runs.size());

    std::transform(
        amq_runs.begin(), amq_runs.end(), amq_runs_diff.begin(),
        [&](const auto &am_run) {
          auto am_rundiff = mk<crocoddyl::ActionModelNumDiff>(am_run, true);
          boost::static_pointer_cast<crocoddyl::ActionModelNumDiff>(am_rundiff)
              ->set_disturbance(1e-4);
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
        ->set_disturbance(1e-4);
    am_terminal = am_terminal_diff;
  }

  ptr<crocoddyl::ShootingProblem> problem =
      mk<crocoddyl::ShootingProblem>(gen_args.start, amq_runs, am_terminal);

  return problem;
};

bool check_dynamics(const std::vector<Eigen::VectorXd> &xs_out,
                    const std::vector<Eigen::VectorXd> &us_out,
                    ptr<Dynamics> dyn) {

  double tolerance = 1e-4;
  CHECK_EQ(xs_out.size(), us_out.size() + 1, AT);

  size_t N = us_out.size();
  bool feasible = true;

  for (size_t i = 0; i < N; i++) {
    Eigen::VectorXd xnext(dyn->nx);

    auto &x = xs_out.at(i);
    auto &u = us_out.at(i);
    dyn->calc(xnext, x, u);

    if ((xnext - xs_out.at(i + 1)).norm() > tolerance) {
      std::cout << "Infeasible at " << i << std::endl;
      std::cout << xnext.transpose() << " " << xs_out.at(i + 1).transpose()
                << std::endl;
      feasible = false;
      break;
    }
  }

  return feasible;
}

Eigen::VectorXd enforce_bounds(const Eigen::VectorXd &us,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub) {

  return us.cwiseMax(lb).cwiseMin(ub);
}

void read_from_file(File_parser_inout &inout) {

  double dt = .1;

  std::cout << "Warning, dt is hardcoded to: " << dt << std::endl;

  YAML::Node init = YAML::LoadFile(inout.init_guess);
  YAML::Node env = YAML::LoadFile(inout.env_file);

  // load the collision checker
  std::cout << "loading collision checker... " << std::endl;
  inout.cl = mk<CollisionChecker>();
  inout.cl->load(inout.env_file);
  std::cout << "DONE" << std::endl;

  inout.name = env["robots"][0]["type"].as<std::string>();

  std::vector<std::vector<double>> states;
  // std::vector<Eigen::VectorXd> xs_init;
  // std::vector<Eigen::VectorXd> us_init;

  size_t N;
  std::vector<std::vector<double>> actions;

  if (!inout.new_format) {
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

    std::transform(
        states.begin(), states.end(), inout.xs.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

    std::transform(
        actions.begin(), actions.end(), inout.us.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

  } else {

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
    std::cout << "DT hardcoded to .1 " << std::endl;

    double total_time = _times.back();

    int num_time_steps = std::ceil(total_time / dt);

    Eigen::VectorXd times = Eigen::VectorXd::Map(_times.data(), _times.size());

    std::vector<Eigen::VectorXd> _xs_init(states.size());
    std::vector<Eigen::VectorXd> _us_init(actions.size());

    std::vector<Eigen::VectorXd> xs_init_new;
    std::vector<Eigen::VectorXd> us_init_new;

    int nx = 3;
    int nu = 2;

    std::transform(
        states.begin(), states.end(), _xs_init.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

    std::transform(
        actions.begin(), actions.end(), _us_init.begin(),
        [](const auto &s) { return Eigen::VectorXd::Map(s.data(), s.size()); });

    auto ts =
        Eigen::VectorXd::LinSpaced(num_time_steps + 1, 0, num_time_steps * dt);

    std::cout << "taking samples at " << ts.transpose() << std::endl;

    for (size_t ti = 0; ti < num_time_steps + 1; ti++) {
      Eigen::VectorXd xout(nx);
      Eigen::VectorXd Jout(nx);

      if (ts(ti) > times.tail(1)(0))
        xout = _xs_init.back();
      else
        linearInterpolation(times, _xs_init, ts(ti), xout, Jout);
      xs_init_new.push_back(xout);
    }

    auto times_u = times.head(times.size() - 1);
    for (size_t ti = 0; ti < num_time_steps; ti++) {
      Eigen::VectorXd uout(nu);
      Eigen::VectorXd Jout(nu);
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

  inout.start = Eigen::VectorXd::Map(_start.data(), _start.size());
  inout.goal = Eigen::VectorXd::Map(_goal.data(), _goal.size());

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

void convert_traj_with_variable_time(const std::vector<Eigen::VectorXd> &xs,
                                     const std::vector<Eigen::VectorXd> &us,
                                     std::vector<Eigen::VectorXd> &xs_out,
                                     std::vector<Eigen::VectorXd> &us_out) {

  size_t N = us.size();
  size_t nx = xs.front().size();
  size_t nu = us.front().size();
  double dt = .1;
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

  auto times = Eigen::VectorXd(N + 1);
  times.setZero();
  for (size_t i = 1; i < times.size(); i++) {
    times(i) = times(i - 1) + dt * us.at(i - 1)(nu - 1);
  }
  std::cout << times.transpose() << std::endl;

  // TODO: be careful with SO(2)
  std::vector<Eigen::VectorXd> x_out, u_out;
  for (size_t i = 0; i < num_time_steps + 1; i++) {
    double t = i * dt / scaling_factor;
    Eigen::VectorXd out(nx);
    Eigen::VectorXd Jout(nx);
    linearInterpolation(times, xs, t, out, Jout);
    x_out.push_back(out);
  }

  std::vector<Eigen::VectorXd> u_nx_orig(us.size());
  std::transform(us.begin(), us.end(), u_nx_orig.begin(),
                 [&nu](auto &s) { return s.head(nu - 1); });

  for (size_t i = 0; i < num_time_steps; i++) {
    double t = i * dt / scaling_factor;
    Eigen::VectorXd out(nu - 1);
    std::cout << " i and time and num_time_steps is " << i << " " << t << " "
              << num_time_steps << std::endl;
    Eigen::VectorXd J(nu - 1);
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

auto smooth_traj(const std::vector<Eigen::VectorXd> &us_init) {

  size_t n = us_init.front().size();
  std::vector<Eigen::VectorXd> us_out(us_init.size());
  // kernel

  Eigen::VectorXd kernel(5);

  kernel << 1, 2, 3, 2, 1;

  kernel /= kernel.sum();

  for (size_t i = 0; i < us_init.size(); i++) {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n);
    for (size_t j = 0; j < kernel.size(); j++) {
      out += kernel(j) * us_init.at(inside_bounds(i - kernel.size() / 2 + j, 0,
                                                  us_init.size() - 1));
    }
    us_out.at(i) = out;
  }
  return us_out;
}

void solve_with_custom_solver(File_parser_inout &file_inout,
                              Result_opti &opti_out) {

  // list of single solver

  std::vector<SOLVER> solvers{SOLVER::traj_opt,      SOLVER::traj_opt_free_time,
                              SOLVER::mpc,           SOLVER::mpcc,
                              SOLVER::mpcc2,         SOLVER::mpcc_linear,
                              SOLVER::mpc_adaptative};

  assert(std::find_if(solvers.begin(), solvers.end(), [](auto &s) {
           return s == static_cast<SOLVER>(opti_params.solver_id);
         }) != solvers.end());

  double dt = .1;
  std::cout << "Warning: "
            << "dt is hardcoded to " << dt << std::endl;

  bool verbose = false;
  auto cl = file_inout.cl;
  auto xs_init = file_inout.xs;
  auto us_init = file_inout.us;
  size_t N = us_init.size();
  auto goal = file_inout.goal;
  auto start = file_inout.start;
  auto name = file_inout.name;

  SOLVER solver = static_cast<SOLVER>(opti_params.solver_id);

  if (opti_params.repair_init_guess) {
    std::cout << "WARNING: reparing init guess, annoying SO2" << std::endl;
    for (size_t i = 1; i < N + 1; i++) {
      xs_init.at(i)(2) = xs_init.at(i - 1)(2) +
                         diff_angle(xs_init.at(i)(2), xs_init.at(i - 1)(2));
    }
    goal(2) = xs_init.at(N)(2) + diff_angle(goal(2), xs_init.at(N)(2));

    std::cout << "goal is now (maybe updated) " << goal.transpose()
              << std::endl;
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
  std::vector<Eigen::VectorXd> xs_out;
  std::vector<Eigen::VectorXd> us_out;

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
    // i could not stop when I reach the goal, only stop when I reach it with
    // the step move. Then, I would do the last at full speed? ( I hope :) )
    // Anyway, now is just fine

    CHECK_GEQ(opti_params.num_steps_to_optimize, opti_params.num_steps_to_move,
              AT);

    bool finished = false;

    std::vector<Eigen::VectorXd> xs_opt;
    std::vector<Eigen::VectorXd> us_opt;

    std::vector<Eigen::VectorXd> xs_init_rewrite = xs_init;
    std::vector<Eigen::VectorXd> us_init_rewrite = us_init;

    std::vector<Eigen::VectorXd> xs_warmstart_mpcc;
    std::vector<Eigen::VectorXd> us_warmstart_mpcc;

    std::vector<Eigen::VectorXd> xs_warmstart_adptative;
    std::vector<Eigen::VectorXd> us_warmstart_adptative;

    xs_opt.push_back(start);
    xs_init_rewrite.at(0) = start;

    debug_file_yaml << "opti:" << std::endl;

    auto times = Eigen::VectorXd::LinSpaced(xs_init.size(), 0,
                                            (xs_init.size() - 1) * dt);

    double max_alpha = times(times.size() - 1);

    ptr<Interpolator> path = mk<Interpolator>(times, xs_init);

    std::vector<Eigen::VectorXd> xs;
    std::vector<Eigen::VectorXd> us;

    double previous_alpha;

    Eigen::VectorXd previous_state = start;
    ptr<crocoddyl::ShootingProblem> problem;

    Eigen::VectorXd goal_mpc(3);

    bool is_last = false;

    double total_time = 0;
    size_t counter = 0;
    size_t total_iterations = 0;
    size_t num_steps_to_optimize_i = 0;

    bool close_to_goal = false;

    Eigen::VectorXd goal_with_alpha(4);
    goal_with_alpha.head(3) = goal;
    goal_with_alpha(3) = max_alpha;

    bool last_reaches_ = false;
    size_t index_first_goal = 0;
    while (!finished) {

      if (solver == SOLVER::mpc || solver == SOLVER::mpc_adaptative) {

        auto start_i = previous_state;
        if (solver == SOLVER::mpc) {
          assert(N - counter * opti_params.num_steps_to_move >= 0);
          size_t remaining_steps = N - counter * opti_params.num_steps_to_move;

          num_steps_to_optimize_i =
              std::min(opti_params.num_steps_to_optimize, remaining_steps);

          int subgoal_index =
              counter * opti_params.num_steps_to_move + num_steps_to_optimize_i;

          is_last = opti_params.num_steps_to_optimize > remaining_steps;

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
                    << counter * opti_params.num_steps_to_move << std::endl;

          num_steps_to_optimize_i = opti_params.num_steps_to_optimize;
          // next goal:
          size_t goal_index = index + num_steps_to_optimize_i;
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

        Generate_params gen_args{
            .free_time = false,
            .name = name,
            .N = num_steps_to_optimize_i,
            .goal = goal_mpc,
            .start = start_i,
            .cl = cl,
            .states = {},
            .actions = {},
        };

        size_t nx, nu;

        problem = generate_problem(gen_args, nx, nu);

        if (opti_params.use_warmstart) {

          if (solver == SOLVER::mpc) {
            xs = std::vector<Eigen::VectorXd>(
                xs_init_rewrite.begin() +
                    counter * opti_params.num_steps_to_move,
                xs_init_rewrite.begin() +
                    counter * opti_params.num_steps_to_move +
                    num_steps_to_optimize_i + 1);

            us = std::vector<Eigen::VectorXd>(
                us_init_rewrite.begin() +
                    counter * opti_params.num_steps_to_move,
                us_init_rewrite.begin() +
                    counter * opti_params.num_steps_to_move +
                    num_steps_to_optimize_i);
          } else {

            if (counter) {
              std::cout << "new warmstart" << std::endl;
              xs = xs_warmstart_adptative;
              us = us_warmstart_adptative;

              size_t missing_steps = num_steps_to_optimize_i - us.size();

              Eigen::VectorXd u_last = Eigen::VectorXd::Zero(nu);
              Eigen::VectorXd x_last = xs.back();

              // TODO: Sample the interpolator to get new init guess.
              for (size_t i = 0; i < missing_steps; i++) {
                us.push_back(u_last);
                xs.push_back(x_last);
              }
            } else {
              xs = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                                gen_args.start);
              us = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i,
                                                Eigen::VectorXd::Zero(nu));
            }
          }
          CHECK_EQ(xs.size(), num_steps_to_optimize_i + 1, AT);
          CHECK_EQ(us.size(), num_steps_to_optimize_i, AT);

          CHECK_EQ(xs.size(), us.size() + 1, AT);
          CHECK_EQ(us.size(), num_steps_to_optimize_i, AT);

        } else {
          xs = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                            gen_args.start);
          us = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i,
                                            Eigen::VectorXd::Zero(nu));
        }
      } else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        bool only_contour_last = true;
        const double threshold_alpha_close = 2.; //
        bool add_contour_to_all_when_close_to_gaol = true;
        const bool regularize_wrt_init_guess = false;

        num_steps_to_optimize_i =
            std::min(opti_params.num_steps_to_optimize, us_init.size());
        // approx alpha of first state
        auto it = std::min_element(
            path->x.begin(), path->x.end(), [&](const auto &a, const auto &b) {
              return (a - previous_state).squaredNorm() <=
                     (b - previous_state).squaredNorm();
            });

        size_t index = std::distance(path->x.begin(), it);
        std::cout << "starting with approx index " << index << std::endl;
        double alpha_of_first = path->times(index);
        std::cout << "alpha of first " << alpha_of_first << std::endl;

        double alpha_ref = alpha_of_first + opti_params.alpha_rate *
                                                num_steps_to_optimize_i * dt;

        // lets create a vector of alpha_refs
        Eigen::VectorXd alpha_refs = Eigen::VectorXd::LinSpaced(
            num_steps_to_optimize_i + 1, alpha_of_first,
            alpha_of_first + num_steps_to_optimize_i * dt);

        Eigen::VectorXd alpha_refs_goal = Eigen::VectorXd::LinSpaced(
            num_steps_to_optimize_i + 1, alpha_of_first,
            alpha_of_first +
                opti_params.alpha_rate * num_steps_to_optimize_i * dt);
        alpha_refs_goal = alpha_refs_goal.cwiseMin(max_alpha);
        alpha_refs = alpha_refs.cwiseMin(max_alpha);

        Eigen::VectorXd cost_alpha_multis =
            Eigen::VectorXd::Ones(num_steps_to_optimize_i + 1);
        cost_alpha_multis(0) =
            1.; // almost zero weight on alpha at the beginning

        std::cout << "vector of alpha refs  is " << std::endl;
        std::cout << alpha_refs.transpose() << std::endl;

        std::cout << "vector of alpha refs goals  is " << std::endl;
        std::cout << alpha_refs_goal.transpose() << std::endl;

        if (alpha_ref > max_alpha) {
          for (size_t i = 0; i < cost_alpha_multis.size(); i++) {
            if (alpha_refs_goal(i) >= max_alpha - 1e-4) {
              cost_alpha_multis(i) = 10.; //  Add higher alpha at the end
            }
          }
        }

        alpha_ref = std::min(alpha_ref, max_alpha);

        double cost_alpha_multi = 1.;
        int missing_indexes = times.size() - index;

        bool use_goal_reaching_formulation =
            std::fabs(alpha_of_first - max_alpha) < threshold_alpha_close;

        std::cout << "hardcoding " << std::endl;
        use_goal_reaching_formulation = true;
        add_contour_to_all_when_close_to_gaol = true;

        if (use_goal_reaching_formulation) {
          std::cout << "we are close to the goal" << std::endl;

#if 0
#endif
          if (add_contour_to_all_when_close_to_gaol) {
            only_contour_last = false;
            cost_alpha_multi = 1.;
          } else {
            std::cout << "alpha is close to the reference alpha " << std::endl;
            std::cout << std::fabs(alpha_of_first - max_alpha) << std::endl;
            std::cout << "adding very high cost to alpha" << std::endl;
            cost_alpha_multi = 100.;
          }
        }

        double cx_init;
        double cu_init;

        // if (!only_contour_last) {
        //   // the alpha_ref
        // }
        //
        // else {

        cx_init = std::min(alpha_of_first + 1.01 * num_steps_to_optimize_i * dt,
                           max_alpha - 1e-3);
        cu_init = cx_init;
        // }

        if (use_goal_reaching_formulation &&
            add_contour_to_all_when_close_to_gaol) {
          cx_init = alpha_of_first;
          cu_init = 0.;
        }

        std::cout << "cx_init:" << cx_init << std::endl;
        std::cout << "alpha_rate:" << opti_params.alpha_rate << std::endl;
        std::cout << "alpha_ref:" << alpha_ref << std::endl;
        std::cout << "max alpha " << max_alpha << std::endl;

        size_t nx, nu;
        const size_t _nx = 3;
        const size_t _nu = 2;

        Eigen::VectorXd start_ic(_nx + 1);
        start_ic.head(_nx) = previous_state.head(_nx);
        start_ic(_nx) = cx_init;

        bool goal_cost = false;

        if (std::fabs(alpha_refs(alpha_refs.size() - 1) - max_alpha) < 1e-3) {
          std::cout << "alpha_refs > max_alpha" << std::endl;
          close_to_goal = true;
        }

        if (close_to_goal) {
          std::cout << "i am close to the goal " << std::endl;
          goal_cost = true;
        }

        std::cout << "goal " << goal_with_alpha.format(FMT) << std::endl;

        //

        size_t index_add_goal_cost = 10;
        std::vector<Eigen::VectorXd> state_weights(num_steps_to_optimize_i);
        for (size_t t = 0; t < num_steps_to_optimize_i; t++) {
          if (t * dt > max_alpha - alpha_of_first && false)
            // if (t * dt > max_alpha - alpha_of_first )
            state_weights.at(t) = 1. * Eigen::VectorXd::Ones(4);
          else
            state_weights.at(t) = Eigen::VectorXd::Zero(4);
        }

        int try_faster = 5;
        if (last_reaches_) {
          std::cout << "last_reaches_"
                    << "special " << std::endl;
          std::cout << "try faster is " << try_faster << std::endl;
          // the goal was at step
          for (size_t t = 0; t < num_steps_to_optimize_i; t++) {
            if (t >
                index_first_goal - opti_params.num_steps_to_move - try_faster)
              state_weights.at(t) = 1. * Eigen::VectorXd::Ones(4);
            else
              state_weights.at(t) = Eigen::VectorXd::Zero(4);
          }
        }

        std::vector<Eigen::VectorXd> _states(num_steps_to_optimize_i);
        for (size_t t = 0; t < num_steps_to_optimize_i; t++) {
          _states.at(t) = goal_with_alpha;
        }

        Generate_params gen_args{.free_time = false,
                                 .name = name,
                                 .N = num_steps_to_optimize_i,
                                 .goal = goal_with_alpha,
                                 .start = start_ic,
                                 .cl = cl,
                                 .states = _states,
                                 .states_weights = state_weights,
                                 .actions = {},
                                 .contour_control = true,
                                 .interpolator = path,
                                 .ref_alpha = alpha_ref,
                                 .max_alpha = max_alpha,
                                 .cost_alpha_multi = cost_alpha_multi,
                                 .only_contour_last = only_contour_last,
                                 .alpha_refs = alpha_refs_goal,
                                 .cost_alpha_multis = cost_alpha_multis,
                                 .linear_contour =
                                     solver == SOLVER::mpcc_linear,
                                 .goal_cost = goal_cost};

        problem = generate_problem(gen_args, nx, nu);

        if (opti_params.use_warmstart) {
          // TODO: I need a more clever initialization. For example, using the
          // ones missing from last time, and then the default?

          std::cout << "warmstarting " << std::endl;
          size_t first_index = index;
          size_t last_index = index + num_steps_to_optimize_i;

          std::vector<Eigen::VectorXd> xs_i;
          std::vector<Eigen::VectorXd> us_i;

          bool reuse_opti = true;
          std::cout << "counter: " << counter << std::endl;
          if (counter && reuse_opti) {

            std::cout << "new warmstart" << std::endl;
            xs_i = xs_warmstart_mpcc;
            us_i = us_warmstart_mpcc;

            size_t missing_steps = num_steps_to_optimize_i - us_i.size();

            Eigen::VectorXd u_last = Eigen::VectorXd::Zero(3);
            Eigen::VectorXd x_last = xs_i.back();

            // TODO: Sample the interpolator to get new init guess.
            for (size_t i = 0; i < missing_steps; i++) {
              us_i.push_back(u_last);
              xs_i.push_back(x_last);
            }

          } else if (last_index < xs_init.size() - 1) {
            std::cout << "last index is " << last_index << std::endl;
            // TODO: I should reuse the ones in OPT
            xs_i = std::vector<Eigen::VectorXd>(&xs_init.at(first_index),
                                                &xs_init.at(last_index) + 1);
            us_i = std::vector<Eigen::VectorXd>(&us_init.at(first_index),
                                                &us_init.at(last_index));
          } else if (first_index < xs_init.size() - 1) {
            std::cout << "here " << std::endl;

            // copy some indexes, and fill the others

            // 0 1 2 3 4

            // starting from index 3, i need five states: 3 4 4
            // size_t num_extra = (xs_init.size() - 1) - first_index;
            auto xs_1 = std::vector<Eigen::VectorXd>(
                xs_init.begin() + first_index, xs_init.end());

            // Eigen::VectorXd xref(4);
            // xref.head(3) = goal;
            // xref(3) = max_alpha;

            auto xs_2 = std::vector<Eigen::VectorXd>(
                num_steps_to_optimize_i + 1 - xs_1.size(), goal);

            xs_1.insert(xs_1.end(), xs_2.begin(), xs_2.end());

            auto us_1 = std::vector<Eigen::VectorXd>(
                us_init.begin() + first_index, us_init.end());

            // Eigen::VectorXd uref(3);
            // uref << 0., 0., 0.;

            auto us_2 = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i -
                                                         us_1.size(),
                                                     Eigen::VectorXd::Zero(2));

            us_1.insert(us_1.end(), us_2.begin(), us_2.end());

            xs_i = xs_1;
            us_i = us_1;

          } else {

            std::cout << "here 2" << std::endl;

            xs_i = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                                xs_init.at(first_index));
            us_i = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i,
                                                Eigen::VectorXd::Zero(2));
          }

          CHECK_EQ(xs_i.size(), num_steps_to_optimize_i + 1, AT);
          CHECK_EQ(us_i.size(), num_steps_to_optimize_i, AT);

          std::vector<Eigen::VectorXd> xcs_i(xs_i.size());
          std::vector<Eigen::VectorXd> ucs_i(us_i.size());

          bool old = false;
          if (old) {
            std::transform(xs_i.begin(), xs_i.end(), xcs_i.begin(),
                           [&_nx, &cx_init](auto &x) {
                             Eigen::VectorXd new_last(_nx + 1);
                             new_last.head(_nx) = x;
                             new_last(_nx) = cx_init;
                             return new_last;
                           });

            std::transform(us_i.begin(), us_i.end(), ucs_i.begin(),
                           [&_nu, &cu_init](auto &u) {
                             Eigen::VectorXd new_last(_nu + 1);
                             new_last.head(_nu) = u;
                             new_last(_nu) = cu_init;
                             return new_last;
                           });
          } else if (counter && reuse_opti) {
            xcs_i = xs_i;
            ucs_i = us_i;
          }

          else {

            for (size_t i = 0; i < xcs_i.size(); i++) {
              Eigen::VectorXd new_last(_nx + 1);
              new_last.head(_nx) = xs_i.at(i);
              new_last(_nx) = alpha_refs(i);
              xcs_i.at(i) = new_last;
            }

            for (size_t i = 0; i < ucs_i.size(); i++) {
              Eigen::VectorXd new_last(_nu + 1);
              new_last.head(_nu) = us_i.at(i);
              new_last(_nu) = alpha_refs(i + 1) - alpha_refs(i);
              ucs_i.at(i) = new_last;
            }
          }

          xs = xcs_i;
          us = ucs_i;

        } else {

          Eigen::VectorXd u0c(3);
          u0c.head(2).setZero();
          u0c(2) = cu_init;
          xs = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i + 1,
                                            gen_args.start);
          us = std::vector<Eigen::VectorXd>(num_steps_to_optimize_i, u0c);
          //
          // std::cout << "warmstarting the alpha's" << std::endl;
          //
          // for (size_t i = 0; i < xs.size(); i++) {
          //   xs.at(i)(3) = alpha_refs(i);
          // }
          //
          // for (size_t i = 0; i < us.size(); i++) {
          //   Eigen::VectorXd new_last(_nu + 1);
          //   us.at(i)(2) = alpha_refs(i + 1) - alpha_refs(i);
          // }
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

      // ENFORCING BOUNDS

      Eigen::VectorXd x_lb, x_ub, u_lb, u_ub;

      if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {
        double inf = 1e6;
        x_lb.resize(4);
        x_ub.resize(4);
        u_lb.resize(3);
        u_ub.resize(3);

        x_lb << -inf, -inf, -inf, 0.;
        x_ub << inf, inf, inf, max_alpha;
        u_lb << -inf, -inf, 0.;
        u_ub << inf, inf, inf;

      } else {
        double inf = 1e6;
        x_lb.resize(3);
        x_ub.resize(3);
        u_lb.resize(2);
        u_ub.resize(2);

        x_lb << -inf, -inf, -inf;
        x_ub << inf, inf, inf;
        u_lb << -inf, -inf;
        u_ub << inf, inf;
      }

      if (opti_params.noise_level > 1e-8) {
        for (size_t i = 0; i < xs.size(); i++) {
          xs.at(i) += opti_params.noise_level *
                      Eigen::VectorXd::Random(xs.front().size());
          xs.at(i) = enforce_bounds(xs.at(i), x_lb, x_ub);
        }

        for (size_t i = 0; i < us.size(); i++) {
          us.at(i) += opti_params.noise_level *
                      Eigen::VectorXd::Random(us.front().size());
          us.at(i) = enforce_bounds(us.at(i), u_lb, u_ub);
        }
      }

      crocoddyl::Timer timer;
      ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
      size_t iterations_i = ddp.get_iter();
      double time_i = timer.get_duration();
      accumulated_time += time_i;
      // solver == SOLVER::traj_opt_smooth_then_free_time) {

      std::cout << "time: " << time_i << std::endl;
      std::cout << "iterations: " << iterations_i << std::endl;
      total_time += time_i;
      total_iterations += iterations_i;
      std::vector<Eigen::VectorXd> xs_i_sol = ddp.get_xs();
      std::vector<Eigen::VectorXd> us_i_sol = ddp.get_us();

      previous_state = xs_i_sol.at(opti_params.num_steps_to_move).head(3);

      size_t copy_steps = 0;

      auto fun_is_goal = [&](const auto &x) {
        return (x.head(3) - goal).norm() < 1e-2;
      };

      if (solver == SOLVER::mpc_adaptative) {
        // check if I reach the goal.

        size_t final_index = num_steps_to_optimize_i;
        Eigen::VectorXd x_last = ddp.get_xs().at(final_index);
        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "true last state: " << ddp.get_xs().back().format(FMT)
                  << std::endl;
        std::cout << "distance to goal: " << (x_last.head(3) - goal).norm()
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
        std::cout << "ideally, I should check if I can check the goal faster, "
                     "with a small linear search "
                  << std::endl;
        // Eigen::VectorXd x_last = ddp.get_xs().back();
        size_t final_index = num_steps_to_optimize_i;
        // size_t final_index = opti_params.num_steps_to_move;
        std::cout << "final index is " << final_index << std::endl;

        double alpha_mpcc = ddp.get_xs().at(final_index)(3);
        Eigen::VectorXd x_last = ddp.get_xs().at(final_index);
        last_reaches_ = fun_is_goal(ddp.get_xs().back());

        std::cout << "**\n" << std::endl;
        std::cout << "checking as final index: " << final_index << std::endl;
        std::cout << "alpha_mpcc:" << alpha_mpcc << std::endl;
        std::cout << "last state: " << x_last.format(FMT) << std::endl;
        std::cout << "true last state: " << ddp.get_xs().back().format(FMT)
                  << std::endl;
        std::cout << "distance to goal: " << (x_last.head(3) - goal).norm()
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

          num_steps_to_optimize_i = std::distance(ddp.get_xs().begin(), it);
          std::cout << "changing the number of steps to optimize(copy) to "
                    << num_steps_to_optimize_i << std::endl;
        }

        std::cout << "checking if i am close to the goal " << std::endl;

        // check 1

        for (size_t i = 0; i < ddp.get_xs().size(); i++) {
          auto &x = ddp.get_xs().at(i);
          if ((x.head(3) - goal).norm() < 1e-1) {
            std::cout << "one state is close to goal! " << std::endl;
            close_to_goal = true;
          }

          if (std::fabs(x(3) - max_alpha) < 1e-1) {
            std::cout << "alpha is close to final " << std::endl;
            close_to_goal = true;
          }
        }
      }

      if (is_last)
        copy_steps = num_steps_to_optimize_i;
      else
        copy_steps = opti_params.num_steps_to_move;

      for (size_t i = 0; i < copy_steps; i++)
        xs_opt.push_back(xs_i_sol.at(i + 1).head(3));

      for (size_t i = 0; i < copy_steps; i++)
        us_opt.push_back(us_i_sol.at(i).head(2));

      if (solver == SOLVER::mpc) {
        for (size_t i = 0; i < num_steps_to_optimize_i; i++) {
          xs_init_rewrite.at(1 + counter * opti_params.num_steps_to_move + i) =
              xs_i_sol.at(i + 1).head(3);

          us_init_rewrite.at(counter * opti_params.num_steps_to_move + i) =
              us_i_sol.at(i).head(2);
        }
      } else if (solver == SOLVER::mpc_adaptative) {

        xs_warmstart_adptative.clear();
        us_warmstart_adptative.clear();

        for (size_t i = copy_steps; i < num_steps_to_optimize_i; i++) {
          xs_warmstart_adptative.push_back(xs_i_sol.at(i));
          us_warmstart_adptative.push_back(us_i_sol.at(i));
        }
        xs_warmstart_adptative.push_back(xs_i_sol.back());

      }

      else if (solver == SOLVER::mpcc || solver == SOLVER::mpcc_linear) {

        xs_warmstart_mpcc.clear();
        us_warmstart_mpcc.clear();

        for (size_t i = copy_steps; i < num_steps_to_optimize_i; i++) {
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
        double alpha_mpcc = ddp.get_xs().back()(3);
        Eigen::VectorXd out(3);
        Eigen::VectorXd Jout(3);
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
    size_t nx = 3;
    size_t nu = 2;
    ptr<Cost> feat_col = mk<Col_cost>(nx, nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(feat_col)->margin = 0.;
    Eigen::VectorXd goal_last = Eigen::VectorXd::Map(goal.data(), goal.size());

    bool feasible_ = check_feas(feat_col, xs_out, us_out, goal_last);

    ptr<Dynamics> dyn = mk<Dynamics_unicycle>();

    bool dynamics_feas = check_dynamics(xs_out, us_out, dyn);

    feasible = feasible_ && dynamics_feas;

    std::cout << "dynamics_feas: " << dynamics_feas << std::endl;

    std::cout << "feasible_ is " << feasible_ << std::endl;
    std::cout << "feasible is " << feasible << std::endl;
  } else if (solver == SOLVER::traj_opt ||
             solver == SOLVER::traj_opt_free_time) {

    if (solver == SOLVER::traj_opt_free_time) {
      std::vector<Eigen::VectorXd> us_init_time(us_init.size());
      size_t nu = us_init.front().size();
      for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd u(nu + 1);
        u.head(nu) = us_init.at(i);
        u(nu) = 1.;
        us_init_time.at(i) = u;
      }
      us_init = us_init_time;
    }

    Generate_params gen_args{
        .free_time = solver == SOLVER::traj_opt_free_time,
        .name = name,
        .N = N,
        .goal = goal,
        .start = start,
        .cl = cl,
        .states = xs_init,
        .actions = us_init,
    };

    size_t nx, nu;

    ptr<crocoddyl::ShootingProblem> problem =
        generate_problem(gen_args, nx, nu);
    std::vector<Eigen::VectorXd> xs(N + 1, gen_args.start);
    std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(nu));

    if (opti_params.use_warmstart) {
      xs = xs_init;
      us = us_init;
    }

    if (opti_params.noise_level > 0.) {
      for (size_t i = 0; i < xs.size(); i++) {
        assert(xs.at(i).size() == nx);
        xs.at(i) += opti_params.noise_level * Eigen::VectorXd::Random(nx);
      }

      for (size_t i = 0; i < us.size(); i++) {
        assert(us.at(i).size() == nx);
        us.at(i) += opti_params.noise_level * Eigen::VectorXd::Random(nu);
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
    ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
    std::cout << "time: " << timer.get_duration() << std::endl;
    accumulated_time += timer.get_duration();

    // check the distance to the goal:
    ptr<Cost> feat_col = mk<Col_cost>(nx, nu, 1, cl);
    boost::static_pointer_cast<Col_cost>(feat_col)->margin = 0.;

    feasible = check_feas(feat_col, ddp.get_xs(), ddp.get_us(), gen_args.goal);
    std::cout << "feasible is " << feasible << std::endl;

    std::cout << "solution" << std::endl;
    std::cout << problem->calc(xs, us) << std::endl;

    if (solver == SOLVER::traj_opt_free_time) {
      std::vector<Eigen::VectorXd> _xs;
      std::vector<Eigen::VectorXd> _us;

      auto dyn = mk<Dynamics_unicycle>(true);

      std::cout << "max error before "
                << max_rollout_error(dyn, ddp.get_xs(), ddp.get_us())
                << std::endl;

      convert_traj_with_variable_time(ddp.get_xs(), ddp.get_us(), _xs, _us);
      xs_out = _xs;
      us_out = _us;

      dyn = mk<Dynamics_unicycle>(false);

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

    if (solver == SOLVER::traj_opt_smooth_then_free_time) {

      CHECK_EQ(opti_params.free_time, false, AT);

      std::cout << "repeating now with free time" << std::endl;
      gen_args.free_time = true;
      gen_args.states = xs_out;
      gen_args.actions = us_out;

      problem = generate_problem(gen_args, nx, nu);
      std::vector<Eigen::VectorXd> xs(N + 1, Eigen::VectorXd::Zero(nx));
      std::vector<Eigen::VectorXd> us(N, Eigen::VectorXd::Zero(nu));

      std::cout << " nu " << nu << std::endl;
      for (size_t t = 0; t < N + 1; t++) {
        xs.at(t) = ddp.get_xs().at(t);
      }
      for (size_t t = 0; t < N; t++) {
        us.at(t) = Eigen::VectorXd::Ones(nu);
        us.at(t).head(nu - 1) = ddp.get_us().at(t);
      }

      for (auto &x : xs) {
        std::cout << x.transpose() << std::endl;
      }

      for (auto &u : us) {
        std::cout << u.transpose() << std::endl;
      }

      std::cout << "feasible is " << feasible << std::endl;

      std::cout << "before..." << std::endl;
      std::cout << "cost " << problem->calc(ddp.get_xs(), us) << std::endl;
      feasible = check_feas(feat_col, ddp.get_xs(), us, gen_args.goal);
      std::cout << "feasible is " << feasible << std::endl;

      ddp = crocoddyl::SolverBoxFDDP(problem);
      ddp.set_th_stop(opti_params.th_stop);
      ddp.set_th_acceptnegstep(opti_params.th_acceptnegstep);

      if (opti_params.CALLBACKS) {
        std::vector<ptr<crocoddyl::CallbackAbstract>> cbs;
        cbs.push_back(mk<crocoddyl::CallbackVerbose>());
        ddp.setCallbacks(cbs);
      }

      crocoddyl::Timer timer;
      ddp.solve(xs, us, opti_params.max_iter, false, opti_params.init_reg);
      std::cout << "time: " << timer.get_duration() << std::endl;
      accumulated_time += timer.get_duration();

      feasible =
          check_feas(feat_col, ddp.get_xs(), ddp.get_us(), gen_args.goal);
      std::cout << "feasible is: " << feasible << std::endl;

      // convert form

      std::vector<Eigen::VectorXd> _xs;
      std::vector<Eigen::VectorXd> _us;

      convert_traj_with_variable_time(ddp.get_xs(), ddp.get_us(), _xs, _us);

      xs_out = _xs;
      us_out = _us;
    };
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
        if (i == 0)
          opti_params.alpha_rate = 1.;
        else
          opti_params.alpha_rate = 1.2;

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

    auto check_with_rate = [](const File_parser_inout &file_inout, double rate,
                              Result_opti &opti_out) {
      const double dt = .1;

      std::vector<Eigen::VectorXd> us_init = file_inout.us;
      std::vector<Eigen::VectorXd> xs_init = file_inout.xs;

      Eigen::VectorXd times = Eigen::VectorXd::LinSpaced(us_init.size() + 1, 0,
                                                         us_init.size() * dt);

      // resample a trajectory
      size_t original_n = us_init.size();
      Eigen::VectorXd times_2 = rate * times;

      // create an interpolator
      Interpolator interp_x(times_2, xs_init);
      Interpolator interp_u(times_2.head(us_init.size()), us_init);

      int new_n = std::ceil(rate * original_n);

      Eigen::VectorXd new_times =
          Eigen::VectorXd::LinSpaced(new_n + 1, 0, new_n * dt);
      std::vector<Eigen::VectorXd> new_xs(new_n + 1);
      std::vector<Eigen::VectorXd> new_us(new_n);

      Eigen::VectorXd x(3);
      Eigen::VectorXd Jx(3);
      Eigen::VectorXd u(2);
      Eigen::VectorXd Ju(2);

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

    double max_rate = 1.5;
    double min_rate = .7;
    int num_rates_to_test = 10;

    Eigen::VectorXd rates =
        Eigen::VectorXd::LinSpaced(num_rates_to_test, min_rate, max_rate);

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

  out << "feasible: " << feasible << std::endl;
  out << "cost: " << cost << std::endl;
  out << "result:" << std::endl;
  out << "  - states:" << std::endl;
  for (auto &x : xs_out) {
    x(2) = std::remainder(x(2), 2 * M_PI);
    out << "      - " << x.format(FMT) << std::endl;
  }

  out << "    actions:" << std::endl;
  for (auto &u : us_out) {
    out << "      - " << u.format(FMT) << std::endl;
  }
};

void File_parser_inout::add_options(po::options_description &desc) {
  desc.add_options()("env", po::value<std::string>(&env_file)->required())(
      "waypoints", po::value<std::string>(&init_guess)->required())(
      "new_format", po::value<bool>(&new_format)->default_value(new_format));
}

void File_parser_inout::print(std::ostream &out) {

  std::string be = "";
  std::string af = ": ";
  out << be << STR(init_guess, af) << std::endl;
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
