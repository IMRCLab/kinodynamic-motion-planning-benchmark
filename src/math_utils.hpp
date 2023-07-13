#pragma once
#include <cmath>

#include "Eigen/Core"
#include "croco_macros.hpp"
#include <Eigen/Geometry>

const Eigen::IOFormat FMT(6, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

bool inline check_bounds(const Eigen::VectorXd &v, const Eigen::VectorXd &v_lb,
                         const Eigen::VectorXd &v_ub, double tol = 1e-10) {

  CHECK_EQ(v.size(), v_lb.size(), AT);
  CHECK_EQ(v.size(), v_lb.size(), AT);
  size_t n = v.size();
  for (size_t i = 0; i < n; i++) {
    if (v(i) > v_ub(i) + tol || v(i) < v_lb(i) - tol) {
      std::cout << "Warning: Outside Bounds " << std::endl;
      std::cout << STR_V(v) << std::endl;
      std::cout << STR_V(v_lb) << std::endl;
      std::cout << STR_V(v_ub) << std::endl;
      return false;
    }
  }
  return true;
}

double inline check_bounds_distance(const Eigen::VectorXd &v,
                                    const Eigen::VectorXd &v_lb,
                                    const Eigen::VectorXd &v_ub) {

  CHECK_EQ(v.size(), v_lb.size(), AT);
  CHECK_EQ(v.size(), v_lb.size(), AT);
  size_t n = v.size();
  double max_distance = 0;
  for (size_t i = 0; i < n; i++) {
    double d1 = std::max(v(i) - v_ub(i), 0.);
    double d2 = std::max(v_lb(i) - v(i), 0.);
    if (d1 > max_distance) {
      max_distance = d1;
    }
    if (d2 > max_distance) {
      max_distance = d2;
    }
  }
  return max_distance;
}

Eigen::VectorXd inline enforce_bounds(const Eigen::VectorXd &us,
                                      const Eigen::VectorXd &lb,
                                      const Eigen::VectorXd &ub) {
  CHECK_EQ(us.size(), lb.size(), AT);
  CHECK_EQ(us.size(), ub.size(), AT);
  return us.cwiseMax(lb).cwiseMin(ub);
}

bool inline check_equal(Eigen::MatrixXd A, Eigen::MatrixXd B, double rtol,
                        double atol) {
  CHECK_EQ(A.rows(), B.rows(), AT);
  CHECK_EQ(A.cols(), B.cols(), AT);

  auto dif = (A - B).cwiseAbs();
  auto max_cwise = A.cwiseAbs().cwiseMax(B.cwiseAbs());

  auto tmp = dif - (rtol * max_cwise +
                    atol * Eigen::MatrixXd::Ones(A.rows(), A.cols()));
  // all element in tmp shoulb be negative
  bool out = (tmp.array() <= 0).all();

  if (!out) {
    std::cout << "**\nERROR" << std::endl;
    std::cout << "A\n" << A << std::endl;
    std::cout << "B\n" << B << std::endl;
    std::cout << "A-B\n" << A - B << std::endl;
    std::cout << "**" << std::endl;
  }
  return out;
}

double inline wrap_angle(double x) {
  x = fmod(x + M_PI, 2 * M_PI);
  if (x < 0)
    x += 2 * M_PI;
  return x - M_PI;
}

bool inline is_diagonal(const Eigen::Ref<const Eigen::MatrixXd> &mat,
                        double tol = 1e-12) {

  if (mat.rows() != mat.cols())
    return false;

  for (int i = 0; i < mat.rows(); i++) {
    for (int j = 0; j < mat.cols(); j++) {
      if (i != j && std::fabs(mat(i, j)) > tol) {
        return false;
      }
    }
  }
  return true;
}

template <class T> T inside_bounds(const T &i, const T &lb, const T &ub) {
  CHECK_GEQ(ub, lb, AT);

  if (i < lb)
    return lb;

  else if (i > ub)
    return ub;
  else
    return i;
}

template <class Derived>
inline Eigen::Matrix<typename Derived::Scalar, 3, 3>
Skew(const Eigen::MatrixBase<Derived> &vec) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);
  return (Eigen::Matrix<typename Derived::Scalar, 3, 3>() << 0.0, -vec[2],
          vec[1], vec[2], 0.0, -vec[0], -vec[1], vec[0], 0.0)
      .finished();
}

void inline finite_diff_grad(std::function<double(const Eigen::VectorXd &)> fun,
                             const Eigen::VectorXd &x,
                             Eigen::Ref<Eigen::VectorXd> D, double eps = 1e-6) {

  assert(x.size() == D.size());
  double y = fun(x);

  for (size_t i = 0; i < static_cast<size_t>(x.size()); i++) {
    Eigen::VectorXd xe = x;
    double ye;
    xe(i) += eps;
    ye = fun(xe);
    D(i) = (ye - y) / eps;
  }
};

void inline finite_diff_hess(std::function<double(const Eigen::VectorXd &)> fun,
                             const Eigen::VectorXd &x,
                             Eigen::Ref<Eigen::MatrixXd> H, double eps = 1e-6) {

  double y = fun(x);
  double h;

  for (size_t i = 0; i < static_cast<size_t>(x.size()); i++) {
    for (size_t j = i; j < static_cast<size_t>(x.size()); j++) {
      if (i == j) {
        Eigen::VectorXd e1 = x;
        e1(i) += eps;

        Eigen::VectorXd e2 = x;
        e2(i) -= eps;

        H(i, j) = (fun(e1) - 2 * y + fun(e2)) / (eps * eps);
      } else {

        Eigen::VectorXd e1 = x;
        e1(i) += eps;
        e1(j) += eps;

        Eigen::VectorXd e2 = x;
        e2(i) -= eps;
        e2(j) -= eps;

        Eigen::VectorXd e3 = x;
        e3(i) += eps;
        e3(j) -= eps;

        Eigen::VectorXd e4 = x;
        e4(i) -= eps;
        e4(j) += eps;

        h = (fun(e1) + fun(e2) - fun(e3) - fun(e4)) / (4 * eps * eps);
        H(i, j) = h;
        H(j, i) = h;
      }
    }
  }
};

void inline finite_diff_jac(
    std::function<void(const Eigen::VectorXd &, Eigen::Ref<Eigen::VectorXd>)>
        fun,
    const Eigen::VectorXd &x, size_t nout, Eigen::Ref<Eigen::MatrixXd> J,
    double eps = 1e-6) {

  Eigen::VectorXd y(nout);

  fun(x, y);

  for (size_t i = 0; i < static_cast<size_t>(x.size()); i++) {
    Eigen::VectorXd xe = x;
    Eigen::VectorXd ye(nout);
    xe(i) += eps;
    fun(xe, ye);
    J.col(i) = (ye - y) / eps;
  }
};

// REF: https://arxiv.org/pdf/1711.02508.pdf
// page 22
void inline __get_quat_from_ang_vel_time(
    const Eigen::Vector3d &v, Eigen::Ref<Eigen::Vector4d> q,
    Eigen::Matrix<double, 4, 3> *J = nullptr) {

  // v = theta * u ( theta \in R , u \in R^3 with ||u||=1)
  // u = v /  || v||
  // theta is ||v||
  // q = Exp( theta * u ) = e^( theta * u / 2 ) = cos( theta / 2 ) + u sin(
  // theta / 2 )
  //
  //  if theta is very small:
  //
  // q = Exp( theta * u ) = cos( theta / 2 ) + v /theta * sin( theta / 2 )
  //
  // sin( theta / 2 ) =  theta / 2 - theta^3 / 48 + ...
  // v /theta * sin( theta / 2 ) =  v * ( 1/2 - theta^2 / 48 )

  double theta = v.norm();
  Eigen::Vector3d u = v / theta;
  //
  //
  double threshold_use_taylor = 1e-6;
  if (theta < threshold_use_taylor) {
    // std::cout << "very small theta " << theta << std::endl;
    q.head<3>() = v * (.5 - theta * theta / 48.);
    q(3) = std::cos(.5 * theta);

    if (J) {
      J->block<3, 3>(0, 0).setZero();
      J->block<3, 3>(0, 0).diagonal() +=
          (.5 - theta * theta / 48.) * Eigen::Vector3d::Ones();
      J->block<3, 3>(0, 0) -= 1. / 24. * v * v.transpose();
      // J->row(3) = -.5 * sin(.5 * theta) * v / theta;
      J->row(3) = -.5 * .5 * v;
    }
  } else {
    q(3) = std::cos(.5 * theta);
    q.head<3>() = u * std::sin(.5 * theta);

    if (J) {
      J->block<3, 3>(0, 0).setZero();
      Eigen::Matrix3d I3;
      I3.setIdentity();
      J->block<3, 3>(0, 0) +=
          (I3 / theta - u * u.transpose() / theta) * std::sin(.5 * theta);
      J->block<3, 3>(0, 0) +=
          u * (.5 * std::cos(.5 * theta) * v.transpose() / theta);
      J->row(3) = -.5 * std::sin(.5 * theta) * v / theta;
    }
  }
}

// TODO: Remove this function.
Eigen::Quaterniond inline get_quat_from_ang_vel_time(
    const Eigen::Vector3d &angular_rotation) {

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

// Eigen::Quaterniond inline qintegrate(const Eigen::Quaterniond &q,
//                                      const Eigen::Vector3d &omega, float dt)
//                                      {
//   Eigen::Quaterniond deltaQ = get_quat_from_ang_vel_time(omega * dt);
//   return q * deltaQ;
// };

void inline quat_product(const Eigen::Vector4d &p, const Eigen::Vector4d &q,
                         Eigen::Ref<Eigen::VectorXd> out, Eigen::Matrix4d *Jp,
                         Eigen::Matrix4d *Jq) {

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

  if (Jq) {

    Eigen::Vector4d dyw_dq{-px, -py, -pz, pw};
    Eigen::Vector4d dyx_dq{pw, -pz, py, px};
    Eigen::Vector4d dyy_dq{pz, pw, -px, py};
    Eigen::Vector4d dyz_dq{-py, px, pw, pz};

    Jq->row(0) = dyx_dq;
    Jq->row(1) = dyy_dq;
    Jq->row(2) = dyz_dq;
    Jq->row(3) = dyw_dq;
  }

  if (Jq) {
    Eigen::Vector4d dyw_dp{-qx, -qy, -qz, qw};
    Eigen::Vector4d dyx_dp{qw, qz, -qy, qx};
    Eigen::Vector4d dyy_dp{-qz, qw, qx, qy};
    Eigen::Vector4d dyz_dp{qy, -qx, qw, qz};

    Jp->row(0) = dyx_dp;
    Jp->row(1) = dyy_dp;
    Jp->row(2) = dyz_dp;
    Jp->row(3) = dyw_dp;
  }
}

void inline normalize(const Eigen::Ref<const Eigen::Vector4d> &q,
                      Eigen::Ref<Eigen::Vector4d> y,
                      Eigen::Ref<Eigen::Matrix4d> J) {
  double norm = q.norm();
  y = q / norm;
  Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
  J.noalias() = I4 / norm - q * q.transpose() / (std::pow(norm, 3));
}

void inline rotate_with_q(const Eigen::Ref<const Eigen::Vector4d> &x,
                          const Eigen::Ref<const Eigen::Vector3d> &a,
                          Eigen::Ref<Eigen::Vector3d> y,
                          Eigen::Ref<Eigen::Matrix<double, 3, 4>> Jx,
                          Eigen::Ref<Eigen::Matrix3d> Ja) {

  Eigen::Vector4d q;
  Eigen::Matrix4d Jnorm;
  Eigen::Matrix<double, 3, 4> Jq;

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

double inline diff_angle(double angle1, double angle2) {
  return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

void inline runge4(
    Eigen::Ref<Eigen::VectorXd> y, const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &u,
    std::function<void(Eigen::Ref<Eigen::VectorXd> y,
                       const Eigen::Ref<const Eigen::VectorXd> &x,
                       const Eigen::Ref<const Eigen::VectorXd> &u)>
        fun,
    double h) {

  size_t n = x.size();
  Eigen::VectorXd k1(n);
  Eigen::VectorXd k2(n);
  Eigen::VectorXd k3(n);
  Eigen::VectorXd k4(n);

  fun(k1, x, u);
  fun(k2, x + h * k1 / 2., u);
  fun(k3, x + h * k2 / 2., u);
  fun(k4, x + k3 * h, u);

  y = x + 1. / 6. * h * (k1 + 2 * k2 + 2 * k3 + k4);
}

void inline euler_second_order_system(
    Eigen::Ref<Eigen::VectorXd> y, const Eigen::Ref<const Eigen::VectorXd> &x,
    const Eigen::Ref<const Eigen::VectorXd> &vel,
    const Eigen::Ref<const Eigen::VectorXd> &acc, double dt) {

  assert(y.size() == 2 * x.size());
  assert(y.size() == 2 * vel.size());
  const size_t n = x.size();

  for (size_t i = 0; i < n; i++) {
    y(i) = x(i) + vel(i) * dt;
  }

  for (size_t i = 0; i < n; i++) {
    y(i + n) = vel(i) + acc(i) * dt;
  }
}

// y = x + v(x,u) dt
void inline euler(Eigen::Ref<Eigen::VectorXd> y,
                  const Eigen::Ref<const Eigen::VectorXd> &x,
                  const Eigen::Ref<const Eigen::VectorXd> &v, double dt) {

  assert(y.size() == x.size());
  assert(y.size() == v.size());
  assert(dt >= 0);

  const size_t n = x.size();
  for (size_t i = 0; i < n; i++) {
    y(i) = x(i) + v(i) * dt;
  }
  // model->integrate(x, v * dt , y);

  // state->
}

// y = x + v(x,u) dt
void inline euler_diff(Eigen::Ref<Eigen::MatrixXd> Jy_x,
                       Eigen::Ref<Eigen::MatrixXd> Jy_u, double dt,
                       const Eigen::Ref<const Eigen::MatrixXd> &Jv_x,
                       const Eigen::Ref<const Eigen::MatrixXd> &Jv_u) {

  const size_t n = Jy_x.cols();
  assert(static_cast<size_t>(Jy_x.cols()) == n);
  assert(static_cast<size_t>(Jy_x.rows()) == n);

  assert(static_cast<size_t>(Jv_x.cols()) == n);
  assert(static_cast<size_t>(Jv_x.rows()) == n);
  assert(static_cast<size_t>(Jv_u.rows()) == n);
  assert(static_cast<size_t>(Jy_u.rows()) == n);
  assert(Jy_u.cols() == Jv_u.cols());

  Jy_x.noalias() = dt * Jv_x;
  for (size_t i = 0; i < n; i++) {
    Jy_x(i, i) += 1;
  }
  Jy_u.noalias() = dt * Jv_u;
}

void inline so2_interpolation(double &state, double from, double to, double t) {
  assert(from <= M_PI && from >= -M_PI);
  assert(to <= M_PI && to >= -M_PI);

  double diff = to - from;
  if (fabs(diff) <= M_PI)
    state = from + diff * t;
  else {
    double &v = state;
    if (diff > 0.0)
      diff = 2.0 * M_PI - diff;
    else
      diff = -2.0 * M_PI - diff;
    v = from - diff * t;
    // input states are within bounds, so the following check is sufficient
    if (v > M_PI)
      v -= 2.0 * M_PI;
    else if (v < -M_PI)
      v += 2.0 * M_PI;
  }
}

double inline so3_distance(Eigen::Vector4d x, Eigen::Vector4d y) {
  double max_quaternion_norm_error = 1e-6;
  // CSTR_V(x);
  // CSTR_V(y);
  CHECK_LEQ(std::abs(x.norm() - 1), max_quaternion_norm_error, AT);
  CHECK_LEQ(std::abs(y.norm() - 1), max_quaternion_norm_error, AT);
  double dq = std::fabs(x.dot(y));
  if (dq > 1.0 - max_quaternion_norm_error)
    return 0.0;
  return std::acos(dq);
}

double inline so2_distance(double x, double y) {
  assert(y <= M_PI && y >= -M_PI);
  assert(x <= M_PI && x >= -M_PI);
  double d = std::fabs(x - y);
  return (d > M_PI) ? 2.0 * M_PI - d : d;
}

inline double normalize_angle(double angle) {
  const double result = fmod(angle + M_PI, 2.0 * M_PI);
  if (result <= 0.0)
    return result + M_PI;
  return result - M_PI;
}

inline double norm_sq(double *x, size_t n) {
  double out = 0;
  for (size_t i = 0; i < n; i++) {
    out += x[i] * x[i];
  }
  return out;
}

double inline l2_squared(const double *v, const double *w, size_t n) {
  double out = 0.;
  for (size_t i = 0; i < n; i++) {
    out += (v[i] - w[i]) * (v[i] - w[i]);
  }
  return out;
}

double inline l2(const double *v, const double *w, size_t n) {
  return std::sqrt(l2_squared(v, w, n));
}

void inline element_wise(double *y, const double *x, const double *r,
                         size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] = x[i] * r[i];
  }
}

void inline element_wise(double *y, const double *r, size_t n) {
  for (size_t i = 0; i < n; i++) {
    y[i] *= r[i];
  }
}
