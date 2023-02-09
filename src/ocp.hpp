#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>

#include "Eigen/Core"
#include <boost/program_options.hpp>

// #include "crocoddyl/core/actions/unicycle.hpp"
#include "crocoddyl/core/numdiff/action.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/core/states/euclidean.hpp"

#include "collision_checker.hpp"
#include "croco_macros.hpp"

template <class T> using ptr = boost::shared_ptr<T>;

template <typename T, typename... Args> auto mk(Args &&...args) {
  return boost::make_shared<T>(std::forward<Args>(args)...);
}

using namespace crocoddyl;

namespace po = boost::program_options;
struct Opti_params {

  bool CALLBACKS = true;
  bool use_finite_diff = false;
  bool use_warmstart = true;
  bool free_time = false;
  bool repair_init_guess = true;
  bool regularize_wrt_init_guess = false;
  bool control_bounds = true;
  bool adaptative_goal_mpc = false;
  int solver_id = 0;

  double th_stop = 1e-2;
  double init_reg = 1e2;
  double th_acceptnegstep = .3;
  double noise_level = 1e-3; // factor on top of [-1., 1.]
  double alpha_rate = 1.5;   // I try to go X times faster
  double k_linear = 10.;
  double k_contour = 10.;

  size_t max_iter = 50;
  size_t num_steps_to_optimize = 20;
  size_t num_steps_to_move = 10;
  size_t max_mpc_iterations = 50;
  std::string debug_file_name = "debug_file.yaml";
  double weight_goal = 200.;
  double collision_weight = 100.;
  bool smooth_traj = false;

  void add_options(po::options_description &desc);
  void print(std::ostream &out);
};


extern Opti_params opti_params;

void linearInterpolation(const Eigen::VectorXd &times,
                         const std::vector<Eigen::VectorXd> &x, double t_query,
                         Eigen::Ref<Eigen::VectorXd> out,
                         Eigen::Ref<Eigen::VectorXd> Jx);

struct Interpolator {

  Eigen::VectorXd times;
  std::vector<Eigen::VectorXd> x;

  Interpolator(const Eigen::VectorXd &times,
               const std::vector<Eigen::VectorXd> &x)
      : times(times), x(x) {
    CHECK_EQ(times.size(), x.size(), AT);
  }

  void inline interpolate(double t_query, Eigen::Ref<Eigen::VectorXd> out,
                          Eigen::Ref<Eigen::VectorXd> J) {
    linearInterpolation(times, x, t_query, out, J);
  }
};

struct ActionDataQ : public ActionDataAbstractTpl<double> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  template <template <typename Scalar> class Model>
  explicit ActionDataQ(Model<Scalar> *const model) : Base(model) {}
};

double inline diff_angle(double angle1, double angle2) {
  return atan2(sin(angle1 - angle2), cos(angle1 - angle2));
}

struct Dynamics {

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) = 0;

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) = 0;

  size_t nx;
  size_t nu;

  virtual ~Dynamics(){};
};

struct Dummy_dynamics : Dynamics {

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  Dummy_dynamics(size_t t_nx, size_t t_nu) {
    nx = t_nx;
    nu = t_nu;
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u){};

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) {}
};

struct Dynamics_contour : Dynamics {

  ptr<Dynamics> dyn;
  bool accumulate = false;
  Dynamics_contour(ptr<Dynamics> dyn, bool accumulate);

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u) override;

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u) override;
};

struct Dynamics_unicycle2 : Dynamics {

  double dt = .1;

  bool free_time;
  Dynamics_unicycle2(bool free_time = false);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

struct Dynamics_unicycle : Dynamics {

  double dt = .1;
  bool free_time;

  Dynamics_unicycle(bool free_time = false);

  typedef MathBaseTpl<double> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual void calc(Eigen::Ref<Eigen::VectorXd> xnext,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Fx,
                        Eigen::Ref<Eigen::MatrixXd> Fu,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);
};

enum class CostTYPE {
  linear,
  least_squares,
};

struct Cost {
  size_t nx;
  size_t nu;
  size_t nr;
  std::string name;
  CostTYPE cost_type = CostTYPE::least_squares;

  Eigen::VectorXd zero_u;
  Eigen::MatrixXd zero_Ju;

  Cost(size_t nx, size_t nu, size_t nr)
      : nx(nx), nu(nu), nr(nr), zero_u(nu), zero_Ju(nr, nu) {
    zero_u.setZero();
    zero_Ju.setZero();
  }
  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u) {

    throw std::runtime_error(AT);
  }

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x) {

    throw std::runtime_error(AT);
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u) {

    throw std::runtime_error(AT);
  }

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x) {

    throw std::runtime_error(AT);
  }
  virtual std::string get_name() const { return name; }
};

struct Contour_cost_alpha_x : Cost {

  double k = 1.5; // bigger than 0
  Contour_cost_alpha_x(size_t nx, size_t nu);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);
};

struct Contour_cost_alpha_u : Cost {

  double k = 1.5; // bigger than 0
  Contour_cost_alpha_u(size_t nx, size_t nu);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);
};

void finite_diff_cost(ptr<Cost> cost, Eigen::Ref<Eigen::MatrixXd> Jx,
                      Eigen::Ref<Eigen::MatrixXd> Ju, const Eigen::VectorXd &x,
                      const Eigen::VectorXd &u, const int nr);

struct Contour_cost_x : Cost {

  ptr<Interpolator> path;
  double weight = 20.;
  bool use_finite_diff = false;

  double last_query;
  Eigen::VectorXd last_out;
  Eigen::VectorXd last_J;

  Contour_cost_x(size_t nx, size_t nu, ptr<Interpolator> path);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct Contour_cost : Cost {

  ptr<Interpolator> path;
  double ref_alpha = 1.;
  double weight_alpha = 2.;
  double weight_diff = 20.;
  double weight_contour = 1.;
  double weight_virtual_control = 2.;
  bool use_finite_diff = false;

  double last_query;
  Eigen::VectorXd last_out;
  Eigen::VectorXd last_J;

  Contour_cost(size_t nx, size_t nu, ptr<Interpolator> path);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct Col_cost : Cost {

  boost::shared_ptr<CollisionChecker> cl;
  double margin = .03;
  double last_raw_d = 0;
  Eigen::VectorXd last_x;
  Eigen::VectorXd last_grad;

  // TODO: check that the sec_factor is actually save in a test
  double sec_factor = .1;

  // what about a log barrier function? -- maybe I get better gradients

  double faraway_zero_gradient_bound = 1.1 * margin;
  // returns 0 gradient if distance is > than THIS.
  double epsilon = 1e-3;            // finite diff
  std::vector<bool> non_zero_flags; // if not nullptr, non_zero_flags[i] = True
                                    // means compute gradient, non_zero_flags[i]
                                    // = false is no compute gradient

  size_t nx_effective;

  Col_cost(size_t nx, size_t nu, size_t nr,
           boost::shared_ptr<CollisionChecker> cl);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct Control_cost : Cost {

  Eigen::VectorXd u_weight;
  Eigen::VectorXd u_ref;

  Control_cost(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &u_weight,
               const Eigen::VectorXd &u_ref);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

// x - ub  <= 0

struct State_bounds : Cost {
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

  Eigen::VectorXd ub;
  Eigen::VectorXd weight;

  State_bounds(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &ub,
               const Eigen::VectorXd &weight);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};
struct State_cost : Cost {

  Eigen::VectorXd x_weight;
  Eigen::VectorXd ref;

  State_cost(size_t nx, size_t nu, size_t nr, const Eigen::VectorXd &x_weight,
             const Eigen::VectorXd &ref);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

struct All_cost : Cost {

  std::vector<boost::shared_ptr<Cost>> costs;

  All_cost(size_t nx, size_t nu, size_t nr,
           const std::vector<boost::shared_ptr<Cost>> &costs);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x,
                    const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calc(Eigen::Ref<Eigen::VectorXd> r,
                    const Eigen::Ref<const Eigen::VectorXd> &x);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        Eigen::Ref<Eigen::MatrixXd> Ju,
                        const Eigen::Ref<const Eigen::VectorXd> &x,
                        const Eigen::Ref<const Eigen::VectorXd> &u);

  virtual void calcDiff(Eigen::Ref<Eigen::MatrixXd> Jx,
                        const Eigen::Ref<const Eigen::VectorXd> &x);
};

size_t
get_total_num_features(const std::vector<boost::shared_ptr<Cost>> &features);

class ActionModelQ : public ActionModelAbstractTpl<double> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef double Scalar;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataQ Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector2s Vector2s;

  boost::shared_ptr<Dynamics> dynamics;
  std::vector<boost::shared_ptr<Cost>> features;
  size_t nx;
  size_t nu;
  size_t nr;

  Eigen::MatrixXd Jx;
  Eigen::MatrixXd Ju;
  // Eigen::VectorXd zero_u;
  // Eigen::MatrixXd zero_Ju;

  ActionModelQ(ptr<Dynamics> dynamics, const std::vector<ptr<Cost>> &features);

  virtual ~ActionModelQ();

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);

  virtual void calc(const boost::shared_ptr<ActionDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &x);

  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &x);

  virtual boost::shared_ptr<ActionDataAbstract> createData();
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract> &data);

  virtual void print(std::ostream &os) const;
};

const Eigen::IOFormat FMT(6, Eigen::DontAlignCols, ",", ",", "", "", "[", "]");

extern double accumulated_time;

enum class SOLVER {
  traj_opt = 0,
  traj_opt_free_time = 1,
  traj_opt_smooth_then_free_time = 2,
  mpc = 3,
  mpcc = 4,
  mpcc2 = 5,
  traj_opt_mpcc = 6,
  mpc_nobound_mpcc = 7,
  mpcc_linear = 8,
  time_search_traj_opt = 9,
  mpc_adaptative = 10,
  none = 11,
};

void PrintVariableMap(const boost::program_options::variables_map &vm,
                             std::ostream &out);

template <typename Derived>
boost::shared_ptr<crocoddyl::ActionModelAbstract>
to_am_base(boost::shared_ptr<Derived> am) {
  return boost::static_pointer_cast<crocoddyl::ActionModelAbstract>(am);
};

void print_data(boost::shared_ptr<ActionDataAbstractTpl<double>> data);

void check_dyn(boost::shared_ptr<Dynamics> dyn, double eps);

struct Generate_params {
  bool free_time = false;
  std::string name;
  size_t N;
  Eigen::VectorXd goal;
  Eigen::VectorXd start;
  ptr<CollisionChecker> cl;
  std::vector<Eigen::VectorXd> states;
  std::vector<Eigen::VectorXd> states_weights;
  std::vector<Eigen::VectorXd> actions;
  bool contour_control = false;
  ptr<Interpolator> interpolator = nullptr;
  double ref_alpha = 1.;
  double max_alpha = 100.;
  double cost_alpha_multi = 1.;
  bool only_contour_last = true;
  Eigen::VectorXd alpha_refs;
  Eigen::VectorXd cost_alpha_multis;
  bool linear_contour = false;
  bool goal_cost = true;

  void print(std::ostream &out) const;
};

double max_rollout_error(ptr<Dynamics> dyn,
                         const std::vector<Eigen::VectorXd> &xs,
                         const std::vector<Eigen::VectorXd> &us);

bool check_feas(ptr<Cost> feat_col, const std::vector<Eigen::VectorXd> &xs,
                const std::vector<Eigen::VectorXd> &us,
                const Eigen::VectorXd &goal);

ptr<crocoddyl::ShootingProblem>
generate_problem(const Generate_params &gen_args, size_t &nx, size_t &nu);

bool check_dynamics(const std::vector<Eigen::VectorXd> &xs_out,
                    const std::vector<Eigen::VectorXd> &us_out,
                    ptr<Dynamics> dyn);

struct File_parser_inout {
  std::string init_guess;
  std::string env_file;
  bool new_format = false;
  std::string name;
  ptr<CollisionChecker> cl;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  Eigen::VectorXd start;
  Eigen::VectorXd goal;

  void add_options(po::options_description &desc);
  void print(std::ostream &out);
};

Eigen::VectorXd enforce_bounds(const Eigen::VectorXd &us,
                               const Eigen::VectorXd &lb,
                               const Eigen::VectorXd &ub);

void read_from_file(File_parser_inout &inout);

void convert_traj_with_variable_time(const std::vector<Eigen::VectorXd> &xs,
                                     const std::vector<Eigen::VectorXd> &us,
                                     std::vector<Eigen::VectorXd> &xs_out,
                                     std::vector<Eigen::VectorXd> &us_out);

struct Result_opti {
  bool feasible = false;
  double cost = -1;
  std::vector<Eigen::VectorXd> xs_out;
  std::vector<Eigen::VectorXd> us_out;

  void write_yaml(std::ostream &out);

  void write_yaml_db(std::ostream &out);
};

int inside_bounds(int i, int lb, int ub);

auto smooth_traj(const std::vector<Eigen::VectorXd> &us_init);

void solve_with_custom_solver(File_parser_inout &file_inout,
                              Result_opti &opti_out);

void compound_solvers(File_parser_inout file_inout, Result_opti &opti_out);

;
