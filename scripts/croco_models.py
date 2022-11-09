import crocoddyl
import numpy as np
import matplotlib.pylab as plt
from unicycle_utils import *
import time
from robots import Quadrotor, RobotUnicycleFirstOrder, RobotCarFirstOrderWithTrailers, RobotUnicycleSecondOrder, qnormalize, qrotate, qintegrate

from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad, jacfwd, jacrev, jit
import jax.numpy as jnp
import sys
from functools import partial
from jax.interpreters import ad, batching, xla
from jax import core, dtypes, lax


def dist_modpi(x, y):
    d = x - y
    if isinstance(x, np.ndarray):
        d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
    else:
        d = d.at[2].set(jnp.arctan2(jnp.sin(d[2]), jnp.cos(d[2])))
    return d


def dist_trailer(x, y):
    d = x - y
    if isinstance(x, np.ndarray):
        d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
        d[3] = np.arctan2(np.sin(d[3]), np.cos(d[3]))
    else:
        d = d.at[2].set(jnp.arctan2(jnp.sin(d[2]), jnp.cos(d[2])))
        d = d.at[3].set(jnp.arctan2(jnp.sin(d[3]), jnp.cos(d[3])))
    return d


def dist_quadcopter(x, y):
    d = x - y
    return d


class CollisionJax():

    def __init__(self, cc):
        self.col_p = core.Primitive("col")
        self.cc = cc
        self.register()

    def col(self, x, u):
        """The JAX-traceable way to use the JAX primitive.

        Note that the traced arguments must be passed as positional arguments
        to `bind`.
        """
        return self.col_p.bind(x, u)

    def col_impl(self, x, u):
        distance, _, _ = self.cc.distance(x)  # Float, Vector, Vector
        return np.array(distance)

    def col_value_and_jvp(self, arg_values, arg_tangents):
        """Evaluates the primal output and the tangents (Jacobian-vector product).

        Given values of the arguments and perturbation of the arguments (tangents),
        compute the output of the primitive and the perturbation of the output.

        This method must be JAX-traceable. JAX may invoke it with abstract values
        for the arguments and tangents.

        Args:
          arg_values: a tuple of arguments
          arg_tangents: a tuple with the tangents of the arguments. The tuple has
            the same length as the arg_values. Some of the tangents may also be the
            special value ad.Zero to specify a zero tangent.
        Returns:
           a pair of the primal output and the tangent.
        """
        x, u = arg_values
        xt, ut = arg_tangents
        distance, grad = self.cc.distanceWithFDiffGradient(
            x)  # Float, Vector, Vector

        def make_zero(tan):
            return lax.zeros_like_array(x) if isinstance(tan, ad.Zero) else tan

        return (np.array(distance), np.array(grad) @ make_zero(xt))

    def col_abstract_eval(self, xs, us):
        """Abstract evaluation of the primitive.

        This fnction does not need to be JAX traceable. It will be invoked with
        abstractions of the actual arguments.
        Args:
          xs, us: abstractions of the arguments.
        Result:
          a ShapedArray for the result of the primitive.
        """
        assert xs.shape == us.shape
        return abstract_arrays.ShapedArray(xs.shape, xs.dtype)

    def register(self):
        self.col_p.def_impl(self.col_impl)
        self.col_p.def_abstract_eval(self.col_abstract_eval)
        ad.primitive_jvps[self.col_p] = self.col_value_and_jvp


class FakeData():
    xnext = np.matrix([])
    r = np.matrix([])
    cost = 0.
    Lx = np.matrix([])
    Lu = np.matrix([])
    Lxx = np.matrix([])
    Lxu = np.matrix([])
    Luu = np.matrix([])
    Fx = np.matrix([])
    Fu = np.matrix([])
    Jx = np.matrix([])
    Ju = np.matrix([])


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def diff_angle(angle1, angle2, use_np=True):
    if use_np:
        return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))
    else:
        return jnp.arctan2(jnp.sin(angle1 - angle2), jnp.cos(angle1 - angle2))


class ActionModelAD(crocoddyl.ActionModelAbstract):

    def __init__(self, robot, features, use_jit=False):
        """
        """
        self.robot = robot
        self.features = features
        self.timer = 0

        self.step_fn = self.robot.step
        self.feat_fn = self.features

        self.Fx_fn = jacfwd(self.step_fn, 0)
        self.Fu_fn = jacfwd(self.step_fn, 1)

        self.Jx_fn = jacfwd(self.feat_fn, 0)
        self.Ju_fn = jacfwd(self.feat_fn, 1)

        if use_jit:
            self.step_fn = jit(self.step_fn)
            self.feat_fn = jit(self.feat_fn)
            self.Fx_fn = jit(self.Fx_fn)
            self.Fu_fn = jit(self.Fu_fn)

            self.Jx_fn = jit(self.Jx_fn)
            self.Ju_fn = jit(self.Ju_fn)

    def calc(self, data, _x, _u=None):
        tic = time.perf_counter()
        if _u is None:
            _u = self.unone
        x = jnp.array(np.array(_x))
        u = jnp.array(np.array(_u))
        _xnext = self.step_fn(x, u)
        _r = self.feat_fn(x, u)
        # should I convert to matrix?
        data.xnext = np.matrix(np.asarray(_xnext)).T
        data.r = np.matrix(np.asarray(_r)).T
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))
        self.timer += time.perf_counter() - tic

    def calcDiff(self, data, x, u=None):
        """
        """

        tic = time.perf_counter()
        xj = jnp.array(x)
        uj = jnp.array(u)

        Fx = self.Fx_fn(xj, uj)
        Fu = self.Fu_fn(xj, uj)

        data.Fx = np.asarray(Fx)
        data.Fu = np.asarray(Fu)

        Jx = self.Jx_fn(xj, uj)
        Ju = self.Ju_fn(xj, uj)

        data.Jx = np.asarray(Jx)
        data.Ju = np.asarray(Ju)

        r = np.asarray(self.feat_fn(xj, uj))

        data.Lx = r @ data.Jx
        data.Lu = r @ data.Ju
        data.Lxx = data.Jx.T @ data.Jx
        data.Lxu = np.zeros((len(x), len(u)))
        data.Luu = data.Ju.T @ data.Ju

        self.timer += time.perf_counter() - tic


# class Trailer_DT():

#     def __init__(self, L, hitch_lengths, free_time):

#         self.L = L
#         self.hitch_lengths = hitch_lengths
#         self.dt = 0.1
#         self.is2D = True
#         self.free_time = free_time

#     def step(self, state, action):
#         """"
#         x_dot = v * cos (theta_0)
#         y_dot = v * sin (theta_0)
#         theta_0_dot = v / L * tan(phi)
#         theta_1_dot = v / hitch_lengths[0] * sin(theta_0 - theta_1)
#         ...
#         """
#         x, y, yaw = state[0], state[1], state[2]

#         if self.free_time:
#             v, phi, t = action
#         else:
#             v, phi = action
#             t = 1.

#         yaw_next = yaw + v / self.L * jnp.tan(phi) * self.dt * t
#         x_next = x + v * jnp.cos(yaw) * self.dt * t
#         y_next = y + v * jnp.sin(yaw) * self.dt * t

#         state_next_list = [x_next, y_next, yaw_next]

#         for i, d in enumerate(self.hitch_lengths):
#             theta_dot = v / d
#             for j in range(0, i):
#                 theta_dot *= jnp.cos(state[2 + j] - state[3 + j])
#             theta_dot *= jnp.sin(state[2 + i] - state[3 + i])

#             theta = state[3 + i]
#             theta_next = theta + theta_dot * self.dt * t
#             # theta_next_norm = normalize_angle(theta_next)
#             state_next_list.append(theta_next)

#         state_next = jnp.array(state_next_list)
#         return state_next


class AM_Trailer_AD(ActionModelAD, crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr, free_time=False):
        nu = 3 if free_time else 2
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), nu, nr)
        self.unone = np.zeros(self.nu)
        ActionModelAD.__init__(
            self,
            RobotCarFirstOrderWithTrailers(
                L=.25,
                hitch_lengths=[.5],
                free_time=free_time,
                semi_implicit=False,
                normalize=False),
            features)

        # Trailer_DT(
        # L=.25, hitch_lengths=[.5], free_time=self.free_time),


# class Unicycle_order1_DT:

#     def __init__(self, free_time=False):
#         self.dt = 0.1
#         self.free_time = free_time

#     def step(self, state, action):
#         x, y, yaw = state
#         if self.free_time:
#             v, w, t = action
#         else:
#             v, w = action
#             t = 1.
#         yaw_next = yaw + w * t * self.dt
#         x_next = x + v * jnp.cos(yaw) * t * self.dt
#         y_next = y + v * jnp.sin(yaw) * t * self.dt

#         state_next = jnp.array([x_next, y_next, yaw_next])
#         return state_next


class AM_unicycle_order1_AD(ActionModelAD,
                            crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr, free_time=False):
        nu = 3 if free_time else 2
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(3), nu, nr)
        self.unone = np.zeros(self.nu)
        ActionModelAD.__init__(
            self, RobotUnicycleFirstOrder(free_time=free_time,
                                          semi_implicit=False,
                                          normalize=False), features)


class AM_unicycle_order2_AD(ActionModelAD,
                            crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr, free_time=False):
        nu = 3 if free_time else 2
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(5), nu, nr)
        self.unone = np.zeros(self.nu)
        ActionModelAD.__init__(
            self, RobotUnicycleSecondOrder(free_time=free_time,
                                           semi_implicit=False,
                                           normalize=False), features)


class AM_quadrotor(ActionModelAD,
                            crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr, free_time=False, use_jit=False):
        nu = 5 if free_time else 4
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(13), nu, nr)
        self.unone = np.zeros(self.nu)
        ActionModelAD.__init__(
            self, Quadrotor(free_time=free_time,
                                           semi_implicit=False,
                                           normalize=False), features, 
            use_jit=use_jit)





# class Unicycle_order2_DT:

#     def __init__(self, free_time=False):
#         self.dt = 0.1
#         self.free_time = free_time

#     def step(self, state, action):
#         x, y, yaw, v, w = state

#         if self.free_time:
#             a, w_dot, t = action
#         else:
#             a, w_dot = action
#             t = 1.

#         # For compatibility with KOMO, update v and yaw first
#         v_next = v + a * self.dt * t
#         w_dot_next = w + w_dot * self.dt * t
#         yaw_next = yaw + w_dot_next * self.dt * t
#         # yaw_next_norm = (yaw_next + np.pi) % (2 * np.pi) - np.pi
#         yaw_next_norm = yaw_next
#         x_next = x + v_next * jnp.cos(yaw_next) * self.dt * t
#         y_next = y + v_next * jnp.sin(yaw_next) * self.dt * t

#         # x_next = x + v * np.cos(yaw) * dt
#         # y_next = y + v * np.sin(yaw) * dt
#         # yaw_next = yaw + w * dt
#         # normalize yaw between -pi and pi

#         state_next = jnp.array(
#             [x_next, y_next, yaw_next_norm, v_next, w_dot_next])
#         return state_next


# class QuadrotorDT:

#     def __init__(self, free_time=False):
#         self.free_time = free_time
#         # parameters (Crazyflie 2.0 quadrotor)
#         self.mass = 0.034  # kg
#         # self.J = np.array([
#         #   [16.56,0.83,0.71],
#         #   [0.83,16.66,1.8],
#         #   [0.72,1.8,29.26]
#         #   ]) * 1e-6  # kg m^2
#         self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

#         # Note: we assume here that our control is forces
#         arm_length = 0.046  # m
#         arm = 0.707106781 * arm_length
#         t2t = 0.006  # thrust-to-torque ratio
#         self.B0 = np.array([
#             [1, 1, 1, 1],
#             [-arm, -arm, arm, arm],
#             [-arm, arm, arm, -arm],
#             [-t2t, t2t, -t2t, t2t]
#         ])
#         self.g = 9.81  # not signed

#         if self.J.shape == (3, 3):
#             # full matrix -> pseudo inverse
#             self.inv_J = np.linalg.pinv(self.J)
#         else:
#             self.inv_J = 1 / self.J  # diagonal matrix -> division

#         self.dt = 0.01
#         self.is2D = False

#     def step(self, _state, _action):
#         state = jnp.array(_state)

#         if self.free_time:
#             action = jnp.asarray(_action[:-1])
#             t = _action[-1]
#         else:
#             action = jnp.asarray(_action)
#             t = 1.

#         q = jnp.concatenate((state[6:7], state[3:6]))
#         q = qnormalize(q)
#         omega = state[10:]

#         eta = jnp.dot(self.B0, action)

#         f_u = jnp.array([0, 0, eta[0]])
#         tau_u = jnp.array([eta[1], eta[2], eta[3]])

#         # dynamics
#         # dot{p} = v
#         pos_next = state[0:3] + state[7:10] * self.dt * t
#         # mv = mg + R f_u
#         vel_next = state[7:10] + (jnp.array([0, 0, -self.g]) +
#                                   qrotate(q, f_u) / self.mass) * self.dt * t

#         # dot{R} = R S(w)
#         # to integrate the dynamics, see
#         # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
#         # https://arxiv.org/pdf/1604.08139.pdf
#         q_next = qnormalize(qintegrate(q, omega, self.dt * t))

#         # mJ = Jw x w + tau_u
#         omega_next = state[10:] + (self.inv_J *
#                                    (jnp.cross(self.J * omega, omega) + tau_u)) * self.dt * t

#         return jnp.concatenate(
#             (pos_next, q_next[1:4], q_next[0:1], vel_next, omega_next))


class OT():
    none = -1
    cost = 0
    penalty = 1
    auglag = 2


class PenaltyFeat():
    mu = 1.
    tt = OT.penalty

    def __init__(self, fn, nr):
        self.fn = fn
        self.nr = nr

    def __call__(self, *args):
        # TODO: what happens with the 1/2?
        return (self.mu / 2.0)**.5 * self.fn(*args)


class CostTerm():
    tt = OT.cost

    def __init__(self, fn, nr):
        self.fn = fn
        self.nr = nr
        self.name = "r-" + fn.name

    def __call__(self, *args):
        return self.fn(*args)


class AuglagTerm():
    mu = 1.
    tt = OT.auglag
    l = np.array([])

    def __init__(self, fn, nr, l=None):
        self.fn = fn
        self.nr = nr
        self.l = l
        if self.l is None:
            self.l = np.zeros(nr)
        self.name = "a-" + fn.name

    def __call__(self, *args):
        # r =  ( mu / 2 ) ** .5  * ( c - l / mu  )
        return ((self.mu / 2.) ** .5 *
                (self.fn(*args) - self.l / self.mu))


class FeatUniversal():
    def __init__(self, list_feats):
        self.list_feats = list_feats
        self.nr = sum(i.nr for i in self.list_feats)

    def append(self, list_feats):
        self.list_feats += list_feats
        self.nr = sum(i.nr for i in self.list_feats)

    def __call__(self, *args):
        a = 0
        if isinstance(args[0], np.ndarray):
            out = np.zeros(self.nr)
            for f in self.list_feats:
                b = f.nr
                o = f(*args)
                out[a:a + b] = o
                a += b
        else:
            out = jnp.zeros(self.nr)
            for f in self.list_feats:
                b = f.nr
                o = f(*args)
                out = out.at[a:a + b].set(o)
                a += b
        return out


class Feat_terminal():
    name = "goal"

    def __init__(self, goal, weight=1., dist_fn=None):
        self.goal = goal
        self.weight_goal = weight
        self.dist_fn = dist_fn
        if dist_fn is None:
            self.dist_fn = lambda x, y: x - y

    def __call__(self, xx, uu):
        out = self.dist_fn(xx, self.goal)
        return self.weight_goal * out


class Feat_terminalTrailer():
    name = "goal"

    def __init__(self, goal, weight=1.):
        self.goal = goal
        self.weight_goal = weight

    def __call__(self, xx, uu):
        angle_distance = True
        out = xx - self.goal
        if angle_distance:
            out[2] = np.arctan2(np.sin(out[2]), np.cos(out[2]))
            out[3] = np.arctan2(np.sin(out[3]), np.cos(out[3]))
        return self.weight_goal * out


class Feat_control():
    name = "control"

    def __init__(self, weight=1, ref=None):
        self.weight = weight
        self.ref = ref

    def __call__(self, xx, uu):
        if self.ref is None:
            ref = np.zeros(len(uu))
        else:
            ref = self.ref
        return self.weight * (uu - ref)


class Feat_regx():
    name = "regx"

    def __init__(self, weight=1., ref=None):
        self.weight = weight
        self.ref = ref

    def __call__(self, xx, uu):
        if self.ref is None:
            ref = np.zeros_like(xx)
        else:
            ref = self.ref
        return self.weight * (xx - ref)


class Feat_obstacles():
    name = "obs-c"

    def __init__(self, obs, radius, weight):
        self.obs = obs
        self.radius = radius
        self.weight = weight

    def __call__(self, xx, uu):
        x, y = xx[0].item(), xx[1].item()
        p = np.array([x, y])
        dist = np.zeros(len(self.obs))
        for i, o in enumerate(self.obs):
            d = np.linalg.norm(p - o)
            dist[i] = min(0, d - self.radius)
        return self.weight * dist


def plot_feats_raw(xs, us, featss, unone):

    multis = []
    datas = []
    OUT = []
    OUTL = []
    T = len(featss) - 1
    for i in range(T + 1):
        data = FakeData()
        xx = xs[i]
        uu = us[i] if i < T else unone
        out = np.array([])
        out_l = np.array([])
        feat = featss[i]
        for f in feat.list_feats:
            OUT.append([i, f.name, f.fn(xx, uu), f(xx, uu)])
            # print("fname", f.name)
            # print("out", out)
            # print("fn",  f.fn(xx, uu))
            out = np.concatenate((out, f.fn(xx, uu)))
            if f.tt == OT.auglag:
                OUTL.append([i, f.name, f.l])
                out_l = np.concatenate((out_l, f.l))

        multis.append(out_l)
        data = FakeData()
        data.r = out
        datas.append(data)

    D = {}
    for i in OUT:
        if i[1] in D:
            D[i[1]].append((i[0], i[2], i[3]))
        else:
            D[i[1]] = [(i[0], i[2], i[3])]

    Dl = {}
    for i in OUTL:
        if i[1] in Dl:
            Dl[i[1]].append((i[0], i[2]))
        else:
            Dl[i[1]] = [(i[0], i[2])]

    for k in D:
        x_ = [i[0] for i in D[k]]
        y_ = [i[2] for i in D[k]]
        plt.plot(x_, y_, '.-', label=k)

    plt.title("costs")
    plt.legend()
    plt.show()

    for k in D:
        x_ = [i[0] for i in D[k]]
        y_ = [i[1] for i in D[k]]
        plt.plot(x_, y_, '.-', label=k)

    plt.title("feats")
    plt.legend()
    plt.show()

    for k in Dl:
        x_ = [i[0] for i in Dl[k]]
        y_ = [i[1] for i in Dl[k]]
        plt.plot(x_, y_, '.-', label=k)

    plt.title("multipliers")
    plt.legend()
    plt.show()

    # plt.clf()
    # nr = len(datas[0].r)
    # nd = len(datas)
    # r =  []
    # for j in range(nr):
    #     rr = [ np.asscalar(d.r[j]) for d in datas]
    #     r.append(rr)

    # for i in range(len(r)):
    #     plt.plot(r[i], 'o-' ,label="f" + str(i))
    # plt.legend()

    # plt.show()

   # # multis to plot
    # nl = len(multis[0])
    # multi_plot = []
    # for j in range(nl):
    #    multi_plot.append( [ m[j] for m in multis ] )

    # for j in range(len(multi_plot)):
    #     plt.plot(multi_plot[j], 'o-' ,label="l" + str(j))

    # plt.legend()
    # plt.show()
    # multi_plot


def plot_feats(xs, us, featss, unone):
    T = len(featss) - 1
    datas = []
    for i in range(T + 1):
        data = FakeData()
        xx = xs[i]
        uu = us[i] if i < T else unone
        f = featss[i]
        data.r = f(xx, uu)
        datas.append(data)

    plt.clf()
    nr = len(datas[0].r)
    r = []
    for j in range(nr):
        rr = [np.asscalar(d.r[j]) for d in datas]
        r.append(rr)

    for i in range(len(r)):
        plt.plot(r[i], 'o-', label="f" + str(i))
    plt.legend()

    plt.show()


def get_cost(xs, us, problem):
    dd = FakeData()
    c_r = 0
    for i, xu in enumerate(zip(xs[:-1], us)):
        x, u = xu
        x = np.array(x)
        u = np.array(u)
        problem.running_model[i].calc(dd, x, u)
        c_r += dd.cost

    problem.terminal_model.calc(dd, xs[-1])
    c_T = dd.cost
    return c_r, c_T


def auglag_solver(ddp, xs, us, problem, unone, visualize, plot_fn=None,
                  **kwargs):
    """
    Ref: eq. 17.39 Nocedal Numerical optimization v2
    and Algorithm 17.4.

    Augmented Lagragian is:
    L = f - l * c + mu / 2 * c^2
    Lambda update is:
    l' = l - mu * ci

    But I want less-squares formulation:
    L = f + || r || ^2
    We rewrite:
    -l * c + mu / 2 * c^2 = ( mu^.5 / 2^.5 * c - l / mu^.5 / 2^.5 )^2 - l^2 / mu / 2
    CHECK
    ( =  mu / 2 * c^2 - 2 * mu.^5 / 2^.5 * c * l / mu^.5 / 2.^.5  + l^2 / mu / 2 - l^2 / mu / 2  )
    feature is: r = mu^.5 / 2^.5 * c - l / mu^.5 / 2^.5
    r =  ( mu / 2 ) ** .5  * ( c - l / mu  )
    CHECK
    -l*c + mu / 2 * c ** 2 = (c - l/mu)**2  * (mu/2) -  l ** 2 / mu / 2
    And the term l ** 2 / mu / 2 is constant in each auglag iteration.

    NOTE: in current implementation, ineqs are formulated as nonsmooth equality
    constraints:
    ineq : g(x) <= 0 ,  eq: max(g(x), 0)
    Another approach is:
        ineq: g(x) <= 0 and eq:  [ g(x) >= 0 or l > 0 ] g(x)
    and the update: l' = max (l - mu * ci, 0 )
    to ensure that the multipliers of ineqs are always positive.
    where l is the previous lagrange multipler.

    This could be more robust against switching between making
    the constraint active or not. So far, i didn't find any problem with
    the easier formulation max(g(x), 0) -- so I use this.
    """

    mu = kwargs.get("mu", 1000.)  # starting penalty for constraints
    max_it = kwargs.get("max_it", 5)  # num outter iterations in Auglag
    max_it_ddp = kwargs.get("max_it_ddp", 75)  # num inner iterations DDP.
    xtol = kwargs.get("xtol", 1e-5)  # terminate if ||step|| < x_tol
    cmax = kwargs.get("cmax", 10.)  # early termination (unsolved) if c > cmax
    th_c = kwargs.get("th_c", 1e-3)  # terminate (solve) if c < th_c
    T = len(problem.featss) - 1

    problem.add_noise(xs, us)

    xprev = np.copy(xs)
    uprev = np.copy(us)

    print_before_start = True
    if print_before_start and visualize:
        print("Plots before start")
        if plot_fn is not None:
            plot_fn(xs, us)
        plot_feats_raw(xs, us, problem.featss, np.zeros(2))

    total_ddp_time = 0.
    total_ddp_time_py = 0.
    total_ddp_time_npy = 0.

    eta = 1. / mu ** .1

    for i in problem.featss:
        for ii in i.list_feats:
            if ii.tt == OT.auglag:
                ii.mu = mu

    # check cost before of initial guuess
    cost_0 = get_cost(xs, us, problem)
    xs_rollout = []
    x0 = np.array(xs[0])
    x = x0.copy()
    xs_rollout.append(x0)
    dd = FakeData()
    for u in us:
        problem.running_model[0].calc(dd, x, u)
        xs_rollout.append(np.array(dd.xnext.copy()).flatten())
        x = np.array(dd.xnext).flatten()
    cost_roll = get_cost(xs_rollout, us, problem)

    print("AUGLAG SOLVER")
    print(f"Initial mu {mu:.3f} eta {eta:.3f}")
    print("Initial Guess")
    print(f"AL of x0, u0 : running,terminal {cost_0[0]:.3f}, {cost_0[1]:.3f}")
    print(
        f"AL of rollout(u0), u0 : running, terminal {cost_roll[0]:.3f}, {cost_roll[1]:.3f}")

    for it in range(max_it):

        # set penalty in Auglag
        for i in problem.featss:
            for ii in i.list_feats:
                if ii.tt == OT.auglag:
                    ii.mu = mu

        # set timers to zero
        for mm in problem.running_model:
            mm.timer = 0
        problem.terminal_model.timer = 0

        # initial regularization in DDP
        if it == 0:
            regInit = 10.
        else:
            regInit = .1

        print(f"Start ddp iteration {it}")
        a = time.perf_counter()
        ddp.solve(init_xs=xs, init_us=us, maxiter=max_it_ddp,
                  isFeasible=False, regInit=regInit)

        ddp_time = time.perf_counter() - a
        ddp_time_py = np.sum([mm.timer for mm in problem.running_model]) + \
            problem.terminal_model.timer
        ddp_time_npy = ddp_time - ddp_time_py
        print(
            f"End ddp iteration {it}. Time total {ddp_time:.3f} python {ddp_time_py:.3f} c++ {ddp_time_npy:.4f} [s]")
        total_ddp_time += ddp_time
        total_ddp_time_py += ddp_time_py
        total_ddp_time_npy += ddp_time_npy

        xs = ddp.xs
        us = ddp.us

        c = 0
        obj = 0
        for i in range(T + 1):
            xx = xs[i]
            uu = us[i] if i < T else unone
            for f in problem.featss[i].list_feats:
                if f.tt == OT.auglag:
                    c += np.sum(np.abs(f.fn(xx, uu)))
                if f.tt == OT.cost:
                    r = f.fn(xx, uu)
                    obj += .5 * np.dot(r, r)

        if c < eta:
            print("Good reduction in c, updating multipliers")
            for i in range(T + 1):
                xx = xs[i]
                uu = us[i] if i < T else unone
                for f in problem.featss[i].list_feats:
                    if f.tt == OT.auglag:
                        f.l -= mu * f.fn(xx, uu)
            eta /= mu ** .9
        else:
            print("Not enough reduction in c, increasing penalty")
            mu *= 10
            eta = 1. / mu ** 0.1


        delta = np.linalg.norm(xs - xprev) + np.linalg.norm(us - uprev)
        print(f"it {it} f {obj:.4f} c {c:.4f} delta xu {delta:4f} eta {eta:3f}")

        if visualize:
            plt.clf()
            if plot_fn is not None:
                plot_fn(xs, us)
            plot_feats_raw(xs, us, problem.featss, np.zeros(2))

        if delta < xtol:
            print("Terminate:  Criteria = XTOL")
            break
        else:
            xprev = np.copy(xs)
            uprev = np.copy(us)

        if c < th_c:
            print("Terminate: Criteria: = c < th_c. Good")
            break

        if c > cmax:
            print("Terminate Criteria: CMAX -- we can not solve this problem")
            break

    return xs, us


class FeatObstaclesFcl():
    name = "obs-fcl"

    def __init__(self, weight, primitive):
        self.weight = weight
        self.col_primitive = primitive

    def __call__(self, xx, uu):
        # d = min( self.col_primitive.col(xx, uu) , 0)
        _d = self.col_primitive.col(xx, uu)
        # d = self.col_primitive.col(xx, uu)
        if isinstance(xx, np.ndarray):
            d = np.minimum(_d, 0.)
            return self.weight * np.array([d])
        elif isinstance(xx, jnp.ndarray):
            d = jnp.minimum(_d, 0)
            return self.weight * jnp.array([d])
        else:
            assert False


class FeatDiffAngle():
    name = "difA"
    bound = np.pi
    # -A <= x <= A
    # -A -x <= 0
    # -(A+x) <= 0  -> A+x >= 0
    # 0 <= A - x

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, xx, uu):
        if isinstance(xx, np.ndarray):
            dangle = diff_angle(xx[2], xx[3])
            return self.weight * np.array([min(0, self.bound - dangle),
                                           min(0, dangle + self.bound)])
        elif isinstance(xx, jnp.ndarray):
            dangle = diff_angle(xx[2], xx[3], False)
            return self.weight * jnp.array([min(0, self.bound - dangle),
                                            min(0, dangle + self.bound)])
        else:
            assert False
        # return np.array([np.minimum(0, self.bound - np.absolute(dangle))])


class FeatBoundsX():
    name = "bounds"

    def __init__(self, lb, ub, weight):
        self.weight = weight
        self.lb = lb
        self.ub = ub

    def __call__(self, xx, uu):
        if isinstance(xx, np.ndarray):
            r_ub = np.minimum(0, self.ub - xx)  # x <= ub
            r_lb = np.minimum(0, xx - self.lb)  # x >= lb
            return self.weight * np.concatenate((r_lb, r_ub))
        else:
            r_ub = jnp.minimum(0, self.ub - xx)  # x <= ub
            r_lb = jnp.minimum(0, xx - self.lb)  # x >= lb
            return self.weight * jnp.concatenate((r_lb, r_ub))


class OCP_abstract():

    def __init__(self, **kwargs):
        self.T = kwargs.get("T")
        self.min_u = kwargs.get("min_u")
        self.max_u = kwargs.get("max_u")
        self.goal = kwargs.get("goal")
        self.x0 = kwargs.get("x0")
        self.min_x = kwargs.get("min_x")
        self.max_x = kwargs.get("max_x")
        self.cc = kwargs.get("cc")
        self.free_time = kwargs.get("free_time", False)

        self.col_primitive = CollisionJax(self.cc)  # register a primitive

        self.disturbance = 1e-4
        self.weight_goal = 10
        # self.weight_control = np.array([.5, .5])
        self.weight_control = .5
        self.weight_bounds = 5
        self.weight_diff = 10
        self.weight_obstacles = 10
        self.featss = []
        self.running_model = []
        self.running_modelx = []
        self.dist_fn = None
        self.feat_run = lambda: None
        self.feat_terminal = lambda: None
        self.use_finite_diff = True  # default

    def get_ctrl_feat(self):
        return CostTerm(Feat_control(self.weight_control), len(self.min_u), 1.)

    def get_obs_feat(self):
        return AuglagTerm(FeatObstaclesFcl(
            self.weight_obstacles, self.col_primitive), 1)

    def get_goal_feat(self):
        return AuglagTerm(Feat_terminal(
            self.goal, self.weight_goal, self.dist_fn), len(self.goal))

    def get_xbounds_feat(self):
        return AuglagTerm(FeatBoundsX(self.min_x, self.max_x,
                                      self.weight_bounds), 2 * len(self.min_x))

    def add_noise(self, xs, us, k=1.):
        """
        """
        delta = 0.001
        for i in range(len(xs)):
            for j in range(len(xs[i])):
                xs[i][j] += k * delta * (2 * np.random.rand() - 1)
        for i in range(len(us)):
            for j in range(len(us[i])):
                us[i][j] += k * delta * (2 * np.random.rand() - 1)

    def create_problem(self, **kwargs):

        for i in range(self.T):
            self.featss.append(self.feat_run())
            model = self.actionModel(
                self.featss[-1], self.featss[-1].nr, **kwargs)
            model.u_lb = self.min_u
            model.u_ub = self.max_u
            if self.use_finite_diff:
                model = crocoddyl.ActionModelNumDiff(model, True)
                model.disturbance = self.disturbance
                model.u_lb = self.min_u
                model.u_ub = self.max_u
            self.running_model.append(model)

        self.featss.append(self.feat_terminal())

        self.terminal_model = self.actionModel(
            self.featss[self.T], self.featss[self.T].nr, **kwargs)

        if self.use_finite_diff:
            self.terminal_model = crocoddyl.ActionModelNumDiff(
                self.terminal_model, True)
            self.terminal_model.disturbance = 1e-5

        self.problem = crocoddyl.ShootingProblem(
            self.x0, self.running_model, self.terminal_model)

    def recompute_init_guess(self, X, U):
        pass

    def normalize(self, X, U):
        pass


class OCP_unicycle_order2(OCP_abstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.actionModel = AM_unicycle_order2_AD
        self.dist_fn = dist_modpi
        self.weight_regx = 1.
        if self.free_time:
            ureg = np.array([0., 0, .7])
            weight_control = np.array([1., 1., 5.])
        else:
            ureg = np.array([0., 0.])
            weight_control = np.array([1., 1.])
        self.feat_run = lambda: FeatUniversal(
            [self.get_obs_feat(),
             CostTerm(Feat_control(weight_control, ureg), len(self.min_u)),
                CostTerm(Feat_regx(self.weight_regx), 5),
                self.get_xbounds_feat()])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.use_finite_diff = False
        self.create_problem(free_time=self.free_time)

    def recompute_init_guess(self, X, U):
        recompute_initguess_unicycle(X, U, self.free_time)

    def normalize(self, X, U):
        normalize_unicycle(X, U)


def recompute_init_guess_trailer(X, U, free_time=False):
    xprev = .0
    angle_index = [2, 3]
    for aa in angle_index:
        first = True
        for x in X:
            if not first:
                x[aa] = xprev + diff_angle(x[aa], xprev)
            else:
                first = False
            xprev = x[aa]

    if free_time:
        for uu in range(len(U)):
            u = np.concatenate((U[uu], np.array([1.])))
            U[uu] = u


def normalize_trailer(X, U):
    angle_index = [2, 3]
    for aa in angle_index:
        for x in X:
            x[aa] = normalize_angle(x[aa])


class OCP_trailer(OCP_abstract):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.weight_regx = 0.1
        self.use_finite_diff = False
        self.actionModel = AM_Trailer_AD

        self.dist_fn = dist_trailer

        if self.free_time:
            ureg = np.array([0, 0, .7])
            weight_control = np.array([1., 1., 5.])
        else:
            ureg = np.array([0., 0.])
            weight_control = np.array([1., 1.])

        self.feat_run = lambda: FeatUniversal([
            CostTerm(Feat_control(weight_control), len(ureg)), self.get_obs_feat(), CostTerm(
                Feat_regx(self.weight_regx), 4), AuglagTerm(FeatDiffAngle(10.), 2)])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.create_problem(free_time=self.free_time)

    def recompute_init_guess(self, X, U):
        recompute_init_guess_trailer(X, U, free_time=self.free_time)

    def normalize(self, X, U):
        normalize_trailer(X, U)


class Feat_Quat():
    name = "quat"

    def __init__(self, weight=1.):
        self.weight = weight
        # self.quat_idx = [3,4,5,6]

    def __call__(self, xx, uu):
        if isinstance(xx, np.ndarray):
            quat_norm = np.sum(np.square(xx[3:7]))
            out = self.weight * np.array([1. - quat_norm])
        else:
            quat_norm = jnp.sum(jnp.square(xx[3:7]))
            out = self.weight * jnp.array([1. - quat_norm])
        return out


def add_noise_quadcopter(xs, us, k):
    """
    """
    deltas_x = np.array([.001, .001, .001,
                         .001, .001, .001, .001,
                         .0001, .0001, .0001,
                         .0001, .0001, .0001])
    deltas_u = 0.001

    for i in range(len(xs)):
        for j in range(len(xs[i])):
            xs[i][j] += k * deltas_x[j] * (2 * np.random.rand() - 1)
        xs[i][3:7] /= np.linalg.norm(xs[i][3:7])

    for i in range(len(us)):
        for j in range(len(us[i])):
            us[i][j] += k * deltas_u * (2 * np.random.rand() - 1)


def add_u_time(U):
    # add time as control
    for uu in range(len(U)):
        u = np.concatenate((U[uu], np.array([1.])))
        U[uu] = u


class OCP_quadrotor(OCP_abstract):
    def __init__(self, free_time=False, use_jit=False, **kwargs):
        super().__init__(**kwargs)

        self.free_time = free_time
        self.use_finite_diff = False

        self.actionModel = AM_quadrotor

        self.dist_fn = dist_quadcopter

        hover_u = 0.0833
        ureg = hover_u * np.ones(4)
        weight_control = 10 * np.ones(4)
        # weight_regx = .01

        if self.free_time:
            ureg = np.append(ureg, [.8])
            weight_control = np.append(weight_control, [1.])

        self.feat_run = lambda: FeatUniversal(
            [CostTerm(Feat_control(weight_control, ureg), len(self.min_u)),
             # CostTerm(Feat_regx(weight_regx), 13, 1.),
             AuglagTerm(FeatBoundsX(jnp.array(self.min_x), jnp.array(self.max_x),
                                    self.weight_bounds), 2 * len(self.min_x))])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.create_problem(use_jit=use_jit, free_time=self.free_time)

    def add_noise(self, xs, us, k=1.):
        add_noise_quadcopter(xs, us, k)

    def recompute_init_guess(self, xs, us):
        if self.free_time:
            add_u_time(us)


def recompute_initguess_unicycle(X, U, free_time):
    """
    Works for first and second order unicycles
    """
    xprev = .0
    angle_index = [2]
    for aa in angle_index:
        first = True
        for x in X:
            if not first:
                x[aa] = xprev + diff_angle(x[aa], xprev)
            else:
                first = False
            xprev = x[aa]

    if free_time:
        # add time as control
        for uu in range(len(U)):
            u = np.concatenate((U[uu], np.array([1.])))
            U[uu] = u


def normalize_unicycle(X, U):
    angle_index = [2]
    for aa in angle_index:
        for x in X:
            x[aa] = normalize_angle(x[aa])


class OCP_unicycle_order1(OCP_abstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.actionModel = AM_unicycle_order1_AD
        self.dist_fn = dist_modpi
        self.weight_regx = 1.
        if self.free_time:
            ureg = np.array([0, 0, .7])
            weight_control = np.array([1., 1., 5.])
        else:
            ureg = np.array([0., 0.])
            weight_control = np.array([1., 1])
        self.feat_run = lambda: FeatUniversal(
            [CostTerm(Feat_control(weight_control, ureg), len(self.min_u)),
             self.get_obs_feat(), CostTerm(Feat_regx(self.weight_regx), 3),
             ])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.use_finite_diff = False
        self.create_problem(free_time=self.free_time)

    def recompute_init_guess(self, X, U):
        recompute_initguess_unicycle(X, U, self.free_time)

    def normalize(self, X, U):
        normalize_unicycle(X, U)
