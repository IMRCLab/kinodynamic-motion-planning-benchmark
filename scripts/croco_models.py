import crocoddyl
import numpy as np
import matplotlib.pylab as plt
from unicycle_utils import *


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def diff_angle(angle1, angle2):
    return np.arctan2(np.sin(angle1 - angle2), np.cos(angle1 - angle2))


class ActionModelUnicycleSecondOrder(crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr):
        self.dt = .1
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(5), 2, nr)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.features = features

    def calc(self, data, xx, u=None):
        if u is None:
            u = self.unone
        # Getting the state and control variables

        x, y, theta, v, w = xx[0].item(), xx[1].item(
        ), xx[2].item(), xx[3].item(), xx[4].item()
        c = np.cos(theta)
        s = np.sin(theta)
        a, aw = u[0].item(), u[1].item()
        dt = self.dt

        xdot = v * c
        ydot = v * s
        data.xnext = np.matrix([x + xdot * dt, y + ydot * dt, theta + w * dt,
                                v + a * dt,
                                w + aw * dt]).T

        data.r = self.features(xx, u)
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


# return RobotCarFirstOrderWithTrailers(-0.1, 0.5, -np.pi/3, np.pi/3,
# 0.25, [0.5])

class ActionModelCarTrailer(crocoddyl.ActionModelAbstract):
    L = .25
    hitch_lengths = [.5]

    def __init__(self, features, nr):
        self.dt = .1
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(4), 2, nr)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.features = features

    def calc(self, data, xx, u=None):
        if u is None:
            u = self.unone
        # Getting the state and control variables

        state = xx
        action = u[0].item(), u[1].item()
        x, y, yaw = state[0].item(), state[1].item(), state[2].item()
        v, phi = action

        yaw_next = yaw + v / self.L * np.tan(phi) * self.dt
        # normalize yaw between -pi and pi
        # yaw_next_norm = normalize_angle(yaw_next)
        yaw_next_norm = yaw_next
        x_next = x + v * np.cos(yaw) * self.dt
        y_next = y + v * np.sin(yaw) * self.dt

        state_next_list = [x_next, y_next, yaw_next_norm]

        for i, d in enumerate(self.hitch_lengths):
            theta_dot = v / d
            for j in range(0, i):
                theta_dot *= np.cos(state[2 + j] - state[3 + j])
            theta_dot *= np.sin(state[2 + i] - state[3 + i])

            theta = state[3 + i]
            theta_next = theta + theta_dot * self.dt
            # theta_next_norm = normalize_angle(theta_next)
            theta_next_norm = theta_next
            state_next_list.append(theta_next_norm)

        data.xnext = np.matrix(state_next_list).T

        data.r = self.features(xx, u)
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


class ActionModelUnicycleFirstOrder(crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr):
        self.dt = .1
        crocoddyl.ActionModelAbstract.__init__(
            self, crocoddyl.StateVector(3), 2, nr)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.features = features

    def calc(self, data, xx, u=None):
        if u is None:
            u = self.unone
        # Getting the state and control variables

        x, y, theta = xx[0].item(), xx[1].item(), xx[2].item()
        c = np.cos(theta)
        s = np.sin(theta)
        v, w = u[0].item(), u[1].item()
        dt = self.dt

        xdot = v * c
        ydot = v * s
        data.xnext = np.matrix(
            [x + xdot * dt, y + ydot * dt, theta + w * dt]).T

        data.r = self.features(xx, u)
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


class FakeData():
    xnext = np.matrix([])
    r = np.matrix([])
    cost = 0.


class OT():
    none = -1
    cost = 0
    penalty = 1
    auglag = 2


class PenaltyFeat():
    mu = 1.
    tt = OT.penalty

    def __init__(self, fun, nr, scale):
        self.fun = fun
        self.nr = nr
        self.scale = scale

    def __call__(self, *args):
        # TODO: what happens with the 1/2?
        return self.scale * (self.mu / 2.0)**.5 * self.fun(*args)


class CostFeat():
    scale = 1.
    nr = 0
    tt = OT.cost
    def fun(x, u): return u  # a funcion

    def __init__(self, fun, nr, scale):
        self.fun = fun
        self.nr = nr
        self.scale = scale
        self.name = "r-" + fun.name

    def __call__(self, *args):
        return self.scale * self.fun(*args)


class AuglagFeat():
    def fun(x, u): return u  # a funcion
    mu = 1.
    scale = 1.
    tt = OT.auglag
    l = np.array([])

    def __init__(self, fun, nr, scale):
        self.fun = fun
        self.nr = nr
        self.scale = scale
        self.l = np.zeros(nr)
        self.name = "a-" + fun.name

    def __call__(self, *args):
        return self.scale * ((self.mu / 2.) ** .5 *
                             (self.fun(*args) - self.l / self.mu))
# r =  ( mu / 2 ) ** .5  * ( c - l / mu  )


class FeatUniversal():
    def __init__(self, list_feats):
        self.list_feats = list_feats
        self.nr = sum(i.nr for i in self.list_feats)

    def append(self, list_feats):
        self.list_feats += list_feats
        self.nr = sum(i.nr for i in self.list_feats)

    def __call__(self, *args):
        out = np.zeros(self.nr)
        a = 0
        for f in self.list_feats:
            b = f.nr
            o = f(*args)
            out[a:a + b] = o
            a += b
        return out


class Feat_terminal():
    name = "goal"

    def __init__(self, goal, weight=1., dist_fun=None):
        self.goal = goal
        self.weight_goal = weight
        self.dist_fun = dist_fun
        if dist_fun is None:
            self.dist_fun = lambda x, y: x - y

    def __call__(self, xx, uu):
        # angle_distance = True
        out = self.dist_fun(xx, self.goal)
        # out = xx - self.goal
        # if angle_distance:
        #     out[2] = np.arctan2( np.sin( out[2] ) , np.cos( out[2]))
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

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, xx, uu):
        return self.weight * uu


class Feat_obstacles():
    name = "obs-c"

    def __init__(self, obs, radius, weight):
        self.obs = obs
        self.radius = radius
        self.weight = weight

    def __call__(self, xx, uu):
        x, y = xx[0].item(), xx[1].item()
        # np.array
        p = np.array([x, y])
        dist = np.zeros(len(self.obs))
        for i, o in enumerate(self.obs):
            # compute distance
            d = np.linalg.norm(p - o)
            dist[i] = min(0, d - self.radius)
        # distance to goal
        return self.weight * dist


def penalty_solver(ddp, xs, us, featss, visualize, plot_fun):
    penalty = [1, 5, 10, 50, 100, 200]
    for p in penalty:

        # weight_goal = weight_goal_ref * p ** .5
        # weight_obstacles = weight_goal_ref * p ** .5
        # weight_control = weight_control_ref
        # done = ddp.solve(init_xs=xs,init_us=us)
        # assert done

        for i in featss:
            for ii in i.list_feats:
                if ii.tt == OT.penalty or ii.tt == OT.auglag:
                    ii.mu = p

        done = ddp.solve(init_xs=xs, init_us=us)

        if visualize:
            plot_fun(ddp.xs, ddp.us)
            # plt.clf()
            # for x in ddp.xs:
            #     plotUnicycle(x)
            # plt.scatter(ddp.xs[-1][0],ddp.xs[-1][1],c="yellow",marker=".")

            # extra_plot(plt)

            # plt.axis([-2, 2, -2, 2])
            # plt.show()
        # plot

        xs = ddp.xs
        us = ddp.us
    return xs, us


def plot_feats_raw(xs, us, featss, unone):

    # update multipliers
    multis = []
    datas = []
    OUT = []
    OUTL = []
    T = len(featss) - 1
    for i in range(T + 1):
        data = FakeData()
        xx = xs[i]
        if i < T:
            # m = rM_basic[i]
            uu = us[i]
        else:
            # m = rT_basic
            uu = unone
        # m.calc(data,xx,uu)
        # print("i" , i )
        out = np.array([])
        out_l = np.array([])
        feat = featss[i]
        for f in feat.list_feats:
            OUT.append([i, f.name, f.fun(xx, uu), f(xx, uu)])
            out = np.concatenate((out, f.scale * f.fun(xx, uu)))
            if f.tt == OT.auglag:
                OUTL.append([i, f.name, f.l])
                out_l = np.concatenate((out_l, f.l))

        multis.append(out_l)
        data = FakeData()
        data.r = out
        datas.append(data)

    # print the data

    # print OUT:
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

    # print :)

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
        if i < T:
            # m = rM_basic[i]
            uu = us[i]
        else:
            # m = rT_basic
            uu = unone
        f = featss[i]
        data.r = f(xx, uu)
        # m.calc(data,xx,uu)
        datas.append(data)

    # print the data

    plt.clf()
    nr = len(datas[0].r)
    nd = len(datas)
    r = []
    for j in range(nr):
        rr = [np.asscalar(d.r[j]) for d in datas]
        r.append(rr)

    for i in range(len(r)):
        plt.plot(r[i], 'o-', label="f" + str(i))
    plt.legend()

    plt.show()


def auglag_solver(ddp, xs, us, problem, unone, visualize, plot_fun, **kwargs):
    """
    update lagrange multipliers
    Eq. 17.39 Nocedal Numerical optimization v2
    lamba <-- lambda - mu * ci
    Using augmented Lagragian L = f - lambda * c + mu / 2 * c^2
    -lambda * c + mu / 2 * c^ 2 = ( mu^.5 / 2^.5 * c - lambda / mu^.5 / 2^.5 )^2 - lambda^2 / mu / 2
    feature is: r = mu^.5 / 2^.5 * c - lambda / mu^.5 / 2^.5
    r =  ( mu / 2 ) ** .5  * ( c - l / mu  )
    -l*c + mu / 2 * c ** 2 = (c - l/mu)**2  * (mu/2) -  l ** 2 / mu / 2
    """

    mu = kwargs.get("mu", 100)
    max_it = kwargs.get("max_it", 10)
    xtol = kwargs.get("xtol", 1e-4)
    T = len(problem.featss) - 1

    for i in problem.featss:
        for ii in i.list_feats:
            if ii.tt == OT.auglag:
                ii.mu = mu

    xprev = np.copy(xs)
    uprev = np.copy(us)
    for it in range(max_it):

        #
        done = ddp.solve(init_xs=xs, init_us=us)
        xs = ddp.xs
        us = ddp.us

        # update multipliers
        # datas = []
        print("udpate multiies")
        for i in range(T + 1):
            # data = FakeData()
            xx = xs[i]
            if i < T:
                # m = rM_basic[i]
                uu = us[i]
            else:
                # m = rT_basic
                uu = unone
            for f in problem.featss[i].list_feats:
                if f.tt == OT.auglag:
                    print("name", f.name)
                    print(f.fun(xx, uu))
                    print(mu * f.scale)
                    f.l -= mu * f.scale * f.fun(xx, uu)
                    print("f.l", f.l)

        if visualize:
            plt.clf()
            plot_fun(xs, us)
            # plot_feats(xs,us,problem.featss, np.zeros(2))
            plot_feats_raw(xs, us, problem.featss, np.zeros(2))

        if np.linalg.norm(xs - xprev) + np.linalg.norm(us - uprev) < xtol:
            print("Terminate Criteria: XTOL")
            break
        else:
            xprev = np.copy(xs)
            uprev = np.copy(us)

    return xs, us


class FeatObstaclesFcl():
    name = "obs-fcl"

    def __init__(self, weight, cc):
        self.weight = weight
        self.cc = cc  # collision checker against the environment

    def __call__(self, xx, uu):
        # x,y,theta= xx[0].item(), xx[1].item(),xx[2].item()
        dist = np.zeros(1)
        dis, _, _ = self.cc.distance(xx)  # Float, Vector, Vector
        # TODO: Only one min distance per time step. Is this a real limitation?
        dist[0] = min(0, dis)
        return self.weight * dist


class FeatDiffAngle():
    name = "difA"
    bound = np.pi / 4

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, xx, uu):
        dangle = diff_angle(xx[2], xx[3])
        return np.array([np.minimum(0, self.bound - np.absolute(dangle))])


class FeatBoundsX():
    name = "bounds"

    def __init__(self, lb, ub, weight):
        self.weight = weight
        self.lb = lb
        self.ub = ub

    def __call__(self, xx, uu):
        # x < ub
        r_ub = np.minimum(0, self.ub - xx)
        # x > lb
        r_lb = np.minimum(0, xx - self.lb)
        return np.concatenate((r_lb, r_ub))


class CroAbstract():

    weight_goal = 10
    weight_control = 1
    weight_bounds = 5
    weight_diff = 5
    weight_obstacles = 10
    featss = []
    rM = []
    dist_fun = None
    def feat_run(): return None
    def feat_terminal(): return None
    actionModel = None

    def __init__(self, **kwargs):
        self.T = kwargs.get("T")
        self.min_u = kwargs.get("min_u")
        self.max_u = kwargs.get("max_u")
        self.goal = kwargs.get("goal")
        self.x0 = kwargs.get("x0")
        self.min_x = kwargs.get("min_x")
        self.max_x = kwargs.get("max_x")
        self.cc = kwargs.get("cc")

    def get_ctrl_feat(self):
        return CostFeat(Feat_control(self.weight_control), len(self.min_u), 1.)

    def get_obs_feat(self):
        return AuglagFeat(FeatObstaclesFcl(
            self.weight_obstacles, self.cc), 1, 1.)

    def get_goal_feat(self):
        return AuglagFeat(Feat_terminal(
            self.goal, self.weight_goal, self.dist_fun), len(self.goal), 1.0)

    def get_xbounds_feat(self):
        return AuglagFeat(FeatBoundsX(self.min_x, self.max_x,
                                      self.weight_bounds), 2 * len(self.min_x), 1.)

    def create_problem(self):

        for i in range(self.T):
            self.featss.append(self.feat_run())
            am = crocoddyl.ActionModelNumDiff(self.actionModel(
                self.featss[-1], self.featss[-1].nr), True)
            am.disturbance = 1e-5
            am.u_lb = self.min_u
            am.u_ub = self.max_u
            self.rM.append(am)

        self.featss.append(self.feat_terminal())
        self.rT = crocoddyl.ActionModelNumDiff(self.actionModel(
            self.featss[self.T], self.featss[self.T].nr), True)
        self.rT.disturbance = 1e-5
        self.problem = crocoddyl.ShootingProblem(self.x0, self.rM, self.rT)

    def recompute_init_guess(self, X, U):
        return X, U


class CroRobotUnicycleFirstOrder(CroAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def dist(x, y):
            d = x - y
            d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
            return d

        self.actionModel = ActionModelUnicycleFirstOrder
        self.dist_fun = dist
        self.feat_run = lambda: FeatUniversal(
            [self.get_ctrl_feat(), self.get_obs_feat()])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.create_problem()

    def recompute_init_guess(self, X, U):
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
        return X, U


class CroRobotUnicycleSecondOrder(CroAbstract):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def dist(x, y):
            d = x - y
            d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
            return d

        self.actionModel = ActionModelUnicycleSecondOrder
        self.dist_fun = dist
        self.feat_run = lambda: FeatUniversal(
            [self.get_ctrl_feat(), self.get_obs_feat(), self.get_xbounds_feat()])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.create_problem()

    def recompute_init_guess(self, X, U):
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
        return X, U


class CarFirstOrder1Trailers(CroAbstract):

    weight_diff = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        def dist(x, y):
            d = x - y
            d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
            d[3] = np.arctan2(np.sin(d[3]), np.cos(d[3]))
            return d

        self.actionModel = ActionModelCarTrailer
        self.dist_fun = dist
        self.feat_run = lambda: FeatUniversal([self.get_ctrl_feat(
        ), self.get_obs_feat(), AuglagFeat(FeatDiffAngle(self.weight_diff), 1, 1.)])
        self.feat_terminal = lambda: FeatUniversal([self.get_goal_feat()])
        self.create_problem()

    def recompute_init_guess(self, X, U):
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
        return X, U
