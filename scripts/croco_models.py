import crocoddyl
import numpy as np
import matplotlib.pylab as plt
from unicycle_utils import *

class ActionModelUnicycleSecondOrder(crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr):
        self.dt = .1
        crocoddyl.ActionModelAbstract.__init__(self,crocoddyl.StateVector(5),2, nr)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.features = features

    def calc(self, data, xx, u=None):
        if u is None: u = self.unone
        # Getting the state and control variables

        x,y,theta,v,w = xx[0].item(), xx[1].item(), xx[2].item(), xx[3].item(), xx[4].item()
        c = np.cos(theta);
        s = np.sin(theta);
        a,aw = u[0].item(), u[1].item()
        dt = self.dt

        xdot = v * c
        ydot = v * s
        data.xnext = np.matrix([x + xdot*dt, y + ydot*dt , theta + w * dt,
            v + a * dt , 
            w + aw * dt ]).T

        data.r = self.features(xx,u)
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass




class ActionModelUnicycle2(crocoddyl.ActionModelAbstract):
    def __init__(self, features, nr):
        self.dt = .1
        crocoddyl.ActionModelAbstract.__init__(self,crocoddyl.StateVector(3),2, nr)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)
        self.features = features

    def calc(self, data, xx, u=None):
        if u is None: u = self.unone
        # Getting the state and control variables

        x,y,theta = xx[0].item(), xx[1].item(), xx[2].item()
        c = np.cos(theta);
        s = np.sin(theta);
        v,w = u[0].item(), u[1].item()
        dt = self.dt

        xdot = v * c
        ydot = v * s
        data.xnext = np.matrix([x + xdot*dt, y + ydot*dt , theta + w * dt ]).T

        data.r = self.features(xx,u)
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
    def __init__(self,fun, nr, scale):
        self.fun = fun
        self.nr = nr
        self.scale = scale
    def __call__(self,*args):
        # TODO: what happens with the 1/2?
        return self.scale * (self.mu / 2.0 )**.5 * self.fun(*args)

class CostFeat():
    scale = 1.
    nr = 0
    tt = OT.cost
    fun = lambda x,u : u  # a funcion 
    def __init__(self,fun, nr,scale):
        self.fun = fun
        self.nr = nr
        self.scale = scale
    def __call__(self,*args):
        return  self.scale * self.fun(*args)

class AuglagFeat():
    fun = lambda x,u : u  # a funcion 
    mu = 1.
    scale = 1.
    tt = OT.auglag
    l = np.array([])

    def __init__(self,fun, nr,scale):
        self.fun = fun
        self.nr = nr
        self.scale = scale
        self.l = np.zeros(nr)
    def __call__(self,*args):
        return self.scale * ( ( self.mu / 2. ) ** .5 * ( self.fun(*args) - self.l / self.mu) )
# r =  ( mu / 2 ) ** .5  * ( c - l / mu  )

class FeatUniversal():
    def __init__(self,list_feats):
        self.list_feats = list_feats
        self.nr = sum( i.nr for i in self.list_feats)
    def __call__(self, *args):
        out = np.zeros(self.nr)
        a = 0
        for f in self.list_feats:
            b = f.nr
            o = f(*args)
            out[a:a+b] = o
            a += b
        return out

class Feat_terminal():
    def __init__(self,goal,weight=1.):
        self.goal = goal
        self.weight_goal = weight
    def __call__(self,xx,uu):
        return self.weight_goal * (xx - self.goal)

class Feat_control():
    def __init__(self,weight=1.):
        self.weight = weight
    def __call__(self,xx,uu):
        return self.weight * uu 

class Feat_obstacles( ):
    def __init__(self,obs, radius,  weight):
        self.obs = obs
        self.radius = radius
        self.weight=weight
    def __call__(self,xx,uu):
        x,y= xx[0].item(), xx[1].item()
        # np.array
        p = np.array([x,y])
        dist = np.zeros(len(self.obs))
        for i , o in enumerate(self.obs):
            # compute distance 
            d = np.linalg.norm( p - o )
            dist[i] = min( 0,  d - self.radius)
        # distance to goal
        return self.weight * dist

def penalty_solver(ddp, xs , us, featss, visualize, extra_plot):
    penalty = [1,5,10,50,100,200]
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

        done = ddp.solve(init_xs=xs,init_us=us)

        if visualize:
            plt.clf()
            for x in ddp.xs: 
                plotUnicycle(x)

            extra_plot(plt)

            plt.axis([-2, 2, -2, 2])
            plt.show()
        # plot

        xs= ddp.xs
        us= ddp.us
    return xs, us

def plot_feats_raw(xs,us, featss, unone ):

    # update multipliers
    multis = []
    datas = []
    T = len(featss)-1
    for i in range(T+1):
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
        for f  in feat.list_feats:
            out = np.concatenate((out, f.scale * f.fun(xx,uu)))
            if f.tt == OT.auglag:
                # update the multipliers
                out_l = np.concatenate((out_l, f.l))

        multis.append(out_l)
        data = FakeData()
        data.r = out
        datas.append(data)

    # print the data

    plt.clf()
    nr = len(datas[0].r)
    nd = len(datas)
    r =  []  
    for j in range(nr):
        rr = [ np.asscalar(d.r[j]) for d in datas]
        r.append(rr)

    for i in range(len(r)):
        plt.plot(r[i], 'o-' ,label="f" + str(i)) 
    plt.legend()

    plt.show()

   # multis to plot  
    nl = len(multis[0])
    multi_plot = []
    for j in range(nl):
       multi_plot.append( [ m[j] for m in multis ] )

    for j in range(len(multi_plot)):
        plt.plot(multi_plot[j], 'o-' ,label="l" + str(j)) 

    plt.legend()
    plt.show()
    multi_plot


def plot_feats(xs,us,featss,unone):
    T = len(featss)-1
    datas = []
    for i in range(T+1):
        data = FakeData()
        xx = xs[i]
        if i < T:
            # m = rM_basic[i]
            uu = us[i]
        else:
            # m = rT_basic
            uu = unone
        f = featss[i]
        data.r = f(xx,uu)
        # m.calc(data,xx,uu)
        datas.append(data)

    # print the data

    plt.clf()
    nr = len(datas[0].r)
    nd = len(datas)
    r =  []  
    for j in range(nr):
        rr = [ np.asscalar(d.r[j]) for d in datas]
        r.append(rr)

    for i in range(len(r)):
        plt.plot(r[i], 'o-' ,label="f" + str(i)) 
    plt.legend()

    plt.show()





def auglag_solver(ddp, xs, us, featss, unone,  visualize , extra_plot):
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

    mu = 100.0
    num_it = 10
    T = len(featss) - 1

    for i in featss:
        for ii in i.list_feats:
            if ii.tt == OT.auglag:
                ii.mu = mu

    for it in range(num_it): 

        #
        done = ddp.solve(init_xs=xs,init_us=us)
        xs = ddp.xs
        us = ddp.us

        # update multipliers
        # datas = []
        for i in range(T+1):
            # data = FakeData()
            xx = xs[i]
            if i < T:
                # m = rM_basic[i]
                uu = us[i]
            else:
                # m = rT_basic
                uu = unone
            for f  in featss[i].list_feats:
                if f.tt == OT.auglag:
                    f.l -= mu * f.scale *  f.fun(xx,uu)

        if visualize:
            plt.clf()
            for x in ddp.xs: 
                plotUnicycle(x)

            extra_plot(plt)

            plt.axis([-2, 2, -2, 2])
            plt.show()

            plot_feats(xs,us,featss, np.zeros(2))
            plot_feats_raw(xs,us, featss, np.zeros(2))

    return xs, us




