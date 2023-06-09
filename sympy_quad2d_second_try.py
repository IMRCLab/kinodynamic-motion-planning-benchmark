import sympy as sp
from sympy import latex
from sympy.physics.mechanics import *
from datetime import datetime


q, r, o, l, m, m_p, I, g, Tx, Ty, ddx, ddy, ddo, ddq, do, dq, f1, f2 = sp.symbols(
    "q, r, o, l, m,m_p, I, g, Tx, Ty, ddx, ddy, ddo, ddq, do, dq, f1, f2")


ax = ddx + (ddo + ddq) * sp.cos(o + q) * r + \
    (do + dq) ** 2 * r * -1 * sp.sin(o + q)
ay = ddy + (ddo + ddq) * sp.sin(o + q) * r + \
    (do + dq) ** 2 * r * 1 * sp.cos(o + q)


eq1 = I * ddo - 1 * (f1 - f2) * l
eq2 = m * ddy - 1 * (-Ty + sp.cos(o) * (f1 + f2) - m * g)
eq3 = m * ddx - 1 * (-Tx - sp.sin(o) * (f1 + f2))
eq4 = Tx * sp.cos(o + q) + Ty * sp.sin(o + q)
eq5 = m_p * ax - Tx
eq6 = m_p * ay - Ty + m_p * g

res = sp.solve([eq1, eq2, eq3, eq4, eq5, eq6], [ddx, ddy, ddo, ddq, Tx, Ty])

ax = ddx
ay = ddy

eq1 = I * ddo - 1 * (f1 - f2) * l
eq2 = m * ddy - 1 * (-Ty + sp.cos(o) * (f1 + f2) - m * g)
eq3 = m * ddx - 1 * (-Tx - sp.sin(o) * (f1 + f2))
eq5 = m_p * ax - Tx
eq6 = m_p * ay - Ty + m_p * g


res2 = sp.solve([eq1, eq2, eq3, eq5, eq6], [ddx, ddy, ddo, Tx, Ty])

print(res2[ddx].subs({o: .1, m: .2, m_p: .3, f1: .4, f2: .5}))
print(res[ddx].subs({o: .1, m: .2, m_p: .3, f1: .4, f2: .5, r: 0, q: 0}))


# from other script
# 0.943537922059427
# 2.44168573493203 - 1.0*g
# -0.1*l/I
# -10.7870746185946 + 0.1*l/I


print(res[ddx].subs({o: .1, m: .2, m_p: .3, f1: .4, f2: .5, r: 0.2, q: 0.5, 
                     dq: .2 ,
                     do: .7 }))

print(res[ddy].subs({o: .1, m: .2, m_p: .3, f1: .4, f2: .5, r: 0.2, q: 0.5, 
                     dq: .2 ,
                     do: .7 }))


print(res[ddo].subs({o: .1, m: .2, m_p: .3, f1: .4, f2: .5, r: 0.2, q: 0.5, 
                     dq: .2 ,
                     do: .7 }))


print(res[ddq].subs({o: .1, m: .2, m_p: .3, f1: .4, f2: .5, r: 0.2, q: 0.5, 
                     dq: .2 ,
                     do: .7 }))


# _xdotdot.subs({_o: .1, m: .2, m_p: .3, f_1: .4, f_2: .5,
#               r: 0.2, _q: 0.5, _vq: .2, _w: .7})


