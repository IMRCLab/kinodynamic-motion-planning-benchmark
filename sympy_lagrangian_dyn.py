

import sympy as sp
from sympy import latex
from sympy.physics.mechanics import *


# %%
print("cartpole")
x, o = dynamicsymbols('x o')
vx,vo = dynamicsymbols('x o', 1)


m_c = sp.Symbol("m_c")
m_p = sp.Symbol("m_p")
l = sp.Symbol("l")
g = sp.Symbol("g")
t = sp.Symbol("t")
f_x = sp.Symbol("f_x")

T_car = .5 * m_c * vx * vx
T_pendulum = .5 * m_p * ( ( vx  +  l * sp.cos(o) * vo ) ** 2 + (vo * l * sp.sin(o) ) ** 2)

U_pendulum = - m_p * g * l * sp.cos(o)

T = T_car + T_pendulum
U = U_pendulum

print( sp.simplify(sp.expand(T_pendulum)))

L = sp.simplify(sp.expand( T - U ))
eq = sp.diff(L, [x,o]) - sp.diff(sp.diff(L,[vx,vo]), t)

def _gradient(f, v): return sp.Matrix([f]).jacobian(v)

# generalized forces:
F = [f_x, 0]
# eq = sp.jacobian(L, [x, z, o]) - sp.diff( sp.jacobian( L , [vx, vz, w] ) ,t )
eq = - _gradient(L, [x, o]) + \
    sp.diff(_gradient(L, [vx, vo]), t) - sp.Matrix(F).T




# {Derivative(z(t), (t, 2)): -g - f1(t)*cos(o(t))/m - f2(t)*cos(o(t))/m}
# >>> sp.solve(eq, sp.Derivative(o, (t, 2)) )
# {Derivative(o(t), (t, 2)): -f1(t)/I + f2(t)/I}



exp = latex(sp.simplify(eq))

with open("tmp.tex", "w") as f:
    f.write(exp.replace("&" , r"\\"))


# xdotdot = sp.solve(eq, sp.Derivative(x, (t, 2)) )
# odotdot = sp.solve(eq, sp.Derivative(o, (t, 2)) )


sol = sp.solve(eq, [  sp.Derivative(x, (t, 2)), sp.Derivative(o,(t,2))] )

sol_list = [ sp.simplify(i) for i in list(sol.values()) ]

xdotdot = sol_list[0]
odotdot = sol_list[1]


def nice_latex(inn: str): 

    out = inn[:]

    changes = [
        ( r"o(t)" , "o" ) , 
        ( r"x(t)" , "x" ) , 
        ( r"\frac{d}{d t} o" , r"\dot{o}"   ) ,
        ( r"\frac{d}{d t} x" , r"\dot{x}"   ) ,
        (r"{\left(t \right)}","")
    ]

    for change in changes:
        out = out.replace(change[0] , change[1])
    return out


with open("tmp_odotdot.tex", "w") as f:
    f.write(nice_latex(latex(sp.simplify(odotdot))))

with open("tmp_xdotdot.tex", "w") as f:
    f.write(nice_latex(latex(sp.simplify(xdotdot))))



# %%




##
l = sp.Symbol("l")
m = sp.Symbol("m")
g = sp.Symbol("g")
t = sp.Symbol('t')
o = dynamicsymbols('o')
w = dynamicsymbols('o', 1)
T = .5 * m * w * w
V = - m * g * l * sp.cos(o)
L = T - V
eq = sp.diff(L, o) - sp.diff(sp.diff(L, w), t)
print(eq)
LM = LagrangesMethod(L, [o])
##

# what about a drone?


f1, f2 = sp.symbols('f1 f2')
l = sp.Symbol('l')
x, z, o = dynamicsymbols('x z o')
vx, vz, w = dynamicsymbols('x z o', 1)

m = sp.Symbol("m")
I = sp.Symbol("I")
T = .5 * m * (vx * vx + vz * vz) + .5 * I * w * w
V = m * z * g
L = T - V


def gradient(f, v): return sp.Matrix([f]).jacobian(v)
# - F


# generalized forces:
F = [(f1 + f2) * sp.sin(o), (f1 + f2) * sp.cos(o), l * (f1 - f2)]
# eq = sp.jacobian(L, [x, z, o]) - sp.diff( sp.jacobian( L , [vx, vz, w] ) ,t )
eq = gradient(L, [x, z, o]) - \
    sp.diff(gradient(L, [vx, vz, w]), t) - sp.Matrix(F).T

# i can get the solutions with:
# >>> sp.solve(eq, sp.Derivative(z, (t, 2)) )
# {Derivative(z(t), (t, 2)): -g - f1(t)*cos(o(t))/m - f2(t)*cos(o(t))/m}
# >>> sp.solve(eq, sp.Derivative(o, (t, 2)) )
# {Derivative(o(t), (t, 2)): -f1(t)/I + f2(t)/I}
