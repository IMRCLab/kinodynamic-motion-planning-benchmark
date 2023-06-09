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
eq = - _gradient(L, [x, o]) + \
    sp.diff(_gradient(L, [vx, vo]), t) - sp.Matrix(F).T




exp = latex(sp.simplify(eq))

with open("tmp.tex", "w") as f:
    f.write(exp.replace("&" , r"\\"))



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
# I can create c code here...




