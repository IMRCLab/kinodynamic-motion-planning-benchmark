import sympy as sp
from sympy import latex
from sympy.physics.mechanics import *
from datetime import datetime


##

def nice_latex(inn: str):

    out = inn[:]

    changes = [
        (r"\frac{d}{d t} o", r"\dot{o}"),
        (r"\frac{d}{d t} x", r"\dot{x}"),
        (r"\frac{d}{d t} y", r"\dot{y}"),
        (r"\frac{d}{d t} q", r"\dot{q}"),
        (r"{\left(t \right)}", "")
    ]

    for change in changes:
        out = out.replace(change[0], change[1])
    return out


##
l, m, I, g, t = sp.symbols("l m I g t")
f_1, f_2 = sp.symbols('f1 f2')
x, y, o, q = dynamicsymbols('x y o q')
v_x, v_y, w, vq = dynamicsymbols('x y o q', 1)


def _gradient(f, v): return sp.Matrix([f]).jacobian(v)


T = .5 * I * w * w + .5 * m * (v_x ** 2 + v_y ** 2)
U = m * g * y
L = T - U

F = [-(f_1 + f_2) * sp.sin(o), (f_1 + f_2) * sp.cos(o), l * (f_1 - f_2)]

eq = - _gradient(L, [x, y, o]) + \
    sp.diff(_gradient(L, [v_x, v_y, w]), t) - sp.Matrix(F).T


sol = sp.solve(eq, [sp.Derivative(x, (t, 2)), sp.Derivative(y, (t, 2)),
                    sp.Derivative(o, (t, 2))])


xdotdot = sp.simplify(sol[sp.Derivative(x, (t, 2))])
ydotdot = sp.simplify(sol[sp.Derivative(y, (t, 2))])
odotdot = sp.simplify(sol[sp.Derivative(o, (t, 2))])

out = [
    ("quad2d_xdotdot.tex", xdotdot),
    ("quad2d_ydotdot.tex", ydotdot),
    ("quad2d_odotdot.tex", odotdot),
]

for filename, varr in out:
    with open(filename, "w") as f:
        f.write(nice_latex(latex(varr)))


# %%
print("now lets add a pole!")

m_p = sp.Symbol("m_p")
r = sp.Symbol("r")

U_pole = m_p * g * (y - r * sp.cos(o + q))
T_pole = .5 * m_p * ((v_x + r * (vq + w) * sp.cos(o + q))
                     ** 2 + (v_y + r * (vq + w) * sp.sin(o + q)))


L = T + T_pole - U - U_pole
F = [-(f_1 + f_2) * sp.sin(o), (f_1 + f_2) * sp.cos(o), l * (f_1 - f_2), 0]

eq = - _gradient(L, [x, y, o, q]) + \
    sp.diff(_gradient(L, [v_x, v_y, w, vq]), t) - sp.Matrix(F).T


sol = sp.solve(eq, [sp.Derivative(x, (t, 2)), sp.Derivative(y, (t, 2)),
                    sp.Derivative(o, (t, 2)), sp.Derivative(q, (t, 2))])

xdotdot = sp.simplify(sol[sp.Derivative(x, (t, 2))])
ydotdot = sp.simplify(sol[sp.Derivative(y, (t, 2))])
odotdot = sp.simplify(sol[sp.Derivative(o, (t, 2))])
qdotdot = sp.simplify(sol[sp.Derivative(q, (t, 2))])

out = [
    ("quad2d_pole_xdotdot.tex", xdotdot),
    ("quad2d_pole_ydotdot.tex", ydotdot),
    ("quad2d_pole_odotdot.tex", odotdot),
    ("quad2d_pole_qdotdot.tex", qdotdot)]


for file, var in out:
    with open(file, "w") as f:
        f.write(nice_latex(latex(var)))


# Continue here

from sympy.utilities.codegen import codegen


_x, _y, _o, _q, _vx, _vy, _w, _vq = sp.symbols("xx yy oo qq vvx vvy ww vvq")


Dsubs = {
    x: _x,
    y: _y,
    o: _o,
    q: _q,
    v_x: _vx,
    v_y: _vy,
    w: _w,
    vq: _vq}

all_vars = list(Dsubs.values())


_xdotdot = xdotdot.subs(Dsubs)
_ydotdot = ydotdot.subs(Dsubs)
_odotdot = odotdot.subs(Dsubs)
_qdotdot = qdotdot.subs(Dsubs)


fdotdot = sp.Matrix([_xdotdot, _ydotdot, _odotdot, _qdotdot])

Jx = fdotdot.jacobian(
    [_x, _y, _o, _q, _vx, _vy, _w, _vq])
Ju = fdotdot.jacobian([f_1, f_2])


# qdotdot_u = _gradient(X, all_vars)

# lets try!!


from sympy.printing.c import C99CodePrinter
# from sympy.printing.ccode import C99CodePrinter

# All printing classes have to be instantiated and then the .doprint()
# method can be used to print SymPy expressions. Let's try to print the
# right hand side of the differential equations.


printer = C99CodePrinter()
Jx_out = sp.MatrixSymbol('Jx', 4, 8)
Jx_ccode = printer.doprint(Jx, assign_to=Jx_out)
# IS this

Ju_out = sp.MatrixSymbol('Ju', 4, 2)
Ju_ccode = printer.doprint(Ju, assign_to=Ju_out)


fout = sp.MatrixSymbol('fdotdot', 4, 1)
f_ccode = printer.doprint(fdotdot, assign_to=fout)


with open("ju_out_quad2dpole.txt", "w") as f:
    f.write(Ju_ccode)

with open("jx_out_quad2dpole.txt", "w") as f:
    f.write(Jx_ccode)

with open("f_quad2dpole.txt", "w") as f:
    f.write(f_ccode)


now = datetime.now()  # current date and time
date_time = now.strftime("%Y-%m-%d--%H-%M-%S")

# %%


header = r"""#include <cmath>
void quadpole_2d( const double* x , const double* u , const double* data,  double* fdotdot , double * Jx , double * Ju) {""" + \
    "\n" + r"// Auto generated " + date_time + " from sympy\n" + r"using namespace std;"

assign_vars = r"""
const double xx = x[0];
const double yy = x[1];
const double oo = x[2];
const double qq = x[3];
const double vvx = x[4];
const double vvy = x[5];
const double ww = x[6];
const double vvq = x[7];
const double f1 = u[0];
const double f2 = u[1];
const double I = data[0];
const double m = data[1];
const double m_p = data[2];
const double l = data[3];
const double r = data[4];
const double g = data[5];""" + "\n"




footer = "\n}"
write_c_code  = False
if write_c_code :
    with open("src/quadpole_acceleration_auto.h", "w") as f:
        f.write(header)
        f.write(assign_vars)
        f.write(f_ccode)
        f.write("\n")
        f.writelines([r"if(Jx) {", "\n"])
        f.write(Jx_ccode)
        f.write("\n")
        f.writelines(["}"])
        f.write("\n")
        f.writelines([r"if(Ju) {", "\n"])
        f.write(Ju_ccode)
        f.write("\n")
        f.writelines(["}"])
        f.write(footer)


# %%

# def _gradient(f, v): return sp.Matrix([f]).jacobian(v)


# lets take some derivatives...
