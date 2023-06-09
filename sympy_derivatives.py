

# write down acrobot expressions

import sympy as sp
import numpy as np

from sympy import init_printing


q = sp.Symbol("q")
c = sp.cos(q)
s = sp.sin(q)
R = sp.Matrix([[c, -s], [s, c]])
print(R)
p1 = sp.Symbol("p1")
p2 = sp.Symbol("p2")
p = sp.Matrix([[p1], [p2]])
v = R * p
jac = sp.simplify(v.jacobian([[q]]))
print(v)  # Matrix([[p1*cos(q) - p2*sin(q)], [p1*sin(q) + p2*cos(q)]])
print(jac)  # Matrix([[-p1*sin(q) - p2*cos(q)], [p1*cos(q) - p2*sin(q)]])
