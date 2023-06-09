

# write down acrobot expressions

import sympy as sp
import numpy as np

from sympy import init_printing

init_printing()

u = sp.Symbol("u")
q1 = sp.Symbol("q1")
q2 = sp.Symbol("q2")
q1dot = sp.Symbol("q1dot")
q2dot = sp.Symbol("q2dot")

qdot = sp.Matrix([[q1dot], [q2dot]])

m1 = sp.Symbol("m1")
m2 = sp.Symbol("m2")
g = sp.Symbol("g")

I1 = sp.Symbol("I1")
I2 = sp.Symbol("I2")

l1 = sp.Symbol("l1")
l2 = sp.Symbol("l2")

lc1 = sp.Symbol("lc1")
lc2 = sp.Symbol("lc2")

c1 = sp.cos(q1)
s1 = sp.sin(q1)

c2 = sp.cos(q2)
s2 = sp.sin(q2)

s12 = sp.sin(q1 + q2)
c12 = sp.cos(q1 + q2)

# REF http://underactuated.mit.edu/acrobot.html
M = sp.Matrix([[I1 + I2 + m2 * l1 * l1 + 2 * m2 * l1 * lc2 * c2,
                I2 + m2 * l1 * lc2 * c2], [I2 + m2 * l1 * lc2 * c2, I2]])

C = sp.Matrix([[-2 * m2 * l1 * lc2 * s2 * q2dot, -m2 * l1 * lc2 * s2 * q2dot],
               [m2 * l1 * lc2 * s2 * q1dot, 0]])

tau = sp.Matrix([[-m1 * g * lc1 * s1 - m2 * g * (l1 * s1 + lc2 * s12)],
                 [-m2 * g * lc2 * s12]])


B = sp.Matrix([[0], [1]])

qdotdot = M.inv() * (tau - C * qdot + B * u)
print(qdotdot)

# print("printing" )
# print("M" )
# sp.pprint(M)
# print("B" )
# sp.pprint(B)
# print("tau" )
# sp.pprint(tau)
# print("C" )
# sp.pprint(C)
# print("")



# sp.latex(qdotdot)
# qdotdot.
# sp.pprint(M)
# sp.pprint(M)

write_c_code = False
if write_c_code:

  qdotdot = sp.simplify(M.inv() * (tau - C * qdot + B * u))

  qdotdot_q1 = sp.simplify(qdotdot.jacobian([[q1]]))
  qdotdot_q2 = sp.simplify(qdotdot.jacobian([[q2]]))

  qdotdot_q1dot = sp.simplify(qdotdot.jacobian([[q1dot]]))
  qdotdot_q2dot = sp.simplify(qdotdot.jacobian([[q2dot]]))


  qdotdot_u = qdotdot.jacobian([[u]])


  ccode = sp.printing.ccode

  q1dotdot = sp.printing.ccode(sp.simplify(qdotdot[0]), assign_to="q1dotdot")
  q2dotdot = sp.printing.ccode(sp.simplify(qdotdot[1]), assign_to="q2dotdot")

  print("****CALC****\n")
  print(q1dotdot)
  print(q2dotdot)
  print("****end of CALC****\n")


  q1dotdot_u = sp.printing.ccode(
      sp.simplify(
          qdotdot_u[0]),
      assign_to="q1dotdot_u")
  q2dotdot_u = sp.printing.ccode(
      sp.simplify(
          qdotdot_u[1]),
      assign_to="q2dotdot_u")

  q1dotdot_q1 = ccode(sp.simplify(qdotdot_q1[0]), assign_to="q1dotdot_q1")
  q2dotdot_q1 = ccode(sp.simplify(qdotdot_q1[1]), assign_to="q2dotdot_q1")

  q1dotdot_q2 = ccode(sp.simplify(qdotdot_q2[0]), assign_to="q1dotdot_q2")
  q2dotdot_q2 = ccode(sp.simplify(qdotdot_q2[1]), assign_to="q2dotdot_q2")


  q1dotdot_q2dot = ccode(
      sp.simplify(
          qdotdot_q2dot[0]),
      assign_to="q1dotdot_q2dot")
  q2dotdot_q2dot = ccode(
      sp.simplify(
          qdotdot_q2dot[1]),
      assign_to="q2dotdot_q2dot")


  q1dotdot_q1dot = ccode(
      sp.simplify(
          qdotdot_q1dot[0]),
      assign_to="q1dotdot_q1dot")
  q2dotdot_q1dot = ccode(
      sp.simplify(
          qdotdot_q1dot[1]),
      assign_to="q2dotdot_q1dot")


  print("****CALC DIFF****\n")
  print(q1dotdot_u)
  print(q2dotdot_u)

  print(q1dotdot_q1)
  print(q2dotdot_q1)

  print(q1dotdot_q2)
  print(q2dotdot_q2)


  print(q1dotdot_q2dot)
  print(q2dotdot_q2dot)


  print(q1dotdot_q1dot)
  print(q2dotdot_q1dot)
  print("****END OF  CALC DIFF****\n")
