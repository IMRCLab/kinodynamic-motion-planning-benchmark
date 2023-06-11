#include <cmath>
void quadpole_2d(const double *x, const double *u, const double *data,
                 double *fdotdot, double *Jx, double *Ju) {
  // Auto generated 2023-06-08--16-58-03 from sympy
  using namespace std;
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
  const double g = data[5];
  fdotdot[0] =
      (-1.0 * f1 * m * sin(oo) +
       0.25 * f1 * m_p * (sin(oo + 2 * qq) + sin(3 * oo + 2 * qq)) +
       1.0 * f1 * m_p * sin(oo) * pow(sin(oo + qq), 2) -
       1.0 * f1 * m_p * sin(oo) - 1.0 * f2 * m * sin(oo) +
       0.25 * f2 * m_p * (sin(oo + 2 * qq) + sin(3 * oo + 2 * qq)) +
       1.0 * f2 * m_p * sin(oo) * pow(sin(oo + qq), 2) -
       1.0 * f2 * m_p * sin(oo) +
       1.0 * m * m_p * r * pow(vvq, 2) * sin(oo + qq) +
       2.0 * m * m_p * r * vvq * ww * sin(oo + qq) +
       1.0 * m * m_p * r * pow(ww, 2) * sin(oo + qq) -
       1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(sin(oo + qq), 3) -
       1.0 * pow(m_p, 2) * r * pow(vvq, 2) * sin(oo + qq) *
           pow(cos(oo + qq), 2) +
       1.0 * pow(m_p, 2) * r * pow(vvq, 2) * sin(oo + qq) -
       2.0 * pow(m_p, 2) * r * vvq * ww * pow(sin(oo + qq), 3) -
       2.0 * pow(m_p, 2) * r * vvq * ww * sin(oo + qq) * pow(cos(oo + qq), 2) +
       2.0 * pow(m_p, 2) * r * vvq * ww * sin(oo + qq) -
       1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(sin(oo + qq), 3) -
       1.0 * pow(m_p, 2) * r * pow(ww, 2) * sin(oo + qq) *
           pow(cos(oo + qq), 2) +
       1.0 * pow(m_p, 2) * r * pow(ww, 2) * sin(oo + qq)) /
      (m * (m + m_p));
  fdotdot[1] =
      (1.0 * f1 * m * cos(oo) -
       0.25 * f1 * m_p * (cos(oo + 2 * qq) - cos(3 * oo + 2 * qq)) -
       1.0 * f1 * m_p * cos(oo) * pow(cos(oo + qq), 2) +
       1.0 * f1 * m_p * cos(oo) + 1.0 * f2 * m * cos(oo) -
       0.25 * f2 * m_p * (cos(oo + 2 * qq) - cos(3 * oo + 2 * qq)) -
       1.0 * f2 * m_p * cos(oo) * pow(cos(oo + qq), 2) +
       1.0 * f2 * m_p * cos(oo) - 1.0 * g * pow(m, 2) +
       1.0 * g * m * m_p * pow(sin(oo + qq), 2) +
       1.0 * g * m * m_p * pow(cos(oo + qq), 2) - 2.0 * g * m * m_p +
       1.0 * g * pow(m_p, 2) * pow(sin(oo + qq), 2) +
       1.0 * g * pow(m_p, 2) * pow(cos(oo + qq), 2) - 1.0 * g * pow(m_p, 2) -
       1.0 * m * m_p * r * pow(vvq, 2) * cos(oo + qq) -
       2.0 * m * m_p * r * vvq * ww * cos(oo + qq) -
       1.0 * m * m_p * r * pow(ww, 2) * cos(oo + qq) +
       1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(sin(oo + qq), 2) *
           cos(oo + qq) +
       1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(cos(oo + qq), 3) -
       1.0 * pow(m_p, 2) * r * pow(vvq, 2) * cos(oo + qq) +
       2.0 * pow(m_p, 2) * r * vvq * ww * pow(sin(oo + qq), 2) * cos(oo + qq) +
       2.0 * pow(m_p, 2) * r * vvq * ww * pow(cos(oo + qq), 3) -
       2.0 * pow(m_p, 2) * r * vvq * ww * cos(oo + qq) +
       1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(sin(oo + qq), 2) *
           cos(oo + qq) +
       1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(cos(oo + qq), 3) -
       1.0 * pow(m_p, 2) * r * pow(ww, 2) * cos(oo + qq)) /
      (m * (m + m_p));
  fdotdot[2] = l * (f1 - f2) / I;
  fdotdot[3] = -f1 * sin(qq) / (m * r) - f2 * sin(qq) / (m * r) - f1 * l / I +
               f2 * l / I;
  if (Jx) {
    Jx[0] = 0;
    Jx[1] = 0;
    Jx[2] = (-1.0 * f1 * m * cos(oo) +
             0.25 * f1 * m_p * (cos(oo + 2 * qq) + 3 * cos(3 * oo + 2 * qq)) +
             2.0 * f1 * m_p * sin(oo) * sin(oo + qq) * cos(oo + qq) +
             1.0 * f1 * m_p * pow(sin(oo + qq), 2) * cos(oo) -
             1.0 * f1 * m_p * cos(oo) - 1.0 * f2 * m * cos(oo) +
             0.25 * f2 * m_p * (cos(oo + 2 * qq) + 3 * cos(3 * oo + 2 * qq)) +
             2.0 * f2 * m_p * sin(oo) * sin(oo + qq) * cos(oo + qq) +
             1.0 * f2 * m_p * pow(sin(oo + qq), 2) * cos(oo) -
             1.0 * f2 * m_p * cos(oo) +
             1.0 * m * m_p * r * pow(vvq, 2) * cos(oo + qq) +
             2.0 * m * m_p * r * vvq * ww * cos(oo + qq) +
             1.0 * m * m_p * r * pow(ww, 2) * cos(oo + qq) -
             1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(sin(oo + qq), 2) *
                 cos(oo + qq) -
             1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(cos(oo + qq), 3) +
             1.0 * pow(m_p, 2) * r * pow(vvq, 2) * cos(oo + qq) -
             2.0 * pow(m_p, 2) * r * vvq * ww * pow(sin(oo + qq), 2) *
                 cos(oo + qq) -
             2.0 * pow(m_p, 2) * r * vvq * ww * pow(cos(oo + qq), 3) +
             2.0 * pow(m_p, 2) * r * vvq * ww * cos(oo + qq) -
             1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(sin(oo + qq), 2) *
                 cos(oo + qq) -
             1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(cos(oo + qq), 3) +
             1.0 * pow(m_p, 2) * r * pow(ww, 2) * cos(oo + qq)) /
            (m * (m + m_p));
    Jx[3] =
        (0.25 * f1 * m_p * (2 * cos(oo + 2 * qq) + 2 * cos(3 * oo + 2 * qq)) +
         2.0 * f1 * m_p * sin(oo) * sin(oo + qq) * cos(oo + qq) +
         0.25 * f2 * m_p * (2 * cos(oo + 2 * qq) + 2 * cos(3 * oo + 2 * qq)) +
         2.0 * f2 * m_p * sin(oo) * sin(oo + qq) * cos(oo + qq) +
         1.0 * m * m_p * r * pow(vvq, 2) * cos(oo + qq) +
         2.0 * m * m_p * r * vvq * ww * cos(oo + qq) +
         1.0 * m * m_p * r * pow(ww, 2) * cos(oo + qq) -
         1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(sin(oo + qq), 2) *
             cos(oo + qq) -
         1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(cos(oo + qq), 3) +
         1.0 * pow(m_p, 2) * r * pow(vvq, 2) * cos(oo + qq) -
         2.0 * pow(m_p, 2) * r * vvq * ww * pow(sin(oo + qq), 2) *
             cos(oo + qq) -
         2.0 * pow(m_p, 2) * r * vvq * ww * pow(cos(oo + qq), 3) +
         2.0 * pow(m_p, 2) * r * vvq * ww * cos(oo + qq) -
         1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(sin(oo + qq), 2) *
             cos(oo + qq) -
         1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(cos(oo + qq), 3) +
         1.0 * pow(m_p, 2) * r * pow(ww, 2) * cos(oo + qq)) /
        (m * (m + m_p));
    Jx[4] = 0;
    Jx[5] = 0;
    Jx[6] = (2.0 * m * m_p * r * vvq * sin(oo + qq) +
             2.0 * m * m_p * r * ww * sin(oo + qq) -
             2.0 * pow(m_p, 2) * r * vvq * pow(sin(oo + qq), 3) -
             2.0 * pow(m_p, 2) * r * vvq * sin(oo + qq) * pow(cos(oo + qq), 2) +
             2.0 * pow(m_p, 2) * r * vvq * sin(oo + qq) -
             2.0 * pow(m_p, 2) * r * ww * pow(sin(oo + qq), 3) -
             2.0 * pow(m_p, 2) * r * ww * sin(oo + qq) * pow(cos(oo + qq), 2) +
             2.0 * pow(m_p, 2) * r * ww * sin(oo + qq)) /
            (m * (m + m_p));
    Jx[7] = (2.0 * m * m_p * r * vvq * sin(oo + qq) +
             2.0 * m * m_p * r * ww * sin(oo + qq) -
             2.0 * pow(m_p, 2) * r * vvq * pow(sin(oo + qq), 3) -
             2.0 * pow(m_p, 2) * r * vvq * sin(oo + qq) * pow(cos(oo + qq), 2) +
             2.0 * pow(m_p, 2) * r * vvq * sin(oo + qq) -
             2.0 * pow(m_p, 2) * r * ww * pow(sin(oo + qq), 3) -
             2.0 * pow(m_p, 2) * r * ww * sin(oo + qq) * pow(cos(oo + qq), 2) +
             2.0 * pow(m_p, 2) * r * ww * sin(oo + qq)) /
            (m * (m + m_p));
    Jx[8] = 0;
    Jx[9] = 0;
    Jx[10] = (-1.0 * f1 * m * sin(oo) -
              0.25 * f1 * m_p * (-sin(oo + 2 * qq) + 3 * sin(3 * oo + 2 * qq)) +
              1.0 * f1 * m_p * sin(oo) * pow(cos(oo + qq), 2) -
              1.0 * f1 * m_p * sin(oo) +
              2.0 * f1 * m_p * sin(oo + qq) * cos(oo) * cos(oo + qq) -
              1.0 * f2 * m * sin(oo) -
              0.25 * f2 * m_p * (-sin(oo + 2 * qq) + 3 * sin(3 * oo + 2 * qq)) +
              1.0 * f2 * m_p * sin(oo) * pow(cos(oo + qq), 2) -
              1.0 * f2 * m_p * sin(oo) +
              2.0 * f2 * m_p * sin(oo + qq) * cos(oo) * cos(oo + qq) +
              1.0 * m * m_p * r * pow(vvq, 2) * sin(oo + qq) +
              2.0 * m * m_p * r * vvq * ww * sin(oo + qq) +
              1.0 * m * m_p * r * pow(ww, 2) * sin(oo + qq) -
              1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(sin(oo + qq), 3) -
              1.0 * pow(m_p, 2) * r * pow(vvq, 2) * sin(oo + qq) *
                  pow(cos(oo + qq), 2) +
              1.0 * pow(m_p, 2) * r * pow(vvq, 2) * sin(oo + qq) -
              2.0 * pow(m_p, 2) * r * vvq * ww * pow(sin(oo + qq), 3) -
              2.0 * pow(m_p, 2) * r * vvq * ww * sin(oo + qq) *
                  pow(cos(oo + qq), 2) +
              2.0 * pow(m_p, 2) * r * vvq * ww * sin(oo + qq) -
              1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(sin(oo + qq), 3) -
              1.0 * pow(m_p, 2) * r * pow(ww, 2) * sin(oo + qq) *
                  pow(cos(oo + qq), 2) +
              1.0 * pow(m_p, 2) * r * pow(ww, 2) * sin(oo + qq)) /
             (m * (m + m_p));
    Jx[11] =
        (-0.25 * f1 * m_p * (-2 * sin(oo + 2 * qq) + 2 * sin(3 * oo + 2 * qq)) +
         2.0 * f1 * m_p * sin(oo + qq) * cos(oo) * cos(oo + qq) -
         0.25 * f2 * m_p * (-2 * sin(oo + 2 * qq) + 2 * sin(3 * oo + 2 * qq)) +
         2.0 * f2 * m_p * sin(oo + qq) * cos(oo) * cos(oo + qq) +
         1.0 * m * m_p * r * pow(vvq, 2) * sin(oo + qq) +
         2.0 * m * m_p * r * vvq * ww * sin(oo + qq) +
         1.0 * m * m_p * r * pow(ww, 2) * sin(oo + qq) -
         1.0 * pow(m_p, 2) * r * pow(vvq, 2) * pow(sin(oo + qq), 3) -
         1.0 * pow(m_p, 2) * r * pow(vvq, 2) * sin(oo + qq) *
             pow(cos(oo + qq), 2) +
         1.0 * pow(m_p, 2) * r * pow(vvq, 2) * sin(oo + qq) -
         2.0 * pow(m_p, 2) * r * vvq * ww * pow(sin(oo + qq), 3) -
         2.0 * pow(m_p, 2) * r * vvq * ww * sin(oo + qq) *
             pow(cos(oo + qq), 2) +
         2.0 * pow(m_p, 2) * r * vvq * ww * sin(oo + qq) -
         1.0 * pow(m_p, 2) * r * pow(ww, 2) * pow(sin(oo + qq), 3) -
         1.0 * pow(m_p, 2) * r * pow(ww, 2) * sin(oo + qq) *
             pow(cos(oo + qq), 2) +
         1.0 * pow(m_p, 2) * r * pow(ww, 2) * sin(oo + qq)) /
        (m * (m + m_p));
    Jx[12] = 0;
    Jx[13] = 0;
    Jx[14] =
        (-2.0 * m * m_p * r * vvq * cos(oo + qq) -
         2.0 * m * m_p * r * ww * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * vvq * pow(sin(oo + qq), 2) * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * vvq * pow(cos(oo + qq), 3) -
         2.0 * pow(m_p, 2) * r * vvq * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * ww * pow(sin(oo + qq), 2) * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * ww * pow(cos(oo + qq), 3) -
         2.0 * pow(m_p, 2) * r * ww * cos(oo + qq)) /
        (m * (m + m_p));
    Jx[15] =
        (-2.0 * m * m_p * r * vvq * cos(oo + qq) -
         2.0 * m * m_p * r * ww * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * vvq * pow(sin(oo + qq), 2) * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * vvq * pow(cos(oo + qq), 3) -
         2.0 * pow(m_p, 2) * r * vvq * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * ww * pow(sin(oo + qq), 2) * cos(oo + qq) +
         2.0 * pow(m_p, 2) * r * ww * pow(cos(oo + qq), 3) -
         2.0 * pow(m_p, 2) * r * ww * cos(oo + qq)) /
        (m * (m + m_p));
    Jx[16] = 0;
    Jx[17] = 0;
    Jx[18] = 0;
    Jx[19] = 0;
    Jx[20] = 0;
    Jx[21] = 0;
    Jx[22] = 0;
    Jx[23] = 0;
    Jx[24] = 0;
    Jx[25] = 0;
    Jx[26] = 0;
    Jx[27] = -f1 * cos(qq) / (m * r) - f2 * cos(qq) / (m * r);
    Jx[28] = 0;
    Jx[29] = 0;
    Jx[30] = 0;
    Jx[31] = 0;
  }
  if (Ju) {
    Ju[0] = (-1.0 * m * sin(oo) +
             0.25 * m_p * (sin(oo + 2 * qq) + sin(3 * oo + 2 * qq)) +
             1.0 * m_p * sin(oo) * pow(sin(oo + qq), 2) - 1.0 * m_p * sin(oo)) /
            (m * (m + m_p));
    Ju[1] = (-1.0 * m * sin(oo) +
             0.25 * m_p * (sin(oo + 2 * qq) + sin(3 * oo + 2 * qq)) +
             1.0 * m_p * sin(oo) * pow(sin(oo + qq), 2) - 1.0 * m_p * sin(oo)) /
            (m * (m + m_p));
    Ju[2] = (1.0 * m * cos(oo) -
             0.25 * m_p * (cos(oo + 2 * qq) - cos(3 * oo + 2 * qq)) -
             1.0 * m_p * cos(oo) * pow(cos(oo + qq), 2) + 1.0 * m_p * cos(oo)) /
            (m * (m + m_p));
    Ju[3] = (1.0 * m * cos(oo) -
             0.25 * m_p * (cos(oo + 2 * qq) - cos(3 * oo + 2 * qq)) -
             1.0 * m_p * cos(oo) * pow(cos(oo + qq), 2) + 1.0 * m_p * cos(oo)) /
            (m * (m + m_p));
    Ju[4] = l / I;
    Ju[5] = -l / I;
    Ju[6] = -sin(qq) / (m * r) - l / I;
    Ju[7] = -sin(qq) / (m * r) + l / I;
  }
}