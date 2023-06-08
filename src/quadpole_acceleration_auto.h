#include <cmath>
void quadpole_2d( const double* x , const double* u , const double* data,  double* fdotdot , double * Jx , double * Ju) {
// Auto generated 2023-06-08--10-09-31 from sympy
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
fdotdot[0] = (-f1*sin(oo) - f2*sin(oo) + g*m_p*tan(oo + qq))/m;
fdotdot[1] = (f1*cos(oo) + f2*cos(oo) - g*m - g*m_p)/m;
fdotdot[2] = l*(f1 - f2)/I;
fdotdot[3] = (-I*g*m*sin(oo + qq) - I*g*m_p*sin(oo + qq) + (1.0/2.0)*I*m*r*(pow(vvq, 2) + 2.0*vvq*ww + pow(ww, 2))*sin(2*oo + 2*qq) + I*(f1 + f2)*sin(oo)*cos(oo + qq) + l*m*r*(-f1 + f2)*pow(cos(oo + qq), 2))/(I*m*r*pow(cos(oo + qq), 2));
if(Jx) {
Jx[0] = 0;
Jx[1] = 0;
Jx[2] = (-f1*cos(oo) - f2*cos(oo) + g*m_p*(pow(tan(oo + qq), 2) + 1))/m;
Jx[3] = g*m_p*(pow(tan(oo + qq), 2) + 1)/m;
Jx[4] = 0;
Jx[5] = 0;
Jx[6] = 0;
Jx[7] = 0;
Jx[8] = 0;
Jx[9] = 0;
Jx[10] = (-f1*sin(oo) - f2*sin(oo))/m;
Jx[11] = 0;
Jx[12] = 0;
Jx[13] = 0;
Jx[14] = 0;
Jx[15] = 0;
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
Jx[26] = 2*(-I*g*m*sin(oo + qq) - I*g*m_p*sin(oo + qq) + (1.0/2.0)*I*m*r*(pow(vvq, 2) + 2.0*vvq*ww + pow(ww, 2))*sin(2*oo + 2*qq) + I*(f1 + f2)*sin(oo)*cos(oo + qq) + l*m*r*(-f1 + f2)*pow(cos(oo + qq), 2))*sin(oo + qq)/(I*m*r*pow(cos(oo + qq), 3)) + (-I*g*m*cos(oo + qq) - I*g*m_p*cos(oo + qq) + I*m*r*(pow(vvq, 2) + 2.0*vvq*ww + pow(ww, 2))*cos(2*oo + 2*qq) - I*(f1 + f2)*sin(oo)*sin(oo + qq) + I*(f1 + f2)*cos(oo)*cos(oo + qq) - 2*l*m*r*(-f1 + f2)*sin(oo + qq)*cos(oo + qq))/(I*m*r*pow(cos(oo + qq), 2));
Jx[27] = 2*(-I*g*m*sin(oo + qq) - I*g*m_p*sin(oo + qq) + (1.0/2.0)*I*m*r*(pow(vvq, 2) + 2.0*vvq*ww + pow(ww, 2))*sin(2*oo + 2*qq) + I*(f1 + f2)*sin(oo)*cos(oo + qq) + l*m*r*(-f1 + f2)*pow(cos(oo + qq), 2))*sin(oo + qq)/(I*m*r*pow(cos(oo + qq), 3)) + (-I*g*m*cos(oo + qq) - I*g*m_p*cos(oo + qq) + I*m*r*(pow(vvq, 2) + 2.0*vvq*ww + pow(ww, 2))*cos(2*oo + 2*qq) - I*(f1 + f2)*sin(oo)*sin(oo + qq) - 2*l*m*r*(-f1 + f2)*sin(oo + qq)*cos(oo + qq))/(I*m*r*pow(cos(oo + qq), 2));
Jx[28] = 0;
Jx[29] = 0;
Jx[30] = (1.0/2.0)*(2.0*vvq + 2*ww)*sin(2*oo + 2*qq)/pow(cos(oo + qq), 2);
Jx[31] = (1.0/2.0)*(2*vvq + 2.0*ww)*sin(2*oo + 2*qq)/pow(cos(oo + qq), 2);
}
if(Ju) {
Ju[0] = -sin(oo)/m;
Ju[1] = -sin(oo)/m;
Ju[2] = cos(oo)/m;
Ju[3] = cos(oo)/m;
Ju[4] = l/I;
Ju[5] = -l/I;
Ju[6] = (I*sin(oo)*cos(oo + qq) - l*m*r*pow(cos(oo + qq), 2))/(I*m*r*pow(cos(oo + qq), 2));
Ju[7] = (I*sin(oo)*cos(oo + qq) + l*m*r*pow(cos(oo + qq), 2))/(I*m*r*pow(cos(oo + qq), 2));
}
}
