#ifndef NEUMANN_UTIL_H
#define NEUMANN_UTIL_H

#include <cmath>
#include <iostream>

int mod(int x, int m) {
    if (x >= 0 && x < m)
        return x;
    else if (x < 0)
        return m - 1 - mod(-1 - x, m);
    else
        return x % m;
}

double fermi_density(double x, double kT, double mu) {
    double alpha = (x - mu) / std::abs(kT);
    if (kT < 1e-15 || std::abs(alpha) > 20) {
        return (x < mu) ? 1.0 : 0.0;
    }
    else {
        return 1.0 / (exp(alpha) + 1.0);
    }
}


#endif //NEUMANN_UTIL_H
