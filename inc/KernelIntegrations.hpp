#pragma once
#include <iostream>
#include <vector>
#include "cheby2.hpp"

namespace KernelIntegration {
  
  typedef cheby::ArrayXd ArrayXd;
  
  ArrayXd coef_to_vals(ArrayXd const &c) {
    ArrayXd a = c;
    a[0] = 2.0*c[0];
    ArrayXd f = cheby::dct3(a);
    return f/2.0;
  }
  
  ArrayXd vals_to_coefs(ArrayXd const &f) {
    ArrayXd c = cheby::dct2(f) / f.size();
    c[0] /= 2.0;
    return c;
  }
  
  floatT integrate_kernel( ArrayXd const &f, ArrayXd const &K ) {
    floatT maxval = K.max();
    ArrayXd c = vals_to_coefs(f * (K / maxval));
    ArrayXd c_new = cheby::Chebyshev_coef_integrate(c, -1.0, true);
    return cheby::Chebyshev_value(1.0,c_new)*maxval;
  }
  
  ArrayXd T(ArrayXd x, floatT n) {
    return cos(n * acos(x));
  }
  
  std::vector<ArrayXd> compute_integrals(int L, std::vector<cheby::ArrayXd> const &Kernel1) {
  
    ArrayXd x = cheby::ChebPts(L);
  
    std::vector<ArrayXd> I_T(L);
    std::vector<ArrayXd> T_j(L);
    for (int i=0; i<L; ++i) {
      I_T[i].resize(L,0.0);
      T_j[i].resize(L,0.0);
    }
    for (int j=0; j<L; ++j) T_j[j] = T(x,j);
  
    for (int i=0; i<L; ++i) {
      ArrayXd K_i;
      K_i.resize(L, 0.0);
      for (int a=0; a<L; ++a) K_i[a] = Kernel1[a][i];
      for (int j=0; j<L; ++j) {
        I_T[j][i] = integrate_kernel(T_j[j], K_i);
      }
    }
  
    return I_T;
  }

};

