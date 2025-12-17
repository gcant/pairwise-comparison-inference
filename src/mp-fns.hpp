#pragma once
#include "cheby2.hpp"
#include "Graph.h"
#include <unordered_map>
#include <vector>

typedef std::vector<std::pair<int,int>> elist_t;
typedef std::unordered_map<int, cheby::Chebyshev> marginal_t;
typedef std::unordered_map<int, marginal_t> message_t;
typedef std::vector<cheby::ArrayXd> Matrix;

//const bool KEEP_ORDER = true;
const bool ENFORCE_POSITIVE = true;
//const floatT kernel_epsilon = 10*std::numeric_limits<floatT>::min();
const floatT marginal_epsilon = 1e-8;
floatT FT_MIN = std::numeric_limits<floatT>::min();

std::vector<cheby::ArrayXd> transpose(std::vector<cheby::ArrayXd> const &X) {
  std::vector<cheby::ArrayXd> XT;
  int L = X.size();
  XT.resize(L);
  for (int i=0; i<L; ++i) {
    XT[i].resize(L, 0.0);
    for (int j=0; j<L; ++j) {
      XT[i][j] = X[j][i];
    }
  }
  return XT;
}

class KernelInt {
  private:
    Matrix K;
    int L;
    std::unordered_map<int, std::unordered_map<int, Matrix>> IK;
    std::unordered_map<int, std::unordered_map<int, Matrix>> WK;
  public:
    void setKernel(int, Matrix);
    const Matrix& getI(int, int);
    const Matrix& getK(int, int);
};

void KernelInt::setKernel(int N, Matrix K1) {
  L = N;
  K.resize(L);
  for (int i=0; i<L; ++i) K[i].resize(L, 0.0);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      K[i][j] = 0.5 * (1 + (K1[i][j] - K1[j][i]));
    }
  }
  std::unordered_map<int, std::unordered_map<int, Matrix>> IK1;
  std::unordered_map<int, std::unordered_map<int, Matrix>> WK1;
  IK = IK1;
  WK = WK1;
}

const Matrix& KernelInt::getI(int W1, int W2) {
  if (not(IK[W1].count(W2))) {
    Matrix tmp;
    tmp.resize(L);
    for (int i=0; i<L; ++i) {
      tmp[i].resize(L, 0.0);
      for (int j=0; j<L; ++j) {
        //tmp[i][j] = kernel_epsilon + (1-2*kernel_epsilon) * pow(K[i][j], W1) * pow(K[j][i], W2);
        tmp[i][j] = std::pow(K[i][j], W1) * std::pow(1-K[i][j], W2);
      }
    }
    WK[W1][W2] = tmp;
    IK[W1][W2] = transpose(KernelIntegration::compute_integrals(L, WK[W1][W2]));
  }
  return IK.at(W1).at(W2);
}

const Matrix& KernelInt::getK(int W1, int W2) {
  if (not(IK[W1].count(W2))) {
    Matrix tmp;
    tmp.resize(L);
    for (int i=0; i<L; ++i) {
      tmp[i].resize(L, 0.0);
      for (int j=0; j<L; ++j) {
        //tmp[i][j] = kernel_epsilon + (1-2*kernel_epsilon) * pow(K[i][j], W1) * pow(K[j][i], W2);
        tmp[i][j] = std::pow(K[i][j], W1) * std::pow(1-K[i][j], W2);
      }
    }
    WK[W1][W2] = tmp;
    IK[W1][W2] = transpose(KernelIntegration::compute_integrals(L, WK[W1][W2]));
  }
  return WK.at(W1).at(W2);
}

elist_t Graph_to_list(Graph &G) {
  elist_t edges;
  for (auto i : G.nodes()) {
    for (auto j : G.neighbors(i)) {
      edges.push_back({i,j});
    }
  }
  return edges;
}

template <class vec_t>
vec_t log0(vec_t X) {
  vec_t ans = X;
  for (int i=0; i<ans.size(); ++i) {
    if (ans[i]>0) ans[i] = log(ans[i]);
    else ans[i] = FT_MIN;
  }
  return ans;
}


cheby::ArrayXd dot(std::vector<cheby::ArrayXd> const &X, cheby::ArrayXd const &a) {
  int L = a.size();
  cheby::ArrayXd ans(0.0,L);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      ans[i] += X[i][j]*a[j];
    }
  }
  return ans;
}

floatT oneD_integral( cheby::ArrayXd const &f ) {
  cheby::Chebyshev f_cheb(f, INIT_BY_VALUES);
  return cheby::Chebyshev_value(1.0, cheby::Chebyshev_coef_integrate(f_cheb.coefs()));
}

floatT twoD_integral( std::vector<cheby::ArrayXd> const &f ) {
  int L = f.size();
  cheby::ArrayXd g(0.0,L);
  for (int k=0; k<L; ++k) {
    g[k] = oneD_integral(f[k]);
  }
  return oneD_integral(g);
}

std::vector<cheby::ArrayXd> two_point_dist( cheby::ArrayXd const &f1, 
		cheby::ArrayXd const &f2, std::vector<cheby::ArrayXd> const &K) {
  int L = f1.size();
  std::vector<cheby::ArrayXd> ans(L);
  for (int i=0; i<L; ++i) ans[i].resize(L,0.0);
  for (int i=0; i<L; ++i) {
    for (int j=0; j<L; ++j) {
      ans[i][j] = K[i][j]*f1[i]*f2[j];
    }
  }
  return ans;
}


floatT iterate_messages_delta_parallel_kernel_edge(Graph const &G, elist_t const &m_order,
		message_t &messages, int const L,
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K,
		cheby::ArrayXd const &prior
		) {
  floatT delta = 0.0;
  #pragma omp parallel for shared(messages, K) reduction(+: delta)
  for (int q=0; q<m_order.size(); ++q) {
    int i=m_order[q].first;
    int j=m_order[q].second;
    cheby::ArrayXd new_values(0.0, L);
    new_values = log0(prior);
    for (auto k : G.neighbors(j)) if (k!=i) {
      new_values += log0(dot(K.getI(W.at(k).at(j), W.at(j).at(k)), messages.at(k).at(j).coefs()));
    }
    new_values = exp(new_values - new_values.max()) + marginal_epsilon;
    cheby::Chebyshev new_message(new_values,INIT_BY_VALUES);
    //if (ENFORCE_POSITIVE) new_message.positives(1000*FT_MIN);
    new_message.normalize();
    delta += abs(messages[j][i].vals() - new_message.vals()).sum();
    messages[j][i] = new_message;
  }
  return delta;
}


floatT iterate_messages_delta_parallel_kernel(Graph const &G,
		message_t &messages, int const L,
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K,
		cheby::ArrayXd const &prior
		) {
  floatT delta = 0.0;
  const int n = G.number_of_nodes();
  #pragma omp parallel for shared(messages, K) reduction(+: delta)
  for (int j=0; j<n; ++j) {
    cheby::ArrayXd new_values(0.0, L);
    new_values = log0(prior);
    for (int k : G.neighbors(j)) {
      new_values += log0(dot(K.getI(W.at(k).at(j), W.at(j).at(k)), messages.at(k).at(j).coefs()));
    }
    for (int i : G.neighbors(j)) {
      cheby::ArrayXd new_m_values(0.0, L);
      new_m_values = new_values - log0(dot(K.getI(W.at(i).at(j), W.at(j).at(i)), messages.at(i).at(j).coefs()));
      new_m_values = exp(new_m_values - new_m_values.max()) + marginal_epsilon;
      cheby::Chebyshev new_message(new_m_values, INIT_BY_VALUES);
      new_message.normalize();
      delta += abs(messages[j][i].vals() - new_message.vals()).sum();
      messages[j][i] = new_message;
    }
  }
  return delta;
}


marginal_t compute_marginals_kernel(Graph const &G, message_t const &messages, int const L, 
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K,
		cheby::ArrayXd const &prior
		) {
  const int n = G.number_of_nodes();
  marginal_t marg;
  //for (int i=0; i<n; ++i) marg[i] = cheby::Chebyshev(L);
  #pragma omp parallel for shared(marg, K)
  for (int i=0; i<n; ++i) {
    auto tmp = cheby::Chebyshev(L);
    cheby::ArrayXd marginal_values(0.0, L);
    marginal_values += log0(prior);
    for (int j : G.neighbors(i)) {
      marginal_values += log0(dot(K.getI(W.at(j).at(i), W.at(i).at(j)), messages.at(j).at(i).coefs()));
    }
    marginal_values = exp(marginal_values - marginal_values.max()) + marginal_epsilon;
    tmp.set_coefs(cheby::Chebyshev_coefs_from_values(marginal_values)[std::slice(0,L,1)]);
    tmp.normalize();
    #pragma omp critical
    {
      marg[i] = tmp;
    }
  }
  return marg;
}

floatT energy_kernel(Graph const &G, message_t const &messages, int L,
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K1
		 ) {

  const int n = G.number_of_nodes();
  floatT U = 0.0;
  #pragma omp parallel for shared(K1) reduction(+: U)
  for (int i=0; i<n; ++i) {
    for (int j : G.neighbors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();
      auto K = K1.getK(W.at(j).at(i), W.at(i).at(j));

      std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), K);
      auto QlnK = Q;
      for (int ii=0; ii<L; ++ii) QlnK[ii] = Q[ii] * log0(K[ii]);

      floatT Z = std::max(twoD_integral(Q), FT_MIN);
      U += (twoD_integral(QlnK))/Z;
    }
  }
  return U/2;
}


floatT marginal_entropy(Graph const &G, marginal_t const &marg, int const L){
  floatT S = 0.0;
  const int N = G.number_of_nodes();
  #pragma omp parallel for reduction(+: S)
  for (int i=0; i<N; ++i) {
    auto mu1 = marg.at(i);
    mu1.normalize();
    cheby::Chebyshev L(mu1.vals() * log0(mu1.vals()), INIT_BY_VALUES);
    floatT di = 0;
    di += G.degree(i);
    S+= (di-1)*cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(L.coefs()));
  }
  return S;
}


floatT message_entropy_kernel(Graph const &G, message_t const &messages, int const L,
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K1
	) {

  const int n = G.number_of_nodes();

  floatT S = 0.0;

  #pragma omp parallel for shared(K1) reduction(+: S)
  for (int i=0; i<n; ++i) {
    for (int j : G.neighbors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();
      auto K = K1.getK(W.at(j).at(i), W.at(i).at(j));

      std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), K);
      auto QlnQ = Q;
      for (int ii=0; ii<L; ++ii) QlnQ[ii] = Q[ii] * log0(Q[ii]);

      floatT Z = std::max(twoD_integral(Q), FT_MIN);
      S -= log(Z) - ((twoD_integral(QlnQ))/Z);

    }
  }
  return S/2;
}

floatT prior_energy(Graph const &G, marginal_t const &marg, 
		cheby::ArrayXd const &prior
		){
  floatT U = 0.0;
  const int N = G.number_of_nodes();
  #pragma omp parallel for reduction(+: U)
  for (int i=0; i<N; ++i) {
    auto mu1 = marg.at(i);
    mu1.normalize();
    cheby::Chebyshev L(mu1.vals() * log0(prior), INIT_BY_VALUES);
    U += cheby::Chebyshev_value(1.0,cheby::Chebyshev_coef_integrate(L.coefs()));
  }
  return U;
}

std::vector<cheby::ArrayXd> compute_mean_marginal(
		Graph const &G, message_t const &messages, int const L,
        std::unordered_map<int, std::unordered_map<int,int>> const &W,
	KernelInt &K1
	) {

  std::vector<cheby::ArrayXd> mean_marginal;
  mean_marginal.resize(L);
  for (int a=0; a<L; ++a) {
    mean_marginal[a].resize(L, 0.0);
      for (int b=0; b<L; ++b) mean_marginal[a][b] = 0.0;
  }
  const int N = G.number_of_nodes();

  #pragma omp parallel for shared(K1)
  for (int i=0; i<N; ++i) {
    for (int j : G.neighbors(i)) {
      auto mu1 = messages.at(i).at(j);
      auto mu2 = messages.at(j).at(i);
      mu1.normalize();
      mu2.normalize();
      auto K = K1.getK(W.at(j).at(i), W.at(i).at(j));

      std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), K);
      auto QlnQ = Q;
      for (int ii=0; ii<L; ++ii) QlnQ[ii] = Q[ii] * log0(Q[ii]);

      floatT Z = std::max(twoD_integral(Q), FT_MIN);
      #pragma omp critical
      {
        for (int a=0; a<L; ++a) mean_marginal[a] += (W.at(i).at(j) * Q[a])/Z;
      }
    }
  }
  return mean_marginal;
}


floatT predict(int i, int j, message_t &messages, int L,
		marginal_t &marg,
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K1
		 ) {
  floatT U = 0.0;
  cheby::Chebyshev mu1;
  cheby::Chebyshev mu2;
  Matrix K;
  if (messages[i].count(j)) {
    mu1 = messages.at(i).at(j);
    mu2 = messages.at(j).at(i);
    K = K1.getK(W.at(j).at(i), W.at(i).at(j));
  }
  else {
    mu1 = marg.at(i);
    mu2 = marg.at(j);
    K = K1.getK(0, 0);
  }
  mu1.normalize();
  mu2.normalize();

  std::vector<cheby::ArrayXd> Q = two_point_dist(mu2.vals(), mu1.vals(), K);

  auto Ki = K1.getK(1, 0);
  auto QK = Q;
  for (int ii=0; ii<L; ++ii) QK[ii] = Q[ii] * Ki[ii];

  return twoD_integral(QK)/twoD_integral(Q);
}


std::vector<cheby::ArrayXd> two_point_pred(int i, int j, message_t &messages, int L,
		marginal_t &marg,
                std::unordered_map<int, std::unordered_map<int,int>> const &W,
		KernelInt &K1
		 ) {
  floatT U = 0.0;
  cheby::Chebyshev mu1;
  cheby::Chebyshev mu2;
  Matrix K;
  if (messages[i].count(j)) {
    mu1 = messages.at(i).at(j);
    mu2 = messages.at(j).at(i);
    K = K1.getK(W.at(j).at(i), W.at(i).at(j));
  }
  else {
    mu1 = marg.at(i);
    mu2 = marg.at(j);
    K = K1.getK(0, 0);
  }
  mu1.normalize();
  mu2.normalize();

  return two_point_dist(mu2.vals(), mu1.vals(), K);

}



////

floatT sample1d(cheby::Chebyshev pdf, floatT u) {
    auto cdf = pdf.integrate();
    return (cdf - u).root();
}

struct sampleT { floatT x; floatT y; };

sampleT sample2d(std::vector<cheby::ArrayXd> Q, floatT u1, floatT u2, int L) {

    cheby::ArrayXd qx(L);
    for (int i=0; i<L; ++i) qx[i] = cheby::Chebyshev(Q[i], INIT_BY_VALUES).integrate().value(1.0);
    auto pdf_x = cheby::Chebyshev(qx, INIT_BY_VALUES);
    pdf_x.normalize();
    floatT x = sample1d(pdf_x, u1);

    auto QT = transpose(Q);
    cheby::ArrayXd qy(L);
    for (int i=0; i<L; ++i) qy[i] = cheby::Chebyshev(QT[i], INIT_BY_VALUES).value(x);
    auto pdf_y = cheby::Chebyshev(qy, INIT_BY_VALUES);
    pdf_y.normalize();
    floatT y = sample1d(pdf_y, u2);
    
    return {x, y};
}

////
