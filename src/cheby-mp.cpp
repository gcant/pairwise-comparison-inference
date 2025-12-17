#include <random>
#include "KernelIntegrations.hpp"
#include "mp-fns.hpp"

std::random_device rd;
std::mt19937 rng(rd());
std::uniform_real_distribution<floatT> unif(0.0,1.0);

// globals to be used when sharing across trials
Graph G;
KernelInt K1;
std::unordered_map<int, std::unordered_map<int,int>> W;
message_t messages;
marginal_t one_node_marginals;
cheby::ArrayXd prior;

extern "C" {

  void compute_Cheb_pts(double *out, int L) {
    auto pts = cheby::ChebPts(L);
    for (int i=0; i<L; ++i) out[i] = (double)pts[i];
  }

  void init_global_messages(int L) {
    elist_t m_order = Graph_to_list(G);  
    for (const auto &[i,j] : m_order)  messages[i][j] = cheby::Chebyshev(L);
  }

  void init_new_messages(int L) {
    elist_t m_order = Graph_to_list(G);  // message update order
    for (const auto &[i,j] : m_order) {
      if (not(messages[i].count(j))) messages[i][j] = cheby::Chebyshev(L);
    }
  }

  void init_global_prior(double *prior1, int L) {
    prior.resize(L);
    for (int i=0; i<L; ++i) prior[i] = prior1[i];
  }

  void init_graph(int num_nodes, int num_edges, int *edges, int *W_vals) {
    Graph G1;
    for (int i=0; i<num_nodes; ++i) G1.add_node(i);
    for (int u=0; u<num_edges; ++u) {
      int i = edges[2*u];
      int j = edges[2*u+1];
      W[i][j] = W_vals[2*u];
      W[j][i] = W_vals[2*u+1];
      G1.add_edge(i, j);
    }
    G = G1;
  }

  void init_kernels(int L, double *K_in, int num_edges, int *W_vals) {
    Matrix tmp;
    tmp.resize(L);
    for (int i=0; i<L; ++i) {
      tmp[i].resize(L, 0.0);
      for (int j=0; j<L; ++j) {
        tmp[i][j] = (floatT)K_in[(i*L)+j];
      }
    }
    K1.setKernel(L, tmp);
    for (int u=0; u<num_edges; ++u) {
      K1.getI(W_vals[2*u], W_vals[2*u+1]);
      K1.getI(W_vals[2*u+1], W_vals[2*u]);
    }
  }

  double pred(int i, int j, int L) {
    auto ans = predict(i, j, messages, L, one_node_marginals, W, K1);
    return (double)ans;
  }

  void two_pd(int i, int j, int L, double *out) {
    auto ans = two_point_pred(i, j, messages, L, one_node_marginals, W, K1);
    for (int u=0; u<L; ++u) {
      for (int v=0; v<L; ++v) {
        out[u*L + v] = (double) ans[u][v];
      }
    }
  }

  int skill_sampler(double *mean_marg_in, int L, int num_samps,
		  double *x_out, double *y_out){
    
    std::vector<cheby::ArrayXd> mean_marginal;
    mean_marginal.resize(L);
    for (int a=0; a<L; ++a) {
      mean_marginal[a].resize(L, 0.0);
      for (int b=0; b<L; ++b) mean_marginal[a][b] = mean_marg_in[a*L+b];
    }

    for (int s=0; s<num_samps; ++s) {
      floatT u1 = unif(rng);
      floatT u2 = unif(rng);
      auto xy = sample2d(mean_marginal, u1, u2, L);
      x_out[s] = xy.x;
      y_out[s] = xy.y;
    }
    return 1;
  }

  int full_algorithm_globals(
        int num_nodes,
        int num_edges,
        int *edges,
	int *W_vals,
        double tol,
        int L,
        double *prior1,
        double *K_in,
        double *output,
	double *mean_marg_out,
	int max_its,
	bool reset_messages,
	bool reset_kernels
        ) {

    init_global_prior(prior1, L);
    init_graph(num_nodes, num_edges, edges, W_vals);

    if (reset_messages) init_global_messages(L);
    init_new_messages(L);

    if (reset_kernels) init_kernels(L, K_in, num_edges, W_vals);
    //elist_t m_order = Graph_to_list(G);  // message update order
    
    int s=0;
    std::cout << "starting mp... " << std::endl;
    for (s; s<max_its; ++s) {
      //std::shuffle(m_order.begin(), m_order.end(), rng);  // randomize update order 
      if (iterate_messages_delta_parallel_kernel(G, messages, L, W, K1, prior) < tol) {
        break;
      }
    }
    std::cout << "done." << std::endl;

    one_node_marginals = compute_marginals_kernel(G, messages, L, W, K1, prior);

    floatT U = energy_kernel(G, messages, L, W, K1) + prior_energy(G, one_node_marginals, prior);   
    floatT S2 = message_entropy_kernel(G, messages, L, W, K1);   // two-point entropy
    floatT S1 = marginal_entropy(G, one_node_marginals, L); // one-point entropy
    floatT S = S2 - S1;
    floatT lnZ = U - S;

    auto mean_marginal = compute_mean_marginal(G, messages, L, W, K1);

    for (int i=0; i<L; ++i) {
      for (int j=0; j<L; ++j) {
        mean_marg_out[(i*L)+j] = mean_marginal[i][j];
      }
    }

    int t = 0;
    auto pts = cheby::ChebPts(L);
    for (t; t<pts.size(); ++t)  output[t] = (double) pts[t];
  
    for (int i=0; i<G.number_of_nodes(); ++i) {
      auto X = one_node_marginals[i].vals();
      for (int x=0; x<X.size(); ++x) {
        output[t] = (double) X[x];
        t++;
      }
    }
    output[t] = (double)S;
    ++t;
    output[t] = (double)lnZ;

    return s+1;

  }

}

