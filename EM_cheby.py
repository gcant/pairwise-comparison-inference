import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
jax.config.update("jax_enable_x64", True)
import scipy.optimize
import networkx as nx
import chebyMP

def monotone_sum(x, L, p=8.0):
    idx = jnp.triu_indices(L, k=1)
    M = jnp.zeros((L,L))
    M = M.at[idx].set(jnp.abs(x)**p)
    tmp = jnp.flip(jnp.cumsum(jnp.flip(jnp.cumsum(M, axis=1),axis=0),axis=0),axis=0)[idx]**(1.0/p)
    M = M.at[idx].set(tmp)
    return M

def toAntiSymMat(x, L):
    M = monotone_sum(x, L)
    return M - M.T

def chebTransform(F):
    c = F.copy()
    c = jax.scipy.fft.dct(c, type=2, axis=0)
    c = c.at[0, :].set(c[0, :] / 2)
    c = jax.scipy.fft.dct(c, type=2, axis=1)
    c = c.at[:, 0].set(c[:, 0] / 2)
    return c / (F.shape[0]**2)

def loglik_integral(f, mu, I):
    logK = jax.nn.log_sigmoid(toAntiSymMat(f, mu.shape[0]))
    return I @ chebTransform(mu * logK) @ I

def regularized_loglik_integral(f, mu, I, l2_reg):
    ll = loglik_integral(f, mu, I)
    c = chebTransform(toAntiSymMat(f, mu.shape[0]))
    return ll - jnp.sum( l2_reg * (c**2) )

rl = jax.jit(regularized_loglik_integral)
d_rl = jax.jit(jax.grad(regularized_loglik_integral, argnums=0))
d2_rl = jax.jit(jax.hessian(regularized_loglik_integral, argnums=0))

class ChebyFit(object):
    def __init__(self, L, l2_reg):
        self.L = L
        I = np.zeros(L)
        I[0] = 2.0
        for n in range(2, L):
            I[n] = ((-1)**n + 1) / (1 - n**2)
        self.I = jnp.array(I)
        self.pts = jnp.cos((2*jnp.arange(L)+1) * jnp.pi / (2*L))
        self.f = jnp.array(np.random.randn(int(L*(L-1)/2)))
        self.l2_reg = l2_reg

    def K(self):
        return jax.nn.sigmoid(toAntiSymMat(self.f, self.L))

    def fit_scipy(self, mu, method):
        l1 = lambda f: -rl(f, mu, self.I, self.l2_reg)
        j1 = lambda f: -d_rl(f, mu, self.I, self.l2_reg)
        h1 = lambda f: -d2_rl(f, mu, self.I, self.l2_reg)
        ans = scipy.optimize.minimize(l1, self.f, jac=j1, hess=h1, method=method)
        self.f = jnp.zeros_like(self.f) + ans.x
        return ans

    def eval(self, x):
        c = chebTransform(toAntiSymMat(self.f, self.L))
        Tx = jnp.cos(jnp.outer(jnp.arange(self.L), jnp.arccos(x)))
        return jax.nn.sigmoid(jnp.einsum('ab,ai,bj->ij', c, Tx, Tx))

    def savetxt(self, fname):
        np.savetxt(fname, self.f)

    def loadtxt(self, fname):
        self.f = jnp.zeros_like(self.f) + np.genfromtxt(fname)


def location_dependence(model, num_samps):
    X = np.linspace(-1, 1, num_samps)
    dxy = np.sqrt(2 * (X[1]-X[0])**2)
    M = np.array(model.eval(X))
    ans = 0.0
    for k in range(1 - M.shape[0], M.shape[0]):
        d = np.diagonal(M, k)
        ans += np.sum((d - np.mean(d))**2) * dxy 
    return np.sqrt(ans * dxy)

def pred_diff(model, x0, x1):
    X = np.array([x0, x1])
    return float(model.eval(X)[1,0])

def average_k_pcnt(model, k, num_samps=100):
    X = np.linspace(-0.999, 0.999 - k*0.02, num_samps)
    return np.mean(np.array([pred_diff(model, X[i], X[i]+k*0.02) for i in range(num_samps)]))

def summarize_model(model):
    return (location_dependence(model, 1024),
            pred_diff(model, -0.98, 0.98),
            average_k_pcnt(model, 10, num_samps=500)
            )


def EM(fname, L, num_its, v=1.):
    W = np.genfromtxt(fname, dtype=int)
    W[np.diag_indices(len(W))] = 0
    W = scipy.sparse.csr_array(W)

    prior = chebyMP.BetaPDF(L, 1.0, 1.0)  # uniform prior for node skills

    # regularization
    a = jnp.outer(jnp.arange(L), jnp.ones(L))
    l2_reg = (v / 64) * (a**2 + (a.T)**2)**2  # equiv to std dev of c_kj ~ 1 / (a**2 + b**2)

    ### infer the kernel
    K = chebyMP.BTKernel(L, 2.0)  # initial guess, logistic function
    s, pts, A, S, lnZ, mean_marg = chebyMP.runBP1(W, K, reset_messages=True, prior=prior, tol=1e-8)

    model = ChebyFit(L, l2_reg) 
    for itr in range(num_its):
        s, pts, A, S, lnZ, mean_marg = chebyMP.runBP1(W, K, reset_messages=False, prior=prior, tol=0.00001)
        opt = model.fit_scipy(mean_marg + 1e-8, method='Newton-CG')
        K = np.array(model.K()).T
        print(itr, fname, lnZ, s)

    return model, K


if __name__=="__main__":
    files = [
     #    'chess.txt',
     #    'baboons.txt',
     #    'mice.txt',
     #    'soccer.txt',
     #    'dogs.txt',
     #    'online_chess_gm_2024_2.txt',
     #    'cs_depts.txt',
     #    'monkeys.txt',
     #    'basketball.txt',
     #    'business_depts.txt',
     #    'tennis_2021_2022.txt',
         'synth_complex.txt',
         'synth_step.txt',
         'synth_uniform.txt',
         'synth_logistic.txt'
         ]

    for f in files:
        model, K = EM("win-loss-mats/"+f, 32, 100)
        model.savetxt("kernels_cheb/f_"+f)
        np.savetxt("kernels_cheb/"+f, K)
        summary = np.array(summarize_model(model))
        np.savetxt("kernels_cheb/summary_"+f, summary)

