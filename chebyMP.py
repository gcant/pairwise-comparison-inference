import ctypes
import numpy as np
import scipy.sparse
import scipy.special

# C arrays of ints/doubles using numpy
array_int = np.ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1, flags='CONTIGUOUS')
array_double = np.ctypeslib.ndpointer(dtype=np.double,ndim=1, flags='CONTIGUOUS')

lib = ctypes.cdll.LoadLibrary("./out/chebyMP.so")

lib.compute_Cheb_pts.argtypes = [ array_double, ctypes.c_int ]
lib.compute_Cheb_pts.restype = None

lib.full_algorithm_globals.argtypes = [
        ctypes.c_int,     # int num_nodes
        ctypes.c_int,     # int num_edges
        array_int,        # int *edges
        array_int,        # int *W_in
        ctypes.c_double,  # double tol
        ctypes.c_int,     # int L
        array_double,     # double *prior1
        array_double,     # double *K_in
        array_double,     # double *output
        array_double,     # double *mean_marg_out
        ctypes.c_int,     # max_its
        ctypes.c_bool,    # reset_messages
        ctypes.c_bool]    # reset_kernels

lib.full_algorithm_globals.restype = ctypes.c_int 



lib.skill_sampler.argtypes = [
        array_double,     # double *mean_marg_in
        ctypes.c_int,     # int L
        ctypes.c_int,     # int num_samps
        array_double,     # double *x_out
        array_double]     # double *y_out

lib.skill_sampler.restype = ctypes.c_int 


lib.pred.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.pred.restype = ctypes.c_double

lib.two_pd.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, array_double]

def twopd(i, j, L):
    ans = np.ascontiguousarray(np.zeros((L*L)).astype(ctypes.c_double))
    lib.two_pd(ctypes.c_int(i), ctypes.c_int(j), ctypes.c_int(L), ans)
    return ans.reshape(L, L)

def predict(i, j, L):
    return lib.pred(ctypes.c_int(i), ctypes.c_int(j), ctypes.c_int(L))

def ChebPts(L):
    ans = np.zeros(L)
    lib.compute_Cheb_pts(ans,L)
    return ans

def BetaPDF(L, alpha, beta):
    X = ChebPts(L)
    P = (((X+1)/2.) ** (alpha-1)) * (((1-X)/2.) ** (beta-1))
    P = P / (2 * scipy.special.beta(alpha,beta))
    return P

def BTKernel(L, beta):
    K = np.zeros((L, L))
    X = ChebPts(L)
    for i in range(L):
        for j in range(L):
            K[j][i] = 1.0 / (1.0 + np.exp(-beta*(X[i]-X[j])))
            #K[i][j] = 1.0 / (1.0 + np.exp(beta*(X[i]-X[j])))
    return K

def runBP1(W, K, reset_messages, prior=None, tol=1e-5, max_its=50, reset_kern=True):
    L = K.shape[0]
    n = W.shape[0]
    ii, jj = (W+W.T).nonzero()

    if prior is None:
        prior = BetaPDF(L, 1, 1)
    prior = np.ascontiguousarray(prior)

    edge_list = []
    W_in = []
    num_edges = 0
    for u in range(len(ii)):
        i = ii[u]
        j = jj[u]
        if (i < j):
            edge_list.extend([i,j])
            W_in.extend([W[i,j], W[j,i]])
            num_edges += 1

    M = np.ascontiguousarray(np.array(edge_list, dtype=ctypes.c_int).flatten().astype(ctypes.c_int))
    W_in = np.ascontiguousarray(np.array(W_in, dtype=ctypes.c_int))
    K_in = np.ascontiguousarray(np.array(K).flatten())

    output = np.ascontiguousarray(np.zeros(L*(n+1) + 2))
    mean_marg_out = np.ascontiguousarray(np.zeros(L*L))

    s = lib.full_algorithm_globals(
        ctypes.c_int(n),
        ctypes.c_int(num_edges),
        M,
        W_in,
        ctypes.c_double(tol),
        ctypes.c_int(L),
        prior,
        K_in,
        output,
        mean_marg_out,
        ctypes.c_int(max_its),
        ctypes.c_bool(reset_messages),
        ctypes.c_bool(reset_kern)
        )

    pts = output[:L].copy()
    A = output[L:-2].reshape(n,L)
    S = output[-2]
    lnZ = output[-1]
    mean_marg = mean_marg_out.reshape(L,L)

    return s, pts, A, S, lnZ, mean_marg 

def sample_skills(mean_marg, num_samps):
    L = len(mean_marg)
    mm = np.ascontiguousarray(mean_marg.flatten())
    x_out = np.ascontiguousarray(np.zeros(num_samps))
    y_out = np.ascontiguousarray(np.zeros(num_samps))
    s = lib.skill_sampler(mm, ctypes.c_int(L), ctypes.c_int(num_samps), x_out, y_out)
    return x_out, y_out

if __name__=="__main__":
    W = np.zeros((4,4))
    W[0,1] = 2
    W[1,2] = 1
    W[2,1] = 1
    W[0,3] = 3
    W = scipy.sparse.csr_array(W)
    K = BTKernel(32, 2.0)
    pr = np.ones(32) / 2.0
    s, pts, A, S, lnZ, mean_marg = runBP1(W, K, reset_messages=True, prior=pr, tol=1e-8)
    print(np.exp(lnZ)) # 0.020205986


