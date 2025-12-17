import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.sparse
import networkx as nx
import chebyMP

def l2_penalty(model):
    ans = 0.0
    for p in model.parameters():
        if p.requires_grad:
            ans += (p**2).sum()
    return ans

class MLP(nn.Module):
    def __init__(self, hidden_size, p, l2_pen=0.0):
        super(MLP, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(2, hidden_size)
        self.drop1 = torch.nn.Dropout(p=p)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.drop2 = torch.nn.Dropout(p=p)
        self.output = nn.Linear(hidden_size, 1)
        self.l2_pen = l2_pen

    def forward(self, x_i, x_j):
        # Ensure both outputs are computed using the same sampled weights
        xij = torch.cat((x_i, x_j), dim=1)
        xji = torch.cat((x_j, x_i), dim=1)
        x = torch.stack([xij, xji], dim=0)
    
        # Forward pass through the network
        x = torch.relu(self.drop1(self.fc1(x)))
        x = torch.relu(self.drop2(self.fc2(x)))
        x = self.output(x)
        return x[0] - x[1]  # f_ij - f_ji

    def cheb_value(self, L):
        x = torch.tensor(chebyMP.ChebPts(L), dtype=torch.float32)
        return self.value(x)

    def value(self, x):
        L = len(x)
        x_i, x_j = torch.meshgrid(x, x, indexing='ij')
        pairs = torch.stack([x_i.flatten(), x_j.flatten()], dim=0)
        diff = self(pairs[0].unsqueeze(1), pairs[1].unsqueeze(1)).reshape(L, L)
        F = np.zeros((L,L))
        F += (1 / (1 + torch.exp(-diff))).detach().numpy().T
        return F

    def fit(self, mean_marg, opt, epochs=64):
        self.train()
        for epoch in range(epochs):
            x, y = chebyMP.sample_skills(mean_marg, 512)
            x_i = torch.tensor(x, dtype=torch.float).reshape(-1,1)
            x_j = torch.tensor(y, dtype=torch.float).reshape(-1,1)
            opt.zero_grad()
            loss = -torch.sum(nn.LogSigmoid()(self(x_i, x_j))) + 512 * self.l2_pen * l2_penalty(self)
            loss.backward()
            opt.step()
        self.eval()

def location_dependence(model, num_samps):
    X = np.linspace(-1, 1, num_samps)
    dxy = np.sqrt(2 * (X[1]-X[0])**2)
    M = np.array(model.value(torch.tensor(X, dtype=torch.float32)))
    ans = 0.0
    for k in range(1 - M.shape[0], M.shape[0]):
        d = np.diagonal(M, k)
        ans += np.sum((d - np.mean(d))**2) * dxy 
    return float(np.sqrt(ans * dxy))

def pred_diff(model, x0, x1):
    X = torch.tensor(np.array([x0, x1]), dtype=torch.float32)
    return float(model.value(X)[1,0])

def average_k_pcnt(model, k, num_samps=100):
    X = np.linspace(-0.999, 0.999 - k*0.02, num_samps)
    return float(np.mean(np.array([pred_diff(model, X[i], X[i]+k*0.02) for i in range(num_samps)])))

def summarize_model(model):
    return np.array([location_dependence(model, 1024),
            pred_diff(model, -0.98, 0.98),
            average_k_pcnt(model, 10, num_samps=500)
            ])

def EM(fname, L, num_its, hidden_size=50, dropout=0.5, l2_pen=0.0):
    W = np.genfromtxt(fname, dtype=int)
    W[np.diag_indices(len(W))] = 0
    W = scipy.sparse.csr_array(W)

    prior = chebyMP.BetaPDF(L, 1.0, 1.0)  # uniform prior for node skills

    ### infer the kernel
    K = chebyMP.BTKernel(L, 2.0)  # initial guess, logistic function
    s, pts, A, S, lnZ, mean_marg = chebyMP.runBP1(W, K, reset_messages=True, prior=prior, tol=1e-8)

    model = MLP(hidden_size, dropout, l2_pen=l2_pen/W.sum())
    opt = optim.Adam(model.parameters())
    for itr in range(num_its):
        s, pts, A, S, lnZ, mean_marg = chebyMP.runBP1(W, K, reset_messages=False, prior=prior, tol=0.00001)
        model.train()
        model.fit(mean_marg + 1e-8, opt)
        model.eval()
        K = 0.5*(K + model.cheb_value(L))
        print(itr, fname, lnZ, s)

    return model, K


if __name__=="__main__":
    files = [
         #'chess.txt',
         #'baboons.txt',
         #'mice.txt',
         #'soccer.txt',
         #'dogs.txt',
         #'online_chess_gm_2024_2.txt',
         #'cs_depts.txt',
         #'monkeys.txt',
         #'basketball.txt',
         #'business_depts.txt',
         #'tennis_2021_2022.txt',
         'synth_complex.txt',
         'synth_step.txt',
         'synth_uniform.txt',
         'synth_logistic.txt'
         ]

    for f in files:
        model, K = EM("win-loss-mats/"+f, 32, 50, hidden_size=64, dropout=0.0, l2_pen=1.0)
        torch.save(model.state_dict(), "kernels_NN/" + f + "_model_weights.pt")
        np.savetxt("kernels_NN/"+f, K)
        summary = np.array(summarize_model(model))
        np.savetxt("kernels_NN/summary_"+f, summary)

