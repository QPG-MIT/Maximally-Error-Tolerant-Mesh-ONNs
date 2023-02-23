import numpy as np
import meshes as ms
import matplotlib.pyplot as plt
from scipy.stats import unitary_group
from numba import njit

class EOIntensityModulation:
    def __init__(self, alpha=0.1, g=np.pi, phi_b=np.pi):
        self.alpha = alpha; self.g = g; self.phi_b = phi_b
    def __call__(self, inputs):
        (a, g, p) = (self.alpha, self.g, self.phi_b); Z = inputs
        return (1j*np.sqrt(1-a) * np.exp(-1j*0.5*g*np.conj(Z)*Z-1j*0.5*p) * np.cos(0.5*g*np.conj(Z)*Z + 0.5*p) * Z)

def clemdec_out(U, dp, eta, mzi3: bool=False):
    U = np.array(U)
    V = np.eye(len(U), dtype=complex); W = np.eye(len(U), dtype=complex)
    (err, d, v, w) = clemdec_out_helper(U, V, W, dp, eta, mzi3=mzi3)
    return (err, v @ np.diag(d) @ w)

@njit
def clemdec_out_helper(U, V, W, dp, eta, mzi3: bool=False):
    N = len(U)
    ind = 0
    for i in range(N-1):
        for j in range(i+1):
            z = np.tan(eta + dp[ind, 0]) if mzi3 else 0 
            (a, b) = (dp[ind, 1], dp[ind, 2])
            ind += 1
            if (i % 2 == 0):
                (k, l) = (N-1-j, i-j); #print (i, j, k, l, 'row')
                (x, y) = U[k, l:l+2]
                s = -y/x
                s = (s + 1j*z) / (1 + 1j*s*z)
                abs_s = np.abs(s);
                s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
                s = (s - 1j*z) / (1 - 1j*s*z)
                (theta, phi) = (2*np.arctan(np.abs(s)), np.angle(1j*s))
                T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
                              [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
                U[:, l:l+2] = U[:, l:l+2].dot(T)
                W[l:l+2, :] = T.T.conj().dot(W[l:l+2, :])
            else:
                (k, l) = (N-2-i+j, j); #print (i, j, k+1, l, 'col')
                (x, y) = U[k:k+2, l]
                s = -x/y
                s = (s + 1j*z) / (1 + 1j*s*z)
                abs_s = np.abs(s);
                s *= min(max(abs_s, np.abs(np.tan(np.abs(a+b)))), np.abs(1/np.tan(np.abs(a-b)+1e-30))) / (abs_s)
                s = (s - 1j*z) / (1 - 1j*s*z)
                (theta, phi) = (2*np.arctan(np.abs(s)), -np.angle(1j*s))
                T = np.array([[np.exp(1j*phi)*np.sin(theta/2), 1j*np.cos(theta/2)],
                              [1j*np.cos(theta/2), np.exp(-1j*phi)*np.sin(theta/2)]])
                U[k:k+2] = T.dot(U[k:k+2])
                V[:, k:k+2] = V[:, k:k+2].dot(T.T.conj())
            #print (np.round(np.abs(U), 3))
    return (np.linalg.norm(U - np.diag(np.diag(U))) / np.sqrt(N), np.diag(U), V, W)

#%%

def run_stuff(nn, data, datalabels, dpList):
    eo = EOIntensityModulation(alpha=0.1, g=0.05*np.pi, phi_b=np.pi)
    U   = [lyr.matrix() for lyr in nn]
    U0  = [lyr.matrix(p_splitter=dp[:,:2]) for (lyr, dp) in zip(nn, dpList)]
    Uc  = [clemdec_out(lyr, dp, 0.)[1] for (lyr, dp) in zip(U, dpList)]
    Uc3 = [clemdec_out(lyr, dp, np.pi/4, mzi3=True)[1] for (lyr, dp) in zip(U, dpList)]
    out = []
    # print (U0[0].shape, U0[1].shape, data.shape, len(U0), len(Uc), len(Uc3))
    for Ulist in [U0, Uc, Uc3]:
        x = data
        for U in Ulist:
            # print (x.shape, U.shape)
            x = x.dot(U.T)       # Synaptic weights (MZI mesh)
            x = eo(x)            # Activations
        x = np.abs(x)**2         # Amplitude -> Power
        x = np.real(x[:, :10])   # Only use first 10 channels.
        (Z, Z0) = (x.argmax(1), datalabels)
        out.append(np.mean(Z == Z0))
    return np.array(out)
# def run_nn(N, err=lambda U: U):
#     nn = onn_models[()][str(N)]; nn = (err(nn['layer1']), err(nn['layer2']))
#     data = data_ft[N]
#     return run_stuff(nn, data)