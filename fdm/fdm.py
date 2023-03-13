import scipy.io as sio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

L = 0.1

# Read layout map
def load_layout(path):
    path = Path(path)
    assert path.suffix == '.mat'
    return sio.loadmat(path).get('F')

def enforce_bcs(F, N):
    # Assume padded F, Neumann BCs
    F[1:N+1,0] = F[1:N+1:,2]
    F[0,1:N+1] = F[2,1:N+1]

    # Dirischlet Boundary Conditions
    F[99,1] = 298
    F[100,1] = 298    
    
# run fdm
def fdm(layout, n_iters=10):
    layout = np.pad(layout,(1,1),mode='constant')
    F = 300 * np.ones(layout.shape)
    F_next = np.copy(F)
    N = F.shape[0] - 2
    h = N / L
    for _ in range(n_iters):
        for i in range(1,N+1):
            for j in range(1,N+1):
                F[i,j] = 0.25 * (F[i+1,j] + F[i-1,j] + F[i,j+1] + F[i,j-1] + 0.001 * h ** 2 * layout[i,j])
                enforce_bcs(F, N)
        F = F_next
    return F