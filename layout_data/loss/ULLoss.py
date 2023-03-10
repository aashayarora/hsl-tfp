import torch
from torch.nn import MSELoss
from torch.nn.functional import conv2d, pad, interpolate
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

""" Loss function with hard constraints 

"""
class Jacobi_layer(torch.nn.Module):
    def __init__(self, nx=21, length=0.1, bcs=None):
        super().__init__()
        self.length = length
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # Padding
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat, n_iter):
        # Source item
        f = self.cof * layout
        # The nodes which are not in boundary
        G = torch.ones_like(heat).detach()

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = F.pad(heat * G, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x

""" Loss function with soft constraints 

"""
class Jacobi_layerSoft(torch.nn.Module):
    def __init__(
            self, nx=21, length=0.1, bcs=None
    ):
        super(Jacobi_layerSoft, self).__init__()
        self.length = length
        self.bcs = bcs
        # The weight 1/4(u_(i, j-1), u_(i, j+1), u_(i-1, j), u_(i+1, j))
        self.weight = torch.Tensor([[[[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]]]])
        # Padding
        self.nx = nx
        self.scale_factor = 1  # self.nx/200
        TEMPER_COEFFICIENT = 1  # 50
        STRIDE = self.length / (self.nx - 1)
        # ((l/(nx))^2)/(4*cof)*m*input(x, y)
        self.cof = 0.25 * STRIDE ** 2 / TEMPER_COEFFICIENT

    def jacobi(self, x):
        return conv2d(x, self.weight.to(device=x.device), bias=None, stride=1, padding=0)

    def forward(self, layout, heat, n_iter):
        # Source item
        f = self.cof * layout
        # G: the nodes which are not in boundary
        G = torch.ones_like(heat).detach()

        if self.bcs is None or len(self.bcs) == 0 or len(self.bcs[0]) == 0:  # all are Dirichlet bcs
            pass
        else:
            for bc in self.bcs:
                if bc[0][1] == 0 and bc[1][1] == 0:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, :1] = torch.zeros_like(G[..., idx_start:idx_end, :1])
                elif bc[0][1] == self.length and bc[1][1] == self.length:
                    idx_start = round(bc[0][0] * self.nx / self.length)
                    idx_end = round(bc[1][0] * self.nx / self.length)
                    G[..., idx_start:idx_end, -1:] = torch.zeros_like(G[..., idx_start:idx_end, -1:])
                elif bc[0][0] == 0 and bc[1][0] == 0:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., :1, idx_start:idx_end] = torch.zeros_like(G[..., :1, idx_start:idx_end])
                elif bc[0][0] == self.length and bc[1][0] == self.length:
                    idx_start = round(bc[0][1] * self.nx / self.length)
                    idx_end = round(bc[1][1] * self.nx / self.length)
                    G[..., -1:, idx_start:idx_end] = torch.zeros_like(G[..., -1:, idx_start:idx_end])
                else:
                    raise ValueError("bc error!")
        for i in range(n_iter):
            if i == 0:
                x = F.pad(heat, [1, 1, 1, 1], mode='reflect')
            else:
                x = F.pad(x, [1, 1, 1, 1], mode='reflect')
            x = G * (self.jacobi(x) + f)
        return x

""" Weighted Loss

"""
class OHEMF12d(torch.nn.Module):


    def __init__(self, loss_fun, weight=None):
        super(OHEMF12d, self).__init__()
        self.weight = weight
        self.loss_fun = loss_fun

    def forward(self, inputs, targets):
        diff = self.loss_fun(inputs, targets, reduction='none').detach()
        min, max = torch.min(diff.view(diff.shape[0], -1), dim=1)[0], torch.max(diff.view(diff.shape[0], -1), dim=1)[0]
        if inputs.ndim == 4:
            min, max = min.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1, 1).expand(diff.shape)
        elif inputs.ndim == 3:
            min, max = min.reshape(diff.shape[0], 1, 1).expand(diff.shape), \
                       max.reshape(diff.shape[0], 1, 1).expand(diff.shape)
        diff = 10.0 * (diff - min) / (max - min)
        return torch.mean(torch.abs(diff * (inputs - targets)))