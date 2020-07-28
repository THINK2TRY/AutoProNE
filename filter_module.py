import math
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import expm
from scipy.special import iv
from sklearn import preprocessing


class HeatKernel(object):
    def __init__(self, t=0.5, theta0=0.6, theta1=0.4):
        self.t = t
        self.theta0 = theta0
        self.theta1 = theta1

    def prop_adjacency(self, mx):
        mx_norm = preprocessing.normalize(mx.transpose(), "l1").transpose()
        adj = self.t * mx_norm
        adj.data = np.exp(adj.data)
        return adj / np.exp(self.t)

    def prop(self, mx, emb):
        adj = self.prop_adjacency(mx)
        return self.theta0 * emb + self.theta1 * adj.dot(emb)


class HeatKernelApproximation(object):
    def __init__(self, t=0.2, k=5):
        self.t = t
        self.k = k

    def taylor(self, mx, emb):
        mx_norm = preprocessing.normalize(mx, "l1")
        result = [math.exp(self.t) * emb]
        for i in range(self.k - 1):
            temp_mx = self.t * mx_norm.dot(result[-1]) / (i + 1)
            result.append(temp_mx)
        return sum(result)

    def chebyshev(self, mx, emb):
        mx = mx + sp.eye(emb.shape[0])
        mx = preprocessing.normalize(mx, "l1")
        conv = iv(0, self.t) * emb
        laplacian = sp.eye(emb.shape[0]) - mx
        Lx0 = emb
        Lx1 = laplacian.dot(emb)
        conv -= 2 * iv(1, self.t) * Lx1

        for i in range(2, self.k):
            Lx2 = 2 * laplacian.dot(Lx1) - Lx0
            conv += (-1) ** i * 2 * iv(i, self.t) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
        return conv

    def prop(self, mx, emb):
        return self.chebyshev(mx, emb)


class Gaussian(object):
    def __init__(self, mu=0.5, theta=1, rescale=False, k=3):
        self.theta = theta
        self.mu = mu
        self.k = k
        self.rescale = rescale
        self.coefs = [(-1) ** i * iv(i, self.theta) for i in range(k+3)]
        self.coefs[0] = self.coefs[0] / 2

    # adj: 1 mul + 3 add,  emb: 2*k mul, 3*k add
    def prop(self, mx, emb):
        row_num, col_sum = mx.shape
        mx = mx + sp.eye(row_num)
        mx_norm = preprocessing.normalize(mx, "l1")
        mx_hat = (1 - self.mu) * sp.eye(row_num) - mx_norm

        Lx0 = emb
        Lx1 = mx_hat.dot(emb)
        Lx1 = 0.5 * mx_hat.dot(Lx1) - emb

        conv = iv(0, self.theta) * Lx0
        conv -= 2 * iv(1, self.theta) * Lx1
        for i in range(2, self.k):
            Lx2 = mx_hat.dot(Lx1)
            Lx2 = (mx_hat.dot(Lx2) - 2 * Lx1) - Lx0

            # Lx2 = 2 * mx_hat.dot(Lx1) - Lx0
            conv += (-1) ** i * 2 * iv(i, self.theta) * Lx2
            Lx0 = Lx1
            Lx1 = Lx2
        if self.rescale:
            conv = mx.dot(emb - conv)
        return conv


class GaussianApproximation(object):
    def __init__(self,  mu=0.2, theta=1, k=2):
        self.theta = theta
        self.mu = mu
        # self.k = k

    def prop(self, mx, emb):
        row_num, col_sum = mx.shape
        mx = mx + sp.eye(row_num)
        mx_norm = preprocessing.normalize(mx, "l1")
        # laplacian = sp.eye(row_num) - mx_norm
        # mx_hat = laplacian - self.mu * sp.eye(row_num)
        mx_hat = (1 - self.mu) * sp.eye(row_num) - mx_norm

        lx1 = mx_hat.dot(emb)
        lx1 = -0.5 * self.theta * (mx_hat.dot(lx1) - emb)

        lx_result = emb + lx1

        lx2 = mx_hat.dot(lx1)
        lx2 = -0.5 * self.theta * (mx_hat.dot(lx2) - lx1)/2
        lx_result += lx2

        return lx_result


class PPR(object):
    """
        applying sparsification to accelerate computation
    """
    def __init__(self, alpha=0.5, k=10):
        self.alpha = alpha
        self.k = k
        self.alpha_list = [self.alpha * (1 - self.alpha) ** i for i in range(self.k)]
        self.epsilon = 1e-3

    def prop(self, mx, emb):
        row_num, col_num = mx.shape
        if row_num < 1000:
            # stable state
            row, col = mx.nonzero
            degree = mx.sum(1).A.squeeze()
            degree_sqrt_inv_mx = sp.csc_matrix((np.sqrt(1. / degree), (row, col)), shape=(row, col))
            identity = sp.csc_matrix((np.ones(row_num), (np.arange(row), np.arange(col_num))), shape=(row, col))
            ppr_mx = identity - (1 - self.alpha) * degree_sqrt_inv_mx.dot(mx).dot(degree_sqrt_inv_mx)
            return self.alpha * sp.linalg.inv(ppr_mx).dot(emb)
        else:
            # k-1 add, k-1 mul
            mx_norm = preprocessing.normalize(mx, "l1")

            Lx = emb
            conv = self.alpha * Lx
            for i in range(1, self.k):
                Lx = (1 - self.alpha) * mx_norm.dot(Lx)
                conv += Lx
            return conv


class SignalRescaling(object):
    """
        - rescale signal of each node according to the degree of the node:
            - sigmoid(degree)
            - sigmoid(1/degree)
    """
    def __init__(self):
        pass

    def prop(self, mx, emb):
        mx = preprocessing.normalize(mx, "l1")
        degree = mx.sum(1).A.squeeze()

        degree_inv = 1. / degree
        signal_val = 1./(1+np.exp(-degree_inv))

        row_num, col_num = mx.shape
        q_ = sp.csc_matrix((signal_val, (np.arange(row_num), np.arange(col_num))), shape=(row_num, col_num))

        adj_norm = mx.dot(q_)
        adj_norm = preprocessing.normalize(adj_norm, "l1")
        conv = adj_norm.dot(emb)
        return conv


class NodeAdaptiveEncoder(object):
    """
        - shrink negative values in signal/feature matrix
        - no learning
    """

    @staticmethod
    def prop(signal):
        mean_signal = signal.mean(1)
        mean_signal = 1./(1 + np.exp(-mean_signal))
        sel_row, sel_col = np.where(signal < 0)
        mean_signal = mean_signal[sel_row]
        signal[sel_row, sel_col] = signal[sel_row, sel_col] * mean_signal
        return signal
