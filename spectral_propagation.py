import numpy as np
import scipy
import time
from sklearn import preprocessing

from filter_module import PPR, Gaussian, SignalRescaling, HeatKernelApproximation


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name} cost time: {time.time() - self.t}s")


def propagate(mx, emb, stype, space=None):
    if space is not None:
        if stype == "heat":
            heat_kernel = HeatKernelApproximation(t=space["t"])
            result = heat_kernel.prop(mx, emb)
        elif stype == "ppr":
            ppr = PPR(alpha=space["alpha"])
            result = ppr.prop(mx, emb)
        elif stype == "gaussian":
            gaussian = Gaussian(mu=space["mu"], theta=space["theta"])
            result = gaussian.prop(mx, emb)
        elif stype == "sc":
            signal_rs = SignalRescaling()
            result = signal_rs.prop(mx, emb)
        else:
            raise ValueError("please use filter in ['heat', 'ppr', 'gaussian', 'sc'], currently use {}".format(stype))
    else:
        if stype == "heat":
            with Timer("HeatKernel") as t:
                heat_kernel = HeatKernelApproximation()
                result = heat_kernel.prop(mx, emb)
        elif stype == "ppr":
            with Timer("PPR") as t:
                ppr = PPR()
                result = ppr.prop(mx, emb)
        elif stype == "gaussian":
            with Timer("Gaussian") as t:
                gaussian = Gaussian()
                result = gaussian.prop(mx, emb)
        elif stype == "sc":
            with Timer("SignalRescaling") as t:
                signal_rs = SignalRescaling()
                result = signal_rs.prop(mx, emb)
        else:
            raise ValueError("please use filter in ['heat', 'ppr', 'gaussian', 'sc'], currently use {}".format(stype))
    return result


def get_embedding_dense(matrix, dimension):
    # get dense embedding via SVD
    U, s, Vh = scipy.linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
    U = np.array(U)
    U = U[:, :dimension]
    s = s[:dimension]
    s = np.sqrt(s)
    U = U * s
    U = preprocessing.normalize(U, "l2")
    return U