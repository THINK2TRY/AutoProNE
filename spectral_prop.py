import numpy as np
import scipy
from sklearn import preprocessing
import time

from filter_module import HeatKernel, PPR, Gaussian, SignalRescaling, HeatKernelApproximation


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"{self.name} cost time: {time.time() - self.t}s")


def propagate(mx, emb, stype, space=None, resale=False):
    if space is not None:
        k = space["k"]
        if stype == "heat":
            # negative
            heat_kernel = HeatKernelApproximation(space["t"], k)
            result = heat_kernel.prop(mx, emb)
        elif stype == "ppr":
            # relatively good,
            ppr = PPR(space["alpha"], k)
            result = ppr.prop(mx, emb)
        elif stype == "gaussian":
            # little positive, but almost zero effects
            rescale = space["rescale"] == 1
            gaussian = Gaussian(space["mu"], space["theta"], k, rescale)
            result = gaussian.prop(mx, emb)
        elif stype == "sc":
            # negative
            signal_rs = SignalRescaling()
            result = signal_rs.prop(mx, emb)
        else:
            raise ValueError("please use filter in ['heat', 'ppr', 'gaussian', 'sc']")
    else:
        if stype == "heat":
            with Timer("HeatKernel") as t:
                # negative
                heat_kernel = HeatKernelApproximation()
                result = heat_kernel.prop(mx, emb)
        elif stype == "ppr":
            with Timer("PPR") as t:
                # relatively good,
                ppr = PPR()
                result = ppr.prop(mx, emb)
        elif stype == "gaussian":
            with Timer("Gaussian") as t:
                # little positive, but almost zero effects
                gaussian = Gaussian(rescale=resale)
                result = gaussian.prop(mx, emb)
        elif stype == "sc":
            with Timer("SignalRescaling") as t:
                # negative
                signal_rs = SignalRescaling()
                result = signal_rs.prop(mx, emb)
        else:
            raise ValueError("please use filter in ['heat', 'ppr', 'gaussian', 'sc']")
    return result


def get_embedding_dense(matrix, dimension):
    # get dense embedding via SVD
    t1 = time.time()
    U, s, Vh = scipy.linalg.svd(matrix, full_matrices=False, check_finite=False, overwrite_a=True)
    U = np.array(U)
    U = U[:, :dimension]
    s = s[:dimension]
    s = np.sqrt(s)
    U = U * s
    U = preprocessing.normalize(U, "l2")
    return U