import numpy as np


def compute_k(n, e, d):
    r = 0.0
    s = 0.0
    for h in range(n + 1):
        if h == 0:
            r = n * np.log(1.0 - e)
        else:
            r += np.log(n - h + 1) - np.log(h) + np.log(e) - np.log(1.0 - e)
        s += np.exp(r)
        if s > d:
            if h == 0:
                # raise Exception()
                return None
            else:
                return h - 1
    return n
