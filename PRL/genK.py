import numpy as np
import random

def apply2prefs(k_fun, p1, p2):
    (x1p, y1p), (x1n, y1n) = p1
    (x2p, y2p), (x2n, y2n) = p2
    res = 0.
    if y1p == y2p:
        res += k_fun(x1p, x2p)
    if y1n == y2n:
        res += k_fun(x1n, x2n)
    if y1p == y2n:
        res -= k_fun(x1p, x2n)
    if y1n == y2p:
        res -= k_fun(x1n, x2p)

    return res


class GenK:
    def __init__(self):
        pass

    def get_random_kernel(self):
        pass

    def get_kernel_function(self):
        pass


class GenKList(GenK):
    def __init__(self, k_list):
        self.kernel_list = k_list

    def get_random_kernel(self):
        return random.randint(0, len(self.kernel_list)-1)

    def get_kernel_function(self, d):
        return self.kernel_list[d]

    def __repr__(self):
        return "GenKList"


class GenHPK(GenK):
    def __init__(self, min_deg=2, max_deg=2):
        self.min_deg = min_deg
        self.max_deg = max(max_deg, min_deg)

    def get_random_kernel(self):
        return random.randint(self.min_deg, self.max_deg)

    def __repr__(self):
        return "GenHPK(dmin=%d, dmax=%d)" %(self.min_deg, self.max_deg)

    def get_kernel_function(self, d):
        return lambda p1, p2: apply2prefs(lambda x,z: np.dot(x,z)**d, p1, p2)
