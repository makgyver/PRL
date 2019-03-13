import numpy as np
import random
import math

#TODO documentation

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

    def get_pref_kernel_function(self):
        pass

    def get_kernel_function(self):
        pass


class GenKList(GenK):
    def __init__(self, k_list):
        self.kernel_list = k_list

    def __repr__(self):
        return "GenKList(n_kernel=%d)" %len(self.kernel_list)

    def get_random_kernel(self):
        return random.randint(0, len(self.kernel_list)-1)

    def get_pref_kernel_function(self, d):
        return lambda p1, p2: apply2prefs(self.get_kernel_function(d), p1, p2)

    def get_kernel_function(self, d):
        return self.kernel_list[d]


class GenHPK(GenK):
    def __init__(self, min_deg=2, max_deg=2):
        self.min_deg = min_deg
        self.max_deg = max(max_deg, min_deg)

    def __repr__(self):
        return "GenHPK(dmin=%d, dmax=%d)" %(self.min_deg, self.max_deg)

    def get_random_kernel(self):
        return random.randint(self.min_deg, self.max_deg)

    def get_pref_kernel_function(self, degree):
        return lambda p1, p2: apply2prefs(self.get_kernel_function(degree), p1, p2)

    def get_kernel_function(self, degree):
        return lambda x,z: np.dot(x,z)**degree


class GenRBFK(GenKList):
    def __init__(self, gamma_range):
        self.gamma_range = gamma_range

    def __repr__(self):
        return "GenRBFK(gamma_range=%s)" %(self.gamma_range)

    def get_random_kernel(self):
        return random.choice(self.gamma_range)

    def get_pref_kernel_function(self, gamma):
        return lambda p1, p2: apply2prefs(self.get_kernel_function(gamma), p1, p2)

    def get_kernel_function(self, gamma):
        return lambda x,z: math.exp(-gamma * np.linalg.norm(x-z)**2)
        #return lambda x,z: math.exp(-gamma * np.sum((x-z)**2))
