import numpy as np
import random, math
from scipy.special import binom

class GenF():
    def __init__(self):
        pass

    def get_random_feat(self):
        pass

    def get_feat_value(self, fid, ex):
        pass


class GenLinF(GenF):
    def __init__(self, X):
        self.d = X.shape[1]

    def get_random_feat(self):
        return random.randint(0, self.d-1)

    def get_feat_value(self, fid, ex):
        return ex[fid]


class GenHPolyF(GenF):
    def __init__(self, X, q):
        self.d = X.shape[1]
        self.q = q

    def get_random_feat(self):
        return tuple([random.randint(0, self.d-1) for i in range(self.q)])

    def get_feat_value(self, fid, ex):
        v = 1.0
        for i in range(self.q):
            v *= ex[fid[i]]
        return v

class GenDPKF(GenF):
    def __init__(self, X, q):
        self.d = X.shape[1]
        self.coeffs = np.array([binom(q, i) * c**(q-i) if i <= q else 0.0 for i in range(q+1)])
        self.n_coeffs = len(coeffs)
        self.p_coeffs = self.coeffs / float(sum(self.coeffs))
        self.nr = [self.d**q if self.coeffs[q] > 0.0 else 0 for q in range(self.n_coeffs)]
        self.sum_nr = sum(self.nr)

    def get_random_feat(self):
        q = np.random.choice(range(self.n_coeffs), p = self.p_coeffs)
        if q == 0: return tuple()
        return tuple(sorted([random.randint(0, self.d-1) for i in range(q)]))

    def get_feat_value(self, fid, ex):
        q = len(fid)
        w = (self.coeffs[q]*self.nr[q]/self.p_coeffs[q])**.5
        v = 1.0
        for f in fid:
            v *= ex[f]
        v *= w
        return v


class GenConjF(GenF):
    def __init__(self, X, q):
        self.d = X.shape[1]
        self.q = q

    def get_random_feat(self):
        return tuple(random.sample(range(self.d), self.q))

    def get_feat_value(self, fid, ex):
        v = 1.0
        for i in range(self.q):
            v *= ex[fid[i]]
        return v


class GenRuleF(GenF):
    def __init__(self, X, q):
        self.d = X.shape[1]
        self.q = q
        self.thresholds = [list(set(X[:,i])) for i in range(X.shape[1])]

    def get_random_feat(self):
        f = []
        for i in range(self.q):
            r = np.random.randint(self.d)
            t = np.random.choice(self.thresholds[r])
            rel = '>=' if np.random.random() < .5 else '<='
            f.append(tuple((r, t, rel)))
        return tuple(f)

    def get_feat_value(self, fid, ex):
        q = len(fid)
        for i in range(q):
            f = fid[i]
            r, t, rel = f
            if (rel == '>=') and (ex[r] < t): return 0.0
            if (rel == '<=') and (ex[r] > t): return 0.0
        return 1.0


class GenRuleEqF(GenF):
    def __init__(self, X, q):
        self.d = X.shape[1]
        self.q = q
        self.thresholds = [list(set(X[:,i])) for i in range(X.shape[1])]
        self.thresholds.append([-1])

    def get_random_feat(self):
        f = []
        for i in range(self.q):
            r = np.random.randint(self.d + 1)
            t = np.random.choice(self.thresholds[r])
            f.append(tuple((r, t, '==')))
        return tuple(f)

    def get_feat_value(self, fid, ex):
        q = len(fid)
        for i in range(q):
            f = fid[i]
            r, t, rel = f
            if (r < self.d) and (ex[r] != t): return 0.0
        return 1.0
