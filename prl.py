import math
import numpy as np

from genF import *
from genP import *

import logging
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)


class PRL:

    def __init__(self, gen_pref, gen_feat, dim, n_cols, solver):
        self.gen_pref = gen_pref
        self.gen_feat = gen_feat
        self.n_cols = n_cols
        self.dim = dim
        self.solver = solver

        self.pref_list = self.gen_pref.get_all_prefs()
        self.n_rows = len(self.pref_list)
        self.feat_list = []
        self.feat_set = set()

        self.M = np.zeros((self.n_rows, self.n_cols))
        self.Q = None

    def __repr__(self):
        return "PRL(gen_pref=%s, gen_feat=%s, n_rows=%d, n_cols=%d, solver=%s)"\
            %(self.gen_pref, self.gen_feat, self.n_rows, self.n_cols, self.solver)

    def get_random_pair(self):
        return (self.gen_pref.get_random_pref(), self.gen_feat.get_random_feat())


    def col_pref_repr(self, q, f):
        r = np.zeros(self.dim)
        (x_p, y_p), (x_n, y_n) = self.gen_pref.get_pref_value(q)
        r[y_p] = +self.gen_feat.get_feat_value(f, x_p)
        r[y_n] = -self.gen_feat.get_feat_value(f, x_n)
        return r


    def row_prefs_repr(self, f):
        R = np.zeros((self.n_rows, self.dim))
        for i, q in enumerate(self.pref_list):
            (x_p, y_p), (x_n, y_n) = self.gen_pref.get_pref_value(q)
            R[i, y_p] = +self.gen_feat.get_feat_value(f, x_p)
            R[i, y_n] = -self.gen_feat.get_feat_value(f, x_n)
        return R


    #TODO optimize this
    def _get_new_col(self):
        (p, f) = self.get_random_pair()
        rp = self.col_pref_repr(p, f)
        while (p, f) in self.feat_set or not rp.any():
            (p, f) = self.get_random_pair()
            rp = self.col_pref_repr(p, f)
        return (p, f), rp


    def fit(self, iterations=1000, verbose=False):
        if verbose:
            logging.info("Starting training of PRL\n%s" %self)
            logging.info("Matrix game initialization...")

        for j in xrange(self.n_cols):
            (p, f), rp = self._get_new_col()
            self.feat_list.append((p, f))
            self.feat_set.add((p, f))
            R = self.row_prefs_repr(f)
            x = np.dot(R, rp)
            self.M[:,j] = x

        #iterative updates
        for t in xrange(iterations):
            if verbose: logging.info("PRL iteration %d" %(t+1))
            (P, Q, V) = self.solver.solve(self.M, self.n_rows, self.n_cols)
            if verbose: logging.info("Value of the game: %.6f" %V)
            if (t+1 < iterations):
                for j in xrange(self.n_cols):
                    if Q[j] <= 0:
                        (p, f), rp = self._get_new_col()
                        self.feat_set.remove(self.feat_list[j])
                        self.feat_list[j] = (p, f)
                        self.feat_set.add((p, f))
                        R = self.row_prefs_repr(f)
                        x = np.dot(R, rp)
                        self.M[:,j] = x
            if verbose:
                logging.info("# of kept columns: %d" %(np.sum(Q>0)))
                logging.info("# of unique fetures: %d\n" %len(set([f for i, (p, f) in enumerate(self.feat_list) if Q[i]>0.])))
        self.Q = Q


    def get_best_features(self, k=10):
        list_w_feat = [(self.Q[i], pf) for i, pf in enumerate(self.feat_list) if self.Q[i] > 0]
        list_w_feat.sort(reverse=True)

        return [(pf, q) for (q, pf) in list_w_feat[:k]]
