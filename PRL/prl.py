import math
import numpy as np

from .genF import *
from .genP import *

import logging

__docformat__ = 'reStructuredText'

class PRL:
    """This class implements the Preference and Rule Learning (PRL) algorithm.

    The current implementation of PRL uses a fixed budget of columns and it is
    designed for label ranking tasks (instance ranking is currently not supported).
    PRL is described in the paper:
    `'Interpretable preference learning: a game theoretic framework for large margin
    on-line feature and rule learning' by M.Polato and F.Aiolli, AAAI 2019.
    <https://arxiv.org/abs/1812.07895>`_.
    """

    def __init__(self, gen_pref, gen_feat, dim, n_cols, solver):
        """Initializes all the useful structures.

        :param gen_pref: the preference generator. See <:genP.GenMacroP> and <:genP.GenMicroP>
        :param gen_feat: the feature generator
        :param dim: number of possible labels
        :param n_cols: number of columns of the matrix sub-game
        :param solver: game solver. See for example <:solvers.FictitiousPlay>
        :type gen_pref: object of class which inherits from <:genP.GenP>, e.g., GenMacroP
        :type gen_feat: object of class which inherits from <:genF.GenF>, e.g., GenHPolyF
        :type dim: int
        :type n_cols: int
        :type solver: object of class which inherits from <:solvers.Solver>
        """
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
        """Returns a string representation of the PRL object.

        :returns: return a string representation of the PRL object
        :rtype: string
        """
        return "PRL(gen_pref=%s, gen_feat=%s, n_rows=%d, n_cols=%d, solver=%s)"\
            %(self.gen_pref, self.gen_feat, self.n_rows, self.n_cols, self.solver)

    def get_random_pair(self):
        """Returns a new random columns composed by a preference-feature pair.

        :returns:  a new random columns composed by a preference-feature pair
        :rtype: tuple(preference, feature)
        """
        return (self.gen_pref.get_random_pref(), self.gen_feat.get_random_feat())


    def col_pref_repr(self, q, f):
        """Computes the representation of the preference q w.r.t. the feature f.

        :param q: the new preference
        :param f: the new feature
        :type q: tuple
        :type f: int, tuple
        :returns: the representation of the preference q w.r.t. f.
        :rtype: numpy.ndarray
        """
        r = np.zeros(self.dim)
        (x_p, y_p), (x_n, y_n) = self.gen_pref.get_pref_value(q)
        r[y_p] = +self.gen_feat.get_feat_value(f, x_p)
        r[y_n] = -self.gen_feat.get_feat_value(f, x_n)
        return r


    def row_prefs_repr(self, f):
        """Computes the representation of all the preferences w.r.t. the selected feature f.

        :param f: a feature
        :type f: int, tuple
        :returns: the representation of all the preferences w.r.t. f
        :rtype: numpy.ndarray
        """
        R = np.zeros((self.n_rows, self.dim))
        for i, q in enumerate(self.pref_list):
            (x_p, y_p), (x_n, y_n) = self.gen_pref.get_pref_value(q)
            R[i, y_p] = +self.gen_feat.get_feat_value(f, x_p)
            R[i, y_n] = -self.gen_feat.get_feat_value(f, x_n)
        return R


    def _get_new_col(self):
        """Internal method that randomly pick a new column in such a way that its representation is not null and it is not already in the game matrix.

        :returns: a not null representation of a random picked preference-feature pair
        :rtype: tuple
        """
        (p, f) = self.get_random_pair()
        rp = self.col_pref_repr(p, f)
        while (p, f) in self.feat_set or not rp.any():
            (p, f) = self.get_random_pair()
            rp = self.col_pref_repr(p, f)
        return (p, f), rp


    def fit(self, iterations=1000, verbose=False):
        """Trains the PRL method.

        :param iterations: the number of iterations of PRL
        :param verbose: whether the output is verbose or not
        :type iterations: int
        :type verbose: bool
        """
        if verbose:
            logging.info("Starting training of %s" %self)
            logging.info("Matrix game initialization...")

        for j in range(self.n_cols):
            (p, f), rp = self._get_new_col()
            self.feat_list.append((p, f))
            self.feat_set.add((p, f))
            R = self.row_prefs_repr(f)
            x = np.dot(R, rp)
            self.M[:,j] = x

        #iterative updates
        for t in range(iterations):
            if verbose: logging.info("PRL iteration %d/%d" %(t+1, iterations))
            (P, Q, V) = self.solver.solve(self.M, self.n_rows, self.n_cols)
            if verbose: logging.info("Value of the game (current margin): %.6f" %V)
            if (t+1 < iterations):
                for j in range(self.n_cols):
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
        """Returns the k best features sorted by their weights.

        :param k: the number of most relevant features to retrieve
        :type k: int
        :returns: the k best features (along with their weights) sorted by their weights
        :rtype: list of tuples
        """
        list_w_feat = [(self.Q[i], pf) for i, pf in enumerate(self.feat_list) if self.Q[i] > 0]
        list_w_feat.sort(reverse=True)

        return [(pf, q) for (q, pf) in list_w_feat[:k]]





class PRL_ext(PRL):
    
    def fit(self, iterations=1000, verbose=False):

        if verbose:
            logging.info("Starting training of %s" %self)
            logging.info("Matrix game initialization...")

        for j in range(self.n_cols):
            (p, f), rp = self._get_new_col()
            self.feat_list.append((p, f))
            self.feat_set.add((p, f))
            R = self.row_prefs_repr(f)
            x = np.dot(R, rp)
            self.M[:,j] = x

        #iterative updates
        for t in range(iterations):
            if verbose: logging.info("PRL_ext iteration %d/%d" %(t+1, iterations))
            (P, Q, V) = self.solver.solve(self.M, self.n_rows, self.M.shape[1])
            if verbose: logging.info("Value of the game (current margin): %.6f" %V)
            if (t+1 < iterations):
                for j in range(self.M.shape[1]):
                    if Q[j] <= 0:
                        (p, f), rp = self._get_new_col()
                        self.feat_set.remove(self.feat_list[j])
                        self.feat_list[j] = (p, f)
                        self.feat_set.add((p, f))
                        R = self.row_prefs_repr(f)
                        x = np.dot(R, rp)
                        self.M[:,j] = x
                
                n_zeros = np.sum(Q <= 0)
                if n_zeros < self.n_cols:
                    M_r = np.zeros((self.n_rows, self.n_cols-n_zeros))
                    for j in range(self.n_cols-n_zeros):
                        (p, f), rp = self._get_new_col()
                        self.feat_list.append((p, f))
                        self.feat_set.add((p, f))
                        R = self.row_prefs_repr(f)
                        x = np.dot(R, rp)
                        M_r[:,j] = x
                    
                    self.M = np.concatenate((self.M, M_r), axis=1)
                    
            if verbose:
                logging.info("# of kept columns: %d" %(np.sum(Q>0)))
                logging.info("# of unique fetures: %d" %len(set([f for i, (p, f) in enumerate(self.feat_list[:len(Q)]) if Q[i]>0.])))
                logging.info("# of columns: %d\n" %self.M.shape[1])
                
        self.Q = Q