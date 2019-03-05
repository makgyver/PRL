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
        self.col_list = []
        self.col_set = set()

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


    def pref_repr(self, q, f):
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

    def compute_column(self, rq, f):
        """Computes the column representation of the preference q w.r.t. the feature f.

        :param rq: the new preference representation
        :param f: the feature identifier
        :type rq: numpy.ndarray
        :type f: int, tuple
        :returns: the representation of the column
        :rtype: numpy.ndarray
        """
        R = np.zeros((self.n_rows, self.dim))
        for i, p in enumerate(self.pref_list):
            (x_p, y_p), (x_n, y_n) = self.gen_pref.get_pref_value(p)
            R[i, y_p] = +self.gen_feat.get_feat_value(f, x_p)
            R[i, y_n] = -self.gen_feat.get_feat_value(f, x_n)

        return np.dot(R, rq)

    def compute_entry(self, p, q, f):
        return np.dot(self.pref_repr(p, f), self.pref_repr(q, f))

    def _get_new_col(self):
        """Internal method that randomly pick a new column in such a way that its representation is not null and it is not already in the game matrix.

        :returns: a not null representation of a random picked preference-feature pair
        :rtype: tuple
        """
        (p, f) = self.get_random_pair()
        rp = self.pref_repr(p, f)
        while (p, f) in self.col_set or not rp.any():
            (p, f) = self.get_random_pair()
            rp = self.pref_repr(p, f)
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
            self.col_list.append((p, f))
            self.col_set.add((p, f))
            self.M[:,j] = self.compute_column(rp, f)

        #iterative updates
        for t in range(iterations):
            if verbose: logging.info("PRL iteration %d/%d" %(t+1, iterations))
            (P, Q, V) = self.solver.solve(self.M, self.n_rows, self.n_cols)
            if verbose: logging.info("Value of the game (current margin): %.6f" %V)
            if (t+1 < iterations):
                for j in range(self.n_cols):
                    if Q[j] <= 0:
                        (p, f), rp = self._get_new_col()
                        self.col_set.remove(self.col_list[j])
                        self.col_list[j] = (p, f)
                        self.col_set.add((p, f))
                        self.M[:,j] = self.compute_column(rp, f)

            if verbose:
                logging.info("## of kept columns: %d" %(np.sum(Q>0)))
                logging.info("# of unique features: %d\n" %len(set([f for i, (p, f) in enumerate(self.col_list) if Q[i]>0.])))
        self.Q = Q


    def get_best_features(self, k=10):
        """Returns the k best features sorted by their weights.

        :param k: the number of most relevant features to retrieve
        :type k: int
        :returns: the k best features (along with their weights) sorted by their weights
        :rtype: list of tuples
        """
        list_w_feat = [(self.Q[i], pf) for i, pf in enumerate(self.col_list) if self.Q[i] > 0]
        list_w_feat.sort(reverse=True)

        return [(pf, q) for (q, pf) in list_w_feat[:k]]





class PRL_ext(PRL):

    def fit(self, iterations=1000, verbose=False):

        if verbose:
            logging.info("Starting training of %s" %self)
            logging.info("Matrix game initialization...")

        for j in range(self.n_cols):
            (p, f), rp = self._get_new_col()
            self.col_list.append((p, f))
            self.col_set.add((p, f))
            self.M[:,j] = self.compute_column(rp, f)

        #iterative updates
        for t in range(iterations):
            if verbose: logging.info("PRL_ext iteration %d/%d" %(t+1, iterations))
            (P, Q, V) = self.solver.solve(self.M, self.n_rows, self.M.shape[1])
            if verbose: logging.info("Value of the game (current margin): %.6f" %V)
            if (t+1 < iterations):
                for j in range(self.M.shape[1]):
                    if Q[j] <= 0:
                        (p, f), rp = self._get_new_col()
                        self.col_set.remove(self.col_list[j])
                        self.col_list[j] = (p, f)
                        self.col_set.add((p, f))
                        self.M[:,j] = self.compute_column(rp, f)

                n_zeros = np.sum(Q <= 0)
                if n_zeros < self.n_cols:
                    M_r = np.zeros((self.n_rows, self.n_cols-n_zeros))
                    for j in range(self.n_cols-n_zeros):
                        (p, f), rp = self._get_new_col()
                        self.col_list.append((p, f))
                        self.col_set.add((p, f))
                        self.M[:,j] = self.compute_column(rp, f)

                    self.M = np.concatenate((self.M, M_r), axis=1)

            if verbose:
                logging.info("# of kept columns: %d" %(np.sum(Q>0)))
                logging.info("# of unique features: %d" %len(set([f for i, (p, f) in enumerate(self.col_list[:len(Q)]) if Q[i]>0.])))
                logging.info("# of columns: %d\n" %self.M.shape[1])

        self.Q = Q


class KPRL:
    """This class implements the Kernelized Preference and Rule Learning (KPRL) algorithm."""

    def __init__(self, gen_pref, gen_kernel, dim, n_cols, solver):
        """Initializes all the useful structures.

        :param gen_pref: the preference generator. See <:genP.GenMacroP> and <:genP.GenMicroP>
        :param gen_kernel: the kernel generator
        :param dim: number of possible labels
        :param n_cols: number of columns of the matrix sub-game
        :param solver: game solver. See for example <:solvers.FictitiousPlay>
        :type gen_pref: object of class which inherits from <:genP.GenP>, e.g., GenMacroP
        :type gen_kernel: object of class which inherits from <:genK.GenK>, e.g., GenHPK
        :type dim: int
        :type n_cols: int
        :type solver: object of class which inherits from <:solvers.Solver>
        """
        self.gen_pref = gen_pref
        self.gen_kernel = gen_kernel
        self.n_cols = n_cols
        self.dim = dim
        self.solver = solver

        self.pref_list = self.gen_pref.get_all_prefs()
        self.n_rows = len(self.pref_list)
        self.col_list = []
        self.col_set = set()

        self.M = np.zeros((self.n_rows, self.n_cols))
        self.Q = None

    def __repr__(self):
        """Returns a string representation of the KPRL object.

        :returns: return a string representation of the KPRL object
        :rtype: string
        """
        return "KPRL(gen_pref=%s, gen_kernel=%s, n_rows=%d, n_cols=%d, solver=%s)"\
            %(self.gen_pref, self.gen_kernel, self.n_rows, self.n_cols, self.solver)

    def get_random_pair(self):
        """Returns a new random columns composed by a preference-kernel pair.

        :returns:  a new random columns composed by a preference-kernel pair
        :rtype: tuple(preference, kernel)
        """
        return (self.gen_pref.get_random_pref(), self.gen_kernel.get_random_kernel())

    def compute_column(self, q, k):
        """Computes the representation of the preference q w.r.t. the kernel k.

        :param q: the new preference
        :param k: the kernel function identifier
        :type q: tuple
        :type k: int, tuple
        :returns: the representation of the column
        :rtype: numpy.ndarray
        """
        p_col = self.gen_pref.get_pref_value(q)
        k_fun = self.gen_kernel.get_kernel_function(k)
        R = np.zeros((self.n_rows, 1))
        for i, r in enumerate(self.pref_list):
            p_row = self.gen_pref.get_pref_value(r)
            R[i, 0] = k_fun(p_col, p_row)

        return R

    def compute_entry(self, p, q, k):
        return self.gen_kernel.get_kernel_function(k)(p, q)

    def _get_new_col(self):
        """Internal method that randomly pick a new column in such a way that its representation is not null and it is not already in the game matrix.

        :returns: a not null representation of a random picked preference-kernel pair
        :rtype: tuple
        """
        (p, k) = self.get_random_pair()
        while (p, k) in self.col_set:
            (p, k) = self.get_random_pair()
        return (p, k)


    def fit(self, iterations=1000, verbose=False):
        """Trains the KPRL method.

        :param iterations: the number of iterations of KPRL
        :param verbose: whether the output is verbose or not
        :type iterations: int
        :type verbose: bool
        """
        if verbose:
            logging.info("Starting training of %s" %self)
            logging.info("Matrix game initialization...")

        for j in range(self.n_cols):
            (p, k) = self._get_new_col()
            self.col_list.append((p, k))
            self.col_set.add((p, k))
            self.M[:,j] = self.compute_column(p, k)

        #iterative updates
        for t in range(iterations):
            if verbose: logging.info("KPRL iteration %d/%d" %(t+1, iterations))
            (P, Q, V) = self.solver.solve(self.M, self.n_rows, self.n_cols)
            if verbose: logging.info("Value of the game (current margin): %.6f" %V)
            if (t+1 < iterations):
                for j in range(self.n_cols):
                    if Q[j] <= 0:
                        (p, k) = self._get_new_col()
                        self.col_set.remove(self.col_list[j])
                        self.col_list[j] = (p, k)
                        self.col_set.add((p, k))
                        self.M[:,j] = self.compute_column(p, k)
            if verbose:
                logging.info("# of kept columns: %d" %(np.sum(Q>0)))

        self.Q = Q
