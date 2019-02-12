import numpy as np


class Solver():

    def solve(self, M, n_rows, n_cols):
        pass

    def get_name(self):
        pass

    def set_params(self, params):
        pass

    def get_params(self):
        pass


class FictitiousPlay(Solver):

    def __init__(self, iterations=1000000):
        self.iterations = iterations

    def get_name(self):
        return "FP"

    def get_params(self):
        return {"iterations" : self.iterations}

    def set_params(self, params):
        if "iterations" in params:
            self.iterations = params["iterations"]

    def solve(self, M, n_rows, n_cols):
        P = np.zeros(n_rows)
        Q = np.zeros(n_cols)
        V = 0.

        Sp = np.zeros(n_rows)

        i_min = np.random.randint(n_rows)
        P[i_min] = 1.0

        Sq = M[i_min,:].copy()

        for t in xrange(self.iterations):
            j_max = np.argmax(Sq / (t+1))
            Q[j_max] += 1.
            Sp += M[:,j_max]
            i_min = np.argmax(-Sp / (t+1))
            P[i_min] += 1.
            Sq += M[i_min,:]

        P /= np.sum(P)
        Q /= np.sum(Q)
        return (P, Q, np.dot(P.T, np.dot(M, Q)))
