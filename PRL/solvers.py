import numpy as np


class Solver():
    """Abstract class that every solver MUST inherit from."""

    def solve(self, M, n_rows, n_cols):
        """Solves the two.players zero-sum game on the given matrix  game.

        :param  M: matrix game
        :param n_rows: number of rows of M
        :param n_cols: number of columns of M
        :type M: numpy.ndarray
        :type n_rows: int
        :type n_cols: int
        :returns: a tuple containing the strategies of the two players as well as the value of the game
        :rtype: tuple(numpy.ndarray, numpy.ndarray, float)
        """
        pass

    def get_name(self):
        """Returns the name of the solver.

        :returns: the name of the solver
        :rtype: string
        """
        pass

    def set_params(self, params):
        """Sets the parameters of the solver.

        :param params: dictionary containing the parameters and their associated values
        :type params: dict
        """
        pass

    def get_params(self):
        """Returns the parameters setting of the solver.

        :returns: dictionary containing the parameters and their associated values
        :rtype: dict
        """
        pass


class FictitiousPlay(Solver):
    """Fictitious Play (FP) algorithm for two-players zero-sum game.

    FP is an approximated solver, based on the algorithm described in
    ``Iterative solutions of games by fictitious play``, G.W. Brown.ì, in Activity Analysis of Production and Allocation 374–376, 1951."""

    def __init__(self, iterations=1000000):
        """Initializes the Fictitious Play algorithm.

        :param iterations: number of iterations of the algorithm
        """
        self.iterations = iterations

    def __repr__(self):
        return "FictitiousPlay(it=%d)" %self.iterations

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
