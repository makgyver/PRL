import numpy as np
import random

class GenP(object):
    """Abstract class which representes a generic preference generator.
    Every specific generator MUST inherit from this class."""

    def __init__(self, X, y):
        """Initializes the preference generator.

        :param X: training instances
        :param y: training labels associated to the instances
        :type X: bidimensional numpy.ndarray
        :type y: numpy.ndarray
        """
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.labelset = set(np.unique(y))

    def get_random_pref(self):
        """Returns the identifier of random preference.

        :returns: a random preference
        :rtype: tuple
        """
        pass

    def get_pref_value(self, p):
        """Retruns the concrete instantiation of a prefernce identifier.

        :param p: preference identifier
        :type p: tuple
        :returns: a preference
        :rtype: tuple(tuple(numpy.ndarray, int), tuple(numpy.ndarray, int))
        """
        (ipos, ypos), (ineg, yneg) = p
        return ((self.X[ipos], ypos), (self.X[ineg], yneg))

    def get_all_prefs(self):
        """Returns the list of all possibile preferences.

        :returns: the list of all possible preferences
        :rtype: list
        """
        pass


class GenMicroP(GenP):
    """Micro preference generator. A micro preference describes preferences like
    (x_i, y_i) is preferred to (x_j, y_j), where (x_i, y_i) in X x Y, while  (x_j, y_j)
    not in X x Y. This kind of preferences are suitable for instance ranking tasks."""

    def __init__(self, X, y):
        GenP.__init__(self, X, y)

    def get_random_pref(self):
        ipos = random.randint(0, self.n-1)
        ypos = self.y[ipos]
        ineg = random.randint(0, self.n-1)
        yneg = random.choice(list(self.labelset - set([self.y[ineg]])))
        return ((ipos, ypos), (ineg, yneg))

    def get_all_prefs(self):
        lp = []
        for i in range(self.n):
            yp = self.y[i]
            for j in range(self.n):
                ypj = self.y[j]
                for yn in (self.labelset - set([ypj])):
                    lp.append(((i, yp), (j, yn)))
        return lp

    def __repr__(self):
        return "Macro preference generator"

class GenMacroP(GenP):
    """Macro preference generator. A macro preference describes preferences like
    y_i is preferred to y_j for the instance x_i, where (x_i, y_i) in X x Y, while (x_i, y_j)
    not in X x Y. This kind of preferences are suitable for label ranking tasks."""

    def __init__(self, X, y):
        GenP.__init__(self, X, y)

    def get_random_pref(self):
        ipos = random.randint(0,self.n-1)
        ypos = self.y[ipos]
        yneg = random.choice(list(self.labelset-set([self.y[ipos]])))
        return ((ipos,ypos),(ipos,yneg))

    def get_all_prefs(self):
        lp = []
        for i in range(self.n):
            yp = self.y[i]
            for yn in (self.labelset - set([yp])):
                lp.append(((i, yp), (i, yn)))
        return lp

    def __repr__(self):
        return "Micro preference generator"


class GenIP(GenP):
    """Instance-based preference generator. These are actually degenerate preferences that are
    simple instances."""

    def __init__(self, X):
        self.X = X
        self.n = X.shape[0]

    def get_random_pref(self):
        return random.randint(0, self.n-1)

    def get_pref_value(self, p):
        return self.X[p]

    def get_all_prefs(self):
        return range(self.n)

    def __repr__(self):
        return "Instance-based preference generator"
