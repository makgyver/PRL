import numpy as np
import random

class GenP():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n = X.shape[0]
        self.labelset = set(np.unique(y))

    def get_random_pref(self):
        pass

    def get_pref_value(self,p):
        (ipos, ypos), (ineg, yneg) = p
        return ((self.X[ipos], ypos), (self.X[ineg], yneg))

    def get_all_prefs(self):
        pass
        
    #def getUniqueVals(self):
    #    return [list(set(self.X[:,i])) for i in range(self.X.shape[1])]


class GenMicroP(GenP):

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

class GenMacroP(GenP):

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
