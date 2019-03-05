import numpy as np

def confusion_matrix(prl, gen_pref_test):
    X = gen_pref_test.X
    y = gen_pref_test.y

    conf_mat = np.zeros((prl.dim, prl.dim), dtype=int)

    N = gen_pref_test.n
    for i in range(N):
        x = X[i,:]
        sco = [0.0 for c in range(prl.dim)]
        for j, (p, f) in enumerate(prl.col_list):
            if prl.Q[j] > 0.0:
                for c in range(prl.dim):
                    sco[c] += prl.Q[j]*prl.compute_entry(, ((x, c), (-x, c)), f)

        y_max = np.argmax(sco)
        conf_mat[y[i], y_max] += 1

    return conf_mat

def accuracy(prl, gen_pref_test, conf_matrix=None):
    if type(conf_matrix) != np.ndarray:
        conf_matrix = confusion_matrix(prl, gen_pref_test)

    acc = sum([float(conf_matrix[y,y]) for y in range(prl.dim)]) / sum(sum(conf_matrix))
    return acc, conf_matrix


def balanced_accuracy(prl, gen_pref_test, conf_matrix=None):
    if type(conf_matrix) != np.ndarray:
        conf_matrix = confusion_matrix(prl, gen_pref_test)

    bacc = 0.0
    for y in range(prl.dim):
        bacc += float(conf_matrix[y,y]) / sum(conf_matrix[y,:])
    bacc /= prl.dim

    return bacc, conf_matrix
