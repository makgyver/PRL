import numpy as np

def confusion_matrix(prl, gen_pref_test):
    y_pred = prl.predict(gen_pref_test)
    y = gen_pref_test.y
    conf_mat = np.zeros((prl.dim, prl.dim), dtype=int)
    for i, y_max in enumerate(y_pred):
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
