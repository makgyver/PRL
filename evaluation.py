import numpy as np

def confusion_matrix(prl, gen_pref_test):
    X = gen_pref_test.X
    y = gen_pref_test.y

    conf_mat = np.zeros((prl.dim, prl.dim), dtype=int)

    N = gen_pref_test.n
    for i in range(N):
        x = X[i,:]
        sco = [0.0 for c in range(prl.dim)]
        for j, (p, f) in enumerate(prl.feat_list):
            if prl.Q[j] > 0.0:
                for c in range(prl.dim):
                    if p[0][1] == c:
                        xp = prl.gen_pref.get_pref_value(p)[0][0]
                        sco[c] += prl.Q[j]*prl.gen_feat.get_feat_value(f, xp)*prl.gen_feat.get_feat_value(f, x)
                    if p[1][1] == c:
                        xn = prl.gen_pref.get_pref_value(p)[1][0]
                        sco[c] -= prl.Q[j]*prl.gen_feat.get_feat_value(f, xn)*prl.gen_feat.get_feat_value(f, x)
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
