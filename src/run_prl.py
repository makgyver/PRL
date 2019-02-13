import sys
import json
import numpy as np
from optparse import OptionParser

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from prl import PRL
from genF import *
from genP import *
from evaluation import *

import logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(filename)s - %(message)s",
                    datefmt='%H:%M:%S-%d%m%y')

def manage_options():
    parser = OptionParser(usage="usage: %prog [options] dataset_file", version="%prog 1.0")

    parser.add_option("-s", "--seed",           dest="seed",            default=42,      help="Pseudo-random seed for replicability", type="int")
    parser.add_option("-t", "--test_size",      dest="test_size",       default=.3,      help="Test set size in percentage [0,1]")
    parser.add_option("-c", "--config_file",    dest="config_file",     default="../config/config.json", help="Configuration file")
    parser.add_option("-v", "--verbose",        dest="verbose",         default=False,   help="Verbose output", action="store_true")

    (options, args) = parser.parse_args()
    if len(args) == 0:
        parser.error("Wrong arguments")

    out_dict = vars(options)
    out_dict["dataset"] = args[0]
    return out_dict


#INPUT
options = manage_options()
logging.info("Options: %s" %options)
#

#LOADING DATA
X, y = load_svmlight_file(options["dataset"])
X, y = X.toarray(), y.astype(int)

# maps labels into the range 0,..,m-1
unique_y = np.unique(y)
dim = len(unique_y)
map_y = dict(zip(unique_y, range(len(unique_y))))
y = np.array([map_y[i] for i in y])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=options["test_size"], random_state=options["seed"])

scaler = MinMaxScaler()
scaler.fit(Xtr)
Xtr = scaler.transform(Xtr)
Xte = scaler.transform(Xte)
#

#CONFIGURATION FILE
with open(options['config_file'], "r") as f:
    data = json.load(f)
    logging.info("Configuration: %s" %data)

    genf_class = getattr(__import__("genF"), data['feat_gen'])
    gen_feat = genf_class(Xtr, *data['feat_gen_params'])

    if data["pref_generator"] == "micro":
        gen_pref_training = GenMicroP(Xtr, ytr)
        gen_pref_test = GenMicroP(Xte, yte)
    else: #if not micro
        gen_pref_training = GenMacroP(Xtr, ytr)
        gen_pref_test = GenMacroP(Xte, yte)

    budget = data["columns_budget"]
    iterations = data["iterations"]

    solver_class = getattr(__import__("solvers"), data['solver'])
    solver = solver_class(*data['solver_params'])
#

#PRL
prl = PRL(gen_pref_training, gen_feat, dim, budget, solver)
prl.fit(iterations, options["verbose"])
#

print prl.get_best_features(k=10)

#EVALUATION
acc, conf = accuracy(prl, gen_pref_test)
bacc, _ = balanced_accuracy(prl, gen_pref_test, conf)

logging.info("Accuracy: %.2f" %acc)
logging.info("Balanced accuracy: %.2f" %bacc)
logging.info("Confusion matrix:\n%s" %conf)
#

logging.shutdown()
