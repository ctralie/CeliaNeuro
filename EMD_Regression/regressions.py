import numpy as np
from scipy import sparse
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from ranks import LABELS, LIDX

# https://scikit-learn.org/dev/auto_examples/cross_decomposition/plot_pcr_vs_pls.html

def do_loo_regression(reg, X, y):
    """
    Do a leave-one-out regression
    Parameters
    ----------
    reg: fn: k -> sklearn pipeline
        Regression handle
    X: ndarray(N, d)
        Independent variables
    y: ndarray(N)
        Observations
    """
    N = X.shape[0]
    y_pred = np.zeros(N)
    rsqr_max = 0
    k_max = 0
    for k in range(1, min(X.shape[0], X.shape[1]+1)):
        reg_k = reg(k)
        y_pred_k = np.zeros(N)
        for i in range(N):
            idx = np.arange(N)
            idx[i] = -1
            idx = idx[idx >= 0]
            Xi = X[idx, :]
            yi = y[idx]
            reg_k.fit(Xi, yi)
            xi = X[i, :]
            y_pred_k[i] = reg_k.predict(xi[None, :])
        rsqr = r2_score(y, y_pred_k)
        if rsqr > rsqr_max:
            rsqr_max = rsqr
            k_max = k
            y_pred = y_pred_k
    return {'rsqr':rsqr_max, 'y_pred':y_pred, 'k':k_max}

def do_regressions_feat(patients, ifn, var, prefix):
    """
    Parameters
    ----------
    patients: pandas dataframe
        Data frame holding node variables for the patients
    ifn: function labels -> ndarray(N)
        Function for extracting labels for the nodes which are used
        as independent variables
    var: string
        Dependent variable on which to regress
    prefix: string
        Prefix of file to which to save the figure
    """
    scores = pd.read_csv("behavioral_scores.csv")
    names = scores['Patient'].to_numpy().tolist()
    labels = patients[LABELS].to_numpy()
    labels[np.isnan(labels)] = 0

    ## Step 1: Setup independent variables
    ## by looping through all combinations of EF_node and others
    orange = scores[scores.columns[1:6]].to_numpy()
    # Setup independent variables for the union of the chosen
    # labels, as well as the "orange variables"
    X = []
    d = 0
    for i, name in enumerate(names):
        x1 = orange[i, :].tolist()
        for key in patients.keys():
            # Sometimes the names in the nodes have extra stuff on the end
            if name in key:
                name = key
        if name in patients:
            x2 = patients[name].to_numpy()
            x2 = x2[ifn(labels) == 1].tolist()
            X.append(x1 + x2)
            d = len(X[-1])
        else:
            print("Missing ", name, prefix)
            X.append([])
    for i in range(len(X)):
        if len(X[i]) == 0:
            X[i] = np.nan*np.ones(d)
    X = np.array(X)

    ## Step 2: Perform regression on the chosen observation
    XSum = np.sum(X, 1)
    # Setup dependent variables and do regression
    figpath = "Results_Regressions/" + prefix + "_vs_" + var + ".svg"
    if os.path.exists(figpath):
        print("Skipping", figpath)
    else:
        y = scores[var].to_numpy()
        # Exclude rows with NaNs (missing values) in X or y
        idx = np.arange(y.size)
        idx[np.isnan(y)] = -1
        idx[np.isnan(XSum)] = -1
        idx = idx[idx >= 0]
        Xv = X[idx, :]
        y = y[idx]
        print(prefix, var, Xv.shape[0], "subjects, ", Xv.shape[1], "indepvars")
        
        plt.clf()
        ## Do pcr regression
        pcr = lambda k: make_pipeline(StandardScaler(), PCA(n_components=k), LinearRegression())
        res = do_loo_regression(pcr, Xv, y)
        plt.subplot(121)
        plt.scatter(res['y_pred'], y)
        plt.legend(["$r^2={:.3f}, k={}$".format(res['rsqr'], res['k'])])
        plt.xlabel("Predicted {}".format(var))
        plt.ylabel("Actual {}".format(var))
        plt.axis("equal")
        plt.title("PCR")
        ## Do pls regression
        pls = lambda k: make_pipeline(StandardScaler(), PLSRegression(n_components=k))
        res = do_loo_regression(pls, Xv, y)
        plt.subplot(122)
        plt.scatter(res['y_pred'], y)
        plt.legend(["$r^2={:.3f}, k={}$".format(res['rsqr'], res['k'])])
        plt.xlabel("Predicted {}".format(var))
        plt.ylabel("Actual {}".format(var))
        plt.axis("equal")
        plt.title("PLS")
        plt.suptitle("{} => {} : {} subj {} indvars".format(prefix, var, Xv.shape[0], Xv.shape[1]))
        plt.savefig(figpath)

def do_regressions():
    plt.figure(figsize=(12, 6))
    for stat in ["WD", "PC", "BC"]:
        patients = pd.read_csv("{}_patients.csv".format(stat))
        ## Group 1: Regressions on language nodes to their corresponding tests
        for (var, label) in [("SpellingPALPA40", "Spelling_node"), ("NamingNNB", "Naming_node"), ("SentprocNAVS", "Syntax_node")]:
            # First do just node type by itself
            ifn = lambda labels: labels[:, LIDX[label]]
            prefix = stat + "_" + label
            do_regressions_feat(patients, ifn, var, prefix)
            # Now use the node type and EFCNN
            ifn = lambda labels: labels[:, LIDX[label]] + labels[:, LIDX["EFCCN_node"]] > 0
            prefix = stat + "_EFCCN+" + label
            do_regressions_feat(patients, ifn, var, prefix)
        ## Group 2: EFComposite_BDS
        ifn = lambda labels: labels[:, LIDX["EFCCN_node"]]
        prefix = stat + "_EFCCN"
        do_regressions_feat(patients, ifn, "EFComposite_BDSnorm", prefix)
        ## Group 3: TxEffect groups
        for typ in ["patients", "prepost"]:
            patients = pd.read_csv("{}_{}.csv".format(stat, typ))
            # Spelling/naming/syntax/efccn vs TxEffect_train_z and TxEffect_gen_z
            ifn = lambda labels: np.sum(labels[:, [LIDX[s+"_node"] for s in ["Spelling", "Naming", "Syntax", "EFCCN"]]], 1) > 0
            prefix = stat + "_" + typ + "_Spelling+Naming+Syntax+EFCCN"
            do_regressions_feat(patients, ifn, "TxEffect_train_z", prefix)
            do_regressions_feat(patients, ifn, "TxEffect_gen_z", prefix)
            # Spelling/naming/syntax vs TxEffect_train_z and TxEffect_gen_z
            ifn = lambda labels: np.sum(labels[:, [LIDX[s+"_node"] for s in ["Spelling", "Naming", "Syntax"]]], 1) > 0
            prefix = stat + "_" + typ + "_Spelling+Naming+Syntax"
            do_regressions_feat(patients, ifn, "TxEffect_train_z", prefix)
            do_regressions_feat(patients, ifn, "TxEffect_gen_z", prefix)


if __name__ == '__main__':
    do_regressions()