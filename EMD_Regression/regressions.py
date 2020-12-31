import numpy as np
from scipy import sparse
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sys import argv

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

def do_monte_carlo_regression(X1, XAll, Xv, y, idx, reg, monte_iters):
    """
    Pick out random sets of independent variables (in addition to the common 4 independent variables)
    to create a regression of the same size, and compute rsquared for many iterations of this
    Parameters
    ----------
    X1: ndarray(N, 4)
        Common independent variables
    XAll: ndarray(N, K)
        All other independent variables
    Xv: ndarray(N2, M)
        Ground truth array of independent variables
    y: ndarray(N2)
        Dependent variables
    idx: ndarray(N2)
        Indices into the original set of patients of which patients
        are included in this regression
    reg: fn: k -> sklearn pipeline
        Regression handle
    monte_iters: int
        Number of monte carlo iterations
    """
    rsqrs = np.zeros(monte_iters)
    M2 = Xv.shape[1] - X1.shape[1]
    X1 = X1[idx, :]
    for i in range(monte_iters):
        if i%10 == 0:
            print("Monte carlo {} of {}".format(i, monte_iters))
        idx2 = np.random.permutation(XAll.shape[1])[0:M2]
        X2 = XAll[idx, :]
        X2 = X2[:, idx2]
        X = np.concatenate((X1, X2), axis=1)
        rsqrs[i] = do_loo_regression(reg, X, y)['rsqr']
    return rsqrs
    

def do_regressions_feat(patients, ifn, var, prefix, fout, monte_iters = 5000, do_plots=True):
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
    fout: file handle
        Handle to file to which to output results in csv format
    monte_iters: int
        Number of monte carlo iterations
    """
    scores = pd.read_csv("behavioral_scores.csv")
    names = scores['patient'].to_numpy().tolist()
    labels = patients[LABELS].to_numpy()
    labels[np.isnan(labels)] = 0

    ## Step 1: Setup independent variables
    ## by looping through all combinations of EF_node and others
    X1 = scores[scores.columns[1:5]].to_numpy()
    # Setup independent variables for the union of the chosen
    # labels, as well as the "orange variables"
    X = np.array([])
    XAll = np.array([])
    for i, name in enumerate(names):
        x1 = X1[i, :].tolist()
        for key in patients.keys():
            # Sometimes the names in the nodes have extra stuff on the end
            if name in key:
                name = key
        if name in patients:
            x2 = patients[name].to_numpy()
            if XAll.size == 0:
                XAll = np.nan*np.ones((len(names), x2.size))
            XAll[i, :] = x2
            x2 = x2[ifn(labels) == 1].tolist()
            x = x1 + x2
            if X.size == 0:
                X = np.nan*np.ones((len(names), len(x)))
            X[i, :] = np.array(x)
        else:
            print("Missing ", name, prefix)
    ## Step 2: Perform regression on the chosen observation
    XSum = np.sum(X, 1)
    # Setup dependent variables and do regression
    y = scores[var].to_numpy()
    # Exclude rows with NaNs (missing values) in X or y
    idx = np.arange(y.size)
    idx[np.isnan(XSum) + np.isnan(y)] = -1
    idx = idx[idx >= 0]
    Xv = X[idx, :]
    y = y[idx]
    print(prefix, var, Xv.shape[0], "subjects, ", Xv.shape[1], "indepvars")
    fout.write("{},{},{},{},{},".format(prefix[0:2], prefix[3::], var, Xv.shape[0], Xv.shape[1]))

    figpath = "Results_Regressions/" + prefix + "_vs_" + var + ".svg"
    ## Do pcr regression
    pcr = lambda k: make_pipeline(StandardScaler(), PCA(n_components=k), LinearRegression())
    res = do_loo_regression(pcr, Xv, y)
    rsqrs = do_monte_carlo_regression(X1, XAll, Xv, y, idx, pcr, monte_iters)
    p = np.sum(rsqrs >= res['rsqr'])/monte_iters
    fout.write("{:.3f},{},{:.3f},".format(res['rsqr'], res['k'], p))
    if do_plots:
        plt.clf()
        plt.subplot(221)
        plt.scatter(res['y_pred'], y)
        plt.legend(["$r^2={:.3f}, k={}$".format(res['rsqr'], res['k'])])
        plt.xlabel("Predicted {}".format(var))
        plt.ylabel("Actual {}".format(var))
        plt.axis("equal")
        plt.title("PCR")
        plt.subplot(223)
        h = plt.hist(rsqrs)
        plt.stem([res['rsqr']], [np.max(h[0])], use_line_collection=True)
        plt.xlabel("$R^2$")
        plt.ylabel("Counts")
        plt.title("PCR Monte Carlo (p = {:.3f})".format(p))


    ## Do pls regression
    pls = lambda k: make_pipeline(StandardScaler(), PLSRegression(n_components=k))
    res = do_loo_regression(pls, Xv, y)
    rsqrs = do_monte_carlo_regression(X1, XAll, Xv, y, idx, pcr, monte_iters)
    p = np.sum(rsqrs >= res['rsqr'])/monte_iters
    fout.write("{:.3f},{},{:.3f}\n".format(res['rsqr'], res['k'], p))
    if do_plots:
        plt.subplot(222)
        plt.scatter(res['y_pred'], y)
        plt.legend(["$r^2={:.3f}, k={}$".format(res['rsqr'], res['k'])])
        plt.xlabel("Predicted {}".format(var))
        plt.ylabel("Actual {}".format(var))
        plt.axis("equal")
        plt.title("PLS")
        plt.suptitle("{} => {} : {} subj {} indvars".format(prefix, var, Xv.shape[0], Xv.shape[1]))
        plt.subplot(224)
        h = plt.hist(rsqrs)
        plt.stem([res['rsqr']], [np.max(h[0])], use_line_collection=True)
        plt.xlabel("$R^2$")
        plt.ylabel("Counts")
        plt.title("PLS Monte Carlo (p = {:.3f})".format(p))
        plt.savefig(figpath)

def do_regressions(batch):
    plt.figure(figsize=(12, 12))
    fout = open("Results_Regressions/results{}.csv".format(batch), "w")
    fout.write("{},{},{},{},{},{},{},{},{},{},{}\n".format("Graph Statistic", "Independent Variables", "Dependent Variable", "Num Subjects", "Num Indep Variables", "PCR R2", "PCR K", "PCR p-value", "PLS R2", "PLS K", "PLS p-value"))
    for stat in ["PC", "BC", "WD"]:
        i = 0
        patients = pd.read_csv("{}_patients.csv".format(stat))
        ## Group 1: Regressions on language nodes to their corresponding tests
        for (var, label) in [("SpellingTrain_prepostPMG", "Spelling_node"), ("SpellingGen_prepostPMG", "Spelling_node"), ("NamingTrain_prepost", "Naming_node"), ("NamingGen_prepost", "Naming_node"), ("SentprocTrain_prepost", "Syntax_node"), ("SentprocGen_prepost", "Syntax_node")]:
            if i == batch:
                # First do just node type by itself
                ifn = lambda labels: labels[:, LIDX[label]]
                prefix = stat + "_" + label
                do_regressions_feat(patients, ifn, var, prefix, fout)
                for executive in ["EFCCN", "MD"]:
                    # Now use the node type and EFCNN
                    ifn = lambda labels: labels[:, LIDX[label]] + labels[:, LIDX["{}_node".format(executive)]] > 0
                    prefix = stat + "_"+executive+"+" + label
                    do_regressions_feat(patients, ifn, var, prefix, fout)
            i += 1
        """
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
        """
    fout.close()

if __name__ == '__main__':
    do_regressions(int(argv[1]))