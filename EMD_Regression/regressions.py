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

LABELS = ['EF_node', 'Spelling_node', 'Naming_node', 'Syntax_node']

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

def do_regressions_feat(feat):
    """
    Parameters
    ----------
    feat: string
        Type of feature (either PD, WD, or BC)
    """
    plt.figure(figsize=(8, 4))
    patients = pd.read_csv("{}_patients.csv".format(feat))
    scores = pd.read_csv("behavioral_scores.csv")
    depvars = scores.columns[5::]
    names = scores['patient'].to_numpy().tolist()
    labels = patients[LABELS].to_numpy()
    labels[np.isnan(labels)] = 0

    ## Step 1: Setup independent variables
    ## by looping through all combinations of EF_node and others
    blues = scores[scores.columns[1:5]].to_numpy()
    plt.figure(figsize=(12, 6))
    Xs = []
    all_labels = []
    # Setup independent variables for each combination of a label
    # and EF_node, as well as the "blue variables"
    for label in range(1, len(LABELS)):
        X = []
        for i, name in enumerate(names):
            x1 = blues[i, :].tolist()
            x2 = patients[name].to_numpy()
            x21 = x2[labels[:, label] == 1].tolist()
            x22 = x2[labels[:, 0] == 1].tolist()
            X.append(x1 + x21 + x22)
        X = np.array(X)
        Xs.append(X)
        all_labels.append(LABELS[label])
    # Setup the union of all independent variables
    X = []
    for i, name in enumerate(names):
        x1 = blues[i, :].tolist()
        x2 = patients[name].to_numpy().tolist()
        X.append(x1 + x2)
    X = np.array(X)
    Xs.append(X)
    all_labels.append("ALL")

    ## Step 2: Perform regression on all observations with each
    ## set of independent variables
    for label, X in zip(all_labels, Xs):
        XSum = np.sum(X, 1)
        # Setup dependent variables and do regression
        for var in depvars:
            figpath = "{}_{}_{}.svg".format(feat, label, var)
            if os.path.exists(figpath):
                continue
            y = scores[var].to_numpy()
            # Exclude rows with NaNs (missing values) in X or y
            idx = np.arange(y.size)
            idx[np.isnan(y)] = -1
            idx[np.isnan(XSum)] = -1
            idx = idx[idx >= 0]
            Xv = X[idx, :]
            y = y[idx]
            print(feat, label, var, Xv.shape[0], "subjects, ", Xv.shape[1], "indepvars")
            
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
            pls = lambda k: PLSRegression(n_components=k)
            res = do_loo_regression(pls, Xv, y)
            plt.subplot(122)
            plt.scatter(res['y_pred'], y)
            plt.legend(["$r^2={:.3f}, k={}$".format(res['rsqr'], res['k'])])
            plt.xlabel("Predicted {}".format(var))
            plt.ylabel("Actual {}".format(var))
            plt.axis("equal")
            plt.title("PLS")

            varnames = "{} + EF_NODE".format(label)
            if label == "ALL":
                varnames = "ALL"
            plt.suptitle("{}, {} : {}, {} subj {} indvars".format(feat, varnames, var, Xv.shape[0], Xv.shape[1]))

            plt.savefig(figpath)

do_regressions_feat("WD")
do_regressions_feat("PC")
do_regressions_feat("BC")