import numpy as np
from scipy import sparse
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

LABELS = ['EF_node', 'Spelling_node', 'Naming_node', 'Syntax_node']

# https://scikit-learn.org/dev/auto_examples/cross_decomposition/plot_pcr_vs_pls.html

def do_pcr(X, y):
    """
    Perform principal component regression
    """
    max_rsqr = 0
    y_pred = []
    kmax = 0
    for k in range(1, 6):
        pcr = make_pipeline(StandardScaler(), PCA(n_components=k), LinearRegression())
        pcr.fit(X, y)
        #pca = pcr.named_steps['pca']  # retrieve the PCA step of the pipeline
        rsqr = pcr.score(X, y)
        if rsqr > max_rsqr:
            max_rsqr = rsqr
            y_pred = pcr.predict(X)
            kmax = k
    return {'rsqr':max_rsqr, 'y_pred':y_pred, 'k':kmax}

def do_pls(X, y):
    """
    Perform partial least squares
    """
    max_rsqr = 0
    y_pred = []
    kmax = 0
    for k in range(1, 6):
        pls = PLSRegression(n_components=k)
        pls.fit(X, y)
        rsqr = pls.score(X, y)
        if rsqr > max_rsqr:
            max_rsqr = rsqr
            y_pred = pls.predict(X)
            kmax = k
    return {'rsqr':max_rsqr, 'y_pred':y_pred, 'k':kmax}

def do_regressions_feat(feat):
    """
    Parameters
    ----------
    feat: string
        Type of feature (either PD or WD)
    """
    plt.figure(figsize=(8, 4))
    patients = pd.read_csv("{}_patients.csv".format(feat))
    scores = pd.read_csv("behavioral_scores.csv")
    depvars = scores.columns[5::]
    names = scores['patient'].to_numpy().tolist()
    labels = patients[LABELS].to_numpy()
    labels[np.isnan(labels)] = 0
    # Loop through all combinations of EF_node and others
    blues = scores[scores.columns[1:5]].to_numpy()
    plt.figure(figsize=(12, 6))
    for label in range(1, len(LABELS)):
        # Setup independent variables
        X = []
        for i, name in enumerate(names):
            x1 = blues[i, :].tolist()
            x2 = patients[name].to_numpy()
            x21 = x2[labels[:, label] == 1].tolist()
            x22 = x2[labels[:, 0] == 1].tolist()
            X.append(x1 + x21 + x22)
        X = np.array(X)
        XSum = np.sum(X, 1)
        # Setup dependent variables and do regression
        for var in depvars:
            y = scores[var].to_numpy()
            # Exclude rows with NaNs (missing values) in X or y
            idx = np.arange(y.size)
            idx[np.isnan(y)] = -1
            idx[np.isnan(XSum)] = -1
            idx = idx[idx >= 0]
            Xv = X[idx, :]
            y = y[idx]
            print(feat, LABELS[label], "+ EF_NODE: ", var, Xv.shape[0], "subjects, ", Xv.shape[1], "indepvars")
            
            plt.clf()
            ## Step 1: Do pcr regression
            res = do_pcr(Xv, y)
            plt.subplot(121)
            plt.scatter(res['y_pred'], y)
            plt.legend(["$r^2={:.3f}, k={}$, {} subj {} indvars".format(res['rsqr'], res['k'], Xv.shape[0], Xv.shape[1])])
            plt.xlabel("Predicted {}".format(var))
            plt.ylabel("Actual {}".format(var))
            plt.title("{}, {} + EF_NODE: {}, PCR".format(feat, LABELS[label], var))
            ## Step 3: Do pls regression
            res = do_pls(Xv, y)
            plt.subplot(122)
            plt.scatter(res['y_pred'], y)
            plt.legend(["$r^2={:.3f}, k={}$, {} subj {} indvars".format(res['rsqr'], res['k'], Xv.shape[0], Xv.shape[1])])
            plt.xlabel("Predicted {}".format(var))
            plt.ylabel("Actual {}".format(var))
            plt.title("PLS")


            plt.savefig("{}_{}_{}.svg".format(feat, LABELS[label], var))





do_regressions_feat("WD")
do_regressions_feat("PC")
do_regressions_feat("BC")