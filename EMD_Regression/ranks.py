import numpy as np
from scipy import sparse
import scipy.io as sio
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from EMD import *

LABELS = ['EFCCN_node', 'Spelling_node', 'Naming_node', 'Syntax_node']
LIDX = {LABELS[i]:i for i in range(len(LABELS))}
PATIENT_END = -6

def get_EMD(idx, TInd, FInd, plot_name = ""):
    """
    Get the earth-movers distances between two ranks
    Parameters
    ----------
    idx: ndarray(N)
        Rank of each node
    TInd: ndarray(N)
        An array which is 1 if it's part of the true type
    FInd: ndarray(N)
        An array which is 1 if it's part of the false type
    plot_name: string
        If non-empty, plot and save to a file with this name
    """
    # Nodes that are part of this type
    T = np.zeros_like(idx)
    T[idx[TInd == 1]] = 1
    T = T/np.sum(T)
    # Rest of nodes
    F = np.zeros_like(idx)
    F[idx[FInd == 1]] = 1 
    F = F/np.sum(F)
    cT = np.cumsum(T) # True CDF
    cF = np.cumsum(F) # False CDF
    dist = np.sum(np.abs(cT-cF))
    if len(plot_name) > 0:
        plt.clf()
        plt.subplot(211)
        plt.stem(T, use_line_collection=True)
        plt.title("%s, %.3g"%(plot_name, dist))
        plt.subplot(212)
        plt.stem(F, use_line_collection=True)
        plt.savefig(plot_name, bbox_inches='tight')
    return dist

def do_permtest(df, tfn, ffn, n_perms, prefix, title):
    """
    Do the analyses on a particular dataframe
    Parameters
    ----------
    df: pandas dataframe
        The dataframe
    tfn: function labels -> ndarray(N)
        Function for extracting labels for the true class
    ffn: function labels -> ndarray(N)
        Function for extracting labels for the false class
    n_perms: int
        Number of random permutations to use as the null hypothesis
    prefix: string
        prefix for filenames
    title: string
        Title of plots
    Return
    ------
    A dictionary of all of the results, by label
    """
    names = df.columns[1:PATIENT_END]
    labels = df[LABELS].to_numpy()
    labels[np.isnan(labels)] = 0
    N = labels.shape[0]
    TInd = tfn(labels)
    FInd = ffn(labels)
    ## Step 1: Scores on actual data
    emds = []
    mean_ranks = []
    for name in names:
        # Get ranks
        idx = np.zeros(N, dtype=int)
        idx[np.argsort(-df[name].to_numpy())] = np.arange(N)
        emd = get_EMD(idx, TInd, FInd)#, "{}_{}.png".format(label, name))
        emds.append(emd)
        mean_ranks.append(np.mean(idx[TInd == 1]))
    emds = np.array(emds)
    mean_ranks = np.array(mean_ranks)
    ## Step 2: Scores on random permutations
    emds_null = np.zeros(n_perms)
    mean_ranks_null = np.zeros(n_perms)
    np.random.seed(0)
    for p in range(n_perms):
        idx = np.random.permutation(N)
        emd = get_EMD(idx, TInd, FInd)
        emds_null[p] = emd
        mean_ranks_null[p] = np.mean(idx[TInd == 1])
    nauroc = 2*(getAUROC(emds, emds_null)['auroc']-0.5)
    plt.clf()
    sns.distplot(emds, kde=True, norm_hist=True, label="True EMDs")
    sns.distplot(emds_null, kde=True, norm_hist=True, label="Monte Carlo EMDs")
    plt.legend()
    plt.title("EMDs of {}, NAUROC = {:.3f}".format(title, nauroc))
    plt.xlabel("Earth Movers Distance")
    plt.ylabel("Density")
    plt.savefig("EMD_{}.svg".format(prefix), bbox_inches='tight')

    plt.clf()
    nauroc = 2*(getAUROC(mean_ranks, mean_ranks_null)['auroc']-0.5)
    sns.distplot(mean_ranks, kde=True, norm_hist=True, label="True Mean Ranks")
    sns.distplot(mean_ranks_null, kde=True, norm_hist=True, label="Monte Carlo Mean Ranks")
    plt.title("Mean Ranks of {}, NAUROC = {:.3f}".format(title, nauroc))
    plt.legend()
    plt.xlabel("Mean Rank")
    plt.ylabel("Density")
    plt.savefig("MR_{}.svg".format(prefix), bbox_inches='tight')
    return {'emds':emds, 'emds_null':emds_null, 'mean_ranks':mean_ranks, 'mean_ranks_null':mean_ranks_null}

def do_analyses_split_feat(feat, tfn, ffn, n_perms, prefix, title):
    """
    Parameters
    ----------
    feat: string
        Type of feature (either PD or WD)
    """
    controls = pd.read_csv("{}_controls.csv".format(feat))
    patients = pd.read_csv("{}_patients.csv".format(feat))
    controls = do_permtest(controls, tfn, ffn, n_perms, "{}_controls".format(prefix), "{} Controls".format(title))
    patients = do_permtest(patients, tfn, ffn, n_perms, "{}_patients".format(prefix), "{} Patients".format(title))
    
    plt.clf()
    control = controls['emds']
    patient = patients['emds']
    nauroc =  2*(getAUROC(control, patient)['auroc']-0.5)
    sns.distplot(control, kde=True, norm_hist=True, label="Control")
    sns.distplot(patient, kde=True, norm_hist=True, label="Patient")
    plt.legend()
    plt.title("EMDs of {}, NAUROC = {:.3f}".format(title, nauroc))
    plt.xlabel("Earth Movers Distance")
    plt.ylabel("Density")
    plt.savefig("{}_EMD.svg".format(prefix), bbox_inches='tight')

    plt.clf()
    control = controls['mean_ranks']
    patient = patients['mean_ranks']
    nauroc =  2*(getAUROC(control, patient)['auroc']-0.5)
    sns.distplot(control, kde=True, norm_hist=True, label="Control")
    sns.distplot(patient, kde=True, norm_hist=True, label="Patient")
    plt.legend()
    plt.title("Mean Ranks of {}, NAUROC = {:.3f}".format(title, nauroc))
    plt.xlabel("Mean Rank")
    plt.ylabel("Density")
    plt.savefig("{}_MR.svg".format(prefix), bbox_inches='tight')

def do_analyses_feat(feat, n_perms = 100000):
    plt.figure(figsize=(8, 4))
    ## Test 1
    ## Compare nodes that are *uniquely* spelling (i.e., don't overlap with other networks) 
    ## to non-language nodes (omit nodes that are naming or syntax)
    tfn = lambda labels: labels[:, LIDX["Spelling_node"]]*(np.sum(labels, 1) == 1)
    ffn = lambda labels: (1-labels[:, LIDX["Spelling_node"]])*(1-labels[:, LIDX["Naming_node"]])*(1-labels[:, LIDX["Syntax_node"]])
    prefix = feat + "_spelling_nonlanguage"
    title = feat + " Uniquely Spelling To Non-Language"
    do_analyses_split_feat(feat, tfn, ffn, n_perms, prefix, title)



do_analyses_feat("WD")
do_analyses_feat("PC")
do_analyses_feat("BC")