"""
Perform a regression on volumes of DTI tracts
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV

def get_data(measurement = 'severity', zero_nans=True):
    """
    Return a set of DTI features
    Parameters
    ----------
    measurement: string
        Dependent variable to use
    zero_nans: boolean
        Whether to replace NaNs with zeros
    Returns
    -------
    V: ndarray(n_subjects, n_tracts)
        An array of features over tracts for each subject
    y: ndarray(n_subjects)
        Measurement for each subject
    tracts: array(string)
        The string associated to each tract
    """
    y = pd.read_csv('BehavioralScores.csv')
    y = y[measurement].values
    V = pd.read_csv('TractVolume.csv')
    tracts = V.columns[1::]
    tracts = [t.replace("_ApproxTractVolume", "") for t in tracts]
    V = np.array(V.values[:, 1::], dtype=float)
    V[np.isnan(V)] = 0
    y[np.isnan(y)] = 0
    return V, y, tracts

def do_ridge_cv(measurement = 'severity', fit_intercept=True, alphas=np.linspace(0.01, 10, 1000)):
    """
    Do ridge regression with leave one out cross-validation
    Parameters
    ----------
    measurement: string
        Parameter to regress on
    fit_intercept: boolean
        Whether to subtract off the y-intercept before regressing
    alphas: ndarray(N)
        A list of regularization parameters to try
    Returns
    -------
    rsqr: float
        The r-squared score for the fit
    """
    V, y, tracts = get_data(measurement)

    clf = RidgeCV(alphas, normalize=True, store_cv_values=True, fit_intercept=fit_intercept).fit(V, y)
    rsqr = clf.score(V, y)
    mses = np.mean(clf.cv_values_, 0)
    pix = np.arange(V.shape[1])

    plt.subplot(131)
    plt.stem(pix, clf.coef_) 
    plt.xticks(pix, tracts, rotation='vertical')
    plt.ylabel("Coefficient")
    plt.xlabel("Tract")
    plt.title("%s Volume Coefficients"%measurement)

    plt.subplot(132)
    plt.plot(alphas, mses)
    plt.scatter(clf.alpha_, mses[np.argmin(np.abs(clf.alpha_-alphas))])
    plt.xlabel("Alpha")
    plt.ylabel("MSE")
    plt.title("MSEs Over Regularization", y=1.08)
    
    plt.subplot(133)
    plt.scatter(y, clf.predict(V))
    plt.axis('equal')
    plt.xlabel(measurement)
    plt.ylabel("Predicted %s"%measurement)
    plt.legend(["rsqr = %.3g"%rsqr])
    plt.title("Linear Model Predictions\nintercept=%.3g"%clf.intercept_)
    plt.tight_layout()

    return rsqr




def do_lasso_cv(measurement = 'severity', fit_intercept=True):
    """
    Do LASSO with leave one out cross-validation
    Parameters
    ----------
    measurement: string
        Parameter to regress on
    fit_intercept: boolean
        Whether to subtract off the y-intercept before regressing
    Returns
    -------
    rsqr: float
        The r-squared score for the fit
    """
    V, y, tracts = get_data(measurement)

    clf = LassoCV(normalize=True, fit_intercept=fit_intercept, cv=y.size-1).fit(V, y)
    rsqr = clf.score(V, y)
    mses = np.mean(clf.mse_path_, 1)
    pix = np.arange(V.shape[1])

    plt.subplot(131)
    plt.stem(pix, clf.coef_) 
    plt.xticks(pix, tracts, rotation='vertical')
    plt.ylabel("Coefficient")
    plt.xlabel("Tract")
    plt.title("%s Volume Coefficients"%measurement)

    plt.subplot(132)
    plt.plot(clf.alphas_, mses)
    plt.scatter(clf.alpha_, mses[np.argmin(np.abs(clf.alpha_-clf.alphas_))])
    plt.xlabel("Alpha")
    plt.ylabel("MSE")
    plt.title("MSEs Over Regularization", y=1.08)
    
    plt.subplot(133)
    plt.scatter(y, clf.predict(V))
    plt.axis('equal')
    plt.xlabel(measurement)
    plt.ylabel("Predicted %s"%measurement)
    plt.legend(["rsqr = %.3g"%rsqr])
    plt.title("Linear Model Predictions\nintercept=%.3g"%clf.intercept_)
    plt.tight_layout()

    return rsqr



def do_all_experiments():
    # Do all LASSO
    res = pd.read_csv('BehavioralScores.csv')
    measurements = res.columns

    plt.figure(figsize=(12, 4))
    fout = open("lasso/lasso.csv", "w")
    fout.write("measurement,intercept,nointercept")
    for measurement in measurements:
        fout.write("\n%s"%measurement)
        for fit_intercept in [True, False]:
            plt.clf()
            rsqr = do_lasso_cv(measurement, fit_intercept=fit_intercept)
            plt.savefig("lasso/%s_%i.png"%(measurement, fit_intercept))
            fout.write(",%.3g"%rsqr)
    fout.close()

    # Do all Ridge
    fout = open("ridge/ridge.csv", "w")
    fout.write("measurement,intercept,nointercept")
    for measurement in measurements:
        fout.write("\n%s"%measurement)
        for fit_intercept in [True, False]:
            plt.clf()
            rsqr = do_ridge_cv(measurement, fit_intercept=fit_intercept)
            plt.savefig("ridge/%s_%i.png"%(measurement, fit_intercept))
            fout.write(",%.3g"%rsqr)
    fout.close()


if __name__ == '__main__':
    do_all_experiments()