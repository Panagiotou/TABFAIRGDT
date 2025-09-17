from fairlearn.metrics import demographic_parity_difference, true_positive_rate_difference, false_positive_rate_difference, demographic_parity_ratio
from scipy.stats import gaussian_kde
import numpy as np 
# from npeet import entropy_estimators as ee


def eq_odd(y_test, y_pred, group_test):
    return true_positive_rate_difference(y_test, y_pred, sensitive_features=group_test)\
                + false_positive_rate_difference(y_test, y_pred, sensitive_features=group_test)

def stat_par(y_test, y_pred, group_test):
    return demographic_parity_difference(y_test, y_pred, sensitive_features=group_test)

def eq_opp(y_test, y_pred, group_test):
    return true_positive_rate_difference(y_test, y_pred, sensitive_features=group_test)

def dpr(y_test, y_pred, group_test):
    return demographic_parity_ratio(y_test, y_pred, sensitive_features=group_test)




def _joint_2(X, Y, damping=1e-10):
    """
    Compute joint density matrix between X and Y
    """
    # Standardize inputs
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    
    # Create density estimator with standardized data
    density = gaussian_kde(np.vstack([X, Y]))
    
    # Create grid for density estimation
    nbins = 50
    x_grid = np.linspace(-2.5, 2.5, nbins)
    y_grid = np.linspace(-2.5, 2.5, nbins)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([xx.ravel(), yy.ravel()])
    
    # Evaluate density on grid
    h2d = density(grid_points).reshape(nbins, nbins)
    h2d = h2d + damping
    h2d = h2d / h2d.sum()
    
    return h2d

def hgr(y_pred_proba, group_test, damping=1e-10):
    """
    Calculate HGR coefficient between predictions and protected group
    """
    # Take class 1 probability and convert to numpy
    X = y_pred_proba.iloc[:, 1].values
    Y = group_test.values
    
    # Get joint density matrix
    h2d = _joint_2(X, Y, damping)
    
    # Calculate marginals
    marginal_x = h2d.sum(axis=1)[:, np.newaxis]
    marginal_y = h2d.sum(axis=0)[np.newaxis, :]
    
    # Compute normalized matrix Q
    Q = h2d / np.sqrt(np.dot(marginal_x, marginal_y))
    
    # Return second singular value
    s = np.linalg.svd(Q, compute_uv=False)
    return s[1]

# def mi_s_y(s, y):
#     return [ee.micd(s.values.reshape(-1, 1), y.values.reshape(-1, 1))]