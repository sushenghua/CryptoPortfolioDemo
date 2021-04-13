import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import norm

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    if isinstance(r, pd.Series):
        statistic, p_value = scipy.stats.jarque_bera(r)
        return p_value > level
    elif isinstance(r, pd.DataFrame): 
        return r.aggregate(is_normal, level=level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')
    


def drawdown(return_series: pd.Series, deposit_fund = 1.):
    """
    Takes a times series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    """
    wealth_index = deposit_fund * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdowns
    })

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computer the skewness of the supplied Serires or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computer the kurtosis of the supplied Serires or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def semideviation(r, filter='-'):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Serires or a DataFrame
    """
    r_filtered = r < 0 if filter == '-' else r >= 0
    return r[r_filtered].std(ddof=0)

def var_historic(r, level=5):
    """
    Returns the historic Value At Risk at a specified level
    i.e. return the number such that 'level' percent of the returns
    fall below that number,and the (100-level) percent are above
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        # level percent change of lose -np.percentile(r, level)*100 percent
        return -np.percentile(r, level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame): 
        return r.aggregate(var_historic, level=level)
    else:
        raise TypeError('Expected r to be Series or DataFrame')
        
def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gaussian VaR of a Series or DataFrame
    If 'modified' is True, then the modified VaR is returned
    using the Cornish-Fisher modification
    """
    # computer the z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * (k - 3) / 24 -
                (2*z**3 - 5*z) * (s**2) / 36
            )
    return -(r.mean() + z * r.std(ddof=0))
