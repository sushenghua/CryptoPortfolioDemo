import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize

def periodized_ret(r, segments_per_period):
    """
    Periodizes a set of segment-returns
    """
    compounded_growth = (1+r).prod()
    n_segments = r.shape[0]
    return compounded_growth ** (segments_per_period/n_segments) - 1     

def periodized_vol(r, segments_per_period):
    """
    Periodizes the volatility of a set of segment-returns
    """
    segment_std = r.std()
    return segment_std * np.sqrt(segments_per_period)

def sharp_ratio(r, riskfree_rate, segments_per_period):
    """
    Computes the periodized sharp ratio of a set of segment-returns
    """
    # convert the periodized riskfree rate to per segment
    rf_per_segment = (1 + riskfree_rate) ** (1/segments_per_period) - 1
    excess_ret = r - rf_per_segment
    periodized_ex_ret = periodized_ret(excess_ret, segments_per_period)
    periodized_vol = periodized_vol(r, segments_per_period)
    return periodized_ex_ret / periodized_vol

def annualized_ret(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods) - 1

def annualized_vol(r, period_per_year):
    """
    Annualized the vol of a set of returns
    We should infer the periods per year
    """
    return r.std() * (period_per_year**0.5)

# def sharp_ratio(r, riskfree_rate, periods_per_year):
#     """
#     Computes the annualized sharp ratio of a set of returns
#     """
#     # convert the annual riskfree rate to per period
#     rf_per_period = (1 + riskfree_rate) ** (1/periods_per_year) - 1
#     excess_ret = r - rf_per_period
#     ann_ex_ret = annualized_ret(excess_ret, periods_per_year)
#     ann_vol = annualized_vol(r, periods_per_year)
#     return ann_ex_ret/ann_vol

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

def drawdown(return_series: pd.Series, deposit_fund = 1., return_dict=False):
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
    if return_dict:
        return {
            'Wealth': wealth_index,
            'Peaks': previous_peaks,
            'Drawdown': drawdowns
        }
    else:
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

def portfolio_ret(weights, ret):
    """
    Weights -> Returns
    """
    #return weights.T @ ret
    return np.dot(ret, weights.T)

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    #return (weights.T @ covmat @ weights)**0.5
    return np.sqrt(np.dot(np.dot(weights, covmat), weights.T))

def plot_ef2(n_points, er, cov, style='.-'):
    """
    Plots the 2-asset efficient frontier
    """
    if er.shape[0] != 2 or cov.shape[0] != 2:
        raise ValueError('plot_ef2 can only plot 2-asset frontier')
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_ret(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    return ef.plot.line(x='Volatility', y='Returns', style=style)

def minimize_vol(target_return, er, cov):
    """
    target_return -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n
    return_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_ret(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol,
                       init_guess,
                       args=(cov,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x

def optimal_weights(n_points, er, cov):
    """
    -> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def gmv(cov):
    """
    Returns the weights of the Global Minimum Vol portfolio
    given the covariance matrix
    """
    n = cov.shape[0]
    return msr(np.repeat(1, n), cov)

def plot_ef(n_points, er, cov,
            show_cml=True, riskfree_rate=0, cml_color='green',
            show_ew=False, ew_color='goldenrod',
            show_gmv=False, gmv_color='tomato',
            show_msr=False, msr_color='magenta',
            style='.-', figsize=(8, 6)):
    """
    Plots the N-asset efficient frontier
    """

    # -------------------------------------------------------------------
    # --- weights, returns, volatility, msr
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_ret(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        'Returns': rets,
        'Volatility': vols
    })
    w_msr = msr(er, cov, riskfree_rate)
    r_msr = portfolio_ret(w_msr, er)
    v_msr = portfolio_vol(w_msr, cov)
    
    # -------------------------------------------------------------------
    # --- efficient frontier line
    ax = ef.plot.line(x='Volatility', y='Returns', style=style, figsize=figsize)
    ax.set_xlim(left=0)
    ax.set_xlabel('Volatility %')
    ax.set_ylabel('Returns %')
    ax.xaxis.set_major_formatter(_ticklabel_formatter)
    ax.yaxis.set_major_formatter(_ticklabel_formatter)

    # -------------------------------------------------------------------
    # --- efficient frontier point
    if (show_ew):
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_ret(w_ew, er)
        v_ew = portfolio_vol(w_ew, cov)
        ax.plot([v_ew], [r_ew], color=ew_color, label='EW (Equal Weights)',
                marker='o', markersize=8)

    # -------------------------------------------------------------------
    # --- global minimum variance point
    if (show_gmv):
        n = er.shape[0]
        w_gmv = gmv(cov)
        r_gmv = portfolio_ret(w_gmv, er)
        v_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([v_gmv], [r_gmv], color=gmv_color, label='GMV (Global Minimum Variance)',
                marker='o', markersize=8)

    # -------------------------------------------------------------------
    # --- capital market line
    if (show_cml):
        cml_x = [0, v_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color=cml_color, label='CML (Capital Market Line)',
                marker='o', markersize=8, linestyle='--')
        
    # -------------------------------------------------------------------
    # --- max sharpe ratio point
    if (show_msr):
        ax.plot([v_msr], [r_msr], label='MSR (Maximum Sharpe Ratio)',
                marker='o', markersize=8, color=msr_color)
    # -------------------------------------------------------------------
    # --- final stuff
    plt.legend()

    return ax

def _ticklabel_formatter(v, pos):
    return round(v * 100, 2)

def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
    r = portfolio_ret(weights, er)
    vol = portfolio_vol(weights, cov)
    return -(r-riskfree_rate)/vol

def msr(er, cov, riskfree_rate=0):
    """
    Riskfree rate + ER + COV -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1),) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(neg_sharpe_ratio,
                       init_guess,
                       args=(riskfree_rate, er, cov,),
                       method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x
