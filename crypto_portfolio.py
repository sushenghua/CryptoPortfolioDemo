import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Binance import BinanceClient as client
import risk_kit as rk
# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

def historical_tradedata_pd(tradepairs, from_datetime, interval='1h', columns='Close'):
    bc = client.instance()
    tradepairs_data_list = []
    for tradepair in tradepairs:
        candles = bc.get_historical_klines_pd(tradepair, from_datetime, interval=interval)
        tradepairs_data_list.append(candles[columns])

    tradepairs_pd = pd.concat(tradepairs_data_list, axis=1)
    tradepairs_pd.columns = tradepairs
    return tradepairs_pd

def historical_tradedata_analysis(tradepairs, from_datetime, interval='1h',
                                  period_info={'name': 'Daily', 'intervals': 24}):
    # -------------------------------------------------------------------
    # --- data frame and display params
    tradedata_pd = historical_tradedata_pd(tradepairs, from_datetime, interval)

    # - return series, covariance
    rets = tradedata_pd.pct_change().dropna()
    cov = rets.cov()
    corr = rets.corr()

    # - display
    titlesize = 14
    figsize = (8, 6)
    glstyle = '-.'
    glwidth = 0.5

    # -------------------------------------------------------------------
    # --- coorelation map
    cellsize = 1.1*len(tradepairs)
    heatmap_figsize = (cellsize+1, cellsize)
    plt.subplots(figsize=heatmap_figsize)
    ax = sns.heatmap(corr, cmap='Blues', annot=True)
    ax.xaxis.tick_top()
    plt.yticks(rotation=0)
    plt.title('Correlations', fontsize=titlesize)
    plt.show()

    # -------------------------------------------------------------------
    # --- plot price line
    for cln in tradepairs:
        plt.figure(figsize=figsize)
        tradedata_pd[cln].plot(xlabel='', ylabel='USDT')
        plt.title(cln, fontsize=titlesize)
        plt.grid(linestyle=glstyle, linewidth=glwidth)
        plt.show()

    # -------------------------------------------------------------------
    # --- return, volatility and sharpe ratio
    # - periodized return, volatility and sharpe ratio
    periodized_ret = rk.periodized_ret(rets, period_info['intervals'])
    periodized_vol = rk.periodized_vol(rets, period_info['intervals'])
    sharpe_ratio = periodized_ret / periodized_vol  # riskfree_rate is 0

    # - Return bar
    plt.figure(figsize=figsize)
    ax = periodized_ret.plot.bar(ylabel='%', color=_value_to_2color(periodized_ret.values))
    ax.yaxis.set_major_formatter(_ticklabel_formatter)
    plt.title('{} Return'.format(period_info['name']) ,fontsize=titlesize)
    plt.axhline(y=0, color='darkblue', linestyle='--', linewidth=1)
    plt.grid(axis='y', linestyle=glstyle, linewidth=glwidth)
    plt.show()

    # - Volatility bar
    plt.figure(figsize=figsize)
    ax = periodized_vol.plot.bar(ylabel='%', color='royalblue')
    ax.yaxis.set_major_formatter(_ticklabel_formatter)
    plt.title('{} Volatility'.format(period_info['name']), fontsize=titlesize)
    plt.grid(axis='y', linestyle=glstyle, linewidth=glwidth)
    plt.show()

    # - Sharpe Ratio bar
    plt.figure(figsize=figsize)
    ax = sharpe_ratio.plot.bar(color=_value_to_2color(sharpe_ratio.values))
    ax.set_ylabel(r'$\frac{Return}{Volatility}$', fontsize=15)
    #ax.yaxis.set_major_formatter(_ticklabel_formatter)
    plt.title('{} Sharpe Ratio (Riskfree Rate 0)'.format(period_info['name']), 
              fontsize=titlesize)
    plt.axhline(y=0, color='darkblue', linestyle='--', linewidth=1)
    plt.grid(axis='y', linestyle=glstyle, linewidth=glwidth)
    plt.show()

    # -------------------------------------------------------------------
    # --- plot drawdown
    dd = rk.drawdown(rets, return_dict=True)

    # - peaks
    dd['Peaks'].plot(ylabel='USDT', xlabel='', figsize=figsize)
    plt.title('Peak (Invest 1 USDT)', fontsize=titlesize)
    plt.grid(linestyle=glstyle, linewidth=glwidth)

    # - wealth
    dd['Wealth'].plot(ylabel='USDT', xlabel='', figsize=figsize)
    plt.title('Wealth (Invest 1 USDT)', fontsize=titlesize)
    plt.grid(linestyle=glstyle, linewidth=glwidth)

    # - drawdown
    ax = dd['Drawdown'].plot(ylabel='%', xlabel='', figsize=figsize)
    ax.yaxis.set_major_formatter(_ticklabel_formatter)
    plt.title('Drawdown', fontsize=titlesize)
    plt.grid(linestyle=glstyle, linewidth=glwidth)

    # -------------------------------------------------------------------
    # --- portfolio efficient frontier
    ax = rk.plot_ef(30, periodized_ret, cov,
                    show_cml=True, show_ew=True, show_gmv=True,
                    show_msr=True, figsize=figsize)
    ax.set_title('Portfolio Efficient Frontier', fontsize=titlesize)
    plt.grid(linestyle=glstyle, linewidth=glwidth)
    plt.show()
    
    # -------------------------------------------------------------------
    # --- portfolio structures
    portfolio_pie_plot(rk.msr(periodized_ret, cov), tradepairs, name='MSR')
    portfolio_pie_plot(rk.gmv(cov), tradepairs, name='GMV')

def _value_to_2color(values):
    return ['forestgreen' if v > 0 else 'crimson' for v in values]

def _ticklabel_formatter(v, pos):
    return round(v * 100, 2)

def portfolio_pie_plot(weights, rows_index, name, figsize=(5, 5)):
    if len(weights) != len(rows_index) or len(weights) == 0:
        raise ValueError('weights not compatible with rows_index or empty weights')
    weights100 = weights * 100
    weights100 = (weights100 > 1e-4) * weights100
    df = pd.DataFrame({name: weights100}, index=rows_index)

#     plt.figure(figsize=figsize)
#     patches, texts, pcts = plt.pie(df[name], labels=df.index,
#                                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
#                                    #textprops=dict(color='w', size='large', weight='bold'),
#                                    autopct=lambda v: '{:.1f}%'.format(v) if v > 0 else '')
#     plt.setp(pcts, color='w', fontweight='bold')
    ax = df.plot.pie(y=name, figsize=figsize,
                     wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                     #explode=(0.01, 0.01, 0.01, 0.01),
                     #textprops=dict(color='w', size='large', weight='bold'),
                     autopct=lambda v: '{:.1f}%'.format(v) if v > 0 else '')
    ax.set_ylabel('')
    plt.title('{} Portfolio'.format(name), fontsize=15)
    plt.legend(bbox_to_anchor=(1.5, 0.8))
    #plt.tight_layout()
    plt.show()
    