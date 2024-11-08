import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
from data import get_stock_data,load_stock_data
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


def analyze_pairs(symbol1, symbol2, sector, start_date, end_date):
    """
    Analyze two stocks for pairs trading suitability using Polygon.io data
    
    Parameters:
    symbol1, symbol2: str, stock symbols
    start_date, end_date: datetime objects
    
    Returns:
    dict: Dictionary containing all metrics
    """
    # Get data for both stocks
    df1 = load_stock_data(symbol1,sector, start_date, end_date)
    df2 = load_stock_data(symbol2,sector, start_date, end_date)
    
    if df1 is None or df2 is None:
        return None
    
    data1 = df1['close']
    data2 = df2['close']
    
    # Ensure same length and clean data
    common_index = data1.index.intersection(data2.index)
    data1 = data1[common_index]
    data2 = data2[common_index]
    
    # Calculate returns
    returns1 = data1.pct_change().dropna()
    returns2 = data2.pct_change().dropna()
    
    # Initialize results dictionary
    results = {
        'returns1':returns1,
        'returns2':returns2,
        'pair': f"{symbol1}-{symbol2}",
        'observations': len(data1),
        'start_date': common_index[0],
        'end_date': common_index[-1],
        'price_data': {
            symbol1: data1,
            symbol2: data2
        }
    }
    
    # 1. Correlation Analysis
    correlation = returns1.corr(returns2)
    results['correlation'] = correlation
    results['correlation_pvalue'] = stats.pearsonr(returns1.values, returns2.values)[1]
    
    # 2. Cointegration Test
    coint_result = coint(data1, data2)
    results['coint_pvalue'] = coint_result[1]
    results['coint_stat'] = coint_result[0]
    
    # 3. Calculate spread
    spread = data1 - (data2 * (data1.mean() / data2.mean()))
    results['spread'] = spread
    
    # 4. Spread Analysis
    results['spread_mean'] = spread.mean()
    results['spread_std'] = spread.std()
    results['spread_zscore'] = (spread - spread.mean()) / spread.std()
    
    # 5. Half-life of mean reversion
    spread_lag = spread.shift(1)
    spread_diff = spread - spread_lag
    spread_lag = spread_lag.dropna()
    spread_diff = spread_diff.dropna()
    
    model = np.polyfit(spread_lag, spread_diff, 1)
    half_life = -np.log(2) / model[0]
    results['half_life'] = half_life
    
    # 6. Stationarity Test (ADF)
    adf_result = adfuller(spread)
    results['adf_pvalue'] = adf_result[1]
    results['adf_stat'] = adf_result[0]
    
    # 7. Beta and Hedge Ratio
    beta = np.cov(returns1, returns2)[0,1] / np.var(returns2)
    results['beta'] = beta
    X = returns2.values.reshape(-1, 1)
    y = returns1.values.reshape(-1, 1)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y)
    hedge_ratio = model.coef_[0][0]
    results['hedge_ratio'] = hedge_ratio
    
    # 9. Volatility Analysis
    results['vol_ratio'] = returns1.std() / returns2.std()
    results['rolling_corr_std'] = returns1.rolling(30).corr(returns2).std()
    
    # 10. Trading Metrics
    spread_zscore = results['spread_zscore']

    
    # 11. Risk Metrics
    results['max_deviation'] = abs(spread_zscore).max()
    results['var_95'] = np.percentile(abs(spread_zscore), 95)

    
    # Maximum drawdown
    rolling_max = spread.expanding().max()
    drawdowns = (spread - rolling_max) / rolling_max
    results['max_drawdown'] = drawdowns.min()
    
    # Value at Risk
    results['var_95'] = np.percentile(spread, 5)
    results['cvar_95'] = spread[spread <= results['var_95']].mean()
    
    check_pair_suitability(results)
    return results

def print_analysis(results):
    """Print formatted analysis results"""
    if results is None:
        print("Analysis failed - no data available")
        return
        
    print(f"\nAnalysis for pair: {results['pair']}")
    print(f"Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
    print(f"Number of observations: {results['observations']}")
    
    print("\nKey Metrics:")
    print(f"Correlation: {results['correlation']:.3f}")
    print(f"Correlation significance: {results['correlation_pvalue']:.3f}")

    print(f"Cointegration : {results['coint_stat']:.3f}")
    print(f"Cointegration significance: {results['coint_pvalue']:.3f}")

    print(f"ADF Unit Root: {results['adf_stat']:.3f}")
    print(f"ADF Unit Root significance: {results['adf_pvalue']:.3f}")

    print(f"Half-life: {results['half_life']:.1f} days")

    print(f"Beta/Hedge Ratio: {results['beta']:.3f}")
        
    print("\nRisk Metrics:")
    print(f"Volatility ratio: {results['vol_ratio']:.3f}")
 
    print("\nSuitability:")
    print(f"Suitable for pairs trading: {results['suitable_pair']}")
    print(f"Rejection if any")
    print(results['rejection_reasons'])

def check_pair_suitability(results):
    rejection_reasons = []
    
    if results['correlation'] <= 0.5:
        rejection_reasons.append("Low correlation")
    if results['correlation_pvalue'] >= 0.05:
        rejection_reasons.append("Correlation not significant")
    
    if results['coint_stat'] >= -3.3377:
        rejection_reasons.append("Weak cointegration")
    if results['coint_pvalue'] >= 0.05:
        rejection_reasons.append("Cointegration not significant")
    
    if results['adf_stat'] >= -1.94:
        rejection_reasons.append("Spread not stationary")
    if results['adf_pvalue'] >= 0.05:
        rejection_reasons.append("ADF test not significant")
    
    if not (1 < results['half_life'] < 30):
        rejection_reasons.append("Half-life outside range")

    results['suitable_pair'] = len(rejection_reasons) == 0
    results['rejection_reasons'] = rejection_reasons if rejection_reasons else "Pair suitable"
    
    return results


