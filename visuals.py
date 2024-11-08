import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import seaborn as sns
from sklearn.linear_model import LinearRegression

def plot_returns_regression(returns1, returns2, symbol1='Stock 1', symbol2='Stock 2'):
    """
    Plot returns scatter plot with regression line and key statistics
    
    Parameters:
    returns1, returns2: pandas Series with datetime index
    symbol1, symbol2: str, names of the securities for the plot
    """
    
    # Create figure with specific size
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    plt.scatter(returns2, returns1, alpha=0.5, color='blue', label='Daily Returns')
    
    # Fit regression line
    X = returns2.values.reshape(-1, 1)
    y = returns1.values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    
    # Get regression line points
    x_range = np.linspace(returns2.min(), returns2.max(), 100).reshape(-1, 1)
    y_pred = reg.predict(x_range)
    
    # Plot regression line
    plt.plot(x_range, y_pred, color='red', label=f'Regression Line (β={reg.coef_[0][0]:.3f})')
    
    # Calculate correlation and p-value
    correlation, p_value = stats.pearsonr(returns1, returns2)
    r_squared = reg.score(X, y)
    
    # Add statistics text box
    stats_text = f'Correlation: {correlation:.3f}\n'
    stats_text += f'R-squared: {r_squared:.3f}\n'
    stats_text += f'Beta: {reg.coef_[0][0]:.3f}\n'
    stats_text += f'P-value: {p_value:.3e}'
    
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot
    plt.title(f'Returns Regression Analysis: {symbol1} vs {symbol2}')
    plt.xlabel(f'{symbol2} Returns')
    plt.ylabel(f'{symbol1} Returns')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.legend()
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    return {
        'beta': reg.coef_[0][0],
        'correlation': correlation,
        'r_squared': r_squared,
        'p_value': p_value
    }

