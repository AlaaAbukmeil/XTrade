from datetime import datetime, timedelta
from pairs import analyze_pairs, print_analysis
from visuals import plot_returns_regression

start_date = datetime(2023, 6, 1)
end_date = datetime(2024, 1, 1)
sector = "post"
stock_1='FDX'
stock_2='UPS'
results = analyze_pairs(stock_1, stock_2, sector,start_date, end_date)
print_analysis(results)
plot_returns_regression(results['returns1'], results['returns2'], stock_1, stock_2)
