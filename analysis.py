from data import get_stock_data
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

symbol = 'GS'
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)

df = get_stock_data(symbol, start_date, end_date)

if df is not None:
    df.to_csv(f'{symbol}_2yr_data.csv', index=False)
