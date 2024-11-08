import requests
import pandas as pd
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta


load_dotenv()
api_key = os.getenv('POLYGON_API_KEY')
def get_stock_data(symbol, start_date, end_date):
    base_url = "https://api.polygon.io/v2/aggs/ticker"
    
    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')
    
    url = f"{base_url}/{symbol}/range/1/day/{start}/{end}?apiKey={api_key}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            
            df = df.rename(columns={
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        else:
            print(f"Error: {data.get('error')}")
            return None
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def load_stock_data(symbol, sector, start_date=None, end_date=None):
    df = pd.read_csv(f'./data/{sector}/{symbol}_2yr_data.csv')
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    df = df.sort_index()
    
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df.index >= start_date]
    
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df.index <= end_date]
    
    return df

symbol = 'FDX'
sector = 'post'
end_date = datetime.now()
start_date = end_date - timedelta(days=3650)

df = get_stock_data(symbol, start_date, end_date)

if df is not None:
    df.to_csv(f'./data/{sector}/{symbol}_2yr_data.csv', index=False)

