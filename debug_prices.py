import yfinance as yf
import pandas as pd
import numpy as np

SECTOR_UNIVERSE = {
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'AXISBANK', 'KOTAKBANK', 'PNB', 'IDFCFIRSTB', 'FEDERALBNK'],
}

def debug_advisor():
    sector = 'Banking'
    stock_type = 'Mid'
    candidate_symbols = SECTOR_UNIVERSE['Banking']
    yf_tickers = [f"{s}.NS" for s in candidate_symbols]
    
    print(f"Fetching prices for {yf_tickers}...")
    try:
        price_df = yf.download(yf_tickers, period="5d", progress=False)['Close']
        print(f"Price DF columns: {price_df.columns.tolist()}")
        print(f"Price DF head:\n{price_df.tail(1)}")
        
        current_prices = {}
        for sym in candidate_symbols:
            ticker_col = f"{sym}.NS"
            if ticker_col in price_df.columns:
                val = price_df[ticker_col].dropna()
                if not val.empty:
                    current_prices[sym] = float(val.iloc[-1])
            else:
                print(f"Ticker {ticker_col} NOT found in columns!")
        
        print(f"Processed prices: {current_prices}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_advisor()
