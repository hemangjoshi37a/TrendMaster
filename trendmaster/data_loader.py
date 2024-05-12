import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from jugaad_trader import Zerodha
import pandas as pd

class DataLoader:
    
    def authenticate_user(self):
        """
        Authenticate the user with Zerodha and return the kite instance.

        :return: Zerodha kite instance after successful authentication
        """
        kite = Zerodha()
        print("Please enter your Zerodha credentials:")
        user_id = input("Zerodha User ID: ")
        password = input("Zerodha Password: ")
        twofa = input("Zerodha 2FA: ")
        kite.login(user_id=user_id, password=password, twofa=twofa)
        return kite

    def get_stock_data(self, kite, symbol):
        """
        Fetch stock data for a given symbol using the provided kite instance.

        :param kite: Zerodha kite instance
        :param symbol: str, stock symbol to fetch data for
        :return: pandas dataframe of given stock name
        """
        print(f"Fetching data for {symbol}")
        from_date = input("Enter start date (YYYY-MM-DD): ")
        to_date = input("Enter end date (YYYY-MM-DD): ")
        interval = 'minute'
        tkn = kite.ltp(f'NSE:{symbol}')[f'NSE:{symbol}']['instrument_token']
        data = kite.historical_data(tkn, from_date, to_date, interval)
        filename = f"{symbol}_data.csv"
        this_df  =  pd.DataFrame(data)
        this_df.to_csv(filename)
        print(f"Data saved to {filename}")
        return this_df

    
    def load(self, symbol, filepath):
        this_inst_df = joblib.load(f'{filepath}/{symbol}_data.pkl')
        amplitude = this_inst_df['close'].to_numpy()
        amplitude = amplitude.reshape(-1)
        
        scaler = MinMaxScaler(feature_range=(-15, 15))
        amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
        
        return amplitude, scaler
    
    def authenticate_user():
        print("Please enter your Zerodha credentials:")
        user_id = input("Zerodha User ID: ")
        password = input("Zerodha Password: ")
        twofa = input("Zerodha 2FA: ")
        kite = Zerodha(user_id=user_id, password=password, twofa=twofa)
        kite.login()
        return kite
    