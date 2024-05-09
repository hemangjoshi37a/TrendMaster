import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from jugaad_trader import Zerodha

class DataLoader:
    def authenticate_user():
        """
        Authenticate the user with Zerodha and return the kite instance.

        :return: Zerodha kite instance after successful authentication
        """
        kite = Zerodha()
        print("Please enter your Zerodha credentials:")
        user_id = input("User ID: ")
        password = input("Password: ")
        twofa = input("2FA: ")
        kite.login(user_id=user_id, password=password, twofa=twofa)
        return kite

    def get_stock_data(kite, symbol):
        """
        Fetch stock data for a given symbol using the provided kite instance.

        :param kite: Zerodha kite instance
        :param symbol: str, stock symbol to fetch data for
        :return: str, filename where the data is saved
        """
        print(f"Fetching data for {symbol}")
        from_date = input("Enter start date (YYYY-MM-DD): ")
        to_date = input("Enter end date (YYYY-MM-DD): ")
        interval = 'day'
        data = kite.historical_data(symbol, from_date, to_date, interval)
        filename = f"{symbol}_data.csv"
        data.to_csv(filename)
        print(f"Data saved to {filename}")
        return filename
    
    
    def load(self, inst, filepath):
        this_inst_df = joblib.load(f'{filepath}/{inst}.p')
        amplitude = this_inst_df['close'].to_numpy()
        amplitude = amplitude.reshape(-1)
        
        scaler = MinMaxScaler(feature_range=(-15, 15))
        amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)
        
        return amplitude, scaler
    def authenticate_user():
        kite = Zerodha()
        print("Please enter your Zerodha credentials:")
        user_id = input("User ID: ")
        password = input("Password: ")
        twofa = input("2FA: ")
        kite.login(user_id=user_id, password=password, twofa=twofa)
        return kite
    def get_stock_data(kite, symbol):
        print(f"Fetching data for {symbol}")
        from_date = input("Enter start date (YYYY-MM-DD): ")
        to_date = input("Enter end date (YYYY-MM-DD): ")
        interval = 'day'
        data = kite.historical_data(symbol, from_date, to_date, interval)
        filename = f"{symbol}_data.csv"
        data.to_csv(filename)
        print(f"Data saved to {filename}")
        return filename