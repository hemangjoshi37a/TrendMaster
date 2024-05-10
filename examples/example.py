from trendmaster import TrendMaster

def main():
    tm = TrendMaster()
    symbol = 'AAPL'
    transformer_params = {'num_layers': 3, 'dropout': 0.1}
    tm.train_model(symbol, transformer_params)
    predictions = tm.infer_model(symbol, '2021-01-01', '2021-01-10')
    print(predictions)

if __name__ == "__main__":
    main()