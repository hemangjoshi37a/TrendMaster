from trendmaster import TrendMaster

def main():
    # Initialize the TrendMaster object
    tm = TrendMaster()

    # Load your data
    data = tm.load_data('path_to_your_data.csv')

    # Train the model with specified transformer parameters
    tm.train(data, transformer_params={'num_layers': 3, 'dropout': 0.1})

    # Save the trained model
    tm.trainer.save_model('path_to_save_trained_model.pth')

    # Perform inference using the trained model
    predictions = tm.infer('path_to_trained_model.pth')
    print(predictions)

if __name__ == "__main__":
    main()