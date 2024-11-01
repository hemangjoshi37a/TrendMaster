# Example usage of merged_module.py

from trendmaster import (
    DataLoader,
    TransAm,
    Trainer,
    Inferencer,
    set_seed,
    plot_results,
    plot_predictions
)

import pyotp

# Set seed for reproducibility
set_seed(42)

user_id = 'YOUR_ZERODHA_USER_ID'
password = 'YOUR_ZERODHA_PASSWORD'  # Replace with your password
totp_key = 'YOUR_ZERODHA_2FA_KEY'   # Replace with your TOTP secret key

# Generate the TOTP code for two-factor authentication
totp = pyotp.TOTP(totp_key)
twofa = totp.now()

# Initialize DataLoader and authenticate
data_loader = DataLoader()
kite = data_loader.authenticate(user_id=user_id, password=password, twofa=twofa)

# Prepare data
train_data, test_data = data_loader.prepare_data(
    symbol='RELIANCE',
    from_date='2023-01-01',
    to_date='2023-02-27',
    input_window=30,
    output_window=10,
    train_test_split=0.8
)
import torch
# Initialize model, trainer, and train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training of {device} device.')
model = TransAm(num_layers=2, dropout=0.2).to(device)

trainer = Trainer(model, device, learning_rate=0.001)
train_losses, val_losses = trainer.train(train_data, test_data, epochs=2, batch_size=64)

# Save the trained model
trainer.save_model('transam_model.pth')

# Initialize inferencer and make predictions
inferencer = Inferencer(model, device, data_loader)
predictions = inferencer.predict(
    symbol='RELIANCE',
    from_date='2023-02-27',
    to_date='2023-12-31',
    input_window=30,
    future_steps=10
)

# Evaluate the model
test_loss = inferencer.evaluate(test_data, batch_size=32)
