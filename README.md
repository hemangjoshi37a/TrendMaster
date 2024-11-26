# TrendMaster: Advanced Stock Price Prediction using Transformer Deep Learning

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/hemangjoshi37a/TrendMaster?style=social)](https://github.com/hemangjoshi37a/TrendMaster/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/hemangjoshi37a/TrendMaster?style=social)](https://github.com/hemangjoshi37a/TrendMaster/fork)
[![GitHub Issues](https://img.shields.io/github/issues/hemangjoshi37a/TrendMaster)](https://github.com/hemangjoshi37a/TrendMaster/issues)

TrendMaster is an advanced stock price prediction library that leverages Transformer deep learning architecture to deliver highly accurate predictions, empowering investors with data-driven insights.

## Table of Contents

- [Features](#features)
- [Why TrendMaster?](#why-trendmaster)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Sample Results](#sample-results)
- [User Interface](#user-interface)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Show Your Support](#show-your-support)
- [Contact](#contact)
- [More from HJ Labs](#more-from-hj-labs)
- [Try Our Algo Trading Platform hjAlgos](#try-our-algo-trading-platform-hjalgos)

## 🚀 Features

- **Advanced Transformer-based prediction model**
- **High accuracy with mean average error of just a few percentage points**
- **Real-time data visualization**
- **User-friendly interface**
- **Customizable model parameters**
- **Support for multiple stock symbols**

## 📊 Why TrendMaster?

TrendMaster stands out as a top-tier tool for financial forecasting by:

- Utilizing a wealth of historical stock data
- Employing sophisticated deep learning algorithms
- Identifying patterns and trends beyond human perception
- Providing actionable insights for smarter investment strategies

## 🛠️ Installation

Get started with TrendMaster in just one command:

```bash
pip install TrendMaster
```

## 📈 Quick Start

Here's how to integrate TrendMaster into your Python projects:

```python
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
print(f'Training on {device} device.')
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
```

## 📈 Backtest Results

Evaluate the performance of TrendMaster using our comprehensive backtesting framework. Our Transformer-based model has been rigorously tested to ensure reliability and accuracy in diverse market conditions.

### 🔍 View Backtest Results

Explore detailed backtest results on our [hjAlgos Backtest Platform](https://hjalgos.hjlabs.in/backtest/).

![bokeh_plot (35)](https://github.com/user-attachments/assets/c2e7a910-3aa8-494d-958e-48199cf85459)


*Sample Backtest Performance Chart*

## 📊 Sample Results

Our Transformer-based prediction model demonstrates impressive accuracy:

![Transformer-Future200](https://user-images.githubusercontent.com/12392345/125791397-a344831b-b28c-4660-b295-924cb7123872.png)

## 🖥️ User Interface

TrendMaster comes with a sleek, user-friendly interface for easy data visualization and analysis:

![TrendMaster UI](https://user-images.githubusercontent.com/12392345/125791827-a4597af0-1292-42d0-9eb1-118d7ef64cbc.png)

## 📘 Documentation

For detailed documentation, including API reference and advanced usage, please visit our [Wiki](https://github.com/hemangjoshi37a/TrendMaster/wiki).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Show Your Support

If you find TrendMaster helpful, please consider giving it a star on GitHub. It helps others discover the project and motivates us to keep improving!

[![GitHub Star History](https://api.star-history.com/svg?repos=hemangjoshi37a/TrendMaster&type=Date)](https://star-history.com/#hemangjoshi37a/TrendMaster&Date)

## 📫 Contact

For questions, suggestions, or collaboration opportunities, please reach out:

- Website: [hjlabs.in](https://hjlabs.in/)
- Email: [hemangjoshi37a@gmail.com](mailto:hemangjoshi37a@gmail.com)
- LinkedIn: [Hemang Joshi](https://www.linkedin.com/in/hemang-joshi-046746aa)

## 🔗 More from HJ Labs

Check out our other exciting projects:
- [pyPortMan](https://github.com/hemangjoshi37a/pyPortMan)
- [AutoCut](https://github.com/hemangjoshi37a/AutoCut)
- [TelegramTradeMsgBacktestML](https://github.com/hemangjoshi37a/TelegramTradeMsgBacktestML)

## 📫 Try Our Algo Trading Platform hjAlgos

Ready to elevate your trading strategy?

<a href="https://hjalgos.hjlabs.in" style="
    display: inline-block;
    padding: 12px 24px;
    background-color: #2563EB;
    color: #FFFFFF;
    text-decoration: none;
    border-radius: 8px;
    font-weight: bold;
    font-size: 16px;
    transition: background-color 0.3s, transform 0.3s;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
">
    Try Our AlgoTrading Platform
</a>

---

Created with ❤️ by [Hemang Joshi](https://github.com/hemangjoshi37a)

