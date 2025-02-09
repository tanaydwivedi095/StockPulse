# StockPulse: A Stock Price Prediction System

StockPulse is a machine learning-based stock price prediction system that leverages LSTMs and GRUs to forecast stock prices based on historical data. It provides data preprocessing, visualization, and predictive analytics to help users analyze stock market trends.

## Features

- Utilizes LSTM and GRU models for time-series forecasting
- Supports multiple stock market datasets
- Performs data normalization and preprocessing
- Provides visualization for stock trends and model accuracy
- Exports predictions to Excel for further analysis

## Installation

Follow these steps to set up and run StockPulse:

### 1. Clone the repository:

```bash
git clone https://github.com/tanaydwivedi095/StockPulse.git
cd StockPulse
```

### 2. Install dependencies:

Ensure you have Python installed, then manually install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn keras tensorflow openpyxl
```

### 3. Run the application:

```bash
python stock_pulse.py
```

## Usage

1. The program prompts you to enter a stock symbol.
2. It processes historical stock data and trains a predictive model.
3. The model forecasts future stock prices and visualizes results.
4. Predictions are saved as `results.xlsx` for reference.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, Keras, TensorFlow
- **Deep Learning Models:** LSTMs, GRUs
- **Data Processing:** MinMax Scaling, Time-series Analysis

## Project Structure

```
StockPulse/
│── data/                 # Contains stock price datasets
│── stock_prediction.py   # Main script for stock prediction
│── results.xlsx          # Output file with stock price predictions
└── README.md             # Project documentation
```

## Contributing

Feel free to contribute by submitting issues or pull requests.

## License

This project is licensed under the MIT License.

---

⭐ **Star this repository** if you find it useful!

