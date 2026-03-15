# 📈 Stock Price Prediction Using LSTM
### Minor Project | Deep Learning | Time Series Forecasting

---

## 📌 Project Overview

The art of forecasting stock prices has been a challenging task for researchers and analysts alike.
This project builds an **LSTM (Long Short-Term Memory)** deep learning model to predict stock prices
based on historical data, and deploys it as an interactive **Streamlit web application**.

---

## 🗂️ Project Structure

```
stock_project/
├── notebooks/
│   ├── 1-data-explanatory-analysis.ipynb   ← EDA, Moving Averages, Correlation
│   ├── 2-data-preprocessing.ipynb          ← Cleaning, Scaling, Train/Val/Test Split
│   └── 3-model-training.ipynb              ← LSTM Training, Metrics, Baseline, Forecast
├── app/
│   └── streamlit_app.py                    ← Interactive Web Application
├── data/
│   ├── raw/                                ← Original CSV dataset
│   └── processed/                          ← Scaled train/validate/test CSVs
├── models/
│   ├── google_stock_price_lstm_model.keras ← Trained LSTM model
│   └── google_stock_price_scaler.gz        ← Fitted MinMaxScaler
├─ reports/
│   └── figures/                            ← Forecast figure, Training Loss figure, etc...
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Download Dataset Manually or Download from Code--->(notebook-1)
Download the Google stock price CSV from:  
👉 https://finance.yahoo.com → Search **GOOGL** → Historical Data → Download  
👉 Convert XLS to CSV at: https://cloudconvert.com/xls-to-csv  
Place the CSV at `data/raw/google_stock_price.csv`

### Step 3 — Run the Notebooks (in order)
```
notebooks/1-data-explanatory-analysis.ipynb
notebooks/2-data-preprocessing.ipynb
notebooks/3-model-training.ipynb
```

### Step 4 — Launch the Web App
```bash
cd app
streamlit run streamlit_app.py
```

---

## 📊 Notebook Descriptions

### Notebook 1 — EDA
- Raw data loading & statistics
- OHLC price history visualization
- **100-day & 200-day Moving Average** analysis (Golden/Death Cross)
- Daily returns distribution & volatility
- Feature correlation heatmap
- Seasonal decomposition

### Notebook 2 — Preprocessing
- Missing value handling
- **Feature Engineering**: MA_10, MA_20, MA_100, MA_200, Daily_Return, Price_Range
- **Chronological 70/15/15 split** (no data leakage)
- MinMaxScaler fitted **only on training data**
- Saves processed CSVs + scaler

### Notebook 3 — Model Training
- 4-layer stacked LSTM (100 units/layer, Dropout 0.2)
- EarlyStopping + ReduceLROnPlateau + ModelCheckpoint callbacks
- Training/validation loss visualization
- **MAE, RMSE, R² Score, MAPE** on all splits
- **Linear Regression baseline comparison**
- **30-day future price forecast** (recursive prediction)
- Comprehensive conclusion

---

## 🌐 Web App Features

| Feature | Description |
|---|---|
| Live Stock Data | Fetches real-time data via yfinance for any ticker |
| Candlestick Chart | Interactive OHLC with MA overlays and volume |
| Moving Average Analysis | 100-day vs 200-day with bullish/bearish zones |
| Returns Distribution | Histogram of daily returns with statistics |
| LSTM Prediction | Actual vs predicted price on historical data |
| Performance Metrics | MAE, RMSE, R², MAPE displayed live |
| Future Forecast | 7–90 day recursive forecast with confidence band |
| Multiple Tickers | Google, Apple, Tesla, TCS, Reliance and more |

---

## 🧠 Model Architecture

```
Input: (60 timesteps × 12 features)
  ↓
LSTM(100) → Dropout(0.2)
  ↓
LSTM(100) → Dropout(0.2)
  ↓
LSTM(100) → Dropout(0.2)
  ↓
LSTM(100) → Dropout(0.2)
  ↓
Dense(1) → Predicted Open Price
```

**Features used**: Open, High, Low, Close, Adj Close, Volume,  
MA_10, MA_20, MA_100, MA_200, Daily_Return, Price_Range

---

## 📈 Results Summary

| Metric | Description |
|---|---|
| MAE | Mean absolute error in USD |
| RMSE | Root mean squared error (penalizes large errors) |
| R² Score | Proportion of variance explained (1.0 = perfect) |
| MAPE | Mean absolute percentage error |
| Baseline | LSTM vs Linear Regression comparison |

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
Stock market predictions are inherently uncertain and should **NOT** be used as financial advice.  
Past performance does not guarantee future results.

---

## 👩‍💻 Technologies Used

- **Python 3.9+**
- **TensorFlow / Keras** — LSTM model
- **Scikit-learn** — Preprocessing & baseline model
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Static charts
- **Plotly** — Interactive charts
- **Streamlit** — Web application
- **yfinance** — Live stock data
- **Statsmodels** — Seasonal decomposition
