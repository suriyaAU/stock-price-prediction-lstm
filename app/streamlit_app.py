"""
Stock Price Prediction — Streamlit Web App
Minor Project | LSTM-based Stock Forecasting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import joblib
import datetime
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #1e3c72, #2a9d8f);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1rem 1.2rem; border-left: 4px solid #2a9d8f;
        margin-bottom: 0.8rem;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #1e3c72; }
    .metric-label { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .positive { color: #2a9d8f !important; }
    .negative { color: #e76f51 !important; }
    .warning-box {
        background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
        padding: 0.8rem 1rem; font-size: 0.85rem; color: #856404; margin-top: 1rem;
    }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #1e3c72; margin: 1.5rem 0 0.8rem; }

    /* ── Remove sidebar bottom padding ── */
    section[data-testid="stSidebar"] {
        padding-bottom: 0rem !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-bottom: 0rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        padding-bottom: 0rem !important;
        gap: 0rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE_SIZE = 60
FEATURES      = ['Open', 'High', 'Low', 'Close', 'Volume',
                  'MA_10', 'MA_20', 'MA_100', 'MA_200', 'Daily_Return', 'Price_Range']
MODEL_PATH    = '../models/google_stock_price_lstm_model.keras'
SCALER_PATH   = '../models/google_stock_price_scaler.gz'

POPULAR_TICKERS = {
    'Google (GOOGL)': 'GOOGL',
    'Netflix (NFLX)': 'NFLX',
    'NVIDIA (NVDA)': 'NVDA', 
    'Apple (AAPL)': 'AAPL',
    'Amazon (AMZN)': 'AMZN', 
}

# ─── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=False)
    df.reset_index(inplace=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA_10']        = df['Close'].rolling(10).mean()
    df['MA_20']        = df['Close'].rolling(20).mean()
    df['MA_100']       = df['Close'].rolling(100).mean()
    df['MA_200']       = df['Close'].rolling(200).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Price_Range']  = df['High'] - df['Low']
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def scale_data(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(df[FEATURES].values)


def inverse_open(preds: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    n = scaler.n_features_in_
    dummy = np.zeros((len(preds), n))
    dummy[:, 0] = preds.ravel()
    return scaler.inverse_transform(dummy)[:, 0]


def build_sequence(data: np.ndarray, seq_size: int):
    X, y = [], []
    for i in range(seq_size, len(data)):
        X.append(data[i-seq_size:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def forecast_future(model, last_seq: np.ndarray, days: int, scaler: MinMaxScaler) -> np.ndarray:
    seq   = last_seq.copy()
    preds = []
    for _ in range(days):
        inp  = seq[-SEQUENCE_SIZE:].reshape(1, SEQUENCE_SIZE, -1)
        pred = model.predict(inp, verbose=0)[0, 0]
        preds.append(pred)
        new_row    = seq[-1].copy()
        new_row[0] = pred
        seq = np.vstack([seq[1:], new_row])
    return inverse_open(np.array(preds).reshape(-1, 1), scaler)


@st.cache_resource
def load_artifacts():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    return None, None


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/3e/STOCK-logo.svg", width=244)
    st.title("Settings")
    st.markdown("---")

    mode = st.radio("Mode", ["📊 Analysis & Predict", "🔮 Future Forecast"], index=0)
    st.markdown("---")

    ticker_choice = st.selectbox("Stock", list(POPULAR_TICKERS.keys()), index=0)
    ticker        = POPULAR_TICKERS[ticker_choice]
    custom_ticker = st.text_input("Or enter custom ticker", placeholder="e.g. INFY.NS")
    if custom_ticker.strip():
        ticker = custom_ticker.strip().upper()

    period = st.selectbox("Data Period", ['1y', '2y', '3y', '5y', '10y'], index=3)

    if mode == "🔮 Future Forecast":
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90, value=30, step=1)

    use_saved_model = st.checkbox("Use saved LSTM model", value=True,
                                  help="Use the pre-trained Google model. Uncheck to train fresh (uses live scaler only).")
    


# ─── Main Page Header ──────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 Stock Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">LSTM Deep Learning · Minor Project · Stock Price Forecasting</div>',
            unsafe_allow_html=True)

# ─── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching {ticker} data..."):
    try:
        df_raw = fetch_stock_data(ticker, period)
        df     = add_features(df_raw)
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker}**: {e}")
        st.stop()

if len(df) < SEQUENCE_SIZE + 50:
    st.error(f"Not enough data for {ticker}. Try a longer period.")
    st.stop()

# ─── Top Metrics Row ───────────────────────────────────────────────────────────
latest   = df.iloc[-1]
prev     = df.iloc[-2]
change   = latest['Close'] - prev['Close']
pct_chg  = change / prev['Close'] * 100
currency = "₹" if ".NS" in ticker or ".BO" in ticker else "$"

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Current Price", f"{currency}{latest['Close']:.2f}",
              delta=f"{change:+.2f} ({pct_chg:+.2f}%)")
with col2:
    st.metric("Today's High",  f"{currency}{latest['High']:.2f}")
with col3:
    st.metric("Today's Low",   f"{currency}{latest['Low']:.2f}")
with col4:
    vol_m = latest['Volume'] / 1e6
    st.metric("Volume", f"{vol_m:.1f}M")
with col5:
    ma100 = latest['MA_100']
    trend = "🟢 Above MA100" if latest['Close'] > ma100 else "🔴 Below MA100"
    st.metric("Trend Signal", trend)

st.markdown("---")

# ─── TAB LAYOUT ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "📉 Analysis", "🤖 LSTM Prediction", "🔮 Forecast"])

# ── TAB 1: Price Chart ─────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Candlestick Chart + Moving Averages</div>', unsafe_allow_html=True)
    show_days = st.slider("Show last N days", 60, min(len(df), 1000), 365, key="chart_days")
    df_view = df.tail(show_days)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=df_view['Date'], open=df_view['Open'], high=df_view['High'],
                                  low=df_view['Low'], close=df_view['Close'], name='OHLC',
                                  increasing_line_color='#2a9d8f', decreasing_line_color='#e76f51'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['MA_100'], name='MA 100',
                              line=dict(color='#e9c46a', width=1.5, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_view['Date'], y=df_view['MA_200'], name='MA 200',
                              line=dict(color='#f4a261', width=1.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Bar(x=df_view['Date'], y=df_view['Volume'], name='Volume',
                          marker_color='#264653', opacity=0.5), row=2, col=1)
    fig.update_layout(title=f"{ticker} — Price Chart",
                      xaxis_rangeslider_visible=False,
                      paper_bgcolor='white', plot_bgcolor='#fafafa',
                      font=dict(family='sans-serif'), height=600,
                      legend=dict(orientation='h', yanchor='bottom', y=1.01))
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Analysis ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Moving Average Analysis</div>', unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['Close'],  name='Close', line=dict(color='steelblue',  width=1.5)))
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_100'], name='MA 100',line=dict(color='darkorange', width=2, dash='dash')))
    fig2.add_trace(go.Scatter(x=df['Date'], y=df['MA_200'], name='MA 200',line=dict(color='crimson',    width=2, dash='dot')))
    fig2.add_traces([
        go.Scatter(x=df['Date'], y=df['MA_100'], fill=None,      line_color='rgba(0,0,0,0)', showlegend=False),
        go.Scatter(x=df['Date'], y=df['MA_200'], fill='tonexty', line_color='rgba(0,0,0,0)',
                   fillcolor='rgba(42,157,143,0.1)', name='MA Spread')
    ])
    fig2.update_layout(title=f"{ticker} — 100-Day vs 200-Day Moving Average",
                       xaxis_title='Date', yaxis_title=f'Price ({currency})',
                       paper_bgcolor='white', plot_bgcolor='#fafafa', height=420)
    st.plotly_chart(fig2, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Daily Returns Distribution</div>', unsafe_allow_html=True)
        fig3 = px.histogram(df, x='Daily_Return', nbins=80, title='Daily Returns (%)',
                            color_discrete_sequence=['steelblue'])
        fig3.add_vline(x=df['Daily_Return'].mean(), line_dash='dash', line_color='red',
                       annotation_text=f"Mean: {df['Daily_Return'].mean():.4f}")
        fig3.update_layout(paper_bgcolor='white', plot_bgcolor='#fafafa', height=350)
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        st.markdown('<div class="section-title">Summary Statistics</div>', unsafe_allow_html=True)
        stats = {
            'Metric': ['All-Time High', 'All-Time Low', 'Avg Daily Return', 'Volatility (Std)',
                       'Best Single Day', 'Worst Single Day', 'Avg Volume'],
            'Value': [
                f"{currency}{df['High'].max():.2f}",
                f"{currency}{df['Low'].min():.2f}",
                f"{df['Daily_Return'].mean()*100:.4f}%",
                f"{df['Daily_Return'].std()*100:.4f}%",
                f"{df['Daily_Return'].max()*100:.2f}%",
                f"{df['Daily_Return'].min()*100:.2f}%",
                f"{df['Volume'].mean()/1e6:.2f}M"
            ]
        }
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

# ── TAB 3: LSTM Prediction ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">LSTM Model — Prediction on Historical Data</div>',
                unsafe_allow_html=True)

    model, scaler = load_artifacts()

    if model is None or not use_saved_model:
        st.info("ℹ️ Pre-trained model not found. Training a fresh scaler on live data...")
        scaler = MinMaxScaler(feature_range=(0, 1))
        n      = len(df)
        train_end = int(n * 0.70)
        scaler.fit(df.iloc[:train_end][FEATURES].values)

    if model is None:
        st.warning("No saved LSTM model found at `../models/google_stock_price_lstm_model.keras`.  \n"
                   "Please run Notebook 3 first to train and save the model.")
    else:
        with st.spinner("Running LSTM predictions..."):
            scaled = scale_data(df, scaler)
            X, y   = build_sequence(scaled, SEQUENCE_SIZE)
            y_pred = model.predict(X, verbose=0)
            y_inv      = inverse_open(y.reshape(-1,1), scaler)
            y_pred_inv = inverse_open(y_pred,          scaler)
            pred_dates  = df['Date'].values[SEQUENCE_SIZE:]

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae  = mean_absolute_error(y_inv, y_pred_inv)
        rmse = np.sqrt(mean_squared_error(y_inv, y_pred_inv))
        r2   = r2_score(y_inv, y_pred_inv)
        mape = np.mean(np.abs((y_inv - y_pred_inv) / y_inv)) * 100

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("MAE",       f"{currency}{mae:.2f}")
        mc2.metric("RMSE",      f"{currency}{rmse:.2f}")
        mc3.metric("R² Score",  f"{r2:.4f}")
        mc4.metric("MAPE",      f"{mape:.2f}%")

        # Show last N days of predictions
        show_n = st.slider("Show last N days", 60, len(pred_dates), 365, key="pred_days")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pred_dates[-show_n:], y=y_inv[-show_n:],
                                   name='Actual Price',    line=dict(color='steelblue', width=2)))
        fig4.add_trace(go.Scatter(x=pred_dates[-show_n:], y=y_pred_inv[-show_n:],
                                   name='LSTM Prediction', line=dict(color='#2a9d8f',  width=1.5, dash='dash')))
        fig4.update_layout(title=f"{ticker} — LSTM Actual vs Predicted",
                           xaxis_title='Date', yaxis_title=f'Open Price ({currency})',
                           paper_bgcolor='white', plot_bgcolor='#fafafa', height=450,
                           legend=dict(orientation='h', yanchor='bottom', y=1.01))
        st.plotly_chart(fig4, use_container_width=True)

        

# ── TAB 4: Future Forecast ─────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">🔮 Future Price Forecast</div>', unsafe_allow_html=True)

    model, scaler = load_artifacts()
    if model is None:
        st.warning("No saved model found. Run Notebook 3 to train and save the model first.")
    else:
        days = st.slider("Forecast days into the future", 7, 90, 30, key="fc_days") \
               if mode != "🔮 Future Forecast" else forecast_days

        with st.spinner(f"Forecasting next {days} trading days..."):
            sc_local = scaler
            if not use_saved_model:
                sc_local = MinMaxScaler(feature_range=(0, 1))
                sc_local.fit(df.iloc[:int(len(df)*0.70)][FEATURES].values)

            scaled_all  = sc_local.transform(df[FEATURES].values)
            last_seq    = scaled_all[-SEQUENCE_SIZE:]
            fc_prices   = forecast_future(model, last_seq, days, sc_local)
            last_date   = df['Date'].iloc[-1]
            fc_dates    = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)

        # Context: last 60 actual days
        ctx_df = df.tail(60)
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=ctx_df['Date'], y=ctx_df['Close'],
                                   name='Historical Close', line=dict(color='steelblue', width=2)))
        fig5.add_trace(go.Scatter(x=fc_dates, y=fc_prices,
                                   name=f'{days}-Day Forecast',
                                   line=dict(color='#e9c46a', width=2.5),
                                   mode='lines+markers', marker=dict(size=5)))
        fig5.add_trace(go.Scatter(
            x=list(fc_dates) + list(reversed(fc_dates)),
            y=list(fc_prices * 1.03) + list(reversed(fc_prices * 0.97)),
            fill='toself', fillcolor='rgba(233,196,106,0.15)',
            line=dict(color='rgba(0,0,0,0)'), name='±3% Band'))
        fig5.add_vline(x=str(last_date.date()), line_dash='dot', line_color='gray')
        fig5.update_layout(title=f"{ticker} — {days}-Day Future Price Forecast",
                           xaxis_title='Date', yaxis_title=f'Price ({currency})',
                           paper_bgcolor='white', plot_bgcolor='#fafafa', height=450,
                           legend=dict(orientation='h', yanchor='bottom', y=1.01))
        st.plotly_chart(fig5, use_container_width=True)

        # Forecast table
        fc_table = pd.DataFrame({
            'Date':            [d.strftime('%Y-%m-%d') for d in fc_dates],
            f'Predicted Open ({currency})': [f"{p:.2f}" for p in fc_prices],
            'Change (%)':      [f"{((fc_prices[i]-fc_prices[i-1])/fc_prices[i-1]*100):+.2f}%" if i > 0
                                 else "—" for i in range(len(fc_prices))]
        })

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast Start", f"{currency}{fc_prices[0]:.2f}")
        c2.metric("Forecast End",   f"{currency}{fc_prices[-1]:.2f}",
                  delta=f"{((fc_prices[-1]-fc_prices[0])/fc_prices[0]*100):+.2f}%")
        c3.metric("Peak Forecast",  f"{currency}{fc_prices.max():.2f}")

        st.markdown("**Day-by-Day Forecast Table**")
        st.dataframe(fc_table, use_container_width=True, hide_index=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Stock Price Prediction</small></center>",
    unsafe_allow_html=True
)
