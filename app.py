
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Apple Stock Analysis Dashboard")
st.caption("Historical prices and technical indicators")

# Load data
stock = pd.read_csv("aapl_stock_data.csv")
stock.columns = stock.columns.str.strip()

date_col = stock.columns[0]
stock[date_col] = pd.to_datetime(stock[date_col], errors="coerce")
stock["Close"] = pd.to_numeric(stock["Close"], errors="coerce")

stock = (
    stock.dropna(subset=[date_col, "Close"])
    .sort_values(date_col)
    .set_index(date_col)
)

# Calculate EMA and RSI
stock["EMA_20"] = stock["Close"].ewm(
    span=20, adjust=False
).mean()

delta = stock["Close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

avg_loss = loss.ewm(
    alpha=1/14, min_periods=14, adjust=False
).mean()

rs = avg_gain / avg_loss
stock["RSI_14"] = 100 - (100 / (1 + rs))

# Sidebar
st.sidebar.header("Dashboard Settings")
period = st.sidebar.selectbox(
    "Historical Period",
    ["1 Year", "3 Years", "5 Years"]
)

days = {"1 Year": 252, "3 Years": 756, "5 Years": len(stock)}
data = stock.tail(min(days[period], len(stock)))

# Summary cards
c1, c2, c3 = st.columns(3)
c1.metric("Stock", "AAPL")
c2.metric("Latest Close", f"${data['Close'].iloc[-1]:.2f}")
c3.metric("Trading Days", len(data))

# Price and EMA chart
st.subheader("Closing Price vs. EMA (20 days)")

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(
    x=data.index, y=data["Close"],
    name="Closing Price", mode="lines"
))
fig_price.add_trace(go.Scatter(
    x=data.index, y=data["EMA_20"],
    name="EMA (20 days)", mode="lines"
))

fig_price.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig_price, use_container_width=True)

# RSI chart
st.subheader("Relative Strength Index (14 days)")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(
    x=data.index, y=data["RSI_14"],
    name="RSI", mode="lines"
))
fig_rsi.add_hline(y=70, line_dash="dash", annotation_text="70")
fig_rsi.add_hline(y=30, line_dash="dash", annotation_text="30")

fig_rsi.update_layout(
    xaxis_title="Date",
    yaxis_title="RSI",
    yaxis_range=[0, 100],
    template="plotly_white"
)
st.plotly_chart(fig_rsi, use_container_width=True)

st.caption("Educational analysis only. Not financial advice.")

# ---------- LSTM Predictions ----------
st.divider()
st.header("🤖 LSTM Model Predictions")

try:
    results = pd.read_csv("lstm_predictions.csv")
    results["Date"] = pd.to_datetime(results["Date"], errors="coerce")
    results = results.dropna(subset=["Date"])

    st.write(
        "Compare actual closing prices with predictions from "
        "both trained LSTM models."
    )

    # Select model to display
    model_choice = st.selectbox(
        "Choose prediction model",
        [
            "Single-feature LSTM",
            "Multivariate LSTM"
        ]
    )

    prediction_column = (
        "Single_LSTM_Prediction"
        if model_choice == "Single-feature LSTM"
        else "Multivariate_LSTM_Prediction"
    )

    fig_prediction = go.Figure()

    fig_prediction.add_trace(go.Scatter(
        x=results["Date"],
        y=results["Actual_Price"],
        mode="lines",
        name="Actual Price"
    ))

    fig_prediction.add_trace(go.Scatter(
        x=results["Date"],
        y=results[prediction_column],
        mode="lines",
        name="Predicted Price"
    ))

    fig_prediction.update_layout(
        title=f"Actual vs Predicted Prices — {model_choice}",
        xaxis_title="Date",
        yaxis_title="Closing Price (USD)",
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig_prediction, use_container_width=True)

    st.caption(
        "These are historical test-set predictions, not guaranteed future prices."
    )

except FileNotFoundError:
    st.warning(
        "Prediction file not found. Save lstm_predictions.csv in the "
        "same directory as app.py."
    )
except KeyError as error:
    st.error(f"A required column is missing from the predictions file: {error}")

# ---------- Model Evaluation ----------
st.divider()
st.header("📊 Model Performance")

try:
    # Use the same actual prices for both models
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np

    actual = results["Actual_Price"].to_numpy()

    single_pred = results["Single_LSTM_Prediction"].to_numpy()
    multi_pred = results["Multivariate_LSTM_Prediction"].to_numpy()

    def evaluate_model(actual, predicted):
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(
            np.abs((actual - predicted) / actual)
        ) * 100

        return mae, rmse, mape

    single_mae, single_rmse, single_mape = evaluate_model(
        actual, single_pred
    )

    multi_mae, multi_rmse, multi_mape = evaluate_model(
        actual, multi_pred
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Single-feature LSTM")
        st.metric("MAE", f"${single_mae:.2f}")
        st.metric("RMSE", f"${single_rmse:.2f}")
        st.metric("MAPE", f"{single_mape:.2f}%")

    with col2:
        st.subheader("Multivariate LSTM")
        st.metric("MAE", f"${multi_mae:.2f}")
        st.metric("RMSE", f"${multi_rmse:.2f}")
        st.metric("MAPE", f"{multi_mape:.2f}%")

    comparison = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "MAPE (%)"],
        "Single-feature LSTM": [
            single_mae, single_rmse, single_mape
        ],
        "Multivariate LSTM": [
            multi_mae, multi_rmse, multi_mape
        ]
    })

    st.subheader("Side-by-side Comparison")
    st.dataframe(comparison, use_container_width=True)

except NameError:
    st.warning(
        "Load the prediction results before displaying the metrics."
    )

# ---------- Project Summary ----------
st.divider()
st.header("📝 Project Summary")

st.markdown("""
### Objective
Predict Apple's next-day closing stock price using
Long Short-Term Memory (LSTM) neural networks.

### Models Developed
- **Single-feature LSTM:** Uses historical closing prices.
- **Multivariate LSTM:** Uses closing prices, 20-day EMA,
  and 14-day RSI.

### Evaluation
Both models are evaluated using:
- **MAE:** Mean Absolute Error, measured in dollars.
- **RMSE:** Root Mean Squared Error, which penalizes
  larger errors more heavily.
- **MAPE:** Mean Absolute Percentage Error.

### Key Finding
In our current experiment, the single-feature model achieved
lower reported prediction errors than the multivariate model.
Adding more input features does not automatically improve
prediction accuracy.

### Limitations
- Stock prices can change unexpectedly due to market events.
- Historical patterns do not guarantee future performance.
- Model results depend on the dataset, preprocessing,
  training settings, and evaluation period.
""")

st.info(
    "This dashboard is an educational machine-learning project, "
    "not a financial advisory tool."
)
