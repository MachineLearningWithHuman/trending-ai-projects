# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
# Get the current credentials
session = get_active_session()

st.title("📈 Live & Historical Trade Dashboard")

# Fetch live orders
query = "SELECT * FROM trade_orders ORDER BY trade_date DESC LIMIT 20;"
df = session.sql(query).collect()
df = pd.DataFrame(df)

# Display live orders
st.subheader("📌 Latest Trade Orders")
st.dataframe(df)

# Historical analysis
st.subheader("📊 Historical Trade Analysis")
offset_time = st.text_input("offset_time(Minutes)")


if st.button("Fetch Historical Trades"):
    hist_query = f"CALL get_historical_trades({offset_time});"
    hist_df = session.sql(hist_query).collect()
    hist_df = pd.DataFrame(hist_df)
    st.dataframe(hist_df)


    # Fetch trading data
    query = "SELECT trade_date, symbol, trade_type, quantity, price FROM trade_orders;"
    df = session.sql(query).to_pandas()
    
    # Convert trade_date to datetime
    df["TRADE_DATE"] = pd.to_datetime(df["TRADE_DATE"])
    
    # Streamlit App
    st.title("📊 Trading Analytics Dashboard")

    # Plot 1: Trade Volume Over Time
    st.subheader("📈 Trade Volume Over Time")
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby("TRADE_DATE")["QUANTITY"].sum().plot(kind="line", marker="o", ax=ax, color="blue")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Trade Volume")
    st.pyplot(fig)
    
    # Plot 2: Stock Price Movements
    st.subheader("💹 Stock Price Movements")
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="TRADE_DATE:T",
        y="PRICE:Q",
        color="SYMBOL:N"
    ).properties(width=700, height=400)
    st.altair_chart(chart, use_container_width=True)
    
    # Plot 3: Trade Type Distribution
    st.subheader("📊 Trade Type Distribution")
    trade_counts = df["TRADE_TYPE"].value_counts()
    fig, ax = plt.subplots()
    trade_counts.plot(kind="bar", color=["green", "red"], ax=ax)
    ax.set_xlabel("Trade Type")
    ax.set_ylabel("Number of Trades")
    st.pyplot(fig)
    
    # Plot 4: Stock-wise Trade Volume
    st.subheader("📊 Stock-wise Trade Volume")
    bar_chart = alt.Chart(df).mark_bar().encode(
        x="SYMBOL:N",
        y="sum(QUANTITY):Q",
        color="SYMBOL:N"
    ).properties(width=600, height=400)
    st.altair_chart(bar_chart, use_container_width=True)

    st.success("✅ Dashboard Updated with Real-Time Data!")

    st.title("📊 Advanced Trading Dashboard")
    hist_query = f"CALL get_trade_analysis();"
    df = session.sql(hist_query).collect()
    df = pd.DataFrame(df)
    st.dataframe(df)
    # 🔹 Volatility Analysis
    st.subheader("📈 Stock Price Volatility")
    fig, ax = plt.subplots(figsize=(8, 4))
    df.groupby("TRADE_DATE")["PRICE"].std().plot(kind="line", marker="o", ax=ax, color="red")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price Volatility")
    st.pyplot(fig)
    
    # 🔹 Order Flow Imbalance (Buy vs. Sell Volume)
    st.subheader("📊 Order Flow Imbalance")
    order_counts = df["TRADE_TYPE"].value_counts()
    fig, ax = plt.subplots()
    order_counts.plot(kind="bar", color=["green", "red"], ax=ax)
    ax.set_xlabel("Trade Type")
    ax.set_ylabel("Number of Trades")
    st.pyplot(fig)
    
    # 🔹 Moving Average Trends (SMA)
    st.subheader("📊 10-day Moving Average")
    # 🔹 Calculate 10-day Simple Moving Average (SMA)
    df["SMA_10"] = df.groupby("SYMBOL")["PRICE"].transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    
    # 🔹 Create Plotly Line Chart
    fig = px.line(df, x="TRADE_DATE", y="SMA_10", color="SYMBOL",
                  title="📊 10-day Moving Average", labels={"SMA_10": "10-day SMA", "TRADE_DATE": "Date"},
                  template="plotly_dark")
    
    # 🔹 Display in Streamlit
    #st.subheader("📊 10-day Moving Average")
    st.plotly_chart(fig, use_container_width=True)
    
    # 🔹 Trade Slippage vs. Market Price (Scatter Plot)
    st.subheader("📊 Trade Slippage vs. Market Price")
    fig, ax = plt.subplots()
    ax.scatter(df["MARKET_PRICE"], df["SLIPPAGE"], c="blue", alpha=0.5)
    ax.set_xlabel("Market Price")
    ax.set_ylabel("Slippage")
    st.pyplot(fig)
    
    st.success("✅ Dashboard Updated with Complex Trading Insights!")