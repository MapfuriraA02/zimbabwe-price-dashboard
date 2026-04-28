#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Zimbabwe Price Dashboard",
    layout="wide"
)

# -------------------------------
# TITLE
# -------------------------------
st.title("Zimbabwe Price Intelligence Dashboard")
st.markdown("Track, analyze, and forecast price trends of essential goods.")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("prices.csv")
df['Date'] = pd.to_datetime(df['Date'])

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
st.sidebar.header("Filter Options")

item = st.sidebar.selectbox(
    "Select Item",
    df["Item"].unique()
)

# -------------------------------
# FILTER DATA
# -------------------------------
filtered = df[df["Item"] == item].copy()
filtered = filtered.sort_values("Date")

# -------------------------------
# FEATURE ENGINEERING (FOR ML)
# -------------------------------
filtered["Date_ordinal"] = filtered["Date"].map(lambda x: x.toordinal())

X = filtered["Date_ordinal"].values.reshape(-1, 1)
y = filtered["Price"].values

# -------------------------------
# TRAIN MODEL
# -------------------------------
model = LinearRegression()
model.fit(X, y)

# -------------------------------
# FUTURE PREDICTIONS
# -------------------------------
future_dates = pd.date_range(start=filtered["Date"].max(), periods=4, freq='MS')[1:]
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

predictions = model.predict(future_ordinals)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predictions
})

# -------------------------------
# METRICS (UI UPGRADE)
# -------------------------------
latest = filtered.iloc[-1]
avg_price = filtered["Price"].mean()
latest_prediction = predictions[-1]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Latest Price", f"{latest['Price']}")

with col2:
    st.metric("Average Price", f"{avg_price:.2f}")

with col3:
    st.metric("Next Month Prediction", f"{latest_prediction:.2f}")

st.markdown("---")

# -------------------------------
# PLOT
# -------------------------------
fig, ax = plt.subplots()

# Actual Data
ax.plot(filtered["Date"], filtered["Price"], marker='o', label='Actual')

# Predicted Data
ax.plot(pred_df["Date"], pred_df["Predicted Price"], marker='x', linestyle='--', label="Predicted")

ax.set_title(f"{item} Price Trend & Forecast", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

st.pyplot(fig)

# -------------------------------
# INSIGHT MESSAGE
# -------------------------------
if latest["Price"] > avg_price:
    st.warning("Current price is above average")
else:
    st.success("Current price is within normal range")

st.markdown("---")

# -------------------------------
# SUMMARY STATISTICS
# -------------------------------
st.subheader("Summary Statistics")
st.dataframe(filtered.describe())

# -------------------------------
# PRICE CHANGES
# -------------------------------
filtered['Change'] = filtered["Price"].diff()

st.subheader("Price Changes")
st.dataframe(filtered)

# -------------------------------
# PREDICTIONS TABLE
# -------------------------------
st.subheader("🔮 Future Price Predictions")
st.dataframe(pred_df)

# -------------------------------
# HIGHEST PRICE
# -------------------------------
st.subheader("Highest Price Recorded")
st.write(filtered.loc[filtered["Price"].idxmax()])
