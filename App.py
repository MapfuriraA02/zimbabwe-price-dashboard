#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

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
st.markdown("Interactive dashboard for tracking, analyzing, and forecasting commodity prices.")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("prices.csv")
df['Date'] = pd.to_datetime(df['Date'])

# -------------------------------
# DATA UPLOAD (SIMULATED REAL-TIME)
# -------------------------------
st.sidebar.subheader("Upload New Data")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    df = pd.concat([df, new_data], ignore_index=True)
    st.sidebar.success("New data uploaded successfully!")

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("Filter Options")

items = st.sidebar.multiselect(
    "Select Items",
    df["Item"].unique(),
    default=[df["Item"].unique()[0]]
)

# -------------------------------
# FILTER DATA
# -------------------------------
filtered = df[df["Item"].isin(items)].copy()
filtered = filtered.sort_values("Date")

# -------------------------------
# METRICS
# -------------------------------
latest_data = filtered.groupby("Item").last().reset_index()

col1, col2 = st.columns(2)

with col1:
    st.metric("Tracked Items", len(items))

with col2:
    st.metric("Total Records", len(filtered))

st.markdown("---")

# -------------------------------
# INTERACTIVE PLOT
# -------------------------------
fig = px.line(
    filtered,
    x="Date",
    y="Price",
    color="Item",
    markers=True,
    title="Price Trends Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# ML FORECAST SECTION
# -------------------------------
st.markdown("---")
st.subheader("Price Forecast")

selected_item = st.selectbox("Select Item for Prediction", items)

item_df = df[df["Item"] == selected_item].copy()
item_df = item_df.sort_values("Date")

# Feature engineering
item_df["Date_ordinal"] = item_df["Date"].map(lambda x: x.toordinal())

X = item_df["Date_ordinal"].values.reshape(-1, 1)
y = item_df["Price"].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Future predictions
future_dates = pd.date_range(start=item_df["Date"].max(), periods=4, freq='MS')[1:]
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

predictions = model.predict(future_ordinals)

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predictions
})

# Plot forecast
forecast_fig = px.line(title=f"{selected_item} Forecast")

forecast_fig.add_scatter(
    x=item_df["Date"],
    y=item_df["Price"],
    mode='lines+markers',
    name="Actual"
)

forecast_fig.add_scatter(
    x=pred_df["Date"],
    y=pred_df["Predicted Price"],
    mode='lines+markers',
    name="Predicted"
)

st.plotly_chart(forecast_fig, use_container_width=True)

# -------------------------------
# USER INPUT PREDICTION
# -------------------------------
st.markdown("### Predict Custom Price")

future_date_input = st.date_input("Select a future date")

if st.button("Predict Price"):
    future_ordinal = np.array([[future_date_input.toordinal()]])
    predicted_price = model.predict(future_ordinal)[0]

    st.success(f"Predicted price on {future_date_input}: {predicted_price:.2f}")

# -------------------------------
# INSIGHTS
# -------------------------------
latest_price = item_df.iloc[-1]["Price"]
avg_price = item_df["Price"].mean()

if latest_price > avg_price:
    st.warning("Current price is above average")
else:
    st.success("Current price is within normal range")

# -------------------------------
# TABLES
# -------------------------------
st.markdown("---")

st.subheader("Data Table")
st.dataframe(filtered)

# Price changes
filtered["Change"] = filtered.groupby("Item")["Price"].diff()

st.subheader("Price Changes")
st.dataframe(filtered)

st.subheader("Predictions Table")
st.dataframe(pred_df)

# -------------------------------
# DOWNLOAD BUTTON
# -------------------------------
st.markdown("---")

csv = filtered.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name="price_data.csv",
    mime="text/csv"
)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with Python, Streamlit, and Machine Learning")
