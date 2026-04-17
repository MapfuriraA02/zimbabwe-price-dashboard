#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# In[10]:


# Loading the dataset
df = pd.read_csv("prices.csv")
df['Date'] = pd.to_datetime(df['Date'])

st.title("Zimbabwe Price Intelligence Dashboard")

# Selecting an item:
item = st.selectbox("Select Item", df["Item"].unique())

# Filtering Data:
filtered = df[df["Item"]==item]

# Convert dates to numeric (for ML)
filtered = filtered.sort_values("Date")
filtered["Date_ordinal"] = filtered["Date"].map(lambda x: x.toordinal())

# Prepare data
X = filtered["Date_ordinal"].values.reshape(-1, 1)
y = filtered["Price"].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 3 months
future_dates = pd.date_range(start=filtered["Date"].max(), periods=4, freq='M')[1:]
future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)

predictions = model.predict(future_ordinals)

# Create prediction dataframe
pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predictions
})

# Plot
fig, ax = plt.subplots()

#Plot Actual Data:
ax.plot(filtered["Date"], filtered["Price"], marker='o', label='Actual')

#Plot Predicted data:
ax.plot(pred_df["Date"], pred_df["Predicted Price"], marker='x', linestyle='--', label="Predicted")

ax.set_title(f"{item} Price Trend & Forecast")
plt.xticks(rotation=45, ha='right')
ax.set_ylabel("Price")
ax.set_xlabel("Date")
ax.legend()
plt.tight_layout()

st.pyplot(fig)

# Show statistics:
st.subheader("Summary Statistics")
st.write(filtered.describe())

# Price Change:
filtered['Change'] = filtered["Price"].diff()
st.subheader("Price Changes")

st.write(filtered)

st.subheader("Future Price Predictions")
st.write(pred_df)

st.subheader("Highest Price Recorded")
st.write(filtered.loc[filtered["Price"].idxmax()])

st.subheader("Future Price Predictions")
st.write(pred_df)

