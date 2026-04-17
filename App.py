#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# In[10]:


# Loading the dataset
df = pd.read_csv("prices.csv")
df['Date'] = pd.to_datetime(df['Date'])

st.title("Zimbabwe Price Intelligence Dashboard")

# Selecting an item:
item = st.selectbox("Select Item", df["Item"].unique())

# Filtering Data:
filtered = df[df["Item"]==item]

# Plot
fig, ax = plt.subplots()
ax.plot(filtered["Date"], filtered["Price"], marker='o')
ax.set_title(f"{item} Price Trend")
ax.set_ylabel("Price")
ax.set_xlabel("Date")

st.pyplot(fig)

# Show statistics:
st.subheader("Summary Statistics")
st.write(filtered.describe())

# Price Change:
filtered['Change'] = filtered["Price"].diff()
st.subheader("Price Changes")

st.write(filtered)


# In[ ]:
st.subheader("Highest Price Recorded")
st.write(filtered.loc[filtered["Price"].idxmax()])


