import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from datetime import datetime

st.set_page_config(page_title="TikTok Analytics", layout="wide")

@st.cache_data
def load_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'tiktok_posts.csv')
    df = pd.read_csv(data_path)
    # Clean numeric columns
    for col in ['likes', 'comments', 'shares', 'plays']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Parse datetime
    if 'create_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
    elif 'posted_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['posted_time'], errors='coerce')
    else:
        df['datetime'] = pd.Timestamp.now()

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()

    # Content metrics
    if 'hashtags' in df.columns:
        df['hashtag_count'] = df['hashtags'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    else:
        df['hashtag_count'] = 0

    if 'description' in df.columns:
        df['caption_length'] = df['description'].fillna('').apply(len)
    else:
        df['caption_length'] = 0

    # Drop rows with NaN plays
    df = df.dropna(subset=['plays'])

    return df

# Load the data
df = load_data()

st.title("TikTok Analytics Dashboard")

# Key metrics
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Posts", f"{len(df):,}")
col2.metric("Avg Views", f"{df['plays'].mean():,.0f}")
col3.metric("Avg Likes", f"{df['likes'].mean():,.0f}")
col4.metric("Avg Comments", f"{df['comments'].mean():.0f}")
col5.metric("Avg Shares", f"{df['shares'].mean():.0f}")

st.markdown("---")

# Hourly performance chart
st.subheader("Best Times to Post")
hourly = df.groupby('hour')['plays'].mean().reset_index()
fig1 = px.bar(hourly, x='hour', y='plays', title="Views by Hour")
st.plotly_chart(fig1, use_container_width=True)

# Daily performance chart
daily = df.groupby('day_of_week')['plays'].mean().reset_index()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily = daily.reindex(day_order).reset_index()
fig2 = px.bar(daily, x='day_of_week', y='plays', title="Views by Day")
st.plotly_chart(fig2, use_container_width=True)

st.success("Dashboard loaded successfully!")