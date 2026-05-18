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
    for col in ['likes', 'comments', 'shares', 'plays']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['hashtag_count'] = df['hashtags'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    df['engagement'] = ((df['likes'] + df['comments']) / df['plays']) * 100
    return df

df = load_data()

st.title("📊 TikTok Analytics")
st.caption(f"Analyzing {len(df):,} posts")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Posts", f"{len(df):,}")
col2.metric("Avg Views", f"{df['plays'].mean():,.0f}")
col3.metric("Avg Likes", f"{df['likes'].mean():,.0f}")
col4.metric("Engagement", f"{df['engagement'].mean():.2f}%")

st.markdown("---")
st.subheader("⏰ Best Time to Post")

hourly = df.groupby('hour')['plays'].mean().reset_index()
fig = px.bar(hourly, x='hour', y='plays', title="Views by Hour")
st.plotly_chart(fig, use_container_width=True)

best_hour = hourly.loc[hourly['plays'].idxmax(), 'hour']
st.success(f"**{int(best_hour)}:00** is your best time to post!")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
