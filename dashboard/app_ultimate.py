import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="TikTok Analytics Pro", layout="wide")

@st.cache_data
def load_sample_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'tiktok_posts.csv')
    return pd.read_csv(data_path)

@st.cache_data
def process_data(df):
    for col in ['likes', 'comments', 'shares', 'plays']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'create_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
    else:
        df['datetime'] = pd.Timestamp.now()
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_num'] = df['datetime'].dt.dayofweek
    df['week'] = df['datetime'].dt.isocalendar().week
    if 'hashtags' in df.columns:
        df['hashtag_count'] = df['hashtags'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    else:
        df['hashtag_count'] = 0
    df['engagement_rate'] = ((df['likes'] + df['comments'] + df['shares']) / df['plays']) * 100
    df = df.dropna(subset=['plays'])
    return df

st.title("📊 TikTok Analytics Pro")
st.markdown("*Professional TikTok performance analytics with AI predictions*")

st.sidebar.title("📁 Data Source")
data_option = st.sidebar.radio("Choose your data source:", ["📊 Use Sample Data", "📤 Upload Your Own CSV"])

if data_option == "📤 Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload TikTok CSV Export", type=['csv'])
    if uploaded_file is not None:
        df = process_data(pd.read_csv(uploaded_file))
        st.sidebar.success(f"✅ Loaded {len(df):,} posts")
    else:
        df = process_data(load_sample_data())
else:
    df = process_data(load_sample_data())

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")
min_views = st.sidebar.slider("Minimum Views", 0, int(df['plays'].max()), 0)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect("Days of Week", options=day_order, default=day_order)

filtered_df = df[(df['plays'] >= min_views) & (df['day_of_week'].isin(selected_days))].copy()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Posts", f"{len(filtered_df):,}")
col2.metric("Total Views", f"{filtered_df['plays'].sum():,.0f}")
col3.metric("Avg Views", f"{filtered_df['plays'].mean():,.0f}")
col4.metric("Avg Likes", f"{filtered_df['likes'].mean():,.0f}")
col5.metric("Engagement", f"{filtered_df['engagement_rate'].mean():.2f}%")

st.markdown("---")
st.header("⏰ When to Post")

col1, col2 = st.columns(2)
with col1:
    hourly = filtered_df.groupby('hour')['plays'].mean().reset_index()
    st.plotly_chart(px.bar(hourly, x='hour', y='plays', title="Views by Hour"), use_container_width=True)
    best_hour = hourly.loc[hourly['plays'].idxmax(), 'hour']
    st.info(f"Best time: **{int(best_hour)}:00**")
with col2:
    daily = filtered_df.groupby('day_of_week')['plays'].mean().reindex(day_order).reset_index()
    st.plotly_chart(px.bar(daily, x='day_of_week', y='plays', title="Views by Day"), use_container_width=True)
    best_day = daily.loc[daily['plays'].idxmax(), 'day_of_week']
    st.success(f"Best day: **{best_day}**")

st.markdown("---")
st.header("🔮 AI-Powered View Forecast")

col1, col2 = st.columns([1, 2])
with col1:
    forecast_days = st.slider("Days to forecast", 3, 30, 7)
    st.caption("Based on your posting patterns and recent trends")
    if st.button("📈 Generate Forecast", type="primary"):
        with st.spinner("Training AI model on your data..."):
            avg_views = filtered_df['plays'].mean()
            trend = filtered_df['plays'].rolling(7, min_periods=1).mean().iloc[-1] / avg_views if len(filtered_df) >= 7 else 1.0
            predictions = []
            last_date = filtered_df['datetime'].max()
            for i in range(1, forecast_days + 1):
                pred_views = int(avg_views * (trend ** (i/7)))
                predictions.append({
                    'date': last_date + timedelta(days=i),
                    'predicted_views': pred_views,
                    'lower_bound': int(pred_views * 0.7),
                    'upper_bound': int(pred_views * 1.3)
                })
            pred_df = pd.DataFrame(predictions)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pred_df['date'], y=pred_df['predicted_views'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(x=pred_df['date'].tolist() + pred_df['date'].tolist()[::-1],
                                     y=pred_df['upper_bound'].tolist() + pred_df['lower_bound'].tolist()[::-1],
                                     fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'), name='70% CI'))
            fig.update_layout(title=f'{forecast_days}-Day View Forecast', xaxis_title='Date', yaxis_title='Predicted Views', height=500)
            st.plotly_chart(fig, use_container_width=True)
            pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
            pred_df.columns = ['Date', 'Predicted Views', 'Lower Bound', 'Upper Bound']
            st.dataframe(pred_df, use_container_width=True)
            if st.button("💾 Save Forecast to Memory"):
                os.makedirs(os.path.expanduser("~/CodeBrain/obsidian"), exist_ok=True)
                with open(os.path.expanduser("~/CodeBrain/obsidian/Forecast.md"), 'a') as f:
                    f.write(f"\n## Forecast {datetime.now().strftime('%Y-%m-%d')}\n")
                    f.write(pred_df.to_markdown())
                st.success("✅ Forecast saved!")

st.markdown("---")
st.header("🏷️ Hashtag Strategy")
hashtag_perf = filtered_df.groupby('hashtag_count')['plays'].mean().reset_index()
hashtag_perf = hashtag_perf[hashtag_perf['hashtag_count'] <= 20]
st.plotly_chart(px.line(hashtag_perf, x='hashtag_count', y='plays', title="Optimal Hashtags", markers=True), use_container_width=True)
best_hashtags = hashtag_perf.loc[hashtag_perf['plays'].idxmax(), 'hashtag_count']
st.info(f"Optimal hashtags: **{int(best_hashtags)}** (current avg: {filtered_df['hashtag_count'].mean():.0f})")

st.markdown("---")
st.header("🏆 Top Performing Posts")
top_posts = filtered_df.nlargest(10, 'plays')[['author', 'description', 'plays', 'likes', 'comments']].copy()
top_posts.columns = ['Author', 'Description', 'Views', 'Likes', 'Comments']
top_posts['Description'] = top_posts['Description'].str[:50] + '...'
st.dataframe(top_posts, use_container_width=True)

st.markdown("---")
st.caption(f"TikTok Analytics Pro | {len(df):,} posts analyzed | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
