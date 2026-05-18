import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="TikTok Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# DATA LOADING WITH CSV UPLOAD
# ============================================
@st.cache_data
def load_sample_data():
    """Load sample data for demo"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'tiktok_posts.csv')
    df = pd.read_csv(data_path)
    return df

@st.cache_data
def process_data(df):
    """Process raw TikTok data into analysis-ready format"""
    # Clean numeric columns
    for col in ['likes', 'comments', 'shares', 'plays']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse datetime (try different possible column names)
    if 'create_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
    elif 'posted_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['posted_time'], errors='coerce')
    else:
        df['datetime'] = pd.Timestamp.now()
    
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_num'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['week'] = df['datetime'].dt.isocalendar().week
    df['date'] = df['datetime'].dt.date
    
    # Engagement metrics
    df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
    df['engagement_rate'] = (df['total_engagement'] / df['plays']) * 100
    
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

# ============================================
# TIME SERIES FORECASTING MODEL
# ============================================
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def train_forecast_model(df):
    """Train time series forecasting model"""
    # Prepare time series features
    df_ts = df.copy()
    df_ts = df_ts.sort_values('datetime')
    df_ts['day_of_year'] = df_ts['datetime'].dt.dayofyear
    df_ts['day_of_week'] = df_ts['datetime'].dt.dayofweek
    df_ts['month'] = df_ts['datetime'].dt.month
    
    # Create lag features (previous days' performance)
    df_ts['plays_lag1'] = df_ts['plays'].shift(1)
    df_ts['plays_lag7'] = df_ts['plays'].shift(7)
    df_ts['plays_rolling7'] = df_ts['plays'].rolling(7).mean()
    
    # Drop NaN rows from lag features
    df_ts = df_ts.dropna()
    
    if len(df_ts) < 30:
        return None, None, None
    
    features = ['day_of_year', 'day_of_week', 'month', 'hashtag_count', 'plays_lag1', 'plays_lag7']
    target = 'plays'
    
    X = df_ts[features]
    y = df_ts[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    return model, features, df_ts

def predict_future(model, features, df, days=7):
    """Predict future performance"""
    if model is None:
        return None
    
    last_date = df['datetime'].max()
    predictions = []
    
    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        
        # Get most recent data for features
        recent = df.tail(30)
        
        pred_features = pd.DataFrame([[
            future_date.timetuple().tm_yday,  # day_of_year
            future_date.weekday(),              # day_of_week
            future_date.month,                  # month
            recent['hashtag_count'].mean(),     # avg hashtags
            recent['plays'].iloc[-1],           # lag1
            recent['plays'].tail(7).mean()      # lag7
        ]], columns=features)
        
        pred = model.predict(pred_features)[0]
        predictions.append({
            'date': future_date,
            'predicted_views': int(pred),
            'lower_bound': int(pred * 0.7),
            'upper_bound': int(pred * 1.3)
        })
    
    return predictions

# ============================================
# UI: DATA SOURCE SELECTION
# ============================================
st.title("📊 TikTok Analytics Pro")
st.markdown("*Professional TikTok performance analytics with AI predictions*")

# Sidebar - Data source
st.sidebar.title("📁 Data Source")

data_option = st.sidebar.radio(
    "Choose your data source:",
    ["📊 Use Sample Data (7,225 posts)", "📤 Upload Your Own CSV"]
)

if data_option == "📤 Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload TikTok CSV Export",
        type=['csv'],
        help="How to export: TikTok App → Profile → Settings → Privacy → Download Data"
    )
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        df = process_data(raw_df)
        st.sidebar.success(f"✅ Loaded {len(df):,} of YOUR posts!")
        data_source = "your data"
    else:
        st.sidebar.info("📤 Upload your TikTok data for personalized insights")
        st.sidebar.caption("No file? Sample data will be used")
        df = process_data(load_sample_data())
        data_source = "sample data (upload yours for real insights!)"
else:
    df = process_data(load_sample_data())
    data_source = "sample data"

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

min_views = st.sidebar.slider(
    "Minimum Views",
    min_value=0,
    max_value=int(df['plays'].max()),
    value=0,
    step=50000,
    format="%d"
)

day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect(
    "Days of Week",
    options=day_order,
    default=day_order
)

# Apply filters
mask = (df['plays'] >= min_views) & (df['day_of_week'].isin(selected_days))
filtered_df = df[mask].copy()

# Train forecast model
forecast_model, forecast_features, forecast_df = train_forecast_model(df)

st.caption(f"📊 Analyzing {len(filtered_df):,} posts from {data_source}")

# ============================================
# KEY METRICS
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("📹 Total Posts", f"{len(filtered_df):,}")
with col2:
    st.metric("👁️ Total Views", f"{filtered_df['plays'].sum():,.0f}")
with col3:
    st.metric("📈 Avg Views", f"{filtered_df['plays'].mean():,.0f}")
with col4:
    st.metric("❤️ Avg Likes", f"{filtered_df['likes'].mean():,.0f}")
with col5:
    st.metric("💬 Engagement", f"{filtered_df['engagement_rate'].mean():.2f}%")

st.markdown("---")

# ============================================
# POSTING TIME ANALYSIS
# ============================================
st.header("⏰ When to Post")

col1, col2 = st.columns(2)

with col1:
    hourly = filtered_df.groupby('hour')['plays'].mean().reset_index()
    fig1 = px.bar(hourly, x='hour', y='plays', title="Views by Hour", color='plays')
    st.plotly_chart(fig1, use_container_width=True)
    best_hour = hourly.loc[hourly['plays'].idxmax(), 'hour']
    st.info(f"💡 Best time: **{int(best_hour)}:00**")

with col2:
    daily = filtered_df.groupby('day_of_week')['plays'].mean().reindex(day_order).reset_index()
    fig2 = px.bar(daily, x='day_of_week', y='plays', title="Views by Day", color='plays')
    st.plotly_chart(fig2, use_container_width=True)
    best_day = daily.loc[daily['plays'].idxmax(), 'day_of_week']
    st.success(f"💡 Best day: **{best_day}**")

st.markdown("---")

# ============================================
# TIME SERIES FORECASTING (NEW!)
# ============================================
st.header("🔮 AI-Powered View Forecast")

if forecast_model is not None and len(filtered_df) > 30:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        forecast_days = st.slider("Days to forecast", 3, 30, 7)
        st.caption(f"Based on {len(forecast_df)} days of historical data")
        
        if st.button("📈 Generate Forecast", type="primary"):
            predictions = predict_future(forecast_model, forecast_features, filtered_df, forecast_days)
            
            if predictions:
                # Create forecast chart
                forecast_df_plot = pd.DataFrame(predictions)
                
                fig_forecast = go.Figure()
                
                # Add historical data (last 30 days)
                hist_data = filtered_df.tail(30).copy()
                fig_forecast.add_trace(go.Scatter(
                    x=hist_data['datetime'],
                    y=hist_data['plays'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add forecast
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df_plot['date'],
                    y=forecast_df_plot['predicted_views'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add confidence interval
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df_plot['date'].tolist() + forecast_df_plot['date'].tolist()[::-1],
                    y=forecast_df_plot['upper_bound'].tolist() + forecast_df_plot['lower_bound'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name='Confidence Interval (70%)'
                ))
                
                fig_forecast.update_layout(
                    title=f'{forecast_days}-Day View Forecast',
                    xaxis_title='Date',
                    yaxis_title='Predicted Views',
                    height=500
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                # Show prediction table
                st.subheader("📊 Daily Predictions")
                pred_df = forecast_df_plot[['date', 'predicted_views', 'lower_bound', 'upper_bound']].copy()
                pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
                pred_df.columns = ['Date', 'Predicted Views', 'Lower Bound', 'Upper Bound']
                st.dataframe(pred_df, use_container_width=True)
                
                # Save forecast to memory
                if st.button("💾 Save Forecast to Memory"):
                    with open(os.path.expanduser("~/CodeBrain/obsidian/Forecast.md"), 'a') as f:
                        f.write(f"\n## Forecast {datetime.now().strftime('%Y-%m-%d')}\n")
                        f.write(pred_df.to_markdown())
                    st.success("✅ Forecast saved to ~/CodeBrain/ossian/Forecast.md")
            else:
                st.info("Click 'Generate Forecast' to see predictions")
        else:
            st.warning("⚠️ Need at least 30 days of data for reliable forecasting. Upload more data or use sample data.")

st.markdown("---")

# ============================================
# HASHTAG ANALYSIS
# ============================================
st.header("🏷️ Hashtag Strategy")

hashtag_perf = filtered_df.groupby('hashtag_count')['plays'].mean().reset_index()
hashtag_perf = hashtag_perf[hashtag_perf['hashtag_count'] <= 20]
fig3 = px.line(hashtag_perf, x='hashtag_count', y='plays', title="Optimal Hashtags", markers=True)
st.plotly_chart(fig3, use_container_width=True)

best_hashtags = hashtag_perf.loc[hashtag_perf['plays'].idxmax(), 'hashtag_count']
st.info(f"💡 Optimal hashtags: **{int(best_hashtags)}** (current avg: {filtered_df['hashtag_count'].mean():.0f})")

st.markdown("---")

# ============================================
# TOP POSTS TABLE
# ============================================
st.header("🏆 Top Performing Posts")

top_posts = filtered_df.nlargest(10, 'plays')[['author', 'description', 'plays', 'likes', 'comments', 'hashtag_count']].copy()
top_posts.columns = ['Author', 'Description', 'Views', 'Likes', 'Comments', 'Hashtags']
top_posts['Description'] = top_posts['Description'].str[:50] + '...'

st.dataframe(top_posts, use_container_width=True)

st.markdown("---")

# ============================================
# EXPORT & MEMORY
# ============================================
st.header("📎 Export & Memory")

col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Download Report (CSV)"):
        csv = filtered_df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name=f"tiktok_export_{timestamp}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🧠 Save Insights to CodeBrain"):
        insights = f"""## Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Data source: {data_source}
- Posts analyzed: {len(filtered_df):,}
- Avg views: {filtered_df['plays'].mean():,.0f}
- Engagement: {filtered_df['engagement_rate'].mean():.2f}%
- Best time: {int(best_hour)}:00
- Best day: {best_day}
- Optimal hashtags: {int(best_hashtags)}
"""
        import os
        os.makedirs(os.path.expanduser("~/CodeBrain/obsidian"), exist_ok=True)
        with open(os.path.expanduser("~/CodeBrain/obsidian/TikTok_Insights.md"), 'a') as f:
            f.write(insights + "\n")
        st.success("✅ Insights saved to CodeBrain!")

st.markdown("---")
st.caption(f"📊 TikTok Analytics Pro | Built with Streamlit + AI | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
