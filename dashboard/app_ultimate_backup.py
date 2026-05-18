import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

warnings.filterwarnings('ignore')

# Add function definitions here

# ============================================
# PAGE CONFIGURATION & CUSTOM CSS
# ============================================
st.set_page_config(
    page_title="TikTok Analytics Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    [data-testid="stMetricLabel"] {
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }
    h1, h2, h3 {
        color: #1d3557;
    }
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING & PROCESSING
# ============================================
@st.cache_data
def load_sample_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'tiktok_posts.csv')
    return pd.read_csv(data_path)

@st.cache_data
def process_data(df):
    """Standard cleaning and feature engineering"""
    for col in ['likes', 'comments', 'shares', 'plays']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'create_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
    elif 'posted_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['posted_time'], errors='coerce')
    else:
        df['datetime'] = pd.Timestamp.now()

    df = df.dropna(subset=['datetime', 'plays'])
    df = df.sort_values('datetime')

    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_num'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['week'] = df['datetime'].dt.isocalendar().week

    if 'hashtags' in df.columns:
        df['hashtag_count'] = df['hashtags'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    else:
        df['hashtag_count'] = 0

    if 'description' in df.columns:
        df['caption_length'] = df['description'].fillna('').apply(len)
    else:
        df['caption_length'] = 0

    df['engagement_rate'] = ((df['likes'] + df['comments'] + df['shares']) / df['plays']) * 100
    df = df.dropna(subset=['plays'])
    return df

# ============================================
# UI: DATA SOURCE SELECTION
# ============================================
st.title("TikTok Analytics Pro")
st.markdown("*Professional performance analytics with AI forecasting*")

# Data Source Selection
st.sidebar.title("Data Source")
data_option = st.sidebar.radio(
    "Choose your data source:",
    ["Use Sample Data (7,225 posts)", "Upload Your Own CSV"]
)

if data_option == "Upload Your Own CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload TikTok CSV Export",
        type=['csv'],
        help="How to export: TikTok App → Profile → Settings → Privacy → Download Data"
    )
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        df = process_data(raw_df)
        st.sidebar.success(f"Loaded {len(df):,} of YOUR posts!")
    else:
        st.sidebar.info("Upload your TikTok data for personalized insights")
        st.sidebar.caption("Using sample data")
        df = process_data(load_sample_data())
else:
    df = process_data(load_sample_data())

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
min_views = st.sidebar.slider("Minimum Views", 0, int(df['plays'].max()), 0, step=50000)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect("Days of Week", options=day_order, default=day_order)

mask = (df['plays'] >= min_views) & (df['day_of_week'].isin(selected_days))
filtered_df = df[mask].copy()
daily_ts = df.groupby(df['datetime'].dt.date).agg({
    'plays': 'mean',
    'hashtag_count': 'mean',
    'likes': 'mean',
    'comments': 'mean',
    'engagement_rate': 'mean'
}).reset_index()
daily_ts.columns = ['date', 'avg_views', 'avg_hashtags', 'avg_likes', 'avg_comments', 'avg_engagement']
daily_ts['date'] = pd.to_datetime(daily_ts['date'])
daily_ts = daily_ts.sort_values('date')

st.caption(f"Analyzing {len(filtered_df):,} posts from sample data")

# KEY METRICS
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Posts", f"{len(filtered_df):,}")
col2.metric("Total Views", f"{filtered_df['plays'].sum():,.0f}")
col3.metric("Avg Views", f"{filtered_df['plays'].mean():,.0f}")
col4.metric("Avg Likes", f"{filtered_df['likes'].mean():.0f}")
col5.metric("Engagement", f"{filtered_df['engagement_rate'].mean():.2f}%")

st.markdown("---")
st.header("When to Post")
c1, c2 = st.columns(2)

with c1:
    hourly = filtered_df.groupby('hour')['plays'].mean().reset_index()
    fig1 = px.bar(hourly, x='hour', y='plays', title="Views by Hour", color='plays')
    st.plotly_chart(fig1, use_container_width=True)
    best_hour = hourly.loc[hourly['plays'].idxmax(), 'hour']
    st.info(f"Best time: **{int(best_hour)}:00**")

with c2:
    daily = filtered_df.groupby('day_of_week')['plays'].mean().reindex(day_order).reset_index()
    fig2 = px.bar(daily, x='day_of_week', y='plays', title="Views by Day", color='plays')
    st.plotly_chart(fig2, use_container_width=True)
    best_day = daily.loc[daily['plays'].idxmax(), 'day_of_week']
    st.success(f"Best day: **{best_day}**")

st.markdown("---")
st.header("AI-Powered View Forecast (Random Forest)")

if daily_ts is not None and len(daily_ts) >= 14:
    with st.spinner("Training forecasting model on your daily data..."):
        model, features, daily_df, mape, mae, importance, eval_data = train_random_forest(daily_ts)

    # Show model metrics
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric("Model Accuracy", f"{100 - mape:.1f}%", delta=None, help="Mean Absolute Percentage Error on test set")
    col_metric2.metric("Mean Absolute Error", f"{mae:,.0f} views", help="Average prediction error on test set")

    # Feature importance chart
    fig_imp = px.bar(importance.head(8), x='importance', y='feature', orientation='h',
                   title="What Drives Your Views? (Feature Importance)", color='importance')
    st.plotly_chart(fig_imp, use_container_width=True)

    # Forecasting controls
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        forecast_days = st.slider("Days to forecast", 3, 30, 7)
        if st.button("Generate Forecast", type="primary", use_container_width=True):
            with st.spinner(f"Generating {forecast_days}-day forecast..."):
                predictions = predict_future(model, features, daily_df, forecast_days)
                if predictions is not None and not predictions.empty:
                    # Chart
                    fig_forecast = go.Figure()
                    # Historical data (last 30 days)
                    hist = daily_df.tail(30)
                    fig_forecast.add_trace(go.Scatter(x=hist['date'], y=hist['avg_views'], mode='lines+markers', name='Historical', line=dict(color='blue')))
                    fig_forecast.add_trace(go.Scatter(x=predictions['date'], y=predictions['predicted_views'], mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
                    # Simple confidence interval (±20% of forecast)
                    lower = predictions['predicted_views'] * 0.8
                    upper = predictions['predicted_views'] * 1.2
                    fig_forecast.add_trace(go.Scatter(x=predictions['date'].tolist() + predictions['date'].tolist()[::-1],
                                                   y=upper.tolist() + lower.tolist()[::-1],
                                                   fill='toself', fillcolor='rgba(255,0,0,0.2)', line=dict(color='rgba(255,0,0,0)'),
                                                   name='80% Confidence Interval'))
                    fig_forecast.update_layout(title=f"{forecast_days}-Day View Forecast", xaxis_title="Date", yaxis_title="Predicted Views", height=500)
                    st.plotly_chart(fig_forecast, use_container_width=True)

                    # Show daily predictions table
                    pred_df = predictions.copy()
                    pred_df['date'] = pred_df['date'].dt.strftime('%Y-%m-%d')
                    pred_df.columns = ['Date', 'Predicted Views', 'Lower Bound', 'Upper Bound']
                    st.dataframe(pred_df, use_container_width=True)

                    # Save to memory option
                    if st.button("Save Forecast to CodeBrain"):
                        os.makedirs(os.path.expanduser("~/CodeBrain/obsidian"), exist_ok=True)
                        with open(os.path.expanduser("~/CodeBrain/obsidian/Forecast.md"), 'a') as f:
                            f.write(f"\n## Forecast - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                            f.write(pred_df.to_markdown())
                        st.success("Forecast saved to ~/CodeBrain/obsidian/Forecast.md")
                else:
                    st.warning("Could not generate forecast. Need at least 14 days of data.")
    with col_f2:
        st.caption("""
        **How it works:**
        - Daily views are aggregated from your posts
        - Random Forest model uses lag features (1,3,7 days), rolling averages, and content metrics
        - Model performance is evaluated on a 20% holdout set
        - Confidence interval represents ±20% around the forecast
        """)

st.markdown("---")
st.header("Hashtag Strategy")
hashtag_perf = filtered_df.groupby('hashtag_count')['plays'].mean().reset_index()
hashtag_perf = hashtag_perf[hashtag_perf["hashtag_count"] <= 20]
fig3 = px.line(hashtag_perf, x='hashtag_count', y='plays', title="Optimal Hashtags", markers=True)
st.plotly_chart(fig3, use_container_width=True)
best_hashtags = hashtag_perf.loc[hashtag_perf['plays'].idxmax(), 'hashtag_count']
st.info(f"Optimal hashtags: **{int(best_hashtags)}** (current avg: {filtered_df['hashtag_count'].mean():.0f})")

st.markdown("---")
st.header("Top Performing Posts")
top_posts = filtered_df.nlargest(10, 'plays')[['author', 'description', 'plays', 'likes', 'comments', 'hashtag_count']].copy()
top_posts.columns = ['Author', 'Description', 'Views', 'Likes', 'Comments', 'Hashtags']
top_posts['Description'] = top_posts['Description'].str[:60] + '...'
st.dataframe(top_posts, use_container_width=True)

st.markdown("---")
st.header("Export & Memory")
col_e1, col_e2 = st.columns(2)
with col_e1:
    if st.button("Download Report (CSV)"):
        csv = filtered_df.to_csv(index=False)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(label="Click to Download", data=csv, file_name=f"tiktok_export_{timestamp}.csv", mime="text/csv")

with col_e2:
    if st.button("Save Insights to CodeBrain"):
        insights = f"""## Session: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Data source: sample data
- Posts analyzed: {len(filtered_df):,}
- Avg views: {filtered_df['plays'].mean():,.0f}
- Engagement: {filtered_df['engagement_rate'].mean():.2f}%
- Best time: {int(best_hour)}:00
- Best day: {best_day}
- Optimal hashtags: {int(best_hashtags)}
"""
        os.makedirs(os.path.expanduser("~/CodeBrain/obsidian"), exist_ok=True)
        with open(os.path.expanduser("~/CodeBrain/obsidian/TikTok_Insights.md"), 'a') as f:
            f.write(insights + "\n")
        st.success("Insights saved to CodeBrain!")

st.markdown("---")
st.caption(f"TikTok Analytics Pro | Built with Streamlit + Random Forest | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
# ============================================
# MISSING FUNCTIONS (ADDED)
# ============================================
def create_daily_features(df):
    if len(df) < 2:
        return None
    daily = df.groupby(df['datetime'].dt.date).agg({
        'plays': 'mean',
        'hashtag_count': 'mean',
        'likes': 'mean',
        'comments': 'mean',
        'engagement_rate': 'mean'
    }).reset_index()
    daily.columns = ['date', 'avg_views', 'avg_hashtags', 'avg_likes', 'avg_comments', 'avg_engagement']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['day_of_year'] = daily['date'].dt.dayofyear
    daily['month'] = daily['date'].dt.month
    daily['lag1'] = daily['avg_views'].shift(1)
    daily['lag3'] = daily['avg_views'].shift(3)
    daily['lag7'] = daily['avg_views'].shift(7)
    daily['rolling7_mean'] = daily['avg_views'].rolling(7, min_periods=1).mean()
    daily['rolling7_std'] = daily['avg_views'].rolling(7, min_periods=1).std().fillna(0)
    daily = daily.dropna().reset_index(drop=True)
    return daily

def train_random_forest(daily_df, test_size=0.2):
    features = ['day_of_week', 'day_of_year', 'month', 'lag1', 'lag3', 'lag7', 
                'rolling7_mean', 'avg_hashtags', 'avg_engagement']
    target = 'avg_views'
    X = daily_df[features]
    y = daily_df[target]
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return model, features, daily_df, mape, mae, importance, (X_test, y_test, y_pred)

def predict_future(model, features, daily_df, days=7):
    if daily_df is None or len(daily_df) < 14:
        return None
    last_date = daily_df['date'].max()
    last_row = daily_df.iloc[-1:].copy()
    predictions = []
    current_df = daily_df.copy()
    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        pred_features = {
            'day_of_week': future_date.weekday(),
            'day_of_year': future_date.timetuple().tm_yday,
            'month': future_date.month,
            'lag1': last_row['avg_views'].values[0],
            'lag3': current_df['avg_views'].iloc[-3] if len(current_df) >= 3 else last_row['avg_views'].values[0],
            'lag7': current_df['avg_views'].iloc[-7] if len(current_df) >= 7 else last_row['avg_views'].values[0],
            'rolling7_mean': current_df['avg_views'].tail(7).mean(),
            'avg_hashtags': daily_df['avg_hashtags'].mean(),
            'avg_engagement': daily_df['avg_engagement'].mean()
        }
        input_df = pd.DataFrame([pred_features])[features]
        pred = model.predict(input_df)[0]
        predictions.append({'date': future_date, 'predicted_views': int(pred)})
        new_row = last_row.copy()
        new_row['date'] = future_date
        new_row['avg_views'] = pred
        for col in ['lag1', 'lag3', 'lag7', 'rolling7_mean']:
            if col in new_row.columns:
                new_row[col] = pred_features.get(col, pred)
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        last_row = new_row
    return pd.DataFrame(predictions)

def create_daily_features(df):
    if len(df) < 2:
        return None
    daily = df.groupby(df['datetime'].dt.date).agg({
        'plays': 'mean',
        'hashtag_count': 'mean',
        'likes': 'mean',
        'comments': 'mean',
        'engagement_rate': 'mean'
    }).reset_index()
    daily.columns = ['date', 'avg_views', 'avg_hashtags', 'avg_likes', 'avg_comments', 'avg_engagement']
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.sort_values('date')
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['day_of_year'] = daily['date'].dt.dayofyear
    daily['month'] = daily['date'].dt.month
    daily['lag1'] = daily['avg_views'].shift(1)
    daily['lag3'] = daily['avg_views'].shift(3)
    daily['lag7'] = daily['avg_views'].shift(7)
    daily['rolling7_mean'] = daily['avg_views'].rolling(7, min_periods=1).mean()
    daily['rolling7_std'] = daily['avg_views'].rolling(7, min_periods=1).std().fillna(0)
    daily = daily.dropna().reset_index(drop=True)
    return daily

def train_random_forest(daily_df, test_size=0.2):
    features = ['day_of_week', 'day_of_year', 'month', 'lag1', 'lag3', 'lag7', 
                'rolling7_mean', 'avg_hashtags', 'avg_engagement']
    target = 'avg_views'
    X = daily_df[features]
    y = daily_df[target]
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    mae = mean_absolute_error(y_test, y_pred)
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    return model, features, daily_df, mape, mae, importance, (X_test, y_test, y_pred)

def predict_future(model, features, daily_df, days=7):
    if daily_df is None or len(daily_df) < 14:
        return None
    last_date = daily_df['date'].max()
    last_row = daily_df.iloc[-1:].copy()
    predictions = []
    current_df = daily_df.copy()
    for i in range(1, days + 1):
        future_date = last_date + timedelta(days=i)
        pred_features = {
            'day_of_week': future_date.weekday(),
            'day_of_year': future_date.timetuple().tm_yday,
            'month': future_date.month,
            'lag1': last_row['avg_views'].values[0],
            'lag3': current_df['avg_views'].iloc[-3] if len(current_df) >= 3 else last_row['avg_views'].values[0],
            'lag7': current_df['avg_views'].iloc[-7] if len(current_df) >= 7 else last_row['avg_views'].values[0],
            'rolling7_mean': current_df['avg_views'].tail(7).mean(),
            'avg_hashtags': daily_df['avg_hashtags'].mean(),
            'avg_engagement': daily_df['avg_engagement'].mean()
        }
        input_df = pd.DataFrame([pred_features])[features]
        pred = model.predict(input_df)[0]
        predictions.append({'date': future_date, 'predicted_views': int(pred)})
        new_row = last_row.copy()
        new_row['date'] = future_date
        new_row['avg_views'] = pred
        for col in ['lag1', 'lag3', 'lag7', 'rolling7_mean']:
            if col in new_row.columns:
                new_row[col] = pred_features.get(col, pred)
        current_df = pd.concat([current_df, new_row], ignore_index=True)
        last_row = new_row
    return pd.DataFrame(predictions)

