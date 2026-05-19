import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import warnings

# Add the parent directory to the Python path
sys.path.append('/Users/eshavijay/Documents/obsvault/TikTok-Dashboard')

from tiktok_data_processor import TikTokDataProcessor
from tiktok_advanced_analytics import AdvancedAnalytics
import sys
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import warnings
from tiktok_data_processor import TikTokDataProcessor
from tiktok_advanced_analytics import AdvancedAnalytics

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="TikTok Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# PROFESSIONAL CSS
# ============================================
st.markdown("""
<style>
    /* Professional typography */
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Metric card styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Insight box styling */
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #0066cc;
        margin: 0.5rem 0;
        color: #333333;
    }

    /* Professional heading spacing */
    h1 {
        margin-bottom: 0rem;
    }

    h2 {
        font-size: 1.3rem;
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    /* Divider styling */
    hr {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA LOADING
# ============================================
@st.cache_data
def load_data():
    # Use our professional data processor
    data_path = '/Users/eshavijay/Documents/obsvault/TikTok-Dashboard/data/tiktok_posts.csv'

    # Use our professional data processor
    processor = TikTokDataProcessor()

    # Load and process data with professional validation
    df = processor.load_data(data_path)
    df = processor.validate_and_clean_data(df)
    df = processor.process_tiktok_data(df)

    # Clean numeric columns
    for col in ['likes', 'comments', 'shares', 'plays']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Parse datetime (already done in process_tiktok_data but keeping for compatibility)
    df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_num'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['date'] = df['datetime'].dt.date

    # Engagement metrics
    df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
    df['engagement_rate'] = (df['total_engagement'] / df['plays']) * 100

    # Content metrics
    df['hashtag_count'] = df['hashtags'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)
    df['caption_length'] = df['description'].fillna('').apply(len)

    # Drop rows with NaN plays for ML
    df = df.dropna(subset=['plays'])

    return df

@st.cache_resource
def train_model(df):
    """Train ML model for view prediction"""
    features = ['hour', 'day_num', 'month', 'hashtag_count', 'caption_length']
    
    X = df[features].fillna(0)
    y = np.log1p(df['plays'])
    
    # Remove any remaining NaN/inf values
    mask = ~(np.isnan(y) | np.isinf(y))
    X = X[mask]
    y = y[mask]
    
    if len(X) == 0:
        return None, features, 0, pd.DataFrame()
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    predictions = np.expm1(model.predict(X))
    actual = np.expm1(y)
    accuracy = 1 - (np.abs(predictions - actual).mean() / actual.mean())
    
    return model, features, accuracy, importance_df

# ============================================
# LOAD DATA
# ============================================
df = load_data()
model, features, model_accuracy, feature_importance = train_model(df)

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.title("Filters")
st.sidebar.markdown("---")

# Date range filter
min_date = df['datetime'].min().date()
max_date = df['datetime'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# View filter
min_views = st.sidebar.slider(
    "Minimum Views",
    min_value=0,
    max_value=int(df['plays'].max()),
    value=0,
    step=50000,
    format="%d"
)

# Day selection
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
selected_days = st.sidebar.multiselect(
    "Days of Week",
    options=day_order,
    default=day_order
)

# Hashtag filter
min_hashtags = st.sidebar.slider("Minimum Hashtags", 0, 20, 0)

# Apply filters
mask = (df['datetime'].dt.date >= date_range[0]) & \
       (df['datetime'].dt.date <= date_range[1]) & \
       (df['plays'] >= min_views) & \
       (df['day_of_week'].isin(selected_days)) & \
       (df['hashtag_count'] >= min_hashtags)

filtered_df = df[mask].copy()

# Export functionality
st.sidebar.markdown("---")
st.sidebar.subheader("Export Data")

if st.sidebar.button("Download CSV"):
    csv = filtered_df.to_csv(index=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    st.sidebar.download_button(
        label="Confirm Download",
        data=csv,
        file_name=f"tiktok_export_{timestamp}.csv",
        mime="text/csv"
    )

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("TikTok Analytics Dashboard")
st.caption(f"Analyzing {len(filtered_df):,} posts from {date_range[0]} to {date_range[1]}")

st.markdown("---")

# ============================================
# KEY METRICS
# ============================================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Posts", f"{len(filtered_df):,}")
with col2:
    st.metric("Total Views", f"{filtered_df['plays'].sum():,.0f}")
with col3:
    st.metric("Average Views", f"{filtered_df['plays'].mean():,.0f}")
with col4:
    st.metric("Average Likes", f"{filtered_df['likes'].mean():,.0f}")
with col5:
    st.metric("Average Hashtags", f"{filtered_df['hashtag_count'].mean():.1f}")

st.markdown("---")

# ============================================
# SECTION 1: POSTING TIME ANALYSIS
# ============================================
st.header("Posting Time Analysis")

col1, col2 = st.columns(2)

with col1:
    if len(filtered_df) > 0:
        hourly = filtered_df.groupby('hour')['plays'].mean().reset_index()
        fig1 = px.bar(
            hourly, 
            x='hour', 
            y='plays',
            title="Average Views by Hour of Day",
            labels={'hour': 'Hour (24h)', 'plays': 'Average Views'},
            color='plays',
            color_continuous_scale='Blues'
        )
        fig1.add_hline(y=filtered_df['plays'].mean(), line_dash="dash", line_color="gray")
        fig1.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        best_hour = hourly.loc[hourly['plays'].idxmax(), 'hour']
        st.markdown(f'<div class="insight-box">Best posting time: <strong>{int(best_hour)}:00</strong> ({(hourly["plays"].max() / filtered_df["plays"].mean() - 1) * 100:.0f}% above average)</div>', unsafe_allow_html=True)

with col2:
    if len(filtered_df) > 0:
        daily = filtered_df.groupby('day_of_week')['plays'].mean().reset_index()
        fig2 = px.bar(
            daily,
            x='day_of_week',
            y='plays',
            title="Average Views by Day of Week",
            color='plays',
            color_continuous_scale='Blues'
        )
        fig2.add_hline(y=filtered_df['plays'].mean(), line_dash="dash", line_color="gray")
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        best_day = daily.loc[daily['plays'].idxmax(), 'day_of_week']
        st.markdown(f'<div class="insight-box">Best posting day: <strong>{best_day}</strong></div>', unsafe_allow_html=True)

st.markdown("---")

# ============================================
# SECTION 2: CONTENT STRATEGY
# ============================================
st.header("Content Strategy Analysis")

col1, col2 = st.columns(2)

with col1:
    if len(filtered_df) > 0:
        hashtag_perf = filtered_df.groupby('hashtag_count')['plays'].mean().reset_index()
        hashtag_perf = hashtag_perf[hashtag_perf['hashtag_count'] <= 20]
        
        fig3 = px.line(
            hashtag_perf,
            x='hashtag_count',
            y='plays',
            title="Impact of Hashtag Count on Views",
            labels={'hashtag_count': 'Number of Hashtags', 'plays': 'Average Views'},
            markers=True
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
        
        best_hashtags = hashtag_perf.loc[hashtag_perf['plays'].idxmax(), 'hashtag_count']
        st.markdown(f'<div class="insight-box">Optimal hashtag count: <strong>{int(best_hashtags)}</strong> (current average: {filtered_df["hashtag_count"].mean():.1f})</div>', unsafe_allow_html=True)

with col2:
    if len(filtered_df) > 0:
        all_hashtags = filtered_df['hashtags'].dropna().str.split(',').explode()
        all_hashtags = all_hashtags.str.strip()
        all_hashtags = all_hashtags[all_hashtags != '']
        top_hashtags = all_hashtags.value_counts().head(10).reset_index()
        top_hashtags.columns = ['hashtag', 'count']
        
        fig4 = px.bar(
            top_hashtags,
            x='count',
            y='hashtag',
            orientation='h',
            title="Most Frequently Used Hashtags",
            labels={'count': 'Usage Count', 'hashtag': ''},
            color='count',
            color_continuous_scale='Blues'
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 3: ENGAGEMENT ANALYSIS
# ============================================
st.header("Engagement Analysis")

if len(filtered_df) > 0:
    fig5 = px.histogram(
        filtered_df,
        x='engagement_rate',
        nbins=40,
        title="Distribution of Engagement Rates",
        labels={'engagement_rate': 'Engagement Rate (%)', 'count': 'Number of Posts'},
        color_discrete_sequence=['#2c3e50']
    )
    fig5.add_vline(x=filtered_df['engagement_rate'].mean(), line_dash="dash", line_color="red",
                   annotation_text=f"Mean: {filtered_df['engagement_rate'].mean():.2f}%")
    fig5.update_layout(height=450)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 4: TOP PERFORMING POSTS
# ============================================
st.header("Top Performing Posts")

if len(filtered_df) > 0:
    top_posts = filtered_df.nlargest(10, 'plays')[[
        'author', 'description', 'plays', 'likes', 'comments', 'shares', 'hour', 'day_of_week'
    ]].copy()
    
    top_posts.columns = ['Author', 'Description', 'Views', 'Likes', 'Comments', 'Shares', 'Hour', 'Day']
    top_posts['Description'] = top_posts['Description'].str[:60] + '...'
    
    st.dataframe(
        top_posts,
        use_container_width=True,
        column_config={
            "Views": st.column_config.NumberColumn(format="%d"),
            "Likes": st.column_config.NumberColumn(format="%d"),
            "Comments": st.column_config.NumberColumn(format="%d"),
            "Shares": st.column_config.NumberColumn(format="%d"),
        }
    )

# ============================================
# SECTION 5: AI-POWERED INSIGHTS
# ============================================
st.header("🔮 AI-Powered Insights")

# Initialize advanced analytics
analytics = AdvancedAnalytics()
model_results = analytics.train_engagement_predictor(df)

if model_results:
    st.markdown("### Engagement Prediction Model")
    st.success(f"Model trained successfully! MAE: {model_results['mae']:.2f}")

    # Show feature importance if available
    if 'model' in model_results and hasattr(model_results['model'], 'feature_importances_'):
        st.markdown("### Feature Importance for Engagement")
        feature_names = ['hour', 'day_num', 'hashtag_count', 'caption_length']
        importance_scores = model_results['model'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)

        fig = px.bar(importance_df, x='Feature', y='Importance',
                    title="What Drives TikTok Engagement?",
                    color='Importance', color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)

        # Show insights
        top_feature = importance_df.iloc[0]['Feature']
        st.info(f"**Key Insight**: {top_feature.replace('_', ' ').title()} is the most important factor for engagement.")
else:
    st.warning("Not enough data to train prediction model.")

st.markdown("---")

# ============================================
# SECTION 5: PERFORMANCE PREDICTOR
# ============================================
st.header("Performance Predictor")

if model is not None and len(filtered_df) > 0:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Post Parameters")
        
        best_hour_val = int(filtered_df.groupby('hour')['plays'].mean().idxmax())
        predict_hour = st.selectbox("Hour", range(24), index=best_hour_val)
        predict_day = st.selectbox("Day", day_order)
        predict_hashtags = st.slider("Hashtag Count", 0, 15, 6)
        
        if st.button("Predict Views", type="primary", use_container_width=True):
            day_map = {day: i for i, day in enumerate(day_order)}
            input_data = pd.DataFrame([[
                predict_hour,
                day_map[predict_day],
                5,
                predict_hashtags,
                120
            ]], columns=features)
            
            pred_log = model.predict(input_data)[0]
            pred_views = int(np.expm1(pred_log))
            
            st.markdown(f"""
            <div class="insight-box">
                <p style="font-size: 1.2rem; margin: 0;">Predicted Views</p>
                <p style="font-size: 2rem; font-weight: 600; margin: 0;">{pred_views:,}</p>
                <p style="font-size: 0.8rem; margin-top: 0.5rem;">Model confidence: {model_accuracy:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Key Performance Drivers")
        
        if len(feature_importance) > 0:
            fig6 = px.bar(
                feature_importance.head(5),
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Impact on View Count",
                labels={'importance': 'Relative Importance', 'feature': ''},
                color='importance',
                color_continuous_scale='Blues'
            )
            fig6.update_layout(height=350)
            st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ============================================
# SECTION 6: RECOMMENDATIONS
# ============================================
st.header("Recommendations")

if len(filtered_df) > 0:
    col1, col2, col3 = st.columns(3)
    
    best_hour_val = int(filtered_df.groupby('hour')['plays'].mean().idxmax())
    best_day_val = filtered_df.groupby('day_of_week')['plays'].mean().idxmax()
    optimal_hashtags = int(filtered_df.groupby('hashtag_count')['plays'].mean().idxmax())
    
    with col1:
        st.markdown("**Posting Schedule**")
        st.markdown(f"- Post at {best_hour_val}:00 on {best_day_val}s")
        st.markdown("- Avoid posting between 22:00-04:00")
    
    with col2:
        st.markdown("**Hashtag Strategy**")
        st.markdown(f"- Use {optimal_hashtags} hashtags per post")
        st.markdown("- Mix popular and niche hashtags")
    
    with col3:
        st.markdown("**Content Focus**")
        st.markdown("- Higher engagement rates drive more views")
        st.markdown("- Caption length: 100-150 characters optimal")

st.markdown("---")

# ============================================
# FOOTER
# ============================================
st.caption(f"""
TikTok Analytics Dashboard | Data from {len(df):,} posts | Filtered: {len(filtered_df):,} posts | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
