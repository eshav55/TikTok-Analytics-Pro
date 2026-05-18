import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'data', 'tiktok_posts.csv')
df = pd.read_csv(data_path)

# Clean data
df['likes'] = pd.to_numeric(df['likes'], errors='coerce')
df['comments'] = pd.to_numeric(df['comments'], errors='coerce')
df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
df['plays'] = pd.to_numeric(df['plays'], errors='coerce')

# Parse datetime
df['datetime'] = pd.to_datetime(df['create_time'], errors='coerce')
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.day_name()
df['date'] = df['datetime'].dt.date

# Calculate metrics
df['total_engagement'] = df['likes'] + df['comments'] + df['shares']
df['engagement_rate'] = (df['total_engagement'] / df['plays']) * 100
df['hashtag_count'] = df['hashtags'].fillna('').apply(lambda x: len(str(x).split(',')) if x else 0)

# Generate insights
print("="*60)
print("📊 TIKTOK ANALYTICS REPORT")
print("="*60)

# 1. Best posting times
hourly = df.groupby('hour')['plays'].mean().sort_values(ascending=False)
print(f"\n⏰ BEST TIME TO POST: {hourly.index[0]}:00")
print(f"   (gets {hourly.iloc[0]/df['plays'].mean():.1f}x more views than average)")

# 2. Best day
daily = df.groupby('day_of_week')['plays'].mean().sort_values(ascending=False)
print(f"\n📅 BEST DAY TO POST: {daily.index[0]}")

# 3. Hashtag strategy
hashtag_perf = df.groupby('hashtag_count')['plays'].mean()
best_hashtag_count = hashtag_perf.idxmax()
print(f"\n🏷️ OPTIMAL HASHTAGS: {best_hashtag_count}")
print(f"   (vs {df['hashtag_count'].mean():.0f} average)")

# 4. Top performers
print(f"\n🔥 TOP 5 POSTS OF ALL TIME:")
top_posts = df.nlargest(5, 'plays')[['author', 'plays', 'likes', 'hashtag_count']]
for i, row in top_posts.iterrows():
    print(f"   • @{row['author']}: {row['plays']:,.0f} views")

# Save cleaned data
output_path = os.path.join(script_dir, '..', 'data', 'cleaned_tiktok_data.csv')
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned data saved to data/cleaned_tiktok_data.csv")