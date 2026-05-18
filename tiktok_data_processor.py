import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from typing import Dict, List, Any

class TikTokDataProcessor:
    """Professional data processor for TikTok analytics with enhanced features"""

    def __init__(self):
        self.data = None
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load TikTok data with comprehensive error handling"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

            self.data = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded {len(self.data)} records")
            return self.data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean data with proper error handling"""
        # Remove rows with missing required fields
        required_columns = ['video_id', 'create_time']
        missing_data = df[df[required_columns].isnull().any(axis=1)]
        if not missing_data.empty:
            self.logger.warning(f"Removing {len(missing_data)} rows with missing data")
            df = df.drop(missing_data.index)

        return df

    def process_tiktok_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process TikTok data with advanced analytics features"""
        # Convert create_time to datetime with error handling
        df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')

        # Sort by date
        df = df.sort_values('create_time')

        # Add time-based features
        df['day_of_week'] = df['create_time'].dt.day_name()
        df['hour'] = df['create_time'].dt.hour
        df['month'] = df['create_time'].dt.month_name()

        return df

    def calculate_advanced_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advanced metrics for TikTok analytics"""
        metrics = {
            'total_videos': len(df),
            'avg_plays': df['plays'].mean() if 'plays' in df.columns else 0,
            'avg_engagement': df[['likes', 'comments', 'shares']].sum(axis=1).mean() if 'likes' in df.columns and 'comments' in df.columns and 'shares' in df.columns else 0,
            'engagement_rate': self._calculate_engagement_rate(df)
        }
        return metrics

    def _calculate_engagement_rate(self, df: pd.DataFrame) -> float:
        """Calculate engagement rate with error handling"""
        if 'likes' in df.columns and 'plays' in df.columns:
            total_plays = df['plays'].sum()
            if total_plays > 0:
                return (df['likes'].sum() / total_plays) * 100
            return 0.0
        return 0.0