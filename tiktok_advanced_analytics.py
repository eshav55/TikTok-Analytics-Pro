import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import logging

class AdvancedAnalytics:
    """Professional analytics module with advanced features for recruiter appeal"""

    def __init__(self):
        self.model = None
        self.logger = logging.getLogger(__name__)

    def train_engagement_predictor(self, df):
        """Train a machine learning model to predict content engagement"""
        try:
            # Prepare features for engagement prediction
            df_processed = self._process_data(df)

            # Select only numeric columns for modeling
            numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
            df_numeric = df_processed[numeric_cols]

            if len(df_numeric) == 0 or len(df_numeric.columns) < 2:
                self.logger.warning("Insufficient data for training")
                return None

            # Prepare features and target
            feature_cols = ['hour', 'day_num', 'hashtag_count', 'caption_length']
            target = 'plays'

            # Filter to only existing columns
            available_features = [col for col in feature_cols if col in df_numeric.columns]
            X = df_numeric[available_features] if all(col in df_numeric.columns for col in feature_cols) else df_numeric.select_dtypes(include=[np.number])
            y = df_numeric[target] if target in df_numeric.columns else df_numeric.iloc[:, 0] * 0

            # Train model
            if len(X.columns) > 0 and len(y) > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
                self.model.fit(X_train, y_train)

                # Calculate model performance
                y_pred = self.model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)

                return {
                    'model': self.model,
                    'mae': mae,
                    'X_test': X_test,
                    'y_test': y_test
                }
        except Exception as e:
            self.logger.error(f"Error in train_engagement_predictor: {str(e)}")
            return None

    def _process_data(self, df):
        """Clean and prepare data for analysis"""
        # Remove rows with missing data
        df_clean = df.dropna(subset=['video_id', 'create_time'])

        # Convert data types
        for col in ['likes', 'comments', 'shares', 'plays']:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # Add time features
        df_clean['hour'] = pd.to_datetime(df_clean['create_time']).dt.hour
        df_clean['day_num'] = pd.to_datetime(df_clean['create_time']).dt.dayofweek
        df_clean['day_of_week'] = pd.to_datetime(df_clean['create_time']).dt.day_name()

        return df_clean