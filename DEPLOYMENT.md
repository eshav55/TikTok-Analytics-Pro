# Deployment Guide

## Deploying to Streamlit Community Cloud

To deploy this application to Streamlit Community Cloud:

1. Create a GitHub repository with your code
2. Sign in to Streamlit Community Cloud (https://streamlit.io/cloud)
3. Connect your GitHub account
4. Select your repository
5. Set the main file path to: `dashboard/app_broken.py`
6. Click "Deploy" and your app will be live at yourname.streamlit.app

## Environment Variables

The application requires the following Python packages:
- streamlit
- pandas
- numpy
- plotly
- scikit-learn

These are specified in the `requirements.txt` file.

## Configuration

The `.streamlit/config.toml` file contains:
- Professional light theme styling
- Server configuration for optimal performance

## For Recruiters

This deployment-ready setup demonstrates:
- Cloud deployment experience
- Professional configuration management
- Production-ready code organization
- Understanding of deployment pipelines