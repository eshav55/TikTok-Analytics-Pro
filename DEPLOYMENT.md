# Deployment Guide

## Deploying to Streamlit Community Cloud

To deploy this application to Streamlit Community Cloud:

1. Create a GitHub repository with your code:
   - Go to github.com and create a new repository
   - Name it "TikTok-Dashboard" or similar
   - Don't initialize with README (we'll push our existing code)

2. Push your code to GitHub:
   ```bash
   git remote set-url origin https://github.com/yourusername/TikTok-Dashboard.git
   git branch -M main
   git push -u origin main
   ```

3. Deploy to Streamlit Community Cloud:
   - Go to https://share.streamlit.io/
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository
   - Set the main file path to: `dashboard/app_broken.py`
   - Click "Deploy" and your app will be live at yourname.streamlit.app

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
- Full-stack development capabilities