# Streamlit Deployment Guide

## 🚀 Quick Deploy to Streamlit Community Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at https://streamlit.io/cloud)
- This repository pushed to GitHub

### Step 1: Prepare Your Repository

Make sure these files are committed to GitHub:
- ✅ `app.py` - Your Streamlit application
- ✅ `requirements.txt` - Dependencies
- ✅ All source code in `src/` folder
- ✅ `train.py` - For training the model
- ❌ Model files (`.pkl`) - These are excluded via `.gitignore` (too large)

### Step 2: Deploy Options

#### Option A: Train Model During Deployment (Easiest) ⭐

1. Go to https://streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set Main file path to: `app.py`
6. Deploy!

**Note:** The app will automatically train the model on first run (takes ~2-3 minutes).

#### Option B: Use Pre-trained Model (Faster Startup)

Since model files are large (~33MB), you have two options:

**B1 - Git LFS (Large File Storage):**
```bash
# Install Git LFS
git lfs install

# Track model files
git lfs track "models/*.pkl"

# Add and commit
git add .gitattributes models/*.pkl
git commit -m "Add model files via Git LFS"
git push origin main
```

**B2 - Cloud Storage (Recommended for Production):**
Upload models to AWS S3 or Google Drive and download them at runtime.

### Step 3: Configure Streamlit Cloud

Create a `packages.txt` file if you need system dependencies:
```
# Example: if you need specific system libraries
# libgl1
```

Create a `.streamlit/config.toml` file:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#31333F"
font = "sans serif"

[server]
maxUploadSize = 200
```

### Step 4: Environment Variables (if needed)

If your app needs API keys or secrets:
1. Go to Streamlit Cloud dashboard
2. Click on your app
3. Go to Settings → Secrets
4. Add your secrets in TOML format:
```toml
[api_keys]
key = "your-api-key"
```

### Step 5: Deploy!

1. Go to https://streamlit.io/cloud
2. Click "New app"
3. Fill in the details:
   - **Repository**: your-username/credit_card_fraud_detection
   - **Branch**: main
   - **Main file path**: app.py
4. Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

## 📊 Local Testing

Before deploying, test locally:

```bash
# Make sure you're in the project directory
cd /Users/felix.onoberevune/Documents/CREDIT_CARD_FRAUD_DETECTION

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the app
streamlit run app.py
```

## 🔧 Troubleshooting

### Issue: "Model not found!"
- Run `python train.py` locally first
- Or let Streamlit train it during deployment (first run will take longer)

### Issue: Memory Error
- Streamlit Cloud free tier has 1GB RAM
- The model training might exceed this
- Solution: Train locally and use Git LFS for model files

### Issue: App is slow
- The model loads on every interaction
- It's cached with `@st.cache_resource` decorator
- This is expected behavior

## 📁 Files Structure

```
credit_card_fraud_detection/
├── app.py                 # Streamlit app (entry point)
├── requirements.txt       # Python dependencies
├── train.py              # Model training script
├── startup.py            # Optional: startup script
├── src/                  # Source code
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── models/               # Model files (excluded from git)
│   └── .gitkeep
├── data/                 # Data files (excluded from git)
└── reports/              # Evaluation reports
```

## 🌟 Alternative Deployment Options

### Heroku
Useful for continuous deployment, but has file size limits.

### AWS EC2 / Google Cloud Run
For production deployments with more resources.

### Docker
Containerize your app for consistent deployment anywhere.

## 📞 Support

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Forum: https://discuss.streamlit.io/
- Deployment issues: Check the "Manage app" logs in Streamlit Cloud

## 🎯 Next Steps

1. Push your code to GitHub
2. Go to https://streamlit.io/cloud
3. Deploy your app!
4. Share the URL with others

Good luck! 🚀
