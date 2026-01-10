# Deployment Instructions for GreenSight

This project is now configured for easy deployment using Docker, or separate deployment to cloud services like Render (Backend) and Vercel (Frontend).

## Prerequisites
1.  **Docker Desktop**: Install Docker Desktop if you want to run proper containerized environments locally.
2.  **Git**: You need Git installed to push changes to GitHub.
3.  **Google Earth Engine Credentials**: You need a service account JSON file for Google Earth Engine.

## 1. Running Locally with Docker Compose

1.  Make sure you have your Google Earth Engine credentials available as a JSON string.
2.  Create a `.env` file in this directory with the following content:
    ```
    GOOGLE_CREDENTIALS_JSON='{"type": "service_account", ...your_json_content_here...}'
    ```
3.  Run the application:
    ```bash
    docker-compose up --build
    ```
4.  Open `http://localhost:3000` for the Frontend.
    Open `http://localhost:8501` for the Backend.

## 2. Deploying "Live"

### Backend (Render.com)
1.  Push this repository to GitHub.
2.  Create a new Web Service on Render connected to your repo.
3.  **Root Directory**: `Backend`
4.  **Runtime**: Python 3
5.  **Build Command**: `pip install -r requirements.txt`
6.  **Start Command**: `python ndvi_app.py`
7.  **Environment Variables**:
    - `PYTHON_VERSION`: `3.9.18`
    - `GOOGLE_CREDENTIALS_JSON`: Paste your service account JSON content here.

### Frontend (Vercel)
1.  Push this repository to GitHub.
2.  Import the project into Vercel.
3.  **Root Directory**: `Frontend`
4.  **Environment Variables**:
    - `NEXT_PUBLIC_API_URL`: The URL of your deployed Render backend (e.g., `https://greensight-backend.onrender.com`).

## 3. Pushing Changes to GitHub

Since I (the AI) cannot access your Git credentials directly, you need to run these commands in your terminal to update your repository:

```bash
# Initialize git if not already done (it seems you have downloaded files, so it might need re-init)
git init
git remote add origin https://github.com/rahuldj001/GreenSight.git

# Add all files
git add .

# Commit changes
git commit -m "Add deployment configuration (Docker, Cloud setup)"

# Push to main branch
git push -u origin main
```

## Troubleshooting
- **Missing Model**: The file `Backend/unet_deforestation_model (2).h5` appears to be a Git LFS pointer (small file). You need the actual large model file for predictions to work accurately. If you have the file, replace the pointer with the actual file before building. The app will use a fallback/mock prediction if the model is missing.
- **Git Command Not Found**: If you see this error, please install Git for Windows from https://git-scm.com/download/win.
