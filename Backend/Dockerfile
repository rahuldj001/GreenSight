FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8501

# Define environment variable
ENV PORT=8501
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/service_account.json

# Run the application
CMD ["python", "ndvi_app.py"]
