# Use an AMD64-compatible Python 3.10 base image (for EC2 compatibility)
FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary files for modelv1
COPY app.py .
COPY modelv1.joblib .
COPY modelv1_threshold.txt .
COPY modelv1_features.joblib .

# Expose FastAPI's default port
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
