# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# Install system deps for building packages (e.g. numpy/shap may need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="/app/models/best_model.pkl"

# Command to run the bot
CMD ["python", "-m", "src.bot"]
