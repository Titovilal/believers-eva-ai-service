FROM python:3.13-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml /app/

# Install dependencies using pip with PyTorch CPU index
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -e .

# Copy application code
COPY main.py /app/
COPY src /app/src

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port (Railway uses dynamic PORT env var)
EXPOSE 8000

# Run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
