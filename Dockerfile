FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY proxy.py .
COPY static/ ./static/

# Create cache directory
RUN mkdir -p /tmp/hls-cache && chmod 777 /tmp/hls-cache

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "proxy.py"]
