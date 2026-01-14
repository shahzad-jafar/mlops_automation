# Base image
FROM docker.io/library/python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Set port environment variable
ENV PORT 8003

# Run FastAPI app
CMD ["uvicorn", "src.app.server:app", "--host", "0.0.0.0", "--port", "8003"]