# Stage 1: Build the application and install dependencies
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Stage 2: Create the final image
FROM python:3.9-slim

WORKDIR /app

# Copy the wheels from the builder stage and install them
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the application code
COPY . .

# Pre-download and cache the sentence-transformer model and NLTK data
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Set the entrypoint for the container
ENTRYPOINT ["python", "src/main.py"]
