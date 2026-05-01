# Use a base image with Python
FROM python:3.11-slim

# Install system dependencies for Librosa, XGBoost, and Rust (for fast_features)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain to build the fast_features extension
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the Rust feature extraction extension
# Assuming maturin or setuptools-rust is used in the project
# Based on the file structure, we need to build the package in /app/fast_features
RUN cd fast_features && pip install .

# Set the entrypoint for inference
# Usage: docker run -v /path/to/data:/data <image_name> /data
ENTRYPOINT ["python", "infer.py"]
