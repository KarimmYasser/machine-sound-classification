# Use a base image with Python
FROM python:3.11-slim

# Install system dependencies for Librosa, XGBoost, and Rust (for fast_features)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust toolchain to build the fast_features extension
RUN curl --proto '=https' --sv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Build the Rust feature extraction extension
RUN cd fast_features && pip install .

# Hugging Face Spaces specific configuration
# The default port is 7860
EXPOSE 7860

# Set environment variable for Gradio to run on the correct port
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Run the Gradio app
CMD ["python", "app.py"]
