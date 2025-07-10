# ---- Stage 1: Build environment ----
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install system build tools (required for some Python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependencies file
COPY requirements.txt .

# Install dependencies into /install directory
RUN pip install --upgrade pip
RUN pip install --prefix=/install -r requirements.txt

# ---- Stage 2: Final image ----
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /install /usr/local

# Copy your FastAPI project files
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Run the app using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]