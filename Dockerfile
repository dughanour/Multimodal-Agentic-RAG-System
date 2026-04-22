# Lightweight Linux image with Python 3.13
FROM python:3.13-slim

# Set /app as the working directory inside the container
WORKDIR /app

# Copy requirements first (for Docker layer caching) to speed up builds if dependencies don't change
COPY requirements.txt .

# Install Python dependencies (mount pip cache so re-installs reuse downloaded packages)
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

# Copy the rest of the project files
COPY . .

# Document the port the app listens on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]