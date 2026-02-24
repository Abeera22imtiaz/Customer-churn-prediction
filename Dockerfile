# 1. Use official Python 3.12 slim image
FROM python:3.12-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Copy only requirements.txt first for caching
COPY requirements.txt .

# 4. Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the entire project
COPY . .

# 6. Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Run the FastAPI app with uvicorn
CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]