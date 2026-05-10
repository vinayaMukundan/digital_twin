FROM python:3.10-slim

WORKDIR /app

# 1. Copy requirements first to leverage Docker caching
COPY requirements.txt .

# 2. Install everything in one layer (add lime to requirements.txt or install it here)
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir lime

# 3. Copy the rest of your application code
COPY . .

# 4. Set environment variable for Hugging Face (optional but recommended)
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
