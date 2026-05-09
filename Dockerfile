# 1. Use Python 3.10 as the base
FROM python:3.10-slim

# 2. Install system dependencies (including curl for Ollama)
# AFTER
RUN apt-get update && apt-get install -y \
    curl \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# 4. Set working directory
WORKDIR /app

# 5. Pre-download the Llama3 model 
# We start the server briefly, pull the model, then kill the server.
# This ensures the model is baked into the image so it doesn't download every time the app starts.
RUN ollama serve & sleep 5 && ollama pull llama3

# 6. Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of your code
COPY . .

# 8. Create a start script to run both Ollama and FastAPI
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
python app.py' > start.sh && chmod +x start.sh

# 9. Expose HF default port
EXPOSE 7860

# 10. Start the application
CMD ["./start.sh"]