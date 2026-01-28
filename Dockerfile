FROM python:3.10-slim

# ===============================
# Set working directory
# ===============================
WORKDIR /app

# ===============================
# System dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Copy requirements (cache friendly)
# ===============================
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ===============================
# Copy entire project
# ===============================
COPY . .

# ===============================
# Streamlit configuration
# ===============================
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=7860

# ===============================
# Expose HF-required port
# ===============================
EXPOSE 7860

# ===============================
# Start Streamlit app
# ===============================
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
