FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
