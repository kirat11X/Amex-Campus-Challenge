FROM python:3.12-slim
WORKDIR /app

COPY profitlens/requirements.txt profitlens/requirements.txt
RUN pip install --no-cache-dir -r profitlens/requirements.txt

COPY . .

# Chat + embeddings are served by the `ollama` service in docker-compose.yml
ENV OLLAMA_HOST=http://ollama:11434
WORKDIR /app/profitlens
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
