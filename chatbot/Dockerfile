# chatbot_/Dockerfile

FROM python:3-slim

WORKDIR /app
COPY ./src/ /app

# COPY ./src/entrypoint.sh /app/entrypoint.sh

COPY ./pyproject.toml /code/pyproject.toml
RUN apt-get update && apt-get install -y gcc g++ build-essential && rm -rf /var/lib/apt/lists/*
RUN pip install --timeout=120 /code/.

# Make entrypoint.sh executable
# RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]