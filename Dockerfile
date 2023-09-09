FROM python:3.9-slim-buster

WORKDIR /app
COPY models ./models/
COPY src ./src/
COPY app.py .
COPY requirements.txt .

RUN apt update && \
    apt install -y \
        gcc \
        python3-dev

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH="${PYTHONPATH}:."

EXPOSE 80

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
