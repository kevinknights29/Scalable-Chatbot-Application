FROM python:3.9-slim-buster

WORKDIR /app
COPY models ./models/
COPY src ./src/
COPY app.py .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80

ENV PYTHONPATH="${PYTHONPATH}:."
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
