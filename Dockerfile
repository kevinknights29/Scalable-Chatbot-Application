FROM python:3.9

WORKDIR /app
ADD . /app

RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 80

ENV PYTHONPATH="${PYTHONPATH}:."
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
