FROM python:3.8-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt setup.py ./
COPY src ./src

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps .

COPY application.py ./
COPY templates ./templates
COPY artifacts ./artifacts

EXPOSE 5000

CMD ["python", "application.py"]
