ARG PYTHON_VERSION=3.13
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir .

CMD ["python", "run_all.py"]
