FROM python:3.10-slim
WORKDIR /app
COPY logs_exporter.py .
RUN pip install flask
CMD ["python", "logs_exporter.py"]
