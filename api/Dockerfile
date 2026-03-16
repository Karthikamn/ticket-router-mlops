FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow==2.11.3 fastapi uvicorn pydantic scikit-learn
COPY app.py /app/app.py
ENV MLFLOW_TRACKING_URI="http://mlflow.ticket-mlops.svc.cluster.local:5000"
ENV MODEL_URI="models:/ticket-router/Production"
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
