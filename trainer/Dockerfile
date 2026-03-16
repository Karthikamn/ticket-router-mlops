FROM python:3.10-slim
WORKDIR /app
RUN pip install --no-cache-dir mlflow==2.11.3 scikit-learn==1.4.2 pandas numpy
COPY train.py /app/train.py
CMD ["python", "train.py"]
