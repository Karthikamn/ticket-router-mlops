import os
import json
import time
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ticket-router-exp")
MODEL_NAME = os.getenv("MODEL_NAME", "ticket-router")
PROMOTE_TO_PROD = os.getenv("PROMOTE_TO_PROD", "true").lower() == "true"

# Tracking URI is set via env MLFLOW_TRACKING_URI
print("Using MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))

def load_data():
    # If you have data/tickets.csv mounted, use it. Else generate simple synthetic dataset.
    csv_path = "data/tickets.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        assert {"text","label"}.issubset(df.columns)
        return df

    # Synthetic mini dataset
    samples = [
        ("VPN not connecting from home", "Network"),
        ("Laptop is very slow", "End-User"),
        ("App throwing 500 error on login", "App"),
        ("Timeout when connecting to Postgres", "Database"),
        ("Received phishing email", "Security"),
        ("WiFi disconnects frequently", "Network"),
        ("Password reset not working", "End-User"),
        ("UI button click not responding", "App"),
        ("Tablespace full error", "Database"),
        ("Malware detected by antivirus", "Security"),
    ]
    df = pd.DataFrame(samples, columns=["text","label"])
    return df

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer()),
        ("clf", MultinomialNB())
    ])

    with mlflow.start_run(run_name=f"train-{int(time.time())}") as run:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_dict(report, "classification_report.json")

        # log model
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        run_id = run.info.run_id
        client = MlflowClient()

        # Find the latest version that was just created by log_model
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        # Pick the highest version
        latest_version = max(int(v.version) for v in versions)
        print("Registered model version:", latest_version)

        # Add accuracy as a tag on the model version
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(latest_version),
            key="accuracy",
            value=str(acc)
        )

        if PROMOTE_TO_PROD:
            # Archive current Production (if any) and promote this version
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=str(latest_version),
                stage="Production",
                archive_existing_versions=True
            )
            print(f"Promoted version {latest_version} to Production")

if __name__ == "__main__":
    main()
