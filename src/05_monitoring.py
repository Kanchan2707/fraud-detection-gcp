import yaml
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from google.cloud import storage, bigquery

with open("config/pipeline_config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_ID = cfg["gcp"]["project_id"]
BUCKET     = cfg["storage"]["bucket_name"]
DATASET_ID = cfg["bigquery"]["dataset_id"]

def load_model():
    client   = storage.Client(project=PROJECT_ID)
    bucket   = client.bucket(BUCKET)
    blobs    = list(bucket.list_blobs(prefix="models/"))
    model    = pickle.loads([b for b in blobs if "model.pkl" in b.name][0].download_as_bytes())
    encoders = pickle.loads([b for b in blobs if "encoders.pkl" in b.name][0].download_as_bytes())
    return model, encoders

def compute_feature_stats(df, label):
    NUM = ["amount","tx_count_1h","merchant_risk_score","distance_from_home"]
    stats = {}
    for col in NUM:
        stats[col] = {
            "mean": round(float(df[col].mean()), 4),
            "std":  round(float(df[col].std()), 4),
            "min":  round(float(df[col].min()), 4),
            "max":  round(float(df[col].max()), 4),
        }
    print(f"\n{label} Feature Statistics:")
    print(f"{'Feature':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*65)
    for col, s in stats.items():
        print(f"{col:<25} {s['mean']:>10.4f} {s['std']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")
    return stats

def detect_drift(train_stats, serve_stats):
    print("\nDrift Detection Report:")
    print("="*65)
    print(f"{'Feature':<25} {'Train Mean':>12} {'Serve Mean':>12} {'Drift':>8} {'Status':>10}")
    print("-"*65)
    alerts = []
    for col in train_stats:
        t_mean = train_stats[col]["mean"]
        s_mean = serve_stats[col]["mean"]
        t_std  = max(train_stats[col]["std"], 0.001)
        drift  = abs(s_mean - t_mean) / t_std
        status = "ALERT" if drift > 1.0 else "OK"
        if drift > 1.0:
            alerts.append(col)
        print(f"{col:<25} {t_mean:>12.4f} {s_mean:>12.4f} {drift:>8.3f} {status:>10}")
    return alerts

def save_monitoring_report(train_stats, serve_stats, alerts, fraud_rate):
    report = {
        "timestamp":      datetime.utcnow().isoformat(),
        "model_version":  "fraud_xgboost_v1",
        "fraud_rate":     fraud_rate,
        "drift_alerts":   alerts,
        "status":         "ALERT" if alerts else "HEALTHY",
        "train_stats":    train_stats,
        "serve_stats":    serve_stats,
    }
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    blob   = bucket.blob(f"monitoring/report_{ts}.json")
    blob.upload_from_string(json.dumps(report, indent=2))
    print(f"\nReport saved to: gs://{BUCKET}/monitoring/report_{ts}.json")
    return report

if __name__ == "__main__":
    print("="*65)
    print("STEP 5: Model Monitoring and Drift Detection")
    print("="*65)

    model, encoders = load_model()
    print("Model loaded for monitoring!")

    # Simulate training baseline data
    np.random.seed(42)
    n = 1000
    train_data = pd.DataFrame({
        "amount":             np.random.lognormal(3.5, 1.2, n),
        "tx_count_1h":        np.random.poisson(0.5, n),
        "merchant_risk_score":np.random.beta(1, 5, n),
        "distance_from_home": np.random.exponential(5, n),
    })

    # Simulate serving data with slight drift
    serve_data = pd.DataFrame({
        "amount":             np.random.lognormal(4.0, 1.3, n),  # Drift!
        "tx_count_1h":        np.random.poisson(0.6, n),
        "merchant_risk_score":np.random.beta(1.2, 4.5, n),
        "distance_from_home": np.random.exponential(6, n),       # Drift!
    })

    train_stats = compute_feature_stats(train_data, "TRAINING")
    serve_stats = compute_feature_stats(serve_data, "SERVING")
    alerts      = detect_drift(train_stats, serve_stats)
    fraud_rate  = 0.023

    print(f"\nCurrent fraud rate: {fraud_rate*100:.1f}%")
    print(f"Drift alerts:       {len(alerts)} feature(s) drifted")
    if alerts:
        print(f"Features drifted:   {', '.join(alerts)}")
        print("ACTION REQUIRED:    Retrain model with recent data!")
    else:
        print("Model status:       HEALTHY")

    report = save_monitoring_report(train_stats, serve_stats, alerts, fraud_rate)

    print("\n" + "="*65)
    print("STEP 5 Complete!")
    print(f"Model Status:  {report['status']}")
    print(f"Fraud Rate:    {fraud_rate*100:.1f}%")
    print(f"Drift Alerts:  {len(alerts)} feature(s)")
    print("Monitoring report saved to Cloud Storage!")
    print("="*65)
