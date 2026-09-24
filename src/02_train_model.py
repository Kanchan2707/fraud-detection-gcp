import yaml, pickle, json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import xgboost as xgb
from google.cloud import bigquery, storage

with open("config/pipeline_config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_ID  = cfg["gcp"]["project_id"]
BUCKET      = cfg["storage"]["bucket_name"]
DATASET_ID  = cfg["bigquery"]["dataset_id"]
TABLE_ID    = cfg["bigquery"]["training_table"]
NUMERICAL   = ["amount","amount_mean_1h","amount_std_1h","tx_count_1h","tx_count_24h","amount_deviation","hour_of_day","day_of_week","days_since_last_tx","merchant_risk_score","distance_from_home"]
CATEGORICAL = ["merchant_category","card_type","transaction_channel"]
ALL_FEATURES= NUMERICAL + CATEGORICAL
TARGET      = "is_fraud"

def load_data():
    print("Loading data from BigQuery...")
    client = bigquery.Client(project=PROJECT_ID)
    query  = f"SELECT {','.join(ALL_FEATURES)},{TARGET} FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` LIMIT 100000"
    df     = client.query(query).to_dataframe()
    print(f"Loaded {len(df):,} rows | Fraud: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.2f}%)")
    return df

def preprocess(df):
    encoders = {}
    df = df.copy()
    for col in CATEGORICAL:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    scaler = StandardScaler()
    df[NUMERICAL] = scaler.fit_transform(df[NUMERICAL])
    encoders["scaler"] = scaler
    return df, encoders

def train(X_train, y_train, X_val, y_val):
    n_neg  = (y_train==0).sum()
    n_pos  = max((y_train==1).sum(), 1)
    scale  = n_neg / n_pos
    print(f"Class ratio: {scale:.1f}")
    model  = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, eval_metric="aucpr",
        early_stopping_rounds=20, random_state=42,
        tree_method="hist", n_jobs=-1, verbosity=1,
    )
    print("Training XGBoost model...")
    model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=50)
    return model

def evaluate(model, X_test, y_test):
    proba   = model.predict_proba(X_test)[:,1]
    auc_pr  = average_precision_score(y_test, proba)
    auc_roc = roc_auc_score(y_test, proba)
    print(f"\n{'='*40}")
    print(f"AUC-PR  (primary):   {auc_pr:.4f}")
    print(f"AUC-ROC (secondary): {auc_roc:.4f}")
    y_pred  = (proba >= 0.3).astype(int)
    print(classification_report(y_test, y_pred, target_names=["Legit","Fraud"]))
    return {"auc_pr":float(auc_pr),"auc_roc":float(auc_roc)}

def save(model, encoders, metrics):
    print("\nSaving model to GCS...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    path   = f"models/fraud_xgboost_{ts}/"
    bucket.blob(f"{path}model.pkl").upload_from_string(pickle.dumps(model))
    bucket.blob(f"{path}encoders.pkl").upload_from_string(pickle.dumps(encoders))
    bucket.blob(f"{path}metrics.json").upload_from_string(json.dumps(metrics,indent=2))
    print(f"Saved to: gs://{BUCKET}/{path}")
    return f"gs://{BUCKET}/{path}"

if __name__ == "__main__":
    print("="*50)
    print("STEP 2: Training Fraud Detection Model")
    print("="*50)
    df           = load_data()
    df, encoders = preprocess(df)
    X, y         = df[ALL_FEATURES], df[TARGET]
    X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)
    X_tr,X_val,y_tr,y_val = train_test_split(X_tr,y_tr,test_size=0.2,random_state=42)
    print(f"Train:{len(X_tr):,} Val:{len(X_val):,} Test:{len(X_te):,}")
    model   = train(X_tr, y_tr, X_val, y_val)
    metrics = evaluate(model, X_te, y_te)
    gcs     = save(model, encoders, metrics)
    print(f"\nStep 2 Complete!")
    print(f"AUC-PR:  {metrics['auc_pr']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"Model saved to GCS successfully!")
