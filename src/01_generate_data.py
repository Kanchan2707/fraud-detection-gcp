import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from google.cloud import bigquery, storage

with open("config/pipeline_config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_ID = cfg["gcp"]["project_id"]
BUCKET     = cfg["storage"]["bucket_name"]
DATASET_ID = cfg["bigquery"]["dataset_id"]
TABLE_ID   = cfg["bigquery"]["training_table"]
np.random.seed(42)

def generate_transactions(n=500000):
    print(f"Generating {n:,} transactions...")
    n_fraud = int(n * 0.01)
    n_legit = n - n_fraud
    hours24 = np.arange(24)
    legit_h = np.exp(-0.5*((hours24-13)/4)**2)
    legit_h[0:6] *= 0.1
    legit_h /= legit_h.sum()
    fraud_h = np.ones(24)
    fraud_h[1:5] *= 4.0
    fraud_h[22:24] *= 3.0
    fraud_h[9:17] *= 0.5
    fraud_h /= fraud_h.sum()
    base = datetime(2026,1,1)
    timestamps = sorted([base + timedelta(days=np.random.uniform(0,90), hours=np.random.uniform(0,24)) for _ in range(n)])
    legit = {
        "amount": np.random.lognormal(3.5,1.2,n_legit).clip(0.01,50000).round(2),
        "hour_of_day": np.random.choice(24,n_legit,p=legit_h),
        "day_of_week": np.random.randint(0,7,n_legit),
        "merchant_risk_score": np.random.beta(1,5,n_legit).round(3),
        "tx_count_1h": np.random.poisson(0.5,n_legit),
        "tx_count_24h": np.random.poisson(3,n_legit),
        "amount_mean_1h": np.random.lognormal(3.5,0.8,n_legit).round(2),
        "amount_std_1h": np.random.exponential(15,n_legit).round(2),
        "amount_deviation": np.random.normal(0,0.3,n_legit).round(3),
        "days_since_last_tx": np.random.exponential(1,n_legit).round(3),
        "distance_from_home": np.random.exponential(5,n_legit).round(2),
        "merchant_category": np.random.choice(["grocery","restaurant","gas","retail","online","travel"],n_legit),
        "card_type": np.random.choice(["credit","debit","prepaid"],n_legit),
        "transaction_channel": np.random.choice(["in-store","online","atm","mobile"],n_legit),
        "is_fraud": np.zeros(n_legit,dtype=int),
    }
    fraud = {
        "amount": np.random.lognormal(5.0,1.5,n_fraud).clip(0.01,50000).round(2),
        "hour_of_day": np.random.choice(24,n_fraud,p=fraud_h),
        "day_of_week": np.random.randint(0,7,n_fraud),
        "merchant_risk_score": np.random.beta(5,2,n_fraud).round(3),
        "tx_count_1h": np.random.poisson(5,n_fraud),
        "tx_count_24h": np.random.poisson(15,n_fraud),
        "amount_mean_1h": np.random.lognormal(5.0,1.0,n_fraud).round(2),
        "amount_std_1h": np.random.exponential(100,n_fraud).round(2),
        "amount_deviation": np.random.normal(2.5,1.0,n_fraud).round(3),
        "days_since_last_tx": np.random.exponential(0.1,n_fraud).round(3),
        "distance_from_home": np.random.exponential(50,n_fraud).round(2),
        "merchant_category": np.random.choice(["online","travel","electronics","jewelry","crypto"],n_fraud),
        "card_type": np.random.choice(["credit","debit","prepaid"],n_fraud),
        "transaction_channel": np.random.choice(["in-store","online","atm","mobile"],n_fraud),
        "is_fraud": np.ones(n_fraud,dtype=int),
    }
    df = pd.concat([pd.DataFrame(legit),pd.DataFrame(fraud)],ignore_index=True)
    df = df.sample(frac=1,random_state=42).reset_index(drop=True)
    df["transaction_id"]   = [f"TXN_{i:08d}" for i in range(len(df))]
    df["cardholder_id"]    = np.random.choice(range(10000),len(df))
    df["transaction_time"] = [timestamps[i] for i in range(len(df))]
    return df

def save_to_bigquery(df):
    print("Saving to BigQuery...")
    client  = bigquery.Client(project=PROJECT_ID)
    dataset = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset.location = "US"
    client.create_dataset(dataset,exists_ok=True)
    schema = [
        bigquery.SchemaField("transaction_id","STRING"),
        bigquery.SchemaField("cardholder_id","INTEGER"),
        bigquery.SchemaField("transaction_time","TIMESTAMP"),
        bigquery.SchemaField("amount","FLOAT"),
        bigquery.SchemaField("hour_of_day","INTEGER"),
        bigquery.SchemaField("day_of_week","INTEGER"),
        bigquery.SchemaField("merchant_risk_score","FLOAT"),
        bigquery.SchemaField("tx_count_1h","INTEGER"),
        bigquery.SchemaField("tx_count_24h","INTEGER"),
        bigquery.SchemaField("amount_mean_1h","FLOAT"),
        bigquery.SchemaField("amount_std_1h","FLOAT"),
        bigquery.SchemaField("amount_deviation","FLOAT"),
        bigquery.SchemaField("days_since_last_tx","FLOAT"),
        bigquery.SchemaField("distance_from_home","FLOAT"),
        bigquery.SchemaField("merchant_category","STRING"),
        bigquery.SchemaField("card_type","STRING"),
        bigquery.SchemaField("transaction_channel","STRING"),
        bigquery.SchemaField("is_fraud","INTEGER"),
    ]
    job_config = bigquery.LoadJobConfig(schema=schema,write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df,f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}",job_config=job_config)
    job.result()
    print(f"Loaded {len(df):,} rows to BigQuery successfully!")

def save_to_gcs(df):
    print("Saving to GCS...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    if not bucket.exists():
        bucket.create(location="US")
    blob = bucket.blob("data/training/transactions.csv")
    blob.upload_from_string(df.to_csv(index=False),content_type="text/csv")
    print("Saved to GCS successfully!")

if __name__ == "__main__":
    print("="*50)
    print("STEP 1: Generating Fraud Detection Training Data")
    print("="*50)
    df = generate_transactions(500000)
    print(f"Total: {len(df):,} | Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    save_to_bigquery(df)
    save_to_gcs(df)
    print("\nStep 1 Complete!")
