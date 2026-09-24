import pickle, yaml
import pandas as pd
from google.cloud import storage

with open("config/pipeline_config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_ID = cfg["gcp"]["project_id"]
BUCKET     = cfg["storage"]["bucket_name"]

def load_model():
    print("Loading model from GCS...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET)
    blobs  = list(bucket.list_blobs(prefix="models/"))
    model    = pickle.loads([b for b in blobs if "model.pkl" in b.name][0].download_as_bytes())
    encoders = pickle.loads([b for b in blobs if "encoders.pkl" in b.name][0].download_as_bytes())
    print("Model loaded!")
    return model, encoders

def score(tx, model, encoders):
    NUM = ["amount","amount_mean_1h","amount_std_1h","tx_count_1h","tx_count_24h","amount_deviation","hour_of_day","day_of_week","days_since_last_tx","merchant_risk_score","distance_from_home"]
    CAT = ["merchant_category","card_type","transaction_channel"]
    df  = pd.DataFrame([tx])
    for c in CAT:
        df[c] = encoders[c].transform(df[c].astype(str))
    df[NUM] = encoders["scaler"].transform(df[NUM])
    prob = model.predict_proba(df[NUM+CAT])[0][1]
    return round(float(prob),4), "FRAUD BLOCKED" if prob>=0.3 else "APPROVED"

if __name__ == "__main__":
    model, encoders = load_model()
    legit = {"amount":25.50,"amount_mean_1h":30.0,"amount_std_1h":10.0,"tx_count_1h":1,"tx_count_24h":3,"amount_deviation":0.2,"hour_of_day":14,"day_of_week":2,"days_since_last_tx":1.0,"merchant_risk_score":0.1,"distance_from_home":5.0,"merchant_category":"grocery","card_type":"credit","transaction_channel":"in-store"}
    fraud = {"amount":850.0,"amount_mean_1h":900.0,"amount_std_1h":250.0,"tx_count_1h":8,"tx_count_24h":20,"amount_deviation":3.5,"hour_of_day":3,"day_of_week":6,"days_since_last_tx":0.02,"merchant_risk_score":0.85,"distance_from_home":150.0,"merchant_category":"online","card_type":"prepaid","transaction_channel":"online"}
    s1,d1 = score(legit, model, encoders)
    s2,d2 = score(fraud, model, encoders)
    print("="*50)
    print("REAL-TIME FRAUD DETECTION RESULTS")
    print("="*50)
    print(f"Transaction 1 - $25.50  grocery  2pm  -> Score:{s1} -> {d1}")
    print(f"Transaction 2 - $850.00 online   3am  -> Score:{s2} -> {d2}")
    print("="*50)
