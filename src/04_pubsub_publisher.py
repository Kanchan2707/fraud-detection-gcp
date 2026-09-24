import json
import time
import yaml
import random
import numpy as np
from datetime import datetime
from google.cloud import pubsub_v1

with open("config/pipeline_config.yaml") as f:
    cfg = yaml.safe_load(f)

PROJECT_ID = cfg["gcp"]["project_id"]
TOPIC_ID   = cfg["pubsub"]["topic_id"]

def create_topic():
    publisher  = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    try:
        publisher.create_topic(request={"name": topic_path})
        print(f"Created topic: {topic_path}")
    except Exception:
        print(f"Topic exists: {topic_path}")
    return publisher, topic_path

def generate_transaction(is_fraud=False):
    if is_fraud:
        return {
            "transaction_id":     f"TXN_{datetime.utcnow().strftime('%H%M%S')}_{random.randint(1000,9999)}",
            "amount":             round(random.uniform(500, 2000), 2),
            "amount_mean_1h":     round(random.uniform(600, 2000), 2),
            "amount_std_1h":      round(random.uniform(100, 500), 2),
            "tx_count_1h":        random.randint(5, 20),
            "tx_count_24h":       random.randint(15, 50),
            "amount_deviation":   round(random.uniform(2.0, 5.0), 3),
            "hour_of_day":        random.choice([1,2,3,4,22,23]),
            "day_of_week":        random.randint(0, 6),
            "days_since_last_tx": round(random.uniform(0.01, 0.1), 3),
            "merchant_risk_score":round(random.uniform(0.7, 1.0), 3),
            "distance_from_home": round(random.uniform(100, 500), 2),
            "merchant_category":  "online",
            "card_type":          "prepaid",
            "transaction_channel":"online",
            "is_fraud_label":     1,
        }
    else:
        return {
            "transaction_id":     f"TXN_{datetime.utcnow().strftime('%H%M%S')}_{random.randint(1000,9999)}",
            "amount":             round(random.uniform(5, 200), 2),
            "amount_mean_1h":     round(random.uniform(10, 150), 2),
            "amount_std_1h":      round(random.uniform(5, 50), 2),
            "tx_count_1h":        random.randint(0, 2),
            "tx_count_24h":       random.randint(1, 5),
            "amount_deviation":   round(random.uniform(-0.5, 0.5), 3),
            "hour_of_day":        random.randint(8, 21),
            "day_of_week":        random.randint(0, 6),
            "days_since_last_tx": round(random.uniform(0.5, 3.0), 3),
            "merchant_risk_score":round(random.uniform(0.0, 0.3), 3),
            "distance_from_home": round(random.uniform(0, 20), 2),
            "merchant_category":  random.choice(["grocery","restaurant","gas","retail"]),
            "card_type":          random.choice(["credit","debit"]),
            "transaction_channel":random.choice(["in-store","mobile"]),
            "is_fraud_label":     0,
        }

if __name__ == "__main__":
    print("="*55)
    print("STEP 4: Publishing Transactions to Pub/Sub")
    print("="*55)

    publisher, topic_path = create_topic()

    print("\nPublishing 20 transactions (18 legit + 2 fraud)...")
    print("-"*55)

    published = 0
    fraud_count = 0

    for i in range(20):
        is_fraud   = (i in [7, 15])
        transaction= generate_transaction(is_fraud)
        data       = json.dumps(transaction).encode("utf-8")
        future     = publisher.publish(topic_path, data)
        future.result()

        status     = "FRAUD" if is_fraud else "legit"
        published += 1
        if is_fraud:
            fraud_count += 1

        print(f"[{i+1:02d}] TXN: ${transaction['amount']:8.2f} | "
              f"Hour: {transaction['hour_of_day']:02d}:00 | "
              f"Risk: {transaction['merchant_risk_score']:.2f} | "
              f"Type: {status}")
        time.sleep(0.5)

    print("-"*55)
    print(f"\nPublished {published} transactions to Pub/Sub!")
    print(f"Legitimate: {published - fraud_count}")
    print(f"Fraudulent: {fraud_count}")
    print(f"\nTopic: projects/{PROJECT_ID}/topics/{TOPIC_ID}")
    print("\nStep 4 Complete!")
    print("Transactions are now streaming in Pub/Sub!")
