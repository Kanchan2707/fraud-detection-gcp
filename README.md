# Real-Time Fraud Detection Pipeline on GCP

**Author:** Kanchan Chauhan | Senior ML Engineer  
**Tech Stack:** Python, GCP Pub/Sub, XGBoost, BigQuery, Cloud Storage

## Results
- AUC-PR: 1.0000 on 100,000 transactions
- Fraud score: 0.9878 on suspicious transaction → FRAUD BLOCKED
- Legit score: 0.0044 on normal transaction → APPROVED
- Model monitoring: automated drift detection

## Project Steps
| Step | Description |
|---|---|
| 01_generate_data.py | Generate 500K transactions → BigQuery |
| 02_train_model.py | Train XGBoost model AUC-PR 1.0 |
| 03_predict.py | Real-time fraud scoring |
| 04_pubsub_publisher.py | Stream transactions via Pub/Sub |
| 05_monitoring.py | Model drift detection |

## How to Run
```bash
python src/01_generate_data.py
python src/02_train_model.py
python src/03_predict.py
python src/04_pubsub_publisher.py
python src/05_monitoring.py
```

## Contact
Kanchan Chauhan | Kanchan9chauhan@gmail.com | Dover NH
