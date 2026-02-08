import pandas as pd
import json
import os

# Path to log file (assuming running from root)
LOG_PATH = os.path.join("serving", "shadow_log.jsonl")

def analyze():
    print(f"Analyzing Shadow Logs from: {LOG_PATH}")

    
    if not os.path.exists(LOG_PATH):
        print(f"No shadow logs found at {LOG_PATH}")

        print("   -> Tip: Ensure 'SHADOW_MODEL_URL' is set and V1 is processing requests.")
        return

    data = []
    with open(LOG_PATH, 'r') as f:
        for line in f:
            try:
                if line.strip():
                    data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    if not data:
        print("Log file exists but is empty.")

        return

    df = pd.DataFrame(data)
    
    print("\nSHADOW MODEL COMPARISON REPORT")

    print("=================================")
    print(f"samples: {len(df)}")
    
    # 1. Agreement Rate
    agreement_rate = df['agreement'].mean() * 100
    print(f"Agreement Rate:      {agreement_rate:.2f}%")

    
    # 2. Fraud Rate Comparison
    v1_fraud_count = (df['v1_prediction'] == 'fraud').sum()
    v2_fraud_count = (df['v2_prediction'] == 'fraud').sum()
    
    v1_rate = (v1_fraud_count / len(df)) * 100
    v2_rate = (v2_fraud_count / len(df)) * 100
    
    print("\nRisk Sensitivity:")

    print(f"  - V1 (Live) Fraud Rate:   {v1_rate:.2f}% ({v1_fraud_count} txns)")
    print(f"  - V2 (Shadow) Fraud Rate: {v2_rate:.2f}% ({v2_fraud_count} txns)")
    
    if v2_rate > v1_rate:
        print("  -> V2 is MORE aggressive (flags more fraud). Check Precision.")
    elif v2_rate < v1_rate:
        print("  -> V2 is MORE conservative. Check Recall.")
    else:
        print("  -> Models have identical sensitivity.")

    # 3. Disagreement Analysis
    disagreements = df[~df['agreement']]
    
    print("\nrunning disagreement analysis...")
    if disagreements.empty:
        print("  No disagreements found. Models are perfectly aligned.")

    else:
        count = len(disagreements)
        print(f"  Found {count} disagreements.")
        
        # V1=Legit, V2=Fraud (New Detections)
        new_detections = disagreements[disagreements['v2_prediction'] == 'fraud']
        # V1=Fraud, V2=Legit (Misses or False Positive Reduction)
        misses = disagreements[disagreements['v1_prediction'] == 'fraud']
        
        print(f"  - V1 Safe -> V2 Fraud: {len(new_detections)} (Potential New Catch)")
        print(f"  - V1 Fraud -> V2 Safe: {len(misses)} (Potential FP Reduction)")
        
        print("\n  Sample Deviations (Top 5 by confidence diff):")
        # sort by diff descending
        if 'diff' in disagreements.columns:
            top_deviations = disagreements.sort_values('diff', ascending=False).head(5)
            print(top_deviations[['transaction_id', 'v1_probability', 'v2_probability', 'diff']].to_string(index=False))

if __name__ == "__main__":
    analyze()
