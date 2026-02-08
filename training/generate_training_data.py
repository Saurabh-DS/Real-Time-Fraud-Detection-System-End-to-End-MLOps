"""
Training Data Generator
=======================
Generates synthetic historical data for training the fraud detection model.
Matches patterns used in the real-time production stream.
"""

import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from faker import Faker


# Configuration
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '../features/data')
NUM_USERS = 1000
NUM_TRANSACTIONS = 100000
FRAUD_RATIO = 0.02  # 2% fraud rate
RANDOM_SEED = 42

# Reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker(['en_GB'])
Faker.seed(RANDOM_SEED)


class TrainingDataGenerator:
    """
    Generates synthetic training data with fraud labels.
    Features are computed to match the real-time feature processor.
    """
    
    def __init__(self):
        self.users = self._generate_user_profiles()
        self.merchants = self._generate_merchant_profiles()
    
    def _generate_user_profiles(self) -> Dict[str, Dict]:
        """Create user behavioral profiles."""
        users = {}
        for i in range(NUM_USERS):
            user_id = f"user_{i:05d}"
            users[user_id] = {
                'user_id': user_id,
                'avg_amount': np.random.lognormal(3.5, 1.0),  # Log-normal for realistic amounts
                'std_amount': np.random.uniform(10, 50),
                'avg_frequency': np.random.uniform(0.5, 5),  # Transactions per day
                'preferred_categories': random.sample([
                    'grocery_stores', 'restaurants', 'gas_stations',
                    'clothing', 'entertainment', 'utilities'
                ], k=3),
                'home_city': random.choice(['London', 'Manchester', 'Birmingham', 'Leeds']),
                'age': random.randint(18, 90),
            }
        return users
    
    def _generate_merchant_profiles(self) -> Dict[str, Dict]:
        """Create merchant profiles with risk scores."""
        merchants = {}
        high_risk = ['gambling', 'crypto_exchange', 'money_transfer']
        
        for i in range(500):
            merchant_id = f"merchant_{i:04d}"
            category = random.choice([
                'grocery_stores', 'restaurants', 'gas_stations', 'hotels',
                'electronics', 'clothing', 'pharmacy', 'entertainment',
                'gambling', 'crypto_exchange', 'money_transfer'
            ])
            merchants[merchant_id] = {
                'merchant_id': merchant_id,
                'category': category,
                'is_high_risk': category in high_risk,
                'fraud_rate': 0.15 if category in high_risk else 0.01,
                'avg_transaction': np.random.uniform(20, 500),
            }
        return merchants
    
    def _compute_user_features(self, user_id: str, transactions: List[Dict], 
                               current_time: datetime) -> Dict[str, Any]:
        """
        Compute features for a user at a specific point in time.
        This mimics what the real-time feature processor does.
        """
        # Filter to transactions before current_time (point-in-time correctness)
        user_txns = [t for t in transactions 
                     if t['user_id'] == user_id and t['timestamp'] < current_time]
        
        if not user_txns:
            return self._empty_features()
        
        # Sort by time
        user_txns.sort(key=lambda x: x['timestamp'])
        
        # Time windows
        time_10m = current_time - timedelta(minutes=10)
        time_1h = current_time - timedelta(hours=1)
        time_24h = current_time - timedelta(hours=24)
        time_30d = current_time - timedelta(days=30)
        
        txns_10m = [t for t in user_txns if t['timestamp'] > time_10m]
        txns_1h = [t for t in user_txns if t['timestamp'] > time_1h]
        txns_24h = [t for t in user_txns if t['timestamp'] > time_24h]
        txns_30d = [t for t in user_txns if t['timestamp'] > time_30d]
        
        # Compute features
        features = {
            # Velocity features
            'transaction_count_10m': len(txns_10m),
            'transaction_count_1h': len(txns_1h),
            'transaction_count_24h': len(txns_24h),
            'transaction_count_30d': len(txns_30d),
            
            # Amount features
            'total_amount_10m': sum(t['amount'] for t in txns_10m),
            'total_amount_1h': sum(t['amount'] for t in txns_1h),
            'max_amount_1h': max((t['amount'] for t in txns_1h), default=0),
            'avg_amount_30d': np.mean([t['amount'] for t in txns_30d]) if txns_30d else 0,
            'std_amount_30d': np.std([t['amount'] for t in txns_30d]) if len(txns_30d) > 1 else 0,
            
            # Diversity features
            'distinct_merchants_30d': len(set(t['merchant_id'] for t in txns_30d)),
            'distinct_categories_30d': len(set(t['merchant_category'] for t in txns_30d)),
            'distinct_cities_1h': len(set(t.get('city', 'Unknown') for t in txns_1h)),
            
            # Time features
            'seconds_since_last': (current_time - user_txns[-1]['timestamp']).total_seconds() if user_txns else 86400,
            'hour_of_day': current_time.hour,
            'day_of_week': current_time.weekday(),
            'is_weekend': 1 if current_time.weekday() >= 5 else 0,
        }
        
        # Derived risk scores
        features['velocity_score'] = min(1.0, features['transaction_count_10m'] / 10.0)
        
        if features['avg_amount_30d'] > 0:
            features['amount_zscore'] = 0  # Will be computed per-transaction
        else:
            features['amount_zscore'] = 0
            
        return features
    
    def _empty_features(self) -> Dict[str, Any]:
        """Return empty features for new users."""
        return {
            'transaction_count_10m': 0,
            'transaction_count_1h': 0,
            'transaction_count_24h': 0,
            'transaction_count_30d': 0,
            'total_amount_10m': 0,
            'total_amount_1h': 0,
            'max_amount_1h': 0,
            'avg_amount_30d': 0,
            'std_amount_30d': 0,
            'distinct_merchants_30d': 0,
            'distinct_categories_30d': 0,
            'distinct_cities_1h': 0,
            'seconds_since_last': 86400,
            'hour_of_day': 12,
            'day_of_week': 0,
            'is_weekend': 0,
            'velocity_score': 0,
            'amount_zscore': 0,
        }
    
    def generate(self) -> pd.DataFrame:
        """Generate training dataset with features and labels."""
        print(f"[DATA] Generating {NUM_TRANSACTIONS:,} transactions...")
        
        transactions = []
        all_transaction_data = []
        
        # Generate base timeline
        start_date = datetime.now() - timedelta(days=60)
        
        for i in range(NUM_TRANSACTIONS):
            # Random timestamp within last 60 days
            timestamp = start_date + timedelta(
                seconds=random.randint(0, 60 * 24 * 60 * 60)
            )
            
            user_id = random.choice(list(self.users.keys()))
            user = self.users[user_id]
            merchant_id = random.choice(list(self.merchants.keys()))
            merchant = self.merchants[merchant_id]
            
            # Decide if fraud
            is_fraud = random.random() < FRAUD_RATIO
            
            if is_fraud:
                # Generate fraud pattern
                fraud_type = random.choice(['velocity', 'amount', 'category'])
                
                if fraud_type == 'velocity':
                    # High transaction count
                    amount = random.uniform(0.50, 10.00)
                elif fraud_type == 'amount':
                    # Unusually high amount
                    amount = user['avg_amount'] * random.uniform(5, 20)
                else:
                    # High-risk category
                    merchant_id = random.choice([m for m, d in self.merchants.items() 
                                                  if d['is_high_risk']])
                    merchant = self.merchants[merchant_id]
                    amount = random.uniform(200, 2000)
            else:
                # Normal transaction
                amount = max(0.50, np.random.normal(user['avg_amount'], user['std_amount']))
            
            txn = {
                'transaction_id': f"txn_{i:08d}",
                'user_id': user_id,
                'merchant_id': merchant_id,
                'merchant_category': merchant['category'],
                'amount': round(amount, 2),
                'timestamp': timestamp,
                'city': user['home_city'] if not is_fraud else random.choice(['Dubai', 'Singapore', 'Miami']),
                'is_fraud': is_fraud,
            }
            transactions.append(txn)
            
            if i % 10000 == 0:
                print(f"   Generated {i:,} transactions...")
        
        # Sort by timestamp for point-in-time feature computation
        transactions.sort(key=lambda x: x['timestamp'])
        
        print("[FEATURES] Computing features for each transaction...")
        
        # Compute features for each transaction
        for i, txn in enumerate(transactions):
            features = self._compute_user_features(
                txn['user_id'], 
                transactions[:i],  # Only transactions before this one
                txn['timestamp']
            )
            
            # Add transaction-level features
            merchant = self.merchants[txn['merchant_id']]
            features.update({
                'transaction_id': txn['transaction_id'],
                'user_id': txn['user_id'],
                'amount': txn['amount'],
                'merchant_category': txn['merchant_category'],
                'is_high_risk_merchant': 1 if merchant['is_high_risk'] else 0,
                'is_fraud': 1 if txn['is_fraud'] else 0,
                'event_timestamp': txn['timestamp'].isoformat(),
                'customer_age': self.users[txn['user_id']]['age'],
            })
            
            # Compute amount z-score
            if features['std_amount_30d'] > 0:
                features['amount_zscore'] = (txn['amount'] - features['avg_amount_30d']) / features['std_amount_30d']
            
            all_transaction_data.append(features)
            
            if i % 10000 == 0:
                print(f"   Computed features for {i:,} transactions...")
        
        df = pd.DataFrame(all_transaction_data)
        
        # Summary
        fraud_count = df['is_fraud'].sum()
        print("\n[DONE] Dataset generated:")
        print(f"   Total transactions: {len(df):,}")
        print(f"   Fraud cases: {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")
        print(f"   Features: {len(df.columns)}")
        
        return df
    
    def save(self, df: pd.DataFrame):
        """Save as Parquet for Feast offline store."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Training data
        train_path = os.path.join(OUTPUT_DIR, 'training_data.parquet')
        df.to_parquet(train_path, index=False)
        print(f"[SAVED] Training data: {train_path}")
        
        # User features (for offline store)
        user_features = df.groupby('user_id').agg({
            'avg_amount_30d': 'last',
            'std_amount_30d': 'last',
            'transaction_count_30d': 'last',
            'distinct_merchants_30d': 'last',
        }).reset_index()
        user_features['event_timestamp'] = datetime.now().isoformat()
        
        user_path = os.path.join(OUTPUT_DIR, 'user_features.parquet')
        user_features.to_parquet(user_path, index=False)
        print(f"[SAVED] User features: {user_path}")
        
        # Merchant features
        merchant_data = []
        for m_id, m_data in self.merchants.items():
            merchant_data.append({
                'merchant_id': m_id,
                'merchant_fraud_rate_30d': m_data['fraud_rate'],
                'merchant_avg_transaction': m_data['avg_transaction'],
                'merchant_risk_category': 'high' if m_data['is_high_risk'] else 'low',
                'merchant_age_days': random.randint(30, 1000),
                'event_timestamp': datetime.now().isoformat(),
            })
        
        merchant_df = pd.DataFrame(merchant_data)
        merchant_path = os.path.join(OUTPUT_DIR, 'merchant_features.parquet')
        merchant_df.to_parquet(merchant_path, index=False)
        print(f"[SAVED] Merchant features: {merchant_path}")


def main():
    generator = TrainingDataGenerator()
    df = generator.generate()
    generator.save(df)

if __name__ == "__main__":
    main()
