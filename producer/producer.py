"""
Real-Time Fraud Detection System - Transaction Producer
========================================================
Simulates a payment gateway stream by pushing synthetic transactions to Kafka.
Includes patterns for high-frequency, geographic, and amount-based fraud.
"""

import json
import random
import time
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from kafka import KafkaProducer
from faker import Faker
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize Faker for realistic data generation
fake = Faker(['en_GB', 'en_US'])  # UK focus for fintech roles

# Configuration from environment
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'transaction_stream')
TRANSACTIONS_PER_SECOND = float(os.getenv('TRANSACTIONS_PER_SECOND', '10'))
FRAUD_RATIO = float(os.getenv('FRAUD_RATIO', '0.02'))  # 2% fraud rate (realistic)


@dataclass
class Transaction:
    """
    Transaction schema matching production payment gateway format.
    All fields are chosen for their fraud detection signal value.
    """
    transaction_id: str
    user_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_name: str
    merchant_category: str  # MCC code category
    timestamp: str
    location_country: str
    location_city: str
    card_present: bool
    channel: str  # 'online', 'pos', 'atm'
    device_id: Optional[str]
    ip_address: Optional[str]
    customer_age: int  # Added for Fairness/Bias Check
    is_fraud: bool  # Ground truth label (for training, hidden in production)


# Merchant Category Codes (MCC) - Common categories for realistic simulation
MERCHANT_CATEGORIES = [
    'grocery_stores', 'restaurants', 'gas_stations', 'hotels',
    'airlines', 'electronics', 'clothing', 'pharmacy',
    'entertainment', 'utilities', 'professional_services',
    'gambling', 'crypto_exchange', 'money_transfer'
]

# High-risk categories (more likely in fraud patterns)
HIGH_RISK_CATEGORIES = ['gambling', 'crypto_exchange', 'money_transfer', 'electronics']

# UK and global cities for geographic simulation
UK_CITIES = ['London', 'Manchester', 'Birmingham', 'Leeds', 'Glasgow', 'Liverpool', 'Bristol']
GLOBAL_CITIES = ['New York', 'Los Angeles', 'Paris', 'Tokyo', 'Dubai', 'Singapore', 'Mumbai']


class UserBehaviorProfile:
    """
    Simulates realistic user spending behavior.
    """
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.age = random.randint(18, 90)  # Simulated age for fairness check
        self.avg_transaction = random.uniform(20, 200)
        self.std_transaction = self.avg_transaction * 0.3
        self.preferred_categories = random.sample(MERCHANT_CATEGORIES[:10], k=4)
        self.home_city = random.choice(UK_CITIES)
        self.transaction_frequency = random.uniform(0.5, 5)  # per day
        self.last_transaction_time = datetime.now() - timedelta(hours=random.randint(1, 48))
        self.last_location = self.home_city


class FraudPatternGenerator:
    """
    Generates realistic fraud patterns based on known attack vectors.
    These patterns are what ML models learn to detect.
    """
    
    @staticmethod
    def high_frequency_attack(user: UserBehaviorProfile) -> List[Transaction]:
        """
        Card testing fraud: Rapid small transactions to test if card is valid.
        Then followed by large purchase.
        """
        transactions = []
        base_time = datetime.now()
        
        # 5-10 rapid small transactions
        for i in range(random.randint(5, 10)):
            txn = Transaction(
                transaction_id=str(uuid.uuid4()),
                user_id=user.user_id,
                amount=round(random.uniform(0.50, 5.00), 2),
                currency='GBP',
                merchant_id=fake.uuid4()[:8],
                merchant_name=fake.company(),
                merchant_category=random.choice(MERCHANT_CATEGORIES),
                timestamp=(base_time + timedelta(seconds=i * random.randint(5, 30))).isoformat(),
                location_country='GB',
                location_city=random.choice(UK_CITIES),
                card_present=False,
                channel='online',
                device_id=fake.uuid4()[:12],
                ip_address=fake.ipv4(),
                customer_age=user.age,
                is_fraud=True
            )
            transactions.append(txn)
        
        # Large fraudulent purchase
        large_txn = Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user.user_id,
            amount=round(random.uniform(500, 5000), 2),
            currency='GBP',
            merchant_id=fake.uuid4()[:8],
            merchant_name=fake.company(),
            merchant_category=random.choice(HIGH_RISK_CATEGORIES),
            timestamp=(base_time + timedelta(minutes=2)).isoformat(),
            location_country='GB',
            location_city=random.choice(UK_CITIES),
            card_present=False,
            channel='online',
            device_id=fake.uuid4()[:12],
            ip_address=fake.ipv4(),
            customer_age=user.age,
            is_fraud=True
        )
        transactions.append(large_txn)
        
        return transactions
    
    @staticmethod
    def geographic_anomaly(user: UserBehaviorProfile) -> Transaction:
        """
        Impossible travel fraud: Transaction from a location too far from last one.
        E.g., London then Dubai within 30 minutes.
        """
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user.user_id,
            amount=round(random.uniform(200, 2000), 2),
            currency=random.choice(['USD', 'EUR', 'AED']),
            merchant_id=fake.uuid4()[:8],
            merchant_name=fake.company(),
            merchant_category=random.choice(HIGH_RISK_CATEGORIES),
            timestamp=datetime.now().isoformat(),
            location_country=random.choice(['US', 'AE', 'SG']),
            location_city=random.choice(GLOBAL_CITIES),
            card_present=True,
            channel='pos',
            device_id=None,
            ip_address=None,
            customer_age=user.age,
            is_fraud=True
        )
    
    @staticmethod
    def amount_anomaly(user: UserBehaviorProfile) -> Transaction:
        """
        Unusual amount: Transaction 10x+ the user's normal spending.
        """
        return Transaction(
            transaction_id=str(uuid.uuid4()),
            user_id=user.user_id,
            amount=round(user.avg_transaction * random.uniform(10, 50), 2),
            currency='GBP',
            merchant_id=fake.uuid4()[:8],
            merchant_name=fake.company(),
            merchant_category=random.choice(HIGH_RISK_CATEGORIES),
            timestamp=datetime.now().isoformat(),
            location_country='GB',
            location_city=user.home_city,
            card_present=False,
            channel='online',
            device_id=fake.uuid4()[:12],
            ip_address=fake.ipv4(),
            customer_age=user.age,
            is_fraud=True
        )


def generate_normal_transaction(user: UserBehaviorProfile) -> Transaction:
    """
    Generate a legitimate transaction matching user's behavioral profile.
    """
    amount = max(0.50, random.gauss(user.avg_transaction, user.std_transaction))
    
    return Transaction(
        transaction_id=str(uuid.uuid4()),
        user_id=user.user_id,
        amount=round(amount, 2),
        currency='GBP',
        merchant_id=fake.uuid4()[:8],
        merchant_name=fake.company(),
        merchant_category=random.choice(user.preferred_categories),
        timestamp=datetime.now().isoformat(),
        location_country='GB',
        location_city=user.home_city,
        card_present=random.choice([True, False]),
        channel=random.choice(['online', 'pos', 'pos', 'pos']),  # POS more common
        device_id=fake.uuid4()[:12] if random.random() > 0.5 else None,
        ip_address=fake.ipv4() if random.random() > 0.5 else None,
        customer_age=user.age,
        is_fraud=False
    )


class TransactionProducer:
    """
    Kafka producer for streaming transactions.
    """
    
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            # Production settings
            acks='all',  # Wait for all replicas (durability)
            retries=3,
            linger_ms=10,  # Batch for efficiency
        )
        
        # Simulate 1000 users with behavior profiles
        self.users: Dict[str, UserBehaviorProfile] = {}
        for i in range(1000):
            user_id = f"user_{i:05d}"
            self.users[user_id] = UserBehaviorProfile(user_id)
        
        self.fraud_generator = FraudPatternGenerator()
        logger.info("Transaction Producer initialized")
        logger.info(f"   Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
        logger.info(f"   Topic: {KAFKA_TOPIC}")
        logger.info(f"   Rate: {TRANSACTIONS_PER_SECOND} TPS")
        logger.info(f"   Fraud Ratio: {FRAUD_RATIO * 100}%")
    
    def send_transaction(self, transaction: Transaction):
        """Send a single transaction to Kafka."""
        # Use user_id as key for partition affinity (all user txns go to same partition)
        self.producer.send(
            KAFKA_TOPIC,
            key=transaction.user_id,
            value=asdict(transaction)
        )
    
    def run(self):
        """Main producer loop."""
        logger.info("Starting transaction stream...")
        
        transaction_count = 0
        fraud_count = 0
        start_time = time.time()
        
        try:
            while True:
                # Select random user
                user_id = random.choice(list(self.users.keys()))
                user = self.users[user_id]
                
                # Calculate dynamic fraud ratio if in Demo Mode
                current_ratio = FRAUD_RATIO
                is_demo_mode = os.getenv('DEMO_MODE', 'false').lower() == 'true'
                
                if is_demo_mode:
                    # Alternating 10-minute cycles
                    # 0-9 mins: Normal (2%)
                    # 10-19 mins: High Fraud (20%) -> Drift
                    current_minute = datetime.now().minute
                    cycle_minute = current_minute % 20
                    
                    if cycle_minute >= 10:
                        current_ratio = 0.20  # 20% fraud (massive drift)
                        if transaction_count % 10 == 0:  # Avoid log spam
                            logger.warning(f"[DEMO MODE] DRIFT ACTIVE! Fraud Rate: {current_ratio:.0%}")
                    else:
                        current_ratio = 0.02  # 2% normal
                        if transaction_count % 10 == 0:
                            logger.info(f"[DEMO MODE] Normal Traffic. Fraud Rate: {current_ratio:.0%}")
                
                # Decide if this should be fraud
                if random.random() < current_ratio:
                    # Generate fraud pattern
                    pattern = random.choice([
                        'high_frequency',
                        'geographic',
                        'amount'
                    ])
                    
                    if pattern == 'high_frequency':
                        transactions = self.fraud_generator.high_frequency_attack(user)
                        for txn in transactions:
                            self.send_transaction(txn)
                            fraud_count += 1
                            transaction_count += 1
                    elif pattern == 'geographic':
                        txn = self.fraud_generator.geographic_anomaly(user)
                        self.send_transaction(txn)
                        fraud_count += 1
                        transaction_count += 1
                    else:
                        txn = self.fraud_generator.amount_anomaly(user)
                        self.send_transaction(txn)
                        fraud_count += 1
                        transaction_count += 1
                else:
                    # Generate normal transaction
                    txn = generate_normal_transaction(user)
                    self.send_transaction(txn)
                    transaction_count += 1
                
                # Update user's last transaction time
                user.last_transaction_time = datetime.now()
                
                # Rate limiting
                time.sleep(1.0 / TRANSACTIONS_PER_SECOND)
                
                # Progress logging every 100 transactions
                if transaction_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_tps = transaction_count / elapsed
                    fraud_rate = fraud_count / transaction_count * 100
                    logger.info(f"Sent: {transaction_count} | "
                          f"Fraud: {fraud_count} ({fraud_rate:.2f}%) | "
                          f"Rate: {actual_tps:.1f} TPS")
                
        except KeyboardInterrupt:
            logger.info("Shutting down producer...")

        finally:
            self.producer.flush()
            self.producer.close()
            logger.info(f"Producer stopped. Total transactions: {transaction_count}")



if __name__ == "__main__":
    producer = TransactionProducer()
    producer.run()
