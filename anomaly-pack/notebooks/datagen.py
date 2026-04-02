import argparse
import os
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import random
from faker import Faker


def generate_synthetic(num_organizations=5,
                       num_vendors_per_org=10,
                       num_users_per_org=5,
                       num_transactions=50000,
                       seed=42,
                       out_path=None):
    """Generate a synthetic enterprise transactions dataset.

    Improvements vs. the original:
    - Reproducible seeding for random, numpy and Faker
    - More realistic (skewed) transaction amounts using log-normal
    - Additional fields: `invoice_number`, `tax_amount`, `vendor_risk_score`,
      `user_tenure_days`, `previous_txn_count`, `approval_status`
    - Risk score weighting scaled so `fraud_score` contributes meaningfully
    - Saves to `data/processed` by default when available
    """

    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    fake = Faker()
    Faker.seed(seed)

    CATEGORIES = ['Travel', 'Office Supplies', 'Equipment', 'Software', 'Consulting']
    PAYMENT_METHODS = ['Bank Transfer', 'Credit Card', 'Cash', 'Paypal']
    CURRENCIES = ['USD', 'EUR', 'GBP', 'AED']
    APPROVAL_STATUSES = ['Approved', 'Pending', 'Rejected']

    # Organizations
    organizations = []
    for _ in range(num_organizations):
        org_id = str(uuid.uuid4())
        organizations.append({
            'organization_id': org_id,
            'organization_name': fake.company(),
            'industry': fake.bs(),
            'country': random.choice(['US', 'UK', 'EU', 'UAE']),
            'currency': random.choice(CURRENCIES)
        })

    # Vendors and users
    vendors = []
    users = []
    for org in organizations:
        for _ in range(num_vendors_per_org):
            vendors.append({
                'vendor_id': str(uuid.uuid4()),
                'organization_id': org['organization_id'],
                'vendor_name': fake.company(),
                'country': org['country'],
                'industry': fake.bs(),
                # vendor risk 0..1 (used as a feature, correlated with unusual_vendor_flag)
                'vendor_risk_score': round(random.random(), 2)
            })
        for _ in range(num_users_per_org):
            # user tenure in days
            tenure = random.randint(30, 3650)
            users.append({
                'user_id': str(uuid.uuid4()),
                'organization_id': org['organization_id'],
                'full_name': fake.name(),
                'email': fake.unique.email(),
                'user_tenure_days': tenure
            })

    # Prepare amount distribution (skewed): log-normal then clipped
    amounts = np.random.lognormal(mean=7.5, sigma=1.0, size=num_transactions)
    # scale down to a reasonable range and clip
    amounts = np.clip(amounts, 10, 200000)
    amounts = np.round(amounts, 2)

    transactions = []
    now = datetime.now()

    for i in range(num_transactions):
        org = random.choice(organizations)
        user = random.choice([u for u in users if u['organization_id'] == org['organization_id']])
        vendor = random.choice([v for v in vendors if v['organization_id'] == org['organization_id']])

        # date within last year up to now
        txn_date = fake.date_time_between(start_date='-1y', end_date='now')
        amount = float(amounts[i])

        # Fraud / anomaly (labels)
        anomaly_flag = 1 if random.random() < 0.02 else 0
        fraud_score = round(random.uniform(0.7, 1.0), 2) if anomaly_flag else round(random.uniform(0.0, 0.3), 2)

        # Rule-based flags
        duplicate_flag = 1 if random.random() < 0.01 else 0
        missing_invoice_flag = 1 if random.random() < 0.02 else 0
        unusual_vendor_flag = 1 if (random.random() < 0.05 or vendor['vendor_risk_score'] > 0.85) else 0
        weekend_or_odd_time_flag = 1 if txn_date.weekday() >= 5 else 0

        # Additional metadata
        invoice_number = None if missing_invoice_flag else f"INV-{random.randint(100000,999999)}"
        tax_amount = round(amount * random.choice([0.0, 0.05, 0.1, 0.2]), 2)
        previous_txn_count = int(np.random.poisson(5))

        # Risk score: scale fraud_score contribution appropriately (0..40), anomaly gives large bump
        risk_score = (anomaly_flag * 50
                      + fraud_score * 40
                      + duplicate_flag * 5
                      + missing_invoice_flag * 3
                      + unusual_vendor_flag * 1
                      + weekend_or_odd_time_flag * 1)
        risk_score = min(round(risk_score, 2), 100)
        if risk_score <= 30:
            risk_level = 'Low'
        elif risk_score <= 70:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        # Approval status more likely Pending/Rejected when high risk
        if risk_level == 'High':
            approval_status = random.choices(APPROVAL_STATUSES, weights=[0.2, 0.6, 0.2])[0]
        elif risk_level == 'Medium':
            approval_status = random.choices(APPROVAL_STATUSES, weights=[0.7, 0.25, 0.05])[0]
        else:
            approval_status = random.choices(APPROVAL_STATUSES, weights=[0.95, 0.04, 0.01])[0]

        transactions.append({
            'transaction_id': str(uuid.uuid4()),
            'organization_id': org['organization_id'],
            'organization_name': org['organization_name'],
            'user_id': user['user_id'],
            'date': txn_date.isoformat(sep=' '),
            'amount': amount,
            'currency': org['currency'],
            'vendor_id': vendor['vendor_id'],
            'vendor_name': vendor['vendor_name'],
            'category': random.choice(CATEGORIES),
            'payment_method': random.choice(PAYMENT_METHODS),
            'description': fake.catch_phrase(),
            'invoice_number': invoice_number,
            'tax_amount': tax_amount,
            'anomaly_flag': anomaly_flag,
            'fraud_score': fraud_score,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'duplicate_flag': duplicate_flag,
            'missing_invoice_flag': missing_invoice_flag,
            'unusual_vendor_flag': unusual_vendor_flag,
            'weekend_or_odd_time_flag': weekend_or_odd_time_flag,
            'vendor_risk_score': vendor.get('vendor_risk_score', 0.0),
            'user_tenure_days': user.get('user_tenure_days', None),
            'previous_txn_count': previous_txn_count,
            'approval_status': approval_status,
            'source_file_id': str(uuid.uuid4())
        })

    df = pd.DataFrame(transactions)

    # Default output path
    if out_path is None:
        default_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
        os.makedirs(default_dir, exist_ok=True)
        out_path = os.path.join(default_dir, 'synthetic_enterprise_audit.csv')

    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset created: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic enterprise audit dataset')
    parser.add_argument('--num-transactions', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default=None, help='Output CSV path')
    args = parser.parse_args()

    generate_synthetic(num_transactions=args.num_transactions, seed=args.seed, out_path=args.out)