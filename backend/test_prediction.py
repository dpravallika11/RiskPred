import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.prediction_service import prediction_service

print('Ready:', prediction_service.is_ready)
print('Features:', len(prediction_service.feature_columns))
print('Threshold:', prediction_service.threshold)

test_txn = {
    'transaction_id': 'TXN_TEST_001',
    'merchant_id': 'MERCHANT_001',
    'customer_id': 'CUST_001',
    'amount': 18500.00,
    'device_id': 'DEV_001',
    'is_new_device': True,
    'location': 'Mumbai',
    'is_new_location': True,
    'payment_method': 'credit_card',
    'velocity_5m': 5,
    'failed_attempts_24h': 3,
    'ProductCD': 'W',
    'card4': 'visa',
    'card6': 'credit',
    'P_emaildomain': 'gmail.com',
    'R_emaildomain': 'gmail.com',
}

result = prediction_service.predict(test_txn)
print()
print('=== PREDICTION RESULT ===')
print(f'Fraud Probability: {result["fraud_probability"]}')
print(f'Risk Score: {result["risk_score"]}/100')
print(f'Risk Level: {result["risk_level"]}')
print(f'Action: {result["recommended_action"]}')
print()
print('Top Risk Factors:')
for f in result['top_risk_factors'][:5]:
    print(f'  - {f["description"]} (impact: +{f["impact"]:.4f})')
print()
print('Top Risk Reducers:')
for f in result['top_risk_reducers'][:3]:
    print(f'  - {f["description"]} (impact: {f["impact"]:.4f})')
