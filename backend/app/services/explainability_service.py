import numpy as np
from typing import List, Dict, Any

class ExplainabilityService:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def explain_transaction(self, raw_data: Dict[str, Any], proba: float) -> List[str]:
        """Generates actionable risk factors based on feature thresholds and signals."""
        reasons = []

        amount = raw_data.get('amount', 0)
        velocity = raw_data.get('velocity_5m', 1)
        failed_attempts = raw_data.get('failed_attempts_24h', 0)
        is_new_device = raw_data.get('is_new_device', False)
        is_new_location = raw_data.get('is_new_location', False)

        if amount > 10000:
            reasons.append(f"Unusually high transaction amount (₹{amount:,.2f})")
        
        if velocity > 3:
            reasons.append(f"High velocity spike ({velocity} transactions in 5 minutes)")
            
        if failed_attempts >= 2:
            reasons.append(f"Multiple failed payment attempts ({failed_attempts} in last 24h)")

        if is_new_device:
            reasons.append("Unrecognized device fingerprint detected")

        if is_new_location:
            reasons.append("Transaction originated from a new/unusual geographical location")

        if not reasons and proba < 0.30:
            reasons.append("Standard transaction profile with low risk signals")
        elif not reasons:
            reasons.append("Elevated risk probability based on historical feature combinations")

        return reasons