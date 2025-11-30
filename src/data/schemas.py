from typing import TypedDict

class TransactionSchema(TypedDict):
    transaction_id: str
    user_id: int
    timestamp: str
    amount: float
    sender_account_age_days: int
    recipient_bank: str
    is_first_time_recipient: int
    internal_risk_score: float
    max_credit_limit: float
    historical_default_rate: float
    time_since_last_login: int
    change_of_device: int
    is_fraud: int
