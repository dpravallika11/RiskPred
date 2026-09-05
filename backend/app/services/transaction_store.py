from typing import Any, Dict, Optional


class TransactionStore:
    """Simple in-memory store for transactions submitted via predict or graph build.

    Allows the investigation endpoint to look up transaction data by ID
    so that ML prediction can run during investigation.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def put(self, transaction: Dict[str, Any]) -> None:
        txn_id = transaction.get("transaction_id")
        if txn_id:
            self._store[txn_id] = transaction

    def get(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(transaction_id)

    def clear(self) -> None:
        self._store.clear()

    @property
    def count(self) -> int:
        return len(self._store)


transaction_store = TransactionStore()
