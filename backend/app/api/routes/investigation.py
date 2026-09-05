from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional

from app.investigation.context import investigation_service
from app.investigation.orchestrator import orchestrator
from app.investigation.report import report_generator
from app.investigation.schemas import InvestigationReport
from app.services.transaction_store import transaction_store
from app.db.repositories import transaction_repo


router = APIRouter()


def _lookup_transaction(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Try in-memory store first, then fall back to Supabase."""
    txn = transaction_store.get(transaction_id)
    if txn is not None:
        return txn
    try:
        txn = transaction_repo.get_by_transaction_id(transaction_id)
        if txn is not None:
            transaction_store.put(txn)
            return txn
    except Exception:
        pass
    return None


@router.get("/investigation/{transaction_id}/context")
async def get_investigation_context(transaction_id: str):
    """Retrieve the deterministic investigation context for a transaction.

    Gathers ML prediction, SHAP explanations, graph evidence, network risk,
    and cluster information into a single structured response.
    Does not call any LLM or agent framework.
    """
    transaction = _lookup_transaction(transaction_id)
    context = investigation_service.build_context(transaction_id, transaction)
    return context.model_dump()


@router.get("/investigation/{transaction_id}", response_model=InvestigationReport)
async def get_investigation_report(transaction_id: str):
    """Run the full investigation pipeline and return a structured report.

    Builds the investigation context, runs all agents through the orchestrator,
    and generates a deterministic report with conclusion and recommended action.
    """
    transaction = _lookup_transaction(transaction_id)
    context = investigation_service.build_context(transaction_id, transaction)
    result = orchestrator.investigate(context)
    report = report_generator.generate(result)
    return report
