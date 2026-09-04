from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, Optional

from app.investigation.context import investigation_service


router = APIRouter()


@router.get("/investigation/{transaction_id}/context")
async def get_investigation_context(transaction_id: str):
    """Retrieve the deterministic investigation context for a transaction.

    Gathers ML prediction, SHAP explanations, graph evidence, network risk,
    and cluster information into a single structured response.
    Does not call any LLM or agent framework.
    """
    context = investigation_service.build_context(transaction_id)
    return context.model_dump()
