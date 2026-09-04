from fastapi import APIRouter, HTTPException, status
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from app.graph.graph_service import graph_service
from app.services.prediction_service import prediction_service


router = APIRouter()


class GraphBuildRequest(BaseModel):
    transactions: List[Dict[str, Any]]


class GraphTransactionRiskRequest(BaseModel):
    transaction_id: str


class GraphClusterResponse(BaseModel):
    cluster_id: int
    transaction_ids: List[str]
    entity_count: int
    entity_types: List[str]
    total_transactions: int
    suspicious_transaction_count: int
    suspicious_ratio: float
    shared_identifiers: Dict[str, List[str]]
    risk_level: str


class GraphClusterListResponse(BaseModel):
    clusters: List[GraphClusterResponse]
    total_clusters: int
    total_transactions_in_clusters: int


@router.get("/graph/status")
async def graph_status():
    return {
        "status": "ready" if graph_service.is_ready else "not_built",
        "transaction_count": graph_service.transaction_count,
        "entity_count": graph_service.entity_count,
        "edge_count": graph_service.edge_count,
        "last_built": graph_service.last_built.isoformat() if graph_service.last_built else None,
    }


@router.post("/graph/build")
async def graph_build(request: GraphBuildRequest):
    try:
        risk_results = {}
        for txn in request.transactions:
            txn_id = txn.get("transaction_id")
            if txn_id and prediction_service.is_ready:
                try:
                    result = prediction_service.predict(txn)
                    risk_results[txn_id] = {
                        "fraud_probability": result["fraud_probability"],
                        "risk_score": result["risk_score"],
                        "risk_level": result["risk_level"],
                    }
                except Exception:
                    pass

        graph_service.build(request.transactions, risk_results)
        return {
            "status": "built",
            "transaction_count": graph_service.transaction_count,
            "entity_count": graph_service.entity_count,
            "edge_count": graph_service.edge_count,
            "build_timestamp": graph_service.last_built.isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Graph build error: {str(e)}",
        )


@router.get("/graph/transaction/{transaction_id}")
async def graph_transaction_info(transaction_id: str):
    try:
        entities = graph_service.get_transaction_entities(transaction_id)
        return {
            "transaction_id": transaction_id,
            "entities": entities,
            "entity_count": len(entities),
        }
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not built",
        )


@router.get("/graph/transaction/{transaction_id}/connections")
async def graph_transaction_connections(transaction_id: str):
    try:
        result = graph_service.get_connected_transactions(transaction_id)
        return result
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not built",
        )


@router.get("/graph/transaction/{transaction_id}/neighborhood")
async def graph_transaction_neighborhood(transaction_id: str, max_hops: int = 2):
    try:
        result = graph_service.get_neighborhood(transaction_id, max_hops)
        return result
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not built",
        )


@router.get("/graph/transaction/{transaction_id}/risk")
async def graph_transaction_risk(transaction_id: str):
    try:
        txn_risk = graph_service._builder.get_transaction_risk(transaction_id)
        ml_risk_score = txn_risk.get("risk_score", 0) if txn_risk else 0
        ml_risk_level = txn_risk.get("risk_level", "UNKNOWN") if txn_risk else "UNKNOWN"
        result = graph_service.get_network_risk(transaction_id, ml_risk_score, ml_risk_level)
        result["transaction_id"] = transaction_id
        return result
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not built",
        )


@router.get("/graph/clusters")
async def graph_clusters():
    try:
        result = graph_service.get_clusters()
        return result
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not built",
        )


@router.get("/graph/clusters/{transaction_id}")
async def graph_cluster_for_transaction(transaction_id: str):
    try:
        result = graph_service.get_cluster_for_transaction(transaction_id)
        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction {transaction_id} not found in graph",
            )
        return result
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph not built",
        )
