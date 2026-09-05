from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class EntityInfo(BaseModel):
    type: str
    value: str
    normalized_value: Optional[str] = None


class ExtractedEntities(BaseModel):
    transaction_id: str
    entities: Dict[str, List[str]] = Field(default_factory=dict)


class ConnectedTransaction(BaseModel):
    transaction_id: str
    shared_entities: List[EntityInfo]


class TransactionConnections(BaseModel):
    transaction_id: str
    connected_transactions: List[ConnectedTransaction]
    total_connections: int


class SuspiciousNeighbor(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    shared_entities: List[EntityInfo]


class NeighborhoodRisk(BaseModel):
    transaction_id: str
    ml_risk_score: float
    ml_risk_level: str
    neighbor_count: int
    suspicious_neighbor_count: int
    suspicious_neighbors: List[SuspiciousNeighbor]
    shared_entity_types: List[str]
    network_context_added: bool


class FraudCluster(BaseModel):
    cluster_id: int
    transaction_ids: List[str]
    entity_count: int
    entity_types: List[str]
    total_transactions: int
    suspicious_transaction_count: int
    suspicious_ratio: float
    shared_identifiers: Dict[str, List[str]]
    risk_level: str


class ClusterList(BaseModel):
    clusters: List[FraudCluster]
    total_clusters: int
    total_transactions_in_clusters: int


class NetworkRiskResult(BaseModel):
    transaction_id: str
    ml_risk_score: float
    ml_risk_level: str
    network_risk_score: float
    network_risk_level: str
    combined_risk_score: float
    combined_risk_level: str
    factors: List[Dict[str, Any]]
    neighbor_count: int
    suspicious_neighbor_count: int
    graph_available: bool


class GraphBuildResponse(BaseModel):
    status: str
    transaction_count: int
    entity_count: int
    edge_count: int
    build_timestamp: datetime


class GraphStatusResponse(BaseModel):
    status: str
    transaction_count: int
    entity_count: int
    edge_count: int
    last_built: Optional[datetime] = None
