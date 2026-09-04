from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class MLPredictionEvidence(BaseModel):
    fraud_probability: float
    risk_score: int
    risk_level: str
    recommended_action: str


class SHAPEvidence(BaseModel):
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    risk_reducers: List[Dict[str, Any]] = Field(default_factory=list)


class GraphEvidence(BaseModel):
    connected_transactions: List[Dict[str, Any]] = Field(default_factory=list)
    total_connections: int = 0
    entities: List[Dict[str, str]] = Field(default_factory=list)
    entity_count: int = 0
    neighborhood_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    neighborhood_edges: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_neighbors: List[Dict[str, Any]] = Field(default_factory=list)
    suspicious_neighbor_count: int = 0
    shared_entity_types: List[str] = Field(default_factory=list)
    graph_available: bool = False


class NetworkRiskEvidence(BaseModel):
    network_risk_score: float = 0.0
    network_risk_level: str = "UNKNOWN"
    combined_risk_score: float = 0.0
    combined_risk_level: str = "UNKNOWN"
    factors: List[Dict[str, Any]] = Field(default_factory=list)
    neighbor_count: int = 0
    suspicious_neighbor_count: int = 0


class ClusterEvidence(BaseModel):
    found: bool = False
    transaction_ids: List[str] = Field(default_factory=list)
    total_transactions: int = 0
    entity_count: int = 0
    entity_types: List[str] = Field(default_factory=list)
    suspicious_transaction_count: int = 0
    suspicious_ratio: float = 0.0
    shared_identifiers: Dict[str, List[str]] = Field(default_factory=dict)
    risk_level: str = "UNKNOWN"
    avg_risk_score: float = 0.0
    strong_entity_types: List[str] = Field(default_factory=list)
    weak_entity_types: List[str] = Field(default_factory=list)


class InvestigationContext(BaseModel):
    transaction_id: str
    transaction: Optional[Dict[str, Any]] = None
    ml_prediction: Optional[MLPredictionEvidence] = None
    shap_explanation: Optional[SHAPEvidence] = None
    graph: GraphEvidence = Field(default_factory=GraphEvidence)
    network_risk: Optional[NetworkRiskEvidence] = None
    cluster: Optional[ClusterEvidence] = None


class RiskAgentResult(BaseModel):
    transaction_id: str
    risk_level: str
    risk_score: float
    assessment: str
    reasons: List[str] = Field(default_factory=list)
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list)
    risk_reducers: List[Dict[str, Any]] = Field(default_factory=list)
    evidence_summary: Dict[str, Any] = Field(default_factory=dict)
