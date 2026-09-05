from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from app.db.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String, unique=True, nullable=False, index=True)
    merchant_id = Column(String, nullable=False)
    customer_id = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    device_id = Column(String, default="UNKNOWN")
    is_new_device = Column(Boolean, default=False)
    location = Column(String, default="UNKNOWN")
    is_new_location = Column(Boolean, default=False)
    payment_method = Column(String, default="credit_card")
    velocity_5m = Column(Integer, default=1)
    failed_attempts_24h = Column(Integer, default=0)
    ProductCD = Column(String, nullable=True)
    card1 = Column(Float, nullable=True)
    card2 = Column(Float, nullable=True)
    card3 = Column(Float, nullable=True)
    card4 = Column(String, nullable=True)
    card5 = Column(Float, nullable=True)
    card6 = Column(String, nullable=True)
    addr1 = Column(Float, nullable=True)
    addr2 = Column(Float, nullable=True)
    dist1 = Column(Float, nullable=True)
    dist2 = Column(Float, nullable=True)
    P_emaildomain = Column(String, nullable=True)
    R_emaildomain = Column(String, nullable=True)
    DeviceType = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="transaction", cascade="all, delete-orphan")
    investigations = relationship("Investigation", back_populates="transaction", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id", ondelete="CASCADE"), nullable=False, index=True)
    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False)
    recommended_action = Column(String, nullable=False)
    prediction_timestamp = Column(DateTime, default=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="predictions")
    risk_factors = relationship("RiskFactor", back_populates="prediction", cascade="all, delete-orphan")


class RiskFactor(Base):
    __tablename__ = "risk_factors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    prediction_id = Column(UUID(as_uuid=True), ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    feature = Column(String, nullable=False)
    impact = Column(Float, nullable=False)
    direction = Column(String, nullable=False)
    description = Column(Text, nullable=True)

    prediction = relationship("Prediction", back_populates="risk_factors")


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(String, default="pending")
    conclusion = Column(Text, nullable=True)
    recommended_action = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transaction = relationship("Transaction", back_populates="investigations")
    evidence = relationship("InvestigationEvidence", back_populates="investigation", cascade="all, delete-orphan")
    patterns = relationship("DetectedPattern", back_populates="investigation", cascade="all, delete-orphan")
    agent_results = relationship("AgentResult", back_populates="investigation", cascade="all, delete-orphan")


class InvestigationEvidence(Base):
    __tablename__ = "investigation_evidence"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    investigation_id = Column(UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True)
    evidence_type = Column(String, nullable=False)
    source = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    details = Column(JSONB, default={})
    available = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    investigation = relationship("Investigation", back_populates="evidence")


class DetectedPattern(Base):
    __tablename__ = "detected_patterns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    investigation_id = Column(UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True)
    pattern_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    evidence = Column(JSONB, default={})
    severity = Column(String, default="UNKNOWN")
    created_at = Column(DateTime, default=datetime.utcnow)

    investigation = relationship("Investigation", back_populates="patterns")


class AgentResult(Base):
    __tablename__ = "agent_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    investigation_id = Column(UUID(as_uuid=True), ForeignKey("investigations.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_name = Column(String, nullable=False)
    result = Column(JSONB, nullable=False)
    status = Column(String, default="success")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    investigation = relationship("Investigation", back_populates="agent_results")


class GraphEdge(Base):
    __tablename__ = "graph_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship = Column(String, nullable=False)
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Entity(Base):
    __tablename__ = "entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type = Column(String, nullable=False, index=True)
    entity_value = Column(String, nullable=False, index=True)
    normalized_value = Column(String, nullable=True)
    node_key = Column(String, nullable=False, unique=True, index=True)
    first_seen_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow)


class TransactionEntity(Base):
    __tablename__ = "transaction_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    transaction_id = Column(String, ForeignKey("transactions.transaction_id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship = Column(String, nullable=False)



