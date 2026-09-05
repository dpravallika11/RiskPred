-- RiskPred Supabase Schema
-- Run this in Supabase SQL Editor to create all tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- 1. TRANSACTIONS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id TEXT UNIQUE NOT NULL,
    merchant_id TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    amount NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    device_id TEXT DEFAULT 'UNKNOWN',
    is_new_device BOOLEAN DEFAULT FALSE,
    location TEXT DEFAULT 'UNKNOWN',
    is_new_location BOOLEAN DEFAULT FALSE,
    payment_method TEXT DEFAULT 'credit_card',
    velocity_5m INTEGER DEFAULT 1 CHECK (velocity_5m >= 0),
    failed_attempts_24h INTEGER DEFAULT 0 CHECK (failed_attempts_24h >= 0),
    ProductCD TEXT,
    card1 NUMERIC,
    card2 NUMERIC,
    card3 NUMERIC,
    card4 TEXT,
    card5 NUMERIC,
    card6 TEXT,
    addr1 NUMERIC,
    addr2 NUMERIC,
    dist1 NUMERIC,
    dist2 NUMERIC,
    P_emaildomain TEXT,
    R_emaildomain TEXT,
    DeviceType TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 2. PREDICTIONS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    fraud_probability NUMERIC(5,4) NOT NULL CHECK (fraud_probability >= 0 AND fraud_probability <= 1),
    risk_score INTEGER NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
    risk_level TEXT NOT NULL CHECK (risk_level IN ('LOW', 'MEDIUM', 'HIGH')),
    recommended_action TEXT NOT NULL,
    prediction_timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_transaction_id ON predictions(transaction_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level);

-- ============================================================
-- 3. RISK FACTORS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS risk_factors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    feature TEXT NOT NULL,
    impact NUMERIC(8,6) NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('increases_risk', 'decreases_risk')),
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_risk_factors_prediction_id ON risk_factors(prediction_id);

-- ============================================================
-- 4. INVESTIGATIONS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS investigations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    conclusion TEXT,
    recommended_action TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_investigations_transaction_id ON investigations(transaction_id);
CREATE INDEX IF NOT EXISTS idx_investigations_status ON investigations(status);

-- ============================================================
-- 5. INVESTIGATION EVIDENCE TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS investigation_evidence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigation_id UUID NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    source TEXT NOT NULL,
    description TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    available BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_evidence_investigation_id ON investigation_evidence(investigation_id);

-- ============================================================
-- 6. DETECTED PATTERNS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS detected_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigation_id UUID NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    pattern_type TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence JSONB DEFAULT '{}',
    severity TEXT DEFAULT 'UNKNOWN',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patterns_investigation_id ON detected_patterns(investigation_id);

-- ============================================================
-- 7. AGENT RESULTS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    investigation_id UUID NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
    agent_name TEXT NOT NULL CHECK (agent_name IN ('RiskAgent', 'PatternAgent', 'EvidenceAgent')),
    result JSONB NOT NULL,
    status TEXT DEFAULT 'success' CHECK (status IN ('success', 'error')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_results_investigation_id ON agent_results(investigation_id);

-- ============================================================
-- 8. ENTITIES TABLE (for entity resolution)
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    entity_type TEXT NOT NULL,
    entity_value TEXT NOT NULL,
    normalized_value TEXT,
    node_key TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(entity_type, entity_value),
    UNIQUE(node_key)
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_value ON entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_entities_node_key ON entities(node_key);

-- ============================================================
-- 9. GRAPH EDGES TABLE (TransactionNode ↔ EntityNode)
-- ============================================================
CREATE TABLE IF NOT EXISTS graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    weight NUMERIC(5,2) DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(transaction_id, entity_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_transaction ON graph_edges(transaction_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_entity ON graph_edges(entity_id);

-- ============================================================
-- 10. TRANSACTION_ENTITIES JUNCTION TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS transaction_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relationship TEXT NOT NULL,
    UNIQUE(transaction_id, entity_id, relationship)
);

CREATE INDEX IF NOT EXISTS idx_tx_entities_transaction ON transaction_entities(transaction_id);
CREATE INDEX IF NOT EXISTS idx_tx_entities_entity ON transaction_entities(entity_id);

-- ============================================================
-- FUNCTIONS: Auto-update updated_at timestamp
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to transactions
CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to investigations
CREATE TRIGGER update_investigations_updated_at
    BEFORE UPDATE ON investigations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


