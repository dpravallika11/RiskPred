-- RiskPred Row Level Security (RLS) Policies
-- Run this AFTER schema.sql in Supabase SQL Editor
--
-- SECURITY MODEL:
--   - Backend uses service_role key (bypasses RLS) for ALL writes
--   - Anon key: SELECT-only on data tables (for potential future frontend reads)
--   - Authenticated key: SELECT-only on data tables
--   - No anon/authenticated INSERT/UPDATE/DELETE on any table
--   - SUPABASE_SERVICE_ROLE_KEY is NEVER exposed to frontend JavaScript

-- ============================================================
-- ENABLE ROW LEVEL SECURITY ON ALL TABLES
-- ============================================================
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_factors ENABLE ROW LEVEL SECURITY;
ALTER TABLE investigations ENABLE ROW LEVEL SECURITY;
ALTER TABLE investigation_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE detected_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE graph_edges ENABLE ROW LEVEL SECURITY;
ALTER TABLE entities ENABLE ROW LEVEL SECURITY;
ALTER TABLE transaction_entities ENABLE ROW LEVEL SECURITY;

-- ============================================================
-- POLICIES: TRANSACTIONS (SELECT only for anon/authenticated)
-- ============================================================
CREATE POLICY "Transactions: Read access for anon"
    ON transactions FOR SELECT TO anon USING (true);

CREATE POLICY "Transactions: Read access for authenticated"
    ON transactions FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: PREDICTIONS (SELECT only)
-- ============================================================
CREATE POLICY "Predictions: Read access for anon"
    ON predictions FOR SELECT TO anon USING (true);

CREATE POLICY "Predictions: Read access for authenticated"
    ON predictions FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: RISK FACTORS (SELECT only)
-- ============================================================
CREATE POLICY "Risk Factors: Read access for anon"
    ON risk_factors FOR SELECT TO anon USING (true);

CREATE POLICY "Risk Factors: Read access for authenticated"
    ON risk_factors FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: INVESTIGATIONS (SELECT only)
-- ============================================================
CREATE POLICY "Investigations: Read access for anon"
    ON investigations FOR SELECT TO anon USING (true);

CREATE POLICY "Investigations: Read access for authenticated"
    ON investigations FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: INVESTIGATION EVIDENCE (SELECT only)
-- ============================================================
CREATE POLICY "Evidence: Read access for anon"
    ON investigation_evidence FOR SELECT TO anon USING (true);

CREATE POLICY "Evidence: Read access for authenticated"
    ON investigation_evidence FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: DETECTED PATTERNS (SELECT only)
-- ============================================================
CREATE POLICY "Patterns: Read access for anon"
    ON detected_patterns FOR SELECT TO anon USING (true);

CREATE POLICY "Patterns: Read access for authenticated"
    ON detected_patterns FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: AGENT RESULTS (SELECT only)
-- ============================================================
CREATE POLICY "Agent Results: Read access for anon"
    ON agent_results FOR SELECT TO anon USING (true);

CREATE POLICY "Agent Results: Read access for authenticated"
    ON agent_results FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: GRAPH EDGES (SELECT only)
-- ============================================================
CREATE POLICY "Graph Edges: Read access for anon"
    ON graph_edges FOR SELECT TO anon USING (true);

CREATE POLICY "Graph Edges: Read access for authenticated"
    ON graph_edges FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: ENTITIES (SELECT only)
-- ============================================================
CREATE POLICY "Entities: Read access for anon"
    ON entities FOR SELECT TO anon USING (true);

CREATE POLICY "Entities: Read access for authenticated"
    ON entities FOR SELECT TO authenticated USING (true);

-- ============================================================
-- POLICIES: TRANSACTION_ENTITIES (SELECT only)
-- ============================================================
CREATE POLICY "Tx Entities: Read access for anon"
    ON transaction_entities FOR SELECT TO anon USING (true);

CREATE POLICY "Tx Entities: Read access for authenticated"
    ON transaction_entities FOR SELECT TO authenticated USING (true);
