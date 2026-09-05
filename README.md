
````markdown
# AI Risk Manager

### Intelligent Fraud Detection, Graph Intelligence & Agentic Investigation

AI Risk Manager is an AI-powered fraud detection and investigation system designed to identify suspicious transactions, explain why they are risky, uncover hidden relationships between transactions, and provide an actionable investigation report.

The system combines:

- Machine Learning-based fraud prediction
- SHAP-based explainability
- Graph-based transaction relationship analysis
- Network risk analysis
- Cluster intelligence
- Deterministic agentic investigation
- Supabase persistence
- A lightweight web dashboard

> **AI Risk Manager doesn't just predict fraud — it explains, connects, investigates and helps act.**

---

## Project Objective

Traditional fraud detection systems often stop after assigning a fraud probability.

AI Risk Manager goes further by answering four important questions:

1. **Is this transaction risky?**
2. **Why is it risky?**
3. **Is it connected to other suspicious activity?**
4. **What action should be taken?**

The system combines ML prediction, explainable AI, graph intelligence and agent-based investigation into a single workflow.

---

## Key Features

### 1. ML Fraud Detection

The system uses the IEEE-CIS Fraud Detection dataset and an XGBoost classification model.

The ML pipeline performs:

```text
Transaction Data
       ↓
Preprocessing
       ↓
Feature Engineering
       ↓
XGBoost Model
       ↓
Fraud Probability
       ↓
Risk Score
       ↓
Risk Level
       ↓
Recommended Action
````

The model uses an optimized decision threshold of `0.40`.

### Model Evaluation

| Metric              | Result |
| ------------------- | -----: |
| Precision           | 36.93% |
| Recall              | 83.74% |
| F1 Score            | 51.26% |
| PR-AUC              | 0.7173 |
| ROC-AUC             | 0.9640 |
| False Positive Rate |  5.18% |
| False Negative Rate | 16.26% |
| Decision Threshold  |   0.40 |

The threshold prioritizes detecting fraudulent transactions while maintaining a manageable false-positive rate.

---

## 2. Explainable AI with SHAP

AI Risk Manager does not treat the ML model as a black box.

SHAP is used to explain individual predictions by identifying:

* Features increasing risk
* Features reducing risk
* Relative feature impact

For every investigation, the dashboard can display:

```text
Top Risk Factors
        +
Top Risk Reducers
        ↓
Explanation of ML Decision
```

This helps investigators understand the reasoning behind a prediction instead of relying only on a numerical score.

---

## 3. Graph Intelligence

Fraud can involve relationships between multiple transactions.

AI Risk Manager builds a transaction-entity graph using relationships such as:

* Customer
* Merchant
* Device
* Card
* Email domain

Example:

```text
             Device
               |
               |
Transaction ─ Entity ─ Transaction
     |           |
   Card       Merchant
```

The graph layer can identify:

* Shared entities
* Connected transactions
* Suspicious neighbors
* Connection density
* Transaction neighborhoods
* Cluster membership

This allows the system to detect patterns that may not be visible from a single transaction alone.

---

## 4. Network Risk Analysis

Graph relationships are incorporated into a network risk analysis layer.

The system evaluates factors such as:

* Number of connected transactions
* Shared entities
* Connection density
* Suspicious neighboring transactions

The result provides:

```text
ML Risk Score
      +
Network Risk
      ↓
Combined Risk Assessment
```

This provides additional context around an individual transaction.

---

## 5. Cluster Intelligence

Connected transactions are grouped into graph-based clusters.

Cluster analysis provides information such as:

* Number of transactions
* Number of entities
* Suspicious transaction count
* Suspicious transaction ratio
* Average risk score
* Strong entity types
* Weak entity types
* Shared identifiers

This helps identify groups of transactions that may be related.

---

# Agentic Investigation

AI Risk Manager uses three deterministic investigation agents.

## Risk Agent

The Risk Agent evaluates the overall risk assessment using the existing ML and graph-derived risk information.

It produces:

* Risk level
* Risk score
* Assessment
* Risk factors
* Risk reducers
* Evidence summary

---

## Pattern Agent

The Pattern Agent identifies suspicious patterns from the investigation context.

It evaluates patterns such as:

* Suspicious neighbors
* Shared entities
* Cluster membership
* Dense connections

The agent reports detected patterns and their severity.

---

## Evidence Agent

The Evidence Agent collects supporting evidence from different system components.

Evidence categories include:

* Transaction
* ML prediction
* SHAP
* Graph
* Network risk
* Cluster

Unavailable information is explicitly marked as unavailable rather than being fabricated.

---

## Investigation Workflow

The complete investigation pipeline is:

```text
Transaction
     ↓
ML Prediction
     ↓
SHAP Explanation
     ↓
Graph Analysis
     ↓
Network Risk
     ↓
Cluster Detection
     ↓
Risk Agent
     ↓
Pattern Agent
     ↓
Evidence Agent
     ↓
Investigation Report
     ↓
Recommended Action
```

The orchestrator coordinates the three investigation agents without creating a separate competing risk engine.

---

# Investigation Report

The final investigation report combines all available intelligence into one result.

It contains:

* Transaction information
* Risk assessment
* Detected patterns
* Evidence items
* Conclusion
* Recommended action
* Agent errors, when applicable
* Investigation metadata

The system explicitly handles missing or unavailable evidence.

---

# Web Dashboard

The frontend is intentionally lightweight and framework-free.

### Technologies

* HTML
* CSS
* Vanilla JavaScript

No React, Vue, Angular, Tailwind, Bootstrap or frontend framework is required.

The dashboard provides:

### Risk Overview

* Risk level
* Risk score
* Fraud probability
* Recommended action

### Risk Score Analysis

Visual representation of the risk assessment.

### SHAP Explanation

Displays:

* Risk-increasing features
* Risk-reducing features
* Feature impact

### Graph Intelligence

Displays:

* Connected transactions
* Shared entities
* Suspicious neighbors
* Neighborhood information

### Network Risk

Displays:

* Network risk score
* Combined risk score
* Network factors
* Neighbor information

### Cluster Intelligence

Displays:

* Cluster size
* Entity count
* Suspicious transaction ratio
* Strong and weak entity types
* Shared identifiers

### Agent Investigation

Displays results from:

* Risk Agent
* Pattern Agent
* Evidence Agent

### Investigation Report

Provides the final consolidated investigation result.

---

# Supabase Integration

Supabase is used for persistent storage of important transaction and prediction information.

The active database schema contains:

* `transactions`
* `predictions`
* `risk_factors`
* `investigations`
* `investigation_evidence`
* `detected_patterns`
* `agent_results`
* `graph_edges`
* `entities`
* `transaction_entities`

The system persists:

* Transactions
* ML predictions
* SHAP risk factors and reducers

The investigation layer can fall back to Supabase when transaction information is not available in the in-memory transaction store.

This allows transaction investigations to continue across backend restarts.

> The current graph intelligence layer is maintained in memory and is rebuilt when the backend restarts.

---

# System Architecture

```text
                    ┌─────────────────────┐
                    │   Web Dashboard     │
                    │ HTML/CSS/JavaScript │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      Backend        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌─────────────┐   ┌─────────────┐
       │ ML Model   │   │   Graph     │   │ Investigation│
       │  XGBoost   │   │ Intelligence│   │    Agents    │
       └─────┬──────┘   └──────┬──────┘   └──────┬──────┘
             │                 │                  │
             ▼                 ▼                  ▼
          SHAP            Network Risk       Risk Agent
                           Cluster           Pattern Agent
                                             Evidence Agent
                                                  │
                                                  ▼
                                        Investigation Report

                               │
                               ▼
                         ┌─────────────┐
                         │  Supabase   │
                         │ Persistence │
                         └─────────────┘
```

---

# Technology Stack

| Component            | Technology                    |
| -------------------- | ----------------------------- |
| Programming Language | Python                        |
| Backend              | FastAPI                       |
| ML Model             | XGBoost                       |
| Explainability       | SHAP                          |
| Graph Intelligence   | Custom Python Graph Layer     |
| Agents               | Deterministic Python Agents   |
| Database             | Supabase                      |
| Frontend             | HTML, CSS, Vanilla JavaScript |
| Model Serialization  | Joblib                        |
| Testing              | Pytest                        |

---

# Project Structure

```text
RiskPred/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── graph/
│   │   ├── investigation/
│   │   ├── models/
│   │   ├── schemas/
│   │   └── services/
│   │
│   ├── ml/
│   │   └── artifacts/
│   │
│   └── tests/
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   └── js/
│       ├── api.js
│       ├── app.js
│       ├── charts.js
│       └── components.js
│
├── supabase/
│   ├── schema.sql
│   └── policies.sql
│
├── start.bat
├── requirements.txt
└── README.md
```

---

# API Endpoints

## Prediction

```text
POST /api/v1/predict
```

Generates a fraud prediction for a transaction.

---

## Investigation

```text
GET /api/v1/investigation/{transaction_id}
```

Returns the complete investigation report.

---

## Investigation Context

```text
GET /api/v1/investigation/{transaction_id}/context
```

Returns the investigation context used by the agents.

---

## Graph

```text
POST /api/v1/graph/build

GET /api/v1/graph/status

GET /api/v1/graph/transaction/{transaction_id}

GET /api/v1/graph/transaction/{transaction_id}/connections

GET /api/v1/graph/transaction/{transaction_id}/neighborhood

GET /api/v1/graph/transaction/{transaction_id}/risk

GET /api/v1/graph/clusters

GET /api/v1/graph/clusters/{transaction_id}
```

---

# Running the Project

## 1. Create and activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

---

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

---

## 3. Configure environment variables

Create a `.env` file in the project root.

Required Supabase configuration:

```text
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
```

Do not commit real Supabase credentials to GitHub.

---

## 4. Start the application

The project includes:

```text
start.bat
```

Run:

```powershell
.\start.bat
```

The application will start on:

```text
http://127.0.0.1:8000
```

Dashboard:

```text
http://127.0.0.1:8000/dashboard
```

The FastAPI backend serves the frontend, so the dashboard and API use the same port.

---

# Example Prediction

Example transaction:

```json
{
  "transaction_id": "TEST_TXN_001",
  "merchant_id": "MERCHANT_001",
  "customer_id": "CUST_1001",
  "amount": 250.00
}
```

Example response:

```text
Transaction ID: TEST_TXN_001
Fraud Probability: 18.09%
Risk Score: 18
Risk Level: LOW
Recommended Action: ALLOW
```

The exact prediction depends on the transaction features supplied to the model.

---

# Testing

The project contains regression and component-level tests covering:

* Prediction
* Graph intelligence
* Investigation context
* Risk Agent
* Pattern Agent
* Evidence Agent
* Investigation orchestration
* Investigation reports
* API routes
* Supabase persistence
* Frontend-related integration behavior

Latest complete test verification:

```text
635 passed
0 failed
```
---

# Future Scope

Possible future improvements include:

* Persistent graph storage
* Real-time transaction streaming
* Larger-scale graph analytics
* Advanced anomaly detection
* Continuous model monitoring
* Investigator feedback loops
* More sophisticated agent reasoning
* Production-grade authentication and authorization
* Real-time fraud alerts

---

# Project Vision

AI Risk Manager aims to move fraud detection from:

```text
Prediction
```

to:

```text
Prediction
     ↓
Explanation
     ↓
Connection
     ↓
Investigation
     ↓
Action
```

The goal is to help fraud investigators make faster, more explainable and better-informed decisions.

---

## Final Statement

**AI Risk Manager doesn't just predict fraud — it explains, connects, investigates and helps act.**


