# RiskPred — AI Risk Manager

RiskPred is an AI-powered fraud risk management system designed to help merchants and risk analysts investigate suspicious financial transactions.

The system combines machine-learning-based transaction risk prediction with explainability, graph-based intelligence, and a deterministic multi-agent investigation pipeline.

---

## Overview

RiskPred follows this investigation flow:

```text
Transaction
     │
     ▼
ML Risk Prediction
     │
     ├── Fraud Probability
     ├── Risk Score
     ├── Risk Level
     ├── Recommended Action
     └── SHAP Explanation
     │
     ▼
Graph Intelligence
     │
     ├── Connections
     ├── Suspicious Neighbors
     ├── Shared Entities
     ├── Network Risk
     └── Cluster Intelligence
     │
     ▼
Agent Investigation
     │
     ├── Risk Agent
     ├── Pattern Agent
     └── Evidence Agent
     │
     ▼
Investigation Report
     │
     ├── Risk Assessment
     ├── Detected Patterns
     ├── Evidence
     ├── Conclusion
     └── Recommended Action
```

---

## Key Features

### Machine Learning Risk Prediction

RiskPred uses the existing trained fraud-detection model to evaluate transactions and provide:

* Fraud probability
* Risk score
* Risk level
* Recommended action
* Risk factors
* Risk reducers
* SHAP-based explanation

The prediction service is reused throughout the investigation pipeline rather than creating a separate risk-scoring system.

### Explainable AI

RiskPred provides explainability through SHAP-derived:

* Risk factors
* Risk reducers

This allows analysts to understand which transaction characteristics contribute to the predicted risk.

### Graph Intelligence

The system builds transaction/entity relationships and provides graph-based investigation capabilities including:

* Transaction connections
* Transaction neighborhoods
* Suspicious neighbors
* Shared entities
* Network risk
* Cluster intelligence

Graph evidence is only displayed when it is actually available.

### Multi-Agent Investigation

RiskPred contains a deterministic investigation pipeline consisting of:

#### Risk Agent

Evaluates the available risk information and produces a structured risk assessment.

#### Pattern Agent

Identifies suspicious patterns from the available graph and transaction context.

#### Evidence Agent

Collects and structures supporting evidence from the available investigation sources.

#### Investigation Orchestrator

Coordinates the agents in the following order:

```text
Risk Agent
     ↓
Pattern Agent
     ↓
Evidence Agent
```

Agent failures are isolated so that one failed agent does not prevent the remaining investigation components from executing.

### Investigation Report

The investigation pipeline produces a structured report containing:

* Risk assessment
* Detected patterns
* Evidence
* Conclusion
* Recommended action
* Agent errors
* Metadata

Unavailable evidence is explicitly represented instead of being fabricated.

### Risk Analyst Dashboard

The frontend provides an AI Risk Manager interface for investigating transactions and viewing:

* Risk overview
* Transaction risk
* Fraud probability
* Risk score
* Risk level
* SHAP explanations
* Graph intelligence
* Network risk
* Cluster information
* Agent investigation
* Evidence
* Investigation conclusion
* Recommended action

The dashboard is implemented using:

* HTML
* CSS
* Vanilla JavaScript

No frontend framework is required.

---

## Architecture

```text
RiskPred
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── health.py
│   │   │       ├── predictions.py
│   │   │       ├── graph.py
│   │   │       └── investigation.py
│   │   │
│   │   ├── graph/
│   │   │   ├── entity_extractor.py
│   │   │   ├── entity_resolver.py
│   │   │   ├── graph_builder.py
│   │   │   ├── graph_queries.py
│   │   │   ├── graph_service.py
│   │   │   ├── network_risk.py
│   │   │   └── cluster_detector.py
│   │   │
│   │   ├── investigation/
│   │   │   ├── context.py
│   │   │   ├── schemas.py
│   │   │   ├── risk_agent.py
│   │   │   ├── pattern_agent.py
│   │   │   ├── evidence_agent.py
│   │   │   ├── orchestrator.py
│   │   │   └── report.py
│   │   │
│   │   └── services/
│   │       └── prediction_service.py
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
├── requirements.txt
└── README.md
```

---

## Backend API

The backend is built using FastAPI.

The API prefix is configured by the application's backend settings.

### Prediction

```text
POST /api/v1/predict
```

Used to obtain the existing ML-based transaction risk prediction.

### Investigation Context

```text
GET /api/v1/investigation/{transaction_id}/context
```

Returns the assembled investigation context for a transaction.

### Investigation Report

```text
GET /api/v1/investigation/{transaction_id}
```

Runs the investigation pipeline and returns the structured investigation report.

### Graph Status

```text
GET /api/v1/graph/status
```

Checks graph availability/status.

### Graph Build

```text
POST /api/v1/graph/build
```

Builds the graph using the existing graph service.

### Transaction Graph

```text
GET /api/v1/graph/transaction/{transaction_id}
```

Retrieves graph information for a transaction.

### Transaction Connections

```text
GET /api/v1/graph/transaction/{transaction_id}/connections
```

Retrieves connections associated with a transaction.

### Transaction Neighborhood

```text
GET /api/v1/graph/transaction/{transaction_id}/neighborhood
```

Supports neighborhood investigation using the existing graph implementation.

### Transaction Network Risk

```text
GET /api/v1/graph/transaction/{transaction_id}/risk
```

Retrieves network-based risk information.

### Clusters

```text
GET /api/v1/graph/clusters
```

Retrieves available graph clusters.

### Transaction Cluster

```text
GET /api/v1/graph/clusters/{transaction_id}
```

Retrieves cluster information associated with a transaction.

> API paths and request/response schemas should be treated as defined by the backend implementation.

---

## Investigation Pipeline

The investigation system is designed to preserve the distinction between prediction, graph intelligence, and investigation reasoning.

```text
InvestigationContext
        │
        ▼
InvestigationOrchestrator
        │
        ├──────────────┐
        ▼              ▼
   Risk Agent     Pattern Agent
        │              │
        └──────┬───────┘
               ▼
        Evidence Agent
               │
               ▼
      Investigation Report
```

The agents are deterministic and operate on the information available in the investigation context.

The system does not generate evidence when evidence is unavailable.

---

## Handling Missing Evidence

Risk investigations may not always have complete information.

RiskPred explicitly represents unavailable information.

Examples include:

```text
Graph intelligence unavailable
SHAP explanation unavailable
Cluster intelligence unavailable
Network risk unavailable
```

Similarly, agent failures are preserved in the investigation report.

This prevents the system from presenting fabricated information as investigative evidence.

---

## Frontend

The RiskPred dashboard is implemented with:

```text
HTML
CSS
Vanilla JavaScript
```

Frontend responsibilities are separated into:

### `api.js`

Handles communication with the FastAPI backend.

### `app.js`

Handles application state, user interactions, and dashboard orchestration.

### `components.js`

Contains reusable DOM rendering components.

### `charts.js`

Contains visualizations such as risk and SHAP representations.

The frontend does not reproduce backend fraud-detection or investigation logic.

---

## Getting Started

### Prerequisites

* Python 3.10+

### Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Start the Application

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend
```

Then open the dashboard:

```text
http://127.0.0.1:8000
```

The dashboard, static assets, and API are all served from the same FastAPI process on port `8000`. No separate frontend server is required.

---

## Testing

Backend tests are located under:

```text
backend/tests/
```

Run the complete test suite with the project's virtual environment activated:

```bash
pytest -q
```

The test suite covers the existing prediction, graph, investigation context, agents, orchestrator, report, and API functionality.

---

## Development Principles

RiskPred follows several important development principles:

### 1. Reuse Existing Intelligence

The investigation pipeline reuses the existing prediction and graph services rather than implementing competing logic.

### 2. No Fabricated Evidence

If data is unavailable, the system explicitly reports it as unavailable.

### 3. Deterministic Investigation

The current investigation agents use deterministic rules and existing system outputs.

### 4. Separation of Responsibilities

Machine learning, graph intelligence, investigation agents, API routes, and frontend presentation remain separate concerns.

### 5. Minimal Integration Changes

The dashboard consumes existing backend APIs instead of unnecessarily redesigning the backend.

### 6. Framework-Free Frontend

The current dashboard uses plain HTML, CSS, and vanilla JavaScript.

---

## Current Technology Stack

### Backend

* Python
* FastAPI
* Pydantic
* Scikit-learn
* XGBoost
* SHAP

### Graph / Investigation

* Graph-based transaction intelligence
* Network risk analysis
* Cluster detection
* Deterministic investigation agents

### Frontend

* HTML5
* CSS3
* Vanilla JavaScript
* SVG/CSS/Canvas-based visualizations where appropriate

### Testing

* Pytest

---

## Project Status

The project has progressed through:

```text
Sprint 1
   ↓
Sprint 2
   ↓
Sprint 3 — Graph Intelligence
   ↓
Sprint 4 — Investigation Agents
   ↓
Sprint 5 — Dashboard + Integration
```

Sprint 3 and Sprint 4 provide the underlying prediction, graph, investigation-context, agent, orchestration, and report functionality.

Sprint 5 provides the analyst-facing dashboard that integrates these capabilities into a single interface.

---

## Important Limitations

RiskPred intentionally does not fabricate information.

If an aggregate transaction dataset or aggregate statistics endpoint is not available, the dashboard does not invent KPI values.

Similarly, if graph, SHAP, cluster, or agent information is unavailable, the dashboard displays an appropriate unavailable state.

The system should therefore be evaluated using the actual transaction and backend data available in the project.

---

## Future Extensions

Potential future work may include:

* Persistent investigation case storage
* Merchant-specific dashboards
* Historical investigation tracking
* Advanced graph visualization
* Analyst feedback loops
* Authentication and role-based access
* Persistent analytics and reporting
* Additional fraud detection models

These are future extensions and are not required for the current Sprint 5 implementation.
