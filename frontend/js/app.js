document.addEventListener('DOMContentLoaded', () => {
    var state = { currentTxnId: null, report: null };

    var $ = (id) => document.getElementById(id);
    var C = RiskPredComponents;

    checkHealth();
    showInitialState();

    $('investigate-btn').addEventListener('click', handleInvestigate);
    $('txn-id-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') handleInvestigate(); });

    function checkHealth() {
        var badge = $('api-status-badge');
        RiskPredAPI.checkHealth()
            .then((data) => {
                if (data.status === 'healthy') {
                    badge.className = 'status-badge status-badge--online';
                    badge.innerHTML = '<span class="status-dot status-dot--online"></span> System Online';
                } else {
                    badge.className = 'status-badge status-badge--offline';
                    badge.innerHTML = '<span class="status-dot status-dot--offline"></span> Degraded';
                }
            })
            .catch(() => {
                badge.className = 'status-badge status-badge--offline';
                badge.innerHTML = '<span class="status-dot status-dot--offline"></span> Backend Offline';
            });
    }

    function showInitialState() {
        $('risk-overview-section').classList.add('hidden');
        $('investigation-section').classList.add('hidden');
        $('risk-shap-container').classList.add('hidden');
        $('empty-state').classList.remove('hidden');
    }

    function hideAllResults() {
        $('risk-overview-section').classList.add('hidden');
        $('investigation-section').classList.add('hidden');
        $('risk-shap-container').classList.add('hidden');
    }

    function handleInvestigate() {
        var input = $('txn-id-input');
        var txnId = (input.value || '').trim();
        if (!txnId) { input.focus(); return; }

        state.currentTxnId = txnId;
        hideAllResults();
        $('empty-state').classList.add('hidden');

        $('investigation-section').classList.remove('hidden');
        $('investigation-section').innerHTML = C.loadingState('Investigating transaction ' + C.esc(txnId) + '...');

        $('investigate-btn').disabled = true;

        RiskPredAPI.getInvestigationReport(txnId)
            .then((report) => {
                state.report = report;
                renderInvestigation(report);
            })
            .catch((err) => {
                $('investigation-section').innerHTML = C.errorState(
                    'Investigation Failed',
                    (err.status === 404 ? 'Transaction "' + txnId + '" not found.' :
                     err.status === 503 ? 'Backend service unavailable. Ensure the model is loaded.' :
                     err.message || 'Could not retrieve investigation report.')
                );
            })
            .finally(() => {
                $('investigate-btn').disabled = false;
            });
    }

    function renderInvestigation(report) {
        $('investigation-section').classList.add('hidden');
        $('risk-overview-section').classList.remove('hidden');
        $('risk-shap-container').classList.remove('hidden');

        renderRiskOverview(report);
        renderRiskScore(report);
        renderShap(report);
        renderGraphSection(report);
        renderAgents(report);
        renderEvidence(report);
        renderConclusion(report);
    }

    /* ── Risk Overview KPIs ──────────────────────────── */
    function renderRiskOverview(report) {
        var ra = report.risk_assessment;
        var txnId = report.transaction_id || state.currentTxnId;
        var level = ra ? ra.risk_level : 'UNKNOWN';
        var score = ra ? ra.risk_score : 0;
        var fraudP = ra && ra.evidence_summary && ra.evidence_summary.fraud_probability != null
            ? ra.evidence_summary.fraud_probability : null;

        $('kpi-txn-id').textContent = txnId;
        $('kpi-risk-level').innerHTML = C.riskBadge(level);
        $('kpi-risk-score').textContent = score;
        $('kpi-fraud-prob').textContent = fraudP != null ? (fraudP * 100).toFixed(1) + '%' : 'N/A';

        var action = report.recommended_action || (ra && ra.assessment) || 'N/A';
        $('kpi-action').textContent = action;
    }

    /* ── Risk Score Display ──────────────────────────── */
    function renderRiskScore(report) {
        var ra = report.risk_assessment;
        var score = ra ? ra.risk_score : 0;
        var level = ra ? ra.risk_level : 'UNKNOWN';
        var fraudP = ra && ra.evidence_summary && ra.evidence_summary.fraud_probability != null
            ? ra.evidence_summary.fraud_probability : null;

        var ring = C.riskScoreRing(score, level);
        var badge = C.riskBadge(level);

        var metaItems = '';
        metaItems += '<div class="risk-meta-item"><div class="risk-meta-label">Risk Level</div><div class="risk-meta-value">' + badge + '</div></div>';
        metaItems += '<div class="risk-meta-item"><div class="risk-meta-label">Fraud Probability</div><div class="risk-meta-value" style="color:' + C.riskColor(level) + '">' + (fraudP != null ? (fraudP * 100).toFixed(1) + '%' : 'N/A') + '</div></div>';
        metaItems += '<div class="risk-meta-item"><div class="risk-meta-label">Risk Score</div><div class="risk-meta-value" style="color:' + C.riskColor(level) + '">' + score + '</div></div>';
        var action = report.recommended_action || 'N/A';
        metaItems += '<div class="risk-meta-item"><div class="risk-meta-label">Recommended Action</div><div class="risk-meta-value" style="font-size:0.85rem">' + C.esc(action) + '</div></div>';

        $('risk-score-content').innerHTML =
            '<div class="risk-score-display">' +
            ring +
            '<div class="risk-meta"><div class="risk-meta-grid">' + metaItems + '</div></div>' +
            '</div>';
    }

    /* ── SHAP Explanation ────────────────────────────── */
    function renderShap(report) {
        var ra = report.risk_assessment;
        var factors = (ra && ra.risk_factors) ? ra.risk_factors : [];
        var reducers = (ra && ra.risk_reducers) ? ra.risk_reducers : [];

        if (factors.length === 0 && reducers.length === 0) {
            $('shap-content').innerHTML = C.emptyState(
                '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
                'SHAP Explanation Unavailable',
                'No risk factor explanation data available for this transaction.'
            );
            return;
        }

        var html = '';

        if (factors.length > 0) {
            html += '<div class="shap-section">';
            html += '<div class="shap-section-title" style="color:#f43f5e">Risk Factors</div>';
            var sorted = factors.slice().sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
            for (var i = 0; i < sorted.length; i++) {
                html += C.shapBar(sorted[i].feature || sorted[i].description || 'Factor', Math.abs(sorted[i].impact), 'risk');
            }
            html += '</div>';
        }

        if (reducers.length > 0) {
            html += '<div class="shap-section" style="margin-top:16px">';
            html += '<div class="shap-section-title" style="color:#10b981">Risk Reducers</div>';
            var sorted = reducers.slice().sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
            for (var i = 0; i < sorted.length; i++) {
                html += C.shapBar(sorted[i].feature || sorted[i].description || 'Factor', Math.abs(sorted[i].impact), 'reduce');
            }
            html += '</div>';
        }

        $('shap-content').innerHTML = html;
    }

    /* ── Graph Intelligence ──────────────────────────── */
    function renderGraphSection(report) {
        var graphAvail = report.evidence && report.evidence.availability && report.evidence.availability.graph;
        if (!graphAvail) {
            $('graph-content').innerHTML = C.emptyState(
                '<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/>',
                'Graph Intelligence Unavailable',
                'Graph has not been built or is unavailable. Build the graph via the API to enable graph intelligence.'
            );
            $('network-risk-content').innerHTML = '';
            $('cluster-content').innerHTML = '';
            return;
        }

        $('graph-content').innerHTML = C.loadingState('Loading graph connections...');
        $('network-risk-content').innerHTML = '';
        $('cluster-content').innerHTML = '';

        var txnId = report.transaction_id;

        Promise.allSettled([
            RiskPredAPI.getTransactionNeighborhood(txnId, 2),
            RiskPredAPI.getTransactionRisk(txnId),
            RiskPredAPI.getClusterForTransaction(txnId),
        ]).then((results) => {
            var neighborhoodResult = results[0];
            var riskResult = results[1];
            var clusterResult = results[2];

            if (neighborhoodResult.status === 'fulfilled') {
                var nd = neighborhoodResult.value;
                var nodeCount = (nd.nodes || []).length;
                var edgeCount = (nd.edges || []).length;
                $('graph-content').innerHTML =
                    '<div style="margin-bottom:8px;font-size:0.8rem;color:var(--text-muted)">' +
                    nodeCount + ' node' + (nodeCount !== 1 ? 's' : '') + ', ' +
                    edgeCount + ' connection' + (edgeCount !== 1 ? 's' : '') +
                    '</div>' +
                    '<div class="graph-canvas" id="graph-canvas"></div>';
                RiskPredCharts.renderGraph($('graph-canvas'), nd);
            } else {
                $('graph-content').innerHTML = C.emptyState(
                    '<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/>',
                    'Graph Unavailable',
                    neighborhoodResult.reason ? neighborhoodResult.reason.message : 'Could not load graph data.'
                );
            }

            if (riskResult.status === 'fulfilled') {
                renderNetworkRisk(riskResult.value);
            }

            if (clusterResult.status === 'fulfilled') {
                renderCluster(clusterResult.value);
            } else if (clusterResult.status === 'rejected' && clusterResult.reason && clusterResult.reason.status === 404) {
                $('cluster-content').innerHTML =
                    '<div class="cluster-info">' +
                    '<div style="color:var(--text-muted);font-size:0.85rem">No cluster found for this transaction.</div>' +
                    '</div>';
            }
        });
    }

    /* ── Network Risk ────────────────────────────────── */
    function renderNetworkRisk(data) {
        if (!data) return;
        var html = '<div class="network-info-grid">';
        html += '<div class="network-info-item"><div class="network-info-value" style="color:' + C.riskColor(data.network_risk_level) + '">' + (data.network_risk_score || 0).toFixed(1) + '</div><div class="network-info-label">Network Risk</div></div>';
        html += '<div class="network-info-item"><div class="network-info-value" style="color:' + C.riskColor(data.combined_risk_level) + '">' + (data.combined_risk_score || 0).toFixed(1) + '</div><div class="network-info-label">Combined Risk</div></div>';
        html += '<div class="network-info-item"><div class="network-info-value" style="color:var(--accent-rose)">' + (data.suspicious_neighbor_count || 0) + '</div><div class="network-info-label">Suspicious Neighbors</div></div>';
        html += '</div>';

        if (data.factors && data.factors.length > 0) {
            html += '<div class="mt-4" style="font-size:0.8rem;color:var(--text-secondary)">';
            html += '<strong>Network Risk Factors:</strong> ';
            var factorTexts = [];
            for (var i = 0; i < data.factors.length; i++) {
                var f = data.factors[i];
                factorTexts.push(f.description || f.type || JSON.stringify(f));
            }
            html += factorTexts.join('; ');
            html += '</div>';
        }

        $('network-risk-content').innerHTML = html;
    }

    /* ── Cluster Intelligence ────────────────────────── */
    function renderCluster(data) {
        if (!data) return;
        var html = '<div class="cluster-info">';
        html += '<div class="cluster-header">';
        html += '<span style="font-weight:600;font-size:0.9rem">Cluster #' + (data.cluster_id || '?') + '</span>';
        html += C.riskBadge(data.risk_level);
        html += '</div>';

        html += '<div class="cluster-detail-grid">';
        html += '<div class="cluster-detail-item"><div class="cluster-detail-label">Transactions</div><div class="cluster-detail-value">' + (data.total_transactions || 0) + '</div></div>';
        html += '<div class="cluster-detail-item"><div class="cluster-detail-label">Entities</div><div class="cluster-detail-value">' + (data.entity_count || 0) + '</div></div>';
        html += '<div class="cluster-detail-item"><div class="cluster-detail-label">Suspicious Ratio</div><div class="cluster-detail-value">' + ((data.suspicious_ratio || 0) * 100).toFixed(1) + '%</div></div>';
        html += '<div class="cluster-detail-item"><div class="cluster-detail-label">Entity Types</div><div class="cluster-detail-value">' + (data.entity_types || []).join(', ') + '</div></div>';
        html += '</div>';

        if (data.shared_identifiers && Object.keys(data.shared_identifiers).length > 0) {
            html += '<div class="shared-identifiers">';
            html += '<div class="shared-id-title">Shared Identifiers</div>';
            for (var key in data.shared_identifiers) {
                var vals = data.shared_identifiers[key];
                html += '<div style="margin-bottom:6px"><span style="color:var(--text-muted);font-size:0.75rem;font-weight:600">' + C.esc(key) + ':</span> ';
                for (var j = 0; j < vals.length; j++) {
                    html += '<span class="shared-id-tag">' + C.esc(vals[j]) + '</span>';
                }
                html += '</div>';
            }
            html += '</div>';
        }

        html += '</div>';
        $('cluster-content').innerHTML = html;
    }

    /* ── Agent Investigation ─────────────────────────── */
    function renderAgents(report) {
        var html = '<div class="agents-grid">';

        html += renderRiskAgent(report.risk_assessment, !!report.agent_errors);
        html += renderPatternAgent(report.detected_patterns, !!report.agent_errors);
        html += renderEvidenceAgent(report.evidence, !!report.agent_errors);

        html += '</div>';

        if (report.agent_errors && report.agent_errors.length > 0) {
            html += C.agentErrorList(report.agent_errors);
        }

        $('agents-content').innerHTML = html;
    }

    function renderRiskAgent(ra, hasErrors) {
        var ok = !!ra;
        var errForAgent = hasErrors ? findAgentError(state.report, 'RiskAgent') : null;

        var header = '<div class="agent-card-header">' +
            '<div class="agent-card-name">&#128737; Risk Agent</div>' +
            C.agentStatusHTML(ok, !ok && !errForAgent) +
            '</div>';

        if (!ok) {
            return '<div class="agent-card">' + header +
                '<div class="agent-card-body"><p style="color:var(--text-muted)">Risk agent did not return results.</p></div></div>';
        }

        var body = '<div class="agent-card-body">';
        body += '<p>' + C.esc(ra.assessment || 'No assessment provided.') + '</p>';

        if (ra.reasons && ra.reasons.length > 0) {
            body += '<ul class="agent-reasons">';
            for (var i = 0; i < ra.reasons.length; i++) {
                body += '<li>' + C.esc(ra.reasons[i]) + '</li>';
            }
            body += '</ul>';
        }
        body += '</div>';

        return '<div class="agent-card">' + header + body + '</div>';
    }

    function renderPatternAgent(pa, hasErrors) {
        var ok = !!pa && pa.pattern_count > 0;
        var errForAgent = hasErrors ? findAgentError(state.report, 'PatternAgent') : null;

        var header = '<div class="agent-card-header">' +
            '<div class="agent-card-name">&#128269; Pattern Agent</div>' +
            C.agentStatusHTML(ok, !ok && !errForAgent) +
            '</div>';

        if (!pa || !pa.patterns || pa.patterns.length === 0) {
            return '<div class="agent-card">' + header +
                '<div class="agent-card-body"><p style="color:var(--text-muted)">No patterns detected.</p></div></div>';
        }

        var body = '<div class="agent-card-body">';
        body += '<p>' + C.esc(pa.summary || pa.pattern_count + ' pattern(s) detected.') + '</p>';
        body += '<div class="pattern-list mt-2">';
        for (var i = 0; i < pa.patterns.length; i++) {
            var p = pa.patterns[i];
            var sevClass = 'pattern-item--' + (p.severity || 'medium').toLowerCase();
            body += '<div class="pattern-item ' + sevClass + '">';
            body += '<div class="pattern-header">';
            body += '<span class="pattern-type">' + C.esc(p.pattern_type) + '</span>';
            body += C.riskBadge(p.severity);
            body += '</div>';
            body += '<div class="pattern-desc">' + C.esc(p.description) + '</div>';
            body += '</div>';
        }
        body += '</div></div>';

        return '<div class="agent-card">' + header + body + '</div>';
    }

    function renderEvidenceAgent(ea, hasErrors) {
        var ok = !!ea && ea.evidence_count > 0;
        var errForAgent = hasErrors ? findAgentError(state.report, 'EvidenceAgent') : null;

        var header = '<div class="agent-card-header">' +
            '<div class="agent-card-name">&#128220; Evidence Agent</div>' +
            C.agentStatusHTML(ok, !ok && !errForAgent) +
            '</div>';

        if (!ea || !ea.evidence || ea.evidence.length === 0) {
            return '<div class="agent-card">' + header +
                '<div class="agent-card-body"><p style="color:var(--text-muted)">No evidence collected.</p></div></div>';
        }

        var body = '<div class="agent-card-body">';
        body += '<p>' + C.esc(ea.summary || ea.evidence_count + ' evidence item(s) collected.') + '</p>';

        if (ea.availability) {
            body += '<div class="mt-2" style="font-size:0.75rem;color:var(--text-muted)">';
            for (var src in ea.availability) {
                var avail = ea.availability[src];
                body += '<span style="margin-right:10px">' +
                    (avail ? '<span style="color:#10b981">&#10003;</span>' : '<span style="color:#f43f5e">&#10007;</span>') +
                    ' ' + C.esc(src) + '</span>';
            }
            body += '</div>';
        }
        body += '</div>';

        return '<div class="agent-card">' + header + body + '</div>';
    }

    /* ── Evidence Display ────────────────────────────── */
    function renderEvidence(report) {
        var ea = report.evidence;
        if (!ea || !ea.evidence || ea.evidence.length === 0) {
            $('evidence-content').innerHTML = C.emptyState(
                '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
                'No Evidence Collected',
                'The evidence agent did not collect any evidence items.'
            );
            return;
        }

        var iconMap = {
            'ml_prediction': { cls: 'evidence-icon--ml', label: 'ML' },
            'shap_explanation': { cls: 'evidence-icon--shap', label: 'SH' },
            'graph': { cls: 'evidence-icon--graph', label: 'GR' },
            'network_risk': { cls: 'evidence-icon--network', label: 'NW' },
            'cluster': { cls: 'evidence-icon--cluster', label: 'CL' },
        };

        var html = '<div class="evidence-list">';
        for (var i = 0; i < ea.evidence.length; i++) {
            var item = ea.evidence[i];
            var iconInfo = iconMap[item.evidence_type] || { cls: 'evidence-icon--ml', label: '?' };
            var unavailClass = item.available === false ? ' evidence-item--unavailable' : '';

            html += '<div class="evidence-item' + unavailClass + '">';
            html += '<div class="evidence-icon ' + iconInfo.cls + '">' + iconInfo.label + '</div>';
            html += '<div class="evidence-content">';
            html += '<div class="evidence-source">' + C.esc(item.evidence_type) + ' &middot; ' + C.esc(item.source) + '</div>';
            html += '<div class="evidence-desc">' + C.esc(item.description) + '</div>';

            if (item.details && Object.keys(item.details).length > 0) {
                html += '<div class="evidence-details">';
                var detailParts = [];
                for (var k in item.details) {
                    detailParts.push(k + ': ' + JSON.stringify(item.details[k]));
                }
                html += C.esc(detailParts.join(' | '));
                html += '</div>';
            }

            if (item.available === false) {
                html += '<div style="color:var(--accent-rose);font-size:0.7rem;margin-top:4px;font-weight:600">&#9888; Unavailable</div>';
            }

            html += '</div></div>';
        }
        html += '</div>';

        $('evidence-content').innerHTML = html;
    }

    /* ── Investigation Conclusion ────────────────────── */
    function renderConclusion(report) {
        var html = '';

        html += '<div class="conclusion-box">';
        html += '<div class="conclusion-label">Investigation Conclusion</div>';
        html += '<div class="conclusion-text">' + C.esc(report.conclusion || 'No conclusion generated.') + '</div>';
        html += '</div>';

        if (report.recommended_action) {
            html += '<div class="conclusion-box" style="margin-top:12px">';
            html += '<div class="conclusion-label">Recommended Action</div>';
            html += '<div class="conclusion-text" style="color:' + C.riskColor(report.risk_assessment ? report.risk_assessment.risk_level : 'UNKNOWN') + ';font-weight:600">' +
                C.esc(report.recommended_action) + '</div>';
            html += '</div>';
        }

        if (report.metadata && Object.keys(report.metadata).length > 0) {
            html += '<div class="conclusion-box" style="margin-top:12px">';
            html += '<div class="conclusion-label">Metadata</div>';
            html += '<div class="evidence-details" style="color:var(--text-secondary)">';
            var metaParts = [];
            for (var k in report.metadata) {
                metaParts.push(k + ': ' + JSON.stringify(report.metadata[k]));
            }
            html += C.esc(metaParts.join(' | '));
            html += '</div></div>';
        }

        $('conclusion-content').innerHTML = html;
    }

    function findAgentError(report, name) {
        if (!report.agent_errors) return null;
        for (var i = 0; i < report.agent_errors.length; i++) {
            if (report.agent_errors[i].agent_name === name) return report.agent_errors[i];
        }
        return null;
    }
});
