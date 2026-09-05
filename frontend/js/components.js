const RiskPredComponents = (() => {
    function esc(str) {
        if (str == null) return '';
        const d = document.createElement('div');
        d.textContent = String(str);
        return d.innerHTML;
    }

    function riskBadgeClass(level) {
        const l = (level || '').toUpperCase();
        if (l === 'LOW') return 'risk-badge--low';
        if (l === 'MEDIUM') return 'risk-badge--medium';
        if (l === 'HIGH') return 'risk-badge--high';
        if (l === 'CRITICAL') return 'risk-badge--critical';
        return 'risk-badge--unknown';
    }

    function riskColor(level) {
        const l = (level || '').toUpperCase();
        if (l === 'LOW') return '#10b981';
        if (l === 'MEDIUM') return '#f59e0b';
        if (l === 'HIGH') return '#f97316';
        if (l === 'CRITICAL') return '#f43f5e';
        return '#64748b';
    }

    function riskBadge(level, extra) {
        extra = extra || '';
        return '<span class="risk-badge ' + riskBadgeClass(level) + '">' + esc(level || 'UNKNOWN') + ' RISK' + (extra ? ' (' + esc(extra) + ')' : '') + '</span>';
    }

    function emptyState(icon, title, text) {
        return '<div class="state-message">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' + icon + '</svg>' +
            '<div class="state-message-title">' + esc(title) + '</div>' +
            '<div class="state-message-text">' + esc(text) + '</div>' +
            '</div>';
    }

    function loadingState(text) {
        return '<div class="loading-overlay">' +
            '<div class="spinner"></div>' +
            '<div class="loading-text">' + esc(text || 'Loading...') + '</div>' +
            '</div>';
    }

    function errorState(title, detail) {
        return '<div class="state-message">' +
            '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#f43f5e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">' +
            '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>' +
            '</svg>' +
            '<div class="state-message-title">' + esc(title) + '</div>' +
            '<div class="state-message-text">' + esc(detail || '') + '</div>' +
            '</div>';
    }

    function sectionTitle(text, iconPath) {
        var icon = iconPath
            ? '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' + iconPath + '</svg>'
            : '';
        return '<div class="section-title">' + icon + esc(text) + '</div>';
    }

    function riskScoreRing(score, level) {
        var r = 48;
        var c = 2 * Math.PI * r;
        var pct = Math.max(0, Math.min(100, score || 0));
        var offset = c - (pct / 100) * c;
        var color = riskColor(level);
        return '<div class="risk-ring">' +
            '<svg viewBox="0 0 110 110">' +
            '<circle class="risk-ring-bg" cx="55" cy="55" r="' + r + '"/>' +
            '<circle class="risk-ring-fill" cx="55" cy="55" r="' + r + '" ' +
            'stroke="' + color + '" stroke-dasharray="' + c + '" stroke-dashoffset="' + offset + '"/>' +
            '</svg>' +
            '<div class="risk-ring-label">' +
            '<div class="risk-ring-value" style="color:' + color + '">' + Math.round(pct) + '</div>' +
            '<div class="risk-ring-sub">Risk Score</div>' +
            '</div>' +
            '</div>';
    }

    function shapBar(label, value, type) {
        var maxVal = 1;
        var pct = Math.min(100, Math.abs(value) / maxVal * 100);
        var cls = type === 'risk' ? 'shap-bar-fill--risk' : 'shap-bar-fill--reduce';
        var sign = type === 'risk' ? '+' : '-';
        return '<div class="shap-bar">' +
            '<div class="shap-bar-label">' + esc(label) + '</div>' +
            '<div class="shap-bar-track"><div class="shap-bar-fill ' + cls + '" style="width:' + pct + '%"></div></div>' +
            '<div class="shap-bar-value">' + sign + Math.abs(value).toFixed(3) + '</div>' +
            '</div>';
    }

    function agentStatusHTML(ok, warn) {
        if (ok) return '<span class="agent-status agent-status--ok">&#10003; Success</span>';
        if (warn) return '<span class="agent-status agent-status--warn">&#9888; Partial</span>';
        return '<span class="agent-status agent-status--error">&#10007; Failed</span>';
    }

    function agentErrorList(errors) {
        if (!errors || errors.length === 0) return '';
        var html = '<div class="agent-errors">';
        for (var i = 0; i < errors.length; i++) {
            var e = errors[i];
            html += '<div class="agent-error-item">' +
                '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>' +
                '<div><strong>' + esc(e.agent_name) + ':</strong> ' + esc(e.error_message) + '</div>' +
                '</div>';
        }
        html += '</div>';
        return html;
    }

    return {
        esc: esc,
        riskBadge: riskBadge,
        riskBadgeClass: riskBadgeClass,
        riskColor: riskColor,
        emptyState: emptyState,
        loadingState: loadingState,
        errorState: errorState,
        sectionTitle: sectionTitle,
        riskScoreRing: riskScoreRing,
        shapBar: shapBar,
        agentStatusHTML: agentStatusHTML,
        agentErrorList: agentErrorList,
    };
})();
