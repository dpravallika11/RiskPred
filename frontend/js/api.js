const RiskPredAPI = (() => {
    const BASE = window.location.origin;
    const API = BASE + '/api/v1';

    async function request(path, options = {}) {
        const url = API + path;
        try {
            const res = await fetch(url, {
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options,
            });
            if (!res.ok) {
                const body = await res.json().catch(() => ({}));
                throw {
                    status: res.status,
                    message: body.detail || body.message || `HTTP ${res.status}`,
                    url,
                };
            }
            return await res.json();
        } catch (err) {
            if (err.status) throw err;
            throw { status: 0, message: 'Network error – backend may be unreachable.', url };
        }
    }

    return {
        checkHealth() {
            return request('/health');
        },

        predictTransaction(payload) {
            return request('/predict', {
                method: 'POST',
                body: JSON.stringify(payload),
            });
        },

        getGraphStatus() {
            return request('/graph/status');
        },

        getTransactionGraph(txnId) {
            return request('/graph/transaction/' + encodeURIComponent(txnId));
        },

        getTransactionConnections(txnId) {
            return request('/graph/transaction/' + encodeURIComponent(txnId) + '/connections');
        },

        getTransactionNeighborhood(txnId, maxHops = 2) {
            return request('/graph/transaction/' + encodeURIComponent(txnId) + '/neighborhood?max_hops=' + maxHops);
        },

        getTransactionRisk(txnId) {
            return request('/graph/transaction/' + encodeURIComponent(txnId) + '/risk');
        },

        getClusters() {
            return request('/graph/clusters');
        },

        getClusterForTransaction(txnId) {
            return request('/graph/clusters/' + encodeURIComponent(txnId));
        },

        getInvestigationContext(txnId) {
            return request('/investigation/' + encodeURIComponent(txnId) + '/context');
        },

        getInvestigationReport(txnId) {
            return request('/investigation/' + encodeURIComponent(txnId));
        },
    };
})();
