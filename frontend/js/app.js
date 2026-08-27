document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('txn-form');
    
    // Check API Health on boot
    RiskPredAPI.checkHealth().then(data => {
        const badge = document.getElementById('api-status-badge');
        if (data.status === 'healthy') {
            badge.className = "px-3 py-1 text-xs rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center gap-2";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> System Online`;
        } else {
            badge.className = "px-3 py-1 text-xs rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-2";
            badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span> Backend Offline`;
        }
    });

    // Handle Transaction Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            transaction_id: "TXN_" + Math.floor(Math.random() * 899999 + 100000),
            merchant_id: "MERCHANT_001",
            customer_id: "CUST_8831",
            amount: parseFloat(document.getElementById('form-amount').value),
            device_id: "DEV_SIM_1",
            is_new_device: document.getElementById('form-new-device').checked,
            location: "Hyderabad, IN",
            is_new_location: document.getElementById('form-new-location').checked,
            payment_method: "credit_card",
            velocity_5m: parseInt(document.getElementById('form-velocity').value),
            failed_attempts_24h: parseInt(document.getElementById('form-failed').value)
        };

        try {
            const res = await RiskPredAPI.predictTransaction(payload);
            renderResult(res);
        } catch (err) {
            alert('Failed to obtain risk prediction from backend server.');
        }
    });
});

function renderResult(res) {
    document.getElementById('res-prob').innerText = (res.fraud_probability * 100).toFixed(1) + "%";
    document.getElementById('res-action').innerText = res.recommended_action;
    
    const badge = document.getElementById('risk-badge');
    badge.innerText = `${res.risk_level} RISK (${res.risk_score}/100)`;
    
    if (res.risk_level === 'CRITICAL') badge.className = "px-3 py-1 rounded text-xs font-bold bg-rose-500/20 text-rose-400 border border-rose-500/30";
    else if (res.risk_level === 'HIGH') badge.className = "px-3 py-1 rounded text-xs font-bold bg-orange-500/20 text-orange-400 border border-orange-500/30";
    else if (res.risk_level === 'MEDIUM') badge.className = "px-3 py-1 rounded text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30";
    else badge.className = "px-3 py-1 rounded text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";

    const reasonsList = document.getElementById('res-reasons');
    reasonsList.innerHTML = res.top_reasons.map(r => `<li class="flex items-center gap-2"><span class="text-rose-400">•</span> ${r}</li>`).join('');
}