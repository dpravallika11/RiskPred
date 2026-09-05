const RiskPredCharts = (() => {
    function renderGraph(container, neighborhoodData) {
        if (!container) return;
        container.innerHTML = '';

        if (!neighborhoodData || !neighborhoodData.nodes || neighborhoodData.nodes.length === 0) {
            container.innerHTML = RiskPredComponents.emptyState(
                '<circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/>',
                'No Graph Data',
                'No graph connections found for this transaction.'
            );
            return;
        }

        var nodes = neighborhoodData.nodes;
        var edges = neighborhoodData.edges || [];
        var w = container.clientWidth || 600;
        var h = container.clientHeight || 320;
        var cx = w / 2;
        var cy = h / 2;

        var nodeMap = {};
        for (var i = 0; i < nodes.length; i++) {
            nodeMap[nodes[i].id] = nodes[i];
        }

        var positions = {};
        var centerId = neighborhoodData.transaction_id || (nodes.length > 0 ? nodes[0].id : null);

        if (nodes.length === 1) {
            positions[nodes[0].id] = { x: cx, y: cy };
        } else {
            for (var i = 0; i < nodes.length; i++) {
                if (nodes[i].id === centerId) {
                    positions[nodes[i].id] = { x: cx, y: cy };
                } else {
                    var angle = (2 * Math.PI * i) / (nodes.length - 1) - Math.PI / 2;
                    var radius = Math.min(w, h) * 0.35;
                    positions[nodes[i].id] = {
                        x: cx + Math.cos(angle) * radius,
                        y: cy + Math.sin(angle) * radius,
                    };
                }
            }
        }

        var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', '0 0 ' + w + ' ' + h);
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');

        for (var i = 0; i < edges.length; i++) {
            var e = edges[i];
            var src = positions[e.source];
            var tgt = positions[e.target];
            if (!src || !tgt) continue;

            var line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', src.x);
            line.setAttribute('y1', src.y);
            line.setAttribute('x2', tgt.x);
            line.setAttribute('y2', tgt.y);

            var isSuspicious = e.suspicious || (e.weight && e.weight > 0.7);
            line.setAttribute('class', isSuspicious ? 'graph-edge graph-edge--suspicious' : 'graph-edge');
            svg.appendChild(line);
        }

        for (var i = 0; i < nodes.length; i++) {
            var n = nodes[i];
            var pos = positions[n.id];
            if (!pos) continue;

            var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            g.setAttribute('class', 'graph-node');

            var circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', pos.x);
            circle.setAttribute('cy', pos.y);

            var isCenter = n.id === centerId;
            var isSuspicious = (n.risk_level || '').toUpperCase() === 'HIGH' || (n.risk_level || '').toUpperCase() === 'CRITICAL';
            var r = isCenter ? 16 : 10;
            var fill = isCenter ? '#6366f1' : (isSuspicious ? '#f43f5e' : '#334155');
            var stroke = isCenter ? '#818cf8' : (isSuspicious ? '#fda4af' : '#475569');

            circle.setAttribute('r', r);
            circle.setAttribute('fill', fill);
            circle.setAttribute('stroke', stroke);
            circle.setAttribute('stroke-width', isCenter ? 3 : 2);
            g.appendChild(circle);

            var text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', pos.x);
            text.setAttribute('y', pos.y + r + 14);
            text.setAttribute('class', 'graph-label');
            var label = n.id || '';
            if (label.length > 12) label = label.substring(0, 10) + '..';
            text.textContent = label;
            g.appendChild(text);

            svg.appendChild(g);
        }

        container.appendChild(svg);
    }

    return { renderGraph: renderGraph };
})();
