const metricsList = document.getElementById('metrics-list');

async function loadMetrics() {
    try {
        const metrics = await api.get('/api/monitoring/metrics');
        
        if (metrics && metrics.length > 0) {
            metricsList.innerHTML = metrics.map(metric => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${metric.source} - ${metric.metric_type || metric.metrictype}</h4>
                        <div class="item-meta">Value: ${metric.value}</div>
                        <div class="item-meta">${new Date(metric.timestamp).toLocaleString()}</div>
                    </div>
                </div>
            `).join('');
        } else {
            metricsList.innerHTML = '<p class="item-meta">No metrics available</p>';
        }
    } catch (error) {
        console.error('Error loading metrics:', error);
    }
}

loadMetrics();
setInterval(loadMetrics, 30000);
