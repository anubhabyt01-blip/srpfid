async function loadDashboard() {
    try {
        const [health, datasets, models, experiments, jobs] = await Promise.all([
            api.get('/api/health'),
            api.get('/api/datasets'),
            api.get('/api/models'),
            api.get('/api/quantum/experiments'),
            api.get('/api/federated/jobs')
        ]);

        document.getElementById('datasets-count').textContent = datasets.length || 0;
        document.getElementById('models-count').textContent = models.length || 0;
        document.getElementById('quantum-count').textContent = experiments.length || 0;
        document.getElementById('federated-count').textContent = jobs.length || 0;

        const healthStatus = document.getElementById('health-status');
        if (health.services) {
            healthStatus.innerHTML = Object.entries(health.services).map(([service, status]) => `
                <div class="health-item">
                    <div class="health-dot ${status ? 'healthy' : 'error'}"></div>
                    <div style="text-transform: capitalize;">${service}</div>
                    <div class="item-meta">${status ? 'Healthy' : 'Error'}</div>
                </div>
            `).join('');
        }

        const recentModels = document.getElementById('recent-models');
        if (models && models.length > 0) {
            recentModels.innerHTML = models.slice(0, 5).map(model => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${model.name}</h4>
                        <div class="item-meta">${model.algorithm}</div>
                    </div>
                    <div>
                        ${model.f1score ? `F1: ${model.f1score.toFixed(3)}` : model.trainingstatus || model.training_status || 'N/A'}
                    </div>
                </div>
            `).join('');
        } else {
            recentModels.innerHTML = '<p class="item-meta">No models trained yet</p>';
        }

        const quantumExperiments = document.getElementById('quantum-experiments');
        if (experiments && experiments.length > 0) {
            quantumExperiments.innerHTML = experiments.slice(0, 5).map(exp => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${exp.name}</h4>
                        <div class="item-meta">${exp.qubits} qubits • ${exp.algorithm}</div>
                    </div>
                    <span class="badge ${exp.status === 'completed' ? 'success' : 'pending'}">${exp.status}</span>
                </div>
            `).join('');
        } else {
            quantumExperiments.innerHTML = '<p class="item-meta">No experiments yet</p>';
        }
    } catch (error) {
        console.error('Error loading dashboard:', error);
    }
}

loadDashboard();
setInterval(loadDashboard, 30000);
