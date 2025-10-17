const quantumForm = document.getElementById('quantum-form');
const quantumStatus = document.getElementById('quantum-status');
const experimentsList = document.getElementById('experiments-list');

quantumForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const experimentData = {
        name: document.getElementById('quantum-name').value,
        algorithm: document.getElementById('quantum-algorithm').value,
        qubits: parseInt(document.getElementById('qubits').value)
    };
    
    quantumStatus.className = 'status-message';
    quantumStatus.textContent = 'Creating experiment...';
    quantumStatus.style.display = 'block';
    
    try {
        const result = await api.post('/api/quantum/experiments', experimentData);
        
        quantumStatus.className = 'status-message success';
        quantumStatus.textContent = 'Quantum experiment created successfully!';
        
        quantumForm.reset();
        loadExperiments();
    } catch (error) {
        quantumStatus.className = 'status-message error';
        quantumStatus.textContent = 'Failed to create experiment: ' + error.message;
    }
});

async function loadExperiments() {
    try {
        const experiments = await api.get('/api/quantum/experiments');
        
        if (experiments && experiments.length > 0) {
            experimentsList.innerHTML = experiments.map(exp => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${exp.name}</h4>
                        <div class="item-meta">${exp.algorithm} | ${exp.qubits} qubits</div>
                        ${exp.execution_time || exp.executiontime ? `<div class="item-meta">Execution Time: ${(exp.execution_time || exp.executiontime).toFixed(2)}s</div>` : ''}
                    </div>
                    <span class="badge ${exp.status === 'completed' ? 'success' : exp.status === 'failed' ? 'error' : 'pending'}">
                        ${exp.status}
                    </span>
                </div>
            `).join('');
        } else {
            experimentsList.innerHTML = '<p class="item-meta">No experiments yet</p>';
        }
    } catch (error) {
        console.error('Error loading experiments:', error);
    }
}

loadExperiments();
