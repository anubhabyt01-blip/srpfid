const trainingForm = document.getElementById('training-form');
const trainingStatus = document.getElementById('training-status');
const modelsList = document.getElementById('models-list');
const datasetSelect = document.getElementById('dataset-select');

async function loadDatasets() {
    try {
        const datasets = await api.get('/api/datasets');
        
        if (datasets && datasets.length > 0) {
            datasetSelect.innerHTML = '<option value="">Select a dataset</option>' +
                datasets.map(dataset => `
                    <option value="${dataset.id}">${dataset.name}</option>
                `).join('');
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

trainingForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const modelData = {
        name: document.getElementById('model-name').value,
        algorithm: document.getElementById('algorithm').value,
        datasetId: document.getElementById('dataset-select').value,
        hyperparameters: {}
    };
    
    trainingStatus.className = 'status-message';
    trainingStatus.textContent = 'Starting training...';
    trainingStatus.style.display = 'block';
    
    try {
        const result = await api.post('/api/models/train', modelData);
        
        trainingStatus.className = 'status-message success';
        trainingStatus.textContent = 'Model training started successfully!';
        
        trainingForm.reset();
        loadModels();
    } catch (error) {
        trainingStatus.className = 'status-message error';
        trainingStatus.textContent = 'Training failed: ' + error.message;
    }
});

async function loadModels() {
    try {
        const models = await api.get('/api/models');
        
        if (models && models.length > 0) {
            modelsList.innerHTML = models.map(model => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${model.name}</h4>
                        <div class="item-meta">${model.algorithm}</div>
                        ${model.f1score || model.f1_score ? `<div class="item-meta">F1 Score: ${(model.f1score || model.f1_score).toFixed(3)}</div>` : ''}
                    </div>
                    <span class="badge ${model.trainingstatus === 'completed' || model.training_status === 'completed' ? 'success' : 'pending'}">
                        ${model.trainingstatus || model.training_status || 'N/A'}
                    </span>
                </div>
            `).join('');
        } else {
            modelsList.innerHTML = '<p class="item-meta">No models trained yet</p>';
        }
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

loadDatasets();
loadModels();
