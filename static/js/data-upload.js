const uploadForm = document.getElementById('upload-form');
const uploadStatus = document.getElementById('upload-status');
const datasetsList = document.getElementById('datasets-list');

uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    formData.append('file', document.getElementById('dataset-file').files[0]);
    formData.append('name', document.getElementById('dataset-name').value);
    formData.append('description', document.getElementById('dataset-description').value);
    
    uploadStatus.className = 'status-message';
    uploadStatus.textContent = 'Uploading...';
    uploadStatus.style.display = 'block';
    
    try {
        const result = await api.uploadFile('/api/datasets/upload', formData);
        
        uploadStatus.className = 'status-message success';
        uploadStatus.textContent = 'Dataset uploaded successfully!';
        
        uploadForm.reset();
        loadDatasets();
    } catch (error) {
        uploadStatus.className = 'status-message error';
        uploadStatus.textContent = 'Upload failed: ' + error.message;
    }
});

async function loadDatasets() {
    try {
        const datasets = await api.get('/api/datasets');
        
        if (datasets && datasets.length > 0) {
            datasetsList.innerHTML = datasets.map(dataset => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${dataset.name}</h4>
                        <div class="item-meta">${dataset.description || 'No description'}</div>
                        <div class="item-meta">Rows: ${dataset.row_count || dataset.rowcount || 'N/A'} | Columns: ${dataset.column_count || dataset.columncount || 'N/A'}</div>
                        ${dataset.data_quality ? `<div class="item-meta">Imbalance: ${(JSON.parse(dataset.data_quality).imbalanceRatio * 100).toFixed(1)}%</div>` : ''}
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button class="btn-small" onclick="exportBalanced('${dataset.id}', '${dataset.target_column || 'defects'}')">Export Balanced</button>
                        <button class="btn-small" onclick="showGeminiInsights('${dataset.id}')">AI Insights</button>
                    </div>
                </div>
            `).join('');
        } else {
            datasetsList.innerHTML = '<p class="item-meta">No datasets uploaded yet</p>';
        }
    } catch (error) {
        console.error('Error loading datasets:', error);
    }
}

async function exportBalanced(datasetId, targetColumn) {
    const techniqueChoice = prompt('Select sampling technique:\n1. SMOTE (default)\n2. ADASYN\n3. BorderlineSMOTE\n4. RandomUndersample\n\nEnter number or name:', '1');
    
    if (!techniqueChoice) return;
    
    const techniqueMap = {
        '1': 'smote',
        '2': 'adasyn',
        '3': 'borderlinesmote',
        '4': 'randomundersample',
        'smote': 'smote',
        'adasyn': 'adasyn',
        'borderlinesmote': 'borderlinesmote',
        'randomundersample': 'randomundersample'
    };
    
    const technique = techniqueMap[techniqueChoice.toLowerCase()] || 'smote';
    
    try {
        const result = await api.post(`/api/datasets/${datasetId}/export-balanced`, {
            samplingTechnique: technique,
            targetColumn: targetColumn
        });
        
        if (result.success) {
            alert(`Dataset balanced successfully!\n\nOriginal: ${result.originalCount} samples\nBalanced: ${result.balancedCount} samples\n\nFile saved: ${result.exportPath}`);
        } else {
            alert('Export failed: ' + result.error);
        }
    } catch (error) {
        alert('Export failed: ' + error.message);
    }
}

async function showGeminiInsights(datasetId) {
    try {
        const insights = await api.get(`/api/datasets/${datasetId}/gemini-insights`);
        
        const message = `AI Insights:\n\n${insights.insights}\n\nRecommendations:\n${insights.recommendations.join('\n')}`;
        alert(message);
    } catch (error) {
        alert('Could not get AI insights: ' + error.message);
    }
}

loadDatasets();
