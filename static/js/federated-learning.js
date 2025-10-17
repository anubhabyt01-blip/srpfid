const nodesList = document.getElementById('nodes-list');
const jobsList = document.getElementById('jobs-list');

async function loadNodes() {
    try {
        const nodes = await api.get('/api/federated/nodes');
        
        if (nodes && nodes.length > 0) {
            nodesList.innerHTML = nodes.map(node => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>Node: ${node.node_id || node.nodeid}</h4>
                        <div class="item-meta">Reputation: ${node.reputation || 0}</div>
                    </div>
                    <span class="badge ${node.status === 'online' ? 'success' : 'pending'}">
                        ${node.status}
                    </span>
                </div>
            `).join('');
        } else {
            nodesList.innerHTML = '<p class="item-meta">No nodes registered</p>';
        }
    } catch (error) {
        console.error('Error loading nodes:', error);
    }
}

async function loadJobs() {
    try {
        const jobs = await api.get('/api/federated/jobs');
        
        if (jobs && jobs.length > 0) {
            jobsList.innerHTML = jobs.map(job => `
                <div class="list-item">
                    <div class="item-info">
                        <h4>${job.name}</h4>
                        <div class="item-meta">${job.model_type || job.modeltype} | Round: ${job.rounds || 0}</div>
                    </div>
                    <span class="badge ${job.status === 'completed' ? 'success' : 'pending'}">
                        ${job.status}
                    </span>
                </div>
            `).join('');
        } else {
            jobsList.innerHTML = '<p class="item-meta">No federated jobs yet</p>';
        }
    } catch (error) {
        console.error('Error loading jobs:', error);
    }
}

loadNodes();
loadJobs();
