const API_BASE = '';

const api = {
    async get(endpoint) {
        const response = await fetch(API_BASE + endpoint);
        return response.json();
    },

    async post(endpoint, data) {
        const response = await fetch(API_BASE + endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        return response.json();
    },

    async uploadFile(endpoint, formData) {
        const response = await fetch(API_BASE + endpoint, {
            method: 'POST',
            body: formData
        });
        return response.json();
    }
};
