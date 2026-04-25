import axios from 'axios';

const API_URL = 'http://localhost:8000';

const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

export const checkHealth = async () => {
    try {
        const response = await api.get('/health');
        return response.data;
    } catch (error) {
        throw new Error('Backend is not running');
    }
};

export const predictYield = async (formData) => {
    try {
        const response = await api.post('/predict/yield', formData);
        return response.data;
    } catch (error) {
        if (error.response && error.response.data) {
            throw new Error(error.response.data.detail || 'Prediction failed');
        }
        throw new Error('Network error. Is the backend running?');
    }
};

export const predictFertilizer = async (formData) => {
    try {
        const response = await api.post('/predict/fertilizer', formData);
        return response.data;
    } catch (error) {
        if (error.response && error.response.data) {
            throw new Error(error.response.data.detail || 'Prediction failed');
        }
        throw new Error('Network error. Is the backend running?');
    }
};

export default api;
