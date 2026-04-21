import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const workflowApi = {
  getState: () => api.get('/session/state'),
  
  initializeGoal: (data: any) => api.post('/goal/initialize', data),
  
  runScoping: () => api.post('/workflow/scoping'),
  
  runLiterature: (maxResults = 5, sources = ['arxiv']) => 
    api.post('/workflow/literature', { max_results: maxResults, sources }),
  
  runHypotheses: (count = 5) => api.post('/workflow/hypotheses', { count }),
  
  generateProtocol: (hypothesisId: string) => api.post(`/workflow/protocol/${hypothesisId}`),
  
  runAnalysis: (hypothesisId: string, filePath: string | null = null) => 
    api.post(`/workflow/analysis/${hypothesisId}`, { file_path: filePath }),
  
  runWriting: () => api.post('/workflow/writing'),
  
  uploadCsv: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/upload/csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
};

export default api;
