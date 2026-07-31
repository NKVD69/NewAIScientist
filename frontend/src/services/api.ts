import axios from 'axios';
import type { Hypothesis, PipelineReport, SessionMeters } from '../types/domain';

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY ?? '';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
  },
});

export interface SessionState {
  phase: string;
  goal: unknown;
  num_hypotheses: number;
  num_papers: number;
  iteration: number;
  has_manuscript: boolean;
  /** Latest pipeline execution. Null before any run. */
  report: PipelineReport | null;
  reports: (PipelineReport | null)[];
  /** False when any task in the session failed or was skipped. */
  run_is_clean: boolean | null;
  meters: SessionMeters;
}

export const workflowApi = {
  getState: () => api.get<SessionState>('/session/state'),
  getHypotheses: () => api.get<Hypothesis[]>('/session/hypotheses'),
  getPapers: () => api.get('/session/papers'),

  initializeGoal: (data: unknown) => api.post('/goal/initialize', data),
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
