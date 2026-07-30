/**
 * Shapes returned by the FastAPI backend.
 *
 * These mirror `models/experiment.py` and `models/hypothesis.py`. Keep
 * the string unions in step with `VerdictStatus` and `ExperimentKind` —
 * the UI's job is to render the distinctions the engine draws, and a
 * drifted union silently collapses them.
 */

export type VerdictStatus =
  | 'corroborated'
  | 'refuted'
  | 'consistent_unscored'
  | 'untested'
  | 'unfalsifiable'
  | 'invalid';

export type ExperimentKind =
  | 'real_data'
  | 'literature_meta'
  | 'simulation'
  | 'infeasible';

export interface Verdict {
  quantity: string;
  status: VerdictStatus;
  expected: number | null;
  observed: number | null;
  unit: string;
  refuting_threshold: number | null;
  deviation: number | null;
  reason: string;
  experiment_kind: string;
}

export interface PriorArtHit {
  title: string;
  year: number | null;
  venue: string;
  url: string;
  similarity: number;
  citation_count: number;
}

export interface NoveltyReport {
  score: number;
  level: 'very_high' | 'high' | 'medium' | 'low' | 'unknown';
  /** False means the prior-art search did not run. Not the same as a low score. */
  searched: boolean;
  corpus_distance: number;
  similarity_method: 'embedding' | 'token';
  edge_is_new: boolean;
  query: string;
  error: string;
  prior_art: PriorArtHit[];
}

export interface Hypothesis {
  id: string;
  title: string;
  description: string;
  mechanism: string;
  status: string;
  generation_method: string;
  parent_ids: string[];
  limitations: string[];
  testable_predictions: string[];

  /** Bayesian Bradley-Terry belief. Never render mu without sigma. */
  rating_mu: number;
  rating_sigma: number;
  rating_conservative: number;
  rating_matches: number;

  verdicts: Verdict[];
  empirical_support: number;
  multiverse_fragility: number;
  novelty_level: string;
  novelty_report?: NoveltyReport;

  prediction_hash: string;
  registered_at: string;
}

export type TaskState = 'succeeded' | 'failed' | 'skipped' | 'pending' | 'running';

export interface TaskResult {
  name: string;
  state: TaskState;
  error: string;
  skipped_because: string;
  duration_s: number;
}

export interface PipelineReport {
  waves: string[][];
  results: Record<string, TaskResult>;
  aborted: boolean;
  abort_reason: string;
  /** True only when every task ran and succeeded. */
  clean: boolean;
  duration_s: number;
}

export interface SessionMeters {
  llm_calls: number;
  max_llm_calls: number | null;
  cost_usd: number;
  max_cost_usd: number | null;
  judge_order_invariance: number | null;
  sandbox_backend: string;
  sandbox_will_execute: boolean;
}
