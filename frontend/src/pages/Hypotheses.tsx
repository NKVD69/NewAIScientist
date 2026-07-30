import { useState } from 'react';
import { RefreshCw, Plus } from 'lucide-react';
import { workflowApi } from '../services/api';
import type { Hypothesis, PipelineReport } from '../types/domain';
import HypothesisEntry from '../components/HypothesisEntry';
import RunHealth from '../components/primitives/RunHealth';

export default function Hypotheses() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([]);
  const [report, setReport] = useState<PipelineReport | null>(null);

  async function generate() {
    setLoading(true);
    setError(null);
    try {
      const response = await workflowApi.runHypotheses(5);
      setHypotheses(response.data.hypotheses ?? response.data ?? []);
      setReport(response.data.report ?? null);
    } catch (e: any) {
      // Errors don't apologise and are never vague about what happened.
      setError(
        e?.response?.data?.detail ??
          e?.message ??
          'The request never reached the backend. Check that it is running on port 8000.',
      );
    } finally {
      setLoading(false);
    }
  }

  // Conservative ranking: mu - 2 sigma, so one lucky win cannot take the top slot.
  const ranked = [...hypotheses].sort(
    (a, b) => b.rating_conservative - a.rating_conservative,
  );

  return (
    <div>
      <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-[26px] leading-tight">Hypotheses</h1>
          <p className="mt-1 max-w-[62ch] text-[13.5px] text-[var(--ink-2)]">
            Generated against the indexed corpus, sealed with falsifiable predictions, then
            ranked by pairwise comparison.
          </p>
        </div>
        <button
          type="button"
          onClick={generate}
          disabled={loading}
          className="flex items-center gap-2 border border-[var(--ink)] bg-[var(--ink)] px-5 py-2.5
                     text-[13px] font-medium text-[var(--ground)] transition-opacity
                     hover:opacity-85 disabled:opacity-40"
        >
          {loading ? <RefreshCw className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
          {loading ? 'Generating' : 'Generate hypotheses'}
        </button>
      </div>

      {error && (
        <div className="mb-[30px] flex items-start gap-4 border border-l-4 border-[var(--oxide)] bg-[var(--oxide-soft)] px-[18px] py-3.5">
          <span className="shrink-0 whitespace-nowrap border border-current px-[7px] py-[3px] font-mono text-[11px] font-semibold uppercase tracking-[0.1em] text-[var(--oxide)]">
            Failed
          </span>
          <p className="m-0 text-[13.5px] leading-normal text-[var(--ink-2)]">{error}</p>
        </div>
      )}

      <RunHealth report={report} />

      {ranked.length === 0 && !loading && !error && (
        <div className="border border-dashed border-[var(--rule-strong)] bg-[var(--raised)] px-8 py-14 text-center">
          <h2 className="text-[17px]">No hypotheses yet</h2>
          <p className="mx-auto mt-2 max-w-[46ch] text-[13.5px] leading-relaxed text-[var(--ink-2)]">
            Index some literature first, then generate. Each hypothesis arrives with sealed
            predictions, so the engine can later report which ones the data refuted.
          </p>
        </div>
      )}

      {ranked.length > 0 && (
        <div className="mb-4 flex items-baseline gap-3">
          <h2 className="text-[15px]">Ranked</h2>
          <span className="h-px flex-1 bg-[var(--rule-strong)]" />
          <span className="panel-label">
            on μ − 2σ · {ranked.length} total
          </span>
        </div>
      )}

      {ranked.map((hyp) => (
        <HypothesisEntry key={hyp.id} hyp={hyp} />
      ))}
    </div>
  );
}
