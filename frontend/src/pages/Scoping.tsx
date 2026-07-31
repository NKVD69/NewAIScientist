import { useState } from 'react';
import { Compass } from 'lucide-react';
import { workflowApi } from '../services/api';
import { PageHeader, Panel, SectionRule } from '../components/primitives/Panel';
import Button from '../components/primitives/Button';
import EmptyState from '../components/primitives/EmptyState';
import { ErrorBanner } from '../components/primitives/Banner';

export default function Scoping() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<unknown>(null);
  const [selected, setSelected] = useState<string[]>([]);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const res = await workflowApi.runScoping();
      setData(res.data);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  const questions: string[] = data?.sub_questions ?? data?.questions ?? [];
  const keywords: string[] = data?.keywords ?? [];

  return (
    <div>
      <PageHeader
        title="Scoping"
        lede="Narrows a broad goal into answerable sub-questions and the search terms that will retrieve literature for them."
        actions={
          <Button onClick={run} loading={loading} icon={<Compass className="h-4 w-4" />}>
            {loading ? 'Scoping' : 'Run scoping'}
          </Button>
        }
      />

      <ErrorBanner error={error} />

      {!data && !loading && !error && (
        <EmptyState title="Not scoped yet">
          Scoping is optional but cheap, and it markedly improves retrieval: the search queries
          it produces are more specific than ones derived from the goal title alone.
        </EmptyState>
      )}

      {questions.length > 0 && (
        <>
          <SectionRule title="Sub-questions" note={`${selected.length} selected`} />
          <Panel>
            <ul className="space-y-2.5">
              {questions.map((q, i) => {
                const on = selected.includes(q);
                return (
                  <li key={i}>
                    <label className="flex cursor-pointer items-start gap-3 py-1">
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() =>
                          setSelected((s) => (on ? s.filter((x) => x !== q) : [...s, q]))
                        }
                        className="mt-1 h-3.5 w-3.5 shrink-0 accent-[var(--ink)]"
                      />
                      <span className="text-[14px] leading-snug">{q}</span>
                    </label>
                  </li>
                );
              })}
            </ul>
          </Panel>
        </>
      )}

      {keywords.length > 0 && (
        <>
          <SectionRule title="Search terms" />
          <Panel>
            <div className="flex flex-wrap gap-2">
              {keywords.map((k, i) => (
                <span
                  key={i}
                  className="num border border-[var(--rule-strong)] bg-[var(--sunken)] px-2 py-1 text-[11.5px] text-[var(--ink-2)]"
                >
                  {k}
                </span>
              ))}
            </div>
          </Panel>
        </>
      )}
    </div>
  );
}
