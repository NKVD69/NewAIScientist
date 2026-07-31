import { useRef, useState } from 'react';
import { Upload } from 'lucide-react';
import { workflowApi } from '../services/api';
import { PageHeader, Panel, SectionRule } from '../components/primitives/Panel';
import Button from '../components/primitives/Button';
import EmptyState from '../components/primitives/EmptyState';
import Banner, { ErrorBanner } from '../components/primitives/Banner';
import VerdictStrip from '../components/primitives/VerdictStrip';
import type { Verdict } from '../types/domain';

export default function Analysis() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [results, setResults] = useState<any>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  async function upload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const uploaded = await workflowApi.uploadCsv(file);
      const res = await workflowApi.runAnalysis('', uploaded.data.path);
      setResults(res.data);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  }

  const kind: string = results?.kind ?? '';
  const verdicts: Verdict[] = results?.verdicts ?? [];
  const isSimulation = kind === 'simulation';

  return (
    <div>
      <PageHeader
        title="Analysis"
        lede="Runs the experiment in a sandboxed container and confronts each measurement with its pre-registered prediction."
        actions={
          <>
            <input
              ref={fileInput}
              type="file"
              accept=".csv"
              onChange={upload}
              className="hidden"
              id="csv-upload"
            />
            <Button
              onClick={() => fileInput.current?.click()}
              loading={loading}
              icon={<Upload className="h-4 w-4" />}
            >
              {loading ? 'Running' : 'Upload CSV'}
            </Button>
          </>
        }
      />

      <ErrorBanner error={error} />

      {isSimulation && (
        <Banner mark="Simulation" tone="amber">
          <strong className="font-semibold text-[var(--ink)]">
            This run cannot corroborate anything.
          </strong>{' '}
          It analysed generated data, so agreement with the prediction earns no evidential
          credit. A contradiction would still count — internal incoherence is informative.
        </Banner>
      )}

      {kind === 'infeasible' && (
        <Banner mark="Not testable">
          <strong className="font-semibold text-[var(--ink)]">No data source was reachable.</strong>{' '}
          Nothing was measured, so nothing can be concluded in either direction.
        </Banner>
      )}

      {!results && !loading && !error && (
        <EmptyState title="No analysis yet">
          Upload a CSV, or let the experiment agent pull from ChEMBL. Execution happens inside a
          container with no network and a read-only filesystem; without a container runtime it
          is refused rather than run with your privileges.
        </EmptyState>
      )}

      {results && (
        <>
          <SectionRule
            title="Run"
            note={[kind, results.data_source, results.sandbox_backend].filter(Boolean).join(' · ')}
          />

          {verdicts.length > 0 && (
            <Panel label="Adjudication" className="mb-6">
              <VerdictStrip verdicts={verdicts} />
              <table className="mt-5 w-full border-collapse text-[13px]">
                <thead>
                  <tr>
                    {['Quantity', 'Expected', 'Measured', 'Verdict'].map((h) => (
                      <th
                        key={h}
                        className="border-b border-[var(--rule-strong)] py-[7px] pr-2.5 text-left font-mono text-[9.5px] font-medium uppercase tracking-[0.11em] text-[var(--ink-3)]"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {verdicts.map((v, i) => (
                    <tr
                      key={i}
                      className={v.status === 'refuted' ? 'bg-[var(--oxide-soft)]' : undefined}
                    >
                      <td className="border-b border-[var(--rule)] py-[9px] pr-2.5 align-top">
                        {v.quantity}
                      </td>
                      <td className="num border-b border-[var(--rule)] py-[9px] pr-2.5 align-top whitespace-nowrap">
                        {v.expected ?? '—'} {v.unit}
                      </td>
                      <td className="num border-b border-[var(--rule)] py-[9px] pr-2.5 align-top whitespace-nowrap">
                        {v.observed ?? '—'} {v.unit}
                      </td>
                      <td className="border-b border-[var(--rule)] py-[9px] pr-2.5 align-top">
                        <span className="capitalize">{v.status.replace('_', ' ')}</span>
                        {v.reason && (
                          <div className="max-w-[46ch] text-[11.5px] leading-snug text-[var(--ink-3)]">
                            {v.reason}
                          </div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Panel>
          )}

          {results.stdout && (
            <>
              <SectionRule title="Output" />
              <Panel>
                <pre className="num max-h-[420px] overflow-auto whitespace-pre-wrap break-words bg-[var(--sunken)] p-4 text-[12px] leading-relaxed text-[var(--ink-2)]">
                  {results.stdout}
                </pre>
              </Panel>
            </>
          )}
        </>
      )}
    </div>
  );
}
