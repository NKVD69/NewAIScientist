import { useState } from 'react';
import { FlaskConical } from 'lucide-react';
import { workflowApi } from '../services/api';
import { PageHeader, Panel, SectionRule } from '../components/primitives/Panel';
import Button from '../components/primitives/Button';
import EmptyState from '../components/primitives/EmptyState';
import { ErrorBanner } from '../components/primitives/Banner';

export default function Protocol() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [protocol, setProtocol] = useState<any>(null);

  async function design() {
    setLoading(true);
    setError(null);
    try {
      const res = await workflowApi.generateProtocol('');
      setProtocol(res.data);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  const steps: string[] = protocol?.steps ?? protocol?.procedure ?? [];
  const materials: string[] = protocol?.materials ?? [];
  const controls: string[] = protocol?.controls ?? [];

  return (
    <div>
      <PageHeader
        title="Protocol"
        lede="A wet-lab procedure for the top-ranked hypothesis, written against its sealed predictions so the measurements it produces can be adjudicated."
        actions={
          <Button onClick={design} loading={loading} icon={<FlaskConical className="h-4 w-4" />}>
            {loading ? 'Designing' : 'Design protocol'}
          </Button>
        }
      />

      <ErrorBanner error={error} />

      {!protocol && !loading && !error && (
        <EmptyState title="No protocol yet">
          Generated for whichever hypothesis leads on μ − 2σ, not on raw score — a hypothesis
          with one lucky win should not send anyone into the lab.
        </EmptyState>
      )}

      {protocol && (
        <>
          {protocol.title && (
            <Panel label="Target" title={protocol.title} className="mb-6">
              {protocol.objective && (
                <p className="max-w-[72ch] text-[13.5px] leading-relaxed text-[var(--ink-2)]">
                  {protocol.objective}
                </p>
              )}
            </Panel>
          )}

          {materials.length > 0 && (
            <>
              <SectionRule title="Materials" />
              <Panel>
                <ul className="grid gap-1.5 sm:grid-cols-2">
                  {materials.map((m, i) => (
                    <li key={i} className="flex gap-2 text-[13.5px]">
                      <span className="num text-[var(--ink-3)]">·</span>
                      {m}
                    </li>
                  ))}
                </ul>
              </Panel>
            </>
          )}

          {steps.length > 0 && (
            <>
              <SectionRule title="Procedure" note={`${steps.length} steps`} />
              <Panel>
                <ol className="space-y-3.5">
                  {steps.map((s, i) => (
                    <li key={i} className="grid grid-cols-[28px_1fr] gap-3">
                      <span className="num text-[12px] text-[var(--ink-3)]">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <span className="text-[13.5px] leading-relaxed">{s}</span>
                    </li>
                  ))}
                </ol>
              </Panel>
            </>
          )}

          {controls.length > 0 && (
            <>
              <SectionRule title="Controls" />
              <Panel>
                <ul className="space-y-1.5">
                  {controls.map((c, i) => (
                    <li key={i} className="text-[13.5px]">
                      {c}
                    </li>
                  ))}
                </ul>
              </Panel>
            </>
          )}
        </>
      )}
    </div>
  );
}
