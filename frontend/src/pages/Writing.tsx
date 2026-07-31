import { useState } from 'react';
import { FileText, Download } from 'lucide-react';
import { workflowApi } from '../services/api';
import { PageHeader, Panel, SectionRule } from '../components/primitives/Panel';
import Button from '../components/primitives/Button';
import EmptyState from '../components/primitives/EmptyState';
import Banner, { ErrorBanner } from '../components/primitives/Banner';
import { cn } from '../utils/cn';

export default function Writing() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [manuscript, setManuscript] = useState<any>(null);
  const [active, setActive] = useState(0);

  async function draft() {
    setLoading(true);
    setError(null);
    try {
      const res = await workflowApi.runWriting();
      setManuscript(res.data);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  const sections: { name: string; content: string }[] =
    manuscript?.sections ??
    (manuscript
      ? Object.entries(manuscript)
          .filter(([, v]) => typeof v === 'string' && (v as string).length > 80)
          .map(([k, v]) => ({ name: k, content: v as string }))
      : []);

  return (
    <div>
      <PageHeader
        title="Manuscript"
        lede="Drafted from the leading hypothesis and its adjudicated results."
        actions={
          <Button onClick={draft} loading={loading} icon={<FileText className="h-4 w-4" />}>
            {loading ? 'Drafting' : 'Draft manuscript'}
          </Button>
        }
      />

      <ErrorBanner error={error} />

      {manuscript && manuscript.run_was_clean === false && (
        <Banner mark="Partial evidence">
          <strong className="font-semibold text-[var(--ink)]">
            This manuscript rests on an incomplete run.
          </strong>{' '}
          Some pipeline tasks failed or were skipped, so parts of the evidence base are missing.
          The limitations section should say so explicitly before this leaves your machine.
        </Banner>
      )}

      {!manuscript && !loading && !error && (
        <EmptyState title="No draft yet">
          Writing runs last and depends on experiment and replication. If those were skipped,
          the draft will be assembled from an incomplete evidence base and will say so.
        </EmptyState>
      )}

      {manuscript && sections.length > 0 && (
        <>
          <SectionRule
            title={manuscript.title ?? 'Draft'}
            note={`${sections.length} sections`}
          />

          <div className="grid gap-6 lg:grid-cols-[196px_1fr]">
            <nav className="lg:sticky lg:top-7 lg:self-start">
              <div className="panel-label mb-2.5">Sections</div>
              <ul className="space-y-0.5">
                {sections.map((s, i) => (
                  <li key={s.name}>
                    <button
                      type="button"
                      onClick={() => setActive(i)}
                      className={cn(
                        'w-full px-2.5 py-1.5 text-left text-[13px] capitalize transition-colors',
                        i === active
                          ? 'bg-[var(--ink)] text-[var(--ground)]'
                          : 'text-[var(--ink-2)] hover:bg-[var(--sunken)]',
                      )}
                    >
                      {s.name.replace(/_/g, ' ')}
                    </button>
                  </li>
                ))}
              </ul>
              <Button variant="secondary" className="mt-5 w-full" icon={<Download className="h-4 w-4" />}>
                Export PDF
              </Button>
            </nav>

            <Panel>
              <article className="max-w-[74ch]">
                <h2 className="mb-4 text-[20px] capitalize">
                  {sections[active]?.name.replace(/_/g, ' ')}
                </h2>
                <div className="whitespace-pre-wrap text-[14.5px] leading-[1.7] text-[var(--ink-2)]">
                  {sections[active]?.content}
                </div>
              </article>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
