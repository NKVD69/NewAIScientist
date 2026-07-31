import { useState } from 'react';
import { Search, ExternalLink } from 'lucide-react';
import { workflowApi } from '../services/api';
import { PageHeader, Panel, SectionRule } from '../components/primitives/Panel';
import { Field, Input } from '../components/primitives/Field';
import Button from '../components/primitives/Button';
import EmptyState from '../components/primitives/EmptyState';
import Banner, { ErrorBanner } from '../components/primitives/Banner';
import { cn } from '../utils/cn';

const SOURCES = [
  { id: 'arxiv', label: 'arXiv' },
  { id: 'pubmed', label: 'PubMed' },
  { id: 'semanticscholar', label: 'Semantic Scholar' },
];

export default function Literature() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [papers, setPapers] = useState<any[]>([]);
  const [excluded, setExcluded] = useState<any[]>([]);
  const [maxResults, setMaxResults] = useState(5);
  const [sources, setSources] = useState<string[]>(['arxiv', 'semanticscholar']);

  async function search() {
    setLoading(true);
    setError(null);
    try {
      const res = await workflowApi.runLiterature(maxResults, sources);
      setPapers(res.data.papers ?? []);
      setExcluded(res.data.retracted_excluded ?? []);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <PageHeader
        title="Literature"
        lede="Retrieved, deduplicated by DOI, screened for retractions, then indexed for retrieval. Semantic Scholar supplies citation counts and publication types the other sources do not."
        actions={
          <Button onClick={search} loading={loading} icon={<Search className="h-4 w-4" />}>
            {loading ? 'Searching' : 'Search'}
          </Button>
        }
      />

      <ErrorBanner error={error} />

      <Panel label="Query" className="mb-6">
        <div className="grid gap-5 sm:grid-cols-[1fr_160px]">
          <Field label="Sources">
            <div className="flex flex-wrap gap-2 pt-0.5">
              {SOURCES.map((s) => {
                const on = sources.includes(s.id);
                return (
                  <button
                    key={s.id}
                    type="button"
                    onClick={() =>
                      setSources((v) => (on ? v.filter((x) => x !== s.id) : [...v, s.id]))
                    }
                    className={cn(
                      'border px-3 py-1.5 font-mono text-[11px] uppercase tracking-wider transition-colors',
                      on
                        ? 'border-[var(--ink)] bg-[var(--ink)] text-[var(--ground)]'
                        : 'border-[var(--rule-strong)] text-[var(--ink-2)] hover:bg-[var(--sunken)]',
                    )}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </Field>
          <Field label="Max per source">
            <Input
              type="number"
              min={1}
              max={50}
              value={maxResults}
              onChange={(e) => setMaxResults(Number(e.target.value))}
            />
          </Field>
        </div>
      </Panel>

      {excluded.length > 0 && (
        <Banner mark="Excluded" tone="amber">
          <strong className="font-semibold text-[var(--ink)]">
            {excluded.length} retracted {excluded.length === 1 ? 'paper' : 'papers'} removed
          </strong>{' '}
          from the corpus before indexing. Grounding a hypothesis on retracted work is a failure
          mode worth preventing at the source.
        </Banner>
      )}

      {papers.length === 0 && !loading && !error && (
        <EmptyState title="No papers indexed">
          Everything downstream depends on this corpus. Hypothesis generation is blocked until
          the literature phase succeeds, deliberately — ungrounded hypotheses read exactly like
          grounded ones.
        </EmptyState>
      )}

      {papers.length > 0 && (
        <>
          <SectionRule title="Corpus" note={`${papers.length} papers`} />
          {papers.map((p, i) => (
            <article key={p.doi || p.url || i} className="entry mb-3 px-[22px] py-4">
              <h3 className="max-w-[70ch] text-[15px] leading-snug">{p.title}</h3>
              <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11.5px] text-[var(--ink-3)]">
                {p.published && <span>{String(p.published).slice(0, 4)}</span>}
                {p.venue && <span>{p.venue}</span>}
                {typeof p.citation_count === 'number' && (
                  <span>{p.citation_count} citations</span>
                )}
                {p.source && <span className="uppercase">{p.source}</span>}
                {typeof p.quality_weight === 'number' && (
                  <span title="Quality weight from recency, venue, citations and publication type">
                    weight {p.quality_weight}
                  </span>
                )}
                {p.url && (
                  <a
                    href={p.url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 border-b border-current text-[var(--violet)] no-underline hover:text-[var(--ink)]"
                  >
                    source <ExternalLink className="h-3 w-3" />
                  </a>
                )}
              </div>
              {p.integrity_flag && (
                <p className="mt-2 border-l-[3px] border-[var(--amber)] bg-[var(--amber-soft)] px-3 py-2 text-[12px] text-[var(--ink-2)]">
                  {p.integrity_flag}
                </p>
              )}
              {(p.tldr || p.summary) && (
                <p className="mt-2.5 max-w-[80ch] text-[13px] leading-relaxed text-[var(--ink-2)]">
                  {(p.tldr || p.summary).slice(0, 320)}
                  {(p.tldr || p.summary).length > 320 ? '…' : ''}
                </p>
              )}
            </article>
          ))}
        </>
      )}
    </div>
  );
}
