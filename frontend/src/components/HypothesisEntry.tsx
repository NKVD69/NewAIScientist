import { useState } from 'react';
import type { Hypothesis, Verdict } from '../types/domain';
import RatingSpread from './primitives/RatingSpread';
import VerdictStrip from './primitives/VerdictStrip';
import { cn } from '../utils/cn';

/** A hypothesis rendered as a lab-record page. */

function Section({
  label,
  defaultOpen = false,
  children,
}: {
  label: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-t border-[var(--rule)]">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 px-[22px] py-3 text-left hover:bg-[var(--sunken)]"
      >
        <span className="panel-label">{label}</span>
        <span
          className={cn(
            'inline-block font-mono text-[11px] text-[var(--ink-3)] transition-transform',
            open && 'rotate-90',
          )}
        >
          {'\u25B6'}
        </span>
      </button>
      {open && <div className="px-[22px] pb-[22px] pt-1">{children}</div>}
    </div>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return (
    <div className="mt-3.5 border-l-[3px] border-[var(--amber)] bg-[var(--amber-soft)] px-3.5 py-[11px] text-[12.5px] leading-relaxed text-[var(--ink-2)]">
      {children}
    </div>
  );
}

const fmt = (n: number | null, unit = '') =>
  n === null ? '\u2014' : `${Number(n.toPrecision(4))}${unit ? ` ${unit}` : ''}`;

function AdjudicationTable({ verdicts }: { verdicts: Verdict[] }) {
  return (
    <table className="w-full border-collapse text-[13px]">
      <thead>
        <tr>
          {['Quantity', 'Expected', 'Measured', 'Deviation', 'Verdict'].map((h, i) => (
            <th
              key={h}
              className={cn(
                'border-b border-[var(--rule-strong)] py-[7px] pr-2.5 text-left font-mono',
                'text-[9.5px] font-medium uppercase tracking-[0.11em] text-[var(--ink-3)]',
                i === 0 && 'w-[22%]',
                i === 4 && 'w-[38%]',
              )}
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {verdicts.map((v, i) => {
          const refuted = v.status === 'refuted';
          return (
            <tr key={`${v.quantity}-${i}`} className={cn(refuted && 'bg-[var(--oxide-soft)]')}>
              <td className="border-b border-[var(--rule)] py-[9px] pr-2.5 align-top">
                {v.quantity}
              </td>
              <td className={cn('num border-b border-[var(--rule)] py-[9px] pr-2.5 align-top whitespace-nowrap', refuted && 'font-semibold text-[var(--oxide)]')}>
                {fmt(v.expected, v.unit)}
              </td>
              <td className={cn('num border-b border-[var(--rule)] py-[9px] pr-2.5 align-top whitespace-nowrap', refuted && 'font-semibold text-[var(--oxide)]')}>
                {fmt(v.observed, v.unit)}
              </td>
              <td className={cn('num border-b border-[var(--rule)] py-[9px] pr-2.5 align-top whitespace-nowrap', refuted && 'font-semibold text-[var(--oxide)]')}>
                {fmt(v.deviation)}
              </td>
              <td className="border-b border-[var(--rule)] py-[9px] pr-2.5 align-top">
                {refuted ? (
                  <strong className="text-[var(--oxide)]">Refuted.</strong>
                ) : (
                  <span className="capitalize">{v.status.replace('_', ' ')}.</span>
                )}
                {v.reason && (
                  <div
                    className={cn(
                      'max-w-[46ch] text-[11.5px] leading-snug',
                      refuted ? 'text-[var(--oxide)]' : 'text-[var(--ink-3)]',
                    )}
                  >
                    {v.reason}
                  </div>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

export default function HypothesisEntry({ hyp }: { hyp: Hypothesis }) {
  const verdicts = hyp.verdicts ?? [];
  const counts = {
    refuted: verdicts.filter((v) => v.status === 'refuted').length,
    corroborated: verdicts.filter((v) => v.status === 'corroborated').length,
    untested: verdicts.filter((v) => v.status === 'untested' || v.status === 'invalid').length,
  };
  const novelty = hyp.novelty_report;

  return (
    <article className="entry mb-[22px]">
      <div className="border-b border-[var(--rule)] px-[22px] pb-4 pt-[18px]">
        <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-start">
          <div className="min-w-0">
            <h2 className="max-w-[62ch] text-[19px] leading-snug">{hyp.title}</h2>

            <div className="mt-[7px] flex flex-wrap items-center gap-3.5">
              <span className="border border-[var(--rule-strong)] px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[var(--ink-2)]">
                {hyp.generation_method}
              </span>
              {hyp.parent_ids?.length > 0 && (
                <span className="border border-[var(--violet)] bg-[var(--violet-soft)] px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[var(--violet)]">
                  {'\u21B3'} evolved from H-{hyp.parent_ids[0].slice(-4)}
                </span>
              )}
              <span className="border border-[var(--rule-strong)] px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider text-[var(--ink-2)]">
                {verdicts.length} {verdicts.length === 1 ? 'prediction' : 'predictions'} sealed
              </span>
            </div>

            <VerdictStrip verdicts={verdicts} />
          </div>

          <RatingSpread mu={hyp.rating_mu} sigma={hyp.rating_sigma} matches={hyp.rating_matches} />
        </div>
      </div>

      {verdicts.length > 0 && (
        <Section
          label={`Adjudication — ${counts.refuted} refuted, ${counts.corroborated} corroborated, ${counts.untested} untested`}
          defaultOpen={counts.refuted > 0}
        >
          <AdjudicationTable verdicts={verdicts} />
          {counts.untested === verdicts.length && (
            <Note>
              <strong className="text-[var(--ink)]">Nothing was measured.</strong> Every sealed
              prediction is untested, so this hypothesis carries no empirical weight in either
              direction.
            </Note>
          )}
        </Section>
      )}

      <Section
        label={
          novelty?.searched
            ? `Novelty — ${novelty.level.replace('_', ' ')} (${novelty.score.toFixed(2)})`
            : 'Novelty — not assessed'
        }
      >
        {!novelty?.searched ? (
          <Note>
            <strong className="text-[var(--ink)]">The prior-art search didn&rsquo;t run</strong>
            {novelty?.error ? ` — ${novelty.error}` : ''}. Novelty is unknown for this
            hypothesis. That is an absence of measurement, not evidence of originality.
          </Note>
        ) : (
          <>
            <p className="m-0 mb-3 max-w-[66ch] text-[13px] text-[var(--ink-2)]">
              Semantic Scholar search on <span className="num text-[12px]">{novelty.query}</span>,
              similarity by {novelty.similarity_method === 'embedding' ? 'embeddings' : 'token overlap'}.
              Judge for yourself:
            </p>
            {novelty.prior_art.length === 0 ? (
              <p className="text-[13px] text-[var(--ink-3)]">
                No closely related work surfaced.
              </p>
            ) : (
              novelty.prior_art.slice(0, 5).map((hit, i) => (
                <div
                  key={i}
                  className="grid grid-cols-[44px_1fr] gap-3 border-b border-[var(--rule)] py-[11px] last:border-b-0"
                >
                  <span
                    className={cn(
                      'num text-[15px] font-semibold',
                      hit.similarity >= 0.85 && 'text-[var(--oxide)]',
                    )}
                  >
                    {hit.similarity.toFixed(2)}
                  </span>
                  <div>
                    <div className="text-[13.5px] leading-snug">{hit.title}</div>
                    <div className="mt-[3px] font-mono text-[11.5px] text-[var(--ink-3)]">
                      {[hit.year, hit.venue, hit.citation_count ? `${hit.citation_count} citations` : null]
                        .filter(Boolean)
                        .join(' · ')}
                      {hit.url && (
                        <>
                          {' · '}
                          <a
                            href={hit.url}
                            target="_blank"
                            rel="noreferrer"
                            className="border-b border-current text-[var(--violet)] no-underline hover:text-[var(--ink)]"
                          >
                            semanticscholar.org {'\u2197'}
                          </a>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              ))
            )}
          </>
        )}
      </Section>

      <Section
        label={
          hyp.multiverse_fragility > 0
            ? `Robustness — ${hyp.multiverse_fragility > 0.5 ? 'fragile' : 'robust'} (${hyp.multiverse_fragility.toFixed(2)})`
            : 'Robustness — not assessed'
        }
      >
        {hyp.multiverse_fragility > 0 ? (
          <>
            <p className="m-0 text-[13px] text-[var(--ink-2)]">
              The conclusion holds in{' '}
              <span className="num">{Math.round((1 - hyp.multiverse_fragility) * 100)}%</span> of
              defensible analytic specifications.
            </p>
            {hyp.multiverse_fragility > 0.5 && (
              <Note>
                <strong className="text-[var(--ink)]">
                  This result depends on arbitrary analytic choices.
                </strong>{' '}
                Most defensible specifications do not reproduce it.
              </Note>
            )}
          </>
        ) : (
          <Note>
            <strong className="text-[var(--ink)]">The multiverse analysis didn&rsquo;t run.</strong>{' '}
            No fragility index is available. That is an absence of measurement, not a sign of
            robustness.
          </Note>
        )}
      </Section>

      {hyp.prediction_hash && (
        <div className="flex flex-wrap items-center gap-2.5 border-t border-[var(--rule)] bg-[var(--sunken)] px-[22px] py-2.5 font-mono text-[11px] text-[var(--ink-3)]">
          <span>Predictions sealed {hyp.registered_at?.slice(0, 16).replace('T', ' ')}</span>
          <span>·</span>
          <span>sha256:{hyp.prediction_hash.slice(0, 12)}</span>
          <span>·</span>
          <span className="font-semibold text-[var(--verdigris)]">intact {'\u2713'}</span>
        </div>
      )}
    </article>
  );
}
