import type { Verdict, VerdictStatus } from '../../types/domain';
import { cn } from '../../utils/cn';

/**
 * SIGNATURE COMPONENT — the verdict strip.
 *
 * One cell per pre-registered prediction. It makes visible at a glance
 * what the engine can finally express: refuted, corroborated, untested,
 * consistent-but-unscored. The refutation stamp overprints, the way a
 * stamp lands on a register — when the product says no, it should look
 * like it means it.
 *
 * The distinction that must never be lost here: `untested` is neutral
 * hatching, not red. Silence is not evidence in either direction, and
 * colouring it as failure would undo the adjudication work.
 */

const CELL: Record<VerdictStatus, { glyph: string; className: string; label: string }> = {
  refuted: {
    glyph: '\u2715',
    label: 'refuted',
    className:
      'bg-[var(--oxide)] border-[var(--oxide)] text-[var(--oxide-on)] shadow-[0_0_0_2px_var(--oxide-soft)]',
  },
  corroborated: {
    glyph: '\u25CF',
    label: 'corroborated',
    className: 'bg-[var(--verdigris-soft)] border-[var(--verdigris)] text-[var(--verdigris)]',
  },
  consistent_unscored: {
    glyph: '\u25CB',
    label: 'consistent, not credited',
    className: 'bg-[var(--sunken)] border-[var(--rule-strong)] text-[var(--ink-3)]',
  },
  untested: {
    glyph: '',
    label: 'untested',
    className:
      'border-dashed border-[var(--rule-strong)] text-[var(--ink-3)] ' +
      '[background:repeating-linear-gradient(45deg,transparent,transparent_4px,var(--rule)_4px,var(--rule)_5px)]',
  },
  unfalsifiable: {
    glyph: '?',
    label: 'unfalsifiable',
    className: 'bg-[var(--amber-soft)] border-[var(--amber)] text-[var(--amber)]',
  },
  invalid: {
    glyph: '\u2298',
    label: 'invalid',
    className: 'bg-[var(--sunken)] border-[var(--amber)] text-[var(--amber)]',
  },
};

function describe(v: Verdict): string {
  const cell = CELL[v.status] ?? CELL.untested;
  const head = `${v.quantity || 'unnamed'} — ${cell.label}`;
  if (v.observed !== null && v.expected !== null) {
    return `${head}: measured ${v.observed} against ${v.expected}${v.unit ? ` ${v.unit}` : ''} expected`;
  }
  return v.reason ? `${head}: ${v.reason}` : head;
}

export default function VerdictStrip({ verdicts }: { verdicts: Verdict[] }) {
  if (!verdicts.length) {
    return (
      <p className="mt-3.5 text-[12px] text-[var(--ink-3)]">
        No predictions were sealed for this hypothesis, so nothing can be adjudicated.
      </p>
    );
  }

  return (
    <div className="mt-3.5 flex gap-[3px]" role="group" aria-label="Verdict per prediction">
      {verdicts.map((v, i) => {
        const cell = CELL[v.status] ?? CELL.untested;
        return (
          <div
            key={`${v.quantity}-${i}`}
            tabIndex={0}
            title={describe(v)}
            className={cn(
              'group relative grid h-[34px] w-[34px] cursor-default place-items-center',
              'border font-mono text-[13px] font-semibold',
              cell.className,
            )}
          >
            {cell.glyph}
            <span
              role="tooltip"
              className="pointer-events-none absolute bottom-[calc(100%+7px)] left-1/2 z-20
                         -translate-x-1/2 whitespace-nowrap bg-[var(--ink)] px-2.5 py-[7px]
                         font-sans text-[11.5px] font-normal normal-case tracking-normal
                         text-[var(--ground)] opacity-0 transition-opacity
                         group-hover:opacity-100 group-focus-visible:opacity-100"
            >
              {describe(v)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
