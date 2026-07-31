import type { PipelineReport, SessionMeters, TaskState } from '../types/domain';
import ThemeToggle from './primitives/ThemeToggle';
import { cn } from '../utils/cn';

/**
 * The rail shows the run's actual DAG state.
 *
 * It replaces the 1→6 step sidebar, which described a waterfall when the
 * pipeline is a loop with revision feeding back into generation. Here:
 * the real waves, which tasks ran in parallel, which failed, and — the
 * part that matters — why the downstream ones were skipped.
 */

const GLYPH: Record<TaskState, string> = {
  succeeded: '\u25CF',
  failed: '\u2715',
  skipped: '\u25A1',
  running: '\u25B6',
  pending: '\u00B7',
};

const TONE: Record<TaskState, string> = {
  succeeded: 'text-[var(--verdigris)]',
  failed: 'text-[var(--oxide)] font-semibold',
  skipped: 'text-[var(--ink-3)]',
  running: 'text-[var(--violet)]',
  pending: 'text-[var(--ink-3)]',
};

function Meter({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex justify-between py-[3px] text-[11.5px] text-[var(--ink-2)]">
      <span>{label}</span>
      <span className={cn('num', tone)}>{value}</span>
    </div>
  );
}

export default function RunRail({
  report,
  meters,
  reachable,
  phase,
  iteration,
}: {
  report: PipelineReport | null;
  meters: SessionMeters | null;
  /** null while the first poll is in flight. */
  reachable?: boolean | null;
  phase?: string;
  iteration?: number;
}) {
  return (
    <aside
      className="sticky top-0 h-screen w-[252px] shrink-0 overflow-y-auto border-r
                 border-[var(--rule-strong)] bg-[var(--raised)] pb-10 pt-[22px]
                 transition-colors"
    >
      <div className="flex items-start justify-between gap-2.5 border-b border-[var(--rule)] px-[22px] pb-[18px]">
        <div>
          <h1 className="text-[17px] leading-tight">NewAIScientist</h1>
          <div className="mt-[3px] font-mono text-[11px] tracking-wide text-[var(--ink-3)]">
            {reachable === false
              ? 'backend unreachable'
              : phase
                ? `${phase}${iteration ? ` · iteration ${iteration}` : ''}`
                : 'idle'}
          </div>
        </div>
        <ThemeToggle />
      </div>

      {reachable === false && (
        <p className="mx-[22px] mt-5 border-l-[3px] border-[var(--oxide)] bg-[var(--oxide-soft)] px-3 py-2.5 text-[12px] leading-relaxed text-[var(--ink-2)]">
          Cannot reach the API on {import.meta.env.VITE_API_URL ?? 'localhost:8000'}. The panel
          below shows the last state received, which may be stale.
        </p>
      )}

      {!report && reachable !== false && (
        <p className="px-[22px] py-6 text-[12.5px] leading-relaxed text-[var(--ink-3)]">
          No run yet. Start one from the dashboard and the pipeline appears here, wave by wave,
          with the reason for anything that fails or gets skipped.
        </p>
      )}

      {report?.waves.map((wave, i) => (
        <div key={i} className="px-[22px] pb-1 pt-3.5">
          <div className="mb-[9px] flex items-baseline justify-between">
            <span className="font-mono text-[10px] tracking-[0.1em] text-[var(--ink-3)]">
              WAVE {i + 1}
            </span>
            {wave.length > 1 && (
              <span className="border border-[var(--rule-strong)] px-1 py-px font-mono text-[9px] tracking-wide text-[var(--ink-3)]">
                parallel
              </span>
            )}
          </div>

          {wave.map((name) => {
            const task = report.results[name];
            const state: TaskState = task?.state ?? 'pending';
            const why = task?.error || task?.skipped_because || '';
            return (
              <div key={name} className="grid grid-cols-[16px_1fr] items-baseline gap-[9px] py-[5px]">
                <span className={cn('text-center font-mono text-[12px] leading-tight', TONE[state])}>
                  {GLYPH[state]}
                </span>
                <span
                  className={cn(
                    'text-[13px]',
                    state === 'failed' && 'font-medium text-[var(--oxide)]',
                    state === 'skipped' && 'text-[var(--ink-3)]',
                  )}
                >
                  {name}
                </span>
                {why && (
                  <span
                    className={cn(
                      'col-start-2 mt-px text-[11.5px] leading-snug',
                      state === 'failed' ? 'text-[var(--oxide)]' : 'text-[var(--ink-3)]',
                    )}
                  >
                    {why}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      ))}

      {report && (
        <div
          className={cn(
            'mx-[22px] mt-4 border-l-[3px] px-3 py-2 text-[11.5px] leading-snug',
            report.clean
              ? 'border-[var(--verdigris)] bg-[var(--verdigris-soft)] text-[var(--ink-2)]'
              : 'border-[var(--oxide)] bg-[var(--oxide-soft)] text-[var(--ink-2)]',
          )}
        >
          {report.clean ? (
            <>Every task completed in <span className="num">{report.duration_s}s</span>.</>
          ) : (
            <>
              <strong className="font-semibold text-[var(--ink)]">Incomplete run.</strong>{' '}
              Anything produced rests on a partial evidence base.
            </>
          )}
        </div>
      )}

      {meters && (
        <div className="mt-3.5 border-t border-[var(--rule)] px-[22px] pt-[18px]">
          <div className="panel-label mb-2">Session</div>
          <Meter
            label="LLM calls"
            value={
              meters.max_llm_calls
                ? `${meters.llm_calls} / ${meters.max_llm_calls}`
                : String(meters.llm_calls)
            }
          />
          <Meter
            label="Spend"
            value={
              meters.max_cost_usd
                ? `$${meters.cost_usd.toFixed(2)} / $${meters.max_cost_usd.toFixed(2)}`
                : `$${meters.cost_usd.toFixed(2)}`
            }
          />
          <Meter
            label="Judge consistency"
            value={
              meters.judge_order_invariance === null
                ? 'n/a'
                : `${Math.round(meters.judge_order_invariance * 100)}%`
            }
            tone={
              meters.judge_order_invariance !== null && meters.judge_order_invariance < 0.6
                ? 'text-[var(--oxide)]'
                : 'text-[var(--verdigris)]'
            }
          />
          <Meter
            label="Sandbox"
            value={meters.sandbox_will_execute ? meters.sandbox_backend : 'unavailable'}
            tone={meters.sandbox_will_execute ? undefined : 'text-[var(--oxide)]'}
          />
        </div>
      )}
    </aside>
  );
}
