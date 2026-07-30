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
  runId,
  startedAt,
}: {
  report: PipelineReport | null;
  meters: SessionMeters | null;
  runId?: string;
  startedAt?: string;
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
            {runId ? `run ${runId}` : 'no active run'}
            {startedAt ? ` · ${startedAt}` : ''}
          </div>
        </div>
        <ThemeToggle />
      </div>

      {!report && (
        <p className="px-[22px] py-6 text-[12.5px] leading-relaxed text-[var(--ink-3)]">
          No run yet. Start one from the dashboard and the pipeline will appear here, wave by
          wave.
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
