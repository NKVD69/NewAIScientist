import type { PipelineReport } from '../../types/domain';

/**
 * A partial run has to look partial.
 *
 * This banner is the only thing standing between a manuscript grounded
 * in nothing and one grounded in something. It is not dismissible, and
 * it does not soften: the copy names what failed, what was skipped as a
 * consequence, and what the user can do about it.
 */
export default function RunHealth({ report }: { report: PipelineReport | null }) {
  if (!report) return null;

  const failed = Object.values(report.results).filter((r) => r.state === 'failed');
  const skipped = Object.values(report.results).filter((r) => r.state === 'skipped');

  if (report.clean) {
    return (
      <div className="mb-[30px] flex items-start gap-4 border border-l-4 border-[var(--verdigris)] bg-[var(--verdigris-soft)] px-[18px] py-3.5">
        <span className="shrink-0 whitespace-nowrap border border-current px-[7px] py-[3px] font-mono text-[11px] font-semibold uppercase tracking-[0.1em] text-[var(--verdigris)]">
          Run complete
        </span>
        <p className="m-0 text-[13.5px] leading-normal text-[var(--ink-2)]">
          Every task finished. Results below rest on a complete evidence base.
        </p>
      </div>
    );
  }

  const firstFailure = failed[0];
  const consequence = skipped.length
    ? ` ${skipped.length === 1 ? 'One task was' : `${skipped.length} tasks were`} skipped as a result: ${skipped
        .map((s) => s.name)
        .join(', ')}.`
    : '';

  return (
    <div className="mb-[30px] flex items-start gap-4 border border-l-4 border-[var(--oxide)] bg-[var(--oxide-soft)] px-[18px] py-3.5">
      <span className="shrink-0 whitespace-nowrap border border-current px-[7px] py-[3px] font-mono text-[11px] font-semibold uppercase tracking-[0.1em] text-[var(--oxide)]">
        {report.aborted ? 'Run aborted' : 'Run incomplete'}
      </span>
      <p className="m-0 text-[13.5px] leading-normal text-[var(--ink-2)]">
        <strong className="font-semibold text-[var(--ink)]">
          {failed.length} {failed.length === 1 ? 'task' : 'tasks'} didn&rsquo;t finish.
        </strong>{' '}
        {firstFailure && (
          <>
            {firstFailure.name} failed
            {firstFailure.error ? `: ${firstFailure.error}` : '.'}
          </>
        )}
        {consequence} Anything produced below rests on a partial evidence base and should be
        reported as such.
      </p>
    </div>
  );
}
