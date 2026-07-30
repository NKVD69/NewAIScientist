/**
 * A Bradley-Terry belief, rendered as a distribution.
 *
 * The deliberate risk in this design: a score is never a point. The band
 * shows mu ± 2 sigma, so a thinly-observed hypothesis looks visibly
 * blurry next to a well-observed one. That blur is the message, not a
 * caveat — the old UI showed a bare Elo number and gave a hypothesis
 * with one lucky win the same visual authority as one with thirty
 * matches behind it.
 */

const FLOOR = 900;
const CEIL = 1500;

const pct = (v: number) => ((Math.min(CEIL, Math.max(FLOOR, v)) - FLOOR) / (CEIL - FLOOR)) * 100;

export default function RatingSpread({
  mu,
  sigma,
  matches,
}: {
  mu: number;
  sigma: number;
  matches: number;
}) {
  const low = mu - 1.96 * sigma;
  const high = mu + 1.96 * sigma;
  const left = pct(low);
  const right = 100 - pct(high);

  return (
    <div className="w-full min-w-0 shrink-0 sm:min-w-[214px]">
      <div className="panel-label">Estimated strength</div>

      <div className="flex items-baseline gap-[7px]">
        <span className="num text-[26px] font-semibold leading-none">
          {Math.round(mu).toLocaleString('en-US')}
        </span>
        <span className="num text-[13px] text-[var(--ink-3)]">± {Math.round(sigma)}</span>
      </div>

      <div
        className="relative mt-[9px] h-[26px] border-x border-[var(--rule-strong)]"
        role="img"
        aria-label={`Estimated strength ${Math.round(mu)}, 95% credible interval ${Math.round(low)} to ${Math.round(high)}, from ${matches} matches`}
      >
        <div className="absolute inset-x-0 top-[13px] h-px bg-[var(--rule)]" />
        <div
          className="absolute top-[7px] h-[13px] opacity-30"
          style={{
            left: `${left}%`,
            right: `${right}%`,
            background:
              'linear-gradient(90deg, transparent, var(--ink-2) 22%, var(--ink-2) 78%, transparent)',
          }}
        />
        <div
          className="absolute top-[2px] h-[23px] w-0.5 bg-[var(--ink)]"
          style={{ left: `${pct(mu)}%` }}
        />
        <span className="num absolute -top-0.5 left-0 text-[9.5px] text-[var(--ink-3)]">
          {FLOOR}
        </span>
        <span className="num absolute -top-0.5 right-0 text-[9.5px] text-[var(--ink-3)]">
          {CEIL}
        </span>
      </div>

      <div className="mt-[5px] text-[11px] leading-snug text-[var(--ink-3)]">
        {matches} {matches === 1 ? 'match' : 'matches'} · 95% interval{' '}
        <span className="num">{Math.round(low).toLocaleString('en-US')}</span> –{' '}
        <span className="num">{Math.round(high).toLocaleString('en-US')}</span>
      </div>
    </div>
  );
}
