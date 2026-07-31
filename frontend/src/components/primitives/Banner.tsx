import { cn } from '../../utils/cn';

type Tone = 'oxide' | 'verdigris' | 'amber';

const TONE: Record<Tone, { border: string; bg: string; text: string }> = {
  oxide: { border: 'border-[var(--oxide)]', bg: 'bg-[var(--oxide-soft)]', text: 'text-[var(--oxide)]' },
  verdigris: {
    border: 'border-[var(--verdigris)]',
    bg: 'bg-[var(--verdigris-soft)]',
    text: 'text-[var(--verdigris)]',
  },
  amber: { border: 'border-[var(--amber)]', bg: 'bg-[var(--amber-soft)]', text: 'text-[var(--amber)]' },
};

/**
 * A stated condition, not a toast.
 *
 * Copy rule: name what happened and what the reader can do. No apology,
 * no "oops", never "something went wrong".
 */
export default function Banner({
  tone = 'oxide',
  mark,
  children,
}: {
  tone?: Tone;
  mark: string;
  children: React.ReactNode;
}) {
  const t = TONE[tone];
  return (
    <div className={cn('mb-[30px] flex items-start gap-4 border border-l-4 px-[18px] py-3.5', t.border, t.bg)}>
      <span
        className={cn(
          'shrink-0 whitespace-nowrap border border-current px-[7px] py-[3px]',
          'font-mono text-[11px] font-semibold uppercase tracking-[0.1em]',
          t.text,
        )}
      >
        {mark}
      </span>
      <p className="m-0 text-[13.5px] leading-normal text-[var(--ink-2)]">{children}</p>
    </div>
  );
}

/** Renders an API failure without hiding what actually happened. */
export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null;
  const e = error as { response?: { data?: { detail?: string } }; message?: string };
  const detail =
    e?.response?.data?.detail ??
    e?.message ??
    'The request never reached the backend. Check that it is running on port 8000.';
  return (
    <Banner mark="Failed">
      <span className="font-mono text-[12.5px]">{detail}</span>
    </Banner>
  );
}
