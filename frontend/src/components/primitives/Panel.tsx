import { cn } from '../../utils/cn';

/** A raised working surface. The base unit of every page. */
export function Panel({
  title,
  label,
  actions,
  className,
  children,
}: {
  title?: string;
  label?: string;
  actions?: React.ReactNode;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section className={cn('entry', className)}>
      {(title || label || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-[var(--rule)] px-[22px] py-3.5">
          <div>
            {label && <div className="panel-label">{label}</div>}
            {title && <h2 className="text-[17px] leading-tight">{title}</h2>}
          </div>
          {actions}
        </header>
      )}
      <div className="px-[22px] py-[18px]">{children}</div>
    </section>
  );
}

export function PageHeader({
  title,
  lede,
  actions,
}: {
  title: string;
  lede?: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
      <div>
        <h1 className="text-[26px] leading-tight">{title}</h1>
        {lede && <p className="mt-1 max-w-[62ch] text-[13.5px] text-[var(--ink-2)]">{lede}</p>}
      </div>
      {actions}
    </div>
  );
}

export function SectionRule({ title, note }: { title: string; note?: string }) {
  return (
    <div className="mb-4 mt-8 flex items-baseline gap-3 first:mt-0">
      <h2 className="text-[15px]">{title}</h2>
      <span className="h-px flex-1 bg-[var(--rule-strong)]" />
      {note && <span className="panel-label">{note}</span>}
    </div>
  );
}
