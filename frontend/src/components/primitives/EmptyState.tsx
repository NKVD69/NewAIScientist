/** Says what is missing and what to do about it. Never just "no data". */
export default function EmptyState({
  title,
  children,
  action,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <div className="border border-dashed border-[var(--rule-strong)] bg-[var(--raised)] px-8 py-14 text-center">
      <h2 className="text-[17px]">{title}</h2>
      <p className="mx-auto mt-2 max-w-[48ch] text-[13.5px] leading-relaxed text-[var(--ink-2)]">
        {children}
      </p>
      {action && <div className="mt-6 flex justify-center">{action}</div>}
    </div>
  );
}
