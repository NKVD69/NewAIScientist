import { cn } from '../../utils/cn';

const control =
  'w-full border border-[var(--rule-strong)] bg-[var(--ground)] px-3.5 py-2.5 text-[14px] ' +
  'text-[var(--ink)] placeholder:text-[var(--ink-3)] transition-colors ' +
  'focus:border-[var(--ink-3)] focus:outline-none';

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block">
      <span className="panel-label mb-1.5 block">{label}</span>
      {children}
      {hint && <span className="mt-1.5 block text-[11.5px] text-[var(--ink-3)]">{hint}</span>}
    </label>
  );
}

export const Input = (p: React.InputHTMLAttributes<HTMLInputElement>) => (
  <input {...p} className={cn(control, p.className)} />
);

export const Textarea = (p: React.TextareaHTMLAttributes<HTMLTextAreaElement>) => (
  <textarea {...p} className={cn(control, 'resize-y', p.className)} />
);

export const Select = (p: React.SelectHTMLAttributes<HTMLSelectElement>) => (
  <select {...p} className={cn(control, p.className)} />
);
