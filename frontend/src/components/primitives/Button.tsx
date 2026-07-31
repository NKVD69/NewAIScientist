import { cn } from '../../utils/cn';

/** Primary is ink-on-ground: one loud element per screen, no gradients. */
export default function Button({
  variant = 'primary',
  loading,
  icon,
  children,
  className,
  ...rest
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'primary' | 'secondary';
  loading?: boolean;
  icon?: React.ReactNode;
}) {
  return (
    <button
      type="button"
      {...rest}
      disabled={rest.disabled || loading}
      className={cn(
        'inline-flex items-center justify-center gap-2 border px-5 py-2.5 text-[13px] font-medium',
        'transition-opacity disabled:cursor-not-allowed disabled:opacity-40',
        variant === 'primary'
          ? 'border-[var(--ink)] bg-[var(--ink)] text-[var(--ground)] hover:opacity-85'
          : 'border-[var(--rule-strong)] bg-transparent text-[var(--ink)] hover:bg-[var(--sunken)]',
        className,
      )}
    >
      {loading ? <span className="num text-[13px]">···</span> : icon}
      {children}
    </button>
  );
}
