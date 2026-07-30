import { useTheme } from '../../theme/useTheme';

export default function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const dark = theme === 'dark';

  return (
    <button
      type="button"
      onClick={toggle}
      title="Switch theme"
      aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      className="grid h-[30px] w-[30px] shrink-0 place-items-center border border-[var(--rule-strong)]
                 bg-[var(--sunken)] p-0 font-mono text-[13px] leading-none text-[var(--ink-2)]
                 transition-colors hover:border-[var(--ink-3)] hover:text-[var(--ink)]"
    >
      {dark ? '\u2600' : '\u25D0'}
    </button>
  );
}
