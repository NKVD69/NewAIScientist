import { useCallback, useEffect, useState } from 'react';

export type Theme = 'light' | 'dark';

const STORAGE_KEY = 'nais-theme';

function currentTheme(): Theme {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

/**
 * Theme state, seeded from whatever the inline <head> script already
 * resolved so the hook never disagrees with the painted DOM.
 *
 * Follows the OS until the user makes an explicit choice, then honours
 * that choice and stops following.
 */
export function useTheme() {
  const [theme, setThemeState] = useState<Theme>(currentTheme);

  const setTheme = useCallback((next: Theme) => {
    document.documentElement.setAttribute('data-theme', next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* private mode: the choice lasts for this session only */
    }
    setThemeState(next);
  }, []);

  const toggle = useCallback(() => {
    setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
  }, [setTheme]);

  // Keep following the OS until an explicit choice has been stored.
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (event: MediaQueryListEvent) => {
      let saved: string | null = null;
      try {
        saved = localStorage.getItem(STORAGE_KEY);
      } catch {
        /* ignore */
      }
      if (saved) return;
      const next: Theme = event.matches ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      setThemeState(next);
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  return { theme, setTheme, toggle };
}
