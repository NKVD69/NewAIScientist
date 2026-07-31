import { useCallback, useEffect, useRef, useState } from 'react';
import { workflowApi, type SessionState } from '../services/api';

/**
 * Polls the backend for session state.
 *
 * Deliberately quiet about failure: if the backend is down the rail keeps
 * showing its last known state rather than flashing an error across the
 * whole interface. A page that issued a request surfaces its own error,
 * with the detail that matters.
 *
 * Polls faster while a run is in flight, then backs off, so an idle tab
 * is not hammering the API.
 */
export function useSession(activeIntervalMs = 2000, idleIntervalMs = 15000) {
  const [state, setState] = useState<SessionState | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);
  const timer = useRef<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const res = await workflowApi.getState();
      setState(res.data);
      setReachable(true);
      return res.data;
    } catch {
      setReachable(false);
      return null;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function tick() {
      const data = await refresh();
      if (cancelled) return;
      // A run is in flight when the latest report still has work outstanding.
      const running = Object.values(data?.report?.results ?? {}).some(
        (r) => r.state === 'running' || r.state === 'pending',
      );
      timer.current = window.setTimeout(tick, running ? activeIntervalMs : idleIntervalMs);
    }

    tick();
    return () => {
      cancelled = true;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [refresh, activeIntervalMs, idleIntervalMs]);

  return { state, reachable, refresh };
}
