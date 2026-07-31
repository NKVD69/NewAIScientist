import { Outlet } from 'react-router-dom';
import RunRail from './RunRail';
import { useSession } from '../session/useSession';

export default function MainLayout() {
  const { state, reachable } = useSession();

  return (
    <div className="flex min-h-screen w-full">
      <RunRail
        report={state?.report ?? null}
        meters={state?.meters ?? null}
        reachable={reachable}
        phase={state?.phase}
        iteration={state?.iteration}
      />
      <main className="min-w-0 flex-1 px-5 pb-24 pt-7 md:px-10">
        <div className="mx-auto max-w-[1180px]">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
