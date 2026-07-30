import { Outlet } from 'react-router-dom';
import { useState } from 'react';
import RunRail from './RunRail';
import type { PipelineReport, SessionMeters } from '../types/domain';

export default function MainLayout() {
  // Replace with a session poll against /session/state once the backend
  // exposes the pipeline report and budget meters.
  const [report] = useState<PipelineReport | null>(null);
  const [meters] = useState<SessionMeters | null>(null);

  return (
    <div className="flex min-h-screen w-full">
      <RunRail report={report} meters={meters} />
      <main className="min-w-0 flex-1 px-5 pb-24 pt-7 md:px-10">
        <div className="mx-auto max-w-[1180px]">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
