import React from 'react';
import { 
  Bell, 
  Settings, 
  User, 
  Database,
  Cpu,
  RefreshCw
} from 'lucide-react';

const Topbar = () => {
  return (
    <header className="flex items-center justify-between px-8 py-4 bg-slate-900/40 border-b border-slate-800 backdrop-blur-md">
      <div>
        <h2 className="text-lg font-semibold text-slate-100 flex items-center gap-2">
          Project: <span className="text-sky-400 font-normal">NewAI Scientist v3</span>
        </h2>
        <p className="text-xs text-slate-500 flex items-center gap-1">
          <Database className="w-3 h-3" /> PubMed + ArXiv + GEO Enabled
        </p>
      </div>

      <div className="flex items-center gap-6">
        <div className="flex items-center gap-4 px-4 py-1.5 bg-slate-800/50 rounded-full border border-slate-700/50">
          <div className="flex items-center gap-2 text-xs">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-300">GPT-4o</span>
          </div>
          <div className="h-4 w-px bg-slate-700"></div>
          <div className="flex items-center gap-2 text-xs">
            <RefreshCw className="w-3.5 h-3.5 text-emerald-400" />
            <span className="text-slate-300">Ready</span>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-full transition-all">
            <Bell className="w-5 h-5" />
          </button>
          <button className="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-full transition-all">
            <Settings className="w-5 h-5" />
          </button>
          <div className="h-8 w-px bg-slate-800 mx-1"></div>
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center text-white text-xs font-bold border-2 border-slate-800 shadow-lg cursor-pointer">
            JD
          </div>
        </div>
      </div>
    </header>
  );
};

export default Topbar;
