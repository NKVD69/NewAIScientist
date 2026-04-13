import React, { useState } from 'react';
import { workflowApi } from '../services/api';
import { 
  Beaker, 
  Code2, 
  GitFork, 
  Settings2, 
  Play,
  RotateCcw,
  CheckCircle2,
  Terminal,
  Cpu,
  Activity
} from 'lucide-react';
import { cn } from '../utils/cn';

const Protocol = () => {
  const [loading, setLoading] = useState(false);
  const [protocol, setProtocol] = useState<any>(null);

  const handleDesign = async () => {
    setLoading(true);
    try {
      const response = await workflowApi.generateProtocol(''); // Uses top hypothesis as default
      setProtocol(response.data);
    } catch (error) {
      console.error("Protocol generation failed", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Phase 4: Experimental Design</h1>
          <p className="text-slate-400 mt-1">Translating hypotheses into rigorous, executable, and statistically powered protocols.</p>
        </div>
        {!protocol && !loading && (
          <button 
            onClick={handleDesign}
            className="px-6 py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-semibold rounded-lg shadow-lg shadow-purple-900/40 transition-all flex items-center gap-2"
          >
            <Beaker className="w-4 h-4" /> Design Formal Protocol
          </button>
        )}
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center h-64 space-y-4">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin"></div>
            <Terminal className="w-6 h-6 text-purple-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
          </div>
          <p className="text-slate-400 animate-pulse">Architecting experimental variables and performing power simulations...</p>
        </div>
      )}

      {protocol && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Protocol Architecture */}
          <div className="space-y-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-purple-400">
                <Settings2 className="w-5 h-5" /> Protocol Architecture
              </h3>
              
              <div className="space-y-6">
                <div>
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Independent Variables (IV)</h4>
                  <div className="grid grid-cols-1 gap-2">
                    {protocol.independent_variables.map((v: any, i: number) => (
                      <div key={i} className="flex items-center gap-3 p-3 bg-slate-800/40 border border-slate-700/50 rounded-lg">
                        <div className="w-1.5 h-1.5 rounded-full bg-purple-500"></div>
                        <span className="text-sm font-medium text-slate-200">{v.name}</span>
                        <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-700 text-slate-400 ml-auto uppercase">{v.role}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-3">Treatment Groups</h4>
                  <div className="flex flex-wrap gap-2">
                    {protocol.experimental_groups.map((group: string, i: number) => (
                      <span key={i} className="px-3 py-1 bg-purple-500/10 border border-purple-500/20 rounded-full text-xs text-purple-400 font-medium">
                        {group}
                      </span>
                    ))}
                    <span className="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full text-xs text-slate-400 font-medium">
                      Control: {protocol.control_group}
                    </span>
                  </div>
                </div>

                <div className="p-4 bg-emerald-500/5 border border-emerald-500/10 rounded-xl flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Activity className="w-5 h-5 text-emerald-400" />
                    <div>
                      <p className="text-xs text-slate-500 font-semibold uppercase">Power Analysis Verdict</p>
                      <p className="text-sm text-slate-200">Required Sample Size: <span className="text-emerald-400 font-mono font-bold">n = {protocol.sample_size}</span></p>
                    </div>
                  </div>
                  <CheckCircle2 className="w-5 h-5 text-emerald-500" />
                </div>
              </div>
            </div>
            
            <div className="glass-card p-6 bg-gradient-to-br from-purple-900/10 to-transparent">
              <h3 className="text-lg font-semibold mb-2 text-slate-200 flex items-center gap-2">
                <Cpu className="w-5 h-5" /> Executable Backend
              </h3>
              <p className="text-sm text-slate-400 mb-6">The AI has generated a self-contained Python validation environment based on this protocol.</p>
              <div className="flex gap-3">
                <button className="flex-1 py-3 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-purple-900/40">
                  <Play className="w-4 h-4 fill-current" /> Execute Simulation
                </button>
                <button className="p-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl border border-slate-700 transition-all">
                  <RotateCcw className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Code Viewer */}
          <div className="glass-card flex flex-col overflow-hidden border-slate-700">
            <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/50">
              <h3 className="text-sm font-semibold flex items-center gap-2 text-slate-400">
                <Code2 className="w-4 h-4" /> validation_script.py
              </h3>
              <div className="flex gap-1.5 text-[8px] uppercase font-bold">
                <span className="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-500 border border-amber-500/20">Review Needed</span>
              </div>
            </div>
            <div className="flex-1 p-0 overflow-hidden font-mono text-[11px] leading-relaxed bg-slate-950">
              <pre className="p-6 custom-scrollbar overflow-auto h-full text-purple-300">
                <code>{protocol.code}</code>
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Protocol;
