import React, { useState } from 'react';
import { workflowApi } from '../services/api';
import { Play, Clipboard, Rocket, CheckCircle2 } from 'lucide-react';

const Dashboard = () => {
  const [goal, setGoal] = useState({
    title: '',
    description: '',
    domain: 'Biomedicine'
  });
  const [loading, setLoading] = useState(false);
  const [initialized, setInitialized] = useState(false);

  const handleInitialize = async () => {
    setLoading(true);
    try {
      await workflowApi.initializeGoal(goal);
      setInitialized(true);
    } catch (error) {
      console.error("Init failed", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Research Dashboard</h1>
          <p className="text-slate-400 mt-1">Configure your scientific mission and track progress.</p>
        </div>
        <div className="flex gap-3">
          <div className="glass-card px-4 py-2 flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
            <span className="text-sm font-medium">Session: Alpha-01</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Configuration */}
        <div className="lg:col-span-2 space-y-6">
          <div className="glass-card p-8">
            <h3 className="text-xl font-semibold mb-6 flex items-center gap-2 text-sky-400">
              <Clipboard className="w-5 h-5" /> Mission Parameters
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Research Title</label>
                <input 
                  type="text" 
                  value={goal.title}
                  onChange={(e) => setGoal({...goal, title: e.target.value})}
                  className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-sky-500 focus:outline-none transition-all"
                  placeholder="e.g., Inhibition of TLR4 Pathway in Neuroinflammation"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-1">Description & Scope</label>
                <textarea 
                  rows={4}
                  value={goal.description}
                  onChange={(e) => setGoal({...goal, description: e.target.value})}
                  className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-sky-500 focus:outline-none transition-all"
                  placeholder="Detail the scientific gap you want to explore..."
                ></textarea>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-1">Scientific Domain</label>
                  <select 
                    value={goal.domain}
                    onChange={(e) => setGoal({...goal, domain: e.target.value})}
                    className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-sky-500 focus:outline-none transition-all"
                  >
                    <option>Biomedicine</option>
                    <option>Physics</option>
                    <option>Computer Science</option>
                    <option>Chemistry</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-1">Primary Engine</label>
                  <div className="w-full bg-slate-800/50 border border-slate-700 rounded-lg px-4 py-3 text-slate-300">
                    GPT-4o (Standard)
                  </div>
                </div>
              </div>

              <div className="pt-4">
                <button 
                  onClick={handleInitialize}
                  disabled={loading || !goal.title}
                  className="w-full py-4 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-bold rounded-xl shadow-lg shadow-sky-900/40 transition-all flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <span className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></span>
                  ) : initialized ? (
                    <><CheckCircle2 className="w-5 h-5" /> Goal Initialized</>
                  ) : (
                    <><Rocket className="w-5 h-5" /> Launch Scientific Mission</>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Progress & Stats */}
        <div className="space-y-6">
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold mb-4 text-slate-200">System Performance</h3>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Total Tokens</span>
                <span className="text-slate-200 font-mono">145k</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5">
                <div className="bg-sky-500 h-1.5 rounded-full w-3/4"></div>
              </div>
              <div className="flex justify-between items-center text-sm">
                <span className="text-slate-400">Research Confidence</span>
                <span className="text-slate-200 font-mono">87%</span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5">
                <div className="bg-emerald-500 h-1.5 rounded-full w-4/5"></div>
              </div>
            </div>
          </div>

          <div className="glass-card p-6 bg-gradient-to-br from-indigo-900/20 to-sky-900/20">
            <h3 className="text-lg font-semibold mb-4 text-slate-200">Auto-Pilot Mode</h3>
            <p className="text-sm text-slate-400 mb-6">Let the AI coordinate all phases autonomously through to the final manuscript.</p>
            <button className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-white font-semibold rounded-lg border border-slate-700 transition-all flex items-center justify-center gap-2">
              <Play className="w-4 h-4 fill-current" /> Start Full Cycle
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
