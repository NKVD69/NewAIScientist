import React, { useState } from 'react';
import { workflowApi } from '../services/api';
import { 
  Lightbulb, 
  Trophy, 
  Dna, 
  Star, 
  TrendingUp,
  RefreshCw,
  MoreHorizontal,
  Plus,
  AlertTriangle,
  GitBranch,
  ArrowRight
} from 'lucide-react';
import { cn } from '../utils/cn';

const Hypotheses = () => {
  const [loading, setLoading] = useState(false);
  const [hypotheses, setHypotheses] = useState<any[]>([]);

  const handleGenerate = async () => {
    setLoading(true);
    try {
      const response = await workflowApi.runHypotheses(5);
      setHypotheses(response.data);
    } catch (error) {
      console.error("Hypothesis generation failed", error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch(status.toLowerCase()) {
      case 'ranked': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
      case 'generated': return 'text-sky-400 bg-sky-500/10 border-sky-500/20';
      case 'evolved': return 'text-purple-400 bg-purple-500/10 border-purple-500/20';
      default: return 'text-slate-400 bg-slate-800 border-slate-700';
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Phase 3: Hypotheses</h1>
          <p className="text-slate-400 mt-1">Multi-step ideation with adversarial critique and causal chaining.</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleGenerate}
            disabled={loading}
            className="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-semibold rounded-lg shadow-lg shadow-indigo-900/40 transition-all flex items-center gap-2"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
            {loading ? 'Generating...' : 'Start Ideation Cycle'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Hypotheses List */}
        <div className="lg:col-span-2 space-y-4">
          {hypotheses.length === 0 && !loading && (
            <div className="glass-card p-12 border-dashed border-slate-700 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mb-4">
                <Lightbulb className="w-8 h-8 text-slate-600" />
              </div>
              <h3 className="text-lg font-medium text-slate-300">Awaiting Hypotheses</h3>
              <p className="text-slate-500 text-sm max-w-xs mt-1">Launch the generation cycle to explore potential scientific breakthroughs.</p>
            </div>
          )}

          {hypotheses.map((hyp) => (
            <div key={hyp.id} className="glass-card p-6 hover:bg-slate-800/40 transition-all border-r-4 border-r-indigo-500/30 group relative overflow-hidden">
              <div className="flex justify-between items-start mb-4">
                <div className="flex flex-col gap-1">
                  <div className="flex items-center gap-2">
                    <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border", getStatusColor(hyp.status))}>
                      {hyp.status}
                    </span>
                    {hyp.parent_id && (
                      <span className="flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-500/10 text-amber-500 border border-amber-500/20">
                        <GitBranch className="w-3 h-3" /> {hyp.link_type} H-{hyp.parent_id.slice(-4)}
                      </span>
                    )}
                  </div>
                </div>
                <button className="text-slate-500 hover:text-slate-300 transition-colors">
                  <MoreHorizontal className="w-5 h-5" />
                </button>
              </div>

              <h3 className="text-xl font-bold text-white mb-2 group-hover:text-indigo-400 transition-colors tracking-tight">
                {hyp.title}
              </h3>
              <p className="text-sm text-slate-400 leading-relaxed line-clamp-2 mb-6">
                {hyp.description}
              </p>

              {/* Adversarial Review Badge */}
              {hyp.adversarial_review && (
                <div className="mb-6 p-3 bg-red-500/5 border border-red-500/20 rounded-lg flex items-start gap-3">
                  <AlertTriangle className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="text-[10px] font-bold text-red-400 uppercase tracking-widest">Adversarial Critique</p>
                    <p className="text-xs text-slate-400 mt-1 italic">"{hyp.adversarial_review.verdict}"</p>
                  </div>
                </div>
              )}

              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2 bg-slate-800/50 px-3 py-1.5 rounded-lg border border-slate-700/50">
                  <Trophy className="w-3.5 h-3.5 text-amber-400" />
                  <span className="text-xs font-bold text-slate-200">Elo: {Math.round(hyp.elo_rating)}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <Star className="w-3.5 h-3.5 text-indigo-400" />
                  <span>Novelty: <span className="text-slate-300 capitalize">{hyp.novelty_level}</span></span>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-500">
                  <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
                  <span>{hyp.reviews?.length || 0} Peer Reviews</span>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Chaining Visualization Mockup */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6 bg-slate-900/40 border-sky-500/20">
            <h3 className="text-lg font-semibold mb-6 flex items-center gap-2 text-sky-400">
              <GitBranch className="w-5 h-5" /> Hypothesis Chaining
            </h3>
            <div className="space-y-4">
              <div className="relative pl-8 pb-4">
                <div className="absolute left-3 top-0 bottom-0 w-px bg-slate-700"></div>
                <div className="absolute left-1 top-2 w-4 h-4 rounded-full bg-sky-500 border-4 border-slate-900 shadow-sky-500/20 shadow-lg"></div>
                <p className="text-xs text-sky-400 font-bold uppercase mb-1">Premise (H-A1)</p>
                <p className="text-xs text-slate-300 font-medium">Initial Mechanistic Theory</p>
              </div>
              <div className="relative pl-8">
                <div className="absolute left-1 top-2 w-4 h-4 rounded-full bg-indigo-500 border-4 border-slate-900 shadow-indigo-500/20 shadow-lg"></div>
                <p className="text-xs text-indigo-400 font-bold uppercase mb-1">Refinement (H-B2)</p>
                <p className="text-xs text-slate-300 font-medium">Deep Dive on Sub-Pathway X</p>
              </div>
            </div>
            <button className="w-full mt-8 py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-bold rounded-xl border border-slate-700 transition-all">
              Visualize Logic Graphes
            </button>
          </div>

          <div className="glass-card p-6 border-red-500/10">
            <h3 className="text-lg font-semibold mb-4 text-red-400 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" /> Adversarial Stress-Tests
            </h3>
            <p className="text-xs text-slate-500 leading-relaxed mb-6">
              The DevilsAdvocate agent identifies logical fallacies and alternative explanations to improve theory robustness.
            </p>
            <div className="space-y-3">
              <div className="flex justify-between text-[10px] font-bold text-slate-500">
                <span>Vulnerability Map</span>
                <span>42% Exposure</span>
              </div>
              <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                <div className="bg-red-500 h-full w-[42%]"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Hypotheses;
