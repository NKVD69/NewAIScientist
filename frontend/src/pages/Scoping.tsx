import React, { useState, useEffect } from 'react';
import { workflowApi } from '../services/api';
import { 
  Search, 
  Target, 
  Layers, 
  ArrowRight, 
  Activity,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { cn } from '../utils/cn';

const Scoping = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<any>(null);
  const [selectedQuestions, setSelectedQuestions] = useState<string[]>([]);

  const handleRunScoping = async () => {
    setLoading(true);
    try {
      const response = await workflowApi.runScoping();
      setData(response.data);
    } catch (error) {
      console.error("Scoping failed", error);
    } finally {
      setLoading(false);
    }
  };

  const toggleQuestion = (id: string) => {
    setSelectedQuestions(prev => 
      prev.includes(id) ? prev.filter(q => q !== id) : [...prev, id]
    );
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Phase 1: Research Scoping</h1>
          <p className="text-slate-400 mt-1">Analyzing state-of-the-art and formulating research questions.</p>
        </div>
        {!data && !loading && (
          <button 
            onClick={handleRunScoping}
            className="px-6 py-2.5 bg-sky-600 hover:bg-sky-500 text-white font-semibold rounded-lg shadow-lg shadow-sky-900/40 transition-all flex items-center gap-2"
          >
            <Search className="w-4 h-4" /> Start Analysis
          </button>
        )}
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center h-64 space-y-4">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin"></div>
            <Activity className="w-6 h-6 text-sky-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
          </div>
          <p className="text-slate-400 animate-pulse">AI Agent is synthesizing literature and identifying gaps...</p>
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* State of the Art Report */}
          <div className="glass-card p-6 flex flex-col">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-sky-400">
              <Layers className="w-5 h-5" /> State of the Art Synthesis
            </h3>
            <div className="flex-1 bg-slate-800/30 rounded-lg p-4 custom-scrollbar overflow-y-auto max-h-[500px] text-slate-300 text-sm leading-relaxed space-y-4">
              <div className="prose prose-invert prose-sm">
                <p className="font-semibold text-slate-100">Known Facts:</p>
                <ul className="list-disc pl-4 space-y-1">
                  {data.soa.known_facts.map((fact: string, i: number) => <li key={i}>{fact}</li>)}
                </ul>
                
                <p className="font-semibold text-slate-100 mt-4">Identified Gaps:</p>
                <div className="space-y-3">
                  {data.soa.gaps.map((gap: any, i: number) => (
                    <div key={i} className="p-3 bg-amber-500/5 border border-amber-500/20 rounded-lg">
                      <p className="text-amber-200 font-medium">{gap.title}</p>
                      <p className="text-slate-400 text-xs mt-1">{gap.description}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Research Questions */}
          <div className="glass-card p-6 flex flex-col">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2 text-indigo-400">
              <Target className="w-5 h-5" /> Formulated Research Questions
            </h3>
            <div className="flex-1 space-y-4 custom-scrollbar overflow-y-auto max-h-[500px]">
              {data.questions.map((q: any) => (
                <div 
                  key={q.id}
                  onClick={() => toggleQuestion(q.id)}
                  className={cn(
                    "p-4 rounded-xl border transition-all cursor-pointer group",
                    selectedQuestions.includes(q.id)
                      ? "bg-sky-500/10 border-sky-500/50"
                      : "bg-slate-800/40 border-slate-700/50 hover:border-slate-600"
                  )}
                >
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-xs font-bold px-2 py-0.5 rounded bg-slate-700 text-slate-300 uppercase letter-spacing-wide">
                      {q.type}
                    </span>
                    {selectedQuestions.includes(q.id) && <CheckCircle2 className="w-4 h-4 text-sky-500" />}
                  </div>
                  <p className="text-slate-200 font-medium leading-tight group-hover:text-white transition-colors">{q.question}</p>
                  
                  <div className="mt-4 flex gap-4 text-[10px] font-mono uppercase tracking-wider text-slate-500">
                    <div className="flex flex-col">
                      <span>Novelty</span>
                      <span className="text-emerald-400">{(q.novelty_score * 10).toFixed(1)}/10</span>
                    </div>
                    <div className="flex flex-col">
                      <span>Impact</span>
                      <span className="text-sky-400">{(q.impact_score * 10).toFixed(1)}/10</span>
                    </div>
                    <div className="flex flex-col">
                      <span>Feasibility</span>
                      <span className="text-purple-400">{(q.feasibility_score * 10).toFixed(1)}/10</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800">
              <button 
                disabled={selectedQuestions.length === 0}
                className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 text-white font-bold rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-900/30"
              >
                Proceed to Literature Review <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      {data && (
        <div className="glass-card p-6">
          <h3 className="text-lg font-semibold mb-6 flex items-center gap-2 text-emerald-400">
            <Activity className="w-5 h-5" /> Conceptual Framework (causal-DAG)
          </h3>
          <div className="bg-slate-900/60 rounded-xl p-8 border border-slate-800 flex items-center justify-center min-h-[200px]">
            <div className="text-center space-y-4">
              <div className="flex gap-4 items-center justify-center">
                {data.framework.variables.map((v: any, i: number) => (
                  <React.Fragment key={v.id}>
                    <div className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg shadow-xl">
                      <span className="text-xs text-slate-500 block text-left uppercase mb-1 font-bold">{v.role}</span>
                      <span className="text-slate-200 font-medium">{v.name}</span>
                    </div>
                    {i < data.framework.variables.length - 1 && <ArrowRight className="text-slate-700" />}
                  </React.Fragment>
                ))}
              </div>
              <p className="text-xs text-slate-500 mt-6 italic">Visualizing hypothesized mechanistic pathways identified in literature.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Scoping;
