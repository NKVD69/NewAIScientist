import React, { useState } from 'react';
import { workflowApi } from '../services/api';
import { 
  Search, 
  BookOpen, 
  ExternalLink, 
  Calendar, 
  User, 
  Database,
  Filter,
  CheckCircle2,
  RefreshCw,
  MoreVertical
} from 'lucide-react';
import { cn } from '../utils/cn';

const Literature = () => {
  const [loading, setLoading] = useState(false);
  const [papers, setPapers] = useState<any[]>([]);
  const [searchParams, setSearchParams] = useState({
    maxResults: 5,
    sources: ['arxiv', 'pubmed']
  });

  const handleSearch = async () => {
    setLoading(true);
    try {
      const response = await workflowApi.runLiterature(
        searchParams.maxResults, 
        searchParams.sources
      );
      setPapers(response.data);
    } catch (error) {
      console.error("Literature search failed", error);
    } finally {
      setLoading(false);
    }
  };

  const toggleSource = (source: string) => {
    setSearchParams(prev => ({
      ...prev,
      sources: prev.sources.includes(source) 
        ? prev.sources.filter(s => s !== source)
        : [...prev.sources, source]
    }));
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Phase 2: Literature Review</h1>
          <p className="text-slate-400 mt-1">Discovering and analyzing foundational papers from global repositories.</p>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold rounded-lg shadow-lg shadow-emerald-900/40 transition-all flex items-center gap-2"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            {loading ? 'Searching...' : 'Search Repositories'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Filters & Sources Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-6">
            <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-4 flex items-center gap-2">
              <Filter className="w-4 h-4" /> Search Sources
            </h3>
            <div className="space-y-2">
              {['arxiv', 'pubmed', 'clinicaltrials', 'biorxiv'].map((source) => (
                <button
                  key={source}
                  onClick={() => toggleSource(source)}
                  className={cn(
                    "w-full flex items-center justify-between px-4 py-2.5 rounded-lg border transition-all text-sm",
                    searchParams.sources.includes(source)
                      ? "bg-emerald-500/10 border-emerald-500/50 text-emerald-400"
                      : "bg-slate-800/40 border-slate-700 text-slate-400 hover:border-slate-600"
                  )}
                >
                  <span className="capitalize">{source}</span>
                  {searchParams.sources.includes(source) && <CheckCircle2 className="w-4 h-4" />}
                </button>
              ))}
            </div>

            <div className="mt-8">
              <h3 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-4">Max Results</h3>
              <input 
                type="range" min="1" max="50" step="1"
                value={searchParams.maxResults}
                onChange={(e) => setSearchParams({...searchParams, maxResults: parseInt(e.target.value)})}
                className="w-full accent-emerald-500" 
              />
              <div className="flex justify-between text-xs text-slate-500 mt-2 font-mono">
                <span>1</span>
                <span>{searchParams.maxResults} items</span>
                <span>50</span>
              </div>
            </div>
          </div>
        </div>

        {/* Paper Results Area */}
        <div className="lg:col-span-3 space-y-4">
          {papers.length === 0 && !loading && (
            <div className="glass-card p-12 border-dashed border-slate-700 flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 bg-slate-800/50 rounded-full flex items-center justify-center mb-4">
                <BookOpen className="w-8 h-8 text-slate-600" />
              </div>
              <h3 className="text-lg font-medium text-slate-300">No papers retrieved yet</h3>
              <p className="text-slate-500 text-sm max-w-xs mt-1">Configure your sources and click search to begin literature discovery.</p>
            </div>
          )}

          {loading && (
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="glass-card p-6 animate-pulse">
                  <div className="h-4 bg-slate-800 rounded w-3/4 mb-4"></div>
                  <div className="h-3 bg-slate-800 rounded w-1/2 mb-2"></div>
                  <div className="h-3 bg-slate-800 rounded w-full"></div>
                </div>
              ))}
            </div>
          )}

          {papers.map((paper, idx) => (
            <div key={idx} className="glass-card p-6 hover:bg-slate-800/40 transition-all border-l-4 border-l-emerald-500/30 group">
              <div className="flex justify-between gap-4">
                <div className="space-y-2">
                  <div className="flex items-center gap-3 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                    <span className="flex items-center gap-1 text-emerald-400">
                      <Database className="w-3 h-3" /> {paper.source || 'Repository'}
                    </span>
                    <span>•</span>
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3 h-3" /> {paper.published || 'N/A'}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-slate-100 leading-snug group-hover:text-emerald-400 transition-colors">
                    {paper.title}
                  </h3>
                  <p className="text-sm text-slate-400 line-clamp-2 leading-relaxed">
                    {paper.summary}
                  </p>
                  <div className="flex items-center gap-4 pt-2">
                    <div className="flex items-center gap-1.5 text-xs text-slate-500">
                      <User className="w-3.5 h-3.5" />
                      {paper.authors ? paper.authors.join(', ') : 'Unknown Authors'}
                    </div>
                  </div>
                </div>
                <div className="flex flex-col gap-2">
                  <a 
                    href={paper.url} 
                    target="_blank" 
                    rel="noreferrer"
                    className="p-2 bg-slate-800 hover:bg-slate-700 rounded-lg text-slate-400 hover:text-white transition-all"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                  <button className="p-2 hover:bg-slate-800 rounded-lg text-slate-500">
                    <MoreVertical className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Literature;
