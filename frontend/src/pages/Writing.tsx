import React, { useState } from 'react';
import { workflowApi } from '../services/api';
import { 
  FileEdit, 
  FileText, 
  Download, 
  RefreshCw, 
  CheckCircle2, 
  BookMarked,
  Printer,
  ChevronRight,
  Send,
  ScrollText
} from 'lucide-react';
import { cn } from '../utils/cn';

const Writing = () => {
  const [loading, setLoading] = useState(false);
  const [manuscript, setManuscript] = useState<any>(null);
  const [exporting, setExporting] = useState<string | null>(null);

  const handleDraft = async () => {
    setLoading(true);
    try {
      const response = await workflowApi.runWriting();
      setManuscript(response.data);
    } catch (error) {
      console.error("Manuscript generation failed", error);
    } finally {
      setLoading(false);
    }
  };

  const sections = [
    { id: 'abstract', label: 'Abstract' },
    { id: 'introduction', label: 'Introduction' },
    { id: 'methods', label: 'Methods' },
    { id: 'results', label: 'Results' },
    { id: 'discussion', label: 'Discussion' },
    { id: 'conclusion', label: 'Conclusion' }
  ];

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Phase 6: Manuscript Drafting</h1>
          <p className="text-slate-400 mt-1">Synthesizing all research phases into a publication-ready scientific manuscript.</p>
        </div>
        {!manuscript && !loading && (
          <button 
            onClick={handleDraft}
            className="px-6 py-2.5 bg-sky-600 hover:bg-sky-500 text-white font-semibold rounded-lg shadow-lg shadow-sky-900/40 transition-all flex items-center gap-2"
          >
            <ScrollText className="w-4 h-4" /> Drafting Full Manuscript
          </button>
        )}
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center h-64 space-y-4">
          <div className="relative">
            <div className="w-16 h-16 border-4 border-sky-500/20 border-t-sky-500 rounded-full animate-spin"></div>
            <FileText className="w-6 h-6 text-sky-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
          </div>
          <p className="text-slate-400 animate-pulse">AI Agent is writing sections and formatting citations...</p>
        </div>
      )}

      {manuscript && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Navigation & Export */}
          <div className="lg:col-span-1 space-y-6">
            <div className="glass-card p-4">
              <h3 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 px-2">Sections</h3>
              <div className="space-y-1">
                {sections.map(section => (
                  <button 
                    key={section.id}
                    className="w-full flex items-center justify-between px-3 py-2 text-sm text-slate-400 hover:text-white hover:bg-slate-800 rounded-lg transition-all group"
                  >
                    <span>{section.label}</span>
                    <ChevronRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </button>
                ))}
              </div>
            </div>

            <div className="glass-card p-6 space-y-4 bg-gradient-to-br from-emerald-900/10 to-transparent border-emerald-500/20">
              <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                <Download className="w-5 h-5" /> Export Options
              </h3>
              <p className="text-xs text-slate-500 mb-4">Export in high-fidelity formats for journal submission.</p>
              
              <div className="space-y-2">
                <button className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl shadow-lg shadow-emerald-900/30 transition-all flex items-center justify-center gap-2">
                  <FileText className="w-4 h-4" /> Download DOCX
                </button>
                <button className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-200 font-bold rounded-xl border border-slate-700 transition-all flex items-center justify-center gap-2">
                  <Printer className="w-4 h-4" /> Download LaTeX
                </button>
              </div>
            </div>

            <button className="w-full group py-4 bg-sky-600/10 hover:bg-sky-600/20 text-sky-400 font-bold rounded-xl border border-sky-600/20 transition-all flex items-center justify-center gap-3">
              <Send className="w-4 h-4 group-hover:translate-x-1 group-hover:-translate-y-1 transition-transform" />
              Submit to ReviewAgent
            </button>
          </div>

          {/* Manuscript Viewer */}
          <div className="lg:col-span-3 space-y-8 pb-20">
            <div className="glass-card p-10 bg-slate-100 text-slate-900 font-serif leading-relaxed shadow-white/5">
              <h1 className="text-3xl font-bold mb-8 text-center border-b border-slate-200 pb-8">{manuscript.title}</h1>
              
              {sections.map(section => (
                <div key={section.id} className="mb-8">
                  <h2 className="text-xl font-bold uppercase tracking-wide mb-4 text-slate-800">{section.label}</h2>
                  <div className="text-slate-700 space-y-4 text-justify">
                    {manuscript.sections[section.id]?.content || `Drafting ${section.label}...`}
                  </div>
                </div>
              ))}

              <div className="mt-16 pt-8 border-t border-slate-200">
                <h2 className="text-xl font-bold uppercase tracking-wide mb-6 text-slate-800 flex items-center gap-2">
                  <BookMarked className="w-5 h-5" /> References
                </h2>
                <div className="space-y-3 text-xs text-slate-600">
                  {manuscript.references?.map((ref: any, i: number) => (
                    <div key={i} className="flex gap-4">
                      <span className="font-bold min-w-[24px]">[{i+1}]</span>
                      <p>{ref.authors} ({ref.year}). {ref.title}. {ref.journal || 'Journal of Scientific Research'}.</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex items-center justify-center gap-2 text-slate-500 text-xs">
              <CheckCircle2 className="w-4 h-4 text-emerald-500" />
              Manuscript compiled with AI Co-Scientist v3.0 logic
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Writing;
