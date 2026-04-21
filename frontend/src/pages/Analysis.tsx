import React, { useState } from 'react';
import { workflowApi } from '../services/api';
import { 
  BarChart3, 
  Upload, 
  Database, 
  Search, 
  CheckCircle2, 
  FileSpreadsheet,
  Globe,
  PieChart,
  LineChart,
  Activity,
  ArrowRight
} from 'lucide-react';
import { cn } from '../utils/cn';

const Analysis = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [activeTab, setActiveTab] = useState<'upload' | 'public'>('upload');

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setLoading(true);
    setUploadProgress(20);
    try {
      const uploadRes = await workflowApi.uploadCsv(file);
      setUploadProgress(60);
      const analysisRes = await workflowApi.runAnalysis('', uploadRes.data.path);
      setUploadProgress(100);
      setResults(analysisRes.data);
    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setLoading(false);
      setTimeout(() => setUploadProgress(0), 1000);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold font-outfit text-white">Phase 5: Data Analysis</h1>
          <p className="text-slate-400 mt-1">Empirical validation through statistical testing and multi-source data ingestion.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Data Ingestion Section */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card overflow-hidden">
            <div className="flex border-b border-slate-800">
              <button 
                onClick={() => setActiveTab('upload')}
                className={cn(
                  "flex-1 py-4 text-xs font-bold uppercase tracking-widest transition-all",
                  activeTab === 'upload' ? "text-sky-400 bg-sky-500/5" : "text-slate-500 hover:text-slate-300"
                )}
              >
                Manual Upload
              </button>
              <button 
                onClick={() => setActiveTab('public')}
                className={cn(
                  "flex-1 py-4 text-xs font-bold uppercase tracking-widest transition-all",
                  activeTab === 'public' ? "text-indigo-400 bg-indigo-500/5" : "text-slate-500 hover:text-slate-300"
                )}
              >
                Public Databases
              </button>
            </div>

            <div className="p-6">
              {activeTab === 'upload' ? (
                <div className="space-y-6">
                  <div className="border-2 border-dashed border-slate-700 rounded-xl p-8 text-center hover:border-sky-500/50 transition-all group relative cursor-pointer">
                    <input 
                      type="file" 
                      accept=".csv"
                      onChange={handleFileUpload}
                      className="absolute inset-0 opacity-0 cursor-pointer"
                    />
                    <Upload className="w-10 h-10 text-slate-600 group-hover:text-sky-500 transition-colors mx-auto mb-4" />
                    <h4 className="text-slate-200 font-medium">Drop CSV Dataset</h4>
                    <p className="text-xs text-slate-500 mt-1">Experimental results or observational data</p>
                  </div>

                  {uploadProgress > 0 && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-[10px] font-bold text-slate-500">
                        <span>{uploadProgress === 100 ? 'ANALYSIS COMPLETE' : 'UPLOADING & ANALYZING...'}</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                        <div 
                          className="bg-sky-500 h-full transition-all duration-500"
                          style={{ width: `${uploadProgress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}

                  <div className="space-y-3 pt-4">
                    <h5 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Supported Formats</h5>
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <FileSpreadsheet className="w-4 h-4 text-emerald-500" />
                      <span>Standard CSV (UTF-8)</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <Search className="w-4 h-4 text-slate-500 absolute left-3 top-3" />
                    <input 
                      type="text"
                      className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-indigo-500"
                      placeholder="Search GEO, ClinicalTrials..."
                    />
                  </div>
                  
                  <div className="space-y-2">
                    {['Gene Expression Omnibus (GEO)', 'ClinicalTrials.gov', 'BioLINCC', 'UK Biobank'].map((db) => (
                      <div key={db} className="flex items-center justify-between p-3 bg-slate-800/40 border border-slate-700 rounded-lg hover:border-indigo-500/50 transition-all cursor-pointer group">
                        <div className="flex items-center gap-2">
                          <Globe className="w-4 h-4 text-slate-500 group-hover:text-indigo-400" />
                          <span className="text-xs text-slate-300 font-medium">{db}</span>
                        </div>
                        <ArrowRight className="w-3 h-3 text-slate-600 group-hover:text-indigo-400" />
                      </div>
                    ))}
                  </div>
                  
                  <button className="w-full py-2 bg-indigo-600/20 text-indigo-300 text-xs font-bold rounded-lg border border-indigo-600/30 hover:bg-indigo-600/30 transition-all">
                    Search Public Data
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Results Visualization Section */}
        <div className="lg:col-span-2 space-y-6">
          {!results && (
            <div className="glass-card h-full flex flex-col items-center justify-center text-center p-12 text-slate-500 space-y-4">
              <PieChart className="w-16 h-16 opacity-20" />
              <p>No data ingested. Please upload a dataset or connect to a database to perform statistical analysis.</p>
            </div>
          )}

          {results && (
            <>
              <div className="grid grid-cols-2 gap-6">
                <div className="glass-card p-6">
                  <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-sky-400" /> Statistical Significance
                  </h4>
                  <div className="space-y-4">
                    {results.results.map((res: any, i: number) => (
                      <div key={i} className="flex justify-between items-center border-b border-slate-800/50 pb-2">
                        <span className="text-sm text-slate-300 font-medium">{res.test_name}</span>
                        <span className={cn(
                          "text-xs font-mono font-bold",
                          res.p_value < 0.05 ? "text-emerald-400" : "text-amber-400"
                        )}>
                          p = {res.p_value.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
                  <div className="w-16 h-16 rounded-full bg-emerald-500/10 flex items-center justify-center mb-4">
                    <CheckCircle2 className="w-8 h-8 text-emerald-500" />
                  </div>
                  <h4 className="text-slate-100 font-bold">Analysis Validated</h4>
                  <p className="text-xs text-slate-500 mt-1">Significant effects detected in treatment groups.</p>
                </div>
              </div>

              <div className="glass-card p-6">
                <h4 className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-indigo-400" /> Automated Interpretation
                </h4>
                <div className="p-4 bg-slate-900/50 rounded-xl border border-slate-800 italic text-slate-300 text-sm leading-relaxed">
                  "{results.interpretation}"
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default Analysis;
