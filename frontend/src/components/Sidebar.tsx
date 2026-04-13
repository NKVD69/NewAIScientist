import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  BarChart3, 
  Beaker, 
  BookOpen, 
  FileEdit, 
  Home, 
  Lightbulb, 
  Search, 
  ChevronLeft,
  ChevronRight,
  Atom
} from 'lucide-react';
import { cn } from '../utils/cn';

const SidebarItem = ({ to, icon: Icon, label, collapsed }) => (
  <NavLink
    to={to}
    className={({ isActive }) => cn(
      "flex items-center gap-3 px-4 py-3 my-1 transition-all duration-300 rounded-lg group",
      isActive 
        ? "sidebar-item-active text-sky-400" 
        : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/50"
    )}
  >
    <Icon className={cn("w-5 h-5", collapsed ? "mx-auto" : "")} />
    {!collapsed && <span className="font-medium">{label}</span>}
  </NavLink>
);

const Sidebar = ({ isOpen, setIsOpen }) => {
  const steps = [
    { to: "/", icon: Home, label: "Dashboard" },
    { to: "/scoping", icon: Search, label: "1. Cadrage" },
    { to: "/literature", icon: BookOpen, label: "2. Littérature" },
    { to: "/hypotheses", icon: Lightbulb, label: "3. Hypothèses" },
    { to: "/protocol", icon: Beaker, label: "4. Design Exp." },
    { to: "/analysis", icon: BarChart3, label: "5. Analyse" },
    { to: "/writing", icon: FileEdit, label: "6. Rédaction" },
  ];

  return (
    <aside className={cn(
      "flex flex-col h-screen bg-slate-900 border-r border-slate-800 transition-all duration-300 ease-in-out z-20",
      isOpen ? "w-64" : "w-20"
    )}>
      <div className="flex items-center justify-between p-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-sky-500 rounded-lg">
            <Atom className="w-6 h-6 text-white" />
          </div>
          {isOpen && (
            <span className="text-xl font-bold font-outfit tracking-tight bg-gradient-to-r from-sky-400 to-indigo-400 bg-clip-text text-transparent">
              Scientist v3
            </span>
          )}
        </div>
      </div>

      <nav className="flex-1 px-3 mt-4 overflow-y-auto custom-scrollbar">
        {steps.map((step) => (
          <SidebarItem 
            key={step.to} 
            {...step} 
            collapsed={!isOpen} 
          />
        ))}
      </nav>

      <div className="p-4 border-t border-slate-800">
        <button 
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center justify-center w-full p-2 text-slate-400 rounded-lg hover:bg-slate-800 transition-colors"
        >
          {isOpen ? <ChevronLeft /> : <ChevronRight />}
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
