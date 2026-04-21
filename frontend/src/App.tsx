import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import MainLayout from './components/MainLayout';
import Dashboard from './pages/Dashboard';
import Scoping from './pages/Scoping';
import Literature from './pages/Literature';
import Hypotheses from './pages/Hypotheses';
import Protocol from './pages/Protocol';
import Analysis from './pages/Analysis';
import Writing from './pages/Writing';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="scoping" element={<Scoping />} />
          <Route path="literature" element={<Literature />} />
          <Route path="hypotheses" element={<Hypotheses />} />
          <Route path="protocol" element={<Protocol />} />
          <Route path="analysis" element={<Analysis />} />
          <Route path="writing" element={<Writing />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
}

export default App;
