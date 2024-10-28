import React from 'react';
import { TabManager } from './components/TabManager';
import { SearchBar } from './components/SearchBar';
import { UrlInput } from './components/UrlInput';
import { TabVisualizations } from './components/TabVisualizations';
import { ExportButton } from './components/ExportButton';
import { EnvWarning } from './components/EnvWarning';

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <h1 className="text-3xl font-bold text-gray-900">WebWeave</h1>
            <div className="flex items-center space-x-4">
              <SearchBar />
              <ExportButton />
            </div>
          </div>
        </div>
      </header>
      <main>
        <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
          <EnvWarning />
          <div className="mb-6 flex justify-center">
            <UrlInput />
          </div>
          <TabVisualizations />
          <TabManager />
        </div>
      </main>
    </div>
  );
}

export default App;