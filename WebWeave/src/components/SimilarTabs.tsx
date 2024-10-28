import React from 'react';
import { Tab } from '../types';

interface SimilarTabsProps {
  tabs: Tab[];
}

export const SimilarTabs: React.FC<SimilarTabsProps> = ({ tabs }) => {
  if (tabs.length === 0) return null;

  return (
    <div className="mt-2 space-y-2">
      <h4 className="text-sm font-medium text-gray-500">Similar Tabs:</h4>
      <div className="space-y-1">
        {tabs.map((tab) => (
          <div key={tab.id} className="text-sm text-gray-600 flex items-center space-x-2">
            <img src={tab.favicon} alt="" className="w-4 h-4" />
            <span className="truncate">{tab.title}</span>
          </div>
        ))}
      </div>
    </div>
  );
};