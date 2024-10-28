import React, { memo, useState } from 'react';
import { Tab } from '../types';
import { SimilarTabs } from './SimilarTabs';
import { useTabStore } from '../store/tabStore';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { TabDetails } from './TabDetails';

interface TabContentProps {
  tab: Tab;
  onRemove: (id: string) => void;
}

export const TabContent = memo(function TabContent({ tab, onRemove }: TabContentProps) {
  const [showDetails, setShowDetails] = useState(false);
  const [showSimilar, setShowSimilar] = useState(false);
  const getSimilarTabs = useTabStore(state => state.getSimilarTabs);

  const similarTabs = showSimilar ? getSimilarTabs(tab.id) : [];

  return (
    <>
      <div className="flex items-center space-x-3">
        <img src={tab.favicon} alt="" className="w-6 h-6" />
        <div className="flex-1">
          <h3 className="font-medium text-gray-900 truncate">{tab.title}</h3>
          <p className="text-sm text-gray-500 truncate">{tab.url}</p>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-gray-400 hover:text-gray-600"
        >
          {showDetails ? (
            <ChevronUpIcon className="h-5 w-5" />
          ) : (
            <ChevronDownIcon className="h-5 w-5" />
          )}
        </button>
        <button
          onClick={() => setShowSimilar(!showSimilar)}
          className="text-gray-400 hover:text-gray-600 ml-2"
        >
          <span className="sr-only">Show similar</span>
          ≡
        </button>
        <button
          onClick={() => onRemove(tab.id)}
          className="text-gray-400 hover:text-gray-600 ml-2"
        >
          <span className="sr-only">Close</span>
          ×
        </button>
      </div>
      
      {showDetails && <TabDetails tab={tab} />}
      {showSimilar && <SimilarTabs tabs={similarTabs} />}
    </>
  );
});