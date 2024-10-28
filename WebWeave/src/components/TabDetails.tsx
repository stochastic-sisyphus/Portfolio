import React, { memo } from 'react';
import { Tab } from '../types';

interface TabDetailsProps {
  tab: Tab;
}

export const TabDetails = memo(function TabDetails({ tab }: TabDetailsProps) {
  return (
    <div className="mt-3 space-y-2">
      {tab.summary && (
        <div>
          <h4 className="text-sm font-medium text-gray-500">Summary:</h4>
          <p className="text-sm text-gray-600">{tab.summary}</p>
        </div>
      )}
      
      {tab.keyPoints && tab.keyPoints.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-500">Key Insights:</h4>
          <ul className="list-disc list-inside text-sm text-gray-600">
            {tab.keyPoints.map((point, i) => (
              <li key={i}>{point}</li>
            ))}
          </ul>
        </div>
      )}
      
      {tab.keywords && tab.keywords.length > 0 && (
        <div>
          <h4 className="text-sm font-medium text-gray-500">Keywords:</h4>
          <div className="flex flex-wrap gap-1">
            {tab.keywords.map((keyword, i) => (
              <span
                key={i}
                className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800"
              >
                {keyword}
              </span>
            ))}
          </div>
        </div>
      )}
      
      {tab.readingTime && (
        <div className="text-sm text-gray-500">
          Reading time: ~{tab.readingTime} min
        </div>
      )}
    </div>
  );
});