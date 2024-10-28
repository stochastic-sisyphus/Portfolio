import React, { useState } from 'react';
import { Draggable, DraggableProvided, DraggableStateSnapshot } from 'react-beautiful-dnd';
import { Tab } from '../types';
import { SimilarTabs } from './SimilarTabs';
import { useTabStore } from '../store/tabStore';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface TabCardProps {
  tab: Tab;
  index: number;
  onRemove: (id: string) => void;
}

export const TabCard: React.FC<TabCardProps> = ({ tab, index, onRemove }) => {
  const [showDetails, setShowDetails] = useState(false);
  const [showSimilar, setShowSimilar] = useState(false);
  const getSimilarTabs = useTabStore(state => state.getSimilarTabs);

  const similarTabs = showSimilar ? getSimilarTabs(tab.id) : [];

  return (
    <Draggable draggableId={tab.id} index={index}>
      {(provided: DraggableProvided, snapshot: DraggableStateSnapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          className={`bg-white rounded-lg shadow-sm p-4 mb-2 transition-shadow ${
            snapshot.isDragging ? 'shadow-lg' : 'hover:shadow-md'
          }`}
        >
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
          
          {showDetails && (
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
          )}
          
          {showSimilar && <SimilarTabs tabs={similarTabs} />}
        </div>
      )}
    </Draggable>
  );
};