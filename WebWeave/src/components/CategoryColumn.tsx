import React from 'react';
import { useDroppable } from '@dnd-kit/core';
import { DraggableTab } from './DraggableTab';
import { Tab, Category } from '../types';
import { useTabStore } from '../store/tabStore';

interface CategoryColumnProps {
  category: Category;
  tabs: Tab[];
}

export function CategoryColumn({ category, tabs }: CategoryColumnProps) {
  const removeTab = useTabStore((state) => state.removeTab);
  const { setNodeRef, isOver } = useDroppable({
    id: category.id,
  });

  return (
    <div className="bg-gray-50 rounded-lg p-4 min-w-[300px]">
      <div className="flex items-center mb-4">
        <div
          className="w-3 h-3 rounded-full mr-2"
          style={{ backgroundColor: category.color }}
        />
        <h2 className="text-lg font-semibold text-gray-900">{category.name}</h2>
        <span className="ml-2 text-sm text-gray-500">({tabs.length})</span>
      </div>
      
      <div
        ref={setNodeRef}
        className={`space-y-2 min-h-[50px] transition-colors ${
          isOver ? 'bg-gray-100' : ''
        }`}
      >
        {tabs.map((tab, index) => (
          <DraggableTab
            key={tab.id}
            tab={tab}
            index={index}
            onRemove={removeTab}
          />
        ))}
      </div>
    </div>
  );
}