import React from 'react';
import { useDraggable } from '@dnd-kit/core';
import { Tab } from '../types';
import { TabContent } from './TabContent';

interface DraggableTabProps {
  tab: Tab;
  index: number;
  onRemove: (id: string) => void;
}

export function DraggableTab({ tab, index, onRemove }: DraggableTabProps) {
  const { attributes, listeners, setNodeRef, isDragging } = useDraggable({
    id: tab.id,
  });

  return (
    <div
      ref={setNodeRef}
      {...listeners}
      {...attributes}
      className={`bg-white rounded-lg shadow-sm p-4 mb-2 transition-shadow ${
        isDragging ? 'shadow-lg' : 'hover:shadow-md'
      }`}
    >
      <TabContent tab={tab} onRemove={onRemove} />
    </div>
  );
}