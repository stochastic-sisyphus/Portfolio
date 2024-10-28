import React from 'react';
import { DndContext, DragEndEvent, closestCenter } from '@dnd-kit/core';
import { CategoryColumn } from './CategoryColumn';
import { useTabStore } from '../store/tabStore';

export function TabManager() {
  const { categories, moveTab, getFilteredTabs } = useTabStore();
  const tabs = getFilteredTabs();

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      moveTab(active.id.toString(), over.id.toString());
    }
  };

  return (
    <DndContext onDragEnd={handleDragEnd} collisionDetection={closestCenter}>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 p-6">
        {categories.map((category) => (
          <CategoryColumn
            key={category.id}
            category={category}
            tabs={tabs.filter((tab) => tab.category === category.id)}
          />
        ))}
      </div>
    </DndContext>
  );
}