import create from 'zustand';
import { Tab, Category } from '../types';
import { findSimilarTabs } from '../services/similarity';
import { persist } from 'zustand/middleware';

interface TabState {
  tabs: Tab[];
  categories: Category[];
  searchQuery: string;
  addTab: (tab: Tab) => void;
  removeTab: (id: string) => void;
  updateTab: (id: string, updates: Partial<Tab>) => void;
  moveTab: (tabId: string, categoryId: string) => void;
  addCategory: (category: Category) => void;
  setSearchQuery: (query: string) => void;
  getSimilarTabs: (tabId: string) => Tab[];
  getFilteredTabs: () => Tab[];
}

export const useTabStore = create<TabState>()(
  persist(
    (set, get) => ({
      tabs: [],
      categories: [
        { id: 'work', name: 'Work', color: '#4F46E5' },
        { id: 'personal', name: 'Personal', color: '#10B981' },
        { id: 'research', name: 'Research', color: '#F59E0B' },
        { id: 'uncategorized', name: 'Uncategorized', color: '#6B7280' },
      ],
      searchQuery: '',
      addTab: (tab) => set((state) => ({ tabs: [...state.tabs, tab] })),
      removeTab: (id) => set((state) => ({
        tabs: state.tabs.filter((tab) => tab.id !== id)
      })),
      updateTab: (id, updates) => set((state) => ({
        tabs: state.tabs.map((tab) => 
          tab.id === id ? { ...tab, ...updates } : tab
        )
      })),
      moveTab: (tabId, categoryId) => set((state) => ({
        tabs: state.tabs.map((tab) =>
          tab.id === tabId ? { ...tab, category: categoryId } : tab
        )
      })),
      addCategory: (category) => set((state) => ({
        categories: [...state.categories, category]
      })),
      setSearchQuery: (query) => set({ searchQuery: query }),
      getSimilarTabs: (tabId) => {
        const state = get();
        const targetTab = state.tabs.find(tab => tab.id === tabId);
        if (!targetTab) return [];
        return findSimilarTabs(targetTab, state.tabs);
      },
      getFilteredTabs: () => {
        const state = get();
        if (!state.searchQuery) return state.tabs;
        
        const query = state.searchQuery.toLowerCase();
        return state.tabs.filter(tab => 
          tab.title.toLowerCase().includes(query) ||
          tab.url.toLowerCase().includes(query) ||
          (tab.summary && tab.summary.toLowerCase().includes(query)) ||
          (tab.keywords && tab.keywords.some(k => k.toLowerCase().includes(query)))
        );
      },
    }),
    {
      name: 'tab-storage',
      version: 1,
    }
  )
);