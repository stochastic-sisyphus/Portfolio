import React from 'react';
import { useTabStore } from '../store/tabStore';

export const ExportButton: React.FC = () => {
  const { tabs, categories } = useTabStore();

  const exportData = () => {
    const exportObj = {
      tabs: tabs.map(tab => ({
        ...tab,
        lastVisited: tab.lastVisited.toISOString(),
      })),
      categories,
      exportDate: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportObj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tab-manager-export-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const exportMarkdown = () => {
    const markdownContent = categories.map(category => {
      const categoryTabs = tabs.filter(tab => tab.category === category.id);
      if (categoryTabs.length === 0) return '';

      return `## ${category.name}\n\n${categoryTabs.map(tab => (
        `### [${tab.title}](${tab.url})\n` +
        (tab.summary ? `\n${tab.summary}\n` : '') +
        (tab.keyPoints?.length ? `\nKey Points:\n${tab.keyPoints.map(point => `- ${point}`).join('\n')}\n` : '') +
        (tab.keywords?.length ? `\nKeywords: ${tab.keywords.join(', ')}\n` : '') +
        (tab.readingTime ? `\nReading time: ~${tab.readingTime} min\n` : '')
      )).join('\n\n')}\n\n`;
    }).join('');

    const blob = new Blob([markdownContent], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `tab-manager-export-${new Date().toISOString().split('T')[0]}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex space-x-2">
      <button
        onClick={exportData}
        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      >
        Export JSON
      </button>
      <button
        onClick={exportMarkdown}
        className="inline-flex items-center px-3 py-2 border border-gray-300 shadow-sm text-sm leading-4 font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      >
        Export Markdown
      </button>
    </div>
  );
};