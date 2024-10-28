import React, { useState } from 'react';
import { useTabStore } from '../store/tabStore';
import { processBatchUrls } from '../services/urlProcessor';

export const UrlInput: React.FC = () => {
  const [input, setInput] = useState('');
  const [processing, setProcessing] = useState(false);
  const addTab = useTabStore((state) => state.addTab);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setProcessing(true);
    try {
      // Extract URLs from input (handles markdown, plain text)
      const urlRegex = /(https?:\/\/[^\s]+)/g;
      const urls = input.match(urlRegex) || [];
      
      const { successful, failed } = await processBatchUrls(urls);
      
      // Add successful tabs to store
      successful.forEach(addTab);
      
      // Show error for failed URLs
      if (failed.length > 0) {
        console.error('Failed to process URLs:', failed);
      }
      
      setInput('');
    } catch (error) {
      console.error('Error processing URLs:', error);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl">
      <div className="mt-1">
        <textarea
          rows={3}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md"
          placeholder="Paste URLs or markdown content here..."
          disabled={processing}
        />
      </div>
      <div className="mt-2">
        <button
          type="submit"
          disabled={processing || !input.trim()}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
        >
          {processing ? 'Processing...' : 'Process URLs'}
        </button>
      </div>
    </form>
  );
};