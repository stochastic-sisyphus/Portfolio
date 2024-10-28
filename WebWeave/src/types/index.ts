export interface Tab {
  id: string;
  title: string;
  url: string;
  favicon: string;
  category: string;
  summary?: string;
  lastVisited: Date;
  content?: string;
  keyPoints?: string[];
  readingTime?: number;
  keywords?: string[];
}

export interface Category {
  id: string;
  name: string;
  color: string;
}

export interface BatchProcessResult {
  successful: Tab[];
  failed: string[];
}