import { Tab } from '../types';
import { summarizeContent } from './openai';
import { extractKeywords } from './keywords';
import { suggestCategory } from './categorization';
import { generateInsights } from './insights';
import { useTabStore } from '../store/tabStore';

export async function processUrl(url: string): Promise<Tab | null> {
  try {
    const response = await fetch(url);
    const html = await response.text();
    
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    
    const title = doc.title || url;
    const content = doc.body.textContent || '';
    const favicon = getFavicon(url, doc);
    
    const wordCount = content.split(/\s+/).length;
    const readingTime = Math.ceil(wordCount / 200);
    
    const summary = await summarizeContent(content);
    const keywords = await extractKeywords(content);
    
    const categories = useTabStore.getState().categories;
    const category = await suggestCategory(content, title, categories);
    
    const insights = await generateInsights({ 
      id: '', title, url, favicon, category, content, summary, 
      keywords, readingTime, lastVisited: new Date() 
    });
    
    return {
      id: crypto.randomUUID(),
      title,
      url,
      favicon,
      category,
      content,
      summary,
      keywords,
      readingTime,
      lastVisited: new Date(),
      keyPoints: insights,
    };
  } catch (error) {
    console.error('Error processing URL:', error);
    return null;
  }
}

function getFavicon(url: string, doc: Document): string {
  const favicon = doc.querySelector('link[rel*="icon"]');
  if (favicon && favicon.getAttribute('href')) {
    const faviconUrl = favicon.getAttribute('href')!;
    return faviconUrl.startsWith('http') ? faviconUrl : new URL(faviconUrl, url).href;
  }
  return new URL('/favicon.ico', url).href;
}

export async function processBatchUrls(urls: string[]): Promise<BatchProcessResult> {
  const successful: Tab[] = [];
  const failed: string[] = [];

  await Promise.all(
    urls.map(async (url) => {
      const result = await processUrl(url);
      if (result) {
        successful.push(result);
      } else {
        failed.push(url);
      }
    })
  );

  return { successful, failed };
}