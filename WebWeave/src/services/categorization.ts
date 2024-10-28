import { Category } from '../types';

const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

export async function suggestCategory(
  content: string,
  title: string,
  availableCategories: Category[]
): Promise<string> {
  if (!OPENAI_API_KEY) {
    return suggestCategoryByKeywords(content, title, availableCategories);
  }

  try {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: "gpt-3.5-turbo",
        messages: [
          {
            role: "system",
            content: `You are a content categorization assistant. Given content and available categories (${availableCategories.map(c => c.name).join(', ')}), suggest the most appropriate category. Respond with just the category name.`
          },
          {
            role: "user",
            content: `Title: ${title}\n\nContent: ${content}`
          }
        ],
        max_tokens: 10,
      })
    });

    const data = await response.json();
    const suggestedCategory = data.choices?.[0]?.message?.content?.trim() || 'Uncategorized';
    const matchedCategory = availableCategories.find(
      c => c.name.toLowerCase() === suggestedCategory.toLowerCase()
    );
    
    return matchedCategory?.id || 'uncategorized';
  } catch (error) {
    console.error('Error suggesting category:', error);
    return suggestCategoryByKeywords(content, title, availableCategories);
  }
}

function suggestCategoryByKeywords(content: string, title: string, categories: Category[]): string {
  const categoryKeywords: Record<string, string[]> = {
    'work': ['project', 'meeting', 'deadline', 'report', 'business', 'client', 'office'],
    'personal': ['hobby', 'family', 'home', 'travel', 'recipe', 'lifestyle'],
    'research': ['study', 'paper', 'analysis', 'research', 'science', 'data', 'theory'],
  };

  const text = `${title} ${content}`.toLowerCase();
  let bestMatch = { id: 'uncategorized', score: 0 };

  categories.forEach(category => {
    const keywords = categoryKeywords[category.id];
    if (keywords) {
      const score = keywords.reduce((acc, keyword) => 
        acc + (text.includes(keyword) ? 1 : 0), 0);
      if (score > bestMatch.score) {
        bestMatch = { id: category.id, score };
      }
    }
  });

  return bestMatch.id;
}