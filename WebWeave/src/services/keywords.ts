const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

export async function extractKeywords(content: string): Promise<string[]> {
  if (!OPENAI_API_KEY) {
    return generateFallbackKeywords(content);
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
            content: "Extract 5-7 relevant keywords from the given content. Return only the keywords separated by commas."
          },
          {
            role: "user",
            content
          }
        ],
        max_tokens: 50,
      })
    });

    const data = await response.json();
    const keywords = data.choices?.[0]?.message?.content?.split(',') || [];
    return keywords.map(k => k.trim());
  } catch (error) {
    console.error('Error extracting keywords:', error);
    return generateFallbackKeywords(content);
  }
}

function generateFallbackKeywords(content: string): string[] {
  const words = content.toLowerCase().split(/\s+/);
  const wordFreq = new Map<string, number>();
  
  // Ignore common words
  const stopWords = new Set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at']);
  
  words.forEach(word => {
    if (word.length > 3 && !stopWords.has(word)) {
      wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
    }
  });
  
  return Array.from(wordFreq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([word]) => word);
}