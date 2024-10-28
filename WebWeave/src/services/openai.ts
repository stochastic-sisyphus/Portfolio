const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

export async function summarizeContent(content: string): Promise<string> {
  if (!OPENAI_API_KEY) {
    return generateFallbackSummary(content);
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
            content: "You are a helpful assistant that summarizes web content concisely."
          },
          {
            role: "user",
            content: `Please summarize the following content in 2-3 sentences: ${content}`
          }
        ],
        max_tokens: 100,
      })
    });

    const data = await response.json();
    return data.choices?.[0]?.message?.content || generateFallbackSummary(content);
  } catch (error) {
    console.error('Error summarizing content:', error);
    return generateFallbackSummary(content);
  }
}

function generateFallbackSummary(content: string): string {
  const words = content.split(/\s+/);
  const firstSentences = content.split(/[.!?]+/).slice(0, 2).join('. ');
  return firstSentences.length > 150 ? firstSentences.slice(0, 150) + '...' : firstSentences;
}