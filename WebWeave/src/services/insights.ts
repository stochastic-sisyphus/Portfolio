import { Tab } from '../types';
import * as tf from '@tensorflow/tfjs';

const OPENAI_API_KEY = import.meta.env.VITE_OPENAI_API_KEY;

let model: tf.LayersModel | null = null;

async function loadModel() {
  if (!model) {
    model = await tf.loadLayersModel('/model/content_classifier.json');
  }
  return model;
}

export async function generateInsights(tab: Tab): Promise<string[]> {
  try {
    if (!OPENAI_API_KEY) {
      return generateBasicInsights(tab);
    }

    const [aiInsights, mlInsights] = await Promise.all([
      generateAIInsights(tab),
      generateMLInsights(tab)
    ]);

    return [...aiInsights, ...mlInsights];
  } catch (error) {
    console.error('Error generating insights:', error);
    return generateBasicInsights(tab);
  }
}

async function generateAIInsights(tab: Tab): Promise<string[]> {
  if (!OPENAI_API_KEY) return [];

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
            content: "Generate 2-3 key insights or action items based on the content. Each insight should be concise and actionable."
          },
          {
            role: "user",
            content: `Title: ${tab.title}\n\nContent: ${tab.content}`
          }
        ],
        max_tokens: 150,
      })
    });

    const data = await response.json();
    return data.choices?.[0]?.message?.content?.split('\n')
      .filter(Boolean)
      .map(insight => insight.replace(/^\d+\.\s*/, '')) || [];
  } catch (error) {
    console.error('Error generating AI insights:', error);
    return [];
  }
}

async function generateMLInsights(tab: Tab): Promise<string[]> {
  try {
    const model = await loadModel();
    const embedding = await generateEmbedding(tab.content || '');
    const prediction = model.predict(embedding) as tf.Tensor;
    const topics = await prediction.data();
    
    return topics
      .map((score, idx) => ({ score, topic: getTopicLabel(idx) }))
      .filter(({ score }) => score > 0.5)
      .map(({ topic }) => `Related to ${topic}`);
  } catch {
    return [];
  }
}

function generateBasicInsights(tab: Tab): string[] {
  const insights: string[] = [];
  
  // Reading time insight
  if (tab.readingTime) {
    insights.push(`Estimated reading time: ${tab.readingTime} minutes`);
  }

  // Content length insight
  if (tab.content) {
    const wordCount = tab.content.split(/\s+/).length;
    insights.push(`Contains ${wordCount} words`);
  }

  // Domain insight
  try {
    const domain = new URL(tab.url).hostname;
    insights.push(`From ${domain}`);
  } catch {
    // Invalid URL, skip domain insight
  }

  return insights;
}

async function generateEmbedding(text: string): Promise<tf.Tensor> {
  const words = text.toLowerCase().split(/\s+/).slice(0, 100);
  return tf.tensor2d([words.map(w => w.length)]);
}

function getTopicLabel(index: number): string {
  const topics = ['Technology', 'Business', 'Science', 'Health', 'Education'];
  return topics[index] || 'General';
}