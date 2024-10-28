import stringSimilarity from 'string-similarity';
import { Tab } from '../types';
import * as d3 from 'd3';

export function findSimilarTabs(targetTab: Tab, allTabs: Tab[]): Tab[] {
  const otherTabs = allTabs.filter(tab => tab.id !== targetTab.id);
  
  // Content-based similarity
  const contentSimilarities = otherTabs.map(tab => ({
    tab,
    similarity: calculateContentSimilarity(targetTab, tab)
  }));

  // Network-based similarity
  const networkSimilarities = calculateNetworkSimilarity(targetTab, otherTabs);
  
  // Combine similarities
  const combinedSimilarities = contentSimilarities.map(({ tab, similarity }, index) => ({
    tab,
    similarity: (similarity + networkSimilarities[index]) / 2
  }));

  return combinedSimilarities
    .filter(({ similarity }) => similarity > 0.3)
    .sort((a, b) => b.similarity - a.similarity)
    .map(({ tab }) => tab);
}

function calculateContentSimilarity(tab1: Tab, tab2: Tab): number {
  const text1 = `${tab1.title} ${tab1.summary || ''} ${tab1.keywords?.join(' ') || ''}`;
  const text2 = `${tab2.title} ${tab2.summary || ''} ${tab2.keywords?.join(' ') || ''}`;
  
  return stringSimilarity.compareTwoStrings(text1, text2);
}

function calculateNetworkSimilarity(targetTab: Tab, otherTabs: Tab[]): number[] {
  // Create a graph of tabs
  const nodes = otherTabs.map(tab => ({
    id: tab.id,
    group: tab.category
  }));

  const links = [];
  for (let i = 0; i < otherTabs.length; i++) {
    for (let j = i + 1; j < otherTabs.length; j++) {
      const similarity = calculateContentSimilarity(otherTabs[i], otherTabs[j]);
      if (similarity > 0.3) {
        links.push({
          source: otherTabs[i].id,
          target: otherTabs[j].id,
          value: similarity
        });
      }
    }
  }

  // Use D3's force simulation to calculate network centrality
  const simulation = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d => (d as any).id))
    .force('charge', d3.forceManyBody())
    .force('center', d3.forceCenter());

  // Run simulation
  for (let i = 0; i < 100; i++) {
    simulation.tick();
  }

  // Calculate network-based similarity scores
  return otherTabs.map(tab => {
    const node = nodes.find(n => n.id === tab.id);
    if (!node) return 0;
    
    const { x = 0, y = 0 } = node;
    return 1 / (1 + Math.sqrt(x * x + y * y));
  });
}