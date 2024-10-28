import React, { useEffect, useRef } from 'react';
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { useTabStore } from '../store/tabStore';
import * as d3 from 'd3';

export const TabVisualizations: React.FC = () => {
  const { tabs, categories } = useTabStore();
  const networkRef = useRef<SVGSVGElement>(null);

  const categoryData = categories.map(category => ({
    name: category.name,
    value: tabs.filter(tab => tab.category === category.id).length,
    color: category.color,
  }));

  useEffect(() => {
    if (!networkRef.current || tabs.length === 0) return;

    const width = networkRef.current.clientWidth;
    const height = 200;

    const nodes = tabs.map(tab => ({
      id: tab.id,
      title: tab.title,
      group: tab.category
    }));

    const links = [];
    for (let i = 0; i < tabs.length; i++) {
      for (let j = i + 1; j < tabs.length; j++) {
        if (tabs[i].category === tabs[j].category) {
          links.push({
            source: tabs[i].id,
            target: tabs[j].id
          });
        }
      }
    }

    const svg = d3.select(networkRef.current);
    svg.selectAll("*").remove();

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => (d as any).id))
      .force("charge", d3.forceManyBody().strength(-50))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", "#999")
      .attr("stroke-opacity", 0.6);

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", 5)
      .attr("fill", d => categories.find(c => c.id === (d as any).group)?.color || "#999");

    simulation.on("tick", () => {
      link
        .attr("x1", d => (d as any).source.x)
        .attr("y1", d => (d as any).source.y)
        .attr("x2", d => (d as any).target.x)
        .attr("y2", d => (d as any).target.y);

      node
        .attr("cx", d => (d as any).x)
        .attr("cy", d => (d as any).y);
    });
  }, [tabs, categories]);

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-6">
      <h2 className="text-lg font-semibold mb-4">Tab Analytics</h2>
      <div className="grid grid-cols-2 gap-4">
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={categoryData}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                label
              >
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="h-64">
          <svg ref={networkRef} width="100%" height="100%" />
        </div>
      </div>
    </div>
  );
};