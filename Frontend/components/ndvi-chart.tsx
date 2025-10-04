"use client";

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface NDVIChartProps {
  afforestation: number;
  deforestation: number;
}

export default function NDVIChart({ afforestation, deforestation }: NDVIChartProps) {
  const data = [
    { name: "Afforestation", area: afforestation, color: "#22c55e" }, // Green
    { name: "Deforestation", area: deforestation, color: "#ef4444" }, // Red
  ];

  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold text-center mb-4">Deforestation & Afforestation Trend</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical">
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" />
          <Tooltip />
          <Legend />
          <Bar dataKey="area" fill="#22c55e" name="Afforestation" />
          <Bar dataKey="area" fill="#ef4444" name="Deforestation" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
