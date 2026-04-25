import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { motion } from 'framer-motion';

const FeatureImportanceChart = ({ data }) => {
    if (!data || data.length === 0) return null;

    // Transform data for recharts
    const chartData = data.map(item => ({
        name: item.feature,
        value: item.direction === 'positive' ? item.importance : -item.importance,
        abs_value: item.importance,
        fill: item.direction === 'positive' ? '#2D6A4F' : '#E63946'
    })).sort((a, b) => b.abs_value - a.abs_value);

    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            const isPositive = payload[0].value > 0;
            return (
                <div className="bg-white p-3 border border-gray-100 shadow-xl rounded-lg">
                    <p className="font-medium text-gray-800 mb-1">{label}</p>
                    <div className="flex items-center gap-2">
                        <span className={`w-3 h-3 rounded-full ${isPositive ? 'bg-nature-primary' : 'bg-red-500'}`}></span>
                        <p className="text-sm text-gray-600">
                            Impact: <span className="font-semibold">{Math.abs(payload[0].value).toFixed(4)}</span>
                        </p>
                    </div>
                    <p className="text-xs text-gray-400 mt-1 mt-1 italic">
                        {isPositive ? 'Pushes prediction higher' : 'Pushes prediction lower'}
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="w-full h-64 mt-6"
        >
            <ResponsiveContainer width="100%" height="100%">
                <BarChart
                    layout="vertical"
                    data={chartData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#E5E7EB" />
                    <XAxis type="number" hide />
                    <YAxis
                        dataKey="name"
                        type="category"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: '#4B5563', fontSize: 12 }}
                        width={120}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ fill: '#F3F4F6' }} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                        {chartData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                    </Bar>
                </BarChart>
            </ResponsiveContainer>
        </motion.div>
    );
};

export default FeatureImportanceChart;
