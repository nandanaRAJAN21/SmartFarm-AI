import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { predictYield } from '../api';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import { Loader2, Sprout, Download } from 'lucide-react';
import jsPDF from 'jspdf';

const YieldPrediction = () => {
    const [formData, setFormData] = useState({
        Crop: 'Wheat',
        Region: 'Region_A',
        Soil_Type: 'Loam',
        Soil_pH: 6.5,
        Rainfall_mm: 800,
        Temperature_C: 25,
        Humidity_pct: 60,
        Fertilizer_Used_kg: 150,
        Irrigation: 'Drip',
        Pesticides_Used_kg: 10,
        Planting_Density: 15,
        Previous_Crop: 'Soybeans',
    });

    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');

    const handleChange = (e) => {
        const { name, value, type } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: type === 'number' ? parseFloat(value) : value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');

        try {
            const data = await predictYield(formData);
            setResult(data);
            console.log('Yield result:', data); // Debug
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // PDF download function
    const downloadYieldPDF = () => {
        if (!result) {
            alert("No data to generate PDF");
            return;
        }

        const pdf = new jsPDF();
        let y = 10;

        pdf.setFontSize(16);
        pdf.text("Yield Prediction Report", 10, y);
        y += 10;

        pdf.setFontSize(12);
        pdf.text(`Crop: ${formData.Crop}`, 10, y); y += 8;
        pdf.text(`Region: ${formData.Region}`, 10, y); y += 8;
        pdf.text(`Predicted Yield: ${result.predicted_value.toFixed(2)} tons/ha`, 10, y); y += 8;
        pdf.text(`Category: ${result.category}`, 10, y); y += 8;
        pdf.text(`Confidence: ${(result.confidence || 0) * 100}%`, 10, y); y += 8;

        if (result.explanation && result.explanation.summary) {
            pdf.text("Explanation:", 10, y); y += 6;
            const lines = pdf.splitTextToSize(result.explanation.summary, 180);
            pdf.text(lines, 10, y);
        }

        pdf.save("yield-prediction-report.pdf");
    };

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
                <h1 className="text-3xl font-bold text-nature-dark mb-2">Yield Prediction Engine</h1>
                <p className="text-gray-600">Enter your farm details to get an AI-powered yield estimate.</p>
            </motion.div>

            <div className="grid lg:grid-cols-2 gap-8">
                {/* Form Section */}
                <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.2 }} className="glass-card p-6">
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="col-span-1">
                                <label className="label-text">Crop</label>
                                <select name="Crop" value={formData.Crop} onChange={handleChange} className="input-field">
                                    <option>Wheat</option>
                                    <option>Maize</option>
                                    <option>Rice</option>
                                    <option>Barley</option>
                                    <option>Soybeans</option>
                                </select>
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Region</label>
                                <input type="text" name="Region" value={formData.Region} onChange={handleChange} className="input-field" placeholder="e.g. Region_A" />
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Soil Type</label>
                                <select name="Soil_Type" value={formData.Soil_Type} onChange={handleChange} className="input-field">
                                    <option>Loam</option>
                                    <option>Clay</option>
                                    <option>Sandy</option>
                                    <option>Silt</option>
                                </select>
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Soil pH</label>
                                <input type="number" step="0.1" name="Soil_pH" value={formData.Soil_pH} onChange={handleChange} className="input-field" />
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Rainfall (mm)</label>
                                <input type="number" name="Rainfall_mm" value={formData.Rainfall_mm} onChange={handleChange} className="input-field" />
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Temperature (°C)</label>
                                <input type="number" step="0.1" name="Temperature_C" value={formData.Temperature_C} onChange={handleChange} className="input-field" />
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Humidity (%)</label>
                                <input type="number" name="Humidity_pct" value={formData.Humidity_pct} onChange={handleChange} className="input-field" />
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Fertilizer Used (kg)</label>
                                <input type="number" name="Fertilizer_Used_kg" value={formData.Fertilizer_Used_kg} onChange={handleChange} className="input-field" />
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Irrigation</label>
                                <select name="Irrigation" value={formData.Irrigation} onChange={handleChange} className="input-field">
                                    <option>Drip</option>
                                    <option>Sprinkler</option>
                                    <option>Flood</option>
                                    <option>Rainfed</option>
                                </select>
                            </div>

                            <div className="col-span-1">
                                <label className="label-text">Pesticides (kg)</label>
                                <input type="number" name="Pesticides_Used_kg" value={formData.Pesticides_Used_kg} onChange={handleChange} className="input-field" />
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full btn-primary flex justify-center items-center mt-6"
                        >
                            {loading ? <Loader2 className="animate-spin mr-2" /> : null}
                            {loading ? 'Analyzing...' : 'Predict Yield'}
                        </button>

                        {error && (
                            <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm mt-4 border border-red-100">
                                {error}
                            </div>
                        )}

                    </form>
                </motion.div>

                {/* Results Section */}
                <div className="space-y-6">
                    {result ? (
                        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass-card p-6 border-l-4 border-l-nature-primary">
                            <h2 className="text-xl font-semibold mb-4 text-gray-800">Prediction Results</h2>
                            <div className="flex items-end gap-3 mb-2">
                                <span className="text-5xl font-bold text-nature-dark">
                                    {result.predicted_value.toFixed(2)}
                                </span>
                                <span className="text-gray-500 mb-1 font-medium">tons / ha</span>
                            </div>

                            <div className="inline-block px-3 py-1 rounded-full bg-nature-primary/10 text-nature-primary text-sm font-semibold mb-6">
                                Category: {result.category} Yield
                            </div>

                            <button
                                onClick={downloadYieldPDF}
                                className="w-full mt-4 bg-green-600 text-white py-2 rounded-lg flex justify-center items-center gap-2 hover:bg-green-700 transition"
                            >
                                <Download size={18} /> Download Yield Report
                            </button>

                            {result.explanation && (
                                <div className="mt-4 pt-4 border-t border-gray-100">
                                    <h3 className="font-medium text-gray-800 mb-2">AI Explanation Insights</h3>
                                    <p className="text-sm text-gray-600 mb-6 bg-blue-50/50 p-4 rounded-lg border border-blue-50 font-medium">
                                        {result.explanation.summary}
                                    </p>
                                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
                                        Top Influencing Factors ({result.explanation.method.toUpperCase()})
                                    </h4>
                                    <FeatureImportanceChart data={result.explanation.feature_importances} />
                                </div>
                            )}
                        </motion.div>
                    ) : (
                        <div className="h-full min-h-[400px] glass-card flex flex-col items-center justify-center text-gray-400 p-8 text-center bg-gray-50/50">
                            <Sprout className="w-16 h-16 mb-4 opacity-50" />
                            <p>Fill out the parameters and click predict to see your expected yield and AI explanations here.</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default YieldPrediction;