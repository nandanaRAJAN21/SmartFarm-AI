import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { predictFertilizer } from '../api';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import { Loader2, TestTube } from 'lucide-react';
import jsPDF from "jspdf";


const FertilizerRecommendation = () => {
    const [formData, setFormData] = useState({
        Soil_Type: 'Loam',
        Soil_pH: 6.5,
        Soil_Moisture: 40,
        Organic_Carbon: 1.0,
        Electrical_Conductivity: 1.2,
        Nitrogen_Level: 80,
        Phosphorus_Level: 45,
        Potassium_Level: 60,
        Temperature: 25,
        Humidity: 60,
        Rainfall: 800,
        Crop_Type: 'Wheat',
        Crop_Growth_Stage: 'Vegetative',
        Season: 'Rabi',
        Irrigation_Type: 'Drip',
        Previous_Crop: 'Soybeans',
        Region: 'Region_A',
        Fertilizer_Used_Last_Season: 150,
        Yield_Last_Season: 4.5
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
            const data = await predictFertilizer(formData);
            setResult(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    const downloadPDF = () => {
    if (!result) return;

    const pdf = new jsPDF();

    let y = 10;

    pdf.setFontSize(16);
    pdf.text("Fertilizer Report", 10, y);

    y += 10;
    pdf.setFontSize(12);

    // Recommendation
    pdf.text(`Recommended: ${result.recommended}`, 10, y);
    y += 6;

    pdf.text(`Confidence: ${(result.confidence * 100).toFixed(1)}%`, 10, y);
    y += 8;

    // Alternatives
    pdf.text("Alternatives:", 10, y);
    y += 6;

    result.top3.forEach((alt) => {
        pdf.text(`${alt.name} - ${(alt.probability * 100).toFixed(1)}%`, 10, y);
        y += 6;
    });

    y += 6;

    // Explanation
    if (result.explanation) {
        pdf.text("Explanation:", 10, y);
        y += 6;

        const lines = pdf.splitTextToSize(result.explanation.summary, 180);
        pdf.text(lines, 10, y);
    }

    pdf.save("report.pdf");
};

    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mb-8"
            >
                <h1 className="text-3xl font-bold text-nature-dark mb-2">Fertilizer Recommendation</h1>
                <p className="text-gray-600">Enter comprehensive soil and farm details for AI-driven nutrient advice.</p>
            </motion.div>

            <div className="grid lg:grid-cols-3 gap-8">

                {/* Form Section */}
                <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                    className="glass-card p-6 lg:col-span-2"
                >
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-100 pb-2 mb-4">Soil Properties</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                            <div>
                                <label className="label-text">Soil Type</label>
                                <select name="Soil_Type" value={formData.Soil_Type} onChange={handleChange} className="input-field">
                                    <option>Loam</option><option>Clay</option><option>Sandy</option><option>Silt</option>
                                </select>
                            </div>
                            <div>
                                <label className="label-text">pH Level</label>
                                <input type="number" step="0.1" name="Soil_pH" value={formData.Soil_pH} onChange={handleChange} className="input-field" />
                            </div>
                            <div>
                                <label className="label-text">Moisture (%)</label>
                                <input type="number" name="Soil_Moisture" value={formData.Soil_Moisture} onChange={handleChange} className="input-field" />
                            </div>
                            <div>
                                <label className="label-text">Organic Carbon</label>
                                <input type="number" step="0.1" name="Organic_Carbon" value={formData.Organic_Carbon} onChange={handleChange} className="input-field" />
                            </div>
                        </div>

                        <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-100 pb-2 mb-4">Macronutrients</h3>
                        <div className="grid grid-cols-3 gap-4 mb-6">
                            <div>
                                <label className="label-text">Nitrogen (N)</label>
                                <input type="number" name="Nitrogen_Level" value={formData.Nitrogen_Level} onChange={handleChange} className="input-field" />
                            </div>
                            <div>
                                <label className="label-text">Phosphorus (P)</label>
                                <input type="number" name="Phosphorus_Level" value={formData.Phosphorus_Level} onChange={handleChange} className="input-field" />
                            </div>
                            <div>
                                <label className="label-text">Potassium (K)</label>
                                <input type="number" name="Potassium_Level" value={formData.Potassium_Level} onChange={handleChange} className="input-field" />
                            </div>
                        </div>

                        <h3 className="text-lg font-semibold text-gray-800 border-b border-gray-100 pb-2 mb-4">Environmental & Crop Info</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div>
                                <label className="label-text">Crop Type</label>
                                <select name="Crop_Type" value={formData.Crop_Type} onChange={handleChange} className="input-field">
                                    <option>Wheat</option><option>Maize</option><option>Rice</option><option>Barley</option><option>Soybeans</option>
                                </select>
                            </div>
                            <div>
                                <label className="label-text">Growth Stage</label>
                                <select name="Crop_Growth_Stage" value={formData.Crop_Growth_Stage} onChange={handleChange} className="input-field">
                                    <option>Vegetative</option><option>Flowering</option><option>Fruiting</option><option>Harvesting</option>
                                </select>
                            </div>
                            <div>
                                <label className="label-text">Season</label>
                                <select name="Season" value={formData.Season} onChange={handleChange} className="input-field">
                                    <option value="Rabi">Winter</option><option value="Kharif">Rainy</option>
                                    <option value="Zaid">Summer</option>
                                </select>
                            </div>
                            <div>
                                <label className="label-text">Irrigation</label>
                                <select name="Irrigation_Type" value={formData.Irrigation_Type} onChange={handleChange} className="input-field">
                                    <option>Drip</option><option>Sprinkler</option><option>Flood</option><option>Rainfed</option>
                                </select>
                            </div>
                        </div>

                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full btn-primary flex justify-center items-center mt-8"
                        >
                            {loading ? <Loader2 className="animate-spin mr-2" /> : null}
                            {loading ? 'Evaluating...' : 'Get Recommendation'}
                        </button>

                        {error && (
                            <div className="p-3 bg-red-50 text-red-600 rounded-lg text-sm mt-4 border border-red-100">
                                {error}
                            </div>
                        )}

                    </form>
                </motion.div>

                {/* Results Section */}
                <div className="lg:col-span-1 space-y-6">
                    {result ? (
                        <div id="report">
                        <motion.div
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="glass-card p-6 border-l-4 border-l-nature-accent"
                        >
                            <h2 className="text-xl font-semibold mb-4 text-gray-800">Top Recommendation</h2>

                            <div className="p-4 bg-nature-accent/10 rounded-xl mb-6 flex items-center gap-4">
                                <div className="p-3 bg-nature-accent/20 rounded-full">
                                    <TestTube className="w-8 h-8 text-nature-brown" />
                                </div>
                                <div>
                                    <div className="text-3xl font-bold text-nature-dark">{result.recommended}</div>
                                    <div className="text-sm font-medium text-nature-brown mt-1">
                                        {(result.confidence * 100).toFixed(1)}% Confidence
                                    </div>
                                </div>
                            </div>

                            <div className="mb-6">
                                <h4 className="text-sm font-semibold text-gray-500 uppercase tracking-wider mb-3">Alternatives</h4>
                                <div className="space-y-2">
                                    {result.top3.filter(a => a.name !== result.recommended).map((alt, idx) => (
                                        <div key={idx} className="flex justify-between items-center p-2 hover:bg-gray-50 rounded-lg transition-colors">
                                            <span className="font-medium text-gray-700">{alt.name}</span>
                                            <span className="text-sm text-gray-500">{(alt.probability * 100).toFixed(1)}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>

                            {result.explanation && (
                                <div className="mt-4 pt-4 border-t border-gray-100">
                                    <h3 className="font-medium text-gray-800 mb-2">Why this fertilizer?</h3>
                                    <p className="text-sm text-gray-600 mb-6 bg-orange-50/50 p-4 rounded-lg border border-orange-50 font-medium">
                                        {result.explanation.summary}
                                    </p>
                                    <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-2">
                                        Key Drivers ({result.explanation.method.toUpperCase()})
                                    </h4>
                                    <FeatureImportanceChart data={result.explanation.feature_importances} />
                                </div>
                            )}
                           <button
    onClick={downloadPDF}
    className="w-full mt-4 bg-green-600 text-white py-2 rounded-lg flex justify-center items-center hover:bg-green-700 transition"
>
    Download PDF
</button>
                        </motion.div>
                        </div>
                    ) : (
                        <div className="h-full min-h-[400px] glass-card flex flex-col items-center justify-center text-gray-400 p-8 text-center bg-gray-50/50">
                            <TestTube className="w-16 h-16 mb-4 opacity-50" />
                            <p>Provide your farm profile and our system will identify the optimal fertilizer mix.</p>
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
};

export default FertilizerRecommendation;