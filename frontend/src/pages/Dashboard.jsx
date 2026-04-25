import React from 'react';
import { motion } from 'framer-motion';
import { NavLink } from 'react-router-dom';
import { Sprout, TestTube } from 'lucide-react';
 


const Dashboard = () => {
    return (
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="text-center mb-16"
            >
                <h1 className="text-5xl font-extrabold text-nature-dark mb-4">
                    Intelligent Farming, <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-nature-primary to-nature-light">
                        Powered by AI
                    </span>
                </h1>
                <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                    Predict your crop yield and get tailored fertilizer recommendations using our hybrid machine learning pipeline equipped with Explainable AI.
                </p>
            </motion.div>
            

            <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">

                {/* Yield Prediction Card */}
                <motion.div
                    whileHover={{ y: -5 }}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                >
                    <NavLink to="/yield" className="block h-full">
                        <div className="glass-card h-full p-8 flex flex-col items-center text-center group cursor-pointer border-2 border-transparent hover:border-nature-light/30 transition-all">
                            <div className="w-20 h-20 bg-nature-primary/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                                <Sprout className="w-10 h-10 text-nature-primary" />
                            </div>
                            <h2 className="text-2xl font-bold text-gray-800 mb-3 block">Yield Prediction</h2>
                            <p className="text-gray-600 mb-6">
                                Estimate tons per hectare based on soil conditions, weather, and farming practices.
                            </p>
                            <div className="mt-auto">
                                <span className="text-nature-primary font-medium flex items-center gap-2 group-hover:gap-3 transition-all">
                                    Start Prediction &rarr;
                                </span>
                            </div>
                        </div>
                    </NavLink>
                </motion.div>

                {/* Fertilizer Recommendation Card */}
                <motion.div
                    whileHover={{ y: -5 }}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.5, delay: 0.4 }}
                >
                    <NavLink to="/fertilizer" className="block h-full">
                        <div className="glass-card h-full p-8 flex flex-col items-center text-center group cursor-pointer border-2 border-transparent hover:border-nature-accent/30 transition-all">
                            <div className="w-20 h-20 bg-nature-accent/10 rounded-full flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300">
                                <TestTube className="w-10 h-10 text-nature-accent" />
                            </div>
                            <h2 className="text-2xl font-bold text-gray-800 mb-3 block">Fertilizer Recommendation</h2>
                            <p className="text-gray-600 mb-6">
                                Get precise nutrient recommendations tailored to your specific crop and soil profile.
                            </p>
                            <div className="mt-auto">
                                <span className="text-nature-accent font-medium flex items-center gap-2 group-hover:gap-3 transition-all">
                                    Get Recommendation &rarr;
                                </span>
                            </div>
                        </div>
                    </NavLink>
                </motion.div>

            </div>
        </div>
    );
};

export default Dashboard;
