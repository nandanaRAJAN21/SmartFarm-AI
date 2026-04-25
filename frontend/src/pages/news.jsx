import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Loader2, AlertCircle, RefreshCw } from 'lucide-react';

const News = () => {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // GNews API details
  //const API_KEY = '';
  // Simplified query: GNews handles logical OR better with simple spaces or "OR"
  const QUERY = 'agriculture OR farming OR "Indian farmers" OR crops';

  const fetchNews = async () => {
    setLoading(true);
    setError('');
    try {
      // 1. We encode the query to handle spaces and special characters
      const url = `https://gnews.io/api/v4/search?q=${encodeURIComponent(QUERY)}&lang=en&country=in&max=10&token=${API_KEY}`;
      
      const res = await fetch(url);
      const data = await res.json();

      // 2. Handle API Errors (like reaching the daily limit)
      if (data.errors) {
        setError(data.errors[0]);
        return;
      }

      // 3. Trust the API results. Manual filtering is usually unnecessary 
      // because GNews' 'q' parameter already does the heavy lifting.
      if (data.articles && data.articles.length > 0) {
        setArticles(data.articles);
      } else {
        setError('No recent news found for these topics in India.');
      }
    } catch (err) {
      setError('Connection failed. Please check your network.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchNews();
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 py-10 min-h-screen">
      <div className="flex justify-between items-end mb-8 border-b border-emerald-100 pb-6">
        <div>
          <h1 className="text-3xl font-black text-emerald-900 uppercase tracking-tight">Agri-News Feed</h1>
          <p className="text-emerald-700/60 font-medium">Real-time updates from across India</p>
        </div>
        <button 
          onClick={fetchNews}
          className="p-2 text-emerald-600 hover:bg-emerald-50 rounded-full transition-colors"
          title="Refresh News"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading ? (
        <div className="flex flex-col justify-center items-center h-64 space-y-4">
          <Loader2 className="animate-spin text-emerald-500 w-10 h-10" />
          <p className="text-emerald-800/40 font-bold animate-pulse uppercase text-xs tracking-widest">Gathering Reports</p>
        </div>
      ) : error ? (
        <div className="bg-amber-50 border border-amber-200 p-8 rounded-3xl text-center max-w-md mx-auto">
          <AlertCircle className="w-12 h-12 text-amber-500 mx-auto mb-4" />
          <h3 className="text-amber-900 font-bold mb-2">Notice</h3>
          <p className="text-amber-800/70 text-sm mb-6">{error}</p>
          <button 
            onClick={fetchNews}
            className="bg-amber-500 text-white px-6 py-2 rounded-xl font-bold hover:bg-amber-600 transition shadow-lg shadow-amber-200"
          >
            Try Again
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {articles.map((article, idx) => (
            <motion.div
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="bg-white rounded-[2rem] overflow-hidden shadow-sm border border-emerald-50 hover:shadow-xl hover:shadow-emerald-100/50 transition-all group"
            >
              <div className="h-48 overflow-hidden relative">
                <img 
                  src={article.image || 'https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800'} 
                  alt="News" 
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                />
                <div className="absolute top-4 left-4 bg-white/90 backdrop-blur px-3 py-1 rounded-full text-[10px] font-bold text-emerald-800">
                   {article.source.name}
                </div>
              </div>
              
              <div className="p-6">
                <h2 className="text-lg font-bold text-slate-800 mb-2 leading-tight line-clamp-2">
                  {article.title}
                </h2>
                <p className="text-sm text-slate-500 mb-6 line-clamp-3">
                  {article.description}
                </p>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 text-emerald-600 font-bold text-sm group-hover:gap-3 transition-all"
                >
                  Read full report <span className="text-lg">→</span>
                </a>
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};

export default News;