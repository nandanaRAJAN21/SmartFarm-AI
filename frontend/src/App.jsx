import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import YieldPrediction from './pages/YieldPrediction';
import FertilizerRecommendation from './pages/FertilizerRecommendation';
import News from './pages/news';
import ChatbotPage from './pages/Chatbot';
import Help from './pages/Help';
import Login from './pages/Login';

function App() {
  const isLoggedIn = localStorage.getItem("isLoggedIn");

  return (
    <Router>
      <div className="min-h-screen bg-nature-bg relative overflow-hidden">

        {isLoggedIn && <Navbar />}

        <main className="relative z-10 w-full">
          <Routes>

            {/* LOGIN FIRST */}
            <Route
              path="/login"
              element={
                isLoggedIn ? <Navigate to="/" /> : <Login />
              }
            />

            {/* PROTECTED ROUTES */}
            <Route
              path="/"
              element={
                isLoggedIn ? <Dashboard /> : <Navigate to="/login" />
              }
            />

            <Route
              path="/yield"
              element={
                isLoggedIn ? <YieldPrediction /> : <Navigate to="/login" />
              }
            />

            <Route
              path="/fertilizer"
              element={
                isLoggedIn ? <FertilizerRecommendation /> : <Navigate to="/login" />
              }
            />

            <Route
              path="/news"
              element={
                isLoggedIn ? <News /> : <Navigate to="/login" />
              }
            />

            <Route
              path="/chatbot"
              element={
                isLoggedIn ? <ChatbotPage /> : <Navigate to="/login" />
              }
            />

            <Route
              path="/help"
              element={
                isLoggedIn ? <Help /> : <Navigate to="/login" />
              }
            />

          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;