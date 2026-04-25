import React, { useState } from "react";
import { Leaf, Lock, User, ArrowRight, Droplets } from "lucide-react";
// Import the image so Vite processes the path correctly
import farmBg from "../assets/tractor-field-ai-generated.jpg"; 

const Login = () => {
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");

  const handleLogin = () => {
    if (user === "Farmer" && pass === "1234") {
      localStorage.setItem("isLoggedIn", "true");
      window.location.href = "/";
    } else {
      alert("Invalid credentials. Try Farmer / 1234");
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center font-sans overflow-hidden relative">
      
      //{/* 🌄 Background Image with Overlay */}
      <div 
        className="absolute inset-0 z-0 bg-cover bg-center bg-no-repeat transition-transform duration-1000 scale-105"
        style={{ 
          //Using the imported variable instead of a string path
          backgroundImage: `url(${farmBg})` 
        }}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-emerald-950/90 via-emerald-900/70 to-transparent"></div>
      </div>

      {/* 🌿 Floating Blur Shapes */}
      <div className="absolute w-72 h-72 bg-lime-400/20 rounded-full blur-3xl top-10 left-10 animate-pulse"></div>

      {/* 💎 Glass Card */}
      <div className="relative z-10 w-full max-w-md px-6">
        <div className="bg-white/80 backdrop-blur-xl border border-white/40 p-10 rounded-[2.5rem] shadow-2xl">

          {/* Header */}
          <div className="flex flex-col items-center mb-10">
            <div className="bg-gradient-to-br from-emerald-500 to-lime-500 p-4 rounded-2xl shadow-xl mb-5 transform -rotate-3">
              <Leaf className="text-white w-8 h-8" />
            </div>
            <h1 className="text-3xl font-black text-emerald-950 tracking-tight">
              SmartFarm <span className="text-lime-600">AI</span>
            </h1>
          </div>

          {/* Form */}
          <div className="space-y-5">
            <div className="group relative">
              <User className="absolute left-4 top-4 text-emerald-700/40 w-5 h-5" />
              <input
                type="text"
                placeholder="USERNAME"
                className="w-full bg-white/70 border border-emerald-100 rounded-2xl py-4 pl-12 pr-4 outline-none focus:ring-4 focus:ring-lime-500/20"
                onChange={(e) => setUser(e.target.value)}
              />
            </div>

            <div className="group relative">
              <Lock className="absolute left-4 top-4 text-emerald-700/40 w-5 h-5" />
              <input
                type="password"
                placeholder="PASSWORD"
                className="w-full bg-white/70 border border-emerald-100 rounded-2xl py-4 pl-12 pr-4 outline-none focus:ring-4 focus:ring-lime-500/20"
                onChange={(e) => setPass(e.target.value)}
              />
            </div>

            <button
              onClick={handleLogin}
              className="w-full bg-gradient-to-r from-emerald-600 to-lime-500 text-white font-bold py-4 rounded-2xl shadow-lg flex items-center justify-center gap-3 active:scale-95 transition-transform"
            >
              <span>LOGIN</span>
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>

          {/* Footer */}
          <div className="mt-10 pt-6 border-t border-emerald-100/60 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-2.5 h-2.5 bg-lime-500 rounded-full animate-ping"></div>
              <span className="text-[10px] font-bold text-emerald-900/50 uppercase tracking-widest">System Live</span>
            </div>
            <div className="flex items-center gap-1 text-emerald-900/40">
              <Droplets className="w-3 h-3" />
              <span className="text-[10px] font-medium">Eco-Safe</span>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default Login;