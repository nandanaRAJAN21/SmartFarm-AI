import React, { useState } from 'react';
import { GoogleGenerativeAI } from "@google/generative-ai";

const ChatbotPage = () => {
  const [messages, setMessages] = useState([
    { text: "Hello! I am your Smart Farm Assistant 🌱. Ask anything about crops, Pests, weather or fetch live weather.", sender: 'ai' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [city, setCity] = useState('');
  const [weather, setWeather] = useState(null);

  //const API_KEY = "";
  const genAI = new GoogleGenerativeAI(API_KEY);

  // 🌤 Fetch Weather (City or GPS)
  const fetchWeather = async () => {
    try {
      setLoading(true);
      //const WEATHER_API_KEY = "";

      let url = "";

      if (city.trim()) {
        url = `https://api.openweathermap.org/data/2.5/weather?q=${city}&units=metric&appid=${WEATHER_API_KEY}`;
      } else {
        navigator.geolocation.getCurrentPosition(async (pos) => {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;

          const res = await fetch(`https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${WEATHER_API_KEY}`);
          const data = await res.json();
          updateWeather(data);
          (err) => { console.error(err) },
  { enableHighAccuracy: true, timeout: 10000 }
        });
        return;
      }

      const res = await fetch(url);
      const data = await res.json();
      updateWeather(data);

    } catch {
      setMessages(prev => [...prev, { text: "❌ Failed to fetch weather", sender: 'ai' }]);
      setLoading(false);
    }
  };

  const updateWeather = (data) => {
    const w = {
      location: data.name,
      temp: data.main.temp,
      humidity: data.main.humidity,
      rainfall: data.rain ? data.rain["1h"] : 0,
      wind: data.wind.speed
    };

    setWeather(w);

    const msg = `🌤 Weather in ${w.location}\nTemp: ${w.temp}°C | Humidity: ${w.humidity}% | Rain: ${w.rainfall}mm | Wind: ${w.wind} m/s`;

    setMessages(prev => [...prev, { text: msg, sender: 'ai' }]);
    setLoading(false);
  };

  // 💬 Smart Chat (uses weather if available)
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { text: input, sender: 'user' };
    const updated = [...messages, userMsg];
    setMessages(updated);
    const question = input;
    setInput('');
    setLoading(true);

    try {
      const model = genAI.getGenerativeModel({ model: "gemini-3.1-flash-lite-preview" });

     const prompt = `
You are a Smart Farming Assistant.

User Question: ${question}

${weather ? `Current Weather Data:
Location: ${weather.location}
Temperature: ${weather.temp}°C
Humidity: ${weather.humidity}%
Rainfall: ${weather.rainfall} mm
Wind: ${weather.wind} m/s
` : ""}

INSTRUCTIONS:

1. Detect user intent:
   - If question is about MARKET PRICE → respond ONLY with market info
   - If question is about CROPS → give crop advice
   - If question is about FERTILIZER → give fertilizer advice
   - If question is about WEATHER IMPACT → explain impact
   - If mixed → combine relevant sections ONLY

2. For MARKET PRICE:
   - Do NOT give exact real-time claims
   - Give approximate range
   - Mention region
   - Keep it SHORT
   - DO NOT include crop advice, risks, or fertilizer unless asked

3. For AGRICULTURE questions:
   Use structured format:

   🌱 Crop Advice:
   🧪 Fertilizer:
   🌦 Weather Impact:
   ⚠ Risks:

4. Keep answers relevant — DO NOT force all sections.
5. IMPORTANT:
- Do NOT force all sections
- Only include relevant sections
- Keep answers clear and practical
- No markdown symbols (**, ##)
- Use simple plain headings

`;

      const result = await model.generateContent(prompt);
      const reply = result.response.text();

      setMessages([...updated, { text: reply, sender: 'ai' }]);
    } catch {
      setMessages([...updated, { text: "❌ AI error", sender: 'ai' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[85vh] bg-white rounded-2xl shadow-2xl m-4 overflow-hidden">

      {/* Header */}
      <div className="bg-emerald-600 p-4 text-white flex flex-col gap-2">
        <div className="flex justify-between">
          <h1 className="font-bold">SmartFarm AI</h1>
          <button onClick={fetchWeather} className="bg-white text-emerald-600 px-3 py-1 rounded-full text-sm">Fetch Weather</button>
        </div>

        <input
          value={city}
          onChange={(e) => setCity(e.target.value)}
          placeholder="Enter city or leave empty for GPS"
          className="px-3 py-1 rounded text-black text-sm"
        />
      </div>

      {/* Weather Display */}
      {weather && (
        <div className="p-3 bg-gray-100 text-sm">
          📍 {weather.location} | 🌡 {weather.temp}°C | 💧 {weather.humidity}% | 🌧 {weather.rainfall}mm | 🌬 {weather.wind}
        </div>
      )}

      {/* Chat */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`p-3 rounded-xl max-w-[75%] text-sm whitespace-pre-line ${m.sender === 'user' ? 'bg-emerald-500 text-white' : 'bg-gray-100'}`}>
              {m.text}
            </div>
          </div>
        ))}
        {loading && <p className="text-xs">Thinking...</p>}
      </div>

      {/* Input */}
      <div className="p-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          className="flex-1 border rounded px-3 py-2"
        />
        <button onClick={sendMessage} className="bg-emerald-600 text-white px-4 rounded">
          →
        </button>
      </div>
    </div>
  );
};
export default ChatbotPage;