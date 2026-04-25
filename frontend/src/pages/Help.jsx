import React, { useState } from "react";

const Help = () => {
  const [search, setSearch] = useState("");

  const data = [
    {
      type: "scheme",
      name: "PM-KISAN",
      desc: "₹6000/year financial support to farmers",
      link: "https://pmkisan.gov.in"
    },
    {
      type: "scheme",
      name: "Crop Insurance (PMFBY)",
      desc: "Protection against crop loss",
      link: "https://pmfby.gov.in"
    },
    {
      type: "scheme",
      name: "Soil Health Card",
      desc: "Soil quality and fertilizer guidance",
      link: "https://soilhealth.dac.gov.in"
    },
    {
      type: "resource",
      name: "ICAR",
      desc: "Agricultural research and innovations",
      link: "https://icar.org.in"
    },
    {
      type: "resource",
      name: "Agriculture Ministry",
      desc: "Government policies and services",
      link: "https://agricoop.gov.in"
    }
  ];

  const filtered = data.filter(item =>
    item.name.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-slate-100 p-6 flex justify-center">
      <div className="max-w-4xl w-full space-y-6">

        {/* HEADER */}
        <div className="bg-emerald-600 text-white p-6 rounded-3xl shadow-lg">
          <h1 className="text-2xl font-bold">🌱 Farmer Help Desk</h1>
          <p className="text-sm opacity-90 mt-1">
            Trusted resources, schemes & support
          </p>
        </div>

        {/* SEARCH */}
        <input
          type="text"
          placeholder="Search schemes or resources..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full p-3 rounded-xl border shadow-sm focus:ring-2 focus:ring-emerald-400 outline-none"
        />

        {/* HELPLINE */}
        <div className="bg-white p-5 rounded-2xl shadow hover:shadow-lg transition">
          <h2 className="font-bold text-emerald-700 mb-2">📞 Helpline</h2>
          <p className="text-lg font-semibold text-slate-800">
            1800-180-1551
          </p>
          <p className="text-sm text-gray-500">
            Kisan Call Center (India)
          </p>
        </div>

        {/* CARDS */}
        <div className="grid md:grid-cols-2 gap-4">
          {filtered.map((item, i) => (
            <div
              key={i}
              className="bg-white p-4 rounded-2xl shadow hover:shadow-xl transition transform hover:-translate-y-1"
            >
              <h3 className="font-semibold text-slate-800">
                {item.name}
              </h3>

              <p className="text-sm text-gray-600 mb-2">
                {item.desc}
              </p>

              <span className="text-xs px-2 py-1 bg-emerald-100 text-emerald-700 rounded-full">
                {item.type.toUpperCase()}
              </span>

              <div className="mt-2">
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-600 text-sm font-semibold"
                >
                  Open Official Site →
                </a>
              </div>
            </div>
          ))}
        </div>

        {/* LOCAL SUPPORT */}
        <div className="bg-amber-50 p-5 rounded-2xl border border-amber-200 shadow">
          <h2 className="font-bold text-amber-700 mb-2">📍 Local Support</h2>
          <p className="text-sm text-slate-700">
            Visit your nearest <b>Krishi Bhavan</b> or agriculture office for direct assistance.
          </p>
        </div>

      </div>
    </div>
  );
};

export default Help;