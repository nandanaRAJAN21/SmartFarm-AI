import React from 'react';
import { NavLink } from 'react-router-dom';
import { Leaf } from 'lucide-react';

const Navbar = () => {
    return (
        <nav className="sticky top-0 z-50 bg-white/90 backdrop-blur-md shadow-sm border-b border-gray-100">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                    <div className="flex items-center">
                        <NavLink to="/" className="flex items-center gap-2 group">
                            <div className="p-2 bg-nature-primary/10 rounded-lg group-hover:bg-nature-primary/20 transition-colors">
                                <Leaf className="w-6 h-6 text-nature-primary" />
                            </div>
                            <span className="font-bold text-xl text-nature-dark tracking-tight">SmartFarm</span>
                        </NavLink>
                    </div>

                    <div className="flex items-center space-x-1 sm:space-x-4 overflow-x-auto">
                        <NavLink
                            to="/"
                            className={({ isActive }) =>
                                `px-4 py-2 rounded-lg text-sm font-medium transition-all ${isActive
                                    ? 'bg-nature-primary text-white shadow-md'
                                    : 'text-gray-600 hover:bg-gray-100 hover:text-nature-dark'
                                }`
                            }
                        >
                            Dashboard
                        </NavLink>
                        <NavLink
                            to="/yield"
                            className={({ isActive }) =>
                                `px-4 py-2 rounded-lg text-sm font-medium transition-all ${isActive
                                    ? 'bg-nature-primary text-white shadow-md'
                                    : 'text-gray-600 hover:bg-gray-100 hover:text-nature-dark'
                                }`
                            }
                        >
                            Yield Prediction
                        </NavLink>
                        <NavLink
                            to="/fertilizer"
                            className={({ isActive }) =>
                                `px-4 py-2 rounded-lg text-sm font-medium transition-all ${isActive
                                    ? 'bg-nature-primary text-white shadow-md'
                                    : 'text-gray-600 hover:bg-gray-100 hover:text-nature-dark'
                                }`
                            }
                        >
                            Fertilizer Recommendation
                        </NavLink>
                        <NavLink
    to="/news"
    className={({ isActive }) =>
        `px-4 py-2 rounded-lg text-sm font-medium transition-all ${isActive
            ? 'bg-nature-primary text-white shadow-md'
            : 'text-gray-600 hover:bg-gray-100 hover:text-nature-dark'
        }`
    }
>
    News
</NavLink>
<NavLink
    to="/chatbot"
    className={({ isActive }) =>
        `px-4 py-2 rounded-lg text-sm font-medium transition-all ${isActive
            ? 'bg-nature-primary text-white shadow-md'
            : 'text-gray-600 hover:bg-gray-100 hover:text-nature-dark'
        }`
    }
>
    AI Chatbot
    
</NavLink>

                    
<NavLink
    to="/help"
    className={({ isActive }) =>
        `px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            isActive
                ? 'bg-nature-primary text-white shadow-md'
                : 'text-gray-600 hover:bg-gray-100 hover:text-nature-dark'
        }`
    }
>
    Help Desk
</NavLink>
<button
  onClick={() => {
    localStorage.removeItem("isLoggedIn");
    window.location.href = "/login";
  }}
  className="ml-2 px-3 py-1 bg-red-500 text-white rounded"
>
  Logout
</button>
</div>
                </div>
            </div>
        </nav>
    );
};
export default Navbar;
