/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        nature: {
          dark: '#1B4332',
          primary: '#2D6A4F',
          light: '#40916C',
          accent: '#F4A261',
          bg: '#F8F9FA',
          brown: '#8B4513'
        }
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
