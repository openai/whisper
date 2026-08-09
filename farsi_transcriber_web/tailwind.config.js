/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: '#4CAF50',
        'primary-hover': '#45a049',
        'primary-active': '#3d8b40',
      }
    },
  },
  plugins: [],
}
