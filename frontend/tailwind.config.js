/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['IBM Plex Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        teal:      '#00c9a7',
        'ai-red':  '#f87171',
        'hu-green':'#34d399',
        'unc-amber':'#fbbf24',
      },
    },
  },
  plugins: [],
}
