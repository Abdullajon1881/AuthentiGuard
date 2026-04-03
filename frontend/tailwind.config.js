/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        serif: ['Libre Baskerville', 'Georgia', 'serif'],
        mono: ['IBM Plex Mono', 'Fira Code', 'monospace'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      colors: {
        accent:      'var(--accent)',
        'accent-dim':'var(--accent-dim)',
        ai:          'var(--ai)',
        'ai-dim':    'var(--ai-dim)',
        human:       'var(--human)',
        'human-dim': 'var(--human-dim)',
        uncertain:   'var(--uncertain)',
        'uncertain-dim':'var(--uncertain-dim)',
        'layer-1':   'var(--layer-1)',
        'layer-2':   'var(--layer-2)',
        'layer-3':   'var(--layer-3)',
        'layer-4':   'var(--layer-4)',
        surface: {
          DEFAULT: 'var(--bg)',
          2: 'var(--bg-2)',
          3: 'var(--bg-3)',
          4: 'var(--bg-4)',
        },
        edge: {
          DEFAULT: 'var(--border)',
          2: 'var(--border-2)',
          3: 'var(--border-3)',
        },
        fg: {
          DEFAULT: 'var(--text)',
          2: 'var(--text-2)',
          3: 'var(--text-3)',
        },
      },
      borderRadius: {
        DEFAULT: 'var(--radius)',
        lg: 'var(--radius-lg)',
      },
      boxShadow: {
        'card': '0 1px 2px rgba(0,0,0,0.15)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.2)',
      },
      animation: {
        'fade-up':   'fade-up 0.4s ease forwards',
        'pulse-dot': 'pulse-dot 1.4s ease-in-out infinite',
        'shimmer':   'shimmer 2s linear infinite',
        'score-bar': 'grow-width 0.6s cubic-bezier(0.34,1.56,0.64,1) forwards',
        'slide-up':  'slide-up 0.3s ease forwards',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%':      { opacity: '0.4', transform: 'scale(0.85)' },
        },
        'shimmer': {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'grow-width': {
          from: { width: '0%' },
          to:   { width: 'var(--target-width)' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
