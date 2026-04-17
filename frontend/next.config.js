/** @type {import('next').NextConfig} */

// Server-side backend URL — used only by the Next.js proxy (never sent to browser).
// In Docker: http://backend:8000 (resolved via Docker DNS).
// In local dev: http://localhost:8000 (default).
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000'

const nextConfig = {
  output: 'standalone',

  reactStrictMode: true,

  async rewrites() {
    return {
      // Serve landing page at root without changing the URL
      beforeFiles: [
        { source: '/', destination: '/landing.html' },
      ],
      // Proxy all API calls from the browser to the FastAPI backend.
      // This lets the landing page (BASE_URL = window.location.origin)
      // and any relative-URL callers reach the backend through port 3000.
      afterFiles: [
        {
          source: '/api/:path*',
          destination: `${BACKEND_URL}/api/:path*`,
        },
        {
          source: '/health',
          destination: `${BACKEND_URL}/health`,
        },
      ],
    }
  },

  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          { key: 'X-Content-Type-Options',    value: 'nosniff' },
          { key: 'X-Frame-Options',            value: 'DENY' },
          { key: 'X-XSS-Protection',           value: '1; mode=block' },
          { key: 'Referrer-Policy',            value: 'strict-origin-when-cross-origin' },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ]
  },

  // Transpile only what's needed
  transpilePackages: [],

  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
  },
}

module.exports = nextConfig
