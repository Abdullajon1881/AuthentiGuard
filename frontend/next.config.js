/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',

  // Strict mode catches accessibility issues during development
  reactStrictMode: true,

  // Step 45: Security headers that also support WCAG accessibility
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
