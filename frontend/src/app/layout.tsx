import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AuthentiGuard — AI Content Detection',
  description: 'Detect AI-generated text, images, video, audio, and code with multi-layer ensemble analysis.',
  themeColor: '#1a1816',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  )
}
