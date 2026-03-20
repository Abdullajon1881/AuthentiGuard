import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AuthentiGuard — AI Content Detection',
  description: 'Detect AI-generated text, images, video, audio, and code with multi-layer ensemble analysis.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  )
}
