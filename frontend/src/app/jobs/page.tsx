'use client'

import Link from 'next/link'

export default function JobsPage() {
  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '2rem' }}>
      <h1 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1rem' }}>
        Analysis Jobs
      </h1>
      <p style={{ color: '#666', marginBottom: '2rem' }}>
        Submit content for analysis on the{' '}
        <Link href="/analyze" style={{ color: '#2563eb', textDecoration: 'underline' }}>
          Analyze page
        </Link>
        , then track your jobs here.
      </p>
      <p style={{ color: '#999' }}>No jobs to display.</p>
    </div>
  )
}
