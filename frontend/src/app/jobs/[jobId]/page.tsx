'use client'

import { useParams } from 'next/navigation'
import Link from 'next/link'

export default function JobDetailPage() {
  const params = useParams<{ jobId: string }>()

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '2rem' }}>
      <Link href="/jobs" style={{ color: '#2563eb', textDecoration: 'underline' }}>
        Back to Jobs
      </Link>
      <h1 style={{ fontSize: '1.5rem', fontWeight: 600, margin: '1rem 0' }}>
        Job Detail
      </h1>
      <p style={{ color: '#666' }}>Job ID: {params.jobId}</p>
      <p style={{ color: '#999', marginTop: '1rem' }}>
        Job details will appear here once the analysis is complete.
      </p>
    </div>
  )
}
