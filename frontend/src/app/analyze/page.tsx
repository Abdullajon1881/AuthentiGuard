'use client'

// Steps 38–41, 46: Main analyze page.
// Orchestrates: Upload → AnalysisViewer (progress) → ReportView (results).
// This page validates end-to-end text analysis works in the browser.

import { useState } from 'react'
import AppShell from '@/components/layout/AppShell'
import UploadModule from '@/components/upload/UploadModule'
import AnalysisViewer from '@/components/analysis/AnalysisViewer'
import ReportView from '@/components/analysis/ReportView'
import type { AnalysisJob, DetectionResult } from '@/types'

type Phase = 'upload' | 'analyzing' | 'result'

export default function AnalyzePage() {
  const [phase, setPhase]   = useState<Phase>('upload')
  const [job,   setJob]     = useState<AnalysisJob | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError]   = useState<string | null>(null)

  function handleJobCreated(j: AnalysisJob) {
    setJob(j)
    setError(null)
    setPhase('analyzing')
  }

  function handleComplete(r: DetectionResult) {
    setResult(r)
    setPhase('result')
  }

  function handleError(msg: string) {
    setError(msg)
    setPhase('upload')
  }

  function reset() {
    setPhase('upload')
    setJob(null)
    setResult(null)
    setError(null)
  }

  return (
    <AppShell activePath="/analyze">
      <div style={{
        maxWidth: 760,
        width: '100%',
        margin: '0 auto',
        padding: '40px 24px',
      }}>
        {/* Page header */}
        <div style={{ marginBottom: 28 }}>
          <div style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.12em',
            textTransform: 'uppercase', color: 'var(--teal)', marginBottom: 6,
          }}>
            AuthentiGuard
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 500, color: 'var(--text)', margin: '0 0 6px' }}>
            {phase === 'upload'    && 'Analyze content'}
            {phase === 'analyzing' && 'Analyzing…'}
            {phase === 'result'    && 'Analysis complete'}
          </h1>
          <p style={{ fontSize: 13, color: 'var(--text-3)', margin: 0 }}>
            {phase === 'upload' &&
              'Paste text, upload a file, or submit a URL to detect AI-generated content.'}
            {phase === 'analyzing' &&
              'Running your content through the 4-layer detection ensemble.'}
            {phase === 'result' &&
              'Review the full forensic breakdown below.'}
          </p>
        </div>

        {/* Error banner */}
        {error && (
          <div role="alert" style={{
            marginBottom: 20,
            padding: '12px 16px',
            background: 'var(--ai-dim)',
            border: '1px solid var(--ai)',
            borderRadius: 'var(--radius)',
            color: 'var(--ai)',
            fontSize: 13,
            display: 'flex', alignItems: 'center', gap: 10,
          }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
              <circle cx="12" cy="12" r="10"/>
              <line x1="12" y1="8" x2="12" y2="12"/>
              <line x1="12" y1="16" x2="12.01" y2="16"/>
            </svg>
            {error}
          </div>
        )}

        {/* Card container */}
        <div style={{
          background: 'var(--bg-2)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          padding: '28px',
        }}>
          {phase === 'upload' && (
            <UploadModule
              onJobCreated={handleJobCreated}
              onError={handleError}
            />
          )}

          {phase === 'analyzing' && job && (
            <AnalysisViewer
              jobId={job.job_id}
              onComplete={handleComplete}
              onError={handleError}
            />
          )}

          {phase === 'result' && result && (
            <ReportView
              result={result}
              onNewAnalysis={reset}
            />
          )}
        </div>
      </div>
    </AppShell>
  )
}
