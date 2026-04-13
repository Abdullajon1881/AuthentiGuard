'use client'

import { useState } from 'react'
import { AlertCircle } from 'lucide-react'
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
      <div className="max-w-[760px] w-full mx-auto px-6 py-10">
        {/* Page header */}
        <div className="mb-8">
          <p className="section-label mb-2">Analyze</p>
          <h1 className="font-serif text-2xl font-bold text-fg mb-1.5">
            {phase === 'upload'    && 'Content analysis'}
            {phase === 'analyzing' && 'Analyzing\u2026'}
            {phase === 'result'    && 'Analysis complete'}
          </h1>
          <p className="text-sm text-fg-3">
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
          <div role="alert" className="mb-5 p-3.5 bg-ai-dim border border-ai rounded text-ai text-sm flex items-center gap-2.5">
            <AlertCircle size={14} />
            {error}
          </div>
        )}

        {/* Card container */}
        <div className="card p-7">
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
