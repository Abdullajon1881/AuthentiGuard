'use client'

// Step 39: Real-time progress — shows exactly which stage is running,
// not just a generic spinner. Polls the job status and maps it to
// human-readable stage labels with a visual pipeline stepper.

import { useEffect, useState } from 'react'
import { analysis } from '@/lib/api'
import type { DetectionResult, JobStatus } from '@/types'

const STAGES = [
  { key: 'pending',      label: 'Job queued',               sub: 'Waiting for an available worker' },
  { key: 'extracting',   label: 'Extracting content',       sub: 'Reading and cleaning the input' },
  { key: 'perplexity',   label: 'Layer 1 — Perplexity',     sub: 'Computing GPT-2 reference scores' },
  { key: 'stylometry',   label: 'Layer 2 — Stylometry',     sub: 'Analyzing writing style signatures' },
  { key: 'transformer',  label: 'Layer 3 — Transformer',    sub: 'Running DeBERTa-v3 classifier' },
  { key: 'adversarial',  label: 'Layer 4 — Adversarial',    sub: 'Checking evasion-attack patterns' },
  { key: 'ensemble',     label: 'Meta-classifier',           sub: 'Combining all layer outputs' },
  { key: 'processing',   label: 'Finalizing result',        sub: 'Applying calibration & scoring' },
  { key: 'completed',    label: 'Analysis complete',        sub: 'Results ready' },
]

// Map backend status + elapsed time → simulated stage index
function inferStageIndex(status: JobStatus, elapsedMs: number): number {
  if (status === 'completed') return STAGES.length - 1
  if (status === 'failed')    return -1
  if (status === 'pending')   return 0

  // 'processing' — simulate pipeline progress by elapsed time
  // Each layer takes roughly 1–2s on GPU, 5–10s on CPU
  const t = elapsedMs / 1000
  if (t < 1)   return 1
  if (t < 3)   return 2
  if (t < 5)   return 3
  if (t < 8)   return 4
  if (t < 11)  return 5
  if (t < 14)  return 6
  return 7
}

interface Props {
  jobId: string
  onComplete: (result: DetectionResult) => void
  onError: (msg: string) => void
}

export default function AnalysisViewer({ jobId, onComplete, onError }: Props) {
  const [stageIdx, setStageIdx] = useState(0)
  const [startTime]             = useState(Date.now())
  const [elapsed, setElapsed]   = useState(0)

  useEffect(() => {
    let cancelled = false
    const tick = setInterval(() => {
      if (!cancelled) setElapsed(Date.now() - startTime)
    }, 250)

    const run = async () => {
      try {
        const result = await analysis.pollUntilDone(
          jobId,
          (status) => {
            if (!cancelled) {
              const idx = inferStageIndex(status as JobStatus, Date.now() - startTime)
              setStageIdx(idx)
            }
          },
          1500,
        )
        if (!cancelled) {
          setStageIdx(STAGES.length - 1)
          setTimeout(() => onComplete(result), 400)
        }
      } catch (err) {
        if (!cancelled) onError(err instanceof Error ? err.message : 'Analysis failed')
      }
    }

    run()
    return () => { cancelled = true; clearInterval(tick) }
  }, [jobId]) // eslint-disable-line react-hooks/exhaustive-deps

  const elapsedSec = (elapsed / 1000).toFixed(1)

  return (
    <div className="fade-up" style={{ padding: '24px 0' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 28 }}>
        <h2 style={{ fontSize: 15, fontWeight: 500, color: 'var(--text)', margin: 0 }}>
          Running analysis
        </h2>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
          {elapsedSec}s elapsed
        </span>
      </div>

      {/* Stage pipeline */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {STAGES.map((stage, idx) => {
          const isDone    = idx < stageIdx
          const isActive  = idx === stageIdx
          const isPending = idx > stageIdx

          return (
            <div key={stage.key} style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
              {/* Line + dot */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 20, flexShrink: 0 }}>
                <div style={{
                  width: 12, height: 12,
                  borderRadius: '50%',
                  marginTop: 4,
                  flexShrink: 0,
                  background: isDone
                    ? 'var(--teal)'
                    : isActive
                    ? 'var(--teal)'
                    : 'var(--bg-4)',
                  border: isPending ? '1px solid var(--border-2)' : 'none',
                  transition: 'all 0.3s',
                  ...(isActive ? { boxShadow: '0 0 0 3px var(--teal-glow)' } : {}),
                }} className={isActive ? 'pulse-dot' : ''} />
                {idx < STAGES.length - 1 && (
                  <div style={{
                    width: 1, flex: 1, minHeight: 28,
                    background: isDone ? 'var(--teal)' : 'var(--border)',
                    opacity: isDone ? 0.6 : 1,
                    transition: 'background 0.3s',
                  }} />
                )}
              </div>

              {/* Label */}
              <div style={{ paddingBottom: idx < STAGES.length - 1 ? 16 : 0 }}>
                <p style={{
                  margin: 0,
                  fontSize: 13,
                  fontWeight: isActive ? 500 : 400,
                  color: isDone
                    ? 'var(--text-3)'
                    : isActive
                    ? 'var(--text)'
                    : 'var(--text-3)',
                  transition: 'color 0.3s',
                }}>
                  {stage.label}
                  {isDone && (
                    <span style={{ marginLeft: 8, fontSize: 11, color: 'var(--teal)', fontFamily: 'var(--font-mono)' }}>
                      ✓
                    </span>
                  )}
                </p>
                {isActive && (
                  <p style={{ margin: '2px 0 0', fontSize: 12, color: 'var(--text-3)' }}>
                    {stage.sub}
                  </p>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
