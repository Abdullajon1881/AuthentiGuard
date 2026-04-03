'use client'

import { useEffect, useState } from 'react'
import { Check } from 'lucide-react'
import { analysis } from '@/lib/api'
import { cn } from '@/lib/cn'
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

function inferStageIndex(status: JobStatus, elapsedMs: number): number {
  if (status === 'completed') return STAGES.length - 1
  if (status === 'failed')    return -1
  if (status === 'pending')   return 0
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
  const progress = Math.min((stageIdx / (STAGES.length - 1)) * 100, 100)

  return (
    <div className="animate-fade-up py-6">
      {/* Progress bar */}
      <div className="h-px bg-edge mb-6 overflow-hidden">
        <div
          className="h-full bg-accent transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Header */}
      <div className="flex items-baseline gap-3 mb-7">
        <h2 className="text-[15px] font-medium text-fg">
          Running analysis
        </h2>
        <span className="font-mono text-xs text-fg-3">
          {elapsedSec}s elapsed
        </span>
      </div>

      {/* Stage pipeline */}
      <div className="flex flex-col">
        {STAGES.map((stage, idx) => {
          const isDone    = idx < stageIdx
          const isActive  = idx === stageIdx
          const isLast    = idx === STAGES.length - 1

          return (
            <div key={stage.key} className="flex items-start gap-3.5">
              {/* Line + dot */}
              <div className="flex flex-col items-center w-5 shrink-0">
                <div className={cn(
                  'w-3 h-3 rounded-full mt-1 shrink-0 transition-all duration-300 flex items-center justify-center',
                  isDone && 'bg-accent',
                  isActive && 'bg-accent animate-pulse-dot',
                  !isDone && !isActive && 'bg-surface-4 border border-edge-2',
                )}>
                  {isDone && <Check size={8} strokeWidth={3} className="text-white" />}
                </div>
                {!isLast && (
                  <div className={cn(
                    'w-px flex-1 min-h-[28px] transition-colors duration-300',
                    isDone ? 'bg-accent/40' : 'bg-edge',
                  )} />
                )}
              </div>

              {/* Label */}
              <div className={cn(!isLast && 'pb-4')}>
                <p className={cn(
                  'text-[13px] transition-colors duration-300',
                  isDone && 'text-fg-3',
                  isActive && 'text-fg font-medium',
                  !isDone && !isActive && 'text-fg-3',
                )}>
                  {stage.label}
                </p>
                {isActive && (
                  <p className="text-xs text-fg-3 mt-0.5">
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
