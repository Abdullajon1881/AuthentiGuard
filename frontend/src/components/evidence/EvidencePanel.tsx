'use client'

import { useState } from 'react'
import { cn } from '@/lib/cn'
import type { EvidenceSignal, SentenceScore } from '@/types'

function scoreToColor(score: number): string {
  if (score >= 0.80) return 'rgba(196, 75, 55, 0.18)'
  if (score >= 0.65) return 'rgba(196,154, 47, 0.14)'
  if (score >= 0.50) return 'rgba(196,154, 47, 0.06)'
  return 'transparent'
}

function scoreToLabel(score: number): string {
  if (score >= 0.80) return 'AI'
  if (score >= 0.65) return 'Likely AI'
  if (score >= 0.50) return 'Uncertain'
  return 'Human'
}

function scoreToTextColor(score: number): string {
  if (score >= 0.80) return 'var(--ai)'
  if (score >= 0.65) return 'var(--uncertain)'
  if (score >= 0.50) return 'var(--text-3)'
  return 'var(--human)'
}

const WEIGHT_COLOR: Record<string, string> = {
  high:   'var(--ai)',
  medium: 'var(--uncertain)',
  low:    'var(--text-3)',
}

interface Props {
  sentenceScores: SentenceScore[]
  topSignals:     EvidenceSignal[]
  overallScore:   number
}

export default function EvidencePanel({ sentenceScores, topSignals, overallScore }: Props) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null)

  const hasSentences = sentenceScores.length > 0
  const hasSignals   = topSignals.length > 0

  return (
    <div className="flex flex-col gap-5">

      {/* Top signals */}
      {hasSignals && (
        <section aria-label="Detection signals">
          <h3 className="section-label mb-3">Key signals</h3>
          <div className="flex flex-col gap-1.5">
            {topSignals.map((sig, i) => (
              <div
                key={i}
                className="flex items-center gap-3 py-2 px-3 border-b border-edge last:border-0"
              >
                <span
                  className="w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ background: WEIGHT_COLOR[sig.weight] ?? 'var(--text-3)' }}
                />
                <span className="text-sm text-fg-2 flex-1">{sig.signal}</span>
                <span
                  className="font-mono text-[11px] px-2 py-0.5 rounded bg-surface-4"
                  style={{ color: WEIGHT_COLOR[sig.weight] ?? 'var(--text-3)' }}
                >
                  {sig.value}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Per-sentence highlight */}
      {hasSentences && (
        <section aria-label="Per-sentence analysis">
          <h3 className="section-label mb-3">Sentence breakdown</h3>

          {/* Legend */}
          <div className="flex flex-wrap gap-x-4 gap-y-1.5 mb-4">
            {[
              { color: 'rgba(196,75,55,0.18)',  label: 'AI (\u226580%)' },
              { color: 'rgba(196,154,47,0.14)', label: 'Likely AI (65\u201380%)' },
              { color: 'rgba(196,154,47,0.06)', label: 'Uncertain (50\u201365%)' },
              { color: 'transparent',           label: 'Human (<50%)', border: true },
            ].map(({ color, label, border }) => (
              <div key={label} className="flex items-center gap-1.5">
                <div
                  className="w-3 h-3 rounded-sm"
                  style={{
                    background: color,
                    border: border ? '1px solid var(--border-2)' : 'none',
                  }}
                />
                <span className="text-[11px] text-fg-3">{label}</span>
              </div>
            ))}
          </div>

          {/* Sentences */}
          <div
            className="leading-[1.85] text-sm text-fg-2 p-4 bg-surface-3 rounded-lg border border-edge"
            role="list"
            aria-label="Sentences with AI probability scores"
          >
            {sentenceScores.map((s, i) => (
              <span
                key={i}
                role="listitem"
                aria-label={`Sentence ${i + 1}: ${scoreToLabel(s.score)} (${Math.round(s.score * 100)}%)`}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx(null)}
                title={`${scoreToLabel(s.score)} \u2014 ${Math.round(s.score * 100)}% AI`}
                className="inline rounded-sm px-0.5 cursor-default transition-all duration-200"
                style={{
                  background: scoreToColor(s.score),
                  outline: hoveredIdx === i ? `1px solid ${scoreToTextColor(s.score)}` : 'none',
                  outlineOffset: 1,
                }}
              >
                {s.text}{' '}
              </span>
            ))}
          </div>

          {/* Hovered sentence tooltip */}
          {hoveredIdx !== null && sentenceScores[hoveredIdx] && (
            <div className="mt-2.5 py-2 px-3.5 bg-surface-4 rounded border border-edge-2 text-xs flex items-center gap-3">
              <span className="text-fg-3 font-mono">
                sentence {hoveredIdx + 1}
              </span>
              <span className="font-medium" style={{ color: scoreToTextColor(sentenceScores[hoveredIdx].score) }}>
                {scoreToLabel(sentenceScores[hoveredIdx].score)}
              </span>
              <span className="text-fg-3 font-mono ml-auto">
                {Math.round(sentenceScores[hoveredIdx].score * 100)}% AI probability
              </span>
            </div>
          )}
        </section>
      )}

      {!hasSentences && !hasSignals && (
        <p className="text-fg-3 text-sm text-center py-6">
          No sentence-level evidence available for this content type.
        </p>
      )}
    </div>
  )
}
