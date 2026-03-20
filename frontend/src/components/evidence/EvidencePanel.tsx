'use client'

// Step 40: Evidence Panel for text results.
// Shows per-sentence AI probability as inline background highlights,
// with a sidebar showing confidence indicators and top signals.

import { useState } from 'react'
import type { EvidenceSignal, SentenceScore } from '@/types'

// Colour-maps a [0,1] AI score to a CSS colour for sentence highlighting
function scoreToColor(score: number): string {
  if (score >= 0.80) return 'rgba(248, 113, 113, 0.22)'   // red   — very likely AI
  if (score >= 0.65) return 'rgba(251, 191,  36, 0.18)'   // amber — probably AI
  if (score >= 0.50) return 'rgba(251, 191,  36, 0.08)'   // faint amber — uncertain
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* ── Top signals ─────────────────────────────────────── */}
      {hasSignals && (
        <section aria-label="Detection signals">
          <h3 style={{ fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: '0.10em',
            textTransform: 'uppercase', color: 'var(--text-3)', marginBottom: 10 }}>
            Key signals
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {topSignals.map((sig, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '8px 12px',
                background: 'var(--bg-3)',
                borderRadius: 'var(--radius)',
                border: '1px solid var(--border)',
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                  background: WEIGHT_COLOR[sig.weight] ?? 'var(--text-3)',
                }} />
                <span style={{ fontSize: 13, color: 'var(--text-2)', flex: 1 }}>
                  {sig.signal}
                </span>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11,
                  color: WEIGHT_COLOR[sig.weight] ?? 'var(--text-3)',
                  background: 'var(--bg-4)',
                  padding: '2px 6px', borderRadius: 3,
                }}>
                  {sig.value}
                </span>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Per-sentence highlight ───────────────────────────── */}
      {hasSentences && (
        <section aria-label="Per-sentence analysis">
          <h3 style={{ fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: '0.10em',
            textTransform: 'uppercase', color: 'var(--text-3)', marginBottom: 10 }}>
            Sentence breakdown
          </h3>

          {/* Legend */}
          <div style={{ display: 'flex', gap: 16, marginBottom: 14, flexWrap: 'wrap' }}>
            {[
              { color: 'rgba(248,113,113,0.22)', label: 'AI (≥80%)' },
              { color: 'rgba(251,191,36,0.18)',  label: 'Likely AI (65–80%)' },
              { color: 'rgba(251,191,36,0.08)',  label: 'Uncertain (50–65%)' },
              { color: 'transparent',            label: 'Human (<50%)', border: true },
            ].map(({ color, label, border }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div style={{
                  width: 12, height: 12, borderRadius: 3,
                  background: color,
                  border: border ? '1px solid var(--border-2)' : 'none',
                }} />
                <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{label}</span>
              </div>
            ))}
          </div>

          {/* Sentences */}
          <div
            style={{
              lineHeight: 1.85,
              fontSize: 14,
              color: 'var(--text-2)',
              padding: '16px',
              background: 'var(--bg-3)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border)',
            }}
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
                title={`${scoreToLabel(s.score)} — ${Math.round(s.score * 100)}% AI`}
                style={{
                  display: 'inline',
                  background: scoreToColor(s.score),
                  borderRadius: 3,
                  padding: '1px 2px',
                  cursor: 'default',
                  transition: 'background 0.2s',
                  outline: hoveredIdx === i ? `1px solid ${scoreToTextColor(s.score)}` : 'none',
                  outlineOffset: 1,
                  position: 'relative',
                }}
              >
                {s.text}{' '}
              </span>
            ))}
          </div>

          {/* Hovered sentence tooltip */}
          {hoveredIdx !== null && sentenceScores[hoveredIdx] && (
            <div style={{
              marginTop: 10,
              padding: '8px 14px',
              background: 'var(--bg-4)',
              borderRadius: 'var(--radius)',
              border: '1px solid var(--border-2)',
              fontSize: 12,
              display: 'flex', alignItems: 'center', gap: 12,
            }}>
              <span style={{ color: 'var(--text-3)', fontFamily: 'var(--font-mono)' }}>
                sentence {hoveredIdx + 1}
              </span>
              <span style={{
                color: scoreToTextColor(sentenceScores[hoveredIdx].score),
                fontWeight: 500,
              }}>
                {scoreToLabel(sentenceScores[hoveredIdx].score)}
              </span>
              <span style={{ color: 'var(--text-3)', fontFamily: 'var(--font-mono)', marginLeft: 'auto' }}>
                {Math.round(sentenceScores[hoveredIdx].score * 100)}% AI probability
              </span>
            </div>
          )}
        </section>
      )}

      {!hasSentences && !hasSignals && (
        <p style={{ color: 'var(--text-3)', fontSize: 13, textAlign: 'center', padding: '24px 0' }}>
          No sentence-level evidence available for this content type.
        </p>
      )}
    </div>
  )
}
