'use client'

// Step 41: Report View — the full detection result breakdown.
// Shows the Authenticity Score dial, confidence, per-layer scores,
// model attribution bar, and links to the evidence panel + report download.

import type { DetectionResult, Label } from '@/types'
import EvidencePanel from '../evidence/EvidencePanel'

// ── Score dial ────────────────────────────────────────────────

function ScoreDial({ score, label }: { score: number; label: Label }) {
  const pct = Math.round(score * 100)

  const labelColor =
    label === 'AI'        ? 'var(--ai)'       :
    label === 'HUMAN'     ? 'var(--human)'    :
                            'var(--uncertain)'

  const labelBg =
    label === 'AI'        ? 'var(--ai-dim)'       :
    label === 'HUMAN'     ? 'var(--human-dim)'    :
                            'var(--uncertain-dim)'

  // SVG arc for the score dial
  const r = 54, cx = 64, cy = 64
  const circ = 2 * Math.PI * r
  const filled = circ * score
  const gap    = circ - filled

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
      <div style={{ position: 'relative', width: 128, height: 128 }}>
        <svg width="128" height="128" viewBox="0 0 128 128" aria-hidden>
          {/* Track */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--bg-4)" strokeWidth="8" />
          {/* Fill */}
          <circle
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={labelColor}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={`${filled} ${gap}`}
            strokeDashoffset={circ * 0.25}  /* start at top */
            style={{ transition: 'stroke-dasharray 0.7s cubic-bezier(0.34,1.56,0.64,1)' }}
          />
        </svg>
        {/* Center text */}
        <div style={{
          position: 'absolute', inset: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
        }}>
          <span style={{ fontSize: 26, fontWeight: 600, fontFamily: 'var(--font-mono)', color: labelColor, lineHeight: 1 }}>
            {pct}
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-3)', letterSpacing: '0.06em', marginTop: 2 }}>
            AI SCORE
          </span>
        </div>
      </div>

      {/* Label badge */}
      <div style={{
        padding: '5px 16px',
        background: labelBg,
        border: `1px solid ${labelColor}`,
        borderRadius: 20,
        fontSize: 11,
        fontFamily: 'var(--font-mono)',
        fontWeight: 500,
        letterSpacing: '0.08em',
        color: labelColor,
      }}>
        {label}
      </div>
    </div>
  )
}

// ── Layer score bar ───────────────────────────────────────────

function LayerBar({ label, score, color }: { label: string; score: number | null; color: string }) {
  if (score === null) return null
  const pct = Math.round(score * 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{
        fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
        minWidth: 100, textAlign: 'right',
      }}>{label}</span>
      <div style={{
        flex: 1, height: 6, background: 'var(--bg-4)',
        borderRadius: 3, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: color, borderRadius: 3,
          transition: 'width 0.6s cubic-bezier(0.34,1.56,0.64,1)',
        }} />
      </div>
      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', minWidth: 34, textAlign: 'right' }}>
        {pct}%
      </span>
    </div>
  )
}

// ── Attribution bar ───────────────────────────────────────────

function AttributionBar({ attribution }: { attribution: DetectionResult['model_attribution'] }) {
  const entries = [
    { key: 'gpt_family',    label: 'GPT',    color: '#60a5fa' },
    { key: 'claude_family', label: 'Claude', color: '#a78bfa' },
    { key: 'llama_family',  label: 'LLaMA',  color: '#34d399' },
    { key: 'human',         label: 'Human',  color: '#94a3b8' },
    { key: 'other',         label: 'Other',  color: '#475569' },
  ] as const

  return (
    <div>
      <h3 style={{ fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: '0.10em',
        textTransform: 'uppercase', color: 'var(--text-3)', marginBottom: 10 }}>
        Model attribution
      </h3>
      {/* Stacked bar */}
      <div style={{ display: 'flex', height: 8, borderRadius: 4, overflow: 'hidden', gap: 1, marginBottom: 10 }}>
        {entries.map(({ key, color }) => {
          const pct = (attribution[key] ?? 0) * 100
          return pct > 0.5 ? (
            <div key={key} style={{ width: `${pct}%`, background: color, transition: 'width 0.5s ease' }} />
          ) : null
        })}
      </div>
      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 16px' }}>
        {entries.map(({ key, label, color }) => {
          const pct = Math.round((attribution[key] ?? 0) * 100)
          if (pct < 1) return null
          return (
            <div key={key} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
              <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{label} {pct}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Stat card ─────────────────────────────────────────────────

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{
      padding: '14px 16px',
      background: 'var(--bg-3)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
    }}>
      <p style={{ fontSize: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.08em',
        textTransform: 'uppercase', color: 'var(--text-3)', marginBottom: 4 }}>
        {label}
      </p>
      <p style={{ fontSize: 20, fontWeight: 500, fontFamily: 'var(--font-mono)', color: 'var(--text)', margin: 0 }}>
        {value}
      </p>
      {sub && <p style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{sub}</p>}
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────

interface Props {
  result: DetectionResult
  onNewAnalysis: () => void
}

export default function ReportView({ result, onNewAnalysis }: Props) {
  const processingStr = result.processing_ms
    ? result.processing_ms < 1000
      ? `${result.processing_ms}ms`
      : `${(result.processing_ms / 1000).toFixed(1)}s`
    : '—'

  const confidence = Math.round(result.confidence * 100)

  return (
    <div className="fade-up stagger" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* ── Score + meta row ─────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 24, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <ScoreDial score={result.authenticity_score} label={result.label} />

        <div style={{ flex: 1, minWidth: 200, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <StatCard label="Confidence"     value={`${confidence}%`} />
            <StatCard label="Processed in"   value={processingStr} />
            <StatCard label="Content type"   value={result.content_type.toUpperCase()} />
            <StatCard label="Job ID"         value={result.job_id.slice(0, 8) + '…'} />
          </div>
        </div>
      </div>

      {/* ── Layer scores ─────────────────────────────────────── */}
      <div style={{
        padding: '16px 20px',
        background: 'var(--bg-3)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
      }}>
        <h3 style={{ fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: '0.10em',
          textTransform: 'uppercase', color: 'var(--text-3)', marginBottom: 14 }}>
          Detection layers
        </h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <LayerBar label="Perplexity"   score={result.layer_scores.perplexity}  color="var(--layer-1)" />
          <LayerBar label="Stylometry"   score={result.layer_scores.stylometry}  color="var(--layer-2)" />
          <LayerBar label="Transformer"  score={result.layer_scores.transformer} color="var(--layer-3)" />
          <LayerBar label="Adversarial"  score={result.layer_scores.adversarial} color="var(--layer-4)" />
        </div>
      </div>

      {/* ── Model attribution ─────────────────────────────────── */}
      <div style={{
        padding: '16px 20px',
        background: 'var(--bg-3)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
      }}>
        <AttributionBar attribution={result.model_attribution} />
      </div>

      {/* ── Evidence panel ───────────────────────────────────── */}
      <div style={{
        padding: '16px 20px',
        background: 'var(--bg-2)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
      }}>
        <EvidencePanel
          sentenceScores={result.sentence_scores}
          topSignals={result.top_signals}
          overallScore={result.authenticity_score}
        />
      </div>

      {/* ── Actions ──────────────────────────────────────────── */}
      <div style={{ display: 'flex', gap: 10 }}>
        <button
          onClick={onNewAnalysis}
          style={{
            padding: '10px 20px',
            background: 'var(--teal)',
            color: '#000',
            border: 'none',
            borderRadius: 'var(--radius)',
            fontFamily: 'var(--font-mono)',
            fontSize: 12,
            fontWeight: 500,
            letterSpacing: '0.06em',
            cursor: 'pointer',
          }}
        >
          New analysis →
        </button>
        {result.report_url && (
          <a
            href={`${process.env.NEXT_PUBLIC_API_URL}${result.report_url}`}
            download
            style={{
              padding: '10px 20px',
              background: 'transparent',
              color: 'var(--text-2)',
              border: '1px solid var(--border-2)',
              borderRadius: 'var(--radius)',
              fontFamily: 'var(--font-mono)',
              fontSize: 12,
              cursor: 'pointer',
              textDecoration: 'none',
              display: 'inline-flex',
              alignItems: 'center',
            }}
            aria-label="Download forensic report"
          >
            Download report
          </a>
        )}
      </div>
    </div>
  )
}
