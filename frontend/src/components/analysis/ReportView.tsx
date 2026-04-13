'use client'

import { Download, ArrowRight } from 'lucide-react'
import type { DetectionResult, Label } from '@/types'
import EvidencePanel from '../evidence/EvidencePanel'

/* ── Score dial ────────────────────────────────────────────── */

function ScoreDial({ score, label }: { score: number; label: Label }) {
  const pct = Math.round(score * 100)
  const labelColor =
    label === 'AI'    ? 'var(--ai)' :
    label === 'HUMAN' ? 'var(--human)' :
                        'var(--uncertain)'
  const labelBg =
    label === 'AI'    ? 'var(--ai-dim)' :
    label === 'HUMAN' ? 'var(--human-dim)' :
                        'var(--uncertain-dim)'

  const r = 54, cx = 64, cy = 64
  const circ = 2 * Math.PI * r
  const filled = circ * score
  const gap = circ - filled

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative w-[128px] h-[128px]">
        <svg width="128" height="128" viewBox="0 0 128 128" aria-hidden>
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--bg-4)" strokeWidth="7" />
          <circle
            cx={cx} cy={cy} r={r}
            fill="none" stroke={labelColor} strokeWidth="7"
            strokeLinecap="round"
            strokeDasharray={`${filled} ${gap}`}
            strokeDashoffset={circ * 0.25}
            style={{ transition: 'stroke-dasharray 0.7s cubic-bezier(0.34,1.56,0.64,1)' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-serif text-[28px] font-bold leading-none" style={{ color: labelColor }}>
            {pct}
          </span>
          <span className="section-label mt-1 text-[9px]">AI Score</span>
        </div>
      </div>
      <div
        className="px-4 py-1.5 rounded-full font-mono text-[11px] font-medium tracking-[0.08em] border"
        style={{ color: labelColor, background: labelBg, borderColor: labelColor }}
      >
        {label}
      </div>
    </div>
  )
}

/* ── Layer score bar ───────────────────────────────────────── */

function LayerBar({ label, score, color }: { label: string; score: number | null; color: string }) {
  if (score === null) return null
  const pct = Math.round(score * 100)
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[11px] text-fg-3 min-w-[100px] text-right">{label}</span>
      <div className="flex-1 h-1.5 bg-surface-4 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-600"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="font-mono text-[11px] text-fg-3 min-w-[34px] text-right">{pct}%</span>
    </div>
  )
}

/* ── Attribution bar ───────────────────────────────────────── */

function AttributionBar({ attribution }: { attribution: DetectionResult['model_attribution'] }) {
  const entries = [
    { key: 'gpt_family',    label: 'GPT',    color: 'var(--layer-1)' },
    { key: 'claude_family', label: 'Claude', color: 'var(--layer-2)' },
    { key: 'llama_family',  label: 'LLaMA',  color: 'var(--human)' },
    { key: 'human',         label: 'Human',  color: 'var(--text-2)' },
    { key: 'other',         label: 'Other',  color: 'var(--text-3)' },
  ] as const

  return (
    <div>
      <h3 className="section-label mb-3">Model attribution</h3>
      <div className="flex h-2 rounded overflow-hidden gap-px mb-3">
        {entries.map(({ key, color }) => {
          const pct = (attribution[key] ?? 0) * 100
          return pct > 0.5 ? (
            <div key={key} className="transition-all duration-500" style={{ width: `${pct}%`, background: color }} />
          ) : null
        })}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {entries.map(({ key, label, color }) => {
          const pct = Math.round((attribution[key] ?? 0) * 100)
          if (pct < 1) return null
          return (
            <div key={key} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-sm" style={{ background: color }} />
              <span className="text-[11px] text-fg-3">{label} {pct}%</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── Stat card ─────────────────────────────────────────────── */

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-edge rounded-lg p-4">
      <p className="section-label mb-2">{label}</p>
      <p className="font-serif text-xl font-bold text-fg leading-none">{value}</p>
    </div>
  )
}

/* ── Main ──────────────────────────────────────────────────── */

interface Props {
  result: DetectionResult
  onNewAnalysis: () => void
}

export default function ReportView({ result, onNewAnalysis }: Props) {
  const processingStr = result.processing_ms
    ? result.processing_ms < 1000
      ? `${result.processing_ms}ms`
      : `${(result.processing_ms / 1000).toFixed(1)}s`
    : '\u2014'

  const confidence = Math.round(result.confidence * 100)

  return (
    <div className="animate-fade-up stagger flex flex-col gap-6">

      {/* Score + meta row */}
      <div className="flex gap-6 items-start flex-wrap">
        <ScoreDial score={result.authenticity_score} label={result.label} />
        <div className="flex-1 min-w-[200px]">
          <div className="grid grid-cols-2 gap-3">
            <StatCard label="Confidence"   value={`${confidence}%`} />
            <StatCard label="Processed in" value={processingStr} />
            <StatCard label="Content type" value={result.content_type.toUpperCase()} />
            <StatCard label="Job ID"       value={result.job_id.slice(0, 8) + '\u2026'} />
          </div>
        </div>
      </div>

      {/* Layer scores */}
      <div className="card p-5">
        <h3 className="section-label mb-4">Detection layers</h3>
        <div className="flex flex-col gap-3">
          <LayerBar label="Perplexity"  score={result.layer_scores.perplexity}  color="var(--layer-1)" />
          <LayerBar label="Stylometry"  score={result.layer_scores.stylometry}  color="var(--layer-2)" />
          <LayerBar label="Transformer" score={result.layer_scores.transformer} color="var(--layer-3)" />
          <LayerBar label="Adversarial" score={result.layer_scores.adversarial} color="var(--layer-4)" />
        </div>
      </div>

      {/* Model attribution */}
      <div className="card p-5">
        <AttributionBar attribution={result.model_attribution} />
      </div>

      {/* Evidence panel */}
      <div className="card p-5">
        <EvidencePanel
          sentenceScores={result.sentence_scores}
          topSignals={result.top_signals}
          overallScore={result.authenticity_score}
        />
      </div>

      {/* Actions */}
      <div className="flex gap-3">
        <button onClick={onNewAnalysis} className="btn-primary">
          New analysis <ArrowRight size={14} />
        </button>
        {result.report_url && (
          <a
            href={`${process.env.NEXT_PUBLIC_API_URL}${result.report_url}`}
            download
            className="btn-secondary"
            aria-label="Download forensic report"
          >
            <Download size={14} />
            Download report
          </a>
        )}
      </div>
    </div>
  )
}
