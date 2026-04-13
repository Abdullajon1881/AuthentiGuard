'use client'

import { useState } from 'react'
import { cn } from '@/lib/cn'
import type { DetectionResult } from '@/types'

interface Props {
  result: DetectionResult
  code?:  string
}

type Tab = 'signals' | 'naming' | 'style'

function FeatureRow({ label, value, isAiSignal, note }: {
  label: string; value: string | number; isAiSignal: boolean; note?: string
}) {
  return (
    <div className={cn(
      'flex items-center gap-3 py-2 px-3 border-b border-edge last:border-0',
      'hover:bg-surface-3/50 transition-colors duration-100',
    )}>
      <span className={cn('w-1 h-4 rounded-full shrink-0', isAiSignal ? 'bg-ai' : 'bg-edge')} />
      <span className="flex-1 text-xs text-fg-2">{label}</span>
      <span className={cn(
        'font-mono text-[11px]',
        isAiSignal ? 'text-ai font-medium' : 'text-fg-3',
      )}>
        {value}
      </span>
      {note && <span className="text-[10px] text-fg-3 max-w-[120px] text-right">{note}</span>}
    </div>
  )
}

function CodePreview({ code }: { code: string }) {
  const lines = code.split('\n').slice(0, 30)
  const hasMore = code.split('\n').length > 30

  return (
    <div className="bg-surface border border-edge rounded-lg overflow-hidden">
      <div className="flex items-center gap-1.5 px-3.5 py-2 border-b border-edge bg-surface-3">
        {['bg-ai', 'bg-uncertain', 'bg-human'].map((c, i) => (
          <div key={i} className={cn('w-2.5 h-2.5 rounded-full opacity-60', c)} />
        ))}
        <span className="ml-1.5 font-mono text-[10px] text-fg-3">submitted code</span>
      </div>
      <pre className="m-0 px-4 py-3.5 font-mono text-xs leading-relaxed text-fg-2 overflow-auto max-h-[280px]">
        {lines.map((line, i) => (
          <div key={i}>
            <span className="text-fg-3 select-none mr-4 text-[10px]">
              {String(i + 1).padStart(3, ' ')}
            </span>
            {line}
          </div>
        ))}
        {hasMore && (
          <div className="text-fg-3 text-[11px] mt-2">
            \u2026 {code.split('\n').length - 30} more lines
          </div>
        )}
      </pre>
    </div>
  )
}

export default function CodeAnalysisViewer({ result, code }: Props) {
  const [tab, setTab] = useState<Tab>('signals')
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ev = (result.evidence_summary ?? {}) as Record<string, any>

  const labelColor = result.label === 'AI' ? 'var(--ai)' : result.label === 'HUMAN' ? 'var(--human)' : 'var(--uncertain)'
  const labelBg = result.label === 'AI' ? 'var(--ai-dim)' : result.label === 'HUMAN' ? 'var(--human-dim)' : 'var(--uncertain-dim)'

  const astSignals = [
    { label: 'Function count',       value: ev.n_functions ?? '\u2014',          isAI: false, note: 'structural' },
    { label: 'Fn length uniformity', value: ev.fn_length_cv != null ? (ev.fn_length_cv as number) < 0.4 ? 'LOW' : 'normal' : '\u2014', isAI: (ev.fn_length_cv as number) < 0.4, note: 'low = AI-like' },
    { label: 'Max nesting depth',    value: ev.max_nesting_depth ?? '\u2014',     isAI: false, note: 'structural' },
    { label: 'Docstring coverage',   value: ev.docstring_coverage != null ? `${Math.round((ev.docstring_coverage as number)*100)}%` : '\u2014', isAI: (ev.docstring_coverage as number) > 0.85, note: '>85% = AI-like' },
    { label: 'TODO/FIXME markers',   value: ev.informal_markers ?? 0,        isAI: (ev.informal_markers as number) === 0, note: 'none = AI-like' },
    { label: 'Magic numbers',        value: ev.n_magic_numbers ?? '\u2014',       isAI: (ev.n_magic_numbers as number) === 0, note: 'none = AI-like' },
    { label: 'Commented-out code',   value: ev.n_commented_code ?? '\u2014',      isAI: false, note: 'human signal' },
  ]

  const namingSignals = [
    { label: 'Mean name length',       value: ev.mean_name_length != null ? `${(ev.mean_name_length as number).toFixed(1)} chars` : '\u2014', isAI: (ev.mean_name_length as number) > 9, note: '>9 = AI-like' },
    { label: 'Abbreviation rate',      value: ev.abbreviation_rate != null ? `${Math.round((ev.abbreviation_rate as number)*100)}%` : '\u2014', isAI: (ev.abbreviation_rate as number) < 0.05, note: '<5% = AI-like' },
    { label: 'Convention consistency', value: ev.convention_consistency != null ? `${Math.round((ev.convention_consistency as number)*100)}%` : '\u2014', isAI: (ev.convention_consistency as number) > 0.90, note: '>90% = AI-like' },
    { label: 'Single-letter names',    value: ev.single_letter_rate != null ? `${Math.round((ev.single_letter_rate as number)*100)}%` : '\u2014', isAI: false, note: 'human signal' },
    { label: 'Numeric suffixes',       value: ev.numeric_suffix_rate != null ? `${Math.round((ev.numeric_suffix_rate as number)*100)}%` : '\u2014', isAI: (ev.numeric_suffix_rate as number) < 0.01, note: 'none = AI-like' },
  ]

  const styleSignals = [
    { label: 'Comment capitalisation',  value: ev.capitalisation_rate != null ? `${Math.round((ev.capitalisation_rate as number)*100)}%` : '\u2014', isAI: (ev.capitalisation_rate as number) > 0.90, note: '>90% = AI-like' },
    { label: 'Sentence-ended comments', value: ev.ends_with_period_rate != null ? `${Math.round((ev.ends_with_period_rate as number)*100)}%` : '\u2014', isAI: (ev.ends_with_period_rate as number) > 0.80, note: '>80% = AI-like' },
    { label: 'Docstring format',        value: ev.google_style_ratio != null ? `Google ${Math.round((ev.google_style_ratio as number)*100)}%` : '\u2014', isAI: (ev.format_consistency as number) > 0.85, note: 'consistent = AI' },
    { label: 'Informal markers',        value: ev.informal_marker_count ?? 0, isAI: (ev.informal_marker_count as number) === 0, note: 'none = AI-like' },
  ]

  const tabSignals = { signals: astSignals, naming: namingSignals, style: styleSignals }

  return (
    <div className="flex flex-col gap-4">
      {/* Verdict */}
      <div className="flex items-center gap-3 px-4 py-3 rounded border" style={{ background: labelBg, borderColor: labelColor }}>
        <span className="font-serif text-lg font-bold" style={{ color: labelColor }}>
          {Math.round(result.authenticity_score * 100)}%
        </span>
        <div>
          <p className="text-sm font-medium text-fg">
            {result.label === 'AI' ? 'AI-generated code detected' : result.label === 'HUMAN' ? 'Likely human-written code' : 'Inconclusive — mixed signals'}
          </p>
          <p className="text-[11px] text-fg-3">
            {Math.round(result.confidence * 100)}% confidence · {result.content_type}
          </p>
        </div>
      </div>

      {code && <CodePreview code={code} />}

      {/* Tabs */}
      <div className="segment-control">
        {(['signals', 'naming', 'style'] as Tab[]).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={cn(tab === t ? 'segment-btn-active' : 'segment-btn')}>
            {t}
          </button>
        ))}
      </div>

      {/* Feature rows */}
      <div className="flex flex-col" aria-label={`${tab} feature breakdown`}>
        {tabSignals[tab].map(s => (
          <FeatureRow key={s.label} label={s.label} value={s.value} isAiSignal={s.isAI} note={s.note} />
        ))}
      </div>

      {/* Sample names */}
      {tab === 'naming' && ev.sample_names && (ev.sample_names as string[]).length > 0 && (
        <div className="card p-3.5">
          <p className="section-label mb-2">Sample identifiers</p>
          <div className="flex flex-wrap gap-1.5">
            {(ev.sample_names as string[]).slice(0, 15).map((name: string) => (
              <span key={name} className={cn(
                'px-2 py-0.5 rounded bg-surface-4 font-mono text-[11px]',
                name.length > 9 ? 'text-ai' : 'text-fg-2',
              )}>
                {name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
