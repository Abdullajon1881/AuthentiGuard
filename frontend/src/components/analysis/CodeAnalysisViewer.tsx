'use client'

// Step 77: Code Analysis Viewer.
// Displays AI detection results for submitted code with syntax-highlighted
// snippet preview, feature breakdown, and naming pattern evidence.

import { useState } from 'react'
import type { DetectionResult } from '@/types'

interface Props {
  result: DetectionResult
  code?:  string   // original submitted code (for preview)
}

type Tab = 'signals' | 'naming' | 'style'

function FeatureRow({
  label, value, isAiSignal, note,
}: {
  label: string
  value: string | number
  isAiSignal: boolean
  note?: string
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 10,
      padding: '7px 12px',
      background: 'var(--bg-3)',
      borderRadius: 'var(--radius)',
      borderLeft: `3px solid ${isAiSignal ? 'var(--ai)' : 'var(--border)'}`,
    }}>
      <span style={{ flex: 1, fontSize: 12, color: 'var(--text-2)' }}>{label}</span>
      <span style={{
        fontFamily: 'var(--font-mono)', fontSize: 11,
        color: isAiSignal ? 'var(--ai)' : 'var(--text-3)',
        fontWeight: isAiSignal ? 500 : 400,
      }}>
        {value}
      </span>
      {note && (
        <span style={{ fontSize: 10, color: 'var(--text-3)', maxWidth: 120, textAlign: 'right' }}>
          {note}
        </span>
      )}
    </div>
  )
}

function CodePreview({ code }: { code: string }) {
  const lines  = code.split('\n').slice(0, 30)
  const hasMore = code.split('\n').length > 30

  return (
    <div style={{
      background: 'var(--bg)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden',
    }}>
      {/* Header bar */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6,
        padding: '8px 14px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-3)',
      }}>
        {['var(--ai)', 'var(--uncertain)', 'var(--human)'].map((c, i) => (
          <div key={i} style={{
            width: 10, height: 10, borderRadius: '50%', background: c, opacity: 0.6,
          }} />
        ))}
        <span style={{
          marginLeft: 6, fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)',
        }}>
          submitted code
        </span>
      </div>
      {/* Code body */}
      <pre style={{
        margin: 0, padding: '14px 16px',
        fontFamily: 'var(--font-mono)',
        fontSize: 12, lineHeight: 1.65,
        color: 'var(--text-2)',
        overflowX: 'auto',
        maxHeight: 280,
        overflowY: 'auto',
      }}>
        {lines.map((line, i) => (
          <div key={i}>
            <span style={{ color: 'var(--text-3)', userSelect: 'none', marginRight: 16, fontSize: 10 }}>
              {String(i + 1).padStart(3, ' ')}
            </span>
            {line}
          </div>
        ))}
        {hasMore && (
          <div style={{ color: 'var(--text-3)', fontSize: 11, marginTop: 8 }}>
            … {code.split('\n').length - 30} more lines
          </div>
        )}
      </pre>
    </div>
  )
}

export default function CodeAnalysisViewer({ result, code }: Props) {
  const [tab, setTab] = useState<Tab>('signals')
  const ev = result.evidence_summary ?? {}

  // Build signal rows from evidence
  const astSignals = [
    { label: 'Function count',        value: ev.n_functions ?? '—',          isAI: false, note: 'structural' },
    { label: 'Fn length uniformity',  value: ev.fn_length_cv != null ? (ev.fn_length_cv as number) < 0.4 ? 'LOW ⚠' : 'normal' : '—', isAI: (ev.fn_length_cv as number) < 0.4, note: 'low = AI-like' },
    { label: 'Max nesting depth',     value: ev.max_nesting_depth ?? '—',     isAI: false, note: 'structural' },
    { label: 'Docstring coverage',    value: ev.docstring_coverage != null ? `${Math.round((ev.docstring_coverage as number)*100)}%` : '—', isAI: (ev.docstring_coverage as number) > 0.85, note: '>85% = AI-like' },
    { label: 'TODO/FIXME markers',    value: ev.informal_markers ?? 0,        isAI: (ev.informal_markers as number) === 0, note: 'none = AI-like' },
    { label: 'Magic numbers',         value: ev.n_magic_numbers ?? '—',       isAI: (ev.n_magic_numbers as number) === 0, note: 'none = AI-like' },
    { label: 'Commented-out code',    value: ev.n_commented_code ?? '—',      isAI: false, note: 'human signal' },
  ]

  const namingSignals = [
    { label: 'Mean name length',      value: ev.mean_name_length != null ? `${(ev.mean_name_length as number).toFixed(1)} chars` : '—', isAI: (ev.mean_name_length as number) > 9, note: '>9 = AI-like' },
    { label: 'Abbreviation rate',     value: ev.abbreviation_rate != null ? `${Math.round((ev.abbreviation_rate as number)*100)}%` : '—', isAI: (ev.abbreviation_rate as number) < 0.05, note: '<5% = AI-like' },
    { label: 'Convention consistency',value: ev.convention_consistency != null ? `${Math.round((ev.convention_consistency as number)*100)}%` : '—', isAI: (ev.convention_consistency as number) > 0.90, note: '>90% = AI-like' },
    { label: 'Single-letter names',   value: ev.single_letter_rate != null ? `${Math.round((ev.single_letter_rate as number)*100)}%` : '—', isAI: false, note: 'human signal' },
    { label: 'Numeric suffixes',      value: ev.numeric_suffix_rate != null ? `${Math.round((ev.numeric_suffix_rate as number)*100)}%` : '—', isAI: (ev.numeric_suffix_rate as number) < 0.01, note: 'none = AI-like' },
  ]

  const styleSignals = [
    { label: 'Comment capitalisation', value: ev.capitalisation_rate != null ? `${Math.round((ev.capitalisation_rate as number)*100)}%` : '—', isAI: (ev.capitalisation_rate as number) > 0.90, note: '>90% = AI-like' },
    { label: 'Sentence-ended comments',value: ev.ends_with_period_rate != null ? `${Math.round((ev.ends_with_period_rate as number)*100)}%` : '—', isAI: (ev.ends_with_period_rate as number) > 0.80, note: '>80% = AI-like' },
    { label: 'Docstring format',       value: ev.google_style_ratio != null ? `Google ${Math.round((ev.google_style_ratio as number)*100)}%` : '—', isAI: (ev.format_consistency as number) > 0.85, note: 'consistent = AI' },
    { label: 'Informal markers',       value: ev.informal_marker_count ?? 0,  isAI: (ev.informal_marker_count as number) === 0, note: 'none = AI-like' },
  ]

  const tabSignals = { signals: astSignals, naming: namingSignals, style: styleSignals }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Verdict banner */}
      <div style={{
        padding: '10px 16px',
        borderRadius: 'var(--radius)',
        background: result.label === 'AI'    ? 'var(--ai-dim)'       :
                    result.label === 'HUMAN' ? 'var(--human-dim)'    :
                                               'var(--uncertain-dim)',
        border: `1px solid ${
          result.label === 'AI' ? 'var(--ai)' : result.label === 'HUMAN' ? 'var(--human)' : 'var(--uncertain)'
        }`,
        display: 'flex', alignItems: 'center', gap: 12,
      }}>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 500,
          color: result.label === 'AI' ? 'var(--ai)' : result.label === 'HUMAN' ? 'var(--human)' : 'var(--uncertain)',
        }}>
          {Math.round(result.authenticity_score * 100)}%
        </span>
        <div>
          <p style={{ margin: 0, fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>
            {result.label === 'AI'    ? 'AI-generated code detected' :
             result.label === 'HUMAN' ? 'Likely human-written code' :
                                         'Inconclusive — mixed signals'}
          </p>
          <p style={{ margin: 0, fontSize: 11, color: 'var(--text-3)' }}>
            {Math.round(result.confidence * 100)}% confidence · {result.content_type}
          </p>
        </div>
      </div>

      {/* Code preview */}
      {code && <CodePreview code={code} />}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
        {(['signals', 'naming', 'style'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '5px 14px', fontSize: 11,
              fontFamily: 'var(--font-mono)', letterSpacing: '0.06em',
              borderRadius: 'var(--radius)',
              border: tab === t ? '1px solid var(--teal)' : '1px solid var(--border-2)',
              background: tab === t ? 'var(--teal-dim)' : 'transparent',
              color: tab === t ? 'var(--teal)' : 'var(--text-3)',
              cursor: 'pointer',
            }}
          >
            {t.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Feature rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}
        aria-label={`${tab} feature breakdown`}>
        {tabSignals[tab].map(s => (
          <FeatureRow
            key={s.label}
            label={s.label}
            value={s.value}
            isAiSignal={s.isAI}
            note={s.note}
          />
        ))}
      </div>

      {/* Sample names from naming tab */}
      {tab === 'naming' && ev.sample_names && (ev.sample_names as string[]).length > 0 && (
        <div style={{
          padding: '12px 14px',
          background: 'var(--bg-3)',
          borderRadius: 'var(--radius)',
          border: '1px solid var(--border)',
        }}>
          <p style={{ fontSize: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.08em',
            textTransform: 'uppercase', color: 'var(--text-3)', margin: '0 0 8px' }}>
            Sample identifiers
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {(ev.sample_names as string[]).slice(0, 15).map((name: string) => (
              <span key={name} style={{
                padding: '2px 8px',
                background: 'var(--bg-4)',
                borderRadius: 3,
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: name.length > 9 ? 'var(--ai)' : 'var(--text-2)',
              }}>
                {name}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
