'use client'

import { useState } from 'react'
import { cn } from '@/lib/cn'
import type { DetectionResult } from '@/types'

interface ImageSignal {
  name:   string
  value:  number
  label:  string
  weight: 'high' | 'medium' | 'low'
}

interface Props {
  result: DetectionResult
  imageUrl?: string
}

function SignalBar({ name, value, weight, label }: ImageSignal) {
  const pct = Math.round(value * 100)
  const color =
    weight === 'high'   ? 'var(--ai)' :
    weight === 'medium' ? 'var(--uncertain)' :
                          'var(--text-3)'
  return (
    <div className="flex items-center gap-3">
      <span className="font-mono text-[11px] text-fg-3 min-w-[180px] text-right">{name}</span>
      <div className="flex-1 h-1.5 bg-surface-4 rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-600" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="font-mono text-[11px] min-w-[40px] text-right" style={{ color }}>{pct}%</span>
      <span className="font-mono text-[10px] text-fg-3 min-w-[60px]">{label}</span>
    </div>
  )
}

function FrequencyViz({ fftGridScore, fftHighFreqRatio }: { fftGridScore: number; fftHighFreqRatio: number }) {
  const SIZE = 120, cx = SIZE / 2, cy = SIZE / 2
  const rings = [0.15, 0.30, 0.45, 0.60, 0.75].map((r, i) => ({ r: r * (SIZE / 2), opacity: 0.1 + i * 0.08 }))
  const gridMarkers = fftGridScore > 0.3
    ? [{ x: cx + SIZE * 0.25, y: cy }, { x: cx - SIZE * 0.25, y: cy }, { x: cx, y: cy + SIZE * 0.25 }, { x: cx, y: cy - SIZE * 0.25 }]
    : []

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}
        aria-label="Frequency domain spectrum" className="bg-surface-4 rounded">
        {rings.map(({ r, opacity }, i) => (
          <circle key={i} cx={cx} cy={cy} r={r} fill="none"
            stroke={fftHighFreqRatio > 0.5 ? 'var(--ai)' : 'var(--accent)'} strokeWidth={1} opacity={opacity} />
        ))}
        {gridMarkers.map((m, i) => (
          <circle key={i} cx={m.x} cy={m.y} r={4} fill="var(--ai)" opacity={fftGridScore} />
        ))}
        <circle cx={cx} cy={cy} r={4} fill="var(--accent)" />
      </svg>
      <p className="font-mono text-[10px] text-fg-3">frequency spectrum</p>
    </div>
  )
}

export default function ImageAnalysisViewer({ result, imageUrl }: Props) {
  const [tab, setTab] = useState<'signals' | 'frequency'>('signals')
  const ev = result.evidence_summary ?? {}

  const signalBars: ImageSignal[] = [
    { name: 'GAN fingerprint',     value: ev.fingerprint_correlation ?? 0, label: 'model residual',  weight: (ev.fingerprint_correlation ?? 0) > 0.5 ? 'high' : 'medium' },
    { name: 'FFT grid artifact',   value: ev.fft_grid_score ?? 0,         label: 'checkerboard',    weight: (ev.fft_grid_score ?? 0) > 0.4 ? 'high' : 'low' },
    { name: 'Over-smoothed HF',    value: ev.fft_high_freq_ratio ?? 0,    label: 'no camera noise', weight: (ev.fft_high_freq_ratio ?? 0) > 0.5 ? 'high' : 'medium' },
    { name: 'Texture uniformity',  value: ev.texture_uniformity ?? 0,      label: 'GLCM analysis',   weight: (ev.texture_uniformity ?? 0) > 0.6 ? 'medium' : 'low' },
    { name: 'Bilateral symmetry',  value: ev.bilateral_symmetry ?? 0,      label: 'StyleGAN signal', weight: (ev.bilateral_symmetry ?? 0) > 0.85 ? 'medium' : 'low' },
    { name: 'Background pattern',  value: ev.background_uniformity ?? 0,   label: 'diffusion signal',weight: (ev.background_uniformity ?? 0) > 0.7 ? 'medium' : 'low' },
  ].filter(s => s.value > 0)

  const labelColor = result.label === 'AI' ? 'var(--ai)' : result.label === 'HUMAN' ? 'var(--human)' : 'var(--uncertain)'
  const labelBg = result.label === 'AI' ? 'var(--ai-dim)' : result.label === 'HUMAN' ? 'var(--human-dim)' : 'var(--uncertain-dim)'

  return (
    <div className="flex flex-col gap-4">
      {/* Image preview + verdict */}
      <div className="flex gap-4 items-start flex-wrap">
        {imageUrl && (
          <div className="w-[120px] h-[120px] rounded-lg border border-edge overflow-hidden shrink-0 relative">
            <img src={imageUrl} alt="Analysed image" className="w-full h-full object-cover" />
            <div className="absolute bottom-1 right-1 bg-black/75 rounded px-1.5 py-0.5 font-mono text-[11px]" style={{ color: labelColor }}>
              {Math.round(result.authenticity_score * 100)}%
            </div>
          </div>
        )}
        <div className="flex-1">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full font-mono text-[11px] font-medium mb-2.5 border"
            style={{ color: labelColor, background: labelBg, borderColor: labelColor }}>
            {result.label === 'AI' ? 'AI-generated image detected' : result.label === 'HUMAN' ? 'Likely authentic photograph' : 'Uncertain — inconclusive signals'}
          </div>
          <p className="text-xs text-fg-3">
            {result.label === 'AI' && 'This image exhibits characteristic patterns of AI generation — GAN fingerprints, frequency domain artifacts, or abnormal texture statistics.'}
            {result.label === 'HUMAN' && 'No significant AI generation artifacts detected. The image shows characteristics consistent with a real photograph.'}
            {result.label === 'UNCERTAIN' && 'Some signals are present but below confidence thresholds. Manual review recommended.'}
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="segment-control">
        {(['signals', 'frequency'] as const).map(t => (
          <button key={t} onClick={() => setTab(t)}
            className={cn(tab === t ? 'segment-btn-active' : 'segment-btn')}>
            {t}
          </button>
        ))}
      </div>

      {/* Signals */}
      {tab === 'signals' && (
        <div className="flex flex-col gap-2" aria-label="Detection signal breakdown">
          {signalBars.length > 0
            ? signalBars.map(s => <SignalBar key={s.name} {...s} />)
            : <p className="text-sm text-fg-3 text-center py-4">No significant signals detected</p>}
        </div>
      )}

      {/* Frequency */}
      {tab === 'frequency' && (
        <div className="flex gap-5 items-start flex-wrap">
          <FrequencyViz fftGridScore={ev.fft_grid_score ?? 0} fftHighFreqRatio={ev.fft_high_freq_ratio ?? 0} />
          <div className="flex-1 flex flex-col gap-2">
            {[
              { label: 'Grid artifacts',     value: ev.fft_grid_score ?? 0,         desc: 'GAN checkerboard pattern' },
              { label: 'HF suppression',     value: ev.fft_high_freq_ratio ?? 0,    desc: 'Missing camera noise' },
              { label: 'Spectral regularity',value: ev.fft_peak_regularity ?? 0,    desc: 'Periodic spectral peaks' },
              { label: 'Azimuthal variance', value: ev.fft_azimuthal_variance ?? 0, desc: 'Directional asymmetry' },
            ].map(({ label, value, desc }) => (
              <div key={label} className="flex items-center gap-3 py-1.5 px-3 bg-surface-3 rounded border-b border-edge last:border-0">
                <span className="text-xs text-fg-2 flex-1">{label}</span>
                <span className="text-[11px] text-fg-3">{desc}</span>
                <span className={cn('font-mono text-[11px] min-w-[40px] text-right', value > 0.5 ? 'text-ai' : 'text-fg-3')}>
                  {Math.round(value * 100)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
