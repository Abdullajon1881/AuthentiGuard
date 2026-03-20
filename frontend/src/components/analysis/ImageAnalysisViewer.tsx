'use client'

// Step 71: Image Analysis Viewer.
// Shows the detection result for an image with a heatmap overlay,
// frequency domain visualisation, and feature breakdown.

import { useState } from 'react'
import type { DetectionResult } from '@/types'

interface ImageSignal {
  name:   string
  value:  number   // 0–1
  label:  string
  weight: 'high' | 'medium' | 'low'
}

interface Props {
  result: DetectionResult
  imageUrl?: string   // presigned URL or data URL of the analysed image
}

// Signal bar component
function SignalBar({ name, value, weight, label }: ImageSignal) {
  const pct = Math.round(value * 100)
  const color =
    weight === 'high'   ? 'var(--ai)'       :
    weight === 'medium' ? 'var(--uncertain)' :
                          'var(--text-3)'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span style={{
        fontSize: 11, color: 'var(--text-3)',
        minWidth: 180, textAlign: 'right',
        fontFamily: 'var(--font-mono)',
      }}>
        {name}
      </span>
      <div style={{
        flex: 1, height: 6,
        background: 'var(--bg-4)', borderRadius: 3, overflow: 'hidden',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: color, borderRadius: 3,
          transition: 'width 0.6s cubic-bezier(0.34,1.56,0.64,1)',
        }} />
      </div>
      <span style={{
        fontSize: 11, fontFamily: 'var(--font-mono)',
        color, minWidth: 40, textAlign: 'right',
      }}>
        {pct}%
      </span>
      <span style={{
        fontSize: 10, color: 'var(--text-3)',
        minWidth: 60, fontFamily: 'var(--font-mono)',
      }}>
        {label}
      </span>
    </div>
  )
}

// Frequency spectrum visualisation (simplified SVG representation)
function FrequencyViz({ fftGridScore, fftHighFreqRatio }: {
  fftGridScore: number
  fftHighFreqRatio: number
}) {
  const SIZE = 120
  const cx = SIZE / 2
  const cy = SIZE / 2

  // Generate mock frequency rings for visualisation
  const rings = [0.15, 0.30, 0.45, 0.60, 0.75].map((r, i) => ({
    r: r * (SIZE / 2),
    opacity: 0.1 + i * 0.08,
  }))

  // Grid artifact markers (appear at regular intervals in GAN images)
  const gridMarkers = fftGridScore > 0.3
    ? [
        { x: cx + SIZE * 0.25, y: cy },
        { x: cx - SIZE * 0.25, y: cy },
        { x: cx, y: cy + SIZE * 0.25 },
        { x: cx, y: cy - SIZE * 0.25 },
      ]
    : []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      <svg
        width={SIZE} height={SIZE}
        viewBox={`0 0 ${SIZE} ${SIZE}`}
        aria-label="Frequency domain spectrum visualisation"
        style={{ background: 'var(--bg-4)', borderRadius: 'var(--radius)' }}
      >
        {/* Frequency rings */}
        {rings.map(({ r, opacity }, i) => (
          <circle
            key={i}
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={fftHighFreqRatio > 0.5 ? 'var(--ai)' : 'var(--teal)'}
            strokeWidth={1}
            opacity={opacity}
          />
        ))}

        {/* Grid artifact markers */}
        {gridMarkers.map((m, i) => (
          <circle
            key={i}
            cx={m.x} cy={m.y} r={4}
            fill="var(--ai)"
            opacity={fftGridScore}
          />
        ))}

        {/* DC component (centre) */}
        <circle cx={cx} cy={cy} r={4} fill="var(--teal)" />
      </svg>
      <p style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-3)', margin: 0 }}>
        frequency spectrum
      </p>
    </div>
  )
}

export default function ImageAnalysisViewer({ result, imageUrl }: Props) {
  const [tab, setTab] = useState<'signals' | 'frequency'>('signals')

  // Extract image-specific signals from evidence
  const ev = result.evidence_summary ?? {}
  const signals = result.top_signals ?? []

  // Map evidence to typed signal bars
  const signalBars: ImageSignal[] = [
    {
      name:   'GAN fingerprint',
      value:  ev.fingerprint_correlation ?? 0,
      label:  'model residual',
      weight: (ev.fingerprint_correlation ?? 0) > 0.5 ? 'high' : 'medium',
    },
    {
      name:   'FFT grid artifact',
      value:  ev.fft_grid_score ?? 0,
      label:  'checkerboard',
      weight: (ev.fft_grid_score ?? 0) > 0.4 ? 'high' : 'low',
    },
    {
      name:   'Over-smoothed HF',
      value:  ev.fft_high_freq_ratio ?? 0,
      label:  'no camera noise',
      weight: (ev.fft_high_freq_ratio ?? 0) > 0.5 ? 'high' : 'medium',
    },
    {
      name:   'Texture uniformity',
      value:  ev.texture_uniformity ?? 0,
      label:  'GLCM analysis',
      weight: (ev.texture_uniformity ?? 0) > 0.6 ? 'medium' : 'low',
    },
    {
      name:   'Bilateral symmetry',
      value:  ev.bilateral_symmetry ?? 0,
      label:  'StyleGAN signal',
      weight: (ev.bilateral_symmetry ?? 0) > 0.85 ? 'medium' : 'low',
    },
    {
      name:   'Background pattern',
      value:  ev.background_uniformity ?? 0,
      label:  'diffusion signal',
      weight: (ev.background_uniformity ?? 0) > 0.7 ? 'medium' : 'low',
    },
  ].filter(s => s.value > 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Image preview + score */}
      <div style={{ display: 'flex', gap: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>

        {/* Image thumbnail */}
        {imageUrl && (
          <div style={{
            width: 120, height: 120,
            borderRadius: 'var(--radius-lg)',
            border: '1px solid var(--border)',
            overflow: 'hidden', flexShrink: 0,
            position: 'relative',
          }}>
            <img
              src={imageUrl}
              alt="Analysed image"
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
            {/* Score overlay */}
            <div style={{
              position: 'absolute', bottom: 4, right: 4,
              background: 'rgba(0,0,0,0.75)',
              borderRadius: 4,
              padding: '2px 6px',
              fontSize: 11, fontFamily: 'var(--font-mono)',
              color: result.label === 'AI' ? 'var(--ai)' :
                     result.label === 'HUMAN' ? 'var(--human)' : 'var(--uncertain)',
            }}>
              {Math.round(result.authenticity_score * 100)}%
            </div>
          </div>
        )}

        {/* Summary */}
        <div style={{ flex: 1 }}>
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            padding: '4px 12px',
            borderRadius: 20,
            background: result.label === 'AI'    ? 'var(--ai-dim)'        :
                        result.label === 'HUMAN' ? 'var(--human-dim)'     :
                                                    'var(--uncertain-dim)',
            border: `1px solid ${
              result.label === 'AI'    ? 'var(--ai)'       :
              result.label === 'HUMAN' ? 'var(--human)'    :
                                          'var(--uncertain)'
            }`,
            marginBottom: 10,
          }}>
            <span style={{
              fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 500,
              color: result.label === 'AI'    ? 'var(--ai)'       :
                     result.label === 'HUMAN' ? 'var(--human)'    :
                                                 'var(--uncertain)',
            }}>
              {result.label === 'AI'    ? 'AI-generated image detected'   :
               result.label === 'HUMAN' ? 'Likely authentic photograph'    :
                                           'Uncertain — inconclusive signals'}
            </span>
          </div>

          <p style={{ fontSize: 12, color: 'var(--text-3)', margin: 0 }}>
            {result.label === 'AI' &&
              'This image exhibits characteristic patterns of AI generation — GAN fingerprints, frequency domain artifacts, or abnormal texture statistics.'}
            {result.label === 'HUMAN' &&
              'No significant AI generation artifacts detected. The image shows characteristics consistent with a real photograph.'}
            {result.label === 'UNCERTAIN' &&
              'Some signals are present but below confidence thresholds. Manual review recommended.'}
          </p>
        </div>
      </div>

      {/* Tab selector */}
      <div style={{ display: 'flex', gap: 6, borderBottom: '1px solid var(--border)', paddingBottom: 10 }}>
        {(['signals', 'frequency'] as const).map(t => (
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

      {/* Signal bars */}
      {tab === 'signals' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}
          aria-label="Detection signal breakdown">
          {signalBars.length > 0 ? (
            signalBars.map(s => <SignalBar key={s.name} {...s} />)
          ) : (
            <p style={{ fontSize: 13, color: 'var(--text-3)', textAlign: 'center', padding: '16px 0' }}>
              No significant signals detected
            </p>
          )}
        </div>
      )}

      {/* Frequency domain */}
      {tab === 'frequency' && (
        <div style={{ display: 'flex', gap: 20, alignItems: 'flex-start', flexWrap: 'wrap' }}>
          <FrequencyViz
            fftGridScore={ev.fft_grid_score ?? 0}
            fftHighFreqRatio={ev.fft_high_freq_ratio ?? 0}
          />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { label: 'Grid artifacts',   value: ev.fft_grid_score ?? 0,       desc: 'GAN checkerboard pattern' },
              { label: 'HF suppression',   value: ev.fft_high_freq_ratio ?? 0,  desc: 'Missing camera noise' },
              { label: 'Spectral regularity', value: ev.fft_peak_regularity ?? 0, desc: 'Periodic spectral peaks' },
              { label: 'Azimuthal variance', value: ev.fft_azimuthal_variance ?? 0, desc: 'Directional asymmetry' },
            ].map(({ label, value, desc }) => (
              <div key={label} style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '6px 12px',
                background: 'var(--bg-3)',
                borderRadius: 'var(--radius)',
              }}>
                <span style={{ fontSize: 12, color: 'var(--text-2)', flex: 1 }}>{label}</span>
                <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{desc}</span>
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: 11,
                  color: value > 0.5 ? 'var(--ai)' : 'var(--text-3)',
                  minWidth: 40, textAlign: 'right',
                }}>
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
