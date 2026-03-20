'use client'

// Step 63: Timeline View for video/audio results.
// Shows a horizontal scrubber with coloured markers at flagged timestamps.
// Severity is colour-coded: high (red) / medium (amber).

import { useState } from 'react'

interface FlaggedSegment {
  start_s:  number
  end_s:    number
  score:    number
  severity: 'high' | 'medium'
}

interface Props {
  durationS:        number
  flaggedSegments:  FlaggedSegment[]
  chunkResults?:    Array<{ start_s: number; end_s: number; score: number }>
  label:            string
  contentType:      'video' | 'audio'
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

function scoreToSeverityColor(score: number): string {
  if (score >= 0.85) return 'var(--ai)'
  if (score >= 0.65) return 'var(--uncertain)'
  return 'var(--human)'
}

export default function TimelineView({
  durationS,
  flaggedSegments,
  chunkResults = [],
  label,
  contentType,
}: Props) {
  const [hoveredSegment, setHoveredSegment] = useState<FlaggedSegment | null>(null)
  const [hoverX, setHoverX] = useState(0)

  const duration = Math.max(durationS, 1)
  const hasFlagged = flaggedSegments.length > 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <h3 style={{ fontSize: 11, fontFamily: 'var(--font-mono)', letterSpacing: '0.10em',
          textTransform: 'uppercase', color: 'var(--text-3)', margin: 0 }}>
          {contentType === 'video' ? 'Video' : 'Audio'} timeline
        </h3>
        <span style={{
          fontSize: 10, fontFamily: 'var(--font-mono)',
          color: 'var(--text-3)',
        }}>
          {formatTime(duration)} total
          {hasFlagged && ` · ${flaggedSegments.length} flagged segment${flaggedSegments.length > 1 ? 's' : ''}`}
        </span>
      </div>

      {/* Timeline track */}
      <div
        style={{
          position: 'relative',
          height: 48,
          background: 'var(--bg-4)',
          borderRadius: 'var(--radius)',
          overflow: 'visible',
          cursor: 'crosshair',
        }}
        role="img"
        aria-label={`Timeline showing ${flaggedSegments.length} flagged segments over ${formatTime(duration)}`}
        onMouseMove={e => {
          const rect = e.currentTarget.getBoundingClientRect()
          setHoverX(e.clientX - rect.left)
        }}
        onMouseLeave={() => setHoveredSegment(null)}
      >
        {/* Chunk score bars (background layer) */}
        {chunkResults.map((chunk, i) => {
          const leftPct  = (chunk.start_s / duration) * 100
          const widthPct = ((chunk.end_s - chunk.start_s) / duration) * 100
          const alpha    = Math.max(0.05, chunk.score * 0.4)
          return (
            <div
              key={i}
              style={{
                position: 'absolute',
                left: `${leftPct}%`,
                width: `${widthPct}%`,
                top: 0,
                height: '100%',
                background: `rgba(248, 113, 113, ${alpha})`,
              }}
              aria-hidden
            />
          )
        })}

        {/* Flagged segment markers */}
        {flaggedSegments.map((seg, i) => {
          const leftPct  = (seg.start_s / duration) * 100
          const widthPct = Math.max(((seg.end_s - seg.start_s) / duration) * 100, 0.5)
          const color    = seg.severity === 'high' ? 'var(--ai)' : 'var(--uncertain)'
          return (
            <div
              key={i}
              onMouseEnter={() => setHoveredSegment(seg)}
              style={{
                position: 'absolute',
                left: `${leftPct}%`,
                width: `${widthPct}%`,
                top: 8,
                height: 32,
                background: color,
                opacity: 0.85,
                borderRadius: 2,
                cursor: 'pointer',
                transition: 'opacity 0.15s',
              }}
              role="button"
              aria-label={`Flagged segment: ${formatTime(seg.start_s)} – ${formatTime(seg.end_s)}, ${Math.round(seg.score * 100)}% AI, severity: ${seg.severity}`}
              tabIndex={0}
            />
          )
        })}

        {/* Time ruler */}
        <div style={{ position: 'absolute', bottom: -18, left: 0, right: 0,
          display: 'flex', justifyContent: 'space-between' }}>
          {[0, 0.25, 0.5, 0.75, 1.0].map(frac => (
            <span key={frac} style={{
              fontSize: 10, fontFamily: 'var(--font-mono)',
              color: 'var(--text-3)',
            }}>
              {formatTime(frac * duration)}
            </span>
          ))}
        </div>
      </div>

      {/* Segment tooltip */}
      <div style={{ minHeight: 36, marginTop: 8 }}>
        {hoveredSegment ? (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 12,
            padding: '8px 14px',
            background: 'var(--bg-4)',
            borderRadius: 'var(--radius)',
            border: `1px solid ${hoveredSegment.severity === 'high' ? 'var(--ai)' : 'var(--uncertain)'}`,
            fontSize: 12,
          }}>
            <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-3)' }}>
              {formatTime(hoveredSegment.start_s)} – {formatTime(hoveredSegment.end_s)}
            </span>
            <span style={{
              color: hoveredSegment.severity === 'high' ? 'var(--ai)' : 'var(--uncertain)',
              fontWeight: 500,
            }}>
              {Math.round(hoveredSegment.score * 100)}% AI · {hoveredSegment.severity} severity
            </span>
          </div>
        ) : hasFlagged ? (
          <p style={{ fontSize: 12, color: 'var(--text-3)', margin: 0 }}>
            Hover over a segment for details
          </p>
        ) : (
          <p style={{ fontSize: 12, color: 'var(--human)', margin: 0 }}>
            No segments flagged — consistent with authentic content
          </p>
        )}
      </div>

      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 4 }}>
        {[
          { color: 'var(--ai)',        label: 'High confidence AI (≥85%)' },
          { color: 'var(--uncertain)', label: 'Medium confidence AI (65–85%)' },
        ].map(({ color, label }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 12, height: 6, borderRadius: 1, background: color }} />
            <span style={{ fontSize: 11, color: 'var(--text-3)' }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
