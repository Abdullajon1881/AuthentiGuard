'use client'

import { useState } from 'react'
import { cn } from '@/lib/cn'

interface FlaggedSegment {
  start_s:  number
  end_s:    number
  score:    number
  severity: 'high' | 'medium'
}

interface Props {
  durationS:       number
  flaggedSegments: FlaggedSegment[]
  chunkResults?:   Array<{ start_s: number; end_s: number; score: number }>
  label:           string
  contentType:     'video' | 'audio'
}

function formatTime(s: number): string {
  const m = Math.floor(s / 60)
  const sec = Math.floor(s % 60)
  return `${m}:${sec.toString().padStart(2, '0')}`
}

export default function TimelineView({
  durationS, flaggedSegments, chunkResults = [], label, contentType,
}: Props) {
  const [hoveredSegment, setHoveredSegment] = useState<FlaggedSegment | null>(null)

  const duration = Math.max(durationS, 1)
  const hasFlagged = flaggedSegments.length > 0

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="section-label">
          {contentType === 'video' ? 'Video' : 'Audio'} timeline
        </h3>
        <span className="font-mono text-[10px] text-fg-3">
          {formatTime(duration)} total
          {hasFlagged && ` \u00b7 ${flaggedSegments.length} flagged segment${flaggedSegments.length > 1 ? 's' : ''}`}
        </span>
      </div>

      {/* Timeline track */}
      <div
        className="relative h-12 bg-surface-4 rounded overflow-visible cursor-crosshair"
        role="img"
        aria-label={`Timeline showing ${flaggedSegments.length} flagged segments over ${formatTime(duration)}`}
        onMouseLeave={() => setHoveredSegment(null)}
      >
        {/* Chunk score bars */}
        {chunkResults.map((chunk, i) => {
          const leftPct = (chunk.start_s / duration) * 100
          const widthPct = ((chunk.end_s - chunk.start_s) / duration) * 100
          const alpha = Math.max(0.05, chunk.score * 0.4)
          return (
            <div key={i} className="absolute top-0 h-full" aria-hidden
              style={{ left: `${leftPct}%`, width: `${widthPct}%`, background: `rgba(196,75,55,${alpha})` }} />
          )
        })}

        {/* Flagged segments */}
        {flaggedSegments.map((seg, i) => {
          const leftPct = (seg.start_s / duration) * 100
          const widthPct = Math.max(((seg.end_s - seg.start_s) / duration) * 100, 0.5)
          const color = seg.severity === 'high' ? 'var(--ai)' : 'var(--uncertain)'
          return (
            <div key={i}
              onMouseEnter={() => setHoveredSegment(seg)}
              className="absolute top-2 h-8 rounded-sm cursor-pointer opacity-85 hover:opacity-100 transition-opacity"
              style={{ left: `${leftPct}%`, width: `${widthPct}%`, background: color }}
              role="button" tabIndex={0}
              aria-label={`Flagged: ${formatTime(seg.start_s)}\u2013${formatTime(seg.end_s)}, ${Math.round(seg.score * 100)}% AI`}
            />
          )
        })}

        {/* Time ruler */}
        <div className="absolute -bottom-5 left-0 right-0 flex justify-between">
          {[0, 0.25, 0.5, 0.75, 1.0].map(frac => (
            <span key={frac} className="font-mono text-[10px] text-fg-3">
              {formatTime(frac * duration)}
            </span>
          ))}
        </div>
      </div>

      {/* Tooltip */}
      <div className="min-h-[36px] mt-2">
        {hoveredSegment ? (
          <div className={cn(
            'flex items-center gap-3 py-2 px-3.5 bg-surface-4 rounded text-xs border',
            hoveredSegment.severity === 'high' ? 'border-ai' : 'border-uncertain',
          )}>
            <span className="font-mono text-fg-3">
              {formatTime(hoveredSegment.start_s)} \u2013 {formatTime(hoveredSegment.end_s)}
            </span>
            <span className="font-medium" style={{ color: hoveredSegment.severity === 'high' ? 'var(--ai)' : 'var(--uncertain)' }}>
              {Math.round(hoveredSegment.score * 100)}% AI \u00b7 {hoveredSegment.severity} severity
            </span>
          </div>
        ) : hasFlagged ? (
          <p className="text-xs text-fg-3">Hover over a segment for details</p>
        ) : (
          <p className="text-xs text-human">No segments flagged \u2014 consistent with authentic content</p>
        )}
      </div>

      {/* Legend */}
      <div className="flex gap-4">
        {[
          { color: 'bg-ai',        label: 'High confidence AI (\u226585%)' },
          { color: 'bg-uncertain', label: 'Medium confidence AI (65\u201385%)' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className={cn('w-3 h-1.5 rounded-sm', color)} />
            <span className="text-[11px] text-fg-3">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
