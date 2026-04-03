'use client'

import { useEffect, useState } from 'react'
import { ArrowRight, AlertCircle } from 'lucide-react'
import { dashboard } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { UsageStats } from '@/types'

/* ── Stat card (editorial: large serif number + monospace label) ── */

function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="border border-edge rounded-lg p-5">
      <p className="section-label mb-3">{label}</p>
      <p className="font-serif text-3xl font-bold leading-none" style={{ color: color ?? 'var(--text)' }}>
        {value}
      </p>
    </div>
  )
}

/* ── Tier usage bar ──────────────────────────────────────────── */

function TierBar({ used, limit, tier }: { used: number; limit: number; tier: string }) {
  const pct = Math.min((used / Math.max(limit, 1)) * 100, 100)
  const color = pct > 80 ? 'var(--ai)' : pct > 50 ? 'var(--uncertain)' : 'var(--accent)'

  return (
    <div className="card p-5">
      <div className="flex justify-between items-baseline mb-3">
        <span className="section-label">Usage — {tier} tier</span>
        <span className="font-mono text-xs text-fg-2">{used} / {limit} req/min</span>
      </div>
      <div className="h-1.5 bg-surface-4 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

/* ── Detection breakdown ─────────────────────────────────────── */

function DetectionBreakdown({ ai, human, uncertain }: { ai: number; human: number; uncertain: number }) {
  const total = ai + human + uncertain || 1
  const aiPct = Math.round((ai / total) * 100)
  const huPct = Math.round((human / total) * 100)
  const unPct = 100 - aiPct - huPct

  return (
    <div className="card p-5">
      <p className="section-label mb-3">Result breakdown</p>
      <div className="flex h-2.5 rounded overflow-hidden gap-px mb-3">
        {aiPct > 0 && <div className="bg-ai transition-all duration-500" style={{ width: `${aiPct}%` }} />}
        {unPct > 0 && <div className="bg-uncertain transition-all duration-500" style={{ width: `${unPct}%` }} />}
        {huPct > 0 && <div className="bg-human transition-all duration-500" style={{ width: `${huPct}%` }} />}
      </div>
      <div className="flex gap-4">
        {[
          { color: 'bg-ai', label: 'AI', pct: aiPct },
          { color: 'bg-uncertain', label: 'Uncertain', pct: unPct },
          { color: 'bg-human', label: 'Human', pct: huPct },
        ].map(({ color, label, pct }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className={cn('w-2 h-2 rounded-sm', color)} />
            <span className="text-[11px] text-fg-3">{label} {pct}%</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Skeleton loader ─────────────────────────────────────────── */

function DashboardSkeleton() {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="border border-edge rounded-lg p-5">
            <div className="shimmer h-3 w-20 rounded mb-4" />
            <div className="shimmer h-8 w-16 rounded" />
          </div>
        ))}
      </div>
      <div className="card p-5">
        <div className="shimmer h-3 w-28 rounded mb-4" />
        <div className="shimmer h-2.5 w-full rounded" />
      </div>
    </div>
  )
}

/* ── Main ──────────────────────────────────────────────────── */

interface Props {
  onNewAnalysis: () => void
}

export default function DashboardView({ onNewAnalysis }: Props) {
  const [stats, setStats]     = useState<UsageStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState<string | null>(null)

  useEffect(() => {
    dashboard.getStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <DashboardSkeleton />

  if (error) return (
    <div className="flex items-center gap-2.5 p-4 text-ai text-sm font-mono border border-ai rounded bg-ai-dim">
      <AlertCircle size={14} />
      Error loading stats: {error}
    </div>
  )

  if (!stats) return null

  return (
    <div className="stagger flex flex-col gap-4">
      {/* Summary stats — large editorial numbers */}
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Total scans"    value={stats.total_scans} />
        <StatCard label="This month"     value={stats.scans_this_month} />
        <StatCard label="AI detected"    value={stats.ai_detected}    color="var(--ai)" />
        <StatCard label="Human verified" value={stats.human_detected} color="var(--human)" />
      </div>

      <DetectionBreakdown
        ai={stats.ai_detected}
        human={stats.human_detected}
        uncertain={stats.uncertain}
      />

      <TierBar used={stats.tier_used} limit={stats.tier_limit} tier="free" />

      <button
        onClick={onNewAnalysis}
        className="btn-primary w-full py-3"
        aria-label="Start a new analysis"
      >
        New analysis <ArrowRight size={14} />
      </button>
    </div>
  )
}
