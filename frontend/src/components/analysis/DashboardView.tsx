'use client'

// Step 42: User Dashboard — scan history, usage statistics, saved reports.

import { useEffect, useState } from 'react'
import { dashboard } from '@/lib/api'
import type { UsageStats } from '@/types'

// ── Stat card ─────────────────────────────────────────────────

function StatCard({ label, value, accent }: { label: string; value: string | number; accent?: string }) {
  return (
    <div style={{
      padding: '18px 20px',
      background: 'var(--bg-3)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
    }}>
      <p style={{ margin: '0 0 6px', fontSize: 10, fontFamily: 'var(--font-mono)',
        letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--text-3)' }}>
        {label}
      </p>
      <p style={{ margin: 0, fontSize: 28, fontWeight: 500,
        fontFamily: 'var(--font-mono)', color: accent ?? 'var(--teal)' }}>
        {value}
      </p>
    </div>
  )
}

// ── Tier usage bar ────────────────────────────────────────────

function TierBar({ used, limit, tier }: { used: number; limit: number; tier: string }) {
  const pct = Math.min((used / Math.max(limit, 1)) * 100, 100)
  const color = pct > 80 ? 'var(--ai)' : pct > 50 ? 'var(--uncertain)' : 'var(--teal)'

  return (
    <div style={{
      padding: '16px 20px',
      background: 'var(--bg-3)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)',
          color: 'var(--text-3)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          Usage — {tier} tier
        </span>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-2)' }}>
          {used} / {limit} req/min
        </span>
      </div>
      <div style={{ height: 6, background: 'var(--bg-4)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: color, borderRadius: 3,
          transition: 'width 0.5s ease',
        }} />
      </div>
    </div>
  )
}

// ── Detection breakdown donut ─────────────────────────────────

function DetectionBreakdown({ ai, human, uncertain }: { ai: number; human: number; uncertain: number }) {
  const total = ai + human + uncertain || 1
  const aiPct = Math.round((ai / total) * 100)
  const huPct = Math.round((human / total) * 100)
  const unPct = 100 - aiPct - huPct

  return (
    <div style={{
      padding: '16px 20px',
      background: 'var(--bg-3)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
    }}>
      <p style={{ margin: '0 0 12px', fontSize: 10, fontFamily: 'var(--font-mono)',
        letterSpacing: '0.10em', textTransform: 'uppercase', color: 'var(--text-3)' }}>
        Result breakdown
      </p>
      {/* Stacked bar */}
      <div style={{ display: 'flex', height: 10, borderRadius: 5, overflow: 'hidden', gap: 2, marginBottom: 10 }}>
        {aiPct > 0 && <div style={{ width: `${aiPct}%`, background: 'var(--ai)' }} />}
        {unPct > 0 && <div style={{ width: `${unPct}%`, background: 'var(--uncertain)' }} />}
        {huPct > 0 && <div style={{ width: `${huPct}%`, background: 'var(--human)' }} />}
      </div>
      <div style={{ display: 'flex', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--ai)' }} />
          <span style={{ fontSize: 11, color: 'var(--text-3)' }}>AI {aiPct}%</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--uncertain)' }} />
          <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Uncertain {unPct}%</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: 'var(--human)' }} />
          <span style={{ fontSize: 11, color: 'var(--text-3)' }}>Human {huPct}%</span>
        </div>
      </div>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────

interface Props {
  onNewAnalysis: () => void
}

export default function DashboardView({ onNewAnalysis }: Props) {
  const [stats, setStats]   = useState<UsageStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]   = useState<string | null>(null)

  useEffect(() => {
    dashboard.getStats()
      .then(setStats)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ padding: '40px 0', textAlign: 'center', color: 'var(--text-3)', fontSize: 13 }}>
      Loading dashboard…
    </div>
  )

  if (error) return (
    <div style={{ padding: '20px', color: 'var(--ai)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>
      Error loading stats: {error}
    </div>
  )

  if (!stats) return null

  return (
    <div className="stagger" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Summary stats grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
        <StatCard label="Total scans"     value={stats.total_scans} />
        <StatCard label="This month"      value={stats.scans_this_month} />
        <StatCard label="AI detected"     value={stats.ai_detected}     accent="var(--ai)" />
        <StatCard label="Human verified"  value={stats.human_detected}  accent="var(--human)" />
      </div>

      {/* Breakdown */}
      <DetectionBreakdown
        ai={stats.ai_detected}
        human={stats.human_detected}
        uncertain={stats.uncertain}
      />

      {/* Tier usage */}
      <TierBar used={stats.tier_used} limit={stats.tier_limit} tier="free" />

      {/* CTA */}
      <button
        onClick={onNewAnalysis}
        style={{
          padding: '12px',
          background: 'var(--teal)',
          color: '#000',
          border: 'none',
          borderRadius: 'var(--radius)',
          fontFamily: 'var(--font-mono)',
          fontSize: 13,
          fontWeight: 500,
          letterSpacing: '0.06em',
          cursor: 'pointer',
          width: '100%',
        }}
        aria-label="Start a new analysis"
      >
        New analysis →
      </button>
    </div>
  )
}
