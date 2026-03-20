'use client'

// Step 44: Mobile-responsive layout shell.
// Sidebar collapses to a bottom nav on mobile.

import { useState } from 'react'
import { ThemeToggle } from '@/hooks/useTheme'

const NAV_ITEMS = [
  { href: '/analyze',   label: 'Analyze',   icon: <SearchIcon /> },
  { href: '/dashboard', label: 'Dashboard', icon: <GridIcon /> },
  { href: '/jobs',      label: 'History',   icon: <ClockIcon /> },
]

function SearchIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
    </svg>
  )
}
function GridIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/>
      <rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
    </svg>
  )
}
function ClockIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
    </svg>
  )
}

interface Props {
  children: React.ReactNode
  activePath?: string
}

export default function AppShell({ children, activePath = '' }: Props) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div style={{
      display: 'flex', minHeight: '100vh',
      background: 'var(--bg)',
    }}>
      {/* ── Sidebar (desktop) ─────────────────────────────── */}
      <aside style={{
        width: 220,
        background: 'var(--bg-2)',
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        padding: '20px 0',
        flexShrink: 0,
      }}
        className="ag-sidebar"
        aria-label="Primary navigation"
      >
        {/* Logo */}
        <div style={{ padding: '0 20px 24px', borderBottom: '1px solid var(--border)' }}>
          <div style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', fontWeight: 500,
            letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--teal)',
            display: 'flex', alignItems: 'center', gap: 8, marginBottom: 2,
          }}>
            <span style={{
              width: 6, height: 6, borderRadius: '50%',
              background: 'var(--teal)',
              boxShadow: '0 0 8px var(--teal-glow)',
            }} className="pulse-dot" />
            AuthentiGuard
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-3)' }}>
            AI Detection Platform
          </div>
        </div>

        {/* Nav links */}
        <nav style={{ padding: '16px 12px', flex: 1 }} aria-label="Main navigation">
          {NAV_ITEMS.map(({ href, label, icon }) => {
            const active = activePath.startsWith(href)
            return (
              <a
                key={href}
                href={href}
                aria-current={active ? 'page' : undefined}
                style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  padding: '9px 10px',
                  marginBottom: 2,
                  borderRadius: 'var(--radius)',
                  color: active ? 'var(--teal)' : 'var(--text-3)',
                  background: active ? 'var(--teal-dim)' : 'transparent',
                  textDecoration: 'none',
                  fontSize: 13,
                  fontWeight: active ? 500 : 400,
                  transition: 'all 0.15s',
                  border: active ? '1px solid var(--teal-dim)' : '1px solid transparent',
                }}
                onMouseEnter={e => { if (!active) { e.currentTarget.style.color = 'var(--text)'; e.currentTarget.style.background = 'var(--bg-3)' } }}
                onMouseLeave={e => { if (!active) { e.currentTarget.style.color = 'var(--text-3)'; e.currentTarget.style.background = 'transparent' } }}
              >
                {icon}
                {label}
              </a>
            )
          })}
        </nav>

        {/* Bottom — theme toggle */}
        <div style={{ padding: '12px 16px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'flex-end' }}>
          <ThemeToggle />
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0 }}
        id="main-content" role="main">
        {children}
      </main>

      {/* ── Mobile bottom nav ─────────────────────────────── */}
      <nav
        aria-label="Mobile navigation"
        style={{
          display: 'none',
          position: 'fixed', bottom: 0, left: 0, right: 0,
          background: 'var(--bg-2)',
          borderTop: '1px solid var(--border)',
          padding: '8px 0',
          justifyContent: 'space-around',
          zIndex: 100,
        }}
        className="ag-mobile-nav"
      >
        {NAV_ITEMS.map(({ href, label, icon }) => {
          const active = activePath.startsWith(href)
          return (
            <a key={href} href={href}
              aria-current={active ? 'page' : undefined}
              style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3,
                color: active ? 'var(--teal)' : 'var(--text-3)',
                textDecoration: 'none', fontSize: 10,
                fontFamily: 'var(--font-mono)',
                padding: '4px 16px',
              }}>
              {icon}
              {label}
            </a>
          )
        })}
      </nav>

      <style>{`
        @media (max-width: 768px) {
          .ag-sidebar  { display: none !important; }
          .ag-mobile-nav { display: flex !important; }
          #main-content { padding-bottom: 64px; }
        }
      `}</style>
    </div>
  )
}
