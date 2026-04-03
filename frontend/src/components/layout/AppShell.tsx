'use client'

import { Search, LayoutGrid, Clock } from 'lucide-react'
import { ThemeToggle } from '@/hooks/useTheme'
import { cn } from '@/lib/cn'

const NAV_ITEMS = [
  { href: '/analyze',   label: 'Analyze',   icon: Search },
  { href: '/dashboard', label: 'Dashboard', icon: LayoutGrid },
  { href: '/jobs',      label: 'History',   icon: Clock },
]

interface Props {
  children: React.ReactNode
  activePath?: string
}

export default function AppShell({ children, activePath = '' }: Props) {
  return (
    <div className="flex min-h-screen bg-surface">
      {/* ── Sidebar (desktop) ─────────────────────────────── */}
      <aside
        className="ag-sidebar hidden md:flex w-[220px] shrink-0 flex-col bg-surface-2 border-r border-edge"
        aria-label="Primary navigation"
      >
        {/* Logo */}
        <div className="px-5 pt-6 pb-5 border-b border-edge">
          <div className="flex items-center gap-2 mb-1">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-dot" />
            <span className="font-mono text-[10px] font-medium uppercase tracking-[0.18em] text-accent">
              AuthentiGuard
            </span>
          </div>
          <p className="text-[11px] text-fg-3 pl-[14px]">
            AI Detection Platform
          </p>
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-3 py-4 space-y-0.5" aria-label="Main navigation">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = activePath.startsWith(href)
            return (
              <a
                key={href}
                href={href}
                aria-current={active ? 'page' : undefined}
                className={cn(
                  'flex items-center gap-2.5 px-3 py-2 rounded no-underline',
                  'font-mono text-[12px] tracking-[0.04em] transition-colors duration-150',
                  active
                    ? 'text-accent border-l-2 border-accent bg-accent-dim ml-0'
                    : 'text-fg-3 border-l-2 border-transparent hover:text-fg-2',
                )}
              >
                <Icon size={15} strokeWidth={1.75} />
                {label}
              </a>
            )
          })}
        </nav>

        {/* Bottom */}
        <div className="px-4 py-3 border-t border-edge flex items-center justify-between">
          <span className="font-mono text-[10px] text-fg-3/50 tracking-wide">v0.1</span>
          <ThemeToggle />
        </div>
      </aside>

      {/* ── Main content ──────────────────────────────────── */}
      <main className="flex-1 flex flex-col min-w-0 md:pb-0 pb-16" id="main-content" role="main">
        {children}
      </main>

      {/* ── Mobile bottom nav ─────────────────────────────── */}
      <nav
        aria-label="Mobile navigation"
        className="md:hidden fixed bottom-0 left-0 right-0 bg-surface-2 border-t border-edge flex justify-around py-2 z-50"
      >
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = activePath.startsWith(href)
          return (
            <a
              key={href}
              href={href}
              aria-current={active ? 'page' : undefined}
              className={cn(
                'flex flex-col items-center gap-1 px-4 py-1 no-underline',
                'font-mono text-[10px] tracking-[0.06em] transition-colors duration-150',
                active ? 'text-accent' : 'text-fg-3 hover:text-fg-2',
              )}
            >
              <Icon size={16} strokeWidth={1.75} />
              {label}
            </a>
          )
        })}
      </nav>
    </div>
  )
}
