'use client'

import { useRouter } from 'next/navigation'
import AppShell from '@/components/layout/AppShell'
import DashboardView from '@/components/analysis/DashboardView'

export default function DashboardPage() {
  const router = useRouter()
  return (
    <AppShell activePath="/dashboard">
      <div style={{ maxWidth: 760, width: '100%', margin: '0 auto', padding: '40px 24px' }}>
        <div style={{ marginBottom: 24 }}>
          <div style={{
            fontSize: 10, fontFamily: 'var(--font-mono)', letterSpacing: '0.12em',
            textTransform: 'uppercase', color: 'var(--teal)', marginBottom: 6,
          }}>
            Dashboard
          </div>
          <h1 style={{ fontSize: 22, fontWeight: 500, color: 'var(--text)', margin: 0 }}>
            Your activity
          </h1>
        </div>
        <DashboardView onNewAnalysis={() => router.push('/analyze')} />
      </div>
    </AppShell>
  )
}
