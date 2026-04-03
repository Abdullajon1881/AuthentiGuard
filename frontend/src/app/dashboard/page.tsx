'use client'

import { useRouter } from 'next/navigation'
import AppShell from '@/components/layout/AppShell'
import DashboardView from '@/components/analysis/DashboardView'

export default function DashboardPage() {
  const router = useRouter()
  return (
    <AppShell activePath="/dashboard">
      <div className="max-w-[760px] w-full mx-auto px-6 py-10">
        <div className="mb-8">
          <p className="section-label mb-2">Dashboard</p>
          <h1 className="font-serif text-2xl font-bold text-fg">
            Your activity
          </h1>
        </div>
        <DashboardView onNewAnalysis={() => router.push('/analyze')} />
      </div>
    </AppShell>
  )
}
