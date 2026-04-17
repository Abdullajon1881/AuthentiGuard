'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { auth } from '@/lib/api'

export default function LoginPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await auth.login(email, password)
      router.push('/analyze')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center px-4">

      {/* Brand mark */}
      <Link href="/" className="mb-10 flex flex-col items-center gap-1 no-underline group">
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse-dot" />
          <span className="font-mono text-[11px] font-medium uppercase tracking-[0.18em] text-accent">
            AuthentiGuard
          </span>
        </div>
        <span className="font-mono text-[10px] text-fg-3 tracking-[0.1em] group-hover:text-fg-2 transition-colors">
          AI Detection Platform
        </span>
      </Link>

      {/* Card */}
      <div className="w-full max-w-[380px] bg-surface-2 border border-edge rounded-lg animate-fade-up">

        {/* Header */}
        <div className="px-7 pt-7 pb-5 border-b border-edge">
          <h1 className="font-serif text-[1.2rem] text-fg leading-snug">Sign in</h1>
          <p className="text-fg-3 text-[13px] mt-1">Welcome back</p>
        </div>

        {/* Form body */}
        <div className="px-7 py-6 space-y-5">

          {error && (
            <div className="px-3 py-2.5 rounded border border-ai bg-ai-dim text-[12px] font-mono text-ai leading-relaxed">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">

            <div className="space-y-1.5">
              <label
                htmlFor="email"
                className="block font-mono text-[10px] uppercase tracking-[0.12em] text-fg-3"
              >
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="you@company.com"
                className="
                  w-full bg-surface border border-edge rounded
                  px-3 py-[9px] text-[13px] text-fg
                  placeholder:text-fg-3
                  focus:outline-none focus:border-edge-3
                  transition-colors duration-150
                "
              />
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="password"
                  className="font-mono text-[10px] uppercase tracking-[0.12em] text-fg-3"
                >
                  Password
                </label>
                <Link
                  href="/auth/forgot-password"
                  className="font-mono text-[10px] text-fg-3 opacity-50 hover:opacity-80 transition-opacity no-underline"
                >
                  Forgot?
                </Link>
              </div>
              <input
                id="password"
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                placeholder="••••••••••"
                className="
                  w-full bg-surface border border-edge rounded
                  px-3 py-[9px] text-[13px] text-fg
                  placeholder:text-fg-3
                  focus:outline-none focus:border-edge-3
                  transition-colors duration-150
                "
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full mt-1 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? 'Signing in…' : 'Sign In →'}
            </button>

          </form>
        </div>

        {/* Footer */}
        <div className="px-7 pb-6 border-t border-edge pt-5 text-center">
          <p className="text-[13px] text-fg-3">
            No account?{' '}
            <Link
              href="/auth/register"
              className="text-accent hover:opacity-80 transition-opacity no-underline"
            >
              Create one
            </Link>
          </p>
        </div>
      </div>

      <Link
        href="/"
        className="mt-8 font-mono text-[10px] uppercase tracking-widest text-fg-3 opacity-40 hover:opacity-70 transition-opacity no-underline"
      >
        ← Back to home
      </Link>
    </div>
  )
}
