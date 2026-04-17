'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { auth } from '@/lib/api'

export default function RegisterPage() {
  const router = useRouter()
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [consent, setConsent] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      await auth.register(email, password, fullName || undefined)
      router.push('/auth/login')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center px-4 py-12">

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
          <h1 className="font-serif text-[1.2rem] text-fg leading-snug">Create account</h1>
          <p className="text-fg-3 text-[13px] mt-1">Start detecting AI-generated content</p>
        </div>

        {/* Form body */}
        <div className="px-7 py-6 space-y-4">

          {error && (
            <div className="px-3 py-2.5 rounded border border-ai bg-ai-dim text-[12px] font-mono text-ai leading-relaxed">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">

            {/* Full name */}
            <div className="space-y-1.5">
              <label
                htmlFor="fullName"
                className="block font-mono text-[10px] uppercase tracking-[0.12em] text-fg-3"
              >
                Full Name{' '}
                <span className="normal-case tracking-normal text-fg-3 opacity-50 font-sans not-italic text-[10px]">
                  (optional)
                </span>
              </label>
              <input
                id="fullName"
                type="text"
                value={fullName}
                onChange={e => setFullName(e.target.value)}
                autoComplete="name"
                placeholder="Jane Doe"
                className="
                  w-full bg-surface border border-edge rounded
                  px-3 py-[9px] text-[13px] text-fg
                  placeholder:text-fg-3
                  focus:outline-none focus:border-edge-3
                  transition-colors duration-150
                "
              />
            </div>

            {/* Email */}
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

            {/* Password */}
            <div className="space-y-1.5">
              <label
                htmlFor="password"
                className="block font-mono text-[10px] uppercase tracking-[0.12em] text-fg-3"
              >
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                minLength={10}
                autoComplete="new-password"
                placeholder="Min 10 characters"
                className="
                  w-full bg-surface border border-edge rounded
                  px-3 py-[9px] text-[13px] text-fg
                  placeholder:text-fg-3
                  focus:outline-none focus:border-edge-3
                  transition-colors duration-150
                "
              />
              <p className="font-mono text-[10px] text-fg-3 opacity-60 leading-relaxed">
                Min 10 chars · at least one uppercase letter and one digit
              </p>
            </div>

            {/* Consent */}
            <div className="flex items-start gap-2.5 pt-1">
              <input
                id="consent"
                type="checkbox"
                checked={consent}
                onChange={e => setConsent(e.target.checked)}
                required
                className="mt-[3px] accent-accent flex-shrink-0"
              />
              <label htmlFor="consent" className="text-[12px] text-fg-3 leading-relaxed cursor-pointer">
                I agree to the{' '}
                <span className="text-fg-2 underline decoration-dotted underline-offset-2 cursor-pointer">
                  Terms of Service
                </span>{' '}
                and{' '}
                <span className="text-fg-2 underline decoration-dotted underline-offset-2 cursor-pointer">
                  Privacy Policy
                </span>
              </label>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full mt-1 disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? 'Creating account…' : 'Create Account →'}
            </button>

          </form>
        </div>

        {/* Footer */}
        <div className="px-7 pb-6 border-t border-edge pt-5 text-center">
          <p className="text-[13px] text-fg-3">
            Already have an account?{' '}
            <Link
              href="/auth/login"
              className="text-accent hover:opacity-80 transition-opacity no-underline"
            >
              Sign in
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
