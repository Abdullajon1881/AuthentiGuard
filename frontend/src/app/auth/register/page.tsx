'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import { auth } from '@/lib/api'

export default function RegisterPage() {
  const router = useRouter()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [fullName, setFullName] = useState('')
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
    <div style={{ maxWidth: 400, margin: '4rem auto', padding: '2rem' }}>
      <h1 style={{ fontSize: '1.5rem', fontWeight: 600, marginBottom: '1.5rem' }}>
        Create Account
      </h1>
      {error && (
        <p style={{ color: '#dc2626', marginBottom: '1rem' }}>{error}</p>
      )}
      <form onSubmit={handleSubmit}>
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
          Full Name
        </label>
        <input
          type="text"
          value={fullName}
          onChange={e => setFullName(e.target.value)}
          style={{ width: '100%', padding: '0.5rem', marginBottom: '1rem', border: '1px solid #ccc', borderRadius: 4 }}
        />
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
          Email
        </label>
        <input
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          style={{ width: '100%', padding: '0.5rem', marginBottom: '1rem', border: '1px solid #ccc', borderRadius: 4 }}
        />
        <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 500 }}>
          Password
        </label>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          minLength={8}
          style={{ width: '100%', padding: '0.5rem', marginBottom: '1.5rem', border: '1px solid #ccc', borderRadius: 4 }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%', padding: '0.75rem', background: '#2563eb', color: '#fff',
            border: 'none', borderRadius: 4, fontWeight: 500, cursor: loading ? 'wait' : 'pointer',
          }}
        >
          {loading ? 'Creating account...' : 'Create Account'}
        </button>
      </form>
      <p style={{ marginTop: '1rem', color: '#666' }}>
        Already have an account?{' '}
        <Link href="/auth/login" style={{ color: '#2563eb', textDecoration: 'underline' }}>
          Sign In
        </Link>
      </p>
    </div>
  )
}
