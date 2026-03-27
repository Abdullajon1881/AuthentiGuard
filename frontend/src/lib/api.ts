// Typed API client. All requests go through here — auth headers,
// error normalisation, and automatic access-token refresh on 401.

import type {
  AnalysisJob,
  DetectionResult,
  JobStatus_,
  TokenResponse,
  User,
  UsageStats,
} from '@/types'

const BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// ── Token storage (memory + localStorage) ────────────────────

let _accessToken: string | null = null

export function setTokens(access: string, refresh: string): void {
  _accessToken = access
  if (typeof window !== 'undefined') {
    localStorage.setItem('ag_refresh', refresh)
  }
}

export function clearTokens(): void {
  _accessToken = null
  if (typeof window !== 'undefined') {
    localStorage.removeItem('ag_refresh')
  }
}

function getRefreshToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('ag_refresh')
}

// ── Core fetch ────────────────────────────────────────────────

interface FetchOptions extends RequestInit {
  skipAuth?: boolean
}

async function apiFetch<T>(path: string, opts: FetchOptions = {}): Promise<T> {
  const { skipAuth = false, ...init } = opts

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(init.headers as Record<string, string> ?? {}),
  }

  if (!skipAuth && _accessToken) {
    headers['Authorization'] = `Bearer ${_accessToken}`
  }

  const res = await fetch(`${BASE}/api/v1${path}`, { ...init, headers })

  // Auto-refresh on 401
  if (res.status === 401 && !skipAuth) {
    const refreshed = await _tryRefresh()
    if (refreshed) {
      headers['Authorization'] = `Bearer ${_accessToken}`
      const retry = await fetch(`${BASE}/api/v1${path}`, { ...init, headers })
      if (!retry.ok) await _throwApiError(retry)
      return retry.json() as Promise<T>
    }
    clearTokens()
    if (typeof window !== 'undefined') window.location.href = '/auth/login'
    throw new Error('Session expired')
  }

  if (!res.ok) await _throwApiError(res)
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

async function _throwApiError(res: Response): Promise<never> {
  let body: { message?: string; error?: string } = {}
  try { body = await res.json() } catch { /* ignore */ }
  throw new Error(body.message ?? body.error ?? `HTTP ${res.status}`)
}

async function _tryRefresh(): Promise<boolean> {
  const token = getRefreshToken()
  if (!token) return false
  try {
    const res = await fetch(`${BASE}/api/v1/auth/refresh`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: token }),
    })
    if (!res.ok) return false
    const data: TokenResponse = await res.json()
    setTokens(data.access_token, data.refresh_token)
    return true
  } catch {
    return false
  }
}

// ── Auth ──────────────────────────────────────────────────────

export const auth = {
  async register(email: string, password: string, fullName?: string): Promise<User> {
    return apiFetch('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ email, password, full_name: fullName, consent_given: true }),
      skipAuth: true,
    })
  },

  async login(email: string, password: string): Promise<TokenResponse> {
    const tokens = await apiFetch<TokenResponse>('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
      skipAuth: true,
    })
    setTokens(tokens.access_token, tokens.refresh_token)
    return tokens
  },

  async logout(): Promise<void> {
    const token = getRefreshToken()
    if (token) {
      await apiFetch('/auth/logout', {
        method: 'POST',
        body: JSON.stringify({ refresh_token: token }),
      }).catch(() => {/* ignore logout errors */})
    }
    clearTokens()
  },
}

// ── Analysis ──────────────────────────────────────────────────

export const analysis = {
  async submitText(text: string, contentType: 'text' | 'code' = 'text'): Promise<AnalysisJob> {
    return apiFetch('/analyze/text', {
      method: 'POST',
      body: JSON.stringify({ text, content_type: contentType }),
    })
  },

  async submitFile(file: File): Promise<AnalysisJob> {
    const form = new FormData()
    form.append('file', file)
    return apiFetch('/analyze/file', {
      method: 'POST',
      body: form,
      headers: { Authorization: `Bearer ${_accessToken}` },
    })
  },

  async submitUrl(url: string): Promise<AnalysisJob> {
    return apiFetch('/analyze/url', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  },

  async getReport(jobId: string, format: 'json' | 'pdf' = 'json'): Promise<Record<string, unknown>> {
    return apiFetch(`/jobs/${jobId}/report?format=${format}`)
  },

  async getStatus(jobId: string): Promise<JobStatus_> {
    return apiFetch(`/jobs/${jobId}`)
  },

  async getResult(jobId: string): Promise<DetectionResult> {
    return apiFetch(`/jobs/${jobId}/result`)
  },

  /** Poll until completed or failed. Resolves with the final result. */
  async pollUntilDone(
    jobId: string,
    onProgress?: (status: string) => void,
    intervalMs = 1500,
    maxWaitMs = 120_000,
  ): Promise<DetectionResult> {
    const deadline = Date.now() + maxWaitMs
    while (Date.now() < deadline) {
      const job = await this.getStatus(jobId)
      onProgress?.(job.status)
      if (job.status === 'completed') return this.getResult(jobId)
      if (job.status === 'failed') throw new Error('Detection job failed')
      await new Promise(r => setTimeout(r, intervalMs))
    }
    throw new Error('Detection timed out')
  },
}

// ── Dashboard ─────────────────────────────────────────────────

export const dashboard = {
  async getStats(): Promise<UsageStats> {
    return apiFetch('/dashboard/stats')
  },
}

// ── Health ────────────────────────────────────────────────────

export async function healthCheck(): Promise<{ status: string }> {
  return apiFetch('/health', { skipAuth: true })
}
