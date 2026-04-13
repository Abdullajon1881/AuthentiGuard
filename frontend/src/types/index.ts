// All shared TypeScript types for the AuthentiGuard frontend.
// Every API response shape is typed here — no `any` anywhere.

export type ContentType = 'text' | 'image' | 'video' | 'audio' | 'code'
export type JobStatus   = 'pending' | 'processing' | 'completed' | 'failed'
export type Label       = 'AI' | 'HUMAN' | 'UNCERTAIN'
export type UserRole    = 'admin' | 'analyst' | 'api_consumer'
export type UserTier    = 'free' | 'pro' | 'enterprise'

// ── Auth ─────────────────────────────────────────────────────

export interface TokenResponse {
  access_token:  string
  refresh_token: string
  token_type:    'bearer'
  expires_in:    number
}

export interface User {
  id:         string
  email:      string
  full_name:  string | null
  role:       UserRole
  tier:       UserTier
  is_active:  boolean
  created_at: string
}

// ── Jobs ─────────────────────────────────────────────────────

export interface AnalysisJob {
  job_id:       string
  status:       JobStatus
  content_type: ContentType
  created_at:   string
  poll_url:     string
}

export interface JobStatus_ {
  job_id:       string
  status:       JobStatus
  progress:     string | null
  created_at:   string
  completed_at: string | null
}

// ── Results ───────────────────────────────────────────────────

export interface SentenceScore {
  text:     string
  score:    number
  evidence: Record<string, unknown>
}

export interface LayerScores {
  perplexity:  number | null
  stylometry:  number | null
  transformer: number | null
  adversarial: number | null
}

export interface EvidenceSignal {
  signal: string
  value:  string
  weight: 'high' | 'medium' | 'low'
}

export interface ModelAttribution {
  gpt_family:    number
  claude_family: number
  llama_family:  number
  human:         number
  other:         number
}

export interface ImageSignal {
  name:   string
  value:  number
  label:  string
  weight: 'high' | 'medium' | 'low'
}

export interface DetectionResult {
  job_id:             string
  status:             JobStatus
  content_type:       ContentType
  authenticity_score: number
  confidence:         number
  label:              Label
  layer_scores:       LayerScores
  sentence_scores:    SentenceScore[]
  top_signals:        EvidenceSignal[]
  model_attribution:  ModelAttribution
  processing_ms:      number | null
  report_url:         string | null
  created_at:         string
  completed_at:       string | null
  evidence_summary?:  Record<string, unknown>
}

// ── Dashboard ─────────────────────────────────────────────────

export interface UsageStats {
  total_scans:      number
  scans_this_month: number
  ai_detected:      number
  human_detected:   number
  uncertain:        number
  avg_score:        number | null
  tier_limit:       number
  tier_used:        number
}

// ── API error ─────────────────────────────────────────────────

export interface ApiError {
  error:   string
  message: string
  detail?: unknown
}
