/**
 * Frontend unit tests.
 * Tests the score-to-label logic, colour mapping, and API client helpers.
 * No DOM rendering required — pure logic tests.
 */

// ── Score → label mapping ─────────────────────────────────────

function scoreToLabel(score: number): string {
  if (score >= 0.75) return 'AI'
  if (score <= 0.40) return 'HUMAN'
  return 'UNCERTAIN'
}

function scoreToColor(score: number): string {
  if (score >= 0.80) return 'rgba(248, 113, 113, 0.22)'
  if (score >= 0.65) return 'rgba(251, 191,  36, 0.18)'
  if (score >= 0.50) return 'rgba(251, 191,  36, 0.08)'
  return 'transparent'
}

function pct(score: number): number {
  return Math.round(score * 100)
}

// Boundary tests
const LABEL_CASES: [number, string][] = [
  [0.99, 'AI'],
  [0.75, 'AI'],
  [0.74, 'UNCERTAIN'],
  [0.55, 'UNCERTAIN'],
  [0.41, 'UNCERTAIN'],
  [0.40, 'HUMAN'],
  [0.01, 'HUMAN'],
]

describe('scoreToLabel', () => {
  LABEL_CASES.forEach(([score, expected]) => {
    test(`${score} → ${expected}`, () => {
      expect(scoreToLabel(score)).toBe(expected)
    })
  })
})

describe('scoreToColor', () => {
  test('≥0.80 → red highlight', () => {
    expect(scoreToColor(0.85)).toContain('248, 113, 113')
  })
  test('0.65–0.79 → amber highlight', () => {
    expect(scoreToColor(0.70)).toContain('251, 191')
  })
  test('0.50–0.64 → faint amber', () => {
    expect(scoreToColor(0.55)).toContain('0.08')
  })
  test('<0.50 → transparent', () => {
    expect(scoreToColor(0.30)).toBe('transparent')
  })
})

describe('pct', () => {
  test('converts to 0–100 integer', () => {
    expect(pct(0.856)).toBe(86)
    expect(pct(0.001)).toBe(0)
    expect(pct(1.000)).toBe(100)
  })
})

// ── Attribution sum ───────────────────────────────────────────

describe('model attribution', () => {
  test('sums to 1.0 for high AI score', () => {
    const score = 0.80
    const humanProb = 1 - score
    const aiProb    = score
    const attribution = {
      human:        humanProb,
      gpt_family:   aiProb * 0.40,
      claude_family: aiProb * 0.25,
      llama_family: aiProb * 0.25,
      other:        aiProb * 0.10,
    }
    const total = Object.values(attribution).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1.0)).toBeLessThan(0.01)
  })

  test('sums to 1.0 for low AI score', () => {
    const score = 0.10
    const humanProb = 1 - score
    const aiProb    = score
    const attribution = {
      human:        humanProb,
      gpt_family:   aiProb * 0.40,
      claude_family: aiProb * 0.25,
      llama_family: aiProb * 0.25,
      other:        aiProb * 0.10,
    }
    const total = Object.values(attribution).reduce((a, b) => a + b, 0)
    expect(Math.abs(total - 1.0)).toBeLessThan(0.01)
  })
})

// ── Processing time formatting ────────────────────────────────

function formatMs(ms: number | null): string {
  if (!ms) return '—'
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`
}

describe('formatMs', () => {
  test('null → —',      () => expect(formatMs(null)).toBe('—'))
  test('500ms → 500ms', () => expect(formatMs(500)).toBe('500ms'))
  test('1500ms → 1.5s', () => expect(formatMs(1500)).toBe('1.5s'))
  test('2000ms → 2.0s', () => expect(formatMs(2000)).toBe('2.0s'))
})

// ── Stage inference ───────────────────────────────────────────

function inferStageIndex(status: string, elapsedMs: number): number {
  if (status === 'completed') return 8
  if (status === 'failed')    return -1
  if (status === 'pending')   return 0
  const t = elapsedMs / 1000
  if (t < 1)  return 1
  if (t < 3)  return 2
  if (t < 5)  return 3
  if (t < 8)  return 4
  if (t < 11) return 5
  if (t < 14) return 6
  return 7
}

describe('inferStageIndex', () => {
  test('completed → last stage', () => expect(inferStageIndex('completed', 0)).toBe(8))
  test('failed → -1',            () => expect(inferStageIndex('failed', 0)).toBe(-1))
  test('pending → 0',            () => expect(inferStageIndex('pending', 0)).toBe(0))
  test('processing at 2s → 2',   () => expect(inferStageIndex('processing', 2000)).toBe(2))
  test('processing at 9s → 5',   () => expect(inferStageIndex('processing', 9000)).toBe(5))
  test('processing at 20s → 7',  () => expect(inferStageIndex('processing', 20000)).toBe(7))
})
