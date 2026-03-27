/**
 * AuthentiGuard browser extension popup.
 * Shows connection status, last scan result, and a "Scan This Page" button.
 */

interface ScanResult {
  score: number
  label: 'AI' | 'HUMAN' | 'UNCERTAIN'
  confidence: number
  url: string
  timestamp: number
}

interface Settings {
  apiUrl: string
  autoScan: boolean
}

const DEFAULT_API_URL = 'http://localhost:8000'

async function getSettings(): Promise<Settings> {
  const data = await chrome.storage.local.get(['ag_api_url', 'ag_auto_scan'])
  return {
    apiUrl: data.ag_api_url || DEFAULT_API_URL,
    autoScan: data.ag_auto_scan ?? false,
  }
}

async function getLastResult(): Promise<ScanResult | null> {
  const data = await chrome.storage.local.get('ag_last_result')
  return data.ag_last_result || null
}

async function getAuthStatus(): Promise<boolean> {
  const data = await chrome.storage.local.get('ag_access_token')
  return !!data.ag_access_token
}

function formatScore(score: number): string {
  return (score * 100).toFixed(1) + '%'
}

function getLabelClass(label: string): string {
  return label.toLowerCase()
}

function timeSince(ts: number): string {
  const seconds = Math.floor((Date.now() - ts) / 1000)
  if (seconds < 60) return 'Just now'
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

async function scanCurrentPage(): Promise<void> {
  const btn = document.getElementById('scan-btn') as HTMLButtonElement
  const statusText = document.getElementById('scan-status')!

  btn.disabled = true
  statusText.textContent = 'Scanning...'

  try {
    // Send message to background script to trigger content analysis
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
    if (!tab?.id) throw new Error('No active tab')

    chrome.runtime.sendMessage(
      { type: 'SCAN_PAGE', tabId: tab.id },
      (response) => {
        if (response?.success) {
          statusText.textContent = 'Scan started!'
          // Refresh popup after a delay to show result
          setTimeout(() => window.location.reload(), 3000)
        } else {
          statusText.textContent = response?.error || 'Scan failed'
          btn.disabled = false
        }
      }
    )
  } catch (err) {
    statusText.textContent = err instanceof Error ? err.message : 'Scan failed'
    btn.disabled = false
  }
}

async function render(): Promise<void> {
  const root = document.getElementById('root')!
  const isAuthed = await getAuthStatus()
  const lastResult = await getLastResult()
  const settings = await getSettings()

  let resultHTML = '<div class="no-result">No scans yet. Click "Scan This Page" to start.</div>'

  if (lastResult) {
    const labelCls = getLabelClass(lastResult.label)
    resultHTML = `
      <div class="result-card">
        <div class="label">Last Scan Result</div>
        <div class="score ${labelCls}">${formatScore(lastResult.score)}</div>
        <span class="result-label ${labelCls}">${lastResult.label}</span>
        <div style="margin-top: 8px; font-size: 11px; color: #64748B;">
          Confidence: ${formatScore(lastResult.confidence)} · ${timeSince(lastResult.timestamp)}
        </div>
      </div>
    `
  }

  root.innerHTML = `
    <div class="header">
      <h1>AuthentiGuard</h1>
      <span class="badge">v1.0</span>
    </div>

    <div class="status">
      <span class="dot ${isAuthed ? 'connected' : 'disconnected'}"></span>
      <span>${isAuthed ? 'Connected' : 'Not logged in'}</span>
      <span style="margin-left: auto; font-size: 11px; color: #64748B;">
        ${settings.autoScan ? 'Auto-scan on' : ''}
      </span>
    </div>

    ${resultHTML}

    <button id="scan-btn" class="btn btn-primary" ${!isAuthed ? 'disabled' : ''}>
      ${isAuthed ? 'Scan This Page' : 'Login Required'}
    </button>
    <div id="scan-status" style="text-align: center; font-size: 12px; color: #94A3B8; margin-top: 8px;"></div>

    <a href="#" id="settings-link" class="settings-link">Settings</a>
  `

  // Event listeners
  const scanBtn = document.getElementById('scan-btn')
  if (scanBtn && isAuthed) {
    scanBtn.addEventListener('click', scanCurrentPage)
  }
}

// Initialize
document.addEventListener('DOMContentLoaded', render)
