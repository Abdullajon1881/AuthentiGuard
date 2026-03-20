/**
 * Background service worker — Step 113/116.
 *
 * Responsibilities:
 *   - Relay analysis requests from content scripts to the AuthentiGuard API
 *   - Cache results in chrome.storage.session (5 min TTL)
 *   - Handle social media platform-specific content extraction (Step 116)
 *   - Context menu: "Check with AuthentiGuard" on right-click
 *   - Keyboard shortcut: Ctrl+Shift+A → scan page
 *
 * Step 116 — Social media integration:
 *   Twitter/X:   Extract tweet text from [data-testid="tweetText"]
 *   Facebook:    Extract post text from [data-ad-preview="message"]
 *   LinkedIn:    Extract post from .feed-shared-update-v2__description
 *   Reddit:      Extract post body from [slot="text-body"]
 */

'use strict';

const API_BASE = 'https://api.authentiguard.io';
const CACHE_TTL_MS = 5 * 60 * 1000;   // 5 min session cache

// ── Auth token management ─────────────────────────────────────

async function getToken(): Promise<string | null> {
  const s = await chrome.storage.sync.get(['apiKey', 'jwtToken', 'tokenExpiry']);
  if (s.apiKey) return null;   // use X-API-Key header instead
  if (s.jwtToken && s.tokenExpiry > Date.now()) return s.jwtToken;
  return null;
}

async function buildHeaders(): Promise<Record<string, string>> {
  const s = await chrome.storage.sync.get(['apiKey', 'jwtToken', 'tokenExpiry']);
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (s.apiKey) {
    headers['X-API-Key'] = s.apiKey;
  } else if (s.jwtToken && Date.now() < (s.tokenExpiry ?? 0)) {
    headers['Authorization'] = `Bearer ${s.jwtToken}`;
  }
  return headers;
}

// ── Cache helpers ─────────────────────────────────────────────

async function getCached(key: string): Promise<any | null> {
  const data = await chrome.storage.session.get(key);
  const entry = data[key];
  if (!entry) return null;
  if (Date.now() - entry.ts > CACHE_TTL_MS) return null;
  return entry.result;
}

async function setCached(key: string, result: any): Promise<void> {
  await chrome.storage.session.set({ [key]: { result, ts: Date.now() } });
}

// ── Text analysis ─────────────────────────────────────────────

async function analyzeText(text: string): Promise<any> {
  // Content hash for cache key
  const hash = await crypto.subtle.digest(
    'SHA-256', new TextEncoder().encode(text)
  );
  const cacheKey = `text:${Array.from(new Uint8Array(hash)).map(b => b.toString(16).padStart(2,'0')).join('').slice(0,16)}`;

  const cached = await getCached(cacheKey);
  if (cached) return cached;

  const headers = await buildHeaders();
  const res = await fetch(`${API_BASE}/api/v1/analyze/text`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ content: text }),
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const result = await res.json();
  await setCached(cacheKey, result);
  return result;
}

// ── Image analysis ────────────────────────────────────────────

async function analyzeImageUrl(src: string): Promise<any> {
  const cacheKey = `img:${src.slice(-40)}`;
  const cached   = await getCached(cacheKey);
  if (cached) return cached;

  // Fetch the image bytes via fetch (service worker has cross-origin access)
  const imageRes = await fetch(src);
  const blob     = await imageRes.blob();
  const file     = new File([blob], 'image.jpg', { type: blob.type });

  const headers = await buildHeaders();
  delete headers['Content-Type'];   // let browser set multipart boundary

  const fd = new FormData();
  fd.append('file', file);

  const res = await fetch(`${API_BASE}/api/v1/analyze/file`, {
    method: 'POST',
    headers,
    body: fd,
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const result = await res.json();
  await setCached(cacheKey, result);
  return result;
}

// ── Step 116: Social media platform extractors ────────────────

function extractSocialContent(url: string, tabId: number): void {
  const platform = detectPlatform(url);
  if (!platform) return;

  chrome.scripting.executeScript({
    target: { tabId },
    func: (platform: string) => {
      // Injected into page — runs in page context
      const extractors: Record<string, () => string[]> = {
        twitter: () => Array.from(
          document.querySelectorAll<HTMLElement>('[data-testid="tweetText"]')
        ).map(el => el.innerText).filter(t => t.length > 30),

        facebook: () => Array.from(
          document.querySelectorAll<HTMLElement>(
            '[data-ad-preview="message"], .userContent, [data-testid="post_message"]'
          )
        ).map(el => el.innerText).filter(t => t.length > 30),

        linkedin: () => Array.from(
          document.querySelectorAll<HTMLElement>(
            '.feed-shared-update-v2__description, .comments-post-meta__headline-text'
          )
        ).map(el => el.innerText).filter(t => t.length > 30),

        reddit: () => Array.from(
          document.querySelectorAll<HTMLElement>('[slot="text-body"], .RichTextJSON-root')
        ).map(el => el.innerText).filter(t => t.length > 30),
      };

      const texts = (extractors[platform] ?? (() => []))();
      return texts.slice(0, 10);
    },
    args: [platform],
  }).then(results => {
    const texts = results[0]?.result as string[] ?? [];
    texts.forEach(text => {
      chrome.tabs.sendMessage(tabId, {
        type: 'ANALYZE_SOCIAL_TEXT',
        platform,
        text,
      });
    });
  });
}

function detectPlatform(url: string): string | null {
  if (url.includes('twitter.com') || url.includes('x.com')) return 'twitter';
  if (url.includes('facebook.com')) return 'facebook';
  if (url.includes('linkedin.com')) return 'linkedin';
  if (url.includes('reddit.com'))   return 'reddit';
  return null;
}

// ── Message handler ───────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === 'ANALYZE_TEXT') {
        const result = await analyzeText(msg.text);
        chrome.tabs.sendMessage(sender.tab!.id!, {
          type: 'RESULT', elementId: msg.id, result,
        });
      } else if (msg.type === 'ANALYZE_IMAGE') {
        const result = await analyzeImageUrl(msg.src);
        chrome.tabs.sendMessage(sender.tab!.id!, {
          type: 'RESULT', elementId: msg.id, result,
        });
      } else if (msg.type === 'GET_SETTINGS') {
        const s = await chrome.storage.sync.get(['settings']);
        sendResponse(s.settings ?? {});
      } else if (msg.type === 'SAVE_SETTINGS') {
        await chrome.storage.sync.set({ settings: msg.settings });
        sendResponse({ ok: true });
      }
    } catch (err) {
      console.error('[AuthentiGuard]', err);
    }
  })();
  return true;   // keep channel open for async sendResponse
});

// ── Context menu ──────────────────────────────────────────────

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id:       'ag-check-selection',
    title:    'Check with AuthentiGuard',
    contexts: ['selection'],
  });
  chrome.contextMenus.create({
    id:       'ag-check-image',
    title:    'Check image with AuthentiGuard',
    contexts: ['image'],
  });
  chrome.contextMenus.create({
    id:       'ag-scan-page',
    title:    'Scan this page',
    contexts: ['page'],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (!tab?.id) return;
  if (info.menuItemId === 'ag-check-selection' && info.selectionText) {
    const result = await analyzeText(info.selectionText);
    chrome.notifications.create({
      type:    'basic',
      iconUrl: 'icons/icon48.png',
      title:   `AuthentiGuard: ${result.label}`,
      message: `AI probability: ${Math.round(result.authenticity_score * 100)}% — ${result.verdict_explanation?.slice(0, 80) ?? ''}`,
    });
  }
  if (info.menuItemId === 'ag-check-image' && info.srcUrl) {
    const result = await analyzeImageUrl(info.srcUrl);
    chrome.notifications.create({
      type:    'basic',
      iconUrl: 'icons/icon48.png',
      title:   `AuthentiGuard: ${result.label}`,
      message: `Image AI probability: ${Math.round(result.authenticity_score * 100)}%`,
    });
  }
  if (info.menuItemId === 'ag-scan-page') {
    chrome.tabs.sendMessage(tab.id, { type: 'SCAN_PAGE' });
  }
});

// ── Keyboard shortcut ─────────────────────────────────────────

chrome.commands.onCommand.addListener((command) => {
  if (command === 'scan-page') {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (tab?.id) chrome.tabs.sendMessage(tab.id, { type: 'SCAN_PAGE' });
    });
  }
});

// ── Tab navigation: trigger social extraction ─────────────────

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    extractSocialContent(tab.url, tabId);
  }
});
