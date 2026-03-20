'use client'

import React, { useCallback, useRef, useState } from 'react'
import { analysis } from '@/lib/api'
import type { AnalysisJob, ContentType } from '@/types'

// ── Types ─────────────────────────────────────────────────────

type InputMode = 'text' | 'file' | 'url' | 'code'

interface Props {
  onJobCreated: (job: AnalysisJob) => void
  onError: (msg: string) => void
}

const ACCEPT = '.txt,.md,.pdf,.docx,.mp3,.wav,.flac,.mp4,.mov,.jpg,.jpeg,.png,.webp,.py,.js,.ts,.java,.go,.cpp,.c,.rs'
const MAX_BYTES = 500 * 1024 * 1024

// ── Mode tab ──────────────────────────────────────────────────

function ModeTab({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: '7px 16px',
        fontSize: 12,
        fontFamily: 'var(--font-mono)',
        fontWeight: 500,
        letterSpacing: '0.06em',
        borderRadius: 'var(--radius)',
        border: active ? '1px solid var(--teal)' : '1px solid var(--border-2)',
        background: active ? 'var(--teal-dim)' : 'transparent',
        color: active ? 'var(--teal)' : 'var(--text-3)',
        cursor: 'pointer',
        transition: 'all 0.15s',
      }}
    >
      {label}
    </button>
  )
}

// ── Drop zone ─────────────────────────────────────────────────

function DropZone({ onFile }: { onFile: (f: File) => void }) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }, [onFile])

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
      aria-label="Upload file for analysis"
      onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
      style={{
        border: `1.5px dashed ${dragging ? 'var(--teal)' : 'var(--border-2)'}`,
        borderRadius: 'var(--radius-lg)',
        padding: '48px 24px',
        textAlign: 'center',
        cursor: 'pointer',
        background: dragging ? 'var(--teal-dim2)' : 'transparent',
        transition: 'all 0.2s',
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        style={{ display: 'none' }}
        onChange={e => e.target.files?.[0] && onFile(e.target.files[0])}
        aria-hidden
      />
      {/* Upload icon */}
      <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-3)"
        strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"
        style={{ margin: '0 auto 12px' }}>
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/>
        <line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
      <p style={{ color: 'var(--text-2)', fontSize: 14, marginBottom: 4 }}>
        Drop a file here, or click to browse
      </p>
      <p style={{ color: 'var(--text-3)', fontSize: 12 }}>
        Text, PDF, DOCX, Images, Audio, Video, Code — up to 500 MB
      </p>
    </div>
  )
}

// ── Main Upload Module ────────────────────────────────────────

export default function UploadModule({ onJobCreated, onError }: Props) {
  const [mode, setMode]       = useState<InputMode>('text')
  const [text, setText]       = useState('')
  const [url, setUrl]         = useState('')
  const [file, setFile]       = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [stage, setStage]     = useState('')

  async function handleSubmit() {
    if (loading) return
    setLoading(true)
    setStage('Submitting…')

    try {
      let job: AnalysisJob

      if (mode === 'text' || mode === 'code') {
        const content = text.trim()
        if (content.length < 20) {
          onError('Please enter at least 20 characters of text.')
          return
        }
        setStage('Queuing analysis…')
        job = await analysis.submitText(content, mode === 'code' ? 'code' : 'text')

      } else if (mode === 'file') {
        if (!file) { onError('Please select a file.'); return }
        if (file.size > MAX_BYTES) { onError('File exceeds 500 MB limit.'); return }
        setStage(`Uploading ${file.name}…`)
        job = await analysis.submitFile(file)

      } else {
        onError('URL analysis coming soon.')
        return
      }

      onJobCreated(job)
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Submission failed')
    } finally {
      setLoading(false)
      setStage('')
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Mode tabs */}
      <div style={{ display: 'flex', gap: 8 }}>
        {(['text', 'file', 'code', 'url'] as InputMode[]).map(m => (
          <ModeTab key={m} label={m.toUpperCase()} active={mode === m} onClick={() => setMode(m)} />
        ))}
      </div>

      {/* Input area */}
      {(mode === 'text' || mode === 'code') && (
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={mode === 'code'
            ? '// Paste code here…'
            : 'Paste or type text to analyze…'}
          rows={10}
          aria-label={mode === 'code' ? 'Code input' : 'Text input'}
          style={{
            width: '100%',
            background: 'var(--bg-3)',
            border: '1px solid var(--border-2)',
            borderRadius: 'var(--radius-lg)',
            color: 'var(--text)',
            fontFamily: mode === 'code' ? 'var(--font-mono)' : 'var(--font-sans)',
            fontSize: mode === 'code' ? 13 : 14,
            lineHeight: 1.65,
            padding: '14px 16px',
            resize: 'vertical',
            outline: 'none',
          }}
          onFocus={e => (e.target.style.borderColor = 'var(--teal)')}
          onBlur={e => (e.target.style.borderColor = 'var(--border-2)')}
        />
      )}

      {mode === 'file' && (
        <div>
          <DropZone onFile={f => setFile(f)} />
          {file && (
            <div style={{
              marginTop: 10,
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '10px 14px',
              background: 'var(--teal-dim2)',
              border: '1px solid var(--teal-dim)',
              borderRadius: 'var(--radius)',
            }}>
              <span style={{ color: 'var(--teal)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>
                {file.name}
              </span>
              <span style={{ color: 'var(--text-3)', fontSize: 11, marginLeft: 'auto' }}>
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </span>
              <button
                onClick={() => setFile(null)}
                aria-label="Remove file"
                style={{ background: 'none', border: 'none', color: 'var(--text-3)', cursor: 'pointer', fontSize: 16, lineHeight: 1 }}
              >×</button>
            </div>
          )}
        </div>
      )}

      {mode === 'url' && (
        <div style={{ display: 'flex', gap: 8 }}>
          <input
            type="url"
            value={url}
            onChange={e => setUrl(e.target.value)}
            placeholder="https://example.com/article"
            aria-label="URL input"
            style={{
              flex: 1,
              background: 'var(--bg-3)',
              border: '1px solid var(--border-2)',
              borderRadius: 'var(--radius)',
              color: 'var(--text)',
              fontSize: 14,
              padding: '10px 14px',
              outline: 'none',
            }}
          />
        </div>
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        aria-busy={loading}
        style={{
          padding: '12px 24px',
          background: loading ? 'var(--teal-dim)' : 'var(--teal)',
          color: loading ? 'var(--teal)' : '#000',
          border: 'none',
          borderRadius: 'var(--radius)',
          fontFamily: 'var(--font-mono)',
          fontSize: 13,
          fontWeight: 500,
          letterSpacing: '0.06em',
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'all 0.15s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 8,
        }}
      >
        {loading ? (
          <>
            <span className="pulse-dot" style={{
              width: 8, height: 8,
              background: 'var(--teal)',
              borderRadius: '50%',
              display: 'inline-block',
            }} />
            {stage || 'Analyzing…'}
          </>
        ) : (
          'Analyze →'
        )}
      </button>
    </div>
  )
}
