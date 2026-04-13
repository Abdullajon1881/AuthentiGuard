'use client'

import React, { useCallback, useRef, useState } from 'react'
import { Upload, X, FileText, Loader2 } from 'lucide-react'
import { analysis } from '@/lib/api'
import { cn } from '@/lib/cn'
import type { AnalysisJob } from '@/types'

type InputMode = 'text' | 'file' | 'url' | 'code'

interface Props {
  onJobCreated: (job: AnalysisJob) => void
  onError: (msg: string) => void
}

const ACCEPT = '.txt,.md,.pdf,.docx,.mp3,.wav,.flac,.mp4,.mov,.jpg,.jpeg,.png,.webp,.py,.js,.ts,.java,.go,.cpp,.c,.rs'
const MAX_BYTES = 500 * 1024 * 1024

/* ── Drop zone ─────────────────────────────────────────────── */

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
      className={cn(
        'border border-dashed rounded-lg py-12 px-6 text-center cursor-pointer transition-all duration-200',
        dragging
          ? 'border-accent bg-accent-dim'
          : 'border-edge-2 hover:border-edge-3',
      )}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={e => e.target.files?.[0] && onFile(e.target.files[0])}
        aria-hidden
      />
      <Upload size={28} strokeWidth={1.5} className="mx-auto mb-3 text-fg-3" />
      <p className="text-fg-2 text-sm mb-1">
        Drop a file here, or click to browse
      </p>
      <p className="text-fg-3 text-xs">
        Text, PDF, DOCX, Images, Audio, Video, Code — up to 500 MB
      </p>
    </div>
  )
}

/* ── Main Upload Module ────────────────────────────────────── */

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
    setStage('Submitting\u2026')

    try {
      let job: AnalysisJob

      if (mode === 'text' || mode === 'code') {
        const content = text.trim()
        if (content.length < 20) {
          onError('Please enter at least 20 characters of text.')
          return
        }
        setStage('Queuing analysis\u2026')
        job = await analysis.submitText(content, mode === 'code' ? 'code' : 'text')

      } else if (mode === 'file') {
        if (!file) { onError('Please select a file.'); return }
        if (file.size > MAX_BYTES) { onError('File exceeds 500 MB limit.'); return }
        setStage(`Uploading ${file.name}\u2026`)
        job = await analysis.submitFile(file)

      } else if (mode === 'url') {
        if (!url.trim()) { onError('Please enter a URL.'); return }
        setStage('Fetching and analyzing URL\u2026')
        job = await analysis.submitUrl(url.trim())
      } else {
        onError('Unknown input mode.')
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
    <div className="flex flex-col gap-5">
      {/* Mode tabs */}
      <div className="segment-control">
        {(['text', 'file', 'code', 'url'] as InputMode[]).map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={cn(
              mode === m ? 'segment-btn-active' : 'segment-btn',
            )}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Input area */}
      {(mode === 'text' || mode === 'code') && (
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder={mode === 'code'
            ? '// Paste code here\u2026'
            : 'Paste or type text to analyze\u2026'}
          rows={10}
          aria-label={mode === 'code' ? 'Code input' : 'Text input'}
          className={cn(
            'w-full bg-surface-3 border border-edge-2 rounded-lg text-fg leading-relaxed',
            'px-4 py-3.5 resize-y outline-none',
            'focus:border-accent transition-colors duration-150',
            'placeholder:text-fg-3',
            mode === 'code' ? 'font-mono text-[13px]' : 'font-sans text-sm',
          )}
        />
      )}

      {mode === 'file' && (
        <div>
          <DropZone onFile={f => setFile(f)} />
          {file && (
            <div className="mt-3 flex items-center gap-3 px-3.5 py-2.5 border border-edge-2 rounded">
              <FileText size={14} className="text-accent shrink-0" />
              <span className="font-mono text-xs text-accent truncate">
                {file.name}
              </span>
              <span className="font-mono text-[11px] text-fg-3 ml-auto shrink-0">
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </span>
              <button
                onClick={() => setFile(null)}
                aria-label="Remove file"
                className="btn-ghost p-0.5"
              >
                <X size={14} />
              </button>
            </div>
          )}
        </div>
      )}

      {mode === 'url' && (
        <input
          type="url"
          value={url}
          onChange={e => setUrl(e.target.value)}
          placeholder="https://example.com/article"
          aria-label="URL input"
          className={cn(
            'w-full bg-surface-3 border border-edge-2 rounded text-fg text-sm',
            'px-4 py-2.5 outline-none',
            'focus:border-accent transition-colors duration-150',
            'placeholder:text-fg-3',
          )}
        />
      )}

      {/* Submit */}
      <button
        onClick={handleSubmit}
        disabled={loading}
        aria-busy={loading}
        className={cn(
          'btn-primary py-3',
          loading && 'opacity-70 cursor-not-allowed',
        )}
      >
        {loading ? (
          <>
            <Loader2 size={14} className="animate-spin" />
            {stage || 'Analyzing\u2026'}
          </>
        ) : (
          'Analyze \u2192'
        )}
      </button>
    </div>
  )
}
