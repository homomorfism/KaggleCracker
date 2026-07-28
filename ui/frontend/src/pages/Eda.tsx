import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { api } from '../api'
import { useEda } from '../hooks'
import { PanelCard } from '../panels'

// While a turn runs, surface the agent's tool activity inside the chat so the
// wait reads as work, not silence.
function AgentActivity({ slug, runId }: { slug: string; runId: string | null }) {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    if (!runId) return
    let alive = true
    let since = 0
    let timer: number | undefined
    setLines([])
    const poll = async () => {
      try {
        const page = await api.edaEvents(slug, runId, since)
        if (!alive) return
        since = page.next_since
        const fresh = page.events
          .map((e) => {
            if (e.type === 'act') return `▸ ${e.tool}`
            if (e.type === 'tool_ok') return `✓ ${e.tool}`
            if (e.type === 'tool_error') return `✗ ${e.tool} — ${e.kind}`
            if (e.type === 'reason' && e.text) return `… ${e.text.slice(0, 120)}`
            return null
          })
          .filter((l): l is string => l !== null)
        if (fresh.length) setLines((prev) => [...prev, ...fresh].slice(-12))
        if (page.status === 'running') timer = window.setTimeout(poll, 1000)
      } catch {
        // The turn directory may not exist yet; the next poll catches up.
        if (alive) timer = window.setTimeout(poll, 1500)
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, runId])

  return (
    <div className="chat-msg chat-agentwork">
      <span className="chat-role">AGENT — WORKING</span>
      {lines.length === 0 && <p className="mono-dim">starting the turn…</p>}
      {lines.map((l, i) => (
        <p key={i} className="chat-workline">
          {l}
        </p>
      ))}
    </div>
  )
}

export default function EdaPage({ slug }: { slug: string }) {
  const { dashboard, chat, agent, error, bump } = useEda(slug)
  const [draft, setDraft] = useState('')
  const [sendError, setSendError] = useState<string | null>(null)
  const logRef = useRef<HTMLDivElement>(null)
  const running = agent.status === 'running'

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [chat.length, running])

  const send = async (e: FormEvent) => {
    e.preventDefault()
    const text = draft.trim()
    if (!text || running) return
    setSendError(null)
    try {
      await api.edaMessage(slug, text)
      setDraft('')
      bump()
    } catch (err) {
      setSendError(String(err instanceof Error ? err.message : err))
    }
  }

  const panels = dashboard?.panels ?? []

  return (
    <section className="eda-grid">
      <div className="eda-board">
        <div className="page-head">
          <h1>Dataset EDA</h1>
          <span className="mono-dim">
            {dashboard ? `v${dashboard.version}${dashboard.updated ? ` · ${dashboard.updated}` : ''}` : '…'}
          </span>
        </div>
        {error && <div className="notice notice-err">{error}</div>}
        {panels.length === 0 && !error && (
          <div className="notice">
            <p>
              {running
                ? 'The EDA agent is building the first dashboard — panels appear here as they land.'
                : 'No dashboard yet. Ask for one in the chat — e.g. “explore the training data”.'}
            </p>
          </div>
        )}
        <div className="panel-grid">
          {panels.map((p) => (
            <PanelCard key={p.id} panel={p} />
          ))}
        </div>
      </div>

      <aside className="chat">
        <h3 className="panel-title">EDA AGENT</h3>
        <div className="chat-log" ref={logRef}>
          {chat.length === 0 && (
            <p className="mono-dim">
              Talk to the dashboard: “histogram of Age”, “drop the drift panel”,
              “correlations for numeric columns only”…
            </p>
          )}
          {chat.map((m) => (
            <div key={m.seq} className={`chat-msg chat-${m.role}`}>
              <span className="chat-role">{m.role === 'user' ? 'YOU' : 'AGENT'}</span>
              <p>{m.text}</p>
            </div>
          ))}
          {running && <AgentActivity slug={slug} runId={agent.run_id} />}
        </div>
        <form className="chat-form" onSubmit={send}>
          <textarea
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                e.currentTarget.form?.requestSubmit()
              }
            }}
            placeholder={running ? 'the agent is working…' : 'modify the dashboard…'}
            disabled={running}
            rows={2}
          />
          <button className="btn btn-primary" disabled={running || !draft.trim()}>
            {running ? 'WORKING…' : 'SEND'}
          </button>
        </form>
        {sendError && <p className="form-error">✗ {sendError}</p>}
      </aside>
    </section>
  )
}
