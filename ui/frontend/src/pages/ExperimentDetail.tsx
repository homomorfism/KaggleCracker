import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import type { ExperimentDetail, JournalEvent, RunStatus } from '../api'
import { api } from '../api'
import { ReportView } from '../charts'

// One journal event, fully expanded: this is where "why did that tool call
// fail" gets answered — args and error text shown whole, nothing elided.
function TurnEvent({ event }: { event: JournalEvent }) {
  if (event.type === 'reason')
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-info">THINK</span>
        <p className="turnlog-text">{event.text}</p>
      </div>
    )
  if (event.type === 'act') {
    const args = event.args ?? {}
    const code = typeof args.code === 'string' ? args.code : null
    const rest = Object.fromEntries(Object.entries(args).filter(([k]) => k !== 'code'))
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-amber">CALL</span>
        <div className="turnlog-body">
          <p className="turnlog-text">
            <code>{event.tool}</code>{' '}
            {Object.keys(rest).length > 0 && (
              <span className="mono-dim">{JSON.stringify(rest)}</span>
            )}
          </p>
          {code && (
            <details>
              <summary className="mono-dim">code — {code.length} chars</summary>
              <pre className="technique-code">{code}</pre>
            </details>
          )}
        </div>
      </div>
    )
  }
  if (event.type === 'tool_ok')
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-ok">OK</span>
        <div className="turnlog-body">
          <p className="turnlog-text">
            <code>{event.tool}</code>
          </p>
          <details>
            <summary className="mono-dim">payload</summary>
            <pre className="technique-code">{JSON.stringify(event.data, null, 2)}</pre>
          </details>
        </div>
      </div>
    )
  if (event.type === 'tool_error')
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-err">ERROR</span>
        <div className="turnlog-body">
          <p className="turnlog-text">
            <code>{event.tool}</code> <strong>{event.kind}</strong>
            {event.retryable === false && <span className="mono-dim"> · not retryable</span>}
          </p>
          <pre className="technique-code turnlog-err">{event.msg}</pre>
        </div>
      </div>
    )
  if (event.type === 'run_started')
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-info">START</span>
        <p className="turnlog-text mono-dim">{event.prompt}</p>
      </div>
    )
  if (event.type === 'run_finished')
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-ok">DONE</span>
        <p className="turnlog-text">{event.text}</p>
      </div>
    )
  if (event.type === 'run_failed')
    return (
      <div className="evt turnlog-evt">
        <span className="badge badge-err">FAILED</span>
        <p className="turnlog-text">{event.reason}</p>
      </div>
    )
  return null
}

// A whole turn's journal, fetched once when opened (and re-fetched while the
// turn is still running).
function TurnLog({
  slug,
  expId,
  turn,
  status,
  defaultOpen,
}: {
  slug: string
  expId: string
  turn: string
  status: RunStatus
  defaultOpen: boolean
}) {
  const [events, setEvents] = useState<JournalEvent[] | null>(null)
  const [open, setOpen] = useState(defaultOpen)

  useEffect(() => {
    if (!open) return
    let alive = true
    let timer: number | undefined
    const load = async () => {
      try {
        const page = await api.experimentEvents(slug, expId, turn, 0)
        if (!alive) return
        setEvents(page.events)
        if (page.status === 'running') timer = window.setTimeout(load, 1500)
      } catch {
        if (alive) setEvents([])
      }
    }
    load()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, expId, turn, open])

  return (
    <details className="turnlog" open={defaultOpen} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <span className={`lamp lamp-${status}`} />
        <span className="technique-name">{turn}</span>
        <span className="mono-dim"> · {status.toUpperCase()}</span>
      </summary>
      {open && events === null && <p className="mono-dim">loading…</p>}
      {open && events !== null && events.length === 0 && (
        <p className="mono-dim">journal is empty</p>
      )}
      {open && events !== null && events.map((e) => <TurnEvent key={e.seq} event={e} />)}
    </details>
  )
}

// Live tool activity for the running turn, shown inside the chat.
function TurnActivity({ slug, expId, turn }: { slug: string; expId: string; turn: string | null }) {
  const [lines, setLines] = useState<string[]>([])

  useEffect(() => {
    if (!turn) return
    let alive = true
    let since = 0
    let timer: number | undefined
    setLines([])
    const poll = async () => {
      try {
        const page = await api.experimentEvents(slug, expId, turn, since)
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
        if (fresh.length) setLines((prev) => [...prev, ...fresh].slice(-14))
        if (page.status === 'running') timer = window.setTimeout(poll, 1000)
      } catch {
        if (alive) timer = window.setTimeout(poll, 1500)
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, expId, turn])

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

export default function ExperimentDetailPage({ slug, expId }: { slug: string; expId: string }) {
  const [detail, setDetail] = useState<ExperimentDetail | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [sendError, setSendError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const logRef = useRef<HTMLDivElement>(null)

  const running = detail?.experiment.running ?? false

  useEffect(() => {
    let alive = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const d = await api.experimentDetail(slug, expId)
        if (!alive) return
        setError(null)
        setDetail(d)
        timer = window.setTimeout(poll, d.experiment.running ? 1500 : 4000)
      } catch (e) {
        if (!alive) return
        setError(String(e instanceof Error ? e.message : e))
        timer = window.setTimeout(poll, 3000)
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, expId, tick])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [detail?.messages.length, running])

  const send = async (e: FormEvent) => {
    e.preventDefault()
    const text = draft.trim()
    if (!text || running) return
    setSendError(null)
    try {
      await api.experimentMessage(slug, expId, text)
      setDraft('')
      setTick((t) => t + 1)
    } catch (err) {
      setSendError(String(err instanceof Error ? err.message : err))
    }
  }

  const stop = async () => {
    try {
      await api.experimentStop(slug, expId)
      setTick((t) => t + 1)
    } catch (e) {
      setSendError(String(e instanceof Error ? e.message : e))
    }
  }

  if (error && !detail) return <div className="notice notice-err">{error}</div>
  if (!detail) return <div className="notice mono-dim">LOADING…</div>

  const exp = detail.experiment
  const approvable = exp.state === 'plan_proposed' && !running

  return (
    <section className="eda-grid">
      <div className="eda-board">
        <div className="page-head">
          <div>
            <h1>{exp.name}</h1>
            <span className="mono-dim">
              {exp.id} · {exp.state.toUpperCase()}
              {exp.cv_score !== null && ` · CV ${exp.cv_score.toFixed(5)}`}
            </span>
          </div>
          <a className="mono-dim" href={`#/p/${slug}/experiments`}>
            ◂ ALL EXPERIMENTS
          </a>
        </div>

        {exp.result_summary && (
          <div className="notice">
            <p>
              <strong>RESULT</strong> — {exp.result_summary}
            </p>
          </div>
        )}

        {detail.plan_md ? (
          <div className="panel">
            <h3 className="panel-title">EXPERIMENT PLAN</h3>
            <ReportView text={detail.plan_md} />
          </div>
        ) : (
          <div className="notice">
            <p>
              No plan yet — the agent is gathering what it needs. Answer its questions
              in the chat; the plan lands here for your review.
            </p>
          </div>
        )}

        {detail.code.length > 0 && (
          <div className="panel" style={{ marginTop: 16 }}>
            <h3 className="panel-title">EXECUTED CODE — {detail.code.length} ATTEMPT{detail.code.length > 1 ? 'S' : ''}</h3>
            {detail.code.map((c, i) => (
              <details key={c.name} className="technique" open={i === detail.code.length - 1}>
                <summary>
                  <span className="technique-name">{c.name}</span>
                </summary>
                <pre className="technique-code">{c.source}</pre>
              </details>
            ))}
          </div>
        )}

        {detail.turns.length > 0 && (
          <div className="panel" style={{ marginTop: 16 }}>
            <h3 className="panel-title">
              AGENT LOG — {detail.turns.length} TURN{detail.turns.length > 1 ? 'S' : ''}
            </h3>
            <p className="mono-dim" style={{ marginBottom: 8 }}>
              every tool call with its arguments, outputs, and full error text
            </p>
            {detail.turns.map((t, i) => (
              <TurnLog
                key={t.id}
                slug={slug}
                expId={expId}
                turn={t.id}
                status={t.status}
                defaultOpen={i === detail.turns.length - 1}
              />
            ))}
          </div>
        )}
      </div>

      <aside className="chat">
        <div className="chat-head">
          <h3 className="panel-title">EXPERIMENT AGENT</h3>
          {running && (
            <button className="btn chat-stop" onClick={stop}>
              ■ STOP
            </button>
          )}
        </div>
        <div className="chat-log" ref={logRef}>
          {detail.messages.map((m) => (
            <div key={m.seq} className={`chat-msg chat-${m.role}`}>
              <span className="chat-role">{m.role === 'user' ? 'YOU' : 'AGENT'}</span>
              <p>{m.text}</p>
            </div>
          ))}
          {running && <TurnActivity slug={slug} expId={expId} turn={detail.latest_turn} />}
        </div>
        {approvable && (
          <div className="chat-approve">
            <button
              className="btn btn-primary"
              onClick={async () => {
                await api.experimentMessage(slug, expId, 'Accepted — run it.')
                setTick((t) => t + 1)
              }}
            >
              ✓ ACCEPT PLAN &amp; RUN
            </button>
          </div>
        )}
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
            placeholder={
              running
                ? 'the agent is working — STOP to interrupt…'
                : approvable
                  ? 'or tell the agent what to change…'
                  : 'answer or steer the agent…'
            }
            disabled={running}
            rows={2}
          />
          <button className="btn btn-primary" disabled={running || !draft.trim()}>
            SEND
          </button>
        </form>
        {sendError && <p className="form-error">✗ {sendError}</p>}
      </aside>
    </section>
  )
}
