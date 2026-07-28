import { useEffect, useMemo, useRef, useState } from 'react'
import type { FormEvent, ReactNode } from 'react'
import type { JournalEvent } from '../api'
import { api } from '../api'
import { useRunEvents } from '../hooks'
import { collectFindings, deriveFacts } from '../findings'
import {
  CorrelationChart,
  DriftChart,
  MissingnessChart,
  NumericTable,
  ReportView,
  TargetBalanceChart,
} from '../charts'

const BADGES: Record<string, [string, string]> = {
  run_started: ['INIT', 'badge-info'],
  reason: ['THINK', 'badge-info'],
  act: ['CALL', 'badge-amber'],
  tool_ok: ['OK', 'badge-ok'],
  tool_error: ['ERROR', 'badge-err'],
  gate_prompt: ['GATE', 'badge-gate'],
  gate_answer: ['GATE', 'badge-gate'],
  run_finished: ['DONE', 'badge-ok'],
  run_failed: ['FAILED', 'badge-err'],
}

// Args for the feed: long string values (a whole plan markdown) collapse to
// their size so a CALL row stays one scannable line; the payload details on
// the matching OK row still hold everything.
function compactArgs(args: Record<string, unknown> | undefined): string {
  if (!args) return ''
  const shown = Object.fromEntries(
    Object.entries(args).map(([k, v]) =>
      typeof v === 'string' && v.length > 100 ? [k, `«${v.length} chars»`] : [k, v],
    ),
  )
  return JSON.stringify(shown)
}

// One line a human can scan: row count plus which checks came back.
function okSummary(data: Record<string, unknown>): string {
  const meta = ['path', 'rows_profiled', 'columns', 'findings_recorded']
  const checks = Object.keys(data).filter((k) => !meta.includes(k))
  const parts: string[] = []
  if (typeof data.path === 'string') parts.push(data.path)
  if (typeof data.rows_profiled === 'number') parts.push(`${data.rows_profiled} rows`)
  if (checks.length) parts.push(checks.join(', '))
  return parts.join(' · ') || 'done'
}

function EventBody({ event }: { event: JournalEvent }) {
  switch (event.type) {
    case 'run_started':
      return <p className="evt-think">{event.prompt}</p>
    case 'reason':
      return <p className="evt-think">{event.text}</p>
    case 'act':
      return (
        <p>
          <code>{event.tool}</code>{' '}
          <span className="evt-args">{compactArgs(event.args)}</span>
        </p>
      )
    case 'tool_ok':
      return (
        <div>
          <p>
            <code>{event.tool}</code> {okSummary(event.data ?? {})}
          </p>
          <details>
            <summary>PAYLOAD</summary>
            <pre>{JSON.stringify(event.data, null, 2)}</pre>
          </details>
        </div>
      )
    case 'tool_error':
      return (
        <p>
          <code>{event.tool}</code> <strong>{event.kind}</strong> — {event.msg}
          {event.retryable === false && <span className="evt-args"> (not retryable)</span>}
        </p>
      )
    case 'gate_prompt':
      return <pre className="gate-pre">{event.text}</pre>
    case 'gate_answer':
      return (
        <p>
          answer: “{event.answer}”{event.scripted && <em className="evt-args"> — scripted, not a human</em>}
        </p>
      )
    case 'run_finished':
      return <p>run complete — report in the panel alongside.</p>
    case 'run_failed':
      return <p>{event.reason}</p>
    default:
      return <pre>{JSON.stringify(event, null, 2)}</pre>
  }
}

function Instrument({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="panel inst">
      <h3 className="panel-title">{title}</h3>
      {children}
    </section>
  )
}

// Shown while a LIVE run is blocked at core/gate.py waiting for a person.
// The box only transports text — what counts as approval is the gate's rule
// (only "yes"/"y"; anything else, including empty, denies).
function GateBox({ slug, runId }: { slug: string; runId: string }) {
  const [answer, setAnswer] = useState('')
  const [sent, setSent] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const send = async (text: string) => {
    setError(null)
    setSent(true)
    try {
      await api.gateAnswer(slug, runId, text)
    } catch (e) {
      setSent(false)
      setError(String(e instanceof Error ? e.message : e))
    }
  }
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (!sent) send(answer)
  }

  return (
    <div className="gatebox">
      <h3 className="panel-title">⚠ HUMAN GATE — THE AGENT IS WAITING FOR YOU</h3>
      <p>
        Read the preview in the feed below. Type <code>yes</code> to approve —
        anything else (including nothing) denies.
      </p>
      <form className="gate-form" onSubmit={submit}>
        <input
          value={answer}
          onChange={(e) => setAnswer(e.target.value)}
          placeholder="type your answer"
          disabled={sent}
          autoFocus
        />
        <button className="btn btn-primary" disabled={sent}>
          {sent ? 'SENT' : 'SEND'}
        </button>
        <button type="button" className="btn" disabled={sent} onClick={() => send('')}>
          DENY
        </button>
      </form>
      {error && <p className="form-error">✗ {error}</p>}
    </div>
  )
}

export default function RunPage({ slug, runId }: { slug: string; runId: string }) {
  const { events, status, error } = useRunEvents(slug, runId)
  const feedRef = useRef<HTMLDivElement>(null)
  const [stick, setStick] = useState(true)

  const findings = useMemo(() => collectFindings(events), [events])
  const facts = useMemo(() => deriveFacts(findings), [findings])
  const finished = events.find((e) => e.type === 'run_finished')
  const hasCorr = !!findings.correlation && Object.keys(findings.correlation).length > 0

  // A gate is pending when the newest gate_prompt has no gate_answer after it
  // and the run is still alive — exactly the moment a human must act.
  const lastPrompt = [...events].reverse().find((e) => e.type === 'gate_prompt')
  const gatePending =
    status === 'running' &&
    lastPrompt !== undefined &&
    !events.some((e) => e.type === 'gate_answer' && e.seq > lastPrompt.seq)

  // Follow the feed like tail -f, but stop following the moment the reader
  // scrolls up to inspect something.
  useEffect(() => {
    const el = feedRef.current
    if (el && stick) el.scrollTop = el.scrollHeight
  }, [events, stick])

  const onScroll = () => {
    const el = feedRef.current
    if (!el) return
    setStick(el.scrollHeight - el.scrollTop - el.clientHeight < 60)
  }

  const t0 = events[0]?.ts ?? 0

  return (
    <section>
      <div className="runhead">
        <div>
          <h1>{runId}</h1>
          <a className="mono-dim" href={`#/p/${slug}`}>◂ #{slug}</a>
        </div>
        <span className="run-meta">
          {status === 'finished' && (
            <a className="btn btn-primary" href={`#/p/${slug}/prep`}>
              CONTINUE → PREPARATION
            </a>
          )}
          <span className="run-meta mono-dim">
            <span className={`lamp lamp-${status}`} /> {status.toUpperCase()}
          </span>
        </span>
      </div>

      {error && <div className="notice notice-err">{error}</div>}

      {gatePending && <GateBox slug={slug} runId={runId} />}

      <div className="run-grid">
        <div className="feed" ref={feedRef} onScroll={onScroll}>
          {events.length === 0 && !error && (
            <div className="evt">
              <span className="evt-t">+0.0s</span>
              <span className="badge badge-info">WAIT</span>
              <div className="evt-body">
                <p className="evt-think">waiting for the first journal event…</p>
              </div>
            </div>
          )}
          {events.map((e) => (
            <div key={e.seq} className={`evt evt-${e.type}`}>
              <span className="evt-t">+{(e.ts - t0).toFixed(1)}s</span>
              <span className={`badge ${BADGES[e.type]?.[1] ?? 'badge-info'}`}>
                {BADGES[e.type]?.[0] ?? e.type.toUpperCase()}
              </span>
              <div className="evt-body">
                <EventBody event={e} />
              </div>
            </div>
          ))}
        </div>

        <aside className="instruments">
          {facts.length === 0 && (
            <div className="panel inst">
              <h3 className="panel-title">INSTRUMENTS</h3>
              <p className="chart-note">warming up — panels light up as results arrive…</p>
            </div>
          )}
          {facts.length > 0 && (
            <Instrument title="SIGNALS">
              <div className="facts">
                {facts.map((f, i) => (
                  <div key={i} className={`fact fact-${f.tone}`}>
                    <span className="fact-glyph">{f.tone === 'warn' ? '▲' : '◆'}</span>
                    <p>{f.text}</p>
                  </div>
                ))}
              </div>
            </Instrument>
          )}
          {findings.targetBalance && (
            <Instrument title="TARGET BALANCE">
              <TargetBalanceChart balance={findings.targetBalance} />
            </Instrument>
          )}
          {findings.missingness && (
            <Instrument title="MISSINGNESS">
              <MissingnessChart miss={findings.missingness} />
            </Instrument>
          )}
          {hasCorr && (
            <Instrument title="CORRELATION WITH TARGET">
              <CorrelationChart corr={findings.correlation!} />
            </Instrument>
          )}
          {findings.drift && (
            <Instrument title="TRAIN / TEST DRIFT">
              <DriftChart drift={findings.drift} />
            </Instrument>
          )}
          {findings.numericSummary && (
            <Instrument title="NUMERIC SUMMARY">
              <NumericTable nums={findings.numericSummary} />
            </Instrument>
          )}
          {finished?.text && (
            <Instrument title="REPORT — report.md">
              <ReportView text={finished.text} />
            </Instrument>
          )}
        </aside>
      </div>
    </section>
  )
}
