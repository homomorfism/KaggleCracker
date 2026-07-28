import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import type { ExperimentMeta } from '../api'
import { api } from '../api'
import { navigate } from '../hooks'

function stateBadge(e: ExperimentMeta): { label: string; cls: string } {
  if (e.running) return { label: 'WORKING', cls: 'badge-info' }
  if (e.state === 'finished') return { label: 'FINISHED', cls: 'badge-ok' }
  if (e.state === 'plan_proposed') return { label: 'PLAN — YOUR REVIEW', cls: 'badge-gate' }
  return { label: 'DRAFT', cls: 'badge-amber' }
}

function CreateBox({ slug }: { slug: string }) {
  const [open, setOpen] = useState(false)
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      const meta = await api.createExperiment(slug, prompt.trim())
      navigate(`/p/${slug}/experiments/${meta.id}`)
    } catch (err) {
      setBusy(false)
      setError(String(err instanceof Error ? err.message : err))
    }
  }

  if (!open)
    return (
      <button className="btn btn-primary" onClick={() => setOpen(true)}>
        + CREATE EXPERIMENT
      </button>
    )
  return (
    <form className="panel exp-create" onSubmit={submit}>
      <h3 className="panel-title">WHAT EXPERIMENT DO YOU WANT TO MAKE?</h3>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            e.currentTarget.form?.requestSubmit()
          }
        }}
        rows={3}
        placeholder="e.g. gradient boosting with title/family-size features — or leave empty and press Enter to let the agent decide"
        autoFocus
      />
      <div className="setup-actions">
        <button className="btn btn-primary" disabled={busy}>
          {busy ? 'CREATING…' : prompt.trim() ? 'START' : 'START — AGENT DECIDES'}
        </button>
        <button type="button" className="btn" onClick={() => setOpen(false)} disabled={busy}>
          CANCEL
        </button>
      </div>
      {error && <p className="form-error">✗ {error}</p>}
    </form>
  )
}

export default function ExperimentsPage({ slug }: { slug: string }) {
  const [experiments, setExperiments] = useState<ExperimentMeta[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const list = await api.experiments(slug)
        if (!alive) return
        setError(null)
        setExperiments(list)
        // Keep the table live while anything is working.
        if (list.some((e) => e.running)) timer = window.setTimeout(poll, 2000)
      } catch (e) {
        if (alive) setError(String(e instanceof Error ? e.message : e))
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug])

  if (error) return <div className="notice notice-err">{error}</div>
  if (!experiments) return <div className="notice mono-dim">LOADING…</div>

  return (
    <section>
      <div className="page-head">
        <div>
          <h1>Experiments</h1>
          <span className="mono-dim">{experiments.length} on record</span>
        </div>
        <CreateBox slug={slug} />
      </div>

      {experiments.length === 0 && (
        <div className="notice">
          <p>
            No experiments yet. CREATE EXPERIMENT starts a planning conversation: the
            agent reads the EDA insights and community knowledge, asks you the open
            questions, proposes a plan — and only implements it after you approve.
          </p>
        </div>
      )}

      {experiments.length > 0 && (
        <div className="panel">
          <div className="table-scroll">
            <table className="stats">
              <thead>
                <tr>
                  <th>NAME</th>
                  <th>DESCRIPTION</th>
                  <th>STATE</th>
                  <th>CV SCORE</th>
                  <th>UPDATED</th>
                </tr>
              </thead>
              <tbody>
                {[...experiments].reverse().map((e) => {
                  const badge = stateBadge(e)
                  return (
                    <tr
                      key={e.id}
                      className="exp-row"
                      onClick={() => navigate(`/p/${slug}/experiments/${e.id}`)}
                    >
                      <td>{e.name}</td>
                      <td className="exp-desc">{e.description || '(agent-chosen)'}</td>
                      <td>
                        <span className={`badge ${badge.cls}`}>{badge.label}</span>
                      </td>
                      <td>{e.cv_score !== null ? e.cv_score.toFixed(5) : '—'}</td>
                      <td className="mono-dim">{e.updated}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
