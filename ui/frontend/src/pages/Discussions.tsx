import { useState } from 'react'
import type { NotebookRecord } from '../api'
import { api } from '../api'
import { useObserver } from '../hooks'
import { ReportView } from '../charts'

function NotebookCard({ nb }: { nb: NotebookRecord }) {
  return (
    <section className="panel notebook-card">
      <div className="notebook-head">
        <div>
          <h3 className="panel-title">{nb.title}</h3>
          <span className="mono-dim">
            {nb.author} · ▲ {nb.votes}
            {nb.cv_claim && ` · CV ${nb.cv_claim}`}
            {nb.lb_claim && ` · LB ${nb.lb_claim}`}
          </span>
        </div>
        <a className="mono-dim" href={nb.url} target="_blank" rel="noreferrer">
          OPEN ↗
        </a>
      </div>
      <p className="prose">{nb.summary}</p>
      {nb.models.length > 0 && (
        <div className="chips">
          {nb.models.map((m) => (
            <span key={m} className="chip">
              {m}
            </span>
          ))}
        </div>
      )}
      {nb.techniques.map((t, i) => (
        <details key={i} className="technique">
          <summary>
            <span className="technique-name">{t.name}</span>
            <span className="mono-dim"> — {t.why_it_matters}</span>
          </summary>
          <pre className="technique-code">{t.code_snippet}</pre>
          <p className="mono-dim">{t.snippet_explanation}</p>
        </details>
      ))}
    </section>
  )
}

export default function DiscussionsPage({ slug }: { slug: string }) {
  const { state, error, bump } = useObserver(slug)
  const [refreshError, setRefreshError] = useState<string | null>(null)
  const running = state?.status === 'running'

  const refresh = async () => {
    setRefreshError(null)
    try {
      await api.observerRefresh(slug)
      bump()
    } catch (e) {
      setRefreshError(String(e instanceof Error ? e.message : e))
    }
  }

  return (
    <section>
      <div className="page-head">
        <div>
          <h1>Discussions observer</h1>
          <span className="mono-dim">
            {state ? `${state.notebooks.length} notebooks summarized` : '…'}
            {running && ' · observing now'}
          </span>
        </div>
        <button className="btn" onClick={refresh} disabled={running}>
          {running ? '⟳ OBSERVING…' : '↻ REFRESH'}
        </button>
      </div>

      {(error || refreshError) && (
        <div className="notice notice-err">{error ?? refreshError}</div>
      )}
      {state?.status === 'failed' && (
        <div className="notice notice-err">
          The last observation failed — check the observer log; a missing
          ANTHROPIC_API_KEY or Kaggle credentials is the usual cause.
        </div>
      )}
      {state && state.status === 'none' && state.notebooks.length === 0 && (
        <div className="notice">
          <p>
            Nothing observed yet. REFRESH pulls the top community notebooks for this
            competition and summarizes what they do — techniques, models, claimed
            scores, and the exact code behind each technique.
          </p>
        </div>
      )}

      {state?.summary_md && (
        <div className="panel" style={{ marginBottom: 16 }}>
          <ReportView text={state.summary_md} />
        </div>
      )}

      {state?.notebooks.map((nb) => <NotebookCard key={nb.ref} nb={nb} />)}
    </section>
  )
}
