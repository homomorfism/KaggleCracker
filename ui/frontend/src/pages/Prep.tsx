import { useEffect, useState } from 'react'
import type { Finding, PlanInfo } from '../api'
import { api } from '../api'
import { ReportView } from '../charts'

// One scannable phrase per finding — the store's value JSON differs per
// check, so each check gets its own short rendering.
export function findingSummary(check: string, value: Record<string, unknown>): string {
  switch (check) {
    case 'missingness':
      return `${value.missing} missing (${((value.pct as number) * 100).toFixed(1)}%)`
    case 'dtypes': {
      const mixed = value.mixed ? 'mixed types — ' : ''
      return `${mixed}${value.int} int / ${value.float} float / ${value.text} text`
    }
    case 'cardinality':
      if (value.constant) return 'constant column'
      if (value.likely_id) return 'looks like an identifier'
      return `${value.distinct} distinct values`
    case 'train_test_drift':
      if (value.smd !== undefined) return `numeric drift, SMD ${value.smd}`
      if (value.tvd !== undefined)
        return `categorical drift, TVD ${value.tvd}${value.unseen_count ? `, ${value.unseen_count} unseen categories` : ''}`
      return 'the two files disagree about which columns exist'
    case 'target_balance':
      return Object.entries(value)
        .map(([cls, v]) => `${cls} ${(((v as { ratio: number }).ratio) * 100).toFixed(1)}%`)
        .join(' · ')
    case 'numeric_summary': {
      const v = value as { min?: number; max?: number; mean?: number }
      return `min ${v.min} · mean ${v.mean} · max ${v.max}`
    }
    case 'correlation_with_target':
      return typeof value === 'object' && 'note' in value
        ? String((value as { note: string }).note)
        : `r = ${value}`
    default:
      return JSON.stringify(value)
  }
}

export default function PrepPage({ slug }: { slug: string }) {
  const [plans, setPlans] = useState<PlanInfo[] | null>(null)
  const [findings, setFindings] = useState<Finding[] | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.plans(slug).then(setPlans, (e) => setError(String(e.message ?? e)))
  }, [slug])
  useEffect(() => {
    setFindings(null)
    api.findings(slug, !showAll).then(setFindings, (e) => setError(String(e.message ?? e)))
  }, [slug, showAll])

  if (error)
    return (
      <div className="notice notice-err">
        <p>{error}</p>
        <a className="mono-dim" href="#/">◂ ALL PROJECTS</a>
      </div>
    )
  if (plans === null) return <div className="notice mono-dim">LOADING…</div>

  const nothingYet = plans.length === 0 && (findings?.length ?? 0) === 0 && !showAll

  return (
    <section>
      <div className="page-head">
        <div>
          <h1>Data preparation</h1>
          <span className="mono-dim">STEP 2 · <a className="mono-dim" href={`#/p/${slug}`}>◂ #{slug}</a></span>
        </div>
      </div>

      <p className="prose">
        Reconnaissance produces two artefacts this step works from: the{' '}
        <strong>flagged findings</strong> — columns where the profile says a decision is
        unavoidable — and the <strong>preprocessing plan</strong> the agent wrote, grounded
        in those findings. The plan is what the training step will execute against.
      </p>

      {nothingYet && (
        <div className="panel">
          <p className="mono-dim">
            nothing here yet — run an analysis first; it records findings and writes the plan
          </p>
        </div>
      )}

      {!nothingYet && (
        <div className="panel" style={{ marginBottom: 16 }}>
          <h3 className="panel-title">PROFILE FINDINGS</h3>
          <div className="mode-row">
            <button
              className={`btn btn-seg ${!showAll ? 'btn-on' : ''}`}
              onClick={() => setShowAll(false)}
            >
              FLAGGED
            </button>
            <button
              className={`btn btn-seg ${showAll ? 'btn-on' : ''}`}
              onClick={() => setShowAll(true)}
            >
              ALL
            </button>
          </div>
          {findings === null && <p className="mono-dim">LOADING…</p>}
          {findings !== null && findings.length === 0 && (
            <p className="mono-dim">no findings recorded{showAll ? '' : ' as flagged'}</p>
          )}
          {findings !== null && findings.length > 0 && (
            <div className="table-scroll">
              <table className="stats prep-table">
                <thead>
                  <tr><th></th><th>dataset</th><th>check</th><th>column</th><th>finding</th></tr>
                </thead>
                <tbody>
                  {findings.map((f, i) => (
                    <tr key={i}>
                      <td>{f.flagged ? <span className="flagtag">▲</span> : ''}</td>
                      <td>{f.dataset}</td>
                      <td>{f.check_name}</td>
                      <td>{f.column_name}</td>
                      <td>{findingSummary(f.check_name, f.value)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <span className="chart-caption mono-dim">▲ = needs a decision in the plan</span>
        </div>
      )}

      {plans.length > 0 && (
        <div className="panel">
          <h3 className="panel-title">PREPROCESSING PLANS</h3>
          {plans.map((p) => (
            <details key={p.dataset} className="plan" open={plans.length === 1}>
              <summary>
                PLAN — {p.dataset} <em className="mono-dim">updated {p.modified}</em>
              </summary>
              <div className="plan-body">
                <ReportView text={p.markdown} />
              </div>
            </details>
          ))}
        </div>
      )}
    </section>
  )
}
