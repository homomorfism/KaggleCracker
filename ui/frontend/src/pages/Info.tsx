import { useEffect, useState } from 'react'
import type { CompetitionMeta, SetupStatus } from '../api'
import { api, fmtBytes } from '../api'
import { useProject, useSetupStatus } from '../hooks'
import { ReportView } from '../charts'

function pct(done: number, total: number): number {
  if (!total) return 0
  return Math.min(100, Math.round((done / total) * 100))
}

function SetupBanner({
  slug,
  status,
  onRetried,
}: {
  slug: string
  status: SetupStatus
  onRetried: () => void
}) {
  const [retrying, setRetrying] = useState(false)

  if (status.state === 'done' || status.state === 'none') return null

  if (status.state === 'failed') {
    const err = status.error
    const rulesUrl =
      err?.kind === 'rules_not_accepted' ? `https://www.kaggle.com/c/${slug}/rules` : null
    return (
      <div className="notice notice-err setup-banner">
        <p>
          <strong>SETUP FAILED</strong> — {err?.msg ?? 'unknown error'}
        </p>
        {err?.actionable && <p>{err.actionable}</p>}
        <div className="setup-actions">
          {rulesUrl && (
            <a className="btn" href={rulesUrl} target="_blank" rel="noreferrer">
              OPEN RULES PAGE ↗
            </a>
          )}
          <button
            className="btn btn-primary"
            disabled={retrying}
            onClick={async () => {
              setRetrying(true)
              try {
                await api.setupRetry(slug)
                onRetried()
              } finally {
                setRetrying(false)
              }
            }}
          >
            {retrying ? 'RETRYING…' : '↻ RETRY SETUP'}
          </button>
        </div>
      </div>
    )
  }

  const label =
    status.state === 'downloading'
      ? `DOWNLOADING COMPETITION DATA — ${fmtBytes(status.bytes_done ?? 0)}${
          status.bytes_total ? ` of ~${fmtBytes(status.bytes_total)}` : ''
        }`
      : status.state === 'extracting'
        ? 'EXTRACTING ARCHIVE…'
        : 'FETCHING COMPETITION METADATA…'
  const width = status.state === 'downloading' ? pct(status.bytes_done ?? 0, status.bytes_total ?? 0) : null

  return (
    <div className="notice setup-banner">
      <p>
        <span className="lamp lamp-running" /> {label}
      </p>
      <div className="progress-track">
        <div
          className={`progress-fill ${width === null ? 'progress-indeterminate' : ''}`}
          style={width === null ? undefined : { width: `${width}%` }}
        />
      </div>
    </div>
  )
}

export default function InfoPage({ slug }: { slug: string }) {
  const { project, reload } = useProject(slug)
  const isCompetition = !!project?.competition
  const { status, bump } = useSetupStatus(slug, isCompetition)
  const [meta, setMeta] = useState<CompetitionMeta | null>(null)

  const setupDone = status?.state === 'done' || project?.setup_state === 'done'
  useEffect(() => {
    if (!isCompetition) return
    api.competition(slug).then(setMeta, () => setMeta(null))
  }, [slug, isCompetition, setupDone])

  // Refresh the file list once the download lands.
  useEffect(() => {
    if (setupDone) reload()
  }, [setupDone, reload])

  if (!project) return <div className="notice mono-dim">LOADING…</div>

  return (
    <section>
      <div className="page-head">
        <div>
          <h1>{meta?.title ?? project.name}</h1>
          <span className="mono-dim">
            {meta?.category ? `${meta.category} · ` : ''}
            {meta?.deadline ? `deadline ${meta.deadline.slice(0, 10)}` : `#${slug}`}
          </span>
        </div>
        {meta?.url && (
          <a className="mono-dim" href={meta.url} target="_blank" rel="noreferrer">
            KAGGLE ↗
          </a>
        )}
      </div>

      {status && (
        <SetupBanner
          slug={slug}
          status={status}
          onRetried={() => {
            bump()
          }}
        />
      )}

      {!isCompetition && (
        <div className="notice">
          <p>
            This project is not linked to a Kaggle competition — data arrives by hand
            upload on the RECON RUNS page.
          </p>
        </div>
      )}

      <div className="info-grid">
        <div className="panel">
          <h3 className="panel-title">EVALUATION</h3>
          {meta?.evaluation_metric ? (
            <p className="metric-badge">{meta.evaluation_metric}</p>
          ) : (
            <p className="mono-dim">metric arrives with setup</p>
          )}
          {project.target && (
            <p className="prose">
              target column: <code className="target-code">{project.target}</code>
            </p>
          )}
          {meta?.reward && <p className="mono-dim">reward: {meta.reward}</p>}
        </div>

        <div className="panel">
          <h3 className="panel-title">DATASET STRUCTURE</h3>
          {project.files.length === 0 && (
            <p className="mono-dim">no files yet — they appear when the download finishes</p>
          )}
          {project.files.length > 0 && (
            <div className="table-scroll">
              <table className="stats">
                <thead>
                  <tr>
                    <th>FILE</th>
                    <th>SIZE</th>
                  </tr>
                </thead>
                <tbody>
                  {project.files.map((f) => (
                    <tr key={f.name}>
                      <td>{f.name}</td>
                      <td>{fmtBytes(f.bytes)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {(meta?.description || project.description) && (
        <div className="panel" style={{ marginTop: 16 }}>
          <h3 className="panel-title">DESCRIPTION</h3>
          <ReportView text={meta?.description || project.description} />
        </div>
      )}
    </section>
  )
}
