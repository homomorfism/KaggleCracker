import { useEffect, useState } from 'react'
import type { Project, RunMode } from '../api'
import { api, fmtBytes } from '../api'
import { navigate, useProject } from '../hooks'
import { ReportView } from '../charts'

type StageState = 'done' | 'ready' | 'todo' | 'soon'

const STATE_GLYPH: Record<StageState, string> = {
  done: '✓',
  ready: '→',
  todo: '',
  soon: '',
}

function Stage({
  n,
  name,
  desc,
  state,
  href,
}: {
  n: number
  name: string
  desc: string
  state: StageState
  href?: string
}) {
  const body = (
    <>
      <span className="stage-n">STEP {n}</span>
      <span className="stage-name">
        {name} <em className={`stage-glyph stage-glyph-${state}`}>{STATE_GLYPH[state]}</em>
      </span>
      <span className="stage-desc">{desc}</span>
      {state === 'soon' && <span className="tag stage-tag">SOON</span>}
    </>
  )
  if (href && state !== 'soon')
    return <a className={`stage stage-${state}`} href={href}>{body}</a>
  return <div className={`stage stage-${state}`}>{body}</div>
}

// Where this project is in the pipeline — also the navigation between steps.
// Training and submission are the teammates' slices, visible so the shape of
// the whole system reads, marked SOON because this UI cannot drive them yet.
function Pipeline({ slug, project, planCount }: {
  slug: string
  project: Project
  planCount: number | null
}) {
  const finished = project.runs.filter((r) => r.status === 'finished').length
  const recon: StageState = finished > 0 ? 'done' : 'todo'
  const prep: StageState = (planCount ?? 0) > 0 ? 'done' : finished > 0 ? 'ready' : 'todo'

  return (
    <div className="stages">
      <Stage
        n={1}
        name="RECON"
        state={recon}
        desc={finished > 0 ? `${finished} finished run${finished > 1 ? 's' : ''}` : 'profile the data'}
      />
      <Stage
        n={2}
        name="PREPARE"
        state={prep}
        href={`#/p/${slug}/prep`}
        desc={
          planCount === null
            ? '…'
            : planCount > 0
              ? 'plan written — open'
              : finished > 0
                ? 'open when the report is in'
                : 'needs a recon run first'
        }
      />
      <Stage n={3} name="TRAIN" state="soon" desc="experiments on the plan" />
      <Stage n={4} name="SUBMIT" state="soon" desc="gated, quota-capped" />
    </div>
  )
}

export default function ProjectPage({ slug }: { slug: string }) {
  const { project, error } = useProject(slug)
  const [mode, setMode] = useState<RunMode>('demo')
  const [liveAvailable, setLiveAvailable] = useState<boolean | null>(null)
  const [planCount, setPlanCount] = useState<number | null>(null)
  const [report, setReport] = useState<string | null>(null)
  const [reportRun, setReportRun] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)

  useEffect(() => {
    api.capabilities().then((c) => setLiveAvailable(c.live), () => setLiveAvailable(false))
    api.plans(slug).then((p) => setPlanCount(p.length), () => setPlanCount(0))
  }, [slug])

  // The recon step's product is the report: show the newest one prominently.
  useEffect(() => {
    if (!project) return
    const latest = [...project.runs].reverse().find((r) => r.has_report)
    if (!latest) return
    setReportRun(latest.run_id)
    api.report(slug, latest.run_id).then(setReport, () => setReport(null))
  }, [slug, project])

  if (error)
    return (
      <div className="notice notice-err">
        <p>{error}</p>
        <a className="mono-dim" href="#/">◂ ALL PROJECTS</a>
      </div>
    )
  if (!project) return <div className="notice mono-dim">LOADING…</div>

  const start = async () => {
    setStarting(true)
    setStartError(null)
    try {
      const { run_id } = await api.startRun(slug, mode)
      navigate(`/p/${slug}/run/${run_id}`)
    } catch (e) {
      setStarting(false)
      setStartError(String(e instanceof Error ? e.message : e))
    }
  }

  return (
    <section>
      <div className="page-head">
        <div>
          <h1>{project.name}</h1>
          <span className="mono-dim">#{project.slug}</span>
        </div>
        <a className="mono-dim" href="#/">◂ ALL PROJECTS</a>
      </div>

      <Pipeline slug={slug} project={project} planCount={planCount} />

      <div className={report ? 'recon-grid' : ''} style={{ marginTop: 16 }}>
        <div>
          <div className="panel">
            <h3 className="panel-title">STEP 1 — RECONNAISSANCE</h3>
            <p className="prose" style={{ marginBottom: 12 }}>
              {project.target
                ? <>predicting <code className="target-code">{project.target}</code></>
                : 'no target column set — profiling will skip target checks'}
            </p>
            <div className="chips" style={{ marginBottom: 14 }}>
              {project.files.map((f) => (
                <span key={f.name} className="chip">
                  {f.name} <em>{fmtBytes(f.bytes)}</em>
                </span>
              ))}
              {project.files.length === 0 && (
                <span className="chip chip-empty">no data — upload a csv first</span>
              )}
            </div>
            <div className="mode-row">
              <button
                className={`btn btn-seg ${mode === 'demo' ? 'btn-on' : ''}`}
                onClick={() => setMode('demo')}
              >
                DEMO
              </button>
              <button
                className={`btn btn-seg ${mode === 'live' ? 'btn-on' : ''}`}
                disabled={liveAvailable === false}
                onClick={() => setMode('live')}
                title={liveAvailable === false ? 'set ANTHROPIC_API_KEY and restart the server' : ''}
              >
                LIVE
              </button>
            </div>
            {liveAvailable === false && mode === 'demo' && (
              <p className="mono-dim" style={{ marginBottom: 12 }}>
                live needs ANTHROPIC_API_KEY in the server environment
              </p>
            )}
            <button
              className="btn btn-primary btn-lg"
              onClick={start}
              disabled={starting || project.files.length === 0}
            >
              {starting ? 'LAUNCHING…' : `▶ ${report ? 'RE-RUN' : 'START'} ${mode.toUpperCase()} ANALYSIS`}
            </button>
            {startError && <p className="form-error">✗ {startError}</p>}
          </div>

          {project.runs.length > 0 && (
            <div className="panel" style={{ marginTop: 16 }}>
              <h3 className="panel-title">HISTORY</h3>
              {[...project.runs].reverse().map((r) => (
                <a key={r.run_id} className="run-row" href={`#/p/${slug}/run/${r.run_id}`}>
                  <span className="run-meta">
                    <span className={`lamp lamp-${r.status}`} />
                    {r.run_id}
                  </span>
                  <span className="mono-dim">{r.status.toUpperCase()}</span>
                </a>
              ))}
            </div>
          )}
        </div>

        {report && (
          <div className="panel report-panel">
            <div className="report-head">
              <h3 className="panel-title">REPORT — {reportRun}</h3>
              <a className="mono-dim" href={`#/p/${slug}/run/${reportRun}`}>VIEW RUN ▸</a>
            </div>
            <ReportView text={report} />
            <a className="btn btn-primary btn-lg btn-continue" href={`#/p/${slug}/prep`}>
              CONTINUE → DATA PREPARATION
            </a>
          </div>
        )}
      </div>
    </section>
  )
}
