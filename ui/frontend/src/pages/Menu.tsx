import type { Project, RunStatus } from '../api'
import { fmtBytes } from '../api'
import { useProjects } from '../hooks'

function projectState(p: Project): { label: string; status: RunStatus | 'idle' } {
  if (p.runs.some((r) => r.status === 'running')) return { label: 'ANALYSING', status: 'running' }
  const last = p.runs[p.runs.length - 1]
  if (!last) return { label: 'NO RUNS YET', status: 'idle' }
  if (last.status === 'failed') return { label: 'LAST RUN FAILED', status: 'failed' }
  return { label: `${p.runs.length} RUN${p.runs.length > 1 ? 'S' : ''} DONE`, status: 'finished' }
}

function ProjectCard({ project, index }: { project: Project; index: number }) {
  const state = projectState(project)
  return (
    <a
      className="card"
      href={`#/p/${project.slug}`}
      style={{ animationDelay: `${index * 70}ms` }}
    >
      <div className="card-top">
        <span className={`lamp lamp-${state.status}`} />
        <span className="card-state">{state.label}</span>
      </div>
      <h2 className="card-name">{project.name}</h2>
      {project.description && <p className="card-desc">{project.description}</p>}
      <div className="chips">
        {project.files.map((f) => (
          <span key={f.name} className="chip">
            {f.name} <em>{fmtBytes(f.bytes)}</em>
          </span>
        ))}
        {project.files.length === 0 && <span className="chip chip-empty">no data yet</span>}
      </div>
      <div className="card-foot">
        <span className="mono-dim">{project.target ? `target: ${project.target}` : 'no target set'}</span>
        <span className="card-go">OPEN ▸</span>
      </div>
    </a>
  )
}

export default function Menu() {
  const { projects, error, reload } = useProjects()

  if (error)
    return (
      <div className="notice notice-err">
        <p>Cannot reach the API server: {error}</p>
        <p className="mono-dim">start it with: python -m ui.server</p>
        <button className="btn" onClick={reload}>RETRY</button>
      </div>
    )
  if (!projects) return <div className="notice mono-dim">SCANNING…</div>

  return (
    <section>
      <div className="page-head">
        <h1>Competitions</h1>
        <span className="mono-dim">
          {projects.length} PROJECT{projects.length === 1 ? '' : 'S'} ON DISK
        </span>
      </div>
      <div className="grid">
        {projects.map((p, i) => (
          <ProjectCard key={p.slug} project={p} index={i} />
        ))}
        <a className="card card-new" href="#/new" style={{ animationDelay: `${projects.length * 70}ms` }}>
          <span className="card-new-plus">+</span>
          NEW PROJECT
        </a>
      </div>
    </section>
  )
}
