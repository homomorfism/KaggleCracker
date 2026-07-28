import type { ReactNode } from 'react'
import { useProject } from '../hooks'

export type Section =
  | 'info'
  | 'eda'
  | 'experiments'
  | 'discussions'
  | 'leaderboard'
  | 'recon'
  | 'prep'
  | 'run'

const NAV: { key: Section; label: string; desc: string; soon?: boolean }[] = [
  { key: 'info', label: 'COMPETITION', desc: 'metric · description · data' },
  { key: 'eda', label: 'DATASET EDA', desc: 'agent dashboard + chat' },
  { key: 'experiments', label: 'EXPERIMENTS', desc: 'plan · approve · run' },
  { key: 'discussions', label: 'DISCUSSIONS', desc: 'community observer' },
  { key: 'leaderboard', label: 'LEADERBOARD', desc: 'pipelines · cv scores' },
]

const UTILITY: { key: Section; label: string }[] = [
  { key: 'recon', label: 'RECON RUNS' },
  { key: 'prep', label: 'DATA PREP' },
]

export default function ProjectShell({
  slug,
  section,
  children,
}: {
  slug: string
  section: Section
  children: ReactNode
}) {
  const { project } = useProject(slug)
  const setup = project?.setup_state ?? 'none'
  const lamp =
    setup === 'failed' ? 'failed' : setup === 'done' || setup === 'none' ? 'ok' : 'running'

  return (
    <div className="shell">
      <nav className="sidenav">
        <div className="sidenav-head">
          <span className={`lamp lamp-${lamp}`} />
          <div>
            <div className="sidenav-name">{project?.name ?? slug}</div>
            <div className="mono-dim">#{slug}</div>
          </div>
        </div>

        <div className="sidenav-group">
          {NAV.map((item) =>
            item.soon ? (
              <div key={item.key} className="sidenav-item sidenav-soon">
                <span className="sidenav-label">
                  {item.label} <span className="tag">SOON</span>
                </span>
                <span className="sidenav-desc">{item.desc}</span>
              </div>
            ) : (
              <a
                key={item.key}
                className={`sidenav-item ${section === item.key ? 'sidenav-on' : ''}`}
                href={`#/p/${slug}/${item.key}`}
              >
                <span className="sidenav-label">{item.label}</span>
                <span className="sidenav-desc">{item.desc}</span>
              </a>
            ),
          )}
        </div>

        <div className="sidenav-group sidenav-utility">
          <span className="sidenav-caption">RECON CONSOLE</span>
          {UTILITY.map((item) => (
            <a
              key={item.key}
              className={`sidenav-item ${section === item.key || (item.key === 'recon' && section === 'run') ? 'sidenav-on' : ''}`}
              href={`#/p/${slug}/${item.key}`}
            >
              <span className="sidenav-label">{item.label}</span>
            </a>
          ))}
        </div>

        <a className="sidenav-back mono-dim" href="#/">
          ◂ ALL PROJECTS
        </a>
      </nav>
      <div className="shell-stage">{children}</div>
    </div>
  )
}
