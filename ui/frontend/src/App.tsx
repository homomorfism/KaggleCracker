import type { ReactNode } from 'react'
import { useHashRoute, useProject } from './hooks'
import Menu from './pages/Menu'
import CreateProject from './pages/CreateProject'
import ProjectPage from './pages/Project'
import PrepPage from './pages/Prep'
import RunPage from './pages/Run'
import type { Section } from './pages/ProjectShell'
import InfoPage from './pages/Info'
import EdaPage from './pages/Eda'
import ExperimentsPage from './pages/Experiments'
import ExperimentDetailPage from './pages/ExperimentDetail'
import DiscussionsPage from './pages/Discussions'
import LeaderboardPage from './pages/Leaderboard'

function projectView(slug: string, route: string[]): { section: Section; view: ReactNode } {
  const sub = route[2] ?? 'info'
  if (sub === 'eda') return { section: 'eda', view: <EdaPage slug={slug} /> }
  // 'train' survives as an alias so old links land on the renamed section.
  if (sub === 'experiments' || sub === 'train') {
    if (route.length === 4)
      return {
        section: 'experiments',
        view: <ExperimentDetailPage slug={slug} expId={route[3]} />,
      }
    return { section: 'experiments', view: <ExperimentsPage slug={slug} /> }
  }
  if (sub === 'discussions')
    return { section: 'discussions', view: <DiscussionsPage slug={slug} /> }
  if (sub === 'leaderboard')
    return { section: 'leaderboard', view: <LeaderboardPage slug={slug} /> }
  if (sub === 'recon') return { section: 'recon', view: <ProjectPage slug={slug} /> }
  if (sub === 'prep') return { section: 'prep', view: <PrepPage slug={slug} /> }
  if (sub === 'run' && route.length === 4)
    return { section: 'run', view: <RunPage slug={slug} runId={route[3]} /> }
  return { section: 'info', view: <InfoPage slug={slug} /> }
}

const SECTIONS: { key: Section; label: string; icon: string }[] = [
  { key: 'info', label: 'Competition', icon: '◈' },
  { key: 'eda', label: 'Dataset EDA', icon: '▤' },
  { key: 'experiments', label: 'Experiments', icon: '◉' },
  { key: 'discussions', label: 'Discussions', icon: '✎' },
  { key: 'leaderboard', label: 'Leaderboard', icon: '▲' },
]

// Recon runs and data prep live under Dataset EDA as tabs, not as their own
// sidebar group — they are the data-understanding workflow, one place.
const EDA_SECTIONS: Section[] = ['eda', 'recon', 'prep', 'run']

function EdaTabs({ slug, section }: { slug: string; section: Section }) {
  const active = section === 'run' ? 'recon' : section
  const tabs = [
    { key: 'eda', label: 'DASHBOARD', href: `#/p/${slug}/eda` },
    { key: 'recon', label: 'RECON RUNS', href: `#/p/${slug}/recon` },
    { key: 'prep', label: 'DATA PREP', href: `#/p/${slug}/prep` },
  ]
  return (
    <div className="tabbar">
      {tabs.map((t) => (
        <a key={t.key} className={`tab ${active === t.key ? 'tab-on' : ''}`} href={t.href}>
          {t.label}
        </a>
      ))}
    </div>
  )
}

function ProjectNav({ slug, section }: { slug: string; section: Section }) {
  const { project } = useProject(slug)
  const setup = project?.setup_state ?? 'none'
  const lamp =
    setup === 'failed' ? 'failed' : setup === 'done' || setup === 'none' ? 'ok' : 'running'

  return (
    <>
      <div className="side-project">
        <span className={`lamp lamp-${lamp}`} />
        <div>
          <div className="side-project-name">{project?.name ?? slug}</div>
          <div className="mono-dim">#{slug}</div>
        </div>
      </div>
      <div className="side-group">
        {SECTIONS.map((item) => {
          const on =
            section === item.key ||
            (item.key === 'eda' && EDA_SECTIONS.includes(section))
          return (
            <a
              key={item.key}
              className={`side-item ${on ? 'side-on' : ''}`}
              href={`#/p/${slug}/${item.key}`}
            >
              <span className="side-icon">{item.icon}</span>
              {item.label}
            </a>
          )
        })}
      </div>
    </>
  )
}

export default function App() {
  const route = useHashRoute()

  const inProject = route[0] === 'p' && route.length >= 2
  const slug = inProject ? route[1] : null

  let view: ReactNode
  let section: Section = 'info'
  if (route.length === 0) view = <Menu />
  else if (route[0] === 'new') view = <CreateProject />
  else if (slug) {
    const r = projectView(slug, route)
    section = r.section
    view = EDA_SECTIONS.includes(r.section) ? (
      <>
        <EdaTabs slug={slug} section={r.section} />
        {r.view}
      </>
    ) : (
      r.view
    )
  } else view = <Menu />

  return (
    <div className="app">
      <aside className="sidebar">
        <a className="side-brand" href="#/">
          <span className="brand-mark">▚▞</span>
          KaggleCracker
        </a>
        <div className="side-group">
          <a className={`side-item ${route.length === 0 ? 'side-on' : ''}`} href="#/">
            <span className="side-icon">⌂</span>
            Projects
          </a>
          <a className={`side-item ${route[0] === 'new' ? 'side-on' : ''}`} href="#/new">
            <span className="side-icon">＋</span>
            New project
          </a>
        </div>
        {slug && <ProjectNav slug={slug} section={section} />}
        <div className="side-foot">
          <span className="lamp lamp-ok" /> local · journal-driven
        </div>
      </aside>
      <main className="main">{view}</main>
    </div>
  )
}
