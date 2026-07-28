import type { ReactNode } from 'react'
import { useHashRoute } from './hooks'
import Menu from './pages/Menu'
import CreateProject from './pages/CreateProject'
import ProjectPage from './pages/Project'
import PrepPage from './pages/Prep'
import RunPage from './pages/Run'
import ProjectShell, { type Section } from './pages/ProjectShell'
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

export default function App() {
  const route = useHashRoute()

  let view: ReactNode
  if (route.length === 0) view = <Menu />
  else if (route[0] === 'new') view = <CreateProject />
  else if (route[0] === 'p' && route.length >= 2) {
    const slug = route[1]
    const { section, view: inner } = projectView(slug, route)
    view = (
      <ProjectShell slug={slug} section={section}>
        {inner}
      </ProjectShell>
    )
  } else view = <Menu />

  return (
    <div className="console">
      <header className="topbar">
        <a className="brand" href="#/">
          <span className="brand-mark">▚▞</span>
          <span className="brand-name">KAGGLECRACKER</span>
          <span className="brand-sub">COMPETITION WORKBENCH</span>
        </a>
        <span className="topbar-status">
          <span className="lamp lamp-ok" /> LOCAL
        </span>
      </header>
      <main className="stage-area">{view}</main>
      <footer className="foot">
        journal-driven · agents on subprocesses · dashboards from files
      </footer>
    </div>
  )
}
