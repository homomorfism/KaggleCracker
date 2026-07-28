import type { ReactNode } from 'react'
import { useHashRoute } from './hooks'
import Menu from './pages/Menu'
import CreateProject from './pages/CreateProject'
import ProjectPage from './pages/Project'
import PrepPage from './pages/Prep'
import RunPage from './pages/Run'

export default function App() {
  const route = useHashRoute()

  let view: ReactNode
  if (route.length === 0) view = <Menu />
  else if (route[0] === 'new') view = <CreateProject />
  else if (route[0] === 'p' && route.length === 2) view = <ProjectPage slug={route[1]} />
  else if (route[0] === 'p' && route.length === 3 && route[2] === 'prep')
    view = <PrepPage slug={route[1]} />
  else if (route[0] === 'p' && route.length === 4 && route[2] === 'run')
    view = <RunPage slug={route[1]} runId={route[3]} />
  else view = <Menu />

  return (
    <div className="console">
      <header className="topbar">
        <a className="brand" href="#/">
          <span className="brand-mark">▚▞</span>
          <span className="brand-name">KAGGLECRACKER</span>
          <span className="brand-sub">RECON CONSOLE</span>
        </a>
        <span className="topbar-status">
          <span className="lamp lamp-ok" /> LOCAL
        </span>
      </header>
      <main className="stage-area">{view}</main>
      <footer className="foot">
        stdlib api · journal-driven · core loop untouched
      </footer>
    </div>
  )
}
