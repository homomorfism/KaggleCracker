import { useCallback, useEffect, useState } from 'react'
import {
  api,
  type ChatMessage,
  type Dashboard,
  type JournalEvent,
  type ObserverState,
  type Project,
  type RunStatus,
  type SetupStatus,
} from './api'

// Hand-rolled hash routing: four routes do not justify a router dependency,
// and the hash keeps the stdlib server's SPA fallback trivial.
//   #/            menu
//   #/new         create project
//   #/p/:slug     project
//   #/p/:slug/run/:id   run
const parseHash = () =>
  window.location.hash.replace(/^#\/?/, '').split('/').filter(Boolean)

export function useHashRoute(): string[] {
  const [route, setRoute] = useState<string[]>(parseHash)
  useEffect(() => {
    const onChange = () => setRoute(parseHash())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export function navigate(path: string) {
  window.location.hash = path
}

export function useProjects() {
  const [projects, setProjects] = useState<Project[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const reload = useCallback(() => {
    setError(null)
    api.listProjects().then(setProjects, (e) => setError(String(e.message ?? e)))
  }, [])
  useEffect(reload, [reload])
  return { projects, error, reload }
}

export function useProject(slug: string) {
  const [project, setProject] = useState<Project | null>(null)
  const [error, setError] = useState<string | null>(null)
  const reload = useCallback(() => {
    setError(null)
    api.getProject(slug).then(setProject, (e) => setError(String(e.message ?? e)))
  }, [slug])
  useEffect(reload, [reload])
  return { project, error, reload }
}

// Poll setup/status.json at 1s until it reaches a terminal state. `bump`
// restarts polling after a retry.
export function useSetupStatus(slug: string, enabled: boolean) {
  const [status, setStatus] = useState<SetupStatus | null>(null)
  const [tick, setTick] = useState(0)
  const bump = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    if (!enabled) return
    let alive = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const s = await api.setupStatus(slug)
        if (!alive) return
        setStatus(s)
        if (s.state !== 'done' && s.state !== 'failed' && s.state !== 'none')
          timer = window.setTimeout(poll, 1000)
      } catch {
        if (alive) timer = window.setTimeout(poll, 2000)
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, enabled, tick])

  return { status, bump }
}

// The EDA page's single poll: dashboard (version-gated), chat, agent status.
// Polls faster while the agent works so panels appear as they land.
export function useEda(slug: string) {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null)
  const [chat, setChat] = useState<ChatMessage[]>([])
  const [agent, setAgent] = useState<{ status: 'running' | 'idle'; run_id: string | null }>({
    status: 'idle',
    run_id: null,
  })
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const bump = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    let alive = true
    let timer: number | undefined
    let version = -1
    const poll = async () => {
      try {
        const state = await api.eda(slug, version)
        if (!alive) return
        setError(null)
        let running = false
        if (!state.unchanged && state.dashboard) {
          version = state.dashboard.version
          setDashboard(state.dashboard)
          setChat(state.chat ?? [])
          setAgent(state.agent ?? { status: 'idle', run_id: null })
          running = state.agent?.status === 'running'
        }
        // While the agent works, chat/journal move without a version bump, so
        // ask for the full payload again on the next poll.
        if (running) version = -1
        timer = window.setTimeout(poll, running ? 1200 : 2000)
      } catch (e) {
        if (!alive) return
        setError(String(e instanceof Error ? e.message : e))
        timer = window.setTimeout(poll, 3000)
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, tick])

  return { dashboard, chat, agent, error, bump }
}

// Observer state, polled while a refresh is running so notebooks stream in.
export function useObserver(slug: string) {
  const [state, setState] = useState<ObserverState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const bump = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    let alive = true
    let timer: number | undefined
    const poll = async () => {
      try {
        const s = await api.observer(slug)
        if (!alive) return
        setError(null)
        setState(s)
        if (s.status === 'running') timer = window.setTimeout(poll, 1500)
      } catch (e) {
        if (alive) setError(String(e instanceof Error ? e.message : e))
      }
    }
    poll()
    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, tick])

  return { state, error, bump }
}

// Poll the run journal exactly the way the backend intends: a since-cursor,
// one second apart, until the journal says the run reached a terminal state.
export function useRunEvents(slug: string, runId: string) {
  const [events, setEvents] = useState<JournalEvent[]>([])
  const [status, setStatus] = useState<RunStatus>('running')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let since = 0
    let alive = true
    let timer: number | undefined
    setEvents([])
    setStatus('running')
    setError(null)

    const poll = async () => {
      try {
        const page = await api.events(slug, runId, since)
        if (!alive) return
        since = page.next_since
        if (page.events.length) setEvents((prev) => [...prev, ...page.events])
        setStatus(page.status)
        if (page.status === 'running') timer = window.setTimeout(poll, 1000)
      } catch (e) {
        if (alive) setError(String(e instanceof Error ? e.message : e))
      }
    }
    poll()

    return () => {
      alive = false
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [slug, runId])

  return { events, status, error }
}
