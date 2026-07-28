import { useCallback, useEffect, useState } from 'react'
import { api, type JournalEvent, type Project, type RunStatus } from './api'

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
