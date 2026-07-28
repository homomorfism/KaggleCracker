// Thin client over ui/server.py. Same-origin /api in both modes: the Vite dev
// server proxies it, the built app is served by ui/server.py itself.

const BASE = '/api'

export type FileInfo = { name: string; bytes: number }
export type RunStatus = 'running' | 'finished' | 'failed'
export type RunInfo = { run_id: string; status: RunStatus; has_report: boolean }
export type Project = {
  name: string
  slug: string
  description: string
  target: string
  created: string
  files: FileInfo[]
  runs: RunInfo[]
}

// One journal line. `type` decides which optional fields are present — the
// same shape ui/journal.py writes.
export type JournalEvent = {
  seq: number
  ts: number
  type: string
  text?: string
  prompt?: string
  tool?: string
  args?: Record<string, unknown>
  data?: Record<string, unknown>
  kind?: string
  retryable?: boolean
  msg?: string
  reason?: string
  answer?: string
  scripted?: boolean
}

export type EventsPage = { events: JournalEvent[]; status: RunStatus; next_since: number }
export type RunMode = 'demo' | 'live'
export type PlanInfo = { dataset: string; markdown: string; modified: string }
export type Finding = {
  dataset: string
  check_name: string
  column_name: string
  value: Record<string, unknown>
  flagged: boolean
  rows_profiled: number
  created_at: string
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method }
  if (body !== undefined) {
    init.body = JSON.stringify(body)
    init.headers = { 'Content-Type': 'application/json' }
  }
  const res = await fetch(BASE + path, init)
  const text = await res.text()
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      msg = JSON.parse(text).error ?? msg
    } catch {
      // non-JSON error body; the status line is all we have
    }
    throw new Error(msg)
  }
  return JSON.parse(text) as T
}

export const api = {
  listProjects: () => req<{ projects: Project[] }>('GET', '/projects').then((r) => r.projects),

  createProject: (p: { name: string; description: string; target: string }) =>
    req<Project>('POST', '/projects', p),

  getProject: (slug: string) => req<Project>('GET', `/projects/${slug}`),

  uploadFile: async (slug: string, file: File): Promise<FileInfo> => {
    const res = await fetch(
      `${BASE}/projects/${slug}/files?name=${encodeURIComponent(file.name)}`,
      { method: 'POST', body: file },
    )
    if (!res.ok) {
      const body = await res.json().catch(() => null)
      throw new Error(body?.error ?? `upload failed: ${res.status}`)
    }
    return res.json()
  },

  startRun: (slug: string, mode: RunMode = 'demo', pace = 0.8) =>
    req<{ run_id: string }>('POST', `/projects/${slug}/runs`, { pace, mode }),

  capabilities: () => req<{ live: boolean }>('GET', '/capabilities'),

  gateAnswer: (slug: string, runId: string, answer: string) =>
    req<{ answer: string }>('POST', `/projects/${slug}/runs/${runId}/gate`, { answer }),

  plans: (slug: string) =>
    req<{ plans: PlanInfo[] }>('GET', `/projects/${slug}/plans`).then((r) => r.plans),

  findings: (slug: string, flaggedOnly = true) =>
    req<{ findings: Finding[] }>(
      'GET',
      `/projects/${slug}/findings${flaggedOnly ? '' : '?flagged=0'}`,
    ).then((r) => r.findings),

  events: (slug: string, runId: string, since: number) =>
    req<EventsPage>('GET', `/projects/${slug}/runs/${runId}/events?since=${since}`),

  report: async (slug: string, runId: string): Promise<string> => {
    const res = await fetch(`${BASE}/projects/${slug}/runs/${runId}/report`)
    if (!res.ok) throw new Error('no report yet')
    return res.text()
  },
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / (1024 * 1024)).toFixed(1)} MB`
}
