// Thin client over ui/server.py. Same-origin /api in both modes: the Vite dev
// server proxies it, the built app is served by ui/server.py itself.

const BASE = '/api'

export type FileInfo = { name: string; bytes: number }
export type RunStatus = 'running' | 'finished' | 'failed'
export type RunInfo = { run_id: string; status: RunStatus; has_report: boolean }
export type SetupState =
  | 'none'
  | 'pending'
  | 'fetching_metadata'
  | 'downloading'
  | 'extracting'
  | 'done'
  | 'failed'
export type Project = {
  name: string
  slug: string
  description: string
  target: string
  competition: string
  created: string
  setup_state: SetupState
  files: FileInfo[]
  runs: RunInfo[]
}

export type SetupStatus = {
  state: SetupState
  ts?: string
  bytes_done?: number
  bytes_total?: number
  files?: string[]
  error?: { kind: string; msg: string; actionable: string }
}

export type CompetitionMeta = {
  slug: string
  title: string
  description: string
  evaluation_metric: string
  deadline: string
  category: string
  reward: string
  url: string
  files: FileInfo[]
}

export type PanelType =
  | 'histogram'
  | 'bar'
  | 'scatter'
  | 'heatmap'
  | 'line'
  | 'table'
  | 'stat'
  | 'markdown'
export type Panel = {
  id: string
  type: PanelType
  title: string
  commentary?: string
  data: Record<string, unknown>
}
export type Dashboard = { version: number; updated: string; panels: Panel[] }
export type ChatMessage = { seq: number; role: 'user' | 'assistant'; text: string; ts: string }
export type EdaState = {
  unchanged?: boolean
  version?: number
  dashboard?: Dashboard
  chat?: ChatMessage[]
  agent?: { status: 'running' | 'idle'; run_id: string | null }
}

export type Technique = {
  name: string
  why_it_matters: string
  code_snippet: string
  snippet_explanation: string
}
export type NotebookRecord = {
  ref: string
  url: string
  title: string
  author: string
  votes: number
  pulled_at: string
  summary: string
  models: string[]
  cv_claim: string
  lb_claim: string
  techniques: Technique[]
}
export type ObserverState = {
  status: RunStatus | 'none'
  notebooks: NotebookRecord[]
  summary_md: string
}

export type LeaderboardEntry = {
  experiment_id: string
  cv_score: number
  fold_scores?: number[]
  notes?: string
}

export type ExperimentState = 'draft' | 'plan_proposed' | 'finished'
export type ExperimentMeta = {
  id: string
  name: string
  description: string
  state: ExperimentState
  cv_score: number | null
  result_summary: string
  created: string
  updated: string
  running: boolean
}
export type ExperimentDetail = {
  experiment: ExperimentMeta
  plan_md: string
  messages: ChatMessage[]
  latest_turn: string | null
  turns: { id: string; status: RunStatus }[]
  code: { name: string; source: string }[]
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
  // prep-run events
  action?: string
  name?: string
  source?: string
  rows?: number
  columns?: number
  bytes?: number
}

export type EventsPage = { events: JournalEvent[]; status: RunStatus; next_since: number }
export type RunMode = 'demo' | 'live'
export type PreparedFile = { name: string; source: string; rows: number; columns: number; bytes: number }
export type PrepManifest = {
  created?: string
  train?: string
  decisions?: {
    dropped: { column: string; why: string }[]
    coerced: string[]
    imputed: { column: string; fill: unknown }[]
    drift_notes: string[]
  }
  files?: PreparedFile[]
}
export type PrepState = {
  status: RunStatus | 'none'
  events: JournalEvent[]
  next_since: number
  manifest: PrepManifest
  log: string
}
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

  createProject: (p: {
    name: string
    description: string
    target: string
    competition_url?: string
  }) => req<Project>('POST', '/projects', p),

  competition: (slug: string) => req<CompetitionMeta>('GET', `/projects/${slug}/competition`),

  setupStatus: (slug: string) => req<SetupStatus>('GET', `/projects/${slug}/setup`),

  setupRetry: (slug: string) => req<{ state: string }>('POST', `/projects/${slug}/setup/retry`),

  eda: (slug: string, version = -1) =>
    req<EdaState>('GET', `/projects/${slug}/eda?version=${version}`),

  edaMessage: (slug: string, text: string) =>
    req<{ message: ChatMessage }>('POST', `/projects/${slug}/eda/messages`, { text }),

  edaEvents: (slug: string, runId: string, since: number) =>
    req<EventsPage>('GET', `/projects/${slug}/eda/runs/${runId}/events?since=${since}`),

  observer: (slug: string) => req<ObserverState>('GET', `/projects/${slug}/observer`),

  observerRefresh: (slug: string) =>
    req<{ status: string }>('POST', `/projects/${slug}/observer/refresh`),

  observerEvents: (slug: string, since: number) =>
    req<EventsPage & { status: RunStatus | 'none' }>(
      'GET',
      `/projects/${slug}/observer/events?since=${since}`,
    ),

  experiments: (slug: string) =>
    req<{ experiments: ExperimentMeta[] }>('GET', `/projects/${slug}/experiments`).then(
      (r) => r.experiments,
    ),

  createExperiment: (slug: string, prompt: string) =>
    req<ExperimentMeta>('POST', `/projects/${slug}/experiments`, { prompt }),

  experimentDetail: (slug: string, expId: string) =>
    req<ExperimentDetail>('GET', `/projects/${slug}/experiments/${expId}`),

  experimentMessage: (slug: string, expId: string, text: string) =>
    req<{ message: ChatMessage }>('POST', `/projects/${slug}/experiments/${expId}/messages`, {
      text,
    }),

  experimentStop: (slug: string, expId: string) =>
    req<{ stopped: boolean }>('POST', `/projects/${slug}/experiments/${expId}/stop`),

  experimentEvents: (slug: string, expId: string, turn: string, since: number) =>
    req<EventsPage>(
      'GET',
      `/projects/${slug}/experiments/${expId}/runs/${turn}/events?since=${since}`,
    ),

  leaderboard: (slug: string) =>
    req<{ entries: LeaderboardEntry[] }>('GET', `/projects/${slug}/leaderboard`).then(
      (r) => r.entries,
    ),

  prepStatus: (slug: string) => req<PrepState>('GET', `/projects/${slug}/prep`),

  prepRun: (slug: string) => req<{ status: string }>('POST', `/projects/${slug}/prep/run`),

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

  capabilities: () => req<{ live: boolean; kaggle: boolean }>('GET', '/capabilities'),

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
