// Turn raw journal events into the structured findings the instruments render.
// Only profile results for the FIRST profiled path (the train file) feed the
// charts — a later profile of original.csv must not overwrite train's stats.

import type { JournalEvent } from './api'

export type MissEntry = { missing: number; pct: number }
export type DtypeEntry = {
  int: number
  float: number
  text: number
  missing: number
  majority: string
  mixed: boolean
}
export type CardEntry = { distinct: number; likely_id: boolean; constant: boolean }
export type BalanceEntry = { count: number; ratio: number }
export type NumericEntry = {
  count: number
  min: number
  max: number
  mean: number
  std: number
  median: number
  p5: number
  p95: number
}
export type DriftColumn =
  | { type: 'numeric'; mean: number; mean_compare: number; smd: number; drifted: boolean }
  | {
      type: 'categorical'
      tvd: number
      unseen_in_compare: string[]
      unseen_count: number
      drifted: boolean
    }
export type Drift = {
  only_in_this_file: string[]
  only_in_compare: string[]
  columns: Record<string, DriftColumn>
  flagged: string[]
}

export type Findings = {
  path?: string
  rowsProfiled?: number
  columns?: string[]
  missingness?: Record<string, MissEntry>
  dtypes?: Record<string, DtypeEntry>
  cardinality?: Record<string, CardEntry>
  targetBalance?: Record<string, BalanceEntry>
  numericSummary?: Record<string, NumericEntry>
  correlation?: Record<string, number>
  correlationNote?: string
  drift?: Drift
}

export function collectFindings(events: JournalEvent[]): Findings {
  const out: Findings = {}
  for (const e of events) {
    if (e.type !== 'tool_ok' || e.tool !== 'profile_dataset' || !e.data) continue
    const d = e.data as Record<string, unknown>
    const path = d.path as string
    if (out.path === undefined) out.path = path
    if (path !== out.path) continue

    out.rowsProfiled = (d.rows_profiled as number) ?? out.rowsProfiled
    out.columns = (d.columns as string[]) ?? out.columns
    if (d.missingness) out.missingness = d.missingness as Findings['missingness']
    if (d.dtypes) out.dtypes = d.dtypes as Findings['dtypes']
    if (d.cardinality) out.cardinality = d.cardinality as Findings['cardinality']
    if (d.target_balance) out.targetBalance = d.target_balance as Findings['targetBalance']
    if (d.numeric_summary) out.numericSummary = d.numeric_summary as Findings['numericSummary']
    if (d.correlation_with_target) {
      const corr = d.correlation_with_target as Record<string, unknown>
      if (typeof corr.note === 'string') out.correlationNote = corr.note
      else out.correlation = corr as Record<string, number>
    }
    if (d.train_test_drift) out.drift = d.train_test_drift as Drift
  }
  return out
}

export type Fact = { tone: 'info' | 'warn'; text: string }

const pct = (x: number) => `${Math.round(x * 100)}%`

// The "interesting facts" cards: short sentences a person scans while the
// model works. Warnings first, capped, never silently — the cap is visible
// as a final "+N more" fact.
export function deriveFacts(f: Findings): Fact[] {
  const facts: Fact[] = []

  if (f.targetBalance) {
    const classes = Object.entries(f.targetBalance)
    const [topName, top] = classes[0] ?? []
    if (top && top.ratio >= 0.7)
      facts.push({
        tone: 'warn',
        text: `Heavy imbalance: “${topName}” is ${pct(top.ratio)} of rows — plain accuracy would lie here.`,
      })
    else if (classes.length)
      facts.push({ tone: 'info', text: `${classes.length} target classes, largest is ${pct(top!.ratio)}.` })
  }

  if (f.missingness) {
    const holes = Object.entries(f.missingness).filter(([, m]) => m.pct >= 0.2)
    for (const [name, m] of holes.slice(0, 2))
      facts.push({ tone: 'warn', text: `“${name}” is ${pct(m.pct)} missing — the plan must impute or drop it.` })
  }

  if (f.dtypes) {
    const mixed = Object.entries(f.dtypes).filter(([, t]) => t.mixed).map(([n]) => n)
    if (mixed.length)
      facts.push({ tone: 'warn', text: `Mixed-type column${mixed.length > 1 ? 's' : ''}: ${mixed.join(', ')} — values disagree about what they are.` })
  }

  if (f.cardinality) {
    const ids = Object.entries(f.cardinality).filter(([, c]) => c.likely_id).map(([n]) => n)
    if (ids.length)
      facts.push({ tone: 'info', text: `Identifier-like: ${ids.join(', ')} — unique per row, useless as a feature.` })
    const consts = Object.entries(f.cardinality).filter(([, c]) => c.constant).map(([n]) => n)
    if (consts.length)
      facts.push({ tone: 'warn', text: `Constant column${consts.length > 1 ? 's' : ''}: ${consts.join(', ')} — carries no information.` })
  }

  if (f.correlation) {
    const ranked = Object.entries(f.correlation).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    const [name, r] = ranked[0] ?? []
    if (name !== undefined)
      facts.push({ tone: 'info', text: `“${name}” moves most with the target (r = ${r!.toFixed(2)}).` })
  }
  if (f.correlationNote) facts.push({ tone: 'info', text: `Correlation ${f.correlationNote}.` })

  if (f.drift) {
    if (f.drift.flagged.length)
      facts.push({ tone: 'warn', text: `Drift flagged in: ${f.drift.flagged.join(', ')} — validate with care.` })
    else facts.push({ tone: 'info', text: 'No train/test drift above threshold — the split looks honest.' })
    const unseen = Object.entries(f.drift.columns).filter(
      ([, c]) => c.type === 'categorical' && c.unseen_count > 0,
    )
    for (const [name, c] of unseen.slice(0, 2))
      if (c.type === 'categorical')
        facts.push({ tone: 'warn', text: `“${name}” has ${c.unseen_count} categor${c.unseen_count > 1 ? 'ies' : 'y'} in the compare file never seen in train.` })
  }

  if (f.rowsProfiled && f.columns)
    facts.push({ tone: 'info', text: `Profiled ${f.rowsProfiled.toLocaleString()} rows × ${f.columns.length} columns of ${f.path}.` })

  const MAX = 8
  const warns = facts.filter((x) => x.tone === 'warn')
  const infos = facts.filter((x) => x.tone === 'info')
  const kept = [...warns, ...infos].slice(0, MAX)
  if (facts.length > MAX) kept.push({ tone: 'info', text: `+${facts.length - MAX} more findings in the payloads below.` })
  return kept
}
