// The instruments: small chart components rendered straight from findings.
// Chart-mark colors are the dark-surface palette validated with the dataviz
// six-checks script (lightness band, chroma, CVD & normal-vision separation,
// contrast on #0f1620): blue #4a8ecb, amber #c8811f, crimson #cf4670.
// Bars stay thin with rounded data-ends; values ride the tips in text tokens;
// every mark has a hover tooltip; flags are icon+label, never color alone.

import type { ReactNode } from 'react'
import type { BalanceEntry, Drift, MissEntry, NumericEntry } from './findings'

const fmtNum = (x: number): string => {
  if (Number.isInteger(x)) return x.toLocaleString()
  if (Math.abs(x) >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 1 })
  return x.toFixed(2)
}

function BarRow({
  label,
  frac,
  display,
  tip,
  flag,
  tick,
}: {
  label: string
  frac: number
  display: string
  tip: string
  flag?: ReactNode
  tick?: number // hairline position as a 0..1 fraction of the track
}) {
  return (
    <div className="bar-row">
      <span className="bar-label" title={label}>{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${Math.min(Math.max(frac * 100, 0.8), 100)}%` }} />
        {tick !== undefined && <div className="bar-tick" style={{ left: `${tick * 100}%` }} />}
        <div className="tip">{tip}</div>
      </div>
      <span className="bar-value">{flag}{display}</span>
    </div>
  )
}

function Truncated({ hidden }: { hidden: number }) {
  if (hidden <= 0) return null
  return <span className="chart-caption mono-dim">+{hidden} more in the payload</span>
}

const MAX_ROWS = 12

/* -- target balance: one measure, one hue, class name left, share at tip -- */

export function TargetBalanceChart({ balance }: { balance: Record<string, BalanceEntry> }) {
  const entries = Object.entries(balance)
  const max = Math.max(...entries.map(([, b]) => b.ratio), 0.0001)
  return (
    <div>
      {entries.slice(0, MAX_ROWS).map(([name, b]) => (
        <BarRow
          key={name}
          label={name}
          frac={b.ratio / max}
          display={`${(b.ratio * 100).toFixed(1)}%`}
          tip={`${name}: ${b.count.toLocaleString()} rows (${(b.ratio * 100).toFixed(2)}%)`}
        />
      ))}
      <Truncated hidden={entries.length - MAX_ROWS} />
    </div>
  )
}

/* -- missingness: share of missing values per column ----------------------- */

export function MissingnessChart({ miss }: { miss: Record<string, MissEntry> }) {
  const entries = Object.entries(miss)
    .filter(([, m]) => m.missing > 0)
    .sort((a, b) => b[1].pct - a[1].pct)
  if (entries.length === 0)
    return <p className="chart-note">no missing values in the profiled sample ✓</p>
  const max = Math.max(...entries.map(([, m]) => m.pct), 0.01)
  return (
    <div>
      {entries.slice(0, MAX_ROWS).map(([name, m]) => (
        <BarRow
          key={name}
          label={name}
          frac={m.pct / max}
          display={`${(m.pct * 100).toFixed(1)}%`}
          tip={`${name}: ${m.missing.toLocaleString()} missing (${(m.pct * 100).toFixed(2)}%)`}
        />
      ))}
      <Truncated hidden={entries.length - MAX_ROWS} />
    </div>
  )
}

/* -- correlation: polarity → diverging bars around a zero line ------------- */
/* Sign is double-encoded: direction from the zero line AND the warm/cool pole,
   so identity never rides on color alone. */

export function CorrelationChart({ corr }: { corr: Record<string, number> }) {
  const ranked = Object.entries(corr).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
  return (
    <div>
      {ranked.slice(0, MAX_ROWS).map(([name, r]) => (
        <div key={name} className="bar-row">
          <span className="bar-label" title={name}>{name}</span>
          <div className="div-track">
            <div className="div-zero" />
            <div
              className={`div-fill ${r < 0 ? 'div-neg' : 'div-pos'}`}
              style={{ width: `${Math.min(Math.abs(r) * 50, 50)}%` }}
            />
            <div className="tip">{`${name}: r = ${r.toFixed(4)} vs target`}</div>
          </div>
          <span className="bar-value">{r >= 0 ? '+' : '−'}{Math.abs(r).toFixed(2)}</span>
        </div>
      ))}
      <span className="chart-caption mono-dim">← negative · positive → · full bar = |r| 1.0</span>
      <Truncated hidden={ranked.length - MAX_ROWS} />
    </div>
  )
}

/* -- drift: SMD / TVD magnitude per column with the 0.1 threshold hairline -- */

export function DriftChart({ drift }: { drift: Drift }) {
  const THRESHOLD = 0.1
  const entries = Object.entries(drift.columns)
  const value = (c: Drift['columns'][string]) => (c.type === 'numeric' ? c.smd : c.tvd)
  const ranked = entries.sort((a, b) => value(b[1]) - value(a[1]))
  const max = Math.max(...ranked.map(([, c]) => value(c)), THRESHOLD * 2)
  return (
    <div>
      {ranked.slice(0, MAX_ROWS).map(([name, c]) => (
        <BarRow
          key={name}
          label={name}
          frac={value(c) / max}
          display={value(c).toFixed(3)}
          tip={
            c.type === 'numeric'
              ? `${name}: SMD ${c.smd.toFixed(4)} (mean ${fmtNum(c.mean)} → ${fmtNum(c.mean_compare)})`
              : `${name}: TVD ${c.tvd.toFixed(4)}${c.unseen_count ? `, ${c.unseen_count} unseen categories` : ''}`
          }
          flag={c.drifted && <span className="flagtag">▲ DRIFT</span>}
          tick={THRESHOLD / max}
        />
      ))}
      <span className="chart-caption mono-dim">hairline = flag threshold 0.1 (smd / tvd) · scale max {max.toFixed(2)}</span>
      {drift.only_in_this_file.length > 0 && (
        <p className="chart-note">only in train: {drift.only_in_this_file.join(', ')}</p>
      )}
      {drift.only_in_compare.length > 0 && (
        <p className="chart-note">only in compare: {drift.only_in_compare.join(', ')}</p>
      )}
      <Truncated hidden={ranked.length - MAX_ROWS} />
    </div>
  )
}

/* -- numeric summary: this one is a table, not a chart ---------------------- */

export function NumericTable({ nums }: { nums: Record<string, NumericEntry> }) {
  const entries = Object.entries(nums)
  return (
    <div className="table-scroll">
      <table className="stats">
        <thead>
          <tr>
            <th>column</th><th>min</th><th>median</th><th>max</th><th>mean</th><th>std</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([name, s]) => (
            <tr key={name}>
              <td>{name}</td>
              <td>{fmtNum(s.min)}</td>
              <td>{fmtNum(s.median)}</td>
              <td>{fmtNum(s.max)}</td>
              <td>{fmtNum(s.mean)}</td>
              <td>{fmtNum(s.std)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* -- the report / plan, rendered from markdown-lite ------------------------- */
/* Headings, **bold** and `code` — enough for what the agent writes, and every
   character stays visible if it writes something fancier. */

function inline(text: string): ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`')) return <code key={i}>{part.slice(1, -1)}</code>
    return part
  })
}

export function ReportView({ text }: { text: string }) {
  const blocks = text.split(/\n\n+/)
  return (
    <div className="report">
      {blocks.map((block, i) => {
        if (block.startsWith('## ')) return <h3 key={i}>{inline(block.slice(3))}</h3>
        if (block.startsWith('# ')) return <h2 key={i}>{inline(block.slice(2))}</h2>
        return <p key={i}>{inline(block)}</p>
      })}
    </div>
  )
}
