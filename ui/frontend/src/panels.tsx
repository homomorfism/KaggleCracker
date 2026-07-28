// Dashboard panels -> rendered cards. Chart-shaped panels become ECharts
// options; table/stat/markdown render as plain DOM. Data shapes mirror
// tools/eda/spec.py — the backend has already validated them, so rendering
// trusts the shape and only guards against absent fields.

import type { EChartsCoreOption } from 'echarts/core'
import type { Panel } from './api'
import { Chart, TOKENS } from './echarts'
import { ReportView } from './charts'

type Bin = { x0: number; x1: number; count: number }
type Series = { name: string; values: number[] }

function fmtNum(v: number): string {
  if (Number.isInteger(v)) return String(v)
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2)
}

function histogramOption(data: Record<string, unknown>): EChartsCoreOption {
  const bins = (data.bins as Bin[]) ?? []
  return {
    grid: { left: 44, right: 12, top: 12, bottom: 28 },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: bins.map((b) => `${fmtNum(b.x0)}–${fmtNum(b.x1)}`),
      axisLabel: { interval: Math.max(0, Math.floor(bins.length / 8) - 1) },
    },
    yAxis: { type: 'value' },
    series: [
      {
        type: 'bar',
        data: bins.map((b) => b.count),
        barCategoryGap: '8%',
        itemStyle: { color: TOKENS.blue },
      },
    ],
  }
}

function barOption(data: Record<string, unknown>): EChartsCoreOption {
  const categories = (data.categories as string[]) ?? []
  const series = (data.series as Series[]) ?? []
  const horizontal = categories.length > 8
  const catAxis = { type: 'category' as const, data: categories }
  const valAxis = { type: 'value' as const }
  return {
    grid: { left: horizontal ? 110 : 44, right: 12, top: 12, bottom: 28 },
    tooltip: { trigger: 'axis' },
    xAxis: horizontal ? valAxis : catAxis,
    yAxis: horizontal ? { ...catAxis, inverse: true } : valAxis,
    series: series.map((s, i) => ({
      type: 'bar',
      name: s.name,
      data: s.values,
      itemStyle: { color: [TOKENS.blue, TOKENS.amber, TOKENS.crimson][i % 3] },
    })),
  }
}

function scatterOption(data: Record<string, unknown>): EChartsCoreOption {
  return {
    grid: { left: 48, right: 16, top: 12, bottom: 34 },
    tooltip: {
      formatter: (p: { value: unknown[] }) =>
        `${data.x_label ?? 'x'}: ${p.value[0]}<br/>${data.y_label ?? 'y'}: ${p.value[1]}`,
    },
    xAxis: { type: 'value', name: String(data.x_label ?? ''), nameGap: 22 },
    yAxis: { type: 'value', name: String(data.y_label ?? '') },
    series: [
      {
        type: 'scatter',
        data: (data.points as number[][]) ?? [],
        symbolSize: 5,
        itemStyle: { color: TOKENS.blue, opacity: 0.65 },
      },
    ],
  }
}

function heatmapOption(data: Record<string, unknown>): EChartsCoreOption {
  const xs = (data.x_labels as string[]) ?? []
  const ys = (data.y_labels as string[]) ?? []
  const values = (data.values as (number | null)[][]) ?? []
  const cells: [number, number, number][] = []
  let lo = Infinity
  let hi = -Infinity
  values.forEach((row, yi) =>
    row.forEach((v, xi) => {
      if (v === null) return
      cells.push([xi, yi, v])
      lo = Math.min(lo, v)
      hi = Math.max(hi, v)
    }),
  )
  const diverging = lo < 0 && hi > 0
  return {
    grid: { left: 110, right: 12, top: 12, bottom: 64 },
    tooltip: {
      formatter: (p: { value: [number, number, number] }) =>
        `${ys[p.value[1]]} × ${xs[p.value[0]]}: ${p.value[2].toFixed(3)}`,
    },
    xAxis: { type: 'category', data: xs, axisLabel: { rotate: 45 } },
    yAxis: { type: 'category', data: ys },
    visualMap: {
      min: diverging ? -Math.max(-lo, hi) : lo,
      max: diverging ? Math.max(-lo, hi) : hi,
      calculable: false,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      itemHeight: 90,
      textStyle: { color: TOKENS.dim, fontSize: 9 },
      inRange: {
        color: diverging
          ? [TOKENS.crimson, '#141d2a', TOKENS.blue]
          : ['#141d2a', TOKENS.blue],
      },
    },
    series: [
      {
        type: 'heatmap',
        data: cells,
        itemStyle: { borderColor: '#0a0f15', borderWidth: 1 },
      },
    ],
  }
}

function lineOption(data: Record<string, unknown>): EChartsCoreOption {
  const xs = (data.x as (string | number)[]) ?? []
  const series = (data.series as Series[]) ?? []
  return {
    grid: { left: 44, right: 16, top: 12, bottom: 28 },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: xs.map(String) },
    yAxis: { type: 'value' },
    series: series.map((s, i) => ({
      type: 'line',
      name: s.name,
      data: s.values,
      showSymbol: false,
      lineStyle: { color: [TOKENS.blue, TOKENS.amber, TOKENS.crimson][i % 3] },
    })),
  }
}

const CHART_BUILDERS: Record<string, (d: Record<string, unknown>) => EChartsCoreOption> = {
  histogram: histogramOption,
  bar: barOption,
  scatter: scatterOption,
  heatmap: heatmapOption,
  line: lineOption,
}

function PanelBody({ panel }: { panel: Panel }) {
  const build = CHART_BUILDERS[panel.type]
  if (build)
    return <Chart option={build(panel.data)} height={panel.type === 'heatmap' ? 320 : 220} />

  if (panel.type === 'table') {
    const columns = (panel.data.columns as string[]) ?? []
    const rows = (panel.data.rows as unknown[][]) ?? []
    return (
      <div className="table-scroll">
        <table className="stats">
          <thead>
            <tr>
              {columns.map((c) => (
                <th key={c}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {row.map((cell, j) => (
                  <td key={j}>{String(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }
  if (panel.type === 'stat')
    return (
      <p className="stat-value">
        {String(panel.data.value)}
        {panel.data.unit ? <em> {String(panel.data.unit)}</em> : null}
      </p>
    )
  if (panel.type === 'markdown') return <ReportView text={String(panel.data.text ?? '')} />
  return <pre>{JSON.stringify(panel.data, null, 2)}</pre>
}

export function PanelCard({ panel }: { panel: Panel }) {
  return (
    <section className={`panel panel-card panel-${panel.type}`}>
      <h3 className="panel-title">{panel.title}</h3>
      <PanelBody panel={panel} />
      {panel.commentary && <p className="panel-commentary">{panel.commentary}</p>}
    </section>
  )
}
