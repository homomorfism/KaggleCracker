// Thin ECharts wrapper: tree-shaken core, one theme built from the CSS
// tokens, one <Chart option/> component. No echarts-for-react — a resize
// observer and a dispose call do not justify a dependency.

import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsCoreOption } from 'echarts/core'

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  HeatmapChart,
  GridComponent,
  TooltipComponent,
  VisualMapComponent,
  CanvasRenderer,
])

// Mirrors styles.css :root — the charts must read as the same instrument
// panel, not an embedded foreign widget.
export const TOKENS = {
  bg: 'transparent',
  panel: '#0f1620',
  edge: '#1e2a3a',
  text: '#e9e2d0',
  dim: '#7d8ba0',
  faint: '#4b586c',
  amber: '#f2a33c',
  blue: '#4a8ecb',
  crimson: '#cf4670',
  ok: '#62d394',
  mono: "'Martian Mono', ui-monospace, monospace",
}

const kcDarkTheme = {
  color: [TOKENS.blue, TOKENS.amber, TOKENS.crimson, TOKENS.ok, TOKENS.dim],
  backgroundColor: TOKENS.bg,
  textStyle: { color: TOKENS.dim, fontFamily: TOKENS.mono, fontSize: 10 },
  axisPointer: { lineStyle: { color: TOKENS.faint } },
  categoryAxis: {
    axisLine: { lineStyle: { color: TOKENS.edge } },
    axisTick: { show: false },
    axisLabel: { color: TOKENS.dim, fontFamily: TOKENS.mono, fontSize: 9 },
    splitLine: { show: false },
  },
  valueAxis: {
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: TOKENS.dim, fontFamily: TOKENS.mono, fontSize: 9 },
    splitLine: { lineStyle: { color: TOKENS.edge, type: 'dashed' as const } },
  },
  tooltip: {
    backgroundColor: TOKENS.panel,
    borderColor: TOKENS.edge,
    textStyle: { color: TOKENS.text, fontFamily: TOKENS.mono, fontSize: 10 },
  },
}

echarts.registerTheme('kc-dark', kcDarkTheme)

export function Chart({ option, height = 240 }: { option: EChartsCoreOption; height?: number }) {
  const ref = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const chart = echarts.init(ref.current, 'kc-dark')
    chartRef.current = chart
    const observer = new ResizeObserver(() => chart.resize())
    observer.observe(ref.current)
    return () => {
      observer.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [])

  useEffect(() => {
    // notMerge: the option is the whole truth each render — panels are
    // replaced wholesale by the agent, never patched.
    chartRef.current?.setOption(option, { notMerge: true })
  }, [option])

  return <div ref={ref} style={{ width: '100%', height }} />
}
