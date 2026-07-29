// The recon instruments, rendered on the EDA dashboard from the newest
// finished recon run's journal — same findings pipeline the run page uses,
// so the dashboard and the run can never disagree.

import { useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import type { JournalEvent } from './api'
import { api } from './api'
import { collectFindings, deriveFacts } from './findings'
import {
  CorrelationChart,
  DriftChart,
  MissingnessChart,
  NumericTable,
  TargetBalanceChart,
} from './charts'

function Card({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="panel inst">
      <h3 className="panel-title">{title}</h3>
      {children}
    </div>
  )
}

export default function ReconSummary({ slug }: { slug: string }) {
  const [events, setEvents] = useState<JournalEvent[] | null>(null)
  const [runId, setRunId] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    setEvents(null)
    api.getProject(slug).then(
      (p) => {
        const latest = [...p.runs].reverse().find((r) => r.status === 'finished')
        if (!latest) {
          if (alive) setEvents([])
          return
        }
        if (alive) setRunId(latest.run_id)
        api.events(slug, latest.run_id, 0).then(
          (page) => alive && setEvents(page.events),
          () => alive && setEvents([]),
        )
      },
      () => alive && setEvents([]),
    )
    return () => {
      alive = false
    }
  }, [slug])

  const findings = useMemo(() => collectFindings(events ?? []), [events])
  const facts = useMemo(() => deriveFacts(findings), [findings])

  // No finished recon run yet: say nothing — the EDA agent's board stands alone.
  if (!events || events.length === 0) return null
  const hasCorr = !!findings.correlation && Object.keys(findings.correlation).length > 0

  return (
    <div style={{ marginTop: 26 }}>
      <div className="report-head" style={{ marginBottom: 12 }}>
        <h3 className="panel-title">
          FROM RECON RUNS{findings.path ? ` — ${findings.path}` : ''}
        </h3>
        {runId && (
          <a className="mono-dim" href={`#/p/${slug}/run/${runId}`}>VIEW RUN ▸</a>
        )}
      </div>
      <div className="panel-grid">
        {facts.length > 0 && (
          <Card title="SIGNALS">
            <div className="facts">
              {facts.map((f, i) => (
                <div key={i} className={`fact fact-${f.tone}`}>
                  <span className="fact-glyph">{f.tone === 'warn' ? '▲' : '◆'}</span>
                  <p>{f.text}</p>
                </div>
              ))}
            </div>
          </Card>
        )}
        {findings.targetBalance && (
          <Card title="TARGET BALANCE">
            <TargetBalanceChart balance={findings.targetBalance} />
          </Card>
        )}
        {findings.missingness && (
          <Card title="MISSINGNESS">
            <MissingnessChart miss={findings.missingness} />
          </Card>
        )}
        {hasCorr && (
          <Card title="CORRELATION WITH TARGET">
            <CorrelationChart corr={findings.correlation!} />
          </Card>
        )}
        {findings.drift && (
          <Card title="TRAIN / TEST DRIFT">
            <DriftChart drift={findings.drift} />
          </Card>
        )}
        {findings.numericSummary && (
          <Card title="NUMERIC SUMMARY">
            <NumericTable nums={findings.numericSummary} />
          </Card>
        )}
      </div>
    </div>
  )
}
