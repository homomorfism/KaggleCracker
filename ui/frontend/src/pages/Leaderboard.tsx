import { useEffect, useState } from 'react'
import type { LeaderboardEntry } from '../api'
import { api } from '../api'

export default function LeaderboardPage({ slug }: { slug: string }) {
  const [entries, setEntries] = useState<LeaderboardEntry[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.leaderboard(slug).then(setEntries, (e) => setError(String(e.message ?? e)))
  }, [slug])

  if (error) return <div className="notice notice-err">{error}</div>
  if (!entries) return <div className="notice mono-dim">LOADING…</div>

  return (
    <section>
      <div className="page-head">
        <h1>Pipeline leaderboard</h1>
        <span className="mono-dim">{entries.length} recorded experiments</span>
      </div>

      {entries.length === 0 && (
        <div className="notice">
          <p>No pipelines scored yet — results appear here once model training runs
          experiments and records their CV scores.</p>
        </div>
      )}

      {entries.length > 0 && (
        <div className="panel">
          <div className="table-scroll">
            <table className="stats">
              <thead>
                <tr>
                  <th>#</th>
                  <th>EXPERIMENT</th>
                  <th>CV SCORE</th>
                  <th>FOLDS</th>
                  <th>NOTES</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e, i) => (
                  <tr key={`${e.experiment_id}-${i}`} className={i === 0 ? 'lb-best' : ''}>
                    <td>{i + 1}</td>
                    <td>{e.experiment_id}</td>
                    <td>{typeof e.cv_score === 'number' ? e.cv_score.toFixed(5) : '—'}</td>
                    <td>{e.fold_scores?.length ?? '—'}</td>
                    <td>{e.notes ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  )
}
