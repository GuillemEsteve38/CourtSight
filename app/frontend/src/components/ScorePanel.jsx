import { useEffect, useState } from 'react'
import { postScore } from '../api'

// Human-in-the-loop scoring. Each rally shows what the pipeline proposed and how
// confident it is; the human confirms or overrides, and only confirmed points reach
// the score. Confidence is shown as a bar rather than a bare number so a weak
// proposal reads as weak at a glance -- these top out around 0.6 by design
// (see src/analysis/rally_segmentation.py).

export default function ScorePanel({ rallies }) {
  const [decisions, setDecisions] = useState(() => rallies.map(() => null))
  const [score, setScore] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    setDecisions(rallies.map(() => null))
  }, [rallies])

  const confirmed = decisions.filter((d) => d !== null)

  useEffect(() => {
    let cancelled = false
    postScore(confirmed)
      .then((s) => !cancelled && setScore(s))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [confirmed.join(',')])

  function decide(index, winner) {
    setDecisions((prev) => {
      const next = [...prev]
      next[index] = next[index] === winner ? null : winner
      return next
    })
  }

  return (
    <section className="card">
      <h2>Score</h2>
      <p className="hint">
        Rally outcomes are <strong>proposals</strong>. Confirm or override each one — only
        confirmed points are scored.
      </p>

      {error && <p className="error">Scoring failed: {error}</p>}

      {score && (
        <div className="scoreboard">
          <div className="scoreline">
            <span className="label">Score</span>
            <span className="value">{score.summary}</span>
          </div>
          <div className="scoreline">
            <span className="label">Confirmed</span>
            <span className="value">
              {confirmed.length} of {rallies.length} rallies
            </span>
          </div>
          {score.winner !== null && (
            <p className="winner">Player {score.winner} wins the match</p>
          )}
        </div>
      )}

      <ol className="rally-list">
        {rallies.map((rally, index) => (
          <li key={`${rally.start_frame}-${index}`} className="rally">
            <div className="rally-head">
              <span className="rally-frames">
                frames {rally.start_frame}–{rally.end_frame}
              </span>
              <span className="rally-bounces">{rally.bounce_frames.length} bounces</span>
            </div>

            <p className="rally-reason">
              {rally.proposed_winner === null
                ? 'No proposal — needs a human call.'
                : `Proposes player ${rally.proposed_winner}: ${rally.proposal_reason}`}
            </p>

            {rally.proposed_winner !== null && (
              <div className="confidence" title={`confidence ${rally.proposal_confidence}`}>
                <div
                  className="confidence-fill"
                  style={{ width: `${Math.round(rally.proposal_confidence * 100)}%` }}
                />
                <span className="confidence-text">
                  confidence {(rally.proposal_confidence * 100).toFixed(0)}%
                </span>
              </div>
            )}

            <div className="rally-actions">
              {[0, 1].map((player) => (
                <button
                  key={player}
                  type="button"
                  className={`chip ${decisions[index] === player ? 'chosen' : ''} ${
                    rally.proposed_winner === player ? 'proposed' : ''
                  }`}
                  onClick={() => decide(index, player)}
                >
                  Player {player}
                  {rally.proposed_winner === player && <span className="tick"> ·proposed</span>}
                </button>
              ))}
              {decisions[index] !== null && (
                <span className="decided">
                  {decisions[index] === rally.proposed_winner ? 'confirmed' : 'overridden'}
                </span>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  )
}
