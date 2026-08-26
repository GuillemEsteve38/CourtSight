import { useState } from 'react'
import { videoUrl } from '../api'
import MiniCourt from './MiniCourt'
import ScorePanel from './ScorePanel'

// Speeds carry an uncertainty band and are labelled as estimates, matching what
// src/analysis/speed.py actually supports. Showing a bare "142 km/h" would be false
// precision -- at 30fps the band is genuinely wide.

function SpeedRow({ shot }) {
  return (
    <li className="speed-row">
      <span className="speed-frames">
        {shot.start_frame}–{shot.end_frame}
      </span>
      <span className="speed-value">{shot.kmh.toFixed(0)}</span>
      <span className="speed-band">
        {shot.kmh_low.toFixed(0)}–{shot.kmh_high.toFixed(0)} km/h
      </span>
    </li>
  )
}

export default function Results({ jobId, result }) {
  const [selectedFrame, setSelectedFrame] = useState(null)

  const inCalls = result.bounces.filter((b) => b.is_in).length
  const outCalls = result.bounces.length - inCalls
  const ballCoverage = result.n_frames
    ? Math.round((result.ball_detected_frames / result.n_frames) * 100)
    : 0
  const courtCoverage = result.n_frames
    ? Math.round((result.court_detected_frames / result.n_frames) * 100)
    : 0

  return (
    <div className="results">
      <section className="card stats-card">
        <h2>{result.video}</h2>
        <div className="stats">
          <Stat label="Frames" value={result.n_frames} />
          <Stat label="FPS" value={result.fps.toFixed(1)} />
          <Stat label="Bounces" value={result.bounces.length} />
          <Stat label="In / Out" value={`${inCalls} / ${outCalls}`} />
          <Stat label="Ball tracked" value={`${ballCoverage}%`} />
          <Stat label="Court detected" value={`${courtCoverage}%`} />
        </div>
        {ballCoverage < 70 && (
          <p className="warning">
            Ball tracked on only {ballCoverage}% of frames. Speeds and rally boundaries
            are unreliable below ~70% — higher frame-rate footage from a fixed tripod
            will improve this.
          </p>
        )}
      </section>

      <div className="columns">
        <section className="card">
          <h2>Annotated video</h2>
          <video className="player" src={videoUrl(jobId)} controls preload="metadata" />
        </section>

        <section className="card">
          <h2>Court map</h2>
          <MiniCourt
            bounces={result.bounces}
            ballTrack={result.ball_track_m}
            selectedFrame={selectedFrame}
          />
          <p className="legend">
            <span className="dot is-in" /> in
            <span className="dot is-out" /> out
          </p>
        </section>
      </div>

      <div className="columns">
        <section className="card">
          <h2>Line calls</h2>
          {result.bounces.length === 0 ? (
            <p className="hint">No bounces detected.</p>
          ) : (
            <ul className="call-list">
              {result.bounces.map((bounce) => (
                <li
                  key={bounce.frame}
                  className={`call ${bounce.is_in ? 'is-in' : 'is-out'}`}
                  onMouseEnter={() => setSelectedFrame(bounce.frame)}
                  onMouseLeave={() => setSelectedFrame(null)}
                >
                  <span className="call-frame">frame {bounce.frame}</span>
                  <span className="call-verdict">{bounce.is_in ? 'IN' : 'OUT'}</span>
                  <span className="call-margin">
                    {Math.abs(bounce.distance_m).toFixed(2)} m from the line
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="card">
          <h2>Shot speeds</h2>
          <p className="hint">
            Estimates with an uncertainty band, not measurements — a monocular camera
            sees only ground-plane motion.
          </p>
          {result.shots.length === 0 ? (
            <p className="hint">No shot speeds could be measured.</p>
          ) : (
            <ul className="speed-list">
              {result.shots.map((shot) => (
                <SpeedRow key={`${shot.start_frame}-${shot.end_frame}`} shot={shot} />
              ))}
            </ul>
          )}
        </section>
      </div>

      <ScorePanel rallies={result.rallies} />
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
    </div>
  )
}
