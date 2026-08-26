import { useEffect, useRef, useState } from 'react'
import { getHealth, getJob, getResult, uploadVideo } from './api'
import Results from './components/Results'

const POLL_INTERVAL_MS = 1000
const TERMINAL = ['done', 'error']

const STAGE_LABELS = {
  read: 'Decoding video',
  court: 'Detecting court',
  ball: 'Tracking ball',
  bounce: 'Detecting bounces',
  analyze: 'Computing calls and speeds',
  render: 'Rendering annotated video',
}

export default function App() {
  const [health, setHealth] = useState(null)
  const [job, setJob] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const pollRef = useRef(null)

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'unreachable' }))
  }, [])

  // Poll while a job is in flight; stop as soon as it reaches a terminal state.
  useEffect(() => {
    if (!job || TERMINAL.includes(job.status)) return undefined

    pollRef.current = setInterval(async () => {
      try {
        const next = await getJob(job.id)
        setJob(next)
        if (next.status === 'done') {
          setResult(await getResult(next.id))
        }
      } catch (e) {
        setError(e.message)
      }
    }, POLL_INTERVAL_MS)

    return () => clearInterval(pollRef.current)
  }, [job?.id, job?.status])

  async function handleFile(file) {
    if (!file) return
    setError(null)
    setResult(null)
    setUploading(true)
    try {
      setJob(await uploadVideo(file))
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  const busy = job && !TERMINAL.includes(job.status)

  return (
    <div className="app">
      <header className="header">
        <h1>CourtSight</h1>
        <p className="tagline">Ball tracking, line calls, speeds and scoring from a fixed camera.</p>
        {health && (
          <p className={`health ${health.cuda ? 'ok' : 'warn'}`}>
            {health.status === 'unreachable'
              ? 'Backend unreachable'
              : health.cuda
                ? `GPU: ${health.device}`
                : 'Running on CPU — analysis will be slow'}
          </p>
        )}
      </header>

      <section className="card upload-card">
        <h2>Upload a clip</h2>
        <p className="hint">
          Fixed camera behind the baseline. Any 16:9 resolution works — it is resized to
          1280×720, which is what the models expect.
        </p>
        <label className={`dropzone ${busy ? 'disabled' : ''}`}>
          <input
            type="file"
            accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
            disabled={busy || uploading}
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
          <span>{uploading ? 'Uploading…' : 'Choose a video file'}</span>
        </label>
        {error && <p className="error">{error}</p>}
      </section>

      {job && (
        <section className="card progress-card">
          <div className="progress-head">
            <span className="filename">{job.filename}</span>
            <span className={`status ${job.status}`}>{job.status}</span>
          </div>
          <div className="progressbar">
            <div className="progressbar-fill" style={{ width: `${job.progress * 100}%` }} />
          </div>
          <p className="progress-message">
            {job.status === 'error'
              ? job.error
              : `${STAGE_LABELS[job.stage] || job.message} · ${Math.round(job.progress * 100)}%`}
          </p>
          {busy && (
            <p className="hint">
              This takes a while — court detection is the slow part. You can leave this tab open.
            </p>
          )}
        </section>
      )}

      {result && <Results jobId={job.id} result={result} />}
    </div>
  )
}
