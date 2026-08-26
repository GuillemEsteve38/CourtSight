// Thin wrapper over the backend API. Kept in one place so components never build URLs.

async function asJson(response) {
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}))
    throw new Error(detail.detail || `${response.status} ${response.statusText}`)
  }
  return response.json()
}

export const getHealth = () => fetch('/api/health').then(asJson)

export function uploadVideo(file) {
  const body = new FormData()
  body.append('file', file)
  return fetch('/api/jobs', { method: 'POST', body }).then(asJson)
}

export const getJob = (id) => fetch(`/api/jobs/${id}`).then(asJson)

export const getResult = (id) => fetch(`/api/jobs/${id}/result`).then(asJson)

export const videoUrl = (id) => `/api/jobs/${id}/video`

export function postScore(winners, { bestOf = 3, firstServer = 0 } = {}) {
  return fetch('/api/score', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ winners, best_of: bestOf, first_server: firstServer }),
  }).then(asJson)
}
