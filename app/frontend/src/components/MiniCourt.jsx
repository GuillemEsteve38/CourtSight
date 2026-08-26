// Top-down court drawn in SVG from court-meters coordinates.
//
// Deliberately vector rather than a server-rendered image: bounces stay hoverable and
// the panel scales cleanly. Court dimensions mirror src/scoring/rules.py -- if these
// ever disagree, rules.py is the source of truth.

const LENGTH = 23.77
const WIDTH = 10.97
const SINGLES_INSET = 1.37
const SERVICE_FROM_NET = 6.4
const NET_Y = LENGTH / 2
const PAD = 1.2

const LEFT_SINGLES = SINGLES_INSET
const RIGHT_SINGLES = WIDTH - SINGLES_INSET
const TOP_SERVICE = NET_Y - SERVICE_FROM_NET
const BOTTOM_SERVICE = NET_Y + SERVICE_FROM_NET

export default function MiniCourt({ bounces = [], ballTrack = [], selectedFrame = null }) {
  const trail = ballTrack.filter(Boolean)

  return (
    <svg
      className="minicourt"
      viewBox={`${-PAD} ${-PAD} ${WIDTH + PAD * 2} ${LENGTH + PAD * 2}`}
      role="img"
      aria-label="Court map of ball bounces"
    >
      <rect x={-PAD} y={-PAD} width={WIDTH + PAD * 2} height={LENGTH + PAD * 2} className="mc-surround" />
      <rect x={0} y={0} width={WIDTH} height={LENGTH} className="mc-court" />

      <g className="mc-lines">
        <rect x={0} y={0} width={WIDTH} height={LENGTH} />
        <line x1={LEFT_SINGLES} y1={0} x2={LEFT_SINGLES} y2={LENGTH} />
        <line x1={RIGHT_SINGLES} y1={0} x2={RIGHT_SINGLES} y2={LENGTH} />
        <line x1={LEFT_SINGLES} y1={TOP_SERVICE} x2={RIGHT_SINGLES} y2={TOP_SERVICE} />
        <line x1={LEFT_SINGLES} y1={BOTTOM_SERVICE} x2={RIGHT_SINGLES} y2={BOTTOM_SERVICE} />
        <line x1={WIDTH / 2} y1={TOP_SERVICE} x2={WIDTH / 2} y2={BOTTOM_SERVICE} />
      </g>
      <line x1={-PAD / 2} y1={NET_Y} x2={WIDTH + PAD / 2} y2={NET_Y} className="mc-net" />

      <g className="mc-trail">
        {trail.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={0.09} />
        ))}
      </g>

      {bounces.map((bounce) => (
        <circle
          key={bounce.frame}
          cx={bounce.point_m[0]}
          cy={bounce.point_m[1]}
          r={selectedFrame === bounce.frame ? 0.55 : 0.36}
          className={`mc-bounce ${bounce.is_in ? 'is-in' : 'is-out'} ${
            selectedFrame === bounce.frame ? 'is-selected' : ''
          }`}
        >
          <title>
            {`frame ${bounce.frame} — ${bounce.is_in ? 'IN' : 'OUT'} (${bounce.distance_m.toFixed(2)} m)`}
          </title>
        </circle>
      ))}
    </svg>
  )
}
