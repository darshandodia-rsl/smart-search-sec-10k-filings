import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

const PANE = 'plan-progress'
// Bumped after each tool call so the pane re-reads PLAN.md, git and budgets.
const tick = atom({ plugin: 'plan-progress', key: 'tick' } as const, 0, {
  shape: 'tick-1',
})


// Colours used by the panel. Fixed hex values so the look is the same in any
// terminal theme; muted/track are mid-greys that read on light and dark.
const C = {
  blue: '#79c0ff',
  violet: '#d2a8ff',
  done: '#7ee787',
  more: '#56d4dd',
  current: '#f2cc60',
  warn: '#ff7b72',
  text: '#c9d1d9',
  muted: '#8b949e',
  pending: '#6e7681',
  track: '#484f58',
  border: '#586069',
}

type Phase = { n: number; title: string; done: boolean; goal: string; bullets: string[] }
type Row = { date: string; step: string; what: string; open: string }

const clip = (s: string, w: number) =>
  s.length > w ? s.slice(0, Math.max(1, w - 1)) + '…' : s

const clean = (s: string) =>
  s.replace(/\*\*|`/g, '').replace(/\\\|/g, '|').trim()

function parsePlan(text: string) {
  const phases: Phase[] = []
  for (const part of text.split(/\n(?=## )/)) {
    const m = part.match(/^## Phase (\d+): (.*)/)
    if (!m || !m[1] || !m[2]) continue
    const goal = part.match(/\*\*Goal:\*\*\s*(.*)/)?.[1] ?? ''
    phases.push({
      n: Number(m[1]),
      title: m[2].replace(/✅.*$/, '').trim(),
      done: m[2].includes('✅'),
      goal: clean(goal),
      bullets: part
        .split('\n')
        .filter(l => /^- /.test(l))
        .map(l => clean(l.slice(2))),
    })
  }
  const rows: Row[] = []
  const seen = new Set<string>()
  for (const line of text.split('\n')) {
    if (!/^\| 20\d\d-/.test(line)) continue
    const c = line.split(/(?<!\\)\|/).map(x => x.trim())
    if (seen.has(line)) continue // the log has a duplicated row
    seen.add(line)
    rows.push({ date: c[1] ?? '', step: clean(c[2] ?? ''), what: clean(c[3] ?? ''), open: clean(c[4] ?? '') })
  }
  return { phases, rows }
}

async function budget($: any, file: string, today: string) {
  try {
    const j = JSON.parse(await $.fs.read(`data/state/${file}`))
    return j.date === today ? Number(j.tokens_used) : 0
  } catch {
    return null
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'progress',
      description: 'Open the PLAN.md progress panel',
    })
    void $.ui.open({ id: PANE, title: 'Project progress' })
    return next(e)
  })

  on('command.run', { command: 'progress' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Project progress' })
    return { text: 'Progress panel opened.' }
  })

  on('tool.call', async ($, e, next) => {
    const ran = await next(e)
    await update($, tick, n => n + 1)
    return ran
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    await read($, tick)
    const cols = (e.viewport?.columns ?? 44) - 6
    const rowsAvail = e.viewport?.rows ?? 40

    let plan
    try {
      plan = parsePlan(await $.fs.read('PLAN.md'))
    } catch {
      return <Text dimColor>PLAN.md not found in the working directory.</Text>
    }
    const { phases, rows } = plan
    const doneCount = phases.filter(p => p.done).length
    const cur = phases.find(p => !p.done)
    const pct = Math.round((doneCount / Math.max(1, phases.length)) * 100)
    const barW = Math.max(8, Math.min(30, cols - 8))
    const filled = Math.round((pct / 100) * barW)

    const curRows = cur ? rows.filter(r => r.step.startsWith(`${cur.n}.`)) : []
    const steps: string[] = []
    for (const r of curRows) if (!steps.includes(r.step)) steps.push(r.step)
    // Upcoming steps: ids like "(2.4)" or "→ 2.4" named in the open items of
    // this phase's log rows that no logged step has used yet.
    const upcoming: string[] = []
    if (cur) {
      const loggedIds = steps.map(x => x.match(/^(\d+\.\d+\w*)/)?.[1] ?? '')
      const seenIds = new Set<string>()
      for (const r of [...curRows].reverse()) {
        for (const seg of r.open.split(/(?<=[.;])\s+/)) {
          for (const m of seg.matchAll(/(\d+)\.(\d+\w*)/g)) {
            const id = `${m[1]}.${m[2]}`
            if (m[1] !== String(cur.n) || seenIds.has(id)) continue
            if (loggedIds.some(l => l.startsWith(id) || id.startsWith(l))) continue
            seenIds.add(id)
            const label = seg.replace(/\(?\d+\.\d+\w*\)?/g, '').replace(/[→.;]+/g, ' ').replace(/\s+/g, ' ').trim()
            upcoming.push(`${id} ${label}`)
          }
        }
      }
      upcoming.sort()
    }
    const last = rows.at(-1)

    // Keep the timeline short: only the newest KEEP completed phases, and only
    // the newest KEEP completed steps of the current phase, are listed. The
    // rest collapse into a one-line count.
    const KEEP = 3
    const hiddenPhases = Math.max(0, doneCount - KEEP)
    const shownPhases = phases.filter(p => !(p.done && p.n < (phases.filter(q => q.done)[hiddenPhases]?.n ?? 0)))
    const hiddenSteps = Math.max(0, steps.length - KEEP)
    const shownSteps = steps.slice(hiddenSteps)

    const today = new Date(await $.clock.now()).toISOString().slice(0, 10)
    const groq = await budget($, 'budget_groq.json', today)
    const limits: { kind: string; percentUsed: number; resetsAt?: string }[] = await $.session
      .usage()
      .then((u: any) => u.rateLimits ?? [])
      .catch(() => [])
    const now = await $.clock.now()
    const compact = rowsAvail < 30
    // Inner text width of a card: the pane width minus border (2) and padding (2).
    const w = cols - 4

    const bar = (ratio: number, width: number, color: string) => {
      const f = Math.round(Math.min(1, Math.max(0, ratio)) * width)
      return (
        <Text color={color}>
          {'█'.repeat(f)}
          <Text color={C.track}>{'░'.repeat(width - f)}</Text>
        </Text>
      )
    }

    // One rounded card: a coloured heading, then the content, with padding.
    const card = (title: string, accent: string, body: any) => (
      <Box
        flexDirection="column"
        borderStyle="round"
        borderColor={C.border}
        paddingX={1}
        paddingY={0}
      >
        <Text bold color={accent}>
          {title}
        </Text>
        <Box flexDirection="column">
          {body}
        </Box>
      </Box>
    )

    // "resets in 2h 14m" from an ISO timestamp.
    const resetsIn = (iso?: string) => {
      if (!iso) return ''
      const mins = Math.max(0, Math.round((Date.parse(iso) - now) / 60000))
      const d = Math.floor(mins / 1440)
      const h = Math.floor((mins % 1440) / 60)
      return ' · resets in ' + (d ? `${d}d ${h}h` : h ? `${h}h ${mins % 60}m` : `${mins}m`)
    }
    const windowRow = (label: string, kind: string) => {
      const l = limits.find(x => x.kind === kind)
      return (
        <Box flexDirection="column">
          <Text color={C.text}>
            {label}{' '}
            <Text color={C.muted}>
              {l ? `${l.percentUsed}%${resetsIn(l.resetsAt)}` : '—'}
            </Text>
          </Text>
          {bar(l ? l.percentUsed / 100 : 0, Math.min(24, w), l && l.percentUsed > 80 ? C.warn : C.blue)}
        </Box>
      )
    }

    const budgetRow = (label: string, used: number | null, cap: number, note = '') => (
      <Box flexDirection="column">
        <Text color={C.text}>
          {label}{' '}
          <Text color={C.muted}>
            {used === null ? '—' : `${used.toLocaleString()} / ${cap.toLocaleString()}${note}`}
          </Text>
        </Text>
        {bar(used === null ? 0 : used / cap, Math.min(24, w), used !== null && used / cap > 0.8 ? C.warn : C.blue)}
      </Box>
    )

    return (
      <Box flexDirection="column">
        {card(
          'Overall',
          C.blue,
          <Box flexDirection="column">
            <Text bold color={C.text}>
              {doneCount}/{phases.length} phases <Text color={C.muted}>· {pct}%</Text>
            </Text>
            {bar(pct / 100, Math.min(30, w), C.done)}
          </Box>,
        )}

        {card(
          'Phases',
          C.blue,
          <Box flexDirection="column">
            {hiddenPhases > 0 && (
              <Text color={C.more}>
                ✓ … {hiddenPhases} earlier phase{hiddenPhases > 1 ? 's' : ''} done
              </Text>
            )}
            {shownPhases.map((p, i) => {
              const isCur = cur?.n === p.n
              const glyph = p.done ? '●' : isCur ? '◐' : '○'
              const color = p.done ? C.done : isCur ? C.current : C.pending
              const rail = p.done ? C.done : C.track
              return (
                <Box flexDirection="column">
                  <Text color={color} bold={isCur}>
                    {glyph} {p.n} {clip(p.title, w - 4)}
                  </Text>
                  {isCur && !compact && (
                    <Box paddingLeft={2}>
                      <Text color={C.muted}>{clip(p.goal, w - 4)}</Text>
                    </Box>
                  )}
                  {isCur && hiddenSteps > 0 && (
                    <Box paddingLeft={2}>
                      <Text color={C.more}>
                        ✓ … {hiddenSteps} earlier step{hiddenSteps > 1 ? 's' : ''} done
                      </Text>
                    </Box>
                  )}
                  {isCur &&
                    shownSteps.map(st => (
                      <Box paddingLeft={2}>
                        <Text color={C.done}>✓ {clip(st, w - 6)}</Text>
                      </Box>
                    ))}
                  {isCur &&
                    upcoming.map(u => (
                      <Box paddingLeft={2}>
                        <Text color={C.pending}>○ {clip(u, w - 6)}</Text>
                      </Box>
                    ))}
                  {i < shownPhases.length - 1 && <Text color={rail}>│</Text>}
                </Box>
              )
            })}
          </Box>,
        )}

        {last &&
          card(
            'Last step',
            C.violet,
            <Box flexDirection="column" gap={1}>
              <Text color={C.text}>
                <Text color={C.muted}>{last.date}</Text> · {clip(last.step, w - 14)}
              </Text>
              {last.open && (
                <Box flexDirection="column">
                  <Text bold color={C.current}>
                    Open items
                  </Text>
                  <Text color={C.text}>{clip(last.open, (w - 2) * (compact ? 2 : 4))}</Text>
                </Box>
              )}
            </Box>,
          )}

        {card(
          'LLM usage',
          C.current,
          <Box flexDirection="column" gap={1}>
            {budgetRow('Groq (today)', groq, 200000)}
            {windowRow('Claude 5-hour', 'five_hour')}
            {windowRow('Claude 7-day', 'seven_day')}
          </Box>,
        )}
      </Box>
    )
  })
}
