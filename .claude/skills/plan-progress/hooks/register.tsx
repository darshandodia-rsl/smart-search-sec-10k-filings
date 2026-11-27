import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { AgyMeters, Meter } from '../types'

const PANE = 'plan-progress'
// Bumped after each tool call so the pane re-reads PLAN.md, git and budgets.
const tick = atom({ plugin: 'plan-progress', key: 'tick' } as const, 0, {
  shape: 'tick-1',
})
// The latest Antigravity (Gemini) usage reading, refreshed by a timer.
const agy = atom({ plugin: 'plan-progress', key: 'agy' } as const, null as AgyMeters | null, {
  shape: 'agy-1',
})
// `agy -p /usage` takes about 10 seconds, so it is read on a timer, not per redraw.
const AGY_POLL_MS = 5 * 60 * 1000

// The Extraction pane: per-company progress of the LLM entity/relationship extraction.
const EXT_PANE = 'extraction-progress'
// Bumped on a timer so the pane follows the runner even when this chat is idle.
const extTick = atom({ plugin: 'plan-progress', key: 'extTick' } as const, 0, {
  shape: 'extTick-1',
})
const EXT_POLL_MS = 30 * 1000
// The job store the extraction runner writes (relative to the project root).
const JOBS_DB = 'data/state/extraction_jobs.sqlite'
// All companies in the corpus (companies.csv); the job store only holds those added so far.
const TOTAL_COMPANIES = 130
// A failed item is retried until it has been tried this many times (jobs.py MAX_ATTEMPTS).
const MAX_ATTEMPTS = 3
// Display names for the job store's backend ids.
const BACKENDS: Record<string, { name: string; color: string }> = {
  groq: { name: 'Groq', color: '#ffa657' },
  antigravity_cli: { name: 'Antigravity', color: '#79c0ff' },
  claude_cli: { name: 'Sonnet', color: '#d2a8ff' },
}

// The fiscal years in the corpus, one box column each in the Filings grid.
const YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
const CORPUS = 'data/edgar-software-corpus'

/**
 * Shortens an SEC company name for the Filings grid, so more of it fits in a
 * narrow pane: drops SEC state tags ("/DE/") and trailing legal or holding
 * words ("Holdings, Inc.", "Corp", "plc", "N.V."), then turns names stored all
 * in capitals into title case. "CrowdStrike Holdings, Inc." becomes
 * "CrowdStrike"; "MICROSOFT CORP" becomes "Microsoft".
 */
function tidyName(name: string): string {
  let s = name.replace(/\s*\/[A-Z]{2}\/?\s*$/, '').trim()
  const suffix = /[\s,]+(inc\.?|corp\.?|corporation|co\.?|ltd\.?|plc|n\.v\.|holdings?)$/i
  while (suffix.test(s)) s = s.replace(suffix, '').trim()
  if (s !== s.toUpperCase()) return s
  return s.toLowerCase().replace(/\b[a-z]/g, c => c.toUpperCase())
}

/**
 * Reads which fiscal years each company filed a 10-K for, from manifest.csv.
 *
 * The Filings grid draws one box per filed year. The manifest is the project's
 * authoritative filing list (one row per company-year), so a company with no
 * 10-K in a year gets no box there. Also returns the companies in
 * companies.csv rank order, so every company shows up even before it is queued.
 */
async function readFilings(
  $: any,
): Promise<{ order: string[]; names: Map<string, string>; filed: Map<string, Set<number>> } | null> {
  try {
    const filed = new Map<string, Set<number>>()
    // cik,ticker,name,fiscal_year,... (name may be quoted and contain commas)
    for (const line of (await $.fs.read(`${CORPUS}/manifest.csv`)).split('\n').slice(1)) {
      const m = line.match(/^\d+,([^,]+),(?:"[^"]*"|[^,]*),(\d{4}),/)
      if (!m || !m[1] || !m[2]) continue
      if (!filed.has(m[1])) filed.set(m[1], new Set())
      filed.get(m[1])!.add(Number(m[2]))
    }
    // rank,cik,ticker,name,... (name may be quoted and contain commas)
    const order: string[] = []
    const names = new Map<string, string>()
    for (const line of (await $.fs.read(`${CORPUS}/companies.csv`)).split('\n').slice(1)) {
      const m = line.match(/^\d+,\d+,([^,]+),(?:"([^"]*)"|([^,]*)),/)
      if (!m || !m[1]) continue
      order.push(m[1])
      names.set(m[1], tidyName(m[2] ?? m[3] ?? ''))
    }
    return { order, names, filed }
  } catch {
    return null
  }
}

/**
 * Reads how far each company-year's extraction has got, keyed "PANW:2015":
 * 'full' when every "extract" item of that year is finished (items out of
 * retries count as finished), 'part' when some but not all are done.
 *
 * The Filings grid colours a year's box from this: bright green for full,
 * dim green for part, grey for years with nothing done yet.
 */
async function readYearProgress($: any): Promise<Map<string, 'full' | 'part'>> {
  const sql = `SELECT ticker || ':' || fiscal_year,
      CASE WHEN SUM(status = 'done' OR (status = 'failed' AND attempts >= ${MAX_ATTEMPTS})) = COUNT(*)
           THEN 'full' ELSE 'part' END
    FROM items WHERE kind = 'extract'
    GROUP BY ticker, fiscal_year HAVING SUM(status = 'done') > 0`
  const out = new Map<string, 'full' | 'part'>()
  try {
    const r = await $.process.run(['/usr/bin/sqlite3', JOBS_DB, sql], { timeoutMs: 10000 })
    if (r.exitCode !== 0) return out
    for (const line of String(r.stdout).split('\n')) {
      const [k, v] = line.split('|')
      if (k && (v === 'full' || v === 'part')) out.set(k, v)
    }
  } catch {}
  return out
}

// The runner's events file (one JSON object per line); `run_stopped` marks the end of a run.
const EVENTS_FILE = 'data/logs/extraction_events.jsonl'

/** The most recent finished extraction run, taken from the events file. */
type LastRun = {
  /** ISO time the run stopped. */
  endedAt: string
  /** ISO time the run started, or null if its start is no longer in the file's tail. */
  startedAt: string | null
  /** The runner's own end-of-run summary, one entry per line. */
  lines: string[]
}

// Pid files of the extraction runners, one per backend (written by /extract-run); each holds a
// caffeinate PID that lives exactly as long as its runner. The `main` and `sonnet` files belong
// to runners started with several backends in one process, and are still checked so such a
// runner counts as alive.
const RUNNER_PIDS = [
  'data/state/extraction_runner_groq.pid',
  'data/state/extraction_runner_antigravity_cli.pid',
  'data/state/extraction_runner_claude_cli.pid',
  'data/state/extraction_runner_main.pid',
  'data/state/extraction_runner_sonnet.pid',
]

/**
 * Tells whether any extraction runner is alive right now.
 *
 * The "Last run" card only makes sense between runs, so the pane hides it while this is
 * true. The runners start and stop independently, so event order alone cannot say
 * whether one is still going; the pid files can.
 */
async function runnerActive($: any): Promise<boolean> {
  const pids: string[] = []
  for (const f of RUNNER_PIDS) {
    try {
      const p = (await $.fs.read(f)).trim()
      if (/^\d+$/.test(p)) pids.push(p)
    } catch {} // no pid file: that runner was never started
  }
  if (pids.length === 0) return false
  try {
    const r = await $.process.run(['/bin/ps', '-p', pids.join(','), '-o', 'pid='], { timeoutMs: 5000 })
    return String(r.stdout).trim().length > 0
  } catch {
    return false
  }
}

/**
 * Finds the latest `run_stopped` event and the `run_started` event before it.
 *
 * The Extraction pane's "Last run" card shows this, so the summary of a finished
 * wave stays visible after the runner has exited. Only the tail of the events
 * file is read, as the file grows for as long as the project runs. Returns null
 * when no run has stopped yet or the file cannot be read.
 */
async function readLastRun($: any): Promise<LastRun | null> {
  try {
    const r = await $.process.run(['/usr/bin/tail', '-c', '100000', EVENTS_FILE], { timeoutMs: 10000 })
    if (r.exitCode !== 0) return null
    const events: { at: string; kind: string; message?: string }[] = []
    for (const line of String(r.stdout).split('\n')) {
      try {
        events.push(JSON.parse(line))
      } catch {} // a blank line, or the line cut in half by tail
    }
    const stopIdx = events.map(x => x.kind).lastIndexOf('run_stopped')
    const stop = events[stopIdx]
    if (!stop) return null
    const start = events.slice(0, stopIdx).reverse().find(x => x.kind === 'run_started')
    return {
      endedAt: stop.at,
      startedAt: start?.at ?? null,
      lines: String(stop.message ?? '').split('\n').filter(l => l.trim()),
    }
  } catch {
    return null
  }
}

// Box colour for a partly extracted year: a dimmer green than a finished one.
const PART_GREEN = '#2f7d43'
// Darker green for companies whose every filing (all years) is extracted.
const COMPLETE_GREEN = '#238636'

/** One company's extraction progress, as read from the job store. */
type CompanyRow = {
  seq: number
  ticker: string
  backend: string | null
  total: number | null
  done: number | null
  running: number | null
  stuck: number | null
  y0: number | null
  y1: number | null
  last: string | null
  /** Items of this company finished in the last ETA_WINDOW_MIN minutes: its recent pace. */
  recent: number | null
}

// The ETA uses the pace of the last this many minutes ("at the current pace").
const ETA_WINDOW_MIN = 60

/**
 * Reads per-company progress from the extraction job store.
 *
 * The Extraction pane draws from this. It counts "extract" items only (the ones
 * that need an LLM call), the same way the pipeline's own status report does.
 * The pane only runs SELECTs, and the job store is in WAL mode (readers never block the
 * runner). It is not opened with `-readonly`: that flag fails when no runner is active.
 * Returns null when the file is missing or unreadable.
 */
async function readCompanies($: any): Promise<CompanyRow[] | null> {
  const sql = `SELECT c.seq, c.ticker, c.backend,
      SUM(i.kind = 'extract') AS total,
      SUM(i.kind = 'extract' AND i.status = 'done') AS done,
      SUM(i.status = 'running') AS running,
      SUM(i.kind = 'extract' AND i.status = 'failed' AND i.attempts >= ${MAX_ATTEMPTS}) AS stuck,
      MIN(i.fiscal_year) AS y0, MAX(i.fiscal_year) AS y1, MAX(i.updated_at) AS last,
      (SELECT COUNT(*) FROM results r JOIN items ri USING (item_id)
        WHERE ri.ticker = c.ticker
          AND r.created_at >= strftime('%Y-%m-%dT%H:%M:%S', 'now', '-${ETA_WINDOW_MIN} minutes')) AS recent
    FROM companies c LEFT JOIN items i USING (ticker)
    GROUP BY c.ticker ORDER BY c.seq`
  try {
    const r = await $.process.run(['/usr/bin/sqlite3', '-json', JOBS_DB, sql], { timeoutMs: 10000 })
    if (r.exitCode !== 0) return null
    const out = String(r.stdout).trim()
    return out ? (JSON.parse(out) as CompanyRow[]) : []
  } catch {
    return null
  }
}

/**
 * Reads Antigravity's Gemini 5-hour and weekly meters from `agy -p /usage`.
 *
 * The panel shows these next to Claude's meters so the user can see how much of
 * each extraction backend's quota is left. `agy` prints tab-separated lines with
 * the % *remaining*; this returns % used, like Claude's. Returns null when the
 * command fails or its output is not understood, so the panel shows a dash.
 */
async function readAgy($: any): Promise<AgyMeters | null> {
  try {
    // An empty folder, so `agy` picks up no project files.
    const r = await $.process.run(['agy', '-p', '/usage'], { cwd: '/private/tmp', timeoutMs: 60000 })
    const found: Record<string, Meter> = {}
    for (const line of String(r.stdout).split('\n')) {
      const p = line.split('\t').map(x => x.trim())
      if (p.length === 4 && p[0] === 'Gemini Models' && p[1] && p[2]) {
        found[p[1]] = { percentUsed: 100 - parseFloat(p[2]), resetsAt: p[3] }
      }
    }
    const fiveHour = found['Five Hour Limit Remaining']
    const sevenDay = found['Weekly Limit Remaining']
    return fiveHour && sevenDay ? { fiveHour, sevenDay } : null
  } catch {
    return null
  }
}

/**
 * Takes a fresh Antigravity reading and stores it, keeping the last good one if
 * the read fails, so a flaky `agy` call doesn't blank the bars.
 */
async function pollAgy($: any) {
  const m = await readAgy($)
  if (m) await update($, agy, () => m)
}


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
    void pollAgy($)
    $.clock.every(AGY_POLL_MS, () => pollAgy($))
    await $.command.register({
      name: 'extraction',
      description: 'Open the per-company extraction progress panel',
    })
    void $.ui.open({ id: EXT_PANE, title: 'Extraction' })
    $.clock.every(EXT_POLL_MS, () => update($, extTick, n => n + 1))
    return next(e)
  })

  on('command.run', { command: 'progress' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Project progress' })
    return { text: 'Progress panel opened.' }
  })

  on('command.run', { command: 'extraction' }, async $ => {
    await $.ui.open({ id: EXT_PANE, title: 'Extraction' })
    return { text: 'Extraction panel opened.' }
  })

  // The Extraction pane: how many companies are done, and live detail for the
  // ones being extracted now (backend, % done, years, last activity).
  on('ui.render', { component: 'Pane', requestId: EXT_PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    await read($, tick)
    await read($, extTick)
    const w = (e.viewport?.columns ?? 44) - 10
    const now = await $.clock.now()

    const rows = await readCompanies($)
    const filings = await readFilings($)
    const yearProgress = await readYearProgress($)
    const running = await runnerActive($)
    const lastRun = running ? null : await readLastRun($)
    // Usage meters for the compact LLM usage card: the same sources as the
    // progress pane (Groq's daily budget file, Claude's session limits, agy).
    const agyMeters = await read($, agy)
    const today = new Date(now).toISOString().slice(0, 10)
    const groq = await budget($, 'budget_groq.json', today)
    const limits: { kind: string; percentUsed: number; resetsAt?: string }[] = await $.session
      .usage()
      .then((u: any) => u.rateLimits ?? [])
      .catch(() => [])
    if (rows === null) {
      return <Text color={C.muted}>No extraction job store yet ({JOBS_DB}).</Text>
    }

    // "left" = items still to do; stuck items (out of retries) don't count as left.
    const left = (r: CompanyRow) => (r.total ?? 0) - (r.done ?? 0) - (r.stuck ?? 0)
    const finished = rows.filter(r => (r.total ?? 0) > 0 && left(r) === 0)
    const active = rows.filter(r => r.backend && left(r) > 0)
    const queued = rows.filter(r => !r.backend && !finished.includes(r))
    const notAdded = Math.max(0, TOTAL_COMPANIES - rows.length)
    // Totally complete: caught up AND every year the company filed a 10-K for is fully extracted
    // (a company can be "caught up" on just the first years added so far).
    const complete: CompanyRow[] = []
    for (const r of finished) {
      const filedYears = filings ? filings.filed.get(r.ticker) : undefined
      if (!filedYears || filedYears.size === 0) continue
      let all = true
      filedYears.forEach(y => {
        if (yearProgress.get(`${r.ticker}:${y}`) !== 'full') all = false
      })
      if (all) complete.push(r)
    }
    // Light part of the bar: caught-up companies with years still to add, plus each
    // in-progress company's fraction done.
    const partial =
      finished.length -
      complete.length +
      active.reduce((n, r) => n + (r.total ? (r.done ?? 0) / r.total : 0), 0)
    const itemsDone = rows.reduce((s, r) => s + (r.done ?? 0), 0)
    const itemsTotal = rows.reduce((s, r) => s + (r.total ?? 0), 0)

    // ETA at the current pace: items left ÷ items finished in the last
    // ETA_WINDOW_MIN minutes. null when there is no recent pace to go on (a
    // company just claimed, or its backend paused by a quota).
    const dur = (mins: number) => {
      const m = Math.max(1, Math.round(mins))
      const d = Math.floor(m / 1440)
      const h = Math.floor((m % 1440) / 60)
      return d ? `${d}d ${h}h` : h ? `${h}h ${m % 60}m` : `${m}m`
    }
    const eta = (itemsLeft: number, recent: number) =>
      itemsLeft > 0 && recent > 0 ? (itemsLeft / recent) * ETA_WINDOW_MIN : null
    const itemsLeft = rows.reduce((s, r) => s + Math.max(0, left(r)), 0)
    const waveEta = eta(itemsLeft, rows.reduce((s, r) => s + (r.recent ?? 0), 0))

    const bar = (ratio: number, width: number, color: string) => {
      const f = Math.round(Math.min(1, Math.max(0, ratio)) * width)
      return (
        <Text color={color}>
          {'█'.repeat(f)}
          <Text color={C.track}>{'░'.repeat(width - f)}</Text>
        </Text>
      )
    }
    // Two-tone bar: green = totally complete companies, amber = partly done (in progress or caught up so far).
    const companiesBar = (dark: number, light: number, total: number, width: number) => {
      const fd = Math.max(0, Math.min(width, Math.round((dark / total) * width)))
      const fl = Math.max(
        0,
        Math.min(width - fd, light > 0 ? Math.max(1, Math.round((light / total) * width)) : 0),
      )
      return (
        <Text>
          <Text color={COMPLETE_GREEN}>{'█'.repeat(fd)}</Text>
          <Text color={C.current}>{'█'.repeat(fl)}</Text>
          <Text color={C.track}>{'░'.repeat(width - fd - fl)}</Text>
        </Text>
      )
    }
    const card = (title: string, accent: string, body: any) => (
      <Box flexDirection="column" borderStyle="round" borderColor={C.border} paddingX={1}>
        <Text bold color={accent}>
          {title}
        </Text>
        <Box flexDirection="column">{body}</Box>
      </Box>
    )
    const backendOf = (id: string | null) =>
      (id && BACKENDS[id]) || { name: id ?? '—', color: C.muted }
    const years = (r: CompanyRow) =>
      r.y0 == null ? '' : r.y0 === r.y1 ? `FY${r.y0}` : `FY${r.y0}–${String(r.y1).slice(2)}`
    const ago = (iso: string | null) => {
      if (!iso) return '—'
      const m = Math.max(0, Math.round((now - Date.parse(iso)) / 60000))
      return m < 60 ? `${m}m ago` : m < 1440 ? `${Math.floor(m / 60)}h ago` : `${Math.floor(m / 1440)}d ago`
    }
    // "2h 14m" until an ISO time.
    const until = (iso?: string) => {
      if (!iso) return ''
      const m = Math.max(0, Math.round((Date.parse(iso) - now) / 60000))
      const d = Math.floor(m / 1440)
      const h = Math.floor((m % 1440) / 60)
      return d ? `${d}d ${h}h` : h ? `${h}h ${m % 60}m` : `${m}m`
    }
    // One compact usage line: label, % used (red above 80%), then a muted detail.
    const usageLine = (label: string, pct: number | null, detail: string) => (
      <Text color={C.text}>
        {label.padEnd(15)}
        <Text bold color={pct !== null && pct > 80 ? C.warn : C.blue}>
          {pct === null ? '—' : `${Math.round(pct)}%`.padStart(4)}
        </Text>
        <Text color={C.muted}>{detail ? clip(`  ${detail}`, Math.max(0, w - 19)) : ''}</Text>
      </Text>
    )
    const windowLine = (label: string, l?: Meter) =>
      usageLine(label, l ? l.percentUsed : null, l?.resetsAt ? `resets ${until(l.resetsAt)}` : '')
    // Filings grid order: queued companies first (extraction priority), then the
    // rest of the corpus by companies.csv rank.
    const gridOrder = filings
      ? [...rows.map(r => r.ticker), ...filings.order.filter(t => !rows.some(r => r.ticker === t))]
      : []
    const tickW = Math.max(4, ...gridOrder.map(t => t.length)) + 1
    // One grid row: ticker, then a box per year (bright green = extracted, dim
    // green = partly, grey = not yet, blank = no 10-K that year), then the
    // company's name. The terminal cuts the row at the pane's real edge (with
    // "…"), so a long name never wraps onto the next row.
    const boxColor = (t: string, y: number) => {
      const p = yearProgress.get(`${t}:${y}`)
      return p === 'full' ? C.done : p === 'part' ? PART_GREEN : C.track
    }
    const gridRow = (t: string) => (
      <Text color={C.text} wrap="truncate-end">
        {t.padEnd(tickW)}
        {YEARS.map(y =>
          !filings?.filed.get(t)?.has(y) ? '  ' : <Text color={boxColor(t, y)}>■ </Text>,
        )}
        <Text color={C.muted}> {filings?.names.get(t) ?? ''}</Text>
      </Text>
    )
    const tickers = (list: CompanyRow[]) => (
      <Text color={C.text}>
        {list.map((r, i) => (
          <Text>
            {i ? ' · ' : ''}
            {r.ticker}
            <Text color={backendOf(r.backend).color}> {backendOf(r.backend).name}</Text>
          </Text>
        ))}
      </Text>
    )

    return (
      <Box flexDirection="column">
        {card(
          'Companies',
          C.blue,
          <Box flexDirection="column">
            <Text bold color={C.text}>
              {finished.length} / {TOTAL_COMPANIES} caught up
              <Text color={C.muted}> · {Math.round((finished.length / TOTAL_COMPANIES) * 100)}%</Text>
            </Text>
            {companiesBar(complete.length, partial, TOTAL_COMPANIES, Math.min(30, w))}
            <Text color={C.muted}>
              <Text color={C.current}>◐ {active.length}</Text> in progress · ○ {queued.length} queued ·{' '}
              {notAdded} not added
            </Text>
            <Text color={C.muted}>
              Items {itemsDone.toLocaleString()} / {itemsTotal.toLocaleString()}
              {itemsTotal ? ` (${Math.round((itemsDone / itemsTotal) * 100)}%)` : ''}
            </Text>
            {itemsLeft > 0 && (
              <Text color={C.muted}>
                Wave ETA{' '}
                <Text bold color={C.text}>
                  {waveEta === null ? '—' : `~${dur(waveEta)}`}
                </Text>
                {waveEta === null ? ' (no items finished in the last hour)' : ' at last hour’s pace'}
              </Text>
            )}
          </Box>,
        )}

        {running &&
          active.length > 0 &&
          card(
            'In progress',
            C.current,
            <Box flexDirection="column" gap={1}>
              {active.map(r => {
                const b = backendOf(r.backend)
                const pct = r.total ? (r.done ?? 0) / r.total : 0
                const rowEta = eta(left(r), r.recent ?? 0)
                return (
                  <Box flexDirection="column">
                    <Text color={C.text}>
                      <Text bold>{r.ticker}</Text> <Text color={b.color}>{b.name}</Text>{' '}
                      <Text color={C.muted}>{Math.round(pct * 100)}%</Text>
                      <Text color={C.muted}> · {rowEta === null ? 'ETA —' : `~${dur(rowEta)} left`}</Text>
                    </Text>
                    {bar(pct, Math.min(24, w), b.color)}
                    <Text color={C.muted}>
                      {clip(
                        `${r.done ?? 0}/${r.total ?? 0} items · ${years(r)}` +
                          (r.running ? ` · ${r.running} running` : '') +
                          (r.stuck ? ` · ${r.stuck} stuck` : '') +
                          ` · ${ago(r.last)}`,
                        w,
                      )}
                    </Text>
                  </Box>
                )
              })}
            </Box>,
          )}

        {finished.length > 0 && card('Caught up', C.done, tickers(finished))}

        {queued.length > 0 && card('Queued', C.muted, tickers(queued))}

        {lastRun &&
          card(
            'Last run',
            C.more,
            <Box flexDirection="column">
              <Text color={C.text}>
                Ended {ago(lastRun.endedAt)}
                {lastRun.startedAt && (
                  <Text color={C.muted}>
                    {' '}
                    · took {dur((Date.parse(lastRun.endedAt) - Date.parse(lastRun.startedAt)) / 60000)}
                  </Text>
                )}
              </Text>
              {lastRun.lines.map(l => (
                <Text color={C.muted} wrap="truncate-end">
                  {l}
                </Text>
              ))}
            </Box>,
          )}

        {card(
          'LLM usage',
          C.current,
          <Box flexDirection="column">
            {usageLine(
              'Groq today',
              groq === null ? null : (groq / 200000) * 100,
              groq === null ? '' : `${Math.round(groq / 1000)}K / 200K`,
            )}
            {windowLine('Claude 5h', limits.find(x => x.kind === 'five_hour'))}
            {windowLine('Claude 7d', limits.find(x => x.kind === 'seven_day'))}
            {windowLine('Antigravity 5h', agyMeters?.fiveHour)}
            {windowLine('Antigravity 7d', agyMeters?.sevenDay)}
          </Box>,
        )}

        {filings &&
          card(
            'Filings',
            C.done,
            <Box flexDirection="column">
              <Text color={C.muted}>
                {' '.repeat(tickW)}
                {YEARS.map(y => String(y % 10) + ' ')}
              </Text>
              {gridOrder.map(gridRow)}
              <Text color={C.muted}>
                <Text color={C.done}>■</Text> extracted <Text color={PART_GREEN}>■</Text> partly{' '}
                <Text color={C.track}>■</Text> not yet · FY2015–24
              </Text>
            </Box>,
          )}
      </Box>
    )
  })

  on('tool.call', async ($, e, next) => {
    const ran = await next(e)
    await update($, tick, n => n + 1)
    return ran
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text } = $.ui.resolve(e)
    await read($, tick)
    const agyMeters = await read($, agy)
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
    // One usage window: label, "% · resets in …", and a bar (red above 80%).
    const windowRow = (label: string, l?: Meter) => {
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
            {windowRow('Claude 5-hour', limits.find(x => x.kind === 'five_hour'))}
            {windowRow('Claude 7-day', limits.find(x => x.kind === 'seven_day'))}
            {windowRow('Antigravity 5-hour', agyMeters?.fiveHour)}
            {windowRow('Antigravity 7-day', agyMeters?.sevenDay)}
          </Box>,
        )}
      </Box>
    )
  })
}
