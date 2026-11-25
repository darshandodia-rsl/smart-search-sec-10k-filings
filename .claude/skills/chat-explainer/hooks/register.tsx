/**
 * chat-explainer: a side panel that explains the current chat in plain language.
 *
 * It keeps a short summary of what was built, what changed and why, and lets
 * the person type a question ("why did we chunk this way?") and read the answer
 * in the panel instead of the main chat. It sits next to the plan-progress
 * panel as another tab. Nothing is sent to the model until the person presses
 * Refresh, runs /recap, or submits a question.
 */
import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { QA } from '../types'

const PANE = 'chat-explainer'
// The model used for summaries and answers. 'haiku' is the cheapest, which
// matters because every call counts against the Claude plan's usage window.
const MODEL = 'haiku'
// How much of the chat is sent with each request, counted in characters. The
// newest part is kept, since that is what the person is usually asking about.
const MAX_TRANSCRIPT_CHARS = 30000

const summary = atom({ plugin: 'chat-explainer', key: 'summary' } as const, '', { shape: 'summary-1' })
const updatedAt = atom({ plugin: 'chat-explainer', key: 'updatedAt' } as const, '', { shape: 'updatedAt-1' })
const qa = atom({ plugin: 'chat-explainer', key: 'qa' } as const, [] as QA[], { shape: 'qa-1' })
// What the panel is waiting on ('' when idle): shown as a status line.
const busy = atom({ plugin: 'chat-explainer', key: 'busy' } as const, '', { shape: 'busy-1' })

// Colours match the plan-progress panel so the two tabs look like one set.
const C = {
  blue: '#79c0ff',
  violet: '#d2a8ff',
  current: '#f2cc60',
  text: '#c9d1d9',
  muted: '#8b949e',
  border: '#586069',
  warn: '#ff7b72',
}

const SYSTEM = [
  'You explain a coding chat to someone who is learning by building the project.',
  'Use plain, simple language and short sentences. Define jargon the first time you use it.',
  'Say what was done and why. Use a small example or analogy when it helps.',
  'Only use facts found in the chat. If the chat does not say, say that plainly instead of guessing.',
  'Reply in markdown. Prefer short bullet lists. No preamble.',
].join('\n')

/** Turns the chat into plain text the model can read: who said what, plus which files were touched. */
async function transcript($: any): Promise<string> {
  const msgs: any[] = await $.session.messages()
  const lines: string[] = []
  for (const m of msgs) {
    const who = m.role === 'user' ? 'USER' : 'CLAUDE'
    if (m.text) lines.push(`${who}: ${m.text}`)
    for (const t of m.toolUses ?? []) {
      const file = t.input?.file_path ?? t.input?.path ?? t.input?.command ?? ''
      lines.push(`  [${t.tool}${file ? ' ' + String(file).slice(0, 160) : ''}]`)
    }
  }
  const all = lines.join('\n')
  return all.length > MAX_TRANSCRIPT_CHARS ? '…(earlier chat cut)…\n' + all.slice(-MAX_TRANSCRIPT_CHARS) : all
}

/** Asks the model one question and returns its text, or a short error message. */
async function ask($: any, prompt: string, maxTokens: number): Promise<string> {
  const r = await $.model.complete({ model: MODEL, system: SYSTEM, prompt, maxTokens, timeoutMs: 90000 })
  return r.isAnswered ? String(r.text).trim() : `Could not get an answer (${r.reason}).`
}

/** Rebuilds the summary from the chat, keeping the old one so cut-off history is not lost. */
async function refresh($: any) {
  // One refresh at a time: a second request while one runs is ignored.
  if (await read($, busy)) return
  await update($, busy, () => 'Summarizing the chat…')
  try {
    const chat = await transcript($)
    const prev = await read($, summary)
    const text = await ask(
      $,
      `${prev ? `Previous summary (older parts of the chat may no longer appear below):\n${prev}\n\n` : ''}Chat so far:\n${chat}\n\n` +
        'Write an updated summary with exactly these three headings: ' +
        '"## What was built", "## What changed recently", "## Why we did it this way". ' +
        'Keep it under 250 words.',
      900,
    )
    await update($, summary, () => text)
    const stamp = new Date(await $.clock.now()).toLocaleTimeString()
    await update($, updatedAt, () => stamp)
  } finally {
    await update($, busy, () => '')
  }
}

/** Answers one typed question using the chat as the only source. */
async function answer($: any, q: string) {
  await update($, busy, () => 'Thinking…')
  try {
    const chat = await transcript($)
    const prev = await read($, summary)
    const a = await ask(
      $,
      `${prev ? `Summary so far:\n${prev}\n\n` : ''}Chat:\n${chat}\n\nQuestion: ${q}\n\nAnswer simply, in under 150 words.`,
      700,
    )
    await update($, qa, list => [...list, { q, a }].slice(-20))
  } finally {
    await update($, busy, () => '')
  }
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    await $.command.register({
      name: 'recap',
      description: 'Refresh the chat explainer panel',
    })
    void $.ui.open({ id: PANE, title: 'Chat explainer' })
    return next(e)
  })

  // Whenever the panel is turned on, build the summary of the chat so far.
  // An empty chat (the panel opening at session start) has nothing to summarize.
  on('ui.open', { id: PANE }, async ($, e, next) => {
    const opened = await next(e)
    const msgs: any[] = await $.session.messages()
    if (msgs.some(m => m.text)) void refresh($)
    return opened
  })

  on('command.run', { command: 'recap' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Chat explainer' })
    void refresh($)
    return { text: 'Refreshing the chat explainer…' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button, Input, Markdown } = $.ui.resolve(e)
    const sum = await read($, summary)
    const when = await read($, updatedAt)
    const history = await read($, qa)
    const status = await read($, busy)

    // One rounded card with a coloured heading, like the progress panel.
    const card = (title: string, accent: string, body: any) => (
      <Box flexDirection="column" borderStyle="round" borderColor={C.border} paddingX={1}>
        <Text bold color={accent}>
          {title}
        </Text>
        {body}
      </Box>
    )

    return (
      <Box flexDirection="column">
        {card(
          'Summary',
          C.blue,
          <Box flexDirection="column">
            {sum ? (
              <Markdown text={sum} />
            ) : (
              <Text color={C.muted}>No summary yet. Press Refresh or run /recap.</Text>
            )}
            <Box gap={2}>
              <Button key="refresh" label="Refresh" variant="primary" onPress={() => void refresh($)} />
              {when ? <Text color={C.muted}>updated {when}</Text> : null}
            </Box>
          </Box>,
        )}

        {history.length > 0 &&
          card(
            'Answers',
            C.violet,
            <Box flexDirection="column" gap={1}>
              {history.slice(-2).map(h => (
                <Box flexDirection="column">
                  <Text bold color={C.current}>
                    Q: {h.q}
                  </Text>
                  <Markdown text={h.a} />
                </Box>
              ))}
            </Box>,
          )}

        {card(
          'Ask',
          C.current,
          <Box flexDirection="column">
            <Input
              key="ask"
              placeholder="Why did we do it this way?"
              submitLabel="ask"
              onSubmit={(value: string) => {
                const q = value.trim()
                if (q) void answer($, q)
              }}
            />
            {status ? <Text color={C.muted}>{status}</Text> : null}
          </Box>,
        )}
      </Box>
    )
  })
}
