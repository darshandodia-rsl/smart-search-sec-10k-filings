/**
 * chat-explainer: a side panel that explains the current chat in plain language.
 *
 * It keeps a short summary of what was built, what changed and why. The person
 * types questions into the panel and a real, separate Claude session answers
 * them: it is forked from the main chat (so it knows everything said so far),
 * can read the project's files, and remembers earlier questions. It sits next to the plan-progress
 * panel as another tab. Nothing is sent to the model until the person presses
 * Refresh, runs /question, or submits a question.
 */
import { atom, read, update } from 'claude-code'
import type { Register } from 'claude-code'

import type { QA } from '../types'

const PANE = 'chat-explainer'
// Summaries always use the cheapest model, because every call counts against
// the Claude plan's usage window. Answers use whichever model the picker holds.
const SUMMARY_MODEL = 'haiku'
const MODELS = ['haiku', 'sonnet', 'opus']
// The key the chosen answer model is saved under, so it survives restarts.
const STORE_KEY = 'answerModel'
// How much of the chat is sent with each request, counted in characters. The
// newest part is kept, since that is what the person is usually asking about.
const MAX_TRANSCRIPT_CHARS = 30000
// The tools the answering session may use. Read-only on purpose: this window
// explains the work, and the main session stays the only place code changes.
const ALLOWED_TOOLS = 'Read,Grep,Glob'
// How long one answer may take (a real session may read several files first).
const ANSWER_TIMEOUT_MS = 300000
// True once session.start saw a person at the prompt. The panel's own headless
// helper sessions load this mod too, and must not open panels or summarize.
let isInteractive = false

const summary = atom({ plugin: 'chat-explainer', key: 'summary' } as const, '', { shape: 'summary-1' })
const updatedAt = atom({ plugin: 'chat-explainer', key: 'updatedAt' } as const, '', { shape: 'updatedAt-1' })
const qa = atom({ plugin: 'chat-explainer', key: 'qa' } as const, [] as QA[], { shape: 'qa-1' })
// The model that answers questions (saved in $.store, mirrored here so the panel redraws).
const model = atom({ plugin: 'chat-explainer', key: 'model' } as const, 'haiku', { shape: 'model-1' })
// Whether the summary card is expanded (collapse it to give the conversation room).
const showSummary = atom({ plugin: 'chat-explainer', key: 'showSummary' } as const, true, { shape: 'showSummary-1' })
// The id of the answering Claude session ('' until the first question), so later questions resume it and it remembers them.
const claudeSession = atom({ plugin: 'chat-explainer', key: 'claudeSession' } as const, '', { shape: 'claudeSession-1' })
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

// Extra rules for the answering session, added to Claude Code's own prompt.
const EXPLAINER_PROMPT =
  'You are a side-window explainer. You were forked from the main coding chat, so you know what was built and why. ' +
  'You are read-only: never edit files or run commands. You may read project files to check facts. ' +
  'The user is learning by building. Answer in plain, simple language with short sentences, define jargon, and keep answers short.'

const SYSTEM = [
  'You explain a coding chat to someone who is learning by building the project.',
  'Use plain, simple language and short sentences. Define jargon the first time you use it.',
  'Say what was done and why. Use a small example or analogy when it helps.',
  'Only use facts found in the chat. If the chat does not say, say that plainly instead of guessing.',
  'This is an ongoing conversation: use the earlier questions and answers to understand follow-ups like "why?" or "and the second one?".',
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
async function ask($: any, prompt: string, maxTokens: number, which: string): Promise<string> {
  const r = await $.model.complete({ model: which, system: SYSTEM, prompt, maxTokens, timeoutMs: 90000 })
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
      SUMMARY_MODEL,
    )
    await update($, summary, () => text)
    const stamp = new Date(await $.clock.now()).toLocaleTimeString()
    await update($, updatedAt, () => stamp)
  } finally {
    await update($, busy, () => '')
  }
}

/**
 * Answers one typed question with a real headless Claude session.
 *
 * The first question forks the main chat, so the helper starts out knowing
 * everything said so far; later questions resume that helper, so it remembers
 * what was asked. The question shows in the panel at once and the answer
 * replaces the "Thinking…" placeholder when it arrives.
 */
async function answer($: any, q: string) {
  await update($, qa, list => [...list, { q, a: 'Thinking…' }].slice(-50))
  await update($, busy, () => 'Thinking…')
  let a: string
  try {
    const sid = await read($, claudeSession)
    const argv = [
      'claude',
      '-p',
      ` ${q}`, // the leading space keeps a question that starts with "-" from being read as a flag
      '--output-format',
      'json',
      '--model',
      await read($, model),
      '--allowedTools',
      ALLOWED_TOOLS,
      '--append-system-prompt',
      EXPLAINER_PROMPT,
      ...(sid ? ['--resume', sid] : ['--resume', await $.session.id(), '--fork-session']),
    ]
    const r = await $.process.run(argv, { timeoutMs: ANSWER_TIMEOUT_MS })
    let out: any = null
    try {
      out = JSON.parse(String(r.stdout))
    } catch {}
    if (r.exitCode !== 0 || !out || out.is_error) {
      a = `The answering session failed: ${String(out?.result ?? r.stderr ?? 'no output').slice(0, 400)}`
    } else {
      a = String(out.result ?? '').trim() || '(empty answer)'
      if (out.session_id) await update($, claudeSession, () => String(out.session_id))
    }
  } catch (err) {
    a = `Something went wrong: ${String(err)}`
  }
  await update($, qa, list => [...list.slice(0, -1), { q, a }])
  await update($, busy, () => '')
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    isInteractive = e.isInteractive
    if (!isInteractive) return next(e)
    await $.command.register({
      name: 'question',
      description: 'Refresh the chat explainer panel',
    })
    const saved = await $.store.get(STORE_KEY)
    if (typeof saved === 'string' && MODELS.includes(saved)) await update($, model, () => saved)
    void $.ui.open({ id: PANE, title: 'Chat explainer' })
    return next(e)
  })

  // Whenever the panel is turned on, build the summary of the chat so far.
  // An empty chat (the panel opening at session start) has nothing to summarize.
  on('ui.open', { id: PANE }, async ($, e, next) => {
    const opened = await next(e)
    if (!isInteractive) return opened
    const msgs: any[] = await $.session.messages()
    if (msgs.some(m => m.text)) void refresh($)
    return opened
  })

  on('command.run', { command: 'question' }, async $ => {
    await $.ui.open({ id: PANE, title: 'Chat explainer' })
    void refresh($)
    return { text: 'Refreshing the chat explainer…' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button, Input, Markdown } = $.ui.resolve(e)
    const sum = await read($, summary)
    const when = await read($, updatedAt)
    const convo = await read($, qa)
    const status = await read($, busy)
    const chosen = await read($, model)
    const open = await read($, showSummary)

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
            {open &&
              (sum ? (
                <Markdown text={sum} />
              ) : (
                <Text color={C.muted}>No summary yet. Press Refresh or run /question.</Text>
              ))}
            <Box gap={2}>
              <Button key="refresh" label="Refresh" variant="primary" onPress={() => void refresh($)} />
              <Button
                key="toggle"
                label={open ? 'Hide' : 'Show'}
                variant="secondary"
                onPress={() => void update($, showSummary, v => !v)}
              />
              {when ? <Text color={C.muted}>updated {when}</Text> : null}
            </Box>
          </Box>,
        )}

        {convo.length > 0 &&
          card(
            'Conversation',
            C.violet,
            <Box flexDirection="column" gap={1}>
              {convo.map(item => (
                <Box flexDirection="column">
                  <Text bold color={C.current}>
                    You: {item.q}
                  </Text>
                  <Markdown text={item.a} />
                </Box>
              ))}
              <Button
                key="clear"
                label="Clear chat"
                variant="secondary"
                onPress={() => {
                  void update($, qa, () => [])
                  void update($, claudeSession, () => '')
                }}
              />
            </Box>,
          )}

        {card(
          'Ask',
          C.current,
          <Box flexDirection="column">
            <Box gap={1}>
              <Text color={C.muted}>Answer with:</Text>
              {MODELS.map(m => (
                <Button
                  key={`model-${m}`}
                  label={m}
                  variant={m === chosen ? 'primary' : 'secondary'}
                  onPress={async () => {
                    await update($, model, () => m)
                    await $.store.set(STORE_KEY, m)
                  }}
                />
              ))}
            </Box>
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
