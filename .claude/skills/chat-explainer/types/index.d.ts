export type QA = { q: string; a: string }

declare module 'claude-code' {
  interface PluginState {
    'chat-explainer': {
      summary: Shaped<string>
      updatedAt: Shaped<string>
      qa: Shaped<QA[]>
      busy: Shaped<string>
      model: Shaped<string>
      showSummary: Shaped<boolean>
      claudeSession: Shaped<string>
    }
  }
}
