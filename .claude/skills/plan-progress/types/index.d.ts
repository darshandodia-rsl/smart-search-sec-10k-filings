declare module 'claude-code' {
  interface PluginState {
    'plan-progress': { tick: Shaped<number> }
  }
}
