/**
 * The plan-progress panel's session state: the values its drawing reads.
 */

/** One usage window: how much is used (0–100) and when it resets (ISO time). */
export type Meter = { percentUsed: number; resetsAt?: string }

/** Antigravity's Gemini meters, read from `agy -p /usage`. */
export type AgyMeters = { fiveHour: Meter; sevenDay: Meter }

declare module 'claude-code' {
  interface PluginState {
    'plan-progress': {
      /** Bumped after each tool call so the panel re-reads its files. */
      tick: Shaped<number>
      /** Bumped on a timer so the Extraction pane re-reads the job store while idle. */
      extTick: Shaped<number>
      /** The latest Antigravity meter reading; null until the first one lands. */
      agy: Shaped<AgyMeters | null>
    }
  }
}
