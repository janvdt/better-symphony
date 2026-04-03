/**
 * OpenCode Runner
 * Launches OpenCode CLI in run mode with JSON output.
 * Parses structured JSON events for real-time monitoring.
 */

import { appendFileSync, writeFileSync, unlinkSync } from "fs";
import { join } from "path";
import type {
  ServiceConfig,
  Issue,
  AgentEvent,
  AgentEventType,
  LiveSession,
} from "../config/types.js";
import { AgentError } from "../config/types.js";
import { logger } from "../logging/logger.js";
import { createSession, updateSessionEvent, updateSessionTokens } from "./session.js";

export type AgentEventCallback = (event: AgentEvent) => void;

export interface OpenCodeRunnerOptions {
  config: ServiceConfig;
  issue: Issue;
  workspacePath: string;
  prompt: string;
  attempt: number | null;
  onEvent: AgentEventCallback;
  abortSignal: AbortSignal;
  /** If set, write a human-readable transcript to this file path */
  transcriptPath?: string;
}

export class OpenCodeRunner {
  private options: OpenCodeRunnerOptions;
  private proc: ReturnType<typeof Bun.spawn> | null = null;
  private session: LiveSession | null = null;
  private lastAssistantMessage: string | null = null;
  private accumulatedTokens = { input: 0, output: 0, total: 0 };
  private accumulatedCost = 0;
  private stepCount = 0;

  constructor(options: OpenCodeRunnerOptions) {
    this.options = options;
    if (options.transcriptPath) {
      const header = `# Agent Transcript: ${options.issue.identifier}\nStarted: ${new Date().toISOString()}\n`;
      writeFileSync(options.transcriptPath, header, "utf-8");
    }
  }

  private writeTranscript(line: string): void {
    if (!this.options.transcriptPath) return;
    try {
      appendFileSync(this.options.transcriptPath, line + "\n", "utf-8");
    } catch {}
  }

  getSession(): LiveSession | null {
    return this.session;
  }

  async run(): Promise<void> {
    const sessionId = `opencode-${Date.now()}`;
    this.session = createSession(sessionId, "turn-1", null);

    this.emitEvent("session_started", {
      session_id: sessionId,
      issue_identifier: this.options.issue.identifier,
    });

    try {
      await this.launchOpenCode(this.options.prompt);
    } finally {
      this.cleanup();
    }
  }

  private async launchOpenCode(prompt: string): Promise<void> {
    const { config, workspacePath, issue } = this.options;

    const args: string[] = [
      "opencode",
      "run",
      "--format",
      "json",
      "--dir",
      workspacePath,
    ];

    if (config.agent.model) {
      args.push("--model", config.agent.model);
    }

    args.push(prompt);

    logger.info("Launching OpenCode", {
      issue_identifier: issue.identifier,
      cwd: workspacePath,
      binary: "opencode",
      promptLength: prompt.length,
    });

    this.proc = Bun.spawn(args, {
      cwd: workspacePath,
      stdin: "ignore",
      stdout: "pipe",
      stderr: "pipe",
      env: {
        ...process.env,
        SYMPHONY_WORKSPACE: workspacePath,
        SYMPHONY_ISSUE_ID: issue.id,
        SYMPHONY_ISSUE_IDENTIFIER: issue.identifier,
        SYMPHONY_LINEAR: new URL("../linear-cli.ts", import.meta.url).pathname,
      },
    });

    if (this.session) {
      this.session.process_pid = this.proc.pid?.toString() ?? null;
    }

    const killProc = () => {
      try {
        this.proc?.kill("SIGTERM");
      } catch {}
      setTimeout(() => {
        try {
          this.proc?.kill("SIGKILL");
        } catch {}
      }, 5000);
    };

    if (this.options.abortSignal.aborted) {
      killProc();
      throw new AgentError("turn_cancelled", "Run aborted before start");
    }
    this.options.abortSignal.addEventListener("abort", killProc, { once: true });

    // Turn timeout
    const timeoutMs = config.agent.turn_timeout_ms;
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      logger.warn("OpenCode turn timeout", {
        issue_identifier: issue.identifier,
        timeout_ms: timeoutMs,
      });
      killProc();
    }, timeoutMs);

    // Stall detection
    const stallTimeoutMs = config.agent.stall_timeout_ms;
    let stallTimer: Timer | null = null;
    const resetStallTimer = () => {
      if (stallTimer) clearTimeout(stallTimer);
      if (stallTimeoutMs > 0) {
        stallTimer = setTimeout(() => {
          logger.warn("OpenCode stall detected", {
            issue_identifier: issue.identifier,
            stall_timeout_ms: stallTimeoutMs,
          });
          killProc();
        }, stallTimeoutMs);
      }
    };
    resetStallTimer();

    try {
      await this.readJsonStream(resetStallTimer);
    } finally {
      clearTimeout(timeout);
      if (stallTimer) clearTimeout(stallTimer);
      this.options.abortSignal.removeEventListener("abort", killProc);
    }

    const rawStderr = this.proc.stderr && typeof this.proc.stderr !== "number"
      ? await new Response(this.proc.stderr).text()
      : "";

    const exitCode = await this.proc.exited;

    logger.info("OpenCode process exited", {
      issue_identifier: issue.identifier,
      exitCode,
      stderr: rawStderr.slice(0, 500) || "(empty)",
    });

    if (timedOut) {
      this.emitEvent("turn_failed", { exitCode, reason: "timeout" });
      throw new AgentError("turn_timeout", `Turn timed out after ${timeoutMs}ms`);
    }

    if (this.options.abortSignal.aborted) {
      this.emitEvent("turn_cancelled", { exitCode });
      throw new AgentError("turn_cancelled", "Run aborted");
    }

    if (exitCode !== 0) {
      const errorContext = this.lastAssistantMessage || rawStderr.slice(0, 500);
      this.emitEvent("turn_failed", { exitCode, stderr: rawStderr.slice(0, 500) });
      throw new AgentError("turn_failed", `OpenCode exited with code ${exitCode}: ${errorContext.slice(0, 200)}`);
    }

    this.emitEvent("turn_completed", { exitCode });
  }

  private async readJsonStream(onActivity: () => void): Promise<void> {
    const stdout = this.proc!.stdout;
    if (!stdout || typeof stdout === "number") {
      throw new AgentError("agent_not_found", "OpenCode stdout not available as stream");
    }
    const reader = stdout.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        if (this.options.abortSignal.aborted) break;

        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed) continue;

          onActivity();

          let message: any;
          try {
            message = JSON.parse(trimmed);
          } catch {
            continue;
          }

          this.handleStreamMessage(message);
        }
      }
    } finally {
      reader.releaseLock();
    }

    if (buffer.trim()) {
      try {
        const message = JSON.parse(buffer.trim());
        this.handleStreamMessage(message);
      } catch {}
    }
  }

  private handleStreamMessage(message: any): void {
    const msgType = message.type;

    if (msgType === "step_start") {
      this.handleStepStart(message);
    } else if (msgType === "text") {
      this.handleText(message);
    } else if (msgType === "tool_use") {
      this.handleToolUse(message);
    } else if (msgType === "step_finish") {
      this.handleStepFinish(message);
    }
  }

  private handleStepStart(message: any): void {
    this.stepCount++;
    const sessionId = message.sessionID;

    if (this.stepCount === 1 && sessionId && this.session) {
      this.session.session_id = sessionId;
      logger.info("OpenCode session init", {
        issue_identifier: this.options.issue.identifier,
        session_id: sessionId,
      });
      this.writeTranscript(`\n## Session Init\nSession: ${sessionId}\n`);
      if (this.session) {
        updateSessionEvent(this.session, "system:init", `session=${sessionId}`);
      }
    }
  }

  private handleText(message: any): void {
    const text = message.part?.text;
    if (!text?.trim()) return;

    const issueId = this.options.issue.identifier;
    this.lastAssistantMessage = text;
    this.emitEvent("assistant_message", { text: text.slice(0, 500) });

    logger.debug(text.slice(0, 200), { issue_identifier: issueId });
    this.writeTranscript(`\n### Assistant\n${text}\n`);

    if (this.session) {
      updateSessionEvent(this.session, "assistant", text.slice(0, 200));
    }
  }

  private handleToolUse(message: any): void {
    const part = message.part;
    if (!part) return;

    const issueId = this.options.issue.identifier;
    const toolName = part.tool ?? "unknown";
    const state = part.state ?? {};
    const input = state.input ?? {};
    const detail =
      input.command ??
      input.filePath ??
      input.file_path ??
      input.pattern ??
      input.description ??
      input.query ??
      input.url ??
      input.prompt ??
      state.title ??
      "";
    const truncated = detail ? String(detail).slice(0, 120) : "";

    this.emitEvent("tool_use", { tool: toolName, detail: truncated });
    logger.info(truncated ? `${toolName} ${truncated}` : toolName, {
      issue_identifier: issueId,
    });
    this.writeTranscript(`\n### Tool: ${toolName}\n${truncated}\n`);

    if (this.session) {
      updateSessionEvent(this.session, `tool:${toolName}`, truncated);
    }

    // OpenCode includes the result inline
    const output = state.output;
    if (output) {
      const summary = `-> ${String(output).slice(0, 200)}`;
      this.emitEvent("tool_result", { summary });
      logger.debug(summary, { issue_identifier: issueId });
      if (this.session) updateSessionEvent(this.session, "tool_result", summary.slice(0, 200));
    }
  }

  private handleStepFinish(message: any): void {
    const part = message.part ?? {};
    const tokens = part.tokens;
    const cost = part.cost ?? 0;

    if (tokens) {
      this.accumulatedTokens.input += tokens.input ?? 0;
      this.accumulatedTokens.output += tokens.output ?? 0;
      this.accumulatedTokens.total += tokens.total ?? 0;
    }
    this.accumulatedCost += cost;

    if (this.session && tokens) {
      updateSessionTokens(this.session, {
        input_tokens: this.accumulatedTokens.input,
        output_tokens: this.accumulatedTokens.output,
        total_tokens: this.accumulatedTokens.total,
      });

      this.emitEvent("token_usage_updated", {
        usage: {
          input_tokens: this.accumulatedTokens.input,
          output_tokens: this.accumulatedTokens.output,
          total_tokens: this.accumulatedTokens.total,
        },
        cost_usd: this.accumulatedCost,
      });
    }

    if (this.accumulatedCost > 0 && this.session) {
      this.session.cost_usd = this.accumulatedCost;
    }

    const reason = part.reason;
    if (reason === "stop") {
      const parts: string[] = [];
      if (this.accumulatedCost > 0) parts.push(`$${this.accumulatedCost.toFixed(4)}`);
      if (this.stepCount > 0) parts.push(`${this.stepCount} steps`);
      const suffix = parts.length ? ` (${parts.join(", ")})` : "";
      logger.info(`OpenCode completed${suffix}`, {
        issue_identifier: this.options.issue.identifier,
        cost_usd: this.accumulatedCost,
        steps: this.stepCount,
      });
      this.writeTranscript(`\n## Result\nCompleted${suffix}\n`);
    }
  }

  private emitEvent(eventType: AgentEventType, payload?: unknown): void {
    const event: AgentEvent = {
      event: eventType,
      timestamp: new Date(),
      process_pid: this.session?.process_pid || this.proc?.pid?.toString(),
      payload,
    };

    if (this.session) {
      event.usage = {
        input_tokens: this.session.input_tokens,
        output_tokens: this.session.output_tokens,
        total_tokens: this.session.total_tokens,
      };
      event.cost_usd = this.session.cost_usd;
      event.duration_ms = this.session.duration_ms;
    }

    this.options.onEvent(event);
  }

  terminate(): void {
    try {
      this.proc?.kill("SIGTERM");
    } catch {}
    setTimeout(() => {
      try {
        this.proc?.kill("SIGKILL");
      } catch {}
    }, 5000);
  }

  private cleanup(): void {
    this.terminate();
  }
}
