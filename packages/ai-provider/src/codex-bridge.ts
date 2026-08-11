import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { AgentMessage, AgentToolCall, AgentToolDef } from '@genoffice/agent-core'
import type { AiChatResponse } from './types'
import type { StreamCallbacks } from './stream'

type JsonRpcMessage = {
  id?: number
  method?: string
  params?: Record<string, unknown>
  result?: unknown
  error?: { message?: string }
}

/** Non-secret renderer marker used only to show that the local bridge is ready. */
export const CODEX_BRIDGE_CONFIGURED = 'configured-by-local-codex'

/** Codex is the suite default; legacy providers require an explicit opt-in. */
export function usesCodexBridgeProvider(): boolean {
  const provider = process.env.GENOFFICE_AI_PROVIDER
  return !provider || provider === 'codex'
}

interface CodexResult {
  content: string
  toolCalls: AgentToolCall[]
}

const CODEX_RESULT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['content', 'toolCalls'],
  properties: {
    content: { type: 'string' },
    toolCalls: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['id', 'name', 'input'],
        properties: {
          id: { type: 'string' },
          name: { type: 'string' },
          // Codex structured outputs require every object to disallow unknown
          // keys. Tool inputs are arbitrary JSON-Schema objects, so carry them
          // as JSON text and parse them at the bridge boundary.
          input: { type: 'string' },
        },
      },
    },
  },
} as const

function conversationPrompt(system: string, messages: AgentMessage[], tools: AgentToolDef[]): string {
  const conversation = messages.map((message) => {
    if (message.role === 'tool') {
      return `Tool results:\n${message.results.map((result) => JSON.stringify(result)).join('\n')}`
    }
    return `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.text}`
  })
  return [
    'You are the AI assistant embedded in a document-editing application.',
    'Do not inspect files, execute commands, browse, or modify the local workspace.',
    'Follow the application system instructions and answer the conversation below.',
    'When an application action is needed, return it only in toolCalls using one of the supplied tools. Each toolCalls.input must be a JSON-encoded string matching that tool input schema.',
    `System instructions:\n${system}`,
    `Available application tools:\n${JSON.stringify(tools)}`,
    `Conversation:\n${conversation.join('\n\n')}`,
  ].join('\n\n')
}

function parseResult(value: string): CodexResult {
  try {
    const parsed = JSON.parse(value) as Partial<CodexResult>
    return {
      content: typeof parsed.content === 'string' ? parsed.content : '',
      toolCalls: Array.isArray(parsed.toolCalls)
        ? parsed.toolCalls.flatMap((call) => {
            if (!call || typeof call !== 'object') return []
            const raw = call as { id?: unknown; name?: unknown; input?: unknown }
            if (typeof raw.id !== 'string' || typeof raw.name !== 'string' || typeof raw.input !== 'string') {
              return []
            }
            try {
              const input = JSON.parse(raw.input) as unknown
              return input && typeof input === 'object' && !Array.isArray(input)
                ? [{ id: raw.id, name: raw.name, input: input as Record<string, unknown> }]
                : []
            } catch {
              return [{ id: raw.id, name: raw.name, input: {}, inputError: 'Codex returned invalid tool input JSON' }]
            }
          })
        : [],
    }
  } catch {
    // A future Codex version might decline structured output. Preserve its answer rather than losing it.
    return { content: value, toolCalls: [] }
  }
}

/**
 * Runs one GenOffice turn through the locally installed Codex App Server. The
 * server uses the user's existing `codex login` session, so no API key ever
 * enters this application. The protocol is deliberately process-local (stdio)
 * and the child is terminated on cancellation or after every ephemeral turn.
 */
export async function streamViaCodexAppServer(
  system: string,
  messages: AgentMessage[],
  tools: AgentToolDef[],
  cb: StreamCallbacks,
): Promise<void> {
  const command = process.env.GENOFFICE_CODEX_COMMAND?.trim() || 'codex'
  // Never expose the host project or a user's documents to a prompt-driven
  // Codex thread. The read-only sandbox gets an empty, per-turn directory.
  const cwd = mkdtempSync(join(tmpdir(), 'genoffice-codex-'))
  const child = spawn(command, ['app-server', '--stdio'], {
    stdio: 'pipe',
    windowsHide: true,
  })
  await runTurn(child, cwd, system, messages, tools, cb)
}

export async function chatViaCodexAppServer(system: string, user: string): Promise<AiChatResponse> {
  let content = ''
  try {
    await streamViaCodexAppServer(system, [{ role: 'user', text: user }], [], {
      signal: new AbortController().signal,
      onDelta: (delta) => {
        content += delta
      },
      onToolCall: () => undefined,
      onActivity: () => undefined,
    })
    return content ? { ok: true, content } : { ok: false, error: 'Codex returned an empty response' }
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) }
  }
}

function runTurn(
  child: ChildProcessWithoutNullStreams,
  cwd: string,
  system: string,
  messages: AgentMessage[],
  tools: AgentToolDef[],
  cb: StreamCallbacks,
): Promise<void> {
  return new Promise((resolve, reject) => {
    let nextId = 1
    let buffer = ''
    let completed = false
    let finalText = ''
    const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>()

    const finish = (error?: Error) => {
      if (completed) return
      completed = true
      child.kill()
      try {
        rmSync(cwd, { recursive: true, force: true })
      } catch {
        // The OS will clean the per-turn temp directory if removal is unavailable.
      }
      if (error) reject(error)
      else resolve()
    }
    const request = (method: string, params: Record<string, unknown>) =>
      new Promise<unknown>((resolveRequest, rejectRequest) => {
        const id = nextId++
        pending.set(id, { resolve: resolveRequest, reject: rejectRequest })
        child.stdin.write(`${JSON.stringify({ id, method, params })}\n`)
      })
    const notify = (method: string, params?: Record<string, unknown>) => {
      child.stdin.write(`${JSON.stringify(params ? { method, params } : { method })}\n`)
    }
    const completeFromText = (text: string) => {
      const result = parseResult(text)
      if (result.content) cb.onDelta(result.content)
      for (const call of result.toolCalls) cb.onToolCall(call)
      finish()
    }
    const handle = (message: JsonRpcMessage) => {
      if (typeof message.id === 'number') {
        const waiter = pending.get(message.id)
        if (!waiter) return
        pending.delete(message.id)
        if (message.error) waiter.reject(new Error(message.error.message || 'Codex App Server request failed'))
        else waiter.resolve(message.result)
        return
      }
      if (message.method === 'item/agentMessage/delta') {
        // Structured-output deltas are intentionally held until they form valid JSON.
        finalText += typeof message.params?.delta === 'string' ? message.params.delta : ''
        cb.onActivity?.()
        return
      }
      if (message.method === 'turn/completed') {
        const turn = message.params?.turn as { status?: string; error?: { message?: string }; items?: unknown[] } | undefined
        if (turn?.status === 'failed') return finish(new Error(turn.error?.message || 'Codex turn failed'))
        if (!finalText) {
          const item = turn?.items?.find(
            (entry): entry is { type: string; text?: string } =>
              typeof entry === 'object' && entry !== null && (entry as { type?: string }).type === 'agentMessage',
          )
          finalText = item?.text || ''
        }
        return completeFromText(finalText)
      }
      if (message.method === 'error') {
        return finish(new Error(String(message.params?.message || 'Codex App Server error')))
      }
    }

    child.stdout.on('data', (chunk: Buffer) => {
      buffer += chunk.toString('utf8')
      for (;;) {
        const newline = buffer.indexOf('\n')
        if (newline < 0) break
        const line = buffer.slice(0, newline).trim()
        buffer = buffer.slice(newline + 1)
        if (!line) continue
        try {
          handle(JSON.parse(line) as JsonRpcMessage)
        } catch {
          // App Server stdout is JSON-RPC. Ignore an unexpected non-protocol diagnostic line.
        }
      }
    })
    child.stderr.on('data', () => cb.onActivity?.())
    child.once('error', (error) => finish(new Error(`Unable to start Codex App Server: ${error.message}`)))
    child.once('exit', (code) => {
      if (!completed) finish(new Error(`Codex App Server exited before completing the request (code ${code ?? 'unknown'})`))
    })
    if (cb.signal.aborted) finish()
    else cb.signal.addEventListener('abort', () => finish(), { once: true })

    void (async () => {
      try {
        await request('initialize', {
          clientInfo: { name: 'genoffice', title: 'GenOffice', version: '0.1.0' },
          capabilities: null,
        })
        notify('initialized')
        const started = (await request('thread/start', {
          cwd,
          approvalPolicy: 'never',
          sandbox: 'read-only',
          ephemeral: true,
          developerInstructions:
            'This is an embedded document assistant. Do not use shell commands, files, browser, or external tools.',
        })) as { thread?: { id?: string } }
        const threadId = started.thread?.id
        if (!threadId) throw new Error('Codex App Server did not return a thread id')
        await request('turn/start', {
          threadId,
          input: [{ type: 'text', text: conversationPrompt(system, messages, tools), text_elements: [] }],
          outputSchema: CODEX_RESULT_SCHEMA,
        })
      } catch (error) {
        finish(error instanceof Error ? error : new Error(String(error)))
      }
    })()
  })
}
