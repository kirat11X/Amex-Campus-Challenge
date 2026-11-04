export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  /** assistant turns that asked for tools */
  tool_calls?: ToolCall[];
  /** tool turns: which tool produced `content` */
  tool_name?: string;
}

export interface ToolSpec {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: { type: 'object'; properties: Record<string, unknown>; required: string[] };
  };
}

export interface CompletionResult {
  content: string;
  toolCalls: ToolCall[];
}

export interface CompleteOptions {
  tools?: ToolSpec[];
  signal?: AbortSignal;
  /** Called with the visible assistant text so far (tool-call markup is hidden). */
  onToken?: (textSoFar: string) => void;
}

/** One chat backend: the in-browser WebLLM model, or the user's local Ollama. */
export interface Engine {
  readonly kind: 'demo' | 'webllm' | 'ollama';
  readonly label: string;
  /** Total context window in tokens (prompt + answer). */
  readonly contextTokens: number;
  complete(messages: ChatMessage[], opts?: CompleteOptions): Promise<CompletionResult>;
}
