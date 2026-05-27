# AgenticRAG Multi-Turn Chat Design

## Goal

Add interactive multi-turn question answering to the local AgenticRAG CLI.
The new mode should support natural follow-up questions such as "第二个模块详细说说" by preserving conversation context in memory during one CLI session.

The first version is intentionally local and session-scoped. It does not persist chat state after the process exits.

## User Experience

Add a new command:

```powershell
python main.py chat
```

The command starts an interactive REPL:

```text
AgenticRAG chat. Type /help for commands, /exit to quit.
> AgenticRAG 的核心模块有哪些？
...streamed answer...
> 第二个模块详细说说
...streamed follow-up answer...
```

Supported commands:

- `/help`: print available chat commands.
- `/reset`: clear the current in-memory conversation and start a fresh state.
- `/exit` or `/quit`: exit the chat session.
- Empty input: ignored.

The existing `ask` command remains a single-turn command. `chat` is the multi-turn entry point.

## Functional Requirements

Each chat answer must stream to the terminal as chunks arrive. Complex agentic answers should continue to print compact tool status lines such as `[tool] search` before or during the final streamed answer.

The chat session must keep one `ConversationState` across turns. The following state must remain available until `/reset` or process exit:

- conversation messages
- Reference ID mappings
- tool results
- token warning state
- turn index used to create Reference IDs

Reference IDs must remain stable across turns. The system may compress old tool result content, but it must not delete or renumber Reference IDs.

## Query Rewrite And Routing

Every user turn uses this pipeline:

1. Read raw user input.
2. Add the raw user turn to the active chat state.
3. Rewrite the current question into a self-contained query using the LLM and recent conversation context.
4. Run the simple/complex switcher on the rewritten query.
5. Answer with either the simple RAG path or the agentic loop path.

The rewrite step handles references such as "它", "第二个", "刚才那个", and similar follow-up phrasing.

The rewrite prompt should request JSON only:

```json
{"query": "self-contained question"}
```

If the rewrite response is malformed, empty, or not a JSON object with a non-empty `query`, the session falls back to the raw user input.

The rewritten query is an internal routing and retrieval aid. It is not printed by default.

## Simple Path

For simple turns:

1. Search with the rewritten query.
2. Build a streaming answer prompt that includes:
   - the raw user question
   - the rewritten self-contained question
   - retrieved context
3. Stream chunks to stdout immediately.
4. Record the final assistant answer in `ConversationState`.

The simple answer must still cite available Reference IDs.

## Complex Path

For complex turns:

1. Use the shared active `ConversationState`.
2. Add enough context for the agentic loop to understand the raw question and rewritten query.
3. Run the existing `run_agentic_loop`.
4. Stream output chunks to stdout immediately.
5. Record the final assistant answer in `ConversationState`.

The existing summarize behavior remains responsible for compressing old tool results when token usage grows. It must preserve Reference IDs.

## Proposed Components

Add `agenticrag/chat.py` with:

- `ChatSession`: owns long-lived state and handles one user turn.
- `build_rewrite_messages(...)`: creates the LLM rewrite prompt from recent history and reference summaries.
- `rewrite_query(...)`: calls the LLM, parses JSON, and falls back safely.
- `run_chat()`: CLI REPL entry point.

Keep `agenticrag/loop.py` focused on one-turn agentic execution. If needed, extract a shared retrieval-tool dispatcher so both `ask` and `chat` use the same `search` / `find` / `open` / `summarize` validation behavior.

Update `main.py` to add the `chat` parser subcommand and dispatch to `agenticrag.chat.run_chat`.

## Error Handling

Single-turn failures inside the REPL should print a concise `[error] ...` message and return to the prompt. They should not terminate the chat session unless the failure is a keyboard interrupt or EOF.

Rewrite failures fall back to the raw user input.

Switcher failures in chat should fall back to the complex route because the agentic loop is better suited to ambiguous follow-ups.

`/reset` creates a fresh `ConversationState` while reusing initialized clients and retriever objects.

## Testing

Add focused tests for:

- parser support for `chat`
- `main(["chat"])` dispatching to `run_chat`
- rewrite JSON parsing
- rewrite fallback behavior
- `ChatSession` passing recent history into the rewrite step
- Reference IDs surviving across turns
- `/reset` clearing chat state
- simple route streaming chunks and recording the assistant answer
- complex route streaming chunks, showing tool statuses, and recording the assistant answer

All tests should avoid real network calls by using fakes and monkeypatches.

## Out Of Scope

The first version does not include:

- persistent session files
- `ask --session`
- `/refs`, `/tokens`, or `/history`
- web UI or API server
- displaying rewritten queries by default

These can be added later without changing the core `ChatSession` abstraction.
