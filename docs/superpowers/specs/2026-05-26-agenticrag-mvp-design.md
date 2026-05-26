# AgenticRAG MVP Design

## Context

This spec defines the first implementation milestone for reproducing the paper "AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases" in this workspace. The implementation follows `PRD.md` and the decisions already recorded in `DESIGN.md`, with the scope narrowed to a CLI-runnable local MVP over the current `docs/` corpus.

The MVP does not attempt to reproduce BRIGHT, WixQA, or FinanceBench scores. Those benchmark workflows are deferred until the core AgenticRAG loop is working locally.

## Goals

- Build a Python CLI that can index local documents and answer questions over them.
- Preserve the paper's core mechanisms: query routing, agentic retrieval loop, `search` / `find` / `open` / `summarize` tools, Reference IDs, token auditing, and forced completion.
- Use streaming output for final answers in both traditional RAG and AgenticRAG paths.
- Keep implementation modular enough for later replacement of Chroma, model providers, or evaluation harnesses.

## Non-Goals

- No web UI.
- No REST API in the first milestone.
- No full public benchmark reproduction.
- No custom embedding model training, graph construction, model fine-tuning, or complex preprocessing beyond local parsing and chunking.

## Architecture

The MVP exposes two command-line entry points:

- `python main.py index`
- `python main.py ask "..."`

The project will use the module layout already selected in `DESIGN.md`:

- `main.py`: CLI entry point.
- `config.py`: environment and runtime configuration.
- `ingest.py`: document scanning, parsing, chunking, and index writing.
- `retriever.py`: Chroma-backed vector retrieval abstraction.
- `switcher.py`: LLM-based simple-versus-complex routing.
- `loop.py`: Agentic Loop orchestration.
- `state.py`: conversation state, token audit, Reference ID mapping, and context compression state.
- `prompts.py`: system prompts, tool instructions, and forced-completion prompt.
- `tools/`: implementations and schemas for `search`, `find`, `open`, and `summarize`.

## Configuration

Runtime configuration comes from `.env` via `python-dotenv`.

Required settings:

- `DEEPSEEK_API_KEY`
- `SILICONFLOW_API_KEY`

Optional settings with defaults:

- `DEEPSEEK_BASE_URL=https://api.deepseek.com`
- `DEEPSEEK_MODEL=<DeepSeek V4 Pro compatible model name>`
- `SILICONFLOW_BASE_URL=https://api.siliconflow.cn`
- `SILICONFLOW_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B`
- `EMBEDDING_DIMS=1536`
- `DOCS_DIR=docs`
- `CHROMA_DIR=.chroma`
- `MAX_CALLS=15`
- `TOKEN_THRESHOLD=128000`
- `TOKEN_WARNING_RATIO=0.9`

The `.gitignore` should exclude `.env`, `.chroma/`, caches, and Python temporary files. If required API keys are missing, the CLI should fail early with a clear message.

## Indexing Flow

`python main.py index` scans `DOCS_DIR` for supported source files. MVP support includes Markdown and PDF.

Parsing rules:

- Markdown is parsed as text and chunked primarily by `##` headings.
- PDF is parsed with `pdfplumber` and chunked by paragraph-like blocks.
- Fallback chunking splits by sentence boundaries.
- Chunking uses `max_chunk_size=1000`, `overlap=100`, and `min_chunk_size=50` characters.

Each parsed chunk is represented as a `DocumentChunk` with:

- `doc_id`
- `path`
- `title`
- `filetype`
- `chunk_index`
- `line_start`
- `line_end`
- `content`

Chroma stores chunk content plus metadata. The parser also keeps or can reload normalized line-based source text so `find` and `open` can operate against document text rather than only vector chunks.

## Ask Flow

`python main.py ask "..."` loads configuration, opens the Chroma index, creates a `ConversationState`, and runs the query through the switcher.

For `simple` queries:

1. Retrieve relevant chunks.
2. Generate a final answer with DeepSeek using streaming output.
3. Include citations from retrieved chunks.

For `complex` queries:

1. Enter the Agentic Loop.
2. Let the model call OpenAI-compatible function tools.
3. Execute tool calls synchronously.
4. Add tool results to `ConversationState`.
5. Continue until the model produces a final answer or `MAX_CALLS` is reached.

The final answer in both paths must stream to the terminal chunk by chunk. Intermediate tool calls should not dump full tool payloads to stdout. They should emit compact status lines such as:

```text
[tool] search: 3 queries, 8 results
[tool] open: turn2search0 lines 120-360
```

## Retrieval Tools

### search

`search(queries)` accepts up to 5 query rewrites. Each query retrieves candidates from Chroma. Results are deduplicated by file and content, assigned Reference IDs in the form `turn{m}search{n}`, and stored in `ConversationState`.

Each result includes:

- Reference ID
- snippet, about 200 characters
- title
- filename
- filetype
- chunk location metadata

The MVP should return at most 10 deduplicated results across all query rewrites.

### find

`find(reference_id, patterns)` searches within the document mapped by the Reference ID.

MVP behavior:

- Case-insensitive substring matching.
- Up to 2 snippets per pattern.
- Around 50 characters of context before and after each match.
- Deduplicate repeated snippets.
- Truncate total output to a practical limit near the PRD's 11k-token target.

### open

`open(reference_id, line_number=0)` returns a line-numbered window over the mapped source document.

MVP behavior:

- Return up to 1,800 lines.
- Include a response header such as `Viewing lines [0-1799] of 3000 lines`.
- Start from `line_number`, clamped to document bounds.

### summarize

`summarize(candidate_reference_ids)` compresses conversation history while preserving the ability to continue investigation.

MVP behavior:

- Replace large tool result payloads unrelated to retained Reference IDs with compact placeholders.
- Preserve user question, model reasoning summaries, retained evidence snippets, and the full Reference ID mapping.
- Do not delete or renumber Reference IDs.

The first implementation can use deterministic compression. LLM-generated summaries can be added later.

## Conversation State

`ConversationState` owns:

- Messages sent to and returned from the model.
- Tool call records.
- Tool result records.
- Reference ID to document/chunk mapping.
- Token estimates.
- Summarization/compression markers.

`total_tokens()` uses `tiktoken` with `cl100k_base` for approximate counting.

When token use reaches `TOKEN_THRESHOLD * TOKEN_WARNING_RATIO`, the system appends an internal warning message. When it reaches `TOKEN_THRESHOLD`, it forces context compression through `summarize`.

## Agentic Loop

The loop runs for at most `MAX_CALLS`, defaulting to 15.

For each turn:

1. Check token budget and manage context if needed.
2. Call DeepSeek with the system prompt, conversation messages, and tool schemas.
3. If tool calls are returned, execute them and append results.
4. If final text is returned, stream it to the terminal and finish.

If the loop reaches `MAX_CALLS` without a final answer, append the `FORCEFINALANSWER` instruction, remove tool schemas from the final model call, and stream the forced final answer.

## Prompting

The system prompt follows the PRD's tool-use instructions:

- Search before answering when uncertain.
- Use `find` and `open` when snippets are insufficient.
- Reuse previous results instead of repeating searches.
- Cite information from tool outputs.

The prompt also instructs the model to answer in Chinese by default for this corpus and to cite evidence using available Reference IDs.

## Error Handling

- Missing API keys fail early with actionable messages.
- Missing Chroma index tells the user to run `python main.py index`.
- Unsupported document files are skipped with a warning.
- PDF parse failures are reported per file without aborting the whole indexing run.
- Tool calls with unknown Reference IDs return a structured tool error that the model can recover from.
- Empty retrieval results are passed to the model as explicit no-result tool outputs.

## Testing And Acceptance

The MVP is accepted when:

- `python main.py index` parses the current `docs/` Markdown/PDF files and writes a Chroma index.
- `python main.py ask "..."` supports streaming final answers.
- Simple queries can route to traditional RAG and produce cited answers.
- Complex queries can route to AgenticRAG and execute at least `search`, `find`, and `open`.
- `summarize` can compact unrelated tool results while preserving selected Reference IDs.
- Reference IDs follow `turn{m}search{n}` and remain stable after summarization.
- `MAX_CALLS` triggers forced completion.
- Unit or focused integration tests cover chunking, Reference ID mapping, tool behavior, switcher parsing, and forced completion.

## Deferred Work

- Benchmark harnesses for BRIGHT, WixQA, and FinanceBench.
- REST API or web UI.
- Semantic `find`.
- LLM-generated context summaries.
- Provider-independent streaming abstractions beyond current DeepSeek and SiliconFlow usage.
