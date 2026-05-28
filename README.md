# Microsoft AgenticRAG Reproduction

Python MVP reproduction of AgenticRAG for local enterprise-style documents.

## Documentation

- Technical architecture and implementation guide: [technical-architecture-and-implementation.md](./technical-architecture-and-implementation.md)

## Setup

Run these commands in Windows PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
copy .env.example .env
```

Fill `DEEPSEEK_API_KEY` and `SILICONFLOW_API_KEY` in `.env`.

## Usage

Index the local document corpus:

```powershell
python main.py index
```

Ask a question:

```powershell
python main.py ask "AgenticRAG 的核心模块有哪些？"
```

## Chat

```powershell
python main.py chat
```

Use `chat` for multi-turn follow-up questions in one terminal session.
The session keeps conversation context and Reference IDs until you exit or run `/reset`.

Available chat commands:

- `/help`: show commands
- `/reset`: clear the current in-memory session
- `/exit` or `/quit`: exit chat

Answers stream to the terminal. Complex AgenticRAG turns may print compact `[tool] ...` status lines before the final answer.

Final answers stream to the terminal. Intermediate AgenticRAG tool calls print compact status lines such as `[tool] search` before the final streamed answer.
