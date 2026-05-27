# Microsoft AgenticRAG Reproduction

Python MVP reproduction of AgenticRAG for local enterprise-style documents.

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

Final answers stream to the terminal. Intermediate AgenticRAG tool calls print compact status lines such as `[tool] search` before the final streamed answer.
