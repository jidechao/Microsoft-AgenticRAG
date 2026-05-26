# AgenticRAG 设计决策文档

> 基于 PRD.md 的论文复现设计决策，通过 grill-me 流程确认。

## 核心技术栈

| # | 决策项 | 结论 |
|---|--------|------|
| 1 | LLM | DeepSeek V4 Pro，官方 API，`https://api.deepseek.com` |
| 2 | 搜索后端 | Chroma + 文件扫描，抽象接口可替换 |
| 3 | 语言/框架 | Python 纯脚本，先跑通核心逻辑 |
| 4 | 项目结构 | main.py / switcher.py / loop.py / tools/ / state.py / prompts.py / config.py |
| 5 | 语料 | `docs/` 下 6 篇文档（2 PDF + 4 MD），中文为主 |
| 6 | Embedding | SiliconFlow Qwen3-Embedding-4B，dims=1536，base_url: https://api.siliconflow.cn |

## 模块设计

| # | 决策项 | 结论 |
|---|--------|------|
| 7 | Switcher | LLM 分类，每查询多一次轻量调用判断简单/复杂 |
| 8 | 工具 Schema | 4 个 function calling（search/find/open/summarize），OpenAI 兼容格式 |
| 9 | PDF 解析 | pdfplumber |
| 10 | Token 计算 | tiktoken cl100k_base 近似估算 |
| 11 | 系统提示词 | 英文工具指令 + 中文回答指令 |
| 12 | 文档切片 | 混合策略：MD 按 ## 标题切、PDF 按段落切，兜底按句子拆分。max_chunk_size=1000 字符，overlap=100，min=50 |
| 13 | summarize 工具 | 只压缩 tool_result，替换为占位描述，保留推理链 |
| 14 | API Key | .env 文件 + python-dotenv，.gitignore 排除 |
| 15 | 强制补全 | 15 轮上限，追加 FORCEFINALANSWER 消息 + 移除 tools 参数 |
| 16 | search 返回格式 | 200 字符 snippet，按文件去重，每查询最多 6 结果，5 queries 并行去重后最多 10 结果 |
| 17 | find 匹配 | 纯子串匹配，每关键词 2 个片段（前后各 50 字符上下文），约 16,500 字符上限 |

## 微软预生产部署的四条设计经验（论文核心价值）

1. **搜索结果展示文档元数据**：标题、文件名、文件类型帮助模型区分语义相似的 snippet，避免重复搜索
2. **行号预览**：让模型锚定内容位置，在后续 open 调用中精准跳转
3. **摘要后保留引用 ID**：压缩上下文后，模型仍然可以继续深入调查之前发现的高价值候选
4. **混合路由**：简单查询走传统 RAG（快、便宜），复杂查询走 Agent RAG（慢、准）。这是生产环境的关键取舍

## 论文关键参数约束

- Agentic Loop 最大 15 轮（max_calls=15）
- open 工具最多返回 1800 行带行号文本
- open 必须包含 Response Header（如 "Viewing lines [0-1799] of 3000 lines"）
- Reference ID 格式：`turn{m}search{n}`
- Token 阈值：128K 的 90%（约 115K）发警告，达 128K 强制 summarize
