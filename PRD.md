这是一份针对论文 **《AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases》** 进行复现的详细需求规格说明书（PRD）。本说明书力求完整还原论文中的设计细节，不添加夸大词汇，旨在提供一份客观、可执行的技术实现指南。

---

# AgenticRAG 系统复现需求规格说明书

## 1. 项目概述

AgenticRAG 是一个轻量级的、推理时（Inference-time）的 Agent 框架，直接运行在现有的企业级搜索基础设施（如 Azure AI Search）之上。该系统不依赖于模型微调、自定义嵌入模型、图谱构建或复杂的语料预处理，而是通过赋予推理大语言模型（Reasoning LLM）主动进行多轮检索、导航和分析的能力，来解决复杂、跨文档、长文本的问答需求。

---

## 2. 系统架构设计

系统由四个核心模块组成：

1. **智能路由分流器 (Query Switcher)**：对输入查询进行预分类，简单查询直接走传统 RAG，复杂或多意图查询路由至 AgenticRAG。
2. **Agentic 循环引擎 (Agentic Loop)**：负责编排 LLM 与工具之间的多轮交互。
3. **检索工具集 (Retrieval Tools)**：提供层次化的文档探索能力，包括 `search`、`find`、`open` 和 `summarize` 四个工具。
4. **对话状态管理器 (Conversation State)**：维护消息历史、进行 Token 审计，以及全局管理 Reference ID（引用标识）。

```
             ┌────────────────────────┐
             │       用户查询         │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │  智能路由分流器(Switcher)│
             └─────┬────────────┬─────┘
                   │ 简单查询   │ 复杂查询
                   ▼            ▼
             Traditional RAG  AgenticRAG Loop
```

---

## 3. 详细功能需求与实现规格

### 3.1 智能路由分流器 (Query Switcher)

- **功能需求**：为权衡用户体验、Token 成本和响应延迟，系统必须在入口处部署一个 Switcher [7]。
- **路由逻辑**：
  - **简单查询**：单意图、事实性问题，直接路由至传统一阶段 RAG（Retrieve-then-Generate），快速返回 [7]。
  - **复杂查询**：多意图、跨文档比较、长文档深度阅读或情况复杂的场景，路由至 AgenticRAG [7]。

### 3.2 Agentic 循环引擎 (Agentic Loop)

- **迭代限制**：默认最大迭代轮数（`max_calls`）为 **15 轮** [3]。
- **强制终止**：如果达到最大迭代次数（第 15 轮）仍未输出最终答案，系统将强制发起一个强制补全请求（Forced Completion），要求模型基于当前已收集的所有信息和上下文，强制生成最终答案并附带引用（`FORCEFINALANSWER`） [3]。

### 3.3 对话状态管理器与 Reference ID 映射

- **引用格式**：在执行 `search` 工具时，返回给 Agent 的每个结果必须分配一个全局递增且唯一的引用标识（Reference ID），格式为：`turn{m}search{n}`，其中 $m$ 代表第几轮交互，$n$ 代表该轮返回的第几个结果 [3]。
- **映射维持**：Conversation State 需维护这些 Reference ID 与原始文档路径、行号及内容的映射关系，以便后续的 `find` 和 `open` 工具精准定位 [3]。

---

## 4. 检索工具集规格 (Retrieval Tools Spec)

系统必须严格实现以下四个工具，接口和行为定义如下 [5]：

| 工具名称            | 定义 (Definition)              | 输入参数 (Input)                                                           | 输出格式与行为 (Output & Behavior)                                                                                                    |
|:--------------- |:---------------------------- |:---------------------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------ |
| **`search`**    | 在整个企业语料库中发现相关文档。             | `queries` (最多 5 个并行的查询重构词) [3, 5]                                      | 每个查询重构词返回最多 10 个去重后的结果 [3]。格式：片段（Snippet）、唯一 Reference ID、标题（Title）、文件名（Filename）、文件类型（Filetype）等元数据 [3, 5]。                   |
| **`find`**      | 在单个指定文档中定位特定信息（词组/语义匹配）。     | `reference_id` (格式如 `turn0search1`), `patterns` (关键词列表，支持正则或子串) [3, 5] | 支持不区分大小写的子串匹配（Lexical matching）或可选的语义查找模式 [3]。每个模式最多返回 2 个匹配片段，根据内容去重并截断，限制在约 **11k Tokens** [3, 5]。                           |
| **`open`**      | 检索单个指定文档在固定窗口大小内的完整正文。       | `reference_id`, `line_number` (可选，默认为 0 或指定的起始行) [4, 5]                | 返回最多 **1,800 行** 带行号的文本正文 [4, 5]。必须包含一个 Response Header（响应头），说明当前的浏览范围和文档总长度（例如：`"Viewing lines [0–1799] of 3000 lines"`） [4]。 |
| **`summarize`** | 上下文管理工具，用于压缩长推理链中的 Token 占用。 | `candidate_reference_ids` (需要保留的文档 Reference ID 列表) [4, 7]             | 记录当前推理进展，删除与被保留 ID 无关的工具返回内容，释放 Token 空间，同时保留已引用的证据 [4, 7]。                                                                    |

---

## 5. 上下文管理与自动摘要机制

由于 `open` 和 `find` 每次调用可能会加载多达约 11k Tokens 的长文本，上下文窗口（以 128K 窗口为例）极易被耗尽 [4]。必须实现以下自动管理逻辑 [4]：

1. **Token 审计监视**：系统在 Agent 交互过程中实时计算当前对话上下文（Conversation State）的 Token 数量 [4]。
2. **90% 阈值警报**：当 Token 使用量达到设定阈值（如 128K）的 **90%**（即约为 115K Tokens）时，系统向 Agent 发送一条内部警告信息（Internal Warning） [4]。
3. **强制触发限制**：一旦达到 128K 限制（阈值），系统将**强制**调用 `summarize` 工具 [4]。
4. **清理逻辑**：
   - 模型在调用 `summarize` 时需指定保留的 `candidate_reference_ids` [4, 7]。
   - 系统接收到参数后，扫描整条对话历史，**移除**与这些被保留 ID 无关的工具调用历史及详细内容（例如已丢弃文档的 1,800 行 Open 文本），仅保留核心的推理结论和已引用的证据片段，从而大幅回收 Token 空间 [4]。

---

## 6. 系统提示词与工具使用指令 (System Prompt)

根据论文附录 A.2，复现时大语言模型（LLM）的系统提示词必须包含以下具体的工具使用指南 [10]：

```markdown
# Overall Instructions
- Search before answering when uncertain.
- Progressively explore using 'find' or 'open' when snippets are insufficient.
- Reuse previous results rather than performing search again.
- Cite every time when information is used from tool outputs.

# When to Use 'search'
- Use 'search' as the primary tool across the enterprise corpus.
- It should be your first choice for any work-related query.
- Use it when users reference current/changing information, enterprise-specific terms, or acronyms.
- Use it to verify details rather than making assumptions.

# When to Use 'find'
- Use 'find' for in-document pattern search for relevant files from search results.
- Use it when search results snippets do not give enough details.
- Use it to get a focused view of a result in relation to certain terms.

# When to Use 'open'
- Use 'open' for windowed full content retrieval for relevant files from search results.
- Use it when search results snippets are insufficient.
- Use it to pull in more content from the most promising results.
- You can open multiple search results.
- Use the option to choose a line number close to the relevant content based on line-numbered document previews.
```

---

## 7. 核心算法流程 (Algorithm 1)

复现的核心循环逻辑应严格依照论文中定义的伪代码实现 [10]：

```python
def agentic_loop(user_query, max_calls=15, token_threshold=128000):
    # 1. 初始化对话状态
    conversation = ConversationState()
    conversation.add(role="user", content=user_query)

    for i in range(1, max_calls + 1):
        # 2. 检查 Token 是否超限并执行上下文管理
        if conversation.total_tokens() >= token_threshold:
            # 强制调用上下文缩减与摘要工具
            conversation = manage_context(conversation) 

        # 3. 运行模型预测（传入系统提示词、对话历史以及工具规范描述）
        response = llm.predict(conversation, tools=TOOL_SCHEMAS)

        # 4. 判断模型是返回工具调用还是最终答案
        if response.has_tool_calls:
            for tool_call in response.tool_calls:
                # 执行工具，根据结果类型（search/find/open/summarize）进行处理
                result = execute_tool(tool_call)
                # 将工具调用及结果追加至对话历史中
                conversation.add(role="tool", tool_call_id=tool_call.id, content=result)
        else:
            # 返回最终格式化的带有引用的答案
            return format_answer(response.text)

    # 5. 若超出最大轮数，强制使用现有可用信息生成答案
    return force_final_answer(conversation)
```

---

## 8. 评测与验收标准 (Evaluation Metrics)

为确保复现版本与论文性能一致，可选用以下三个公开基准数据集进行回归验证：

1. **BRIGHT (Long-context 检索基准)** [4, 10]:
   
   - **评测指标**：Recall@1 [4, 10]。
   - **基准目标**：Qwen (最佳 Embedding) 约为 27.8%，AgenticRAG (基于 Claude Sonnet 4.5) 平均 Recall@1 应接近 **49.6%** [5]。

2. **WixQA (企业支持与多文档故障排查基准)** [4, 10]:
   
   - **评测指标**：基于 LLM 评判的 Factuality [4]。
   - **基准目标**：在 Expert Written 集合上，Factuality 指标达到 **0.96** 左右 [5]。

3. **FinanceBench (复杂财务分析基准)** [4]:
   
   - **评测指标**：Answer Correctness (答案准确率) [4]。
   - **基准目标**：GPT-5-mini 实验准确率应达到约 **92.00%** (接近 94.00% 的 Oracle True Evidence 水平) [6]。
