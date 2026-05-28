SWITCHER_PROMPT = """Classify the user query for a RAG system.
Return only JSON: {"route": "simple"} or {"route": "complex"}.
Use simple for single-intent factual questions.
Use complex for multi-step, multi-document, comparison, long-document, or ambiguous questions.
"""

QUERY_REWRITE_PROMPT = """Rewrite the current user question into a self-contained question for a multi-turn AgenticRAG session.
Use the recent conversation and available Reference IDs only for pronouns, ellipsis, explicit follow-ups, or Reference IDs.
If the current question is self-contained or introduces an independent new topic, return it exactly unchanged.
Do not blend the current question with prior conversation or change its topic just because history exists.
Do not answer the question.
Return only strict JSON in this exact shape: {"query": "..."}.
"""

SYSTEM_PROMPT = """# Overall Instructions
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

请默认使用中文回答，并使用可用的 Reference ID 标注证据。
"""

SIMPLE_RAG_PROMPT = """请基于给定检索片段回答用户问题。必须引用片段中的 Reference ID。如果证据不足，请明确说明。
"""

CHAT_SIMPLE_RAG_PROMPT = """你将看到原始用户问题、改写后的自包含问题和检索结果。
请回答原始用户问题；必要时使用改写后的问题来理解上下文、指代、省略或追问。
回答必须基于检索结果，并引用使用到的 Reference ID。
如果证据不足，请清楚说明证据不足，不要编造答案。"""

FORCE_FINAL_ANSWER_PROMPT = """FORCEFINALANSWER:
You have reached the maximum number of tool calls. Produce the best final answer using only the collected context.
Include citations with Reference IDs whenever possible.
"""
