+++
date = '2026-06-22T10:00:00-07:00'
draft = false
title = 'Agentic AI Search 原理和代码: 极简演示'
+++

传统的基于RAG的 AI Search里，LLM 只负责把RAG 搜索的结果整合成最终答案， LLM 不参与"该怎么搜"。

**Agentic AI Search** 把 LLM 变成主动的 **Orchestrator**（指挥者）。它自己决定：

- 搜**什么** query
- 要不要**再搜**一次（换个 query）
- 什么时候信息**够了**，可以直接作答

举两个直观例子：

- 问 "1 + 2" —— LLM 直接回答 "3"，根本不触发搜索。
- 问 "Claude Code 是怎么实现的？" —— LLM 自己发起了两次不同 query 的搜索，攒够信息后才作答。

这就是从 RAG 到 **Agentic AI Search** 的转变，也是现代 AI 聊天与搜索引擎背后的核心范式。下面用 ~120 行 Python 把它从零讲清楚。

## 一、原理：Agent Loop + Tool Use

整个机制就是一个循环：给 LLM 一个 `web_search` 工具，让它在循环里自己决定**搜**还是**答**。

```mermaid
flowchart TD
    Q(["用户 Query"]) --> L["调用 LLM<br/>（带 web_search 工具）"]
    L --> D{"LLM 返回的是<br/>Tool Use 还是答案？"}
    D -- "Tool Use: web_search(query)" --> S["执行搜索<br/>结果去重后加入 Context"]
    S --> L
    D -- "纯文本答案" --> O["格式化输出<br/>Answer + 引用 + 结果列表"]
    O --> R(["返回最终答案"])
```

几个关键点：

1. **工具**：LLM 拿到一个 `web_search` 工具，用 JSON schema 描述。
2. **每轮二选一**：要么调工具去搜（结果加进 context，继续循环），要么返回纯文本答案（退出循环）。
3. **去重**：跨多次搜索的结果按 URL 去重，每个 URL 只出现一次，并分配一个全局编号 `[n]`。
4. **`MAX_LOOP_TIMES`**：循环上限（默认 3），防止 Agent 反复搜索却永远不肯作答。

## 二、完整代码

整个程序就一个 Python 文件，约 120 行：

```python
import os, json, sys
import litellm
from tavily import TavilyClient

# --- API Keys ----------------------------------------------------------------
TAVILY_API_KEY = "tvly-..."  # your Tavily API key
OPENAI_API_KEY = "sk-..."    # your OpenAI API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Configuration -----------------------------------------------------------
MODEL          = "gpt-4o-mini"  # change to e.g. "gpt-4o", "claude-3-5-haiku-20241022"
TOP_K          = 10             # max web results per search
MAX_LOOP_TIMES = 3              # max LLM calls per request (including final answer)

litellm.set_verbose = False


# --- Step 1: Web Search Tool -------------------------------------------------

def search_web(query, k=TOP_K):
    """Return top-k Tavily results for query."""
    response = TavilyClient(api_key=TAVILY_API_KEY).search(query, max_results=k)
    return response["results"]  # each: {title, url, content, score}


WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information on a topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    }
}


# --- Step 2: Agent Loop ------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful research assistant with access to a web_search tool. "
    "Use it to find information needed to answer the user's query; you may search multiple times with different queries. "
    "Each search result is labeled [n] with its URL. "
    "Cite supporting text in your answer as [n](URL) using those numbers. "
    "When you have enough information, write a clear, comprehensive markdown answer."
)


def run_agent(query, model=MODEL, max_loops=MAX_LOOP_TIMES):
    """Run the agent loop; return (answer_text, all_results, model_name)."""
    seen_urls = {}   # url -> result dict (augmented with "num" key)
    counter   = 0    # global discovery counter across all searches
    messages  = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]

    last_content = ""
    model_name   = model
    for loop_num in range(max_loops):
        print(f"\n[Loop {loop_num + 1}] Calling LLM...", file=sys.stderr)
        response = litellm.completion(
            model=model, messages=messages,
            tools=[WEB_SEARCH_TOOL], tool_choice="auto"
        )
        msg        = response.choices[0].message
        model_name = response.model  # actual model name returned by the API
        last_content = msg.content or ""

        if not msg.tool_calls:  # LLM produced a plain answer — exit loop
            print(f"[Loop {loop_num + 1}] LLM returned answer. Exiting loop.", file=sys.stderr)
            return last_content, list(seen_urls.values()), model_name

        # --- Execute tool calls and add results to context ---
        messages.append(msg)
        for tc in msg.tool_calls:
            search_query = json.loads(tc.function.arguments)["query"]
            print(f"[Loop {loop_num + 1}] Tool call: web_search('{search_query}')", file=sys.stderr)
            results = search_web(search_query)
            print(f"[Loop {loop_num + 1}] Got {len(results)} results", file=sys.stderr)

            tool_text = ""
            for r in results:
                if r["url"] not in seen_urls:  # deduplicate; first occurrence wins
                    counter += 1
                    r["num"] = counter
                    seen_urls[r["url"]] = r
                num = seen_urls[r["url"]]["num"]
                tool_text += f"[{num}] {r['title']}\nURL: {r['url']}\n{r['content']}\n\n"

            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "name": "web_search", "content": tool_text
            })

    # Max loops reached — return whatever the LLM last said
    print(f"[Loop {max_loops}] Max loops reached. Returning last LLM response.", file=sys.stderr)
    return last_content, list(seen_urls.values()), model_name


# --- Step 3: Format Output ---------------------------------------------------

def format_output(answer_text, all_results, model_name):
    """Return final markdown string with Answer and Web Search Results sections."""
    # Display results in ascending [n] (discovery) order
    ordered = sorted(all_results, key=lambda r: r["num"])

    results_md = "## Web Search Results\n\n"
    for r in ordered:
        results_md += f"- [{r['num']}] [{r['title']}]({r['url']})\n"

    return f"## Answer *(model: {model_name})*\n\n{answer_text}\n\n---\n\n{results_md}"


# --- Entry Point -------------------------------------------------------------

if __name__ == "__main__":
    query = "How was Claude Code implemented?"
    answer_text, all_results, model_name = run_agent(query)
    print(format_output(answer_text, all_results, model_name))
```

## 三、代码讲解

核心是中间的 **Agent Loop**（`run_agent()`）：

1. **定义 Web Search 为工具。** 调用 LLM 回答用户 Query 时，把工具传给它：

    ```python
    response = litellm.completion(
        model=model, messages=messages,
        tools=[WEB_SEARCH_TOOL], tool_choice="auto"
    )
    ```

2. **看LLM Call返回结果**
    - 返回 `tool_calls` —— LLM 决定要搜，那就执行 Web Search，结果去重存入 `seen_urls` 并追加回 `messages`，进入循环下一轮。
    - 不返回 `tool_calls` —— LLM 已拿到足够信息、直接作答，结束 Agentic Loop。

## 四、运行示例

以 `query = "Claude Code 是如何实现的？"` 为例。能看到 LLM **自己决定**搜了两次——先用中文 query，再自动换成英文 query 补充信息，第三轮才作答：

```
[Loop 1] Calling LLM...
[Loop 1] Tool call: web_search('Claude Code 实现 研究')
[Loop 1] Got 10 results

[Loop 2] Calling LLM...
[Loop 2] Tool call: web_search('Claude Code implementation details')
[Loop 2] Got 10 results

[Loop 3] Calling LLM...
[Loop 3] LLM returned answer. Exiting loop.
```

最终输出分两段——带 `[n](URL)` 引用的 **Answer**，和按发现顺序排列的 **Web Search Results**（下面是节选）：

> **Answer**（model: gpt-4o-mini）
>
> Claude Code 是由 Anthropic 开发的一款先进编程助手，旨在通过综合的 AI 技术与工具集成来提升软件开发效率……
>
> **1. 系统架构和设计**：用户接口层、代理循环、权限系统、工具管理、状态与持久性。根据源码分析，超过 98% 的代码用于基础设施管理 [1][2]。
>
> **2. 功能特性**：代码编辑和运行、上下文理解、任务驱动、用户和项目记忆系统 [1]。
>
> **3. 实现技术**：用 TypeScript 构建，代码库接近 51 万行 [2]。
>
> **Web Search Results**
> - [1] 深度解析 Claude Code 51 万行源码背后的设计实现 - 张永清 - 博客园
> - [2] ThreeFish-AI/analysis_claude_code: Claude Code 逆向工程研究仓库

一个有意思的细节：LLM 第一轮用中文搜，发现还不够，第二轮自己换成英文 query 再补一次——这正是 Agentic Search 里 LLM 主动驱动搜索的体现。

## 小结

- **Agentic 架构让 LLM 主动驱动搜索**：自己改写 query、评估结果、攒够 context 再作答——和人类查资料的"先宽后窄"很像。
- **Tool Use 是干净的抽象**：把 `web_search` 定义成一个工具，让 LLM 决定调用。
- 这套 **Agentic Loop + Tool Use** 正是 Perplexity、ChatGPT、Claude 等现代 AI 搜索的底层范式，也是现代 AI Agent 的核心。

---

#ai #agenticaisearch #aiagents #llm #rag #tooluse #aisearch #aicoding #人工智能 #ai学习 #大模型

If you like this post, consider star [the repo](https://github.com/sleepingbear-ai/sleepingbear-ai.github.io).
