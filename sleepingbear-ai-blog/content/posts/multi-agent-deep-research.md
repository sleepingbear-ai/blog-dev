+++
date = '2026-07-20T10:00:00-07:00'
draft = false
title = 'Multi-Agent Deep Research: 原理和代码'
+++

之前写过一篇 [Anthropic Deep Research: Multi-Agent 架构](/posts/anthropic-multi-agent-research/)，讲 Anthropic 是怎么用 Multi-Agent 做 Deep Research 的。光看架构不过瘾，我按照那套架构写了一份开源实现：**139 行 Python**，跑起来就是一个能并行搜索、带引用的 Deep Research。

代码在这里：[github.com/tiejun-ai/deep_research](https://github.com/tiejun-ai/deep_research)

## 一、原理：Orchestrator-Worker

Anthropic 的核心思路是 **Orchestrator-Worker**（指挥者—工人）：

- **Lead Agent（指挥者）** —— 拆解问题，把一个大 query 切成几个互不重叠的子任务。
- **Subagents（工人）** —— 每人领一个子任务，**并行**去搜，各自有独立的 context window。
- **CitationAgent** —— 最后统一给报告里的每个论断挂上来源链接。
- **Synthesis** —— Lead Agent 把各路 findings 汇总成一篇连贯的报告。

为什么要多个 Agent？两个关键原因：

1. **并行探索**。开放式问题没有预设路径，多个 Subagent 同时往不同方向挖，比单个 Agent 顺序试快得多。
2. **Context 隔离**。每个 Subagent 有自己的 context window，返回给 Lead 的是**压缩后的 findings**，不是原始搜索结果。这等于突破了单个 context window 的上限。

代价也很明确：Anthropic 的数据是 Multi-Agent 大约烧 **15 倍** 于普通 chat 的 token（单 Agent 约 4 倍）。所以这套架构只适合**高价值、可并行、超出单 context** 的任务。

还有一个很重要的设计原则：**协调逻辑写在 Prompt 里，不写在代码里**。谁干什么、干到什么程度、输出什么格式，全靠 Prompt 表达。这也是为什么实现能压到 139 行。

## 二、代码架构图

```mermaid
flowchart TD
    Q(["用户 Query"]) --> P["Lead Agent: plan<br/>拆成 ≤3 个子任务（JSON）"]
    P --> F{"Fan Out<br/>ThreadPoolExecutor"}
    F --> A0["subagent-0<br/>web_search loop"]
    F --> A1["subagent-1<br/>web_search loop"]
    F --> A2["subagent-2<br/>web_search loop"]
    A0 --> M["压缩后的 findings + source URLs"]
    A1 --> M
    A2 --> M
    M --> S["Lead Agent: synthesize<br/>汇总成 markdown 报告"]
    S --> C["Citation Agent<br/>给每个论断挂 inline 引用"]
    C --> R(["最终报告"])
```

三个 Subagent 各自跑一个独立的 Agent Loop（调 LLM → 决定搜什么 → 搜 → 再判断），互不干扰。

## 三、完整代码

```python
"""Multi-Agent Deep Research (minimal). Deps: litellm tavily-python."""

import os
import json
from concurrent.futures import ThreadPoolExecutor

import litellm
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

LEAD_MODEL = "gpt-4.1-mini"
SUBAGENT_MODEL = "gpt-4o-mini"
MAX_NUM_AGENTS = 3
MAX_AGENT_LOOP_TIMES = 3

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns titles, URLs, and snippets.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

def web_search(query, agent="system"):
    print(f"[{agent}] search: {query!r}")
    return [
        {"title": r["title"], "url": r["url"], "content": r["content"]}
        for r in tavily.search(query=query, max_results=4)["results"]
    ]

def llm(model, messages, tools=None, response_format=None, max_tokens=2000):
    return litellm.completion(
        model=model, messages=messages, tools=tools,
        response_format=response_format, max_tokens=max_tokens, temperature=0.2,
    ).choices[0].message

def _assistant_to_dict(msg):
    d = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return d

class Agent:
    def __init__(self, model, system_prompt, name="agent"):
        self.model = model
        self.system_prompt = system_prompt
        self.name = name
        self.sources = []

    def run(self, task):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        for _ in range(MAX_AGENT_LOOP_TIMES):
            msg = llm(self.model, messages, tools=[WEB_SEARCH_TOOL])
            messages.append(_assistant_to_dict(msg))
            if not msg.tool_calls:
                return msg.content
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                results = web_search(args.get("query", ""), agent=self.name)
                self.sources.extend(r["url"] for r in results)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(results)})
        messages.append({"role": "user", "content": "Stop searching and give your final answer now."})
        return llm(self.model, messages).content

SUBAGENT_SYSTEM = (
    "You are a research subagent. You are given one focused objective.\n"
    "Use the web_search tool to gather evidence, refining queries until you can answer well.\n"
    "Then write condensed findings. For every claim, note the source URL inline, e.g. "
    "(source: https://example.com/page).\n"
    "Output format requested by the lead: {output_format}"
)

PLANNER_SYSTEM = (
    "You are the lead agent of a multi-agent research system.\n"
    "Decompose the query into focused, non-overlapping subtasks for parallel subagents.\n"
    'Respond ONLY with JSON: {"subtasks": [{"objective": "...", "output_format": "..."}]}.'
)

SYNTH_SYSTEM = (
    "You are the lead agent synthesizing a final research report from subagents' findings.\n"
    "Write a well-structured markdown report that answers the query as a coherent narrative.\n"
    "Format the report properly: in a list, all items should be of the same category.\n"
    "Preserve the subagents' inline source attributions. Do NOT add a bibliography or URL list."
)

CITATION_SYSTEM = (
    "You are the citation agent. Ensure every substantive claim carries an inline citation to one "
    "of the available source URLs, formatted as a markdown link like ([example.com](https://example.com/page)).\n"
    "Keep citations inline and minimal. Do NOT append a bibliography. Return the full report markdown."
)

def research(query):
    print(f"[lead] query: {query!r}")
    plan = llm(LEAD_MODEL,
               [{"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"Research query: {query}\nReturn at most {MAX_NUM_AGENTS} subtasks."}],
               response_format={"type": "json_object"})
    subtasks = json.loads(plan.content)["subtasks"][:MAX_NUM_AGENTS]
    print(f"[lead] plan: {[s['objective'] for s in subtasks]}")

    def run_subagent(i, st):
        agent = Agent(SUBAGENT_MODEL,
                      SUBAGENT_SYSTEM.format(output_format=st.get("output_format", "concise bullet points")),
                      name=f"subagent-{i}")
        findings = agent.run(st["objective"])
        return {"objective": st["objective"], "findings": findings, "sources": agent.sources}

    # Fan out: subagents research concurrently in parallel threads.
    with ThreadPoolExecutor(max_workers=MAX_NUM_AGENTS) as ex:
        results = list(ex.map(lambda p: run_subagent(*p), enumerate(subtasks)))

    blocks = "\n\n".join(f"## Subtask: {r['objective']}\n{r['findings']}" for r in results)
    report = llm(LEAD_MODEL,
                 [{"role": "system", "content": SYNTH_SYSTEM},
                  {"role": "user", "content": f"Original query: {query}\n\nSubagent findings:\n{blocks}"}],
                 max_tokens=3000).content

    sources = sorted({u for r in results for u in r["sources"]})
    return llm(LEAD_MODEL,
               [{"role": "system", "content": CITATION_SYSTEM},
                {"role": "user", "content": f"Report:\n{report}\n\nAvailable source URLs:\n" + "\n".join(sources)}],
               max_tokens=3000).content

if __name__ == "__main__":
    print(research("Best practices for prompt engineering?"))
```

## 四、代码讲解

### 1. 一个 `Agent` Class，Lead 和 Subagent 共用

`Agent.run()` 就是一个标准的 Agent Loop：调 LLM（带 `web_search` 工具）→ 返回 `tool_calls` 就去搜、把结果塞回 `messages`、进入下一轮 → 返回纯文本就说明信息够了，直接作答。

超出 `MAX_AGENT_LOOP_TIMES` 还没收敛，就补一句 `"Stop searching and give your final answer now."` 强制收尾。这一个 Class，Subagent 用，Lead 也能用。

### 2. Plan：用 JSON mode 拆任务

```python
plan = llm(LEAD_MODEL, [...], response_format={"type": "json_object"})
subtasks = json.loads(plan.content)["subtasks"][:MAX_NUM_AGENTS]
```

拆解规则全在 `PLANNER_SYSTEM` 里 —— "focused, non-overlapping subtasks"。代码只负责截断到 `MAX_NUM_AGENTS`。注意每个子任务除了 `objective` 还带一个 `output_format`，由 Lead 告诉 Subagent 该怎么组织输出，这就是 Anthropic 说的"教会 Agent 如何委派"。

### 3. Fan Out：真并行

```python
with ThreadPoolExecutor(max_workers=MAX_NUM_AGENTS) as ex:
    results = list(ex.map(lambda p: run_subagent(*p), enumerate(subtasks)))
```

Web Search 和 LLM call 都是 I/O bound，多线程就够了，不需要 asyncio。三个 Subagent 真的在同时搜。

### 4. Context 隔离在哪里

关键一行是 `run_subagent()` 的返回值：只返回 `findings`（Subagent 自己压缩过的结论）和 `sources`，**原始搜索结果留在 Subagent 自己的 `messages` 里，不会进 Lead 的 context**。这就是"用多个 context window 换更大的信息吞吐"。

### 5. 两个成本旋钮

- `MAX_NUM_AGENTS` —— 并行几个 Subagent
- `MAX_AGENT_LOOP_TIMES` —— 每个 Subagent 最多搜几轮

Multi-Agent 烧 token 是真的，这两个参数直接决定成本和延迟。默认 3 × 3 在便宜模型上跑一次大约几分钱。

### 6. Citation Pass 单独一轮

合成报告和挂引用**分成两次 LLM call**。合成的时候只管把内容写连贯，引用的时候只管把 URL 对应上。一次让模型同时干两件事，两件都做不好。

## 五、运行 Demo

用中文 query 跑一次：`"巴黎旅行攻略 几天合适"`（完整 notebook：[deep_research.demo2.ipynb](https://github.com/tiejun-ai/deep_research/blob/main/deep_research.demo2.ipynb)，可以直接在 Colab 打开）。

先看日志。Lead 把问题拆成了三个不重叠的角度 —— **需要几天**、**有哪些景点**、**不同天数怎么安排**：

```
[       lead] query    text='巴黎旅行攻略 几天合适'
[       lead] plan     subtasks=[
    'Research the ideal number of days to spend in Paris ...',
    'Identify key attractions and activities in Paris that influence the length of stay.',
    'Provide travel tips and itinerary suggestions for different trip lengths (3/5/7 days).']

[ subagent-0] start    objective='Research the ideal number of days ...'
[ subagent-1] start    objective='Identify key attractions ...'
[ subagent-2] start    objective='Provide travel tips and itinerary ...'

[ subagent-1] search   query='top attractions activities Paris length of stay'
[ subagent-0] search   query='ideal number of days to spend in Paris for tourists'
[ subagent-2] search   query='3 days itinerary Paris travel tips'
[ subagent-2] search   query='5 days itinerary Paris travel tips'
[ subagent-1] search   query='top attractions in Paris visit duration'
[ subagent-0] search   query='Paris tourist itinerary 3 days 5 days'
[ subagent-1] search   query='popular activities in Paris length of stay'
[ subagent-0] search   query='how many days to visit Paris recommendations'
[ subagent-2] search   query='7 days itinerary Paris travel tips'

[ subagent-0] done
[ subagent-1] done
[ subagent-2] done
[       lead] synthesized
[   citation] citation_pass    num_sources=28
```

几个值得注意的点：

- 三个 `search` 日志是**交错**出现的 —— 这就是并行的证据。
- 用户问的是中文，Subagent 自己判断出**英文 query 搜得到更多资料**，全部转成了英文搜索。
- 每个 Subagent 都搜了不止一轮，第二轮的 query 是根据第一轮结果**refine** 过的。
- 一共收集了 **28 个来源**，但进入 Lead context 的只是三段压缩后的 findings。

最终报告（节选）：

> ## 巴黎旅行攻略：几天合适？
>
> ### 理想停留天数推荐
>
> 综合多方资料，**4 到 5 天** 是游览巴黎的理想时长。这个时间段既能覆盖主要景点，又能保证行程不至于过于匆忙。
>
> - **4 天**：足够游览埃菲尔铁塔、卢浮宫、巴黎圣母院、蒙马特区等标志性景点([travelherstory.com](https://www.travelherstory.com/how-many-days-in-paris))。
> - **5 天**：适合首次来访者，可安排一日游（如凡尔赛宫）([nomadicmatt.com](https://www.nomadicmatt.com/travel-blogs/how-to-spend-5-days-in-paris))。
> - **3 天及以下**：只能快速浏览主要景点，缺少深入体验的机会([tripadvisor.com](https://www.tripadvisor.com/Articles-l7eKZyigPQnI-How_many_days_paris.html))。
>
> ### 影响停留时间的主要景点及建议游览时长
>
> | 景点 | 建议游览时间 |
> |---|---|
> | 埃菲尔铁塔 | 约 2 小时 |
> | 卢浮宫 | 约 3 小时（重点参观） |
> | 凡尔赛宫 | 半天至 1 天（含交通） |
> | 蒙马特区 + 圣心大教堂 | 2-3 小时 |
> | 塞纳河游船 | 约 1 小时 15 分钟 |
>
> ### 5 天行程（较为全面）
>
> - **第 1 天**：抵达 → 凯旋门 → 香榭丽舍大街 → 埃菲尔铁塔附近晚餐
> - **第 2 天**：卢浮宫 → 奥赛博物馆 → 塞纳河游船
> - **第 3 天**：巴黎圣母院 → 圣礼拜堂 → 拉丁区 → 晚餐
> - **第 4 天**：凡尔赛宫或吉维尼一日游
> - **第 5 天**：马莱区购物 → 卢森堡公园休闲 → 告别晚餐
>
> 5 天行程适合首次来访者，行程宽裕且丰富([dangerous-business.com](https://www.dangerous-business.com/5-days-in-paris))。

报告的结构 —— 先给结论、再列景点表格、再给三档行程 —— 正好对应 Lead 拆出的三个子任务。这就是 Multi-Agent 的价值：**问题的结构决定了报告的结构**，而不是让一个 Agent 顺着一条线往下想。

## 小结

- **Orchestrator-Worker 是 Deep Research 的标准架构**：Lead 拆解 + Subagent 并行 + 压缩回传 + 统一引用。
- **协调逻辑放在 Prompt 里**，代码只负责跑 loop、开线程、截断预算 —— 所以 139 行就够了。
- **Context 隔离比并行更重要**。Subagent 返回压缩后的 findings 而不是原始搜索结果，这才是突破单 context 上限的关键。
- **Token 是有代价的**。`MAX_NUM_AGENTS` 和 `MAX_AGENT_LOOP_TIMES` 这两个旋钮要根据 query 的价值来调，不是越大越好。

代码：[github.com/tiejun-ai/deep_research](https://github.com/tiejun-ai/deep_research) —— 欢迎 star。

---

#ai #aiagents #multiagent #deepresearch #llm #aisearch #tooluse #aicoding #claude #anthropic #人工智能 #ai学习 #ai编程 #多智能体 #大模型

If you like this post, consider star [the repo](https://github.com/sleepingbear-ai/sleepingbear-ai.github.io).
