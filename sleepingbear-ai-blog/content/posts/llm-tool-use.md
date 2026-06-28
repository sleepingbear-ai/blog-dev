+++
date = '2026-06-27T10:00:00-07:00'
draft = false
title = 'LLM 是怎么学会 Tool Use 的？'
summary = """
  *AI Agent 的强大，来自 LLM 看似神奇的调用 **Tools** 的能力。这份高层次的拆解，帮你看清魔法背后的原理。*
"""
+++

*AI Agent 的强大，来自 LLM 看似神奇的调用 **Tools** 的能力。这份魔法到底从哪来？下面做个高层次的拆解，把它讲清楚。*

---

## TL;DR

一个常见的误解是：LLM 用工具时，是它自己直接连网、跑代码、或启动某个程序。实际上，**LLM 从来不会真的执行工具**，它依然只是一个文本预测器。

另一个误解是：只要在 Prompt 里要求一下，*任何* LLM 都能可靠地吐出格式正确的 Tool Call（JSON）。强一点的模型确实**可以**靠 Prompt 被"哄"出来，但要做到**可靠**——选对工具、传对参数、格式合法、时机正确——就得靠**专门的训练**。正是这种训练，把 Tool Use 从一个脆弱的 Prompt 技巧，变成了原生、可依赖的能力。

所谓 "Tool Use"（也叫 *Function Calling*），其实是三个部分精心编排的配合：

1. **结构化的 Prompt** —— AI Agent（比如一个聊天机器人，或 Claude Code）在 System Prompt 里，把可用工具列表（以 JSON Schema 的形式）交给 LLM。
2. **专门的训练** —— LLM 被 fine-tune 过，当它识别到需要时，会*发出*一个结构化的 Tool Call 请求，而不是用自然语言作答。
3. **外部的 Runtime** —— AI Agent 盯着模型的输出，替它执行工具的函数调用，再把结果喂回到对话里。

模型决定*调用什么*、*用什么参数*。真正的调用，由 AI Agent 这层外部软件来做。

把这几部分拼起来，就构成了 **Agent Loop**：

```
       用户 Query + 工具列表
                │
                ▼
          ┌──────────┐   发出 Tool Call    ┌──────────────────┐
   ┌─────►│   LLM    │ ──────────────────► │  AI Agent 执行   │ ──► 🌐 / 💻 / 🗄
   │      └──────────┘                     │      工具        │
   │           │                           └──────────────────┘
   │           │ 不再需要工具                        │
   │           ▼                                     │ tool_result
   │       最终答案 ──► 用户                          │
   │                                                 │
   └─────────────────────────────────────────────────┘
                 循环，直到 LLM 不再调用工具
```

## 原理拆解

端到端地看，一次完整的 Tool Use 是这样的：

```
            ┌──────────────────────────── 对话历史 ────────────────────────────┐
 SYSTEM     │ "你有这些工具：                                                   │
 PROMPT     │   get_current_weather(location: str, unit: 'celsius'|'fahrenheit')│
            │    → 获取某个城市的当前天气数据"                                  │
            └──────────────────────────────────────────────────────────────────┘
 USER  ──►  "巴黎现在天气怎么样？"
              │
              ▼
        ┌───────────┐   (1) 意识到静态知识答不了；把请求匹配到
        │    LLM    │       天气工具的描述，于是不写自然语言，而是发出：
        └───────────┘
              │
              ▼  (2) 结构化的 Tool Call 请求
        { "name": "get_current_weather",
          "arguments": { "location": "Paris", "unit": "celsius" } }
              │
              ▼  (3) AI AGENT 看到这个调用，
        ┌───────────┐      向天气服务发起真正的 API 调用……
        │ AI AGENT  │ ───► 🌐 天气 API ───► { "temp_c": 22, "sky": "sunny" }
        └───────────┘      ……再把结果以 `tool_result` 消息喂回去
              │
              ▼  (4) 模型读到新数据，写出自然语言
        ┌───────────┐
        │    LLM    │ ───► "巴黎现在 22°C，晴天！"
        └───────────┘
```

逐个拆开这四个环节：

### 1. 蓝图（System Prompt）

在你输入任何内容之前，AI Agent 已经注入了一段隐藏的 **System Prompt**，里面列着可用的工具。每个工具都配有严格的蓝图——通常是 JSON Schema——讲清楚：

- **这个工具做什么：** 比如 `"获取某个城市的当前天气数据"`。
- **需要哪些输入：** 比如一个 `location` 字符串，和一个可选的 `unit` 枚举（`celsius` / `fahrenheit`）。

因为模型对语义高度敏感，它读了这些描述，就能*在概念上*理解每个工具是干嘛的。

具体到 OpenAI API，你在请求的 `tools` 参数里传入工具蓝图：

```python
from openai import OpenAI
client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather data for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Paris'"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather like in Paris right now?"}],
    tools=tools,
)

# 模型不会用自然语言回答 —— 它返回一个要执行的 Tool Call：
call = response.choices[0].message.tool_calls[0]
print(call.function.name)       # "get_current_weather"
print(call.function.arguments)  # '{"location": "Paris", "unit": "celsius"}'
```

### 2. 触发（识别到需要）

当你问巴黎天气时，模型意识到自己的静态训练数据答不了，把你的请求和工具描述做匹配，然后决定行动。它不再用自然语言回复，而是切换状态，输出一个 Tool Use 请求（常常包在 `<tool_call>` 之类的特殊语法里，或就是裸 JSON），声明它的意图：

```json
{
  "name": "get_current_weather",
  "arguments": { "location": "Paris", "unit": "celsius" }
}
```

### 3. 交接（外部环境）

运行模型的那段代码——也就是 AI Agent——一直盯着输出流。一旦看到 LLM 吐出的那段 Tool Call JSON，它就拿着生成的参数（`"Paris"`、`"celsius"`），向天气服务发起*真正的* API 调用，取回实时数据，再以一条标着 `tool_result` 的新消息塞回对话历史。

整个把戏的关键就在这里：模型自始至终只产出了*一段描述函数调用的文本*，真正去调用的，是 AI Agent 里的软件。

### 4. 作答（综合出答案）

工具结果现在已经躺在它的 Context 里，LLM 据此生成回复：*"巴黎现在 22°C，晴天！"*

## LLM 是怎么学会这个行为的？

LLM 不是天生就会写 JSON Tool Call 请求的，这是训练出来的：

- **Supervised Fine-Tuning（SFT）：** 用成千上万条模拟对话来训练模型，形态是 *[用户提问] → [模型发出 JSON Tool Call] → [系统插入工具结果] → [模型给出最终答案]*。它学会复现这个模式。
- **Reinforcement Learning（RLHF / DPO）：** 模型因生成不会让 Runtime 崩溃的合法参数而被奖励，因幻觉出输入、或选错工具而被惩罚。

### 各大实验室实际怎么训练

旗舰实验室的技术报告，透露了高层次的"配方"：

#### OpenAI

原生的 "Function Calling" 在 2023 年 6 月（`gpt-3.5-turbo-0613`）上线。他们用这样的数据做 fine-tune：System Prompt 里含 API Schema，目标输出是字符串化的 JSON。更晚的模型（GPT-4o）则针对 **parallel function calling**（并行函数调用）做了调优。

#### Anthropic

Claude 被训练成把工具描述当作**硬约束**，用 RLHF 和 DPO 来惩罚幻觉的、或未列出的参数。更近期的 **programmatic tool calling**，训练模型直接吐出*编排代码*（Python 循环），在沙箱里运行——处理成千上万 token 的中间数据，只把压缩后的最终答案返回主 Context。

#### Google

Gemini 的技术报告提到，Tool Use 不是在 post-training 阶段才"装"上去的——结构化的 API 交互、数据库查询、多轮编程日志，是和普通文本一起混进 **pre-training** 的。自动化的校验循环，会惩罚任何过不了 JSON 校验器或编译器的工具参数。

## 研究与工程版图

Tool Use 从一个新颖的点子，演变成了标准的 AI 工程基础设施。几个里程碑：

**奠基性研究**

- **Toolformer**（Schick et al., 2023）—— 开创性地证明了模型可以通过自监督*自学*：什么时候该发 API 调用、传什么参数、以及如何把结果折回到 next-token 预测里，同时不丢失它的文本生成能力。
- **Gorilla**（Patil et al., 2023）—— 把这个想法从少数几个工具（计算器、日历）扩展到*成千上万个*复杂、不断变化的 Web API，靠 instruction tuning 加一个 retriever 来缓解 API 文档幻觉。

**工程标准**

从研究走向生产，暴露出严重的碎片化：每个厂商都想要一套不同的 Schema，每个数据源都要写一坨专门的胶水代码。行业的答案是统一的协议层——其中最有名的就是 Anthropic 的 **Model Context Protocol（MCP）**。MCP 标准化了 AI Agent 与 "MCP Server"（工具或数据源的封装）之间的对话方式。工程师部署模块化的 MCP Server，LLM 则从这些 MCP Server 里**动态发现**可用的工具和数据源。

---

### 参考文献

- Schick, T., et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools.* [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Patil, S. G., et al. (2023). *Gorilla: Large Language Model Connected with Massive APIs.* [arXiv:2305.15334](https://arxiv.org/abs/2305.15334)
- Hou, X., Zhao, Y., Wang, S., & Wang, H. (2025). *Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions.* [arXiv:2503.23278](https://arxiv.org/abs/2503.23278)

---

#ai #aiagents #llm #tooluse #functioncalling #mcp #人工智能 #ai学习 #大模型

If you like this post, consider star [the repo](https://github.com/sleepingbear-ai/sleepingbear-ai.github.io).
