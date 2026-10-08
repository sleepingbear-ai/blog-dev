+++
date = '2026-10-08T10:00:00-07:00'
draft = false
title = 'Anthropic：如何构建有效的 AI Agent'
summary = '解读 Anthropic《Building effective agents》：增强型 LLM、五种 Workflow、自主 Agent，以及 Agent Frameworks 选择、工具设计和持续评估。'
+++

Anthropic 在 [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) 中总结了一组简单、可组合的架构模式。核心建议是：**从最简单的方案开始，只有评估证明有收益时，才增加复杂度。**

原文发表于 2024 年 12 月，Agent的构建方法日新月异， 但此文仍值得学习。下面聚焦其架构与工程原则，配图均来自 Anthropic 原文。

## 先区分 Workflow 与 Agent

「Agent」这个词有不同用法。Anthropic 将相关系统统称为 agentic systems，同时区分两种架构：

- **Workflow：** 由代码预先规定执行路径，编排 LLM 与工具，适合任务边界清楚、需要可预测行为的场景。
- **Agent：** 由 LLM 动态决定流程与工具使用，适合步骤难以预知、需要灵活决策的开放任务。

选择时先问：单次 LLM 调用能否解决问题？如是则不必用Agent, 用Workflow: 很多 AI 应用通过检索（Retrieval）补充信息、通过上下文示例（In-Context Examples）明确要求，就能满足需求。Agent 通常要用更多成本和延迟换取任务效果，需要评估这笔交换是否值得。

### 基础组件：Augmented LLM

后面的模式都建立在 **Augmented LLM（增强型 LLM）** 之上：为模型接入检索、工具和记忆。模型可以生成搜索请求、选择工具，也可以决定保留哪些信息。

实现时有两个重点：增强能力要匹配具体任务，接口要清晰、文档要完整。原文提到 MCP 是接入外部工具与数据源的一种方式。

![Augmented LLM：模型连接检索、工具与记忆，图源 Anthropic](/images/building-effective-agents/augmented-llm.png)

## 五种常见 Workflow

### Prompt chaining：串联

把任务拆成固定步骤，每次 LLM 调用处理前一步的输出。这样每次调用只需完成一个更简单的子任务，以额外延迟换取更高准确性。

图中的 **Gate** 是程序化检查：确认中间结果符合要求，再继续下一步。例如先生成提纲，检查是否覆盖必要内容，通过后才写正文。另一个例子是先生成营销文案，再翻译。

**适用条件：** 能提前将任务清楚地拆成固定子任务，各步骤之间有明确的输入输出关系。

![Prompt chaining：串联调用与中间检查，图源 Anthropic](/images/building-effective-agents/prompt-chaining.png)

### Routing：路由

先对输入分类，再交给专门的模型、Prompt 或流程。每条路径可以单独优化，减少一个通用 Prompt 同时处理多类问题时的相互干扰。

例如，客服问题分到一般咨询、退款和技术支持；也可以将简单问题交给成本较低的模型，复杂问题交给能力更强的模型。分类既可以用 LLM，也可以用传统分类器或算法。

**适用条件：** 输入类别有清晰差异，并且系统能够可靠地判断类别。分流准确性会影响后续处理效果。

![Routing：分类后分流到专门的处理路径，图源 Anthropic](/images/building-effective-agents/routing.png)

### Parallelization：并行

让多个 LLM 调用同时执行，再由程序汇总。原文区分两种形式：

- **Sectioning（分工）：** 将独立子任务并行处理。例如，一次调用回答用户问题，另一次检查安全性；或分别评估答案的不同质量维度。
- **Voting（投票）：** 对同一任务进行多次尝试，使用不同 Prompt 获得多种判断，再聚合结果。例如多次审查代码漏洞，或按不同投票阈值判断内容是否违规。

**适用条件：** 子任务可以独立执行，或者多个视角能提高判断的可信度。分工可以降低等待时间，也让每次调用更专注；投票阈值则需要根据误报与漏报的取舍来选择。

![Parallelization：并行执行后汇总，图源 Anthropic](/images/building-effective-agents/parallelization.png)

### Orchestrator-workers：协调与分工

由一个中央 LLM（Orchestrator）理解任务，动态拆分子任务，分派给 Worker LLM，最后综合结果。

例如，修改代码时，需要改哪些文件、每个文件如何改，取决于具体需求；深度搜索（Deep Search）时，也需要根据问题决定调查哪些方向。

**适用条件：** 无法提前列出完整的子任务。它与并行模式的关键区别是分工方式：并行模式的子任务预先确定，这里的子任务由协调者根据输入动态产生。Anthropic 仍将这种结构列为 Workflow。

![Orchestrator-workers：动态拆分、分派与汇总，图源 Anthropic](/images/building-effective-agents/orchestrator-workers.png)

### Evaluator-optimizer：评估与优化

一个 LLM 生成结果，另一个 LLM 提供评价和反馈，再据此修改，形成循环。这里的重点是建立有效的评价标准，让反馈能够指导下一轮改进。

原文给出两个判断信号：**人类提出反馈后，结果确实能改善；LLM 也能提供类似的有效反馈。** 两者成立时，迭代才更可能带来价值。

例如，文学翻译需要反复处理细微语义和表达；深度搜索（Deep Search）需要评价当前信息是否充分，再决定是否补充检索。**适用条件：** 标准明确，而且改进效果可以验证。

![Evaluator-optimizer：生成、评价与反馈循环，图源 Anthropic](/images/building-effective-agents/evaluator-optimizer.png)

## Agent：根据环境反馈行动

Agent 通常从用户指令或一段澄清对话开始。目标明确后，模型制定计划并执行，在需要额外信息或判断时再请求人类介入。

实现的核心是一个循环：**决定行动 → 执行工具 → 获得环境反馈 → 决定下一步。** 工具返回值、代码执行结果等提供了判断进展的依据，帮助 Agent 调整行动并从错误中恢复。

这类架构适合难以预知步骤数量、无法写死执行路径的任务，例如修复代码问题或操作计算机。系统可以在完成目标时结束，也应设置最大迭代次数等停止条件，并在检查点或阻塞时请求人类反馈。

![Autonomous agent：与环境交互并根据反馈继续行动，图源 Anthropic](/images/building-effective-agents/autonomous-agent.png)

自主性带来更高成本，也可能让错误在多轮行动中累积。因此，原文建议在沙箱中充分测试，并设置适当的边界控制（Guardrails）。任务越开放，越需要验证模型决策和工具使用是否可靠。

## 工程原则：简单、透明、清晰的工具接口

### Agent Frameworks：理解底层，再选择抽象

Agent Frameworks 可以简化模型调用、工具定义与解析、调用串联等工作，但额外的抽象可能遮住 Prompt 和响应，使调试更困难，也容易促使开发者过早增加复杂度。

Anthropic 建议从直接调用 LLM API 开始，很多模式只需少量代码即可实现。使用 Agent Frameworks 时，要理解它实际怎样运行；对 Framework 底层机制的错误假设，是常见的问题来源。

### 透明：让计划与执行步骤可观察

原文将明确展示 Agent 的计划步骤列为核心原则。系统需要让人看清它准备做什么、已经执行了哪些关键行动，以便理解行为、定位错误，并在需要时介入。

### ACI：把工具接口当成产品来设计

**Agent-Computer Interface（ACI）** 是模型与工具交互的接口。原文建议像设计人机界面一样投入精力：工具定义和文档也需要认真做 Prompt Engineering。

- **描述清楚：** 说明用途、参数含义、输入格式、示例、边界情况，以及与其他相似工具的区别。
- **降低格式负担：** 避免要求模型精确计算 diff 行数，或为代码进行大量 JSON 转义。等价的操作格式，对模型的难度可能不同。
- **观察真实用法：** 用多种输入查看模型怎样选择工具和传参，再迭代接口。
- **从接口上防错：** 调整参数与约束，让常见错误更难发生。

原文中的具体例子: Coding Agent 改变工作目录后容易用错相对文件路径，将工具改为要求绝对文件路径后，这类错误得到解决。Anthropic 在其 SWE-bench Agent 上，优化工具所花的时间甚至超过了整体 Prompt调试。

## 组合模式，持续评估

这些模式可以按任务组合和调整。例如，路由后的某条路径可以串联多个步骤，也可以加入评估与优化循环。

原文反复强调：先建立简单方案，再通过评估迭代。每次增加步骤、自主性或模型调用，都应确认任务效果的收益是否值得额外成本和延迟。
