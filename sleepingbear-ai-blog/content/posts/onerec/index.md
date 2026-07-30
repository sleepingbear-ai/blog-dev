+++
date = '2026-07-29T10:00:00-07:00'
draft = false
title = 'OneRec（快手）详解和思考：一个生成式推荐模型统一召回与排序'
tags = ['onerec', '推荐系统', '生成式推荐', 'semanticid', 'moe', 'dpo', '偏好对齐', 'llm', '快手', 'ai学习']
summary = """
  *一个生成式推荐模型，代替整套"召回→排序"流程；一次生成一整个 session；再用 Reward Model 驱动的 Self-Improving DPO 循环做 Preference Alignment。*
"""
+++

*一个生成式推荐模型，代替整套"召回→排序"流程；一次生成一整个 session；再用 Reward Model 驱动的 Self-Improving DPO 循环做 Preference Alignment。*

论文：**[OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment](https://arxiv.org/abs/2502.18965)**（Deng et al., 快手, 2025）

---

## TL;DR

推荐系统通常是**多阶段**流程：召回 → 粗排 → 精排。每一段都有一个独立的模型，而上一段的质量，就是下一段的天花板。像 **TIGER**（[参见我之前的这篇](../tiger-generative-retrieval/)）这样的生成式召回工作，把召回变成了 LLM 式的 **next-Semantic-ID 预测**——但它只改变了**召回**这一段。

**OneRec** 更彻底：把**整条**推荐链路统一成**一个 encoder–decoder Transformer** 模型。它读用户的观看历史，**直接生成下一个 session 的 item list**——召回和排序合并为一。三个关键设计：

- **稀疏 MoE（Mixture-of-Experts）decoder**：让模型能高效 scale up。
- **Session-wise Generation**：一次生成 5~10 个视频的 item list，而不是一次一个 item。
- **IPA（Iterative Preference Alignment，迭代式偏好对齐）**：用 Reward Model + DPO 的循环，教模型什么样的 session 用户真的喜欢。

OneRec 在快手 main feed（数亿 DAU）上线，A/B 测试**总观看时长 +1.68%**——在工业级规模上，这是生成式推荐的一个相当大的胜利。OneRec 更是一次很有意思的尝试：用**一个**端到端训练的模型，替掉**整套**多阶段架构——这背后是行业更大的趋势：**少一些人工设计的 pipeline 结构，多给模型一些自由，换取更高的天花板。**

## 问题在哪

**多阶段架构。** 要在 10¹⁰ 量级的视频库里高效推荐，工业推荐系统会把候选一层层漏斗下来，每一层更重、更准：召回（~10⁵ 个候选）→ 粗排（~10³）→ 精排（~10²）。OneRec 把这一整套换成一次 encoder→decoder：

![图 1：(a) OneRec 的统一架构，用单个 encoder–decoder 从视频库端到端生成推荐结果；(b) 传统的多阶段架构，把视频库依次经过召回、粗排、精排三个阶段层层过滤。](fig1-unified-vs-cascade.png)

*OneRec 的统一生成（a）vs 传统多阶段架构（b）。[论文](https://arxiv.org/abs/2502.18965) Figure 1。*

多阶段架构的两个结构性问题：

- **每一段都是下一段的上限。** 排序只能对召回送上来的东西重新排一遍；被前面丢掉的，就永远回不来了。而且各阶段模型常是独立训练的，误差会层层累积。
- **生成式推荐只用在了召回上。** TIGER 这类方法把**召回**重新定义成生成 next item ID，但生成之后，还是要用下游的传统排序链路。

**OneRec 的思路：用一个生成式模型替掉整个多阶段架构。** 给定用户历史，**直接生成**最终的 session 的 item list——不再有独立的召回、粗排、精排模型。召回和排序，就是同一模型的自回归解码（autoregressive decoding）。

## 方法详解

OneRec 的结构：

- **item tokenization**：把每个视频变成一个短的 Semantic ID tuple。
- **生成模型**：读用户历史，解码出下一个 session。
- **迭代式 DPO 偏好对齐（IPA）**：用一个 **Reward Model** 来刻画用户到底喜欢什么。

### Step 1 —— 把视频 tokenize 成 Semantic ID（balanced K-means）

和 TIGER 一样，OneRec 给每个 item 一个 **Semantic ID**：一个短 codeword tuple，相似的 item 共享前缀。每个视频先是一个**多模态 embedding** `e`，再被量化（quantized）成 `L = 3` 个 token（每层一个）。

但方法不同：TIGER 训练并使用了 **RQ-VAE**；OneRec 发现 RQ-VAE 产出的 **code 分布严重不均衡**——少数 code 吸走了大部分 item，而大部分 code 几乎是空的（所谓 *"hourglass phenomenon"*，沙漏现象）。这对生成模型很不友好：它希望概率能均匀分布在一个被充分利用的 codebook 上。

所以 OneRec 换成了 **residual balanced K-means**（残差式均衡 K-means）。每一层把 item 聚成 `K = 8192` 个 cluster，而且**加了约束：每个 cluster 的 item 数量必须一样多**（`w = |V| / K`）。一个 item 的 Semantic ID，就是逐层最近的那些 centroid 的序号：

![residual balanced K-means 的 Semantic ID 分配过程：视频 embedding e 逐层往下走，每一层在这一层 8192 条的 codebook 里找最近的质心，得到一个 code（依次是 a、b、c），剩下的残差（r² = e − code_a，然后 r³ = r² − code_b）传给下一层。三个 code 合起来就是 Semantic ID (a, b, c)。](semantic-id.svg)

*用 residual balanced K-means 分配 Semantic ID（示意图；论文里是 Algorithm 1 的伪代码）。*

3 层各有一个 8192 个 code 的 codebook，所以只用 `3 × 8192` 个 codeword embedding，就能表示 `8192³ ≈ 5.5×10¹¹` 个 item。一个视频，从此就是**3 个 token**。

### Step 2 —— Session-wise 生成（而不是 point-wise）

传统的 point-wise 推荐模型只预测或评价"下一个视频"。OneRec 的 **session-wise 生成**直接生成一个**高价值 session**——一次用户请求返回的那 5~10 个视频。

举个例子：用户刚看完一个**钓鱼**视频，现在要返回 5 个：

- **point-wise**：每个候选单独打分，取最高的 5 个——很可能是 5 个雷同的钓鱼视频。
- **session-wise**：直接生成一整串 item list，比如钓鱼技巧 → 水下咬钩高光 → 钓鱼翻车合集 → 渔具测评 → 户外做饭。因为以 session 为目标，模型会自己优化**多样性**和**顺序**。

训练目标是从日志里挖出来的 **high-value session**：比如用户看了 ≥5 个视频、总观看时长超过某个阈值、并且有互动（点赞 / 收藏 / 分享）的 session。模型是一个 seq2seq 生成模型，以用户行为历史序列为输入，学习生成这些 high-value session（`m = 5` 个目标 item）。

#### 模型结构（encoder–decoder + MoE）

统一生成推荐模型是一个 **encoder–decoder Transformer**：

![图 2a：OneRec 架构。Encoder 用全可见的 self-attention 和 FFN 读入用户行为历史序列（每个视频是用 SEP 分隔的 semantic token），产出上下文 H。Decoder 用 causal self-attention、对 H 的 cross-attention，以及一个 Mixture-of-Experts 层（router 从 N 个 expert 里激活 2 个）自回归地生成高价值 session，用 next-token-prediction loss 训练。](fig2-architecture.png)

*OneRec 的 encoder–decoder（decoder 里是 MoE），[论文](https://arxiv.org/abs/2502.18965) Figure 2a。*

- **Encoder** 读用户行为序列 `H_u`（看过 / 点赞 / 关注 / 分享的视频，最多 `n = 256` 个）构成的 Semantic ID 序列。
- **Decoder** 逐 token 生成目标 **session**（`L_NTP` 就是 seq2seq 里经典的 Next-Token-Prediction loss），再把每 3 个 token 一组映射回视频。
- **Decoder 里的稀疏 MoE** 把 FFN 换成了 `N_MoE = 24` 个 expert，router 每个 token 只激活 `K_MoE = 2` 个。这样可以只 scale 模型**容量**，而不成比例地增加**推理成本**——线上服务时**只有约 13% 的参数是激活的**。而且和 LLM 的 Scaling Law 一致：论文发现模型越大，推荐效果越好。

### Step 3 —— 用 Reward Model 和 DPO 做迭代式偏好对齐

Next-Token-Prediction 训练只教会模型**模仿**好的 session。为了让它学会生成**更好的** session，OneRec 从 LLM 的 post-training 里借鉴了 **DPO**（Direct Preference Optimization）。但推荐场景有个特别的困难：

> 在 NLP 里，人工标注可以给出"回答 A 优于回答 B"这样的 pair。但在推荐系统里，**每一次请求，推荐系统只会给用户展示一个 item list**——不能同时把"好版本"和"坏版本"都展示给用户。**Preference Pair 在日志里不存在。**

OneRec 的解法：训练一个 **Reward Model** 来替代真实的用户反馈，然后用它在循环里不断改进生成模型。

#### Reward Model

Reward Model（RM）给一个候选 session 打分，而且是**多目标同时打分**——它很像传统推荐系统里常用的多任务排序模型（Multi-task Ranking Model）。对于 session `S = {v₁, …, vₘ}` 和用户 `u`：先做 **target-aware fusion**（`eᵢ = vᵢ ⊙ u`，比如对用户行为序列做 target attention），再让这些 item **通过 self-attention 相互交互**，然后 **sum-pooling** 成一个向量，最后过**四个 `Sigmoid(MLP)` 塔**——每个塔预测一个目标：**观看时长（swt）、完播（vtr）、关注（wtr）、点赞（ltr）**。Reward Model 用海量的日志反馈、以 binary cross-entropy loss 预训练。

![Reward Model 架构：session 里的每个 item vᵢ 先与用户 u 做 target-aware 融合（eᵢ = vᵢ ⊙ u）；得到的 item 向量通过 self-attention 相互交互，再 sum-pooling 成一个向量，最后送进四个 Sigmoid(MLP) 塔，各自预测一个 reward——观看时长（swt）、完播（vtr）、关注（wtr）、点赞（ltr）。](reward-model.svg)

*Reward Model 结构（依据[论文](https://arxiv.org/abs/2502.18965) 3.3.1 节绘制，论文中没有这张图）。*

论文没有说最终的总 reward 怎么计算，但很可能是各个目标预测值的某种组合（比如 weighted sum）。

#### 用 DPO 做迭代式偏好对齐

有了 Reward Model，OneRec 就能自我迭代改进：

用生成模型生成候选 session → 用 Reward Model 打分 → 选择偏好 pair → 用 DPO 训练生成模型 → 拿改进后的模型再来一轮：

![图 2b：迭代式偏好对齐循环。当前模型 OneRec_t 在训练数据上 beam search 出 N 个候选 session；Reward Model 给每个打分；reward 最高的作为 chosen、最低的作为 rejected；在这个 pair 上用 DPO loss 训出 OneRec_{t+1}，再送回去做下一轮自我提升。](fig3-ipa.png)

*Iterative Preference Alignment（[论文](https://arxiv.org/abs/2502.18965) Figure 2b）。*

**生成偏好 pair。** 对每个用户，用当前模型 `OneRec_t` 做推理，**beam search** 出 `N = 128` 个候选 session，再用 Reward Model 打分。**reward 最高的那个当 *chosen***，**最低的那个当 *rejected***——一个（chosen, rejected）偏好 pair 就有了。

**迭代式 DPO 训练。** 每一轮训一个新模型 `OneRec_{t+1}`，从当前的 `OneRec_t` 初始化——而 `OneRec_t` 随即被**冻住，当作 reference model**。对每个偏好 pair，**DPO** loss 鼓励 `OneRec_{t+1}` 提高生成 *chosen* session 的概率、压低生成 *rejected* 的概率（两者都是**相对于冻住的 reference** 来衡量）：

![单个偏好 pair 的 DPO loss：负的 log-sigmoid，括号里是 β 乘以（chosen session 在训练模型 P_{t+1} 下与冻住的 reference P_t 下的似然之比取 log），再减去 rejected session 的同一个 log 比值。](dpo-loss.svg)

有意思的是：OneRec **并不是只优化 DPO 目标**，而是训练组合 loss：**`L = L_NTP + λ·L_DPO`**——保留 Next-Token-Prediction loss，让模型在学 Preference 信号的同时（`L_DPO`），仍然**模仿**生成好的 session（`L_NTP`）。纯 DPO 训练有可能会让模型偏离生成合法 session 的能力。

整个过程是**迭代**的：`OneRec_{t+1}` 成为下一轮的 reference model，重新生成候选和偏好 pair，循环好几轮（`OneRec_1 → OneRec_2 → … → OneRec_T`）——每一代都从上一代**自己的输出**里 bootstrapping，也就是**自我提升（self-improvement）**。为了控制训练成本，只有 **`r_DPO = 1%`** 的training examples用 `L_DPO` loss （其余的用普通的 `L_NTP` loss）。实验显示，**1% 的 DPO 采样比例**平均能拿到**最优效果的 95%**，而算力开销远小于更高的比例。

## 实验与结果

在快手的大规模工业数据集上做离线评测，加上线上 A/B 测试。指标：观看侧 **swt**（session 观看时长）、**vtr**（完播）；互动侧 **wtr**（关注）、**ltr**（点赞）。用来比较的模型包括：判别式 point-wise 模型（SASRec、BERT4Rec、FDSA）、生成式 point-wise 模型（TIGER），以及各种 DPO 变体（IPO、cDPO、rDPO、CPO、simPO、S-DPO）。

主要结论：

- **Session-wise 生成打败 point-wise 生成。** OneRec-1B（不带 IPA）比 TIGER-1B 高 **+1.78% max swt**、**+3.36% max ltr**。
- **一点点偏好对齐，收益巨大。** 只用 **1% 的 DPO** 数据，OneRec-1B+IPA 在 OneRec-1B（不带 IPA）之上拿到 **+4.04% max swt** 和 **+5.43% max ltr**，而且 **IPA 优于所有其他 DPO 变体**。
- **OneRec 像 LLM 一样 scale。** 模型从 0.05B 涨到 1B，精度稳定提升（0.05B→0.1B 提升 +14.45%，之后每一档再涨 5~6%）——而 MoE 设计正是 scale up **效率**的关键。注意：1B 用 LLM 的标准看，还是很小。
- **线上 A/B（快手 main feed，数亿 DAU）：** OneRec-1B+IPA 带来 **+1.68% 总观看时长**和 **+6.56% 平均播放时长**——在这个体量上，是很实在的收益，而且收益随模型变大而增长：

| 模型 | 总观看时长 | 平均播放时长 |
|:---|:---:|:---:|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| **OneRec-1B + IPA** | **+1.68%** | **+6.56%** |

*[论文](https://arxiv.org/abs/2502.18965) Table 2：线上 A/B 测试中相对快手现有多阶段推荐系统的提升。*

## 我的一些想法

### OneRec 的优势

- **单个端到端模型，没有多阶段架构的误差累积。** 把召回和排序统一起来，去掉了"每段都是下一段天花板"的瓶颈，也省掉了维护好几个独立训练模型的负担。更重要的是，OneRec 证明了这样一个统一的生成式推荐模型**能在工业级规模上真正上线，并且拿到实打实的收益**。
- **Session-wise 很有效。** 推荐的输出本来就是一个 item list，不是一个 item。直接生成 item list——把"选什么"和"怎么排"一起学——效果好于传统的 point-wise 推荐。
- **MoE 让 scaling 变得可行。** 拿到 LLM 式的 scaling up 收益，同时推理只激活约 13% 的参数——正是这一点，让"一个很大的生成式推荐模型"从研究玩具变成快手真能上线的服务。
- **DPO 式的自我提升对齐，在推荐系统里是有效的。** 针对"推荐系统里拿不到偏好 pair"这一问题，用 Reward Model 驱动的自我提升循环来优化模型，是非常漂亮的创新！它把 LLM 的 post-training 技术移植进推荐系统，而且真的带来了性能提升。

### OneRec 的缺点

- **服务成本和延迟。** 在自回归 decoder 上做 beam search（size 128），比传统的"召回再排序"链路重得多。要靠 KV-cache 解码、float16、MoE 稀疏化，才勉强跑得动。
- **Reward Model 成了新的天花板。** IPA 优化的是 **RM 分数**，不是真实用户。RM 的任何偏差（比如过度偏爱观看时长）都会被放大进策略里——这是典型的 reward hacking 风险。
- **复杂度上升。** 相比传统推荐系统，balanced K-means tokenization + MoE encoder-decoder + Reward Model + 迭代 DPO 循环，是一堆新增的活动零件。
- **到底是 scale up 的功劳，还是统一架构的功劳？** OneRec 用一个 scale up 之后的统一模型拿到了显著收益，但也可以争辩说：把传统的召回和排序模型分别 scale up，说不定也能拿到其中一部分收益。

### 大方向

- **这篇论文为什么重要。** TIGER 让生成式**召回**变得有竞争力；OneRec 是最早做到**一个生成式模型**、在真实的大规模生产系统里**打败一套成熟且调优充分的完整多阶段推荐系统**（召回 + 排序）的工作之一。这正是这个领域一直在等的里程碑——它展示了生成式推荐这个新方向的巨大潜力。
- **行业趋势。** "推荐正在变成一个 LLM 问题"这个论断，在这里走到了它的逻辑终点：item 即 token，整条链路就是一个序列模型，用 Scaling Law + MoE 换容量，用 RLHF 式的偏好对齐（DPO + Reward Model）换质量。随着推理成本继续下降，可以预期推荐链路里更多的部分会塌缩进这样的生成式结构里——而自我提升的对齐循环，很可能成为推荐系统里一项激动人心的关键技术！
- **把推荐模型 scale up。** OneRec 只是其中一条路，还有很多值得探索。OneRec-1B 按 LLM 的标准还很小，随着 LLM 推理越来越便宜，把模型继续做大，是一条很有希望拿到更大收益的路。

## 参考文献

- **OneRec**：[OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment](https://arxiv.org/abs/2502.18965)（Deng et al., 快手, 2025）
- **TIGER**：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)（Rajput et al., NeurIPS 2023）——[我的解读](../tiger-generative-retrieval/)
- **DPO**：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)（Rafailov et al., NeurIPS 2023）

---

#ai #推荐系统 #生成式推荐 #onerec #semanticid #moe #dpo #偏好对齐 #llm #快手 #人工智能 #ai学习 #大模型

If you like this post, consider star [the repo](https://github.com/sleepingbear-ai/sleepingbear-ai.github.io).
