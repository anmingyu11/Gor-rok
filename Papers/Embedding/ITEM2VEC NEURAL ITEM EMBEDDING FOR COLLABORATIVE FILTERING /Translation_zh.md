> 微软-特拉维夫大学

# ITEM2VEC: NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING 

许多协同过滤(CF) 算法都是基于 item 的，因为它们分析 item - item 的关系以产生 item 之间的相似性。近期，自然语言处理(NLP)领域的一些研究工作提出了利用 neural embedding 算法学习单词的 latent vector。其中，Skip-gram with Negative sampling(SGNS)，也被称为 word2vec，在各种语言学任务中提供了 state-of-art 的结果。在本文中，我们证明了基于 item 的 CF 可以在相同的 neural word embedding 框架下进行转换。受SGNS的启发，我们描述了一种名为 item2vec 的方法，用基于 item 的 CF，生成了在 latent space 中的 item embedding 。即使在用户信息不可用的情况下，该方法也能够推断 item-item 关系。实验结果证明了 item2vec 方法的有效性，并表明该方法与 SVD 具有较好的竞争优势。

**索引词** - skip-gram, word2vec, neural word embedding, collaborative filtering, item similarity, recommender systems, market basket analysis, itemitem collaborative filtering, item recommendations.

## 1. INTRODUCTION AND RELATED WORK

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Fig1.png)

**Fig. 1. Windows 10应用商店中基于与 Need For Speed 类似的 item 的推荐. **

计算 item 相似度是现代推荐系统中的关键组成部分。 虽然许多推荐算法专注于同时学习 user 和 item 的低维 embedding[1,2,3]，计算 item 相似性本身就是目的。 线上 retailers 广泛使用 item 相似性来完成许多不同的推荐任务。 本文讨论了通过在低维空间中 embedding 项 item 来学习项目相似性这一被忽视的任务。

基于 item 的相似性被 online retailers 用于基于单个 item 的推荐。例如，在Windows 10 App Store中，每个 App 或 Game 的详细信息页面都包含一个名为“People also like”的其他类似应用列表。这个列表可以扩展为与原始 App 相似的完整页面推荐列表，如图1所示。类似的推荐列表仅仅基于与单个的 item 的相似性在大多数在线商店中存在，如 Amazon，Netflix，Google Play, iTunes store和许多其他的。

单个 item 推荐与更“传统”的 user-to-item 推荐不同，因为它们通常是在用户对特定 item 具有明确的兴趣的上下文中，以及在明确的 user 购买意图的上下文中显示的。 因此，基于 item 相似性的单个 item 建议通常比  user-to-item 的建议具有更高的点击率（CTR），因此在销售额或收入中所占的比例更高。

基于 item 相似度的单一 item 推荐也可用于其他各种推荐任务：在“candy rank”中，类似 item (通常价格较低) 的推荐会在付款前的结账页面上提出。在 “bundle” 推荐中，一组由几个 item 组成的 item被组合在一起进行推荐。最后，在 online store 中使用 item 相似性，以便更好地探索和发现，并改善整体用户体验。通过为 user 定义松弛变量 (在优化问题中，松弛变量是添加到不等式约束中以将其转换为等式的变量。) 来隐式学习 item 之间的联系的 user-to-item CF 方法，比经过优化以直接学习 item 关系的方法，不可能产生更好的 item 表示。

item 相似性也是基于 item 的 CF 算法的核心，该算法旨在直接从 item - item 关系学习表示[4，5]。有几种场景需要基于 item 的 CF 方法：在大规模数据集中，当 user 数量明显大于 item 数量时，仅对 item 建模的方法的计算复杂度明显低于同时对 user 和 item 建模的方法。例如，在线音乐服务可能有数亿注册用户，只有数万名艺术家(item)。

在某些场景下，user-item 关系不可用。例如，今天的网上购物有很大一部分是在没有明确的用户识别过程的情况下完成的。相反，可用的信息是按 session 计算的。把这些 session 当作“user”对待，不仅代价高得令人望而却步，而且信息量也会更少。

近年来，用于语言任务的 neural embdding 方法的最新进展已极大地提高了NLP技术的先进水平[6、7、8、12]。这些方法试图将单词和短语映射到捕获单词之间语义关系的低维向量空间。其中，也被称为 word2vec [8]的S kip-gram with Negative Sampling (SGNS) 在各种自然语言处理任务中屡创新高[7,8]，其应用已经扩展到自然语言处理之外的其他领域[9,10]。

在本文中，我们建议将 SGNS 应用于基于 item 的CF中。由于 SGNS 在其他领域的巨大成功，我们建议 SGNS 稍加修改就可以捕获协同过滤数据集中不同 item 之间的关系。为此，我们提出了 SGNS 的一个修改版本，命名为 item2vec。我们表明，item2vec可以 induce 一个相似度度量，与使用SVD的item-based-CF相竞争。而将与其他更复杂方法的比较留到未来的研究中。

论文的其余部分组织如下：第 2 节概述了 SGNS 背景下的相关工作，第 3 节描述了如何将 SGNS 应用于 item-based CF，第4节描述了实验设置，并给出了定性和定量的 结果。

#### QA

- 推导SVD

- 词向量可以怎么取

  > 可以取输入向量，也可以取输出向量，也可以取两者相加，或者将两者拼接起来。

- 向量除了计算相似度，还有哪些应用

