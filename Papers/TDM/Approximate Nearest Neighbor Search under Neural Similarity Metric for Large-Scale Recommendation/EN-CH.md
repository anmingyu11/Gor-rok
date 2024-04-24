# Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation

## ABSTRACT

Model-based methods for recommender systems have been studied extensively for years. Modern recommender systems usually resort to 1) representation learning models which define user-item preference as the distance between their embedding representations, and 2) embedding-based Approximate Nearest Neighbor (ANN) search to tackle the efficiency problem introduced by large-scale corpus. While providing efficient retrieval, the embedding-based retrieval pattern also limits the model capacity since the form of user-item preference measure is restricted to the distance between their embedding representations. However, for other more precise user-item preference measures, e.g., preference scores directly derived from a deep neural network, they are computationally intractable because of the lack of an efficient retrieval method, and an exhaustive search for all user-item pairs is impractical.

> 长期以来，针对推荐系统的基于模型的方法已经得到广泛研究。现代推荐系统通常采用以下两种方法：1) 使用表示学习模型来 user-item 偏好，将其表示为 embedding 表示之间的距离；2) 使用基于 embedding 的近似最近邻（ANN）搜索来解决大规模数据集引入的效率问题。虽然 embedding 检索提供了高效的检索功能，但由于 user-item 偏好度量形式受限于 embedding 表示之间的距离，这也限制了模型的容量。然而，对于其他更精确的 user-item 偏好度量方法（例如直接从深度神经网络获得的偏好评分），由于缺乏高效的检索方法，计算上是难以处理的，并且对所有 user-item 对进行穷举搜索是不可行的。

In this paper, we propose a novel method to extend ANN search to arbitrary matching functions, e.g., a deep neural network. Our main idea is to perform a greedy walk with a matching function in a similarity graph constructed from all items. To solve the problem that the similarity measures of graph construction and user-item matching function are heterogeneous, we propose a pluggable adversarial training task to ensure the graph search with arbitrary matching function can achieve fairly high precision. Experimental results in both open source and industry datasets demonstrate the effectiveness of our method. The proposed method has been fully deployed in the Taobao display advertising platform and brings a considerable advertising revenue increase. We also summarize our detailed experiences in deployment in this paper.

> 在本文中，我们提出了一种新颖的方法，将近似最近邻（ANN）搜索扩展到任意匹配函数，例如深度神经网络。我们的主要思想是在从所有 item 构建的相似性图中，利用匹配函数进行贪婪搜索。为了解决图构建的相似性度量和 user-item 匹配函数异质的问题，我们提出了可插拔的对抗训练任务，以确保具有任意匹配函数的图搜索能够达到相当高的精度。在开源数据集和工业数据集上的实验结果表明了我们方法的有效性。该方法已经完全部署在淘宝展示广告平台，并带来了相当可观的广告收入增长。我们还总结了在部署过程中的详细经验。

## 1 INTRODUCTION

Constantly growing amount of available information has posed great challenges to modern recommenders. To deal with the information explosion, modern recommender system is usually designed with a multi-stage cascade architecture that mainly consists of candidate generation and ranking. In the candidate generation stage, also known as matching, thousands of targets are retrieved from a very large corpus, and then, in the ranking stage, these retrieved targets are ranked according to the user’s preference. Notably, given the constraints of computational resources and latency in real-world systems, candidate generation cannot be solved by sequentially scanning the entire corpus while facing a large-scale corpus.

>不断增长的可用信息量对现代推荐系统提出了巨大挑战。为了应对信息爆炸，现代推荐系统通常采用多阶段级联架构，主要包括候选生成和排序两个阶段。在候选生成阶段，也被称为匹配阶段，从一个非常庞大的语料库中检索出成千上万个目标，然后在排序阶段，根据用户的偏好对这些检索出的目标进行排序。值得注意的是，在实际系统中，由于计算资源和延迟的限制，无法通过顺序扫描整个语料库来解决候选生成问题，尤其是面对大规模的语料库时。

To bypass the prohibitive computational cost of scanning the entire corpus, embedding-based retrieval (EBR) has prevailed in recommender systems for years due to its simplicity and efficiency [13, 15]. However, EBR is insufficient to model the complex structure of useritem preferences. Many works have already shown that more complex models usually generalize better [11, 21, 32]. And researchers have striven to develop techniques to tackle the large-scale retrieval problem with more complex models as well. To overcome computation barriers and benefit from arbitrarily advanced models, the idea of regularizing the total computational cost through an index has recently been presented. These methods [8, 33–35] typically have a learnable index and follow the Expectation Maximization (EM) type optimization paradigm, updating between deep model and index alternatively. As a consequence, the deep model, together with beam search, can be leveraged to retrieve relevant items from a large corpus in a sub-linear complexity w.r.t. corpus size. Even though these end-to-end methods can introduce a deep model to large-scale retrieval, there are two aspects that should not be ignored: 1) the joint training of index and model for large-scale data necessitates a costly training budget in terms of both training time and computational resources; 2) the existence of index structure’s internal nodes, such as non-leaf nodes in TDMs [33–35] and path nodes in DR [8], makes it difficult to utilize side-information from items.

> 为了避免扫描整个语料库带来的计算成本过高，embedding-based的检索（EBR）已经在推荐系统中盛行多年，因其简单和高效的特点而受到青睐[13, 15]。然而，EBR无法很好地建模用户-物品偏好的复杂结构。许多研究已经表明，更复杂的模型通常具有更好的泛化性能[11, 21, 32]。研究人员一直努力开发技术来解决使用更复杂模型进行大规模检索的问题。为了克服计算瓶颈并从任意先进的模型中受益，最近提出了通过索引来规范总计算成本的思想。这些方法[8, 33–35]通常具有可学习的索引，并遵循期望最大化（EM）类型的优化范式，在深度模型和索引之间交替更新。因此，深度模型和 beam search 可以以与语料库大小相关的次线性复杂度从大规模语料库中检索相关项。尽管这些端到端方法可以将深度模型引入到大规模检索中，但有两个方面不能忽视：1) 对于大规模数据，索引和模型的联合训练需要昂贵的训练预算，包括训练时间和计算资源；2) 索引结构的内部节点（如TDMs中的非叶节点[33–35]和DR中的路径节点[8]）使得难以利用物品的附加信息。

This work tackles the aforementioned problems by solving largescale retrieval with an arbitrarily advanced model in a lightweight manner, called Neural Approximate Nearest Neighbour Search (NANN). More specifically, we leverage the deep model as a greedy walker to explore a similarity graph constructed after model training. The joint training budget of the end-to-end methods can be greatly released by following the decoupled paradigm. Besides, the similarity graph that the deep model traverses contains no internal nodes, which facilitates the usage of side information from candidate items. To improve the efficiency and effectiveness of graph search, we creatively come up with both a heuristic retrieval method and a auxiliary training task in our NANN framework. The main contributions of our paper are summarized as follows:

> 本研究通过以轻量级的方式使用任意先进的模型来解决大规模检索问题，即神经网络近似最近邻搜索（NANN）。具体而言，我们利用深度模型作为 greedy walker，在模型训练后构建的相似性图中进行探索。采用解耦的范式可以大大释放端到端方法的联合训练预算。此外，深度模型遍历的相似性图不包含内部节点，这有助于利用候选项的附加信息。为了提高图搜索的效率和效果，在我们的NANN框架中创造性地提出了启发式检索方法和辅助训练任务。本文的主要贡献总结如下：

- We present a unified and lightweight framework that can introduce arbitrarily advanced models as the matching function to large-scale ANN retrieval. The basic idea is to leverage similarity graph search with the matching function. 
- To make the computational cost and latency controllable in graph search, we propose a heuristic retrieval method called Beam-retrieval, which can reach better results with fewer computations. And we also propose an auxiliary adversarial task in model training, which can greatly mitigate the effect of heterogeneity between similarity measures and improve the retrieval quality. 
- We conduct extensive experiments on both a publicly accessible benchmark dataset and a real industry dataset, which demonstrate the proposed NANN is an excellent empirical solution to ANN search under neural similarity metric. Besides, NANN has been fully deployed in the Taobao display advertising platform and contributes 3.1% advertising revenue improvements. 
- We describe in detail the hands-on deployment experiences of NANN in Taobao display advertising platform. The deployment and its corresponding optimizations are based on the Tensorflow framework [1]. We hope that our experiences in developing such a lightweight yet effective large-scale retrieval framework will be e helpful to outstretch NANN to other scenarios with ease.

> - 我们提出了一个统一且轻量级的框架，可以将任意先进的模型引入到大规模ANN检索中作为匹配函数。基本思想是利用匹配函数进行相似性图搜索。
> - 为了使图搜索中的计算成本和延迟可控，我们提出了一种启发式检索方法称为Beam-retrieval，它可以在较少的计算量下获得更好的结果。我们还在模型训练中提出了一个辅助对抗任务，可以显著减轻相似性度量异质性的影响，并提高检索质量。
> - 我们在公开可访问的基准数据集和实际行业数据集上进行了广泛的实验，结果表明所提出的NANN是在神经相似性度量下ANN搜索的优秀经验性解决方案。此外，NANN已经完全部署在淘宝展示广告平台上，并带来了3.1%的广告收入改善。
> - 我们详细描述了NANN在淘宝展示广告平台上的实际部署经验。该部署及其相应的优化是基于Tensorflow框架[1]进行的。我们希望我们在开发这样一个轻量级但有效的大规模检索框架方面的经验能够有助于将NANN轻松应用于其他场景。

## 2 RELATED WORK

Hereafter, let $\mathcal{V}$ and $\mathcal{U}$ denote the item set and the user set. In recommendation, we strive to retrieve a set of relevant items $\mathcal{B}_u$ from a large-scale corpus $\mathcal{V}$ for each user $𝑢 ∈ \mathcal{U}$. Mathematically,
$$
\mathcal{B}_u=\underset{v \in \mathcal{V}}{\operatorname{argTopk}} s(v, u)
\\ (1)
$$
where $𝑠(𝑣, 𝑢)$ is the similarity function.

> 下面，让我们用 $\mathcal{V}$ 和 $\mathcal{U}$ 分别表示物品集合和用户集合。在推荐系统中，我们努力为每个用户 $𝑢 ∈ \mathcal{U}$ 从一个大规模的语料库 $\mathcal{V}$ 中检索一组相关的物品 $\mathcal{B}_u$。数学上，我们可以表示为： 
> $$
>  \mathcal{B}_u=\underset{v \in \mathcal{V}}{\operatorname{argTopk}} s(v, u) \\ (1) 
> $$
> 其中 $𝑠(𝑣, 𝑢)$ 是相似度函数。该公式描述了推荐系统的目标，即根据相似度函数 $s(v, u)$，找出与用户 $u$ 相似度最高的前 $k$ 个物品，并将它们作为集合 $\mathcal{B}_u$ 返回。

**Search on graph**. Search on graph popularized by its exceptional efficiency, performance, and flexibility (in terms of similarity function) is a fundamental and powerful approach for NNS. The theoretical foundation to search on graph is the 𝑠-Delaunay graph defined by similarity function $𝑠(𝑣, 𝑢)$. Previous work [20] has shown that $𝑠(𝑣, 𝑢) = −||𝑣 −𝑢||_2$ can find the exact solution to Equation (1), when $𝑘 = 1$ and $𝑢, 𝑣 ∈ \mathbb{R}_𝑑$ , by certain greedy walk on the 𝑠-Delaunay graph constructed from $\mathcal{V}$. More generally, many existing works attempt to extend the conclusion to non-metric cases, such as inner product [2, 22, 24, 25], Mercer kernel [5, 6] and Bregman divergence [3]. In addition, researchers also set foot in approximating the 𝑠-Delaunay graph as the construction of a perfect 𝑠-Delaunay graph with a large corpus is infeasible. Navigable Small World (NSW) [17] is proposed to greatly optimize both graph construction and search process. On top of that, Hierarchical NSW (HNSW) [18] incrementally builds a multi-layer structure from proximity graphs and provides state-of-the-art for NNS. Our approach will resort to HNSW, although other graph-based NNS methods can also work.

> **图搜索**。图搜索以其卓越的效率、性能和灵活性（在相似度函数方面）而广为人知，是最近邻搜索（NNS）的一种基本且强大的方法。图搜索的理论基础是由相似度函数 $𝑠(𝑣, 𝑢)$ 定义的 s-Delaunay图。之前的研究[20]表明，当 $𝑘 = 1$ 并且 $𝑢, 𝑣 ∈ \mathbb{R}_𝑑$ 时，通过从 $\mathcal{V}$ 构建的 s-Delaunay图上的某种贪婪行走，可以找到方程（1）的精确解，其中 $𝑠(𝑣, 𝑢) = −||𝑣 −𝑢||_2$ 。更一般地，许多现有工作试图将这个结论推广到非度量情况，如内积[2, 22, 24, 25]、Mercer核[5, 6]和Bregman散度[3]等。此外，研究人员还尝试近似 s-Delaunay图，因为使用大规模语料库构建完美的 s-Delaunay 图是不可行的。Navigable Small World (NSW) [17] 提出了一种极大地优化图的构建和搜索过程的方法。此外，Hierarchical NSW (HNSW) [18]通过逐步构建多层结构从邻近图中提供了最先进的NNS技术。我们的方法将使用HNSW算法，尽管其他基于图的NNS方法也可以工作。

**Deep model-based retrieval**. Model-based, especially deep model-based methods have been an active topic in large-scale retrieval recently. In recommendation, many works focus on an end-to-end fashion to simultaneously train index and deep model. Tree-based methods, including TDM [34], JTM [33] and BSAT [35],build its index as a tree structure and model user interests from coarse to fine. Deep retrieval (DR) [8] encodes all candidate items with learnable paths and train the item paths along with the deep model to maximize the same objective. These approaches traverse their index to predict user interests and achieve sub-linear computational complexity w.r.t corpus size by beam search. However, these methods usually require additional internal nodes to parametrize the learnable index, which imposes difficulties in using side information of items. Moreover, additional model parameters and training time have to be paid for these end-to-end manners due to the existence of a learnable index and EM-type training paradigm.

> **基于深度模型的检索**。近年来，基于模型，尤其是基于深度模型的方法在大规模检索中成为一个活跃的研究领域。在推荐系统中，许多工作致力于采用端到端的方式同时训练索引和深度模型。树结构方法，包括TDM [34]、JTM [33]和BSAT [35]，将索引构建为一棵树结构，并从粗到细建模用户兴趣。深度检索（DR）[8]使用可学习的路径对所有候选物品进行编码，并通过训练物品路径以及深度模型来最大化相同的目标函数。这些方法通过遍历索引来预测用户兴趣，并通过 beam search 实现了与语料库大小呈亚线性的计算复杂度。然而，这些方法通常需要额外的内部节点来参数化可学习的索引，这在利用物品的附加信息时会带来困难。此外，由于存在可学习的索引和EM类型的训练范式，这些端到端的方法还需支付额外的模型参数和训练时间。

**Search on the graph with deep model** A few works have already tried to extend the similarity function to deep neural networks. The closest work to ours is SL2G [28] which constructs the index graph by l2 distance and traverses the post-training graph with deep neural networks. However, their approach can be only generalized to the $𝑠(𝑣, 𝑢)$ with convexity or quasi-convexity. For the non-convex similarity function (most common case for deep neural network), they apply SL2G directly without adaption. Another work [19] defines the index graph without similarity for item pairs. They exploit the idea that relevant items should have close $𝑠(𝑣, 𝑢)$ for the same user and represent a candidate item by a subsample of $\left\{s\left(v, u_i\right) \mid j=1, \ldots, m\right\}$. However, it is difficult to sample a representative set in practice, especially for large-scale corpus $\mathcal{V}$.

> **基于深度模型的图搜索** 一些工作已经尝试将相似度函数扩展到深度神经网络。与我们的工作最接近的是SL2G[28]，它通过l2距离构建索引图，并使用深度神经网络遍历训练后的图。然而，他们的方法只能推广到具有凸性或拟凸性的 $𝑠(𝑣, 𝑢)$。对于非凸相似度函数（深度神经网络的最常见情况），他们直接应用SL2G而没有进行调整。另一个工作[19]在 item 对之间定义了没有相似度的索引图。他们利用了相关的item 应该对于同一用户具有接近的 $𝑠(𝑣, 𝑢)$的思想，并通过子样本 $\left\{s(v, u_i) \mid j=1, \ldots, m\right\}$ 来表示候选item。然而，在实践中对于大规模语料库 $\mathcal{V}$ 来说，很难采样出一个代表性的集合。

## 3 METHODOLOGY

In this section, we firstly give a general framework about EBR and model-based retrieval in Section 3.1, including model architecture and training paradigm. Then, we introduce the similarity graph construction and graph-based retrieval method respectively for the proposed NANN in Section 3.2 and Section 3.3. Given these preliminary concepts, we accordingly introduce the pluggable adversarial training task and demonstrate how it can eliminate the gap of similarity measures between graph construction and model-based matching function in Section 3.4.

> 在本节中，我们首先在第3.1节中提供了关于EBR和基于模型的检索的总体框架，包括模型架构和训练范式。然后，我们分别在第3.2节和第3.3节介绍了所提出的NANN的相似性图构建和基于图的检索方法。在理解这些初步概念之后，我们在第3.4节中详细介绍可插入的对抗训练任务，并演示它如何消除图构建和基于模型匹配函数之间的相似性度量差距。

### 3.1 General Framework

3.1.1 *Review Embedding Based Retrieval*. Our proposed method can be generally deemed as an extension of the EBR framework where we generalize the simple similarity metrics to arbitrary neural ones. Therefore, we briefly review the EBR framework for clarity. EBR is designed with a two-sided model architecture where one side is to encode the user profile and behaviour sequence, and the other side is to encode the item. Mathematically, 
$$
\mathbf{e}_u=\mathrm{NN}_u\left(\mathbf{f}_u\right), \mathbf{e}_v=\mathrm{NN}_v\left(\mathbf{f}_v\right)
\\(2)
$$
where two deep neural networks $NN_𝑢$ and $NN_𝑣$ (i.e., the user and the item network) encode the inputs of $f_𝑢$ and $f_𝑣$ to the dense vectors $e_𝑢 ∈ \mathbb{R}^𝑑$ and $e_𝑣 ∈ \mathbb{R}_𝑑$ separately. And the user-item preference forms as the inner product of the semantic embedding, i.e. $e^𝑇_𝑢e_𝑣$ . The candidate sampling based criterion such as Noise Contrastive Estimation (NCE) [10] and Sampled-softmax [14] are usually used to train the EBR models due to the computational difficulty to evaluate partition functions by summing over the entire vocabulary of large corpus.

> 3.1.1 *基于 embedding 的检索综述*。我们提出的方法可以被普遍视为EBR框架的扩展，其中我们将简单的相似度度量推广到任意的神经网络模型中。因此，为了清晰起见，我们简要回顾一下EBR框架。EBR采用双塔模型架构，其中一侧用于编码用户的个人资料和行为序列，另一侧用于编码物品。数学上表示为：
> $$
>  \mathbf{e}_u=\mathrm{NN}_u\left(\mathbf{f}_u\right), \mathbf{e}_v=\mathrm{NN}_v\left(\mathbf{f}_v\right) \\(2) 
> $$
> 其中两个深度神经网络 $NN_𝑢$ 和 $NN_𝑣$（即用户网络和物品网络）将$f_𝑢$和$f_𝑣$的输入编码为分别属于$\mathbb{R}^𝑑$的稠密向量 $e_𝑢$ 和 $e_𝑣$。用户-物品偏好形式可以由语义 embedding 的内积表示，即 $e^𝑇_𝑢e_𝑣$。由于计算整个大规模语料库的分区函数的困难性，通常使用基于候选采样的准则，如噪声对比估计（NCE）[10] Sampled-softmax [14]来训练EBR模型。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure1.png)

**Figure 1**: General Framework. In the left part, the deep model contains three basic branches, user network, target attention network, and item network. The user network is responsible for modeling the user’s profile. And we adopt the attention mechanism to flexibly learn the interaction between user behavior sequence and target item. The information of items is learned through the item network. In the right part, we approximate the Delaunay graph defined on $𝑙_2$ distance among $e_𝑣$ (the output of item network) following HNSW. Note that the online inference starts from $e_𝑣$ for item network branch.

> **图1**: 总体框架。在左侧部分，深度模型包含三个基本分支：用户网络、目标注意力网络和物品网络。用户网络负责建模用户的个人资料。我们采用注意力机制来灵活地学习用户行为序列与目标物品之间的交互。物品的信息通过物品网络进行学习。在右侧部分，我们使用HNSW方法近似定义在 $e_𝑣$ 上的 $𝑙_2$ 距离的Delaunay图。请注意，在线推断从 item 网络分支的 $e_𝑣$ 开始。

*3.1.2 Model Architecture*. Compared to the traditional EBR method in large-scale retrieval, NANN greatly outstretches the model capacity by more complex architecture with user network, target attention network, and item network akin to a standard CTR prediction model, as shown in Figure 1. In other words, we substitute the inner product e 𝑇 𝑢 e𝑣 with a more general and expressive $𝑠(𝑣, 𝑢)$. The generalized form $𝑠(𝑣, 𝑢)$ with deep neural network, in turn, poses both theoretical and practical challenges to us: 1) how to generalize the search on the graph-based index to any non-linear and non-convex $𝑠(𝑣, 𝑢)$ reasonably; 2) how to integrate the graph-based index with complex deep model and deploy the computation-intensive retrieval framework in a lightweight and efficient way.

> **3.1.2 模型架构**。与大规模检索中的传统EBR方法相比，NANN通过更复杂的架构，包括用户网络、目标注意力网络和物品网络，大大提升了模型容量，如图1所示。换句话说，我们用一个更普适和表达能力更强的 $𝑠(𝑣, 𝑢)$ 替代了内积 $e^𝑇_𝑢e_𝑣$。然而，基于深度神经网络的广义形式 $𝑠(𝑣, 𝑢)$ 也给我们带来了理论和实践上的挑战：1）如何合理地将基于图的检索推广到任意非线性和非凸的 $𝑠(𝑣, 𝑢)$ ；2）如何将基于图的检索与复杂的深度模型集成，并以轻量高效的方式部署计算密集型的检索框架。

*3.1.3 Training*. Same with EBR, we reduce the computationally intractable problem to the problem of estimating the parameters of a binary classifier by NCE. The positive samples come from the true distribution that user $𝑢$ engages with item $𝑣$, while the negative samples are drawn from a “noise” distribution $𝑞(𝑣)$, e.g., the unigram distribution over $𝑣 ∈ \mathcal{V}$. We denote the corresponding loss function as $L_{𝑁𝐶𝐸}$. Moreover, we extend the search on the graph-based index to any metric $𝑠(𝑣, 𝑢)$ by using an auxiliary task with the loss denoted by $L_{𝐴𝑈𝑋}
$ (details are in Section 3.4). Hence, the overall objective is
$$
\mathcal{L}_{\text {all }}=\mathcal{L}_{N C E}+\mathcal{L}_{A U X}
\\(3)
$$

> **3.1.3 训练**。与EBR相同，我们将计算复杂的问题转化为通过NCE估计二分类器的参数的问题。正样本来自用户 $𝑢$ 与物品 $𝑣$ 之间真实分布的交互，而负样本则从“噪声”分布$𝑞(𝑣)$中抽取，例如在 $\mathcal{V}$ 上的单字分布。我们将相应的损失函数表示为 $L_{𝑁𝐶𝐸}$。此外，我们通过使用一个辅助任务，其损失由$L_{𝐴𝑈𝑋}$表示（详细内容见第3.4节），将基于图的检索推广到任意度量$𝑠(𝑣, 𝑢)$上。因此，整体目标函数为： 
> $$
> \mathcal{L}_{\text {all }}=\mathcal{L}_{N C E}+\mathcal{L}_{A U X} \\(3)
> $$

*3.1.4 Search on post-training similarity graph*. The graph-based index is built from the precomputed item embedding $e_𝑣$ extracted from item network $NN_𝑣$ . In the prediction stage, we traverse the similarity graph in a way that is tailored to both real-world systems and arbitrary $𝑠(𝑣, 𝑢)$.

> **3.1.4 在训练后的相似性图上进行搜索**。基于预计算的 item embedding $e_𝑣$（从物品网络$NN_𝑣$中提取）构建了基于图的索引。在预测阶段，我们以适应实际系统和任意 $𝑠(𝑣, 𝑢)$ 的方式遍历相似性图。

### 3.2 Graph Construction

Search on similarity graphs was originally proposed for metric spaces and extended to the symmetric non-metric scenarios, e.g, Mercer kernel and Maximum Inner Product Search (MIPS). The $𝑠(𝑣, 𝑢)$ can be also generalized to the certain asymmetric case, i.e. Bregman divergence, by exploiting convexity in place of triangle inequality [3]. However, s-Delaunay graph with arbitrary $𝑠(𝑣, 𝑢)$ is not guaranteed to exist or be unique. Furthermore, to construct such s-Delaunay graphs from the large-scale corpus are even computationally prohibitive for both exact and approximate ones. Hence, we follow the way of SL2G [28] to simplify this problem by building the graph index with the item embedding e𝑣 . The graph is defined with $𝑙_2$ distance among $e_𝑣$ and agnostic to $𝑠(𝑣, 𝑢)$. In practice, we build the HNSW graph directly, which is claimed a proper way to approximate the Delaunay graph defined on $𝑙_2$ distance.

> 最初，基于相似性图的搜索是针对度量空间提出的，并扩展到了对称非度量情景，例如Mercer核和最大内积搜索（MIPS）。通过利用凸性代替三角不等式，$𝑠(𝑣, 𝑢)$ 也可以推广到某些非对称情况，例如Bregman散度[3]。然而，对于任意的 $𝑠(𝑣, 𝑢)$，s-Delaunay图不能保证存在或唯一。此外，从大规模语料库构建这样的s-Delaunay图即使在精确和近似的情况下都具有计算上的限制。因此，我们采用SL2G的方法[28]，通过使用 item embedding $e_𝑣$ 来构建图索引。该图是由 $e_𝑣$ 之间的 $𝑙_2$ 距离定义的，并且与 $𝑠(𝑣, 𝑢)$ 无关。在实践中，我们直接构建HNSW图，据称这是一种适当的方法来近似定义在 $𝑙_2$ 距离上的Delaunay图。

### 3.3 Online Retrieval

We equip the original HNSW with beam search and propose a Beam-retrieval to handle the online retrieval in production. With precomputed $e_𝑣$ , the online retrieval stage can be represented as
$$
\mathcal{B}_u=\underset{v \in \mathcal{V}}{\operatorname{argTopk}} s_u\left(\mathbf{e}_v\right)
\\(4)
$$
where $𝑠_𝑢 (.)$ is the user specfic function computed in real time and $e_𝑣$ is the only variable w.r.t $𝑠_𝑢 (.)$ when search on graph-based index.

> 我们通过在原始HNSW上增加beam search的方法，提出了一种称为Beam-retrieval的算法来处理生产环境下的在线检索。利用预计算的$e_𝑣$值，在线检索阶段可以表示为： 
> $$
>  \mathcal{B}_u=\underset{v \in \mathcal{V}}{\operatorname{argTopk}} s_u\left(\mathbf{e}_v\right) \\(4) 
> $$
> 其中，$𝑠_𝑢(.)$ 是实时计算的用户特定函数，而 $e_𝑣$ 是唯一关于 $𝑠_𝑢(.)$ 的可变量，当在基于图的索引上进行搜索时。

The search process of HNSW traverses a hierarchy of proximity graphs in a layer-wise and top-down way, as shown in Algorithm 1. The original HNSW retrieval algorithm referred to as HNSW-retrieval for convenience, employs simple greedy searches where $𝑒𝑓_𝑙(𝑙 > 0)$ in Algorithm 1 is set to 1 at the top layers and assigns a larger value to $𝑒𝑓_0$ to ensure retrieval performance at the ground layer. However, the HNSW-retrieval is practically insufficient to tackle large-scale retrieval in real-world recommender systems since it suffers from the following deficiencies: 1) the subroutine SEARCH − LAYER in HNSW-retrieval explores the graph in a while-loop, which makes the online inference’s computation and latency uncontrollable; 2) the traversal with simple greedy search is more prone to stuck into local optimum, especially for our case where $𝑠_𝑢 (e_𝑣 )$ is usually non-convex. Hence, we reform the SEARCH − LAYER subroutine in HNSW-retrieval according to Algorithm 2. We firstly replace the while-loop with a for-loop to control the prediction latency and the amount of candidates $𝑣$ to evaluate. Despite having an early-stopping strategy, the for-loop can still guarantee the retrieval performance, shown in Figure 7. We secondly break the limits on $𝑒𝑓_𝑙 (𝑙 > 0)$ and enlarge it at top layers to utilize batch computing. Traversal with multiple paths is equivalent to beam search on the similarity graph, which is proved more efficient than the original version demonstrated in Figure 3.

> HNSW的搜索过程以逐层自顶向下的方式遍历一系列近邻图，如算法1所示。为了方便起见，将原始的HNSW检索算法称为HNSW-retrieval。在HNSW-retrieval中，采用简单的贪婪搜索，其中算法1中的 $𝑒𝑓_𝑙(𝑙 > 0)$ 被设置为顶层的值1，并且给 $𝑒𝑓_0$ 赋予较大的值，以保证在底层具有良好的检索性能。然而，在实际的大规模推荐系统中，HNSW-retrieval不足以应对大规模检索问题，因为它存在以下问题：1）HNSW-retrieval中的SEARCH − LAYER子程序使用while循环探索图，使得在线推断的计算和延迟无法控制；2）采用简单贪婪搜索进行遍历更容易陷入局部最优解，特别是在我们的情况下，$𝑠_𝑢 (e_𝑣 )$通常是非凸的。因此，我们根据算法2重新设计了HNSW-retrieval中的SEARCH − LAYER子程序。首先，我们使用for循环替代了while循环，以控制预测延迟和要评估的候选项$𝑣$的数量。尽管有 early stop 策略，但for循环仍然可以保证检索性能，如图7所示。其次，我们打破了$𝑒𝑓_𝑙 (𝑙 > 0)$的限制，并在顶层进行扩大以利用批量计算。使用多条路径进行遍历等效于在相似性图上进行beam search，这被证明比图中原始版本更高效，如图3所示。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Alg1.png)

![Alg2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Alg2.png)

### 3.4 Search with Arbitrary Neural Metric

*3.4.1 Motivation.* When facing arbitrarily models, triangle inequality, symmetry and convexity can no longer be exploited to validate the rationality of similarity graph search with $𝑠_𝑢 (e_𝑣 )$. In practice, the reaction of $𝑠_𝑢 (e_𝑣 )$ to small perturbation of $e_𝑣$ is highly uncertain, e.g., $𝑠_𝑢 (𝑒_𝑣 )$ may fluctuate drastically when $e_𝑣$ is slightly perturbed. Intuitively, this uncertainty plagues the retrieval perforamnce especially when the similarity metrics used in graph construction stage ($l_2$ distance among $e_𝑣$ ) and retrieval stage ($𝑠_𝑢 (e_𝑣 )$) are highly heterogeneous, shown in Table 3. And in this work, we show that retrieval performance can be empirically augmented if we intentionally bias $𝑠_𝑢 (e_𝑣 )$ to avoid uncertainty w.r.t $e_𝑣$ .

> *3.4.1 动机.* 在面对任意模型时，无法再利用三角不等式、对称性和凸性来验证使用 $𝑠_𝑢 (e_𝑣 )$ 进行相似性图搜索的合理性。实际上，当对 $e_𝑣$ 进行微小扰动时，$𝑠_𝑢 (e_𝑣 )$ 的反应是高度不确定的，例如，当稍微扰动 $e_𝑣$ 时，$𝑠_𝑢 (𝑒_𝑣 )$ 可能会剧烈波动。直观上，这种不确定性会严重影响检索性能，特别是当在图构建阶段（例如，$e_𝑣$ 之间的 $l_2$ 距离）和检索阶段（$𝑠_𝑢 (e_𝑣 )$）中使用的相似度度量非常异质时，如表 3 所示。而在本文中，我们展示了通过有意地偏置 $𝑠_𝑢 (e_𝑣 )$ 来避免与 $e_𝑣$ 相关的不确定性，从而可以在实证上增强检索性能。

Our philosophy is based upon an analogy to identifying the local optimum of a differentiable function $𝑓 : \mathbb{R}_𝑑 → \mathbb{R}$ along with a certain direction. Suppose that the solution to $min −𝑠_𝑢 (𝑒_𝑣 )$ is an arbitrary vector defined in $𝑅^𝑑$ , gradient descent and coordinate descent are commonly used to find the local optimum. And we claim that the graph search is analogous to block coordinate descent, of which the update direction is governed by graph structure and top-k procedure instead of gradients. Hence, given the above, we can interpret the uncertainty of $𝑠_𝑢 (e_𝑣 )$ w.r.t $e_𝑣$ as analogous to the flat/sharpness of loss landscape in gradient-based optimization. Although disputable, it is widely thought that "flat minimal" usually generalize better compared to "sharp minimal" [16, 30] because of their robustness to small perturbation of inputs. Earlier works have attempted to change the optimization algorithm to favor flat minimal and find "better" regions [4, 7, 12]. Inspired by these works, we leverage the adversarial training [9, 26, 27, 30, 31] to both mitigate the uncertainty and improve the robustness of arbitrary $𝑠_𝑢 (e_𝑣 )$ w.r.t $e_𝑣$ .

> 我们的理念基于对于一个可微函数 $𝑓 : \mathbb{R}_𝑑 → \mathbb{R}$ 在某个特定方向上寻找局部最优解的类比。假设 $min −𝑠_𝑢 (𝑒_𝑣 )$ 的解是在 $𝑅^𝑑$ 中定义的任意向量，梯度下降和坐标下降是常用的方法来寻找局部最优解。我们声称图搜索类似于块坐标下降，其中更新方向由图结构和前 $k$ 个节点的过程决定，而不是由梯度决定。因此，根据上述，我们可以将 $𝑠_𝑢 (e_𝑣 )$ 相对于 $e_𝑣$ 的不确定性解释为类似于基于梯度优化中损失函数  flat/sharpness of loss landscape。虽然有争议，但普遍认为与“尖锐最小值”相比，“平坦最小值”通常具有更好的泛化性能 [16, 30]，因为它们对输入的微小扰动具有鲁棒性。早期的研究试图改变优化算法以偏好平坦最小值并找到“更好”的区域 [4, 7, 12]。受到这些工作的启发，我们利用对抗性训练 [9, 26, 27, 30, 31] 来减轻 $𝑠_𝑢 (e_𝑣 )$ 相对于 $e_𝑣$ 的不确定性，并提高任意 $𝑠_𝑢 (e_𝑣 )$ 对于 $e_𝑣$ 的鲁棒性。

*3.4.2 Adversarial Gradient Method*. Generally speaking, we resort to the adversarial gradient method and introduce flatness into $𝑠_𝑢 (e_𝑣 )$ in an end-to-end learning-based method [30]. To achieve the robustness of deep neural networks by the defense against adversarial examples has been widely applied to various computer vision tasks in recent years [9, 26, 27, 31]. Adversarial examples refer to normal inputs with crafted perturbations which are usually human-imperceptible but can fool deep neural networks maliciously. The adversarial training utilized in our work is one of the most effective approaches [23, 29] defending against adversarial examples for deep learning. More specifically, we flatten the landscape of $𝑠_𝑢(e_𝑣)$ w.r.t. $e_𝑣$ via training on adversarially perturbed $\tilde{\mathbf{e}_v}$ .

> *3.4.2 Adversarial Gradient Method*. 一般而言，我们采用对抗梯度方法，并在基于端到端学习的方法中引入平坦性到 $𝑠_𝑢 (e_𝑣 )$ [30]。近年来，通过对抗性示例的防御来实现深度神经网络的鲁棒性已被广泛应用于各种计算机视觉任务[9, 26, 27, 31]。对抗性示例是指带有精心设计的扰动的正常输入，这些扰动通常对人类来说不可察觉，但可以恶意欺骗深度神经网络。我们工作中使用的对抗训练是最有效的方法之一[23, 29]，用于对抗性示例进行深度学习的防御。更具体地说，我们通过对对抗性扰动 $\tilde{\mathbf{e}_v}$ 进行训练，使得 $e_𝑣$ 关于 $𝑠_𝑢(e_𝑣)$ 的风景线变得平坦。

In our case, our solutions to maximize $𝑠_𝑢 (e_𝑣)$ are limited to corpus $\mathcal{V}$. Hence, we mainly focus on the landscape of $𝑠_𝑢 (.)$ around each $e_𝑣$ instead of the overall landscape. We formulate the training objective in terms of flatness as follow：
$$
\begin{aligned}
\mathcal{L}_{A U X} & =\sum_u \sum_{v \in \mathcal{Y}_u} s_u\left(\mathbf{e}_v\right) \log \frac{s_u\left(\mathbf{e}_v\right)}{s_u\left(\tilde{\mathbf{e}_v}\right)} \\
\tilde{\mathbf{e}_v} & =\mathbf{e}_v+\Delta
\end{aligned}
\\(5)
$$
where $\mathcal{Y}_𝑢$ consists of the labels from both true distribution and noise distribution for each $𝑢 ∈ \mathcal{U}$ according to NCE.

> 在我们的情况下，我们要最大化 $𝑠_𝑢 (e_𝑣)$ 的解决方案局限于语料库 $\mathcal{V}$。因此，我们主要关注每个 $e_𝑣$ 周围 $𝑠_𝑢 (.)$ 的风景线，而不是整体风景线。我们将训练目标以平坦性表示如下： 
> $$
>  \begin{aligned} \mathcal{L}_{A U X} & =\sum_u \sum_{v \in \mathcal{Y}_u} s_u\left(\mathbf{e}_v\right) \log \frac{s_u\left(\mathbf{e}_v\right)}{s_u\left(\tilde{\mathbf{e}_v}\right)} \\ \tilde{\mathbf{e}_v} & =\mathbf{e}_v+\Delta \end{aligned} \\(5)
> $$
> 其中$\mathcal{Y}_𝑢$根据NCE原理由每个$𝑢 ∈ \mathcal{U}$对应的真实分布和噪声分布的标签组成。

In detail, we generate the adversarial examples by fast gradient sign method (FGSM) [9], which computes the perturbation as:
$$
\Delta=\epsilon \operatorname{sign}\left(\nabla_{\mathbf{e}_v} s_u\left(\mathbf{e}_v\right)\right)
\\(6)
$$
where the $\nabla_{\mathbf{e}_v} s_u\left(\mathbf{e}_v\right)$ stands for the gradient of $s_u\left(\mathbf{e}_v\right)$ w.r.t. $e_𝑣$ that can be easily computed by backpropagation and the max-norm of perturbation $Δ$ is bounded by $𝜖$.

Put simply, we achieve the search with the arbitrary measure without utilizing the convexity of $𝑠_𝑢 (e_𝑣)$. Instead, our framework is built upon the flatness of $𝑠_𝑢 (e_𝑣)$ w.r.t each $e_𝑣$ , which can be achieved with a simple yet effective auxiliary task.

> 具体而言，我们使用快速梯度符号方法（FGSM）[9]生成对抗性示例，该方法计算扰动为： 
> $$
>  \Delta=\epsilon \operatorname{sign}\left(\nabla_{\mathbf{e}_v} s_u\left(\mathbf{e}_v\right)\right) \\(6) 
> $$
> 其中$\nabla_{\mathbf{e}_v} s_u\left(\mathbf{e}_v\right)$表示$s_u\left(\mathbf{e}_v\right)$相对于$e_𝑣$的梯度，可以通过反向传播轻松计算得到，并且扰动$Δ$的最大范数受到$𝜖$的限制。 简单来说，我们在搜索中实现了任意度量，而不利用$𝑠_𝑢 (e_𝑣)$的凸性。相反，我们的框架建立在$𝑠_𝑢 (e_𝑣)$相对于每个$e_𝑣$的平坦性上，这可以通过一个简单而有效的辅助任务来实现。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure2.png)

**Figure 2**: Online Serving System. The NANN service is provided by real-time prediction server, where graph-based index and deep neural network constitute a unified Tensorflow graph. The NANN service receives the user features from real-time feature server and output the retrieved candidate items to downstream task directly.

## 4 SYSTEM IMPLEMENTATION

Figure 2 illustrates the online serving architecture of the proposed method. In general, almost any off-the-shelf inference system, e.g. TensorFlow Serving, for deep neural networks can provide outof-the-box services for NANN. The framework is flexible to use and maintain since we integrate graph-based index with the deep neural network and form a unified Tensorflow graph. The neural network inference and graph-based retrieval of NANN can thus serve as a unified module.

> 图2展示了所提出方法的在线服务架构。一般来说，几乎任何现成的深度神经网络推理系统，例如TensorFlow Serving，都可以为NANN提供即开即用的服务。由于我们将基于图的索引与深度神经网络集成，并形成一个统一的Tensorflow图，因此该框架具有灵活的使用和维护性。因此，NANN的神经网络推理和基于图的检索可以作为一个统一的模块进行服务。

As described in Algorithm 1, the online inference is mainly composed of feed-forward of $𝑠_𝑢 (e_𝑣 )$ and search on the graph, which performs alternatively. For online computation, we place the search on the graph on the Central Processing Unit (CPU) to maintain the flexibility of retrieval, while placing feed-forward of $𝑠_𝑢 (e_𝑣 )$ on the Graphics Processing Unit (GPU) for efficiency. Correspondingly, both the graph-based index and the precomputed $e_𝑣$ are represented as Tensorflow tensor and cached in CPU memory. The host-to-device and device-to-host communications follow the latest Peripheral Component Interconnect Express (PCIe) bus standard. This design can achieve a balance between flexibility and efficiency while just introducing slight communication overheads.

> 如算法1所描述的，在线推断主要由$𝑠_𝑢 (e_𝑣 )$的前向传播和图上的搜索组成，两者交替进行。对于在线计算，我们将图上的搜索放置在中央处理器（CPU）上以保持检索的灵活性，而将$𝑠_𝑢 (e_𝑣 )$的前向传播放置在图形处理器（GPU）上以提高效率。相应地，基于图的索引和预先计算的$e_𝑣$都表示为Tensorflow tensor，并缓存在CPU内存中。主机与设备之间的通信遵循最新的Peripheral Component Interconnect Express（PCIe）总线标准。这种设计可以在灵活性和效率之间取得平衡，同时只引入轻微的通信开销。

Graph representation lays the foundation for online retrieval. In our implementation, each $𝑣 ∈ \mathcal{V}$ is firstly serially numbered and assigned with a unique identifier. The hierarchical structure of HNSW is then represented by multiple Tensorflow RaggedTensors 1 .

Here, we mainly emphasize the online serving efficiency optimizations of our proposed method, which are based on the Tensorflow framework.

> 图表示为在线检索奠定了基础。在我们的实现中，首先对每个$𝑣 ∈ \mathcal{V}$进行串行编号，并分配一个唯一的标识符。然后，HNSW的层次结构由多个Tensorflow RaggedTensors表示。 
>
> 在这里，我们主要强调我们提出的方法的在线服务效率优化，这是基于Tensorflow框架实现的。

## 4.1 Mark with Bitmap

To ensure the online retrieval performance, it is of importance to increase the outreach of candidate items within limited rounds of neighborhood propagation, as shown in Algorithm 2. Hence, we need to mark the visited items and bypass them to traverse further. The idea of Bitmap comes to mind as the $𝑣$ is serially numbered. We invent the Bitmap procedure by building C++ custom operators (Ops) within the Tensorflow framework. We summarize the performance of Bitmap Ops in terms of queries per second (QPS) and response time (RT) in milliseconds in Table 1.

> 为了确保在线检索性能，重要的是在有限的邻域传播轮次内增加候选项的扩展范围，如算法2所示。因此，我们需要标记已访问的 item 并绕过它们以进一步遍历。由于 $𝑣$ 是连续编号的，位图的思想就浮现了出来。我们通过在Tensorflow框架中构建C++自定义操作（Ops）来实现位图过程。我们在表1中以每秒查询数（QPS）和响应时间（RT）（以毫秒为单位）总结了Bitmap Ops的性能。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Table1.png)**Table 1**: Different implementations of “Mark” procedure.

We test the performance of Bitmap Ops with the model architecture of $𝑠_𝑢 (e_𝑣)$ deployed in production, of which detailed configuration will be introduced in Section 5. We traverse a three-layer graphbased index with the $|\mathcal{V}|$ equal to 1,300,000 and tune the parameters in Algorithm 2 to control the number of candidates, roughly 17,000 for the benchmark testing, to be evaluated. As demonstrated Table 1, our custom Bitmap Ops significantly outperform the Tensorflow Raw Set Ops.

> 我们使用生产环境中部署的 $𝑠_𝑢 (e_𝑣)$ 模型架构来测试 Bitmap Ops 的性能，其详细配置将在第5节中介绍。我们遍历了一个三层图形索引，其中$|\mathcal{V}|$等于1,300,000，并调整算法2中的参数来控制要评估的候选人数，大约为17,000个用于基准测试。正如表1所示，我们的自定义Bitmap Ops明显优于Tensorflow原始集合操作（Raw Set Ops）。

### 4.2 Dynamic Shape with XLA

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate the TensorFlow model . XLA can automatically optimize the model execution in terms of speed and memory usage by fusing the individual Tensorflow Ops into coarsen-grained clusters. Our model has achieved a ~3x performance improvement with the help of XLA. However, it requires all tensors of the computation graph to have fixed shapes and compiled codes are specialized to concrete shapes. In our scenario, the number of unvisited items $|\mathcal{𝐶}|$ to be evaluated by $𝑠_𝑢 (e_𝑣)$ is dynamic for each neighborhood propagation in Algorithm 2. Therefore, we present an "auto-padding" strategy to transform the dynamic shapes, e.g., $|\mathcal{𝐶}|$ in Algorithm 2, to certain predefined and fixed shapes. In detail, we set in advance a grid of potential shapes of $|\mathcal{𝐶}|$ and generate compiled codes for these predefined shapes with XLA’s Just-inTime ( JIT ) compilation, which is triggered by replaying the logs from the production environment. For online inference, the "autopadding" strategy automatically pad the tensor with size $|\mathcal{𝐶}|$ to its nearest greater point on the grid and execute efficiently with its corresponding compiled code by XLA, and slice the tensor to its original shape afterward. In short, we extend the capacity of XLA to dynamic shapes with an automatic "padding-slicing" strategy.

> XLA（加速线性代数）是一种针对线性代数的领域特定编译器，可以加速TensorFlow模型的执行。通过将单个TensorFlow操作融合成粗粒度的集群，XLA可以自动优化模型的执行速度和内存使用情况。在我们的模型中，借助XLA的帮助，我们实现了大约3倍的性能提升。然而，它要求计算图中的所有张量具有固定的形状，并且编译的代码是专门针对具体形状的。在我们的场景中，根据算法2中的每次邻居传播，待评估的未访问 item $|\mathcal{𝐶}|$ 的数量是动态的。因此，我们提出了一种“自动填充”策略，将动态形状（例如算法2中的 $|\mathcal{𝐶}|$）转换为预定义的固定形状。具体而言，我们预先设定了一个潜在形状网格，并使用XLA的即时编译（JIT）在这些预定义形状上生成编译代码，该编译由生产环境中的日志重放触发。对于在线推断，"自动填充"策略会将大小为 $|\mathcal{𝐶}|$ 的张量自动填充到网格上最接近的更大点，并通过XLA使用相应的编译代码高效执行，之后再将张量切片回其原始形状。简而言之，我们通过自动的“填充-切片”策略扩展了XLA对动态形状的支持能力。

## 5 EXPERIMENTS

We study the performance of the proposed method as well as present the corresponding analysis in this section. Besides comparison to baseline, we put more emphasis on the retrieval performance of NANN and the corresponding ablation study due to the inadequacies of directly related works. Experiments on both an open-source benchmark dataset and an industry dataset from Taobao are conducted to demonstrate the effectiveness of the proposed method. We observe that our proposed method can significantly outperform the baseline and achieve almost the same retrieval performance as its brute-force counterpart with much fewer computations.

> 在本节中，我们研究了所提出的方法的性能，并进行了相应的分析。除了与基准方法的比较外，由于直接相关工作的不足，我们更加重视NANN的检索性能及其对应的消融研究。我们在一个开源基准数据集和淘宝的一个行业数据集上进行实验，以展示所提出方法的有效性。我们观察到，我们所提出的方法可以显著优于基准方法，并且在计算量明显减少的情况下，几乎达到与其暴力搜索对应的检索性能。

### 5.1 Setup

*5.1.1 Datasets*. We do experiments with two large-scale datasets: 1) a publicly accessible user-item behavior dataset from Taobao called UserBehavior 3 ; 2) a real industry dataset of Taobao collected from traffic logs. Table 2 summarizes the main statistics for these two datasets.

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Table2.png)

**Table 2**: Statistics of evaluation datasets

> *5.1.1 数据集*。我们使用了两个大规模数据集进行实验：1) 一个来自淘宝的公开可访问的用户-物品行为数据集，称为 UserBehavior3；2) 一份来自淘宝流量日志的真实行业数据集。表2总结了这两个数据集的主要统计信息。

**UserBehavior**. UserBehavior is a subset of Taobao user behaviors for recommendation problems with implicit feedback. Each record includes user ID, item ID, item category ID, behavior type, and timestamp. The behavior type indicates how the user interacts with the item, including click, purchase, adding items to the shopping cart, and adding items to favorites. We filter some of the users with high sparsity and keep the users with at least 10 behaviors. Suppose that the behaviors of user $𝑢$ be $(𝑏_{𝑢_1} , . . . , 𝑏_{𝑢_𝑘} , . . . , 𝑏_{𝑢_𝑛} )$, the task is to predict $𝑏_{𝑢_{𝑘+1}}$ based on the preceding behaviors. The validation and test sets are constituted by the samples from randomly selected 10,000 users respectively. We take the $⌈𝑙_𝑢/2⌉$-th ($𝑙_𝑢$ denotes the length of behavior sequence for user $𝑢$) behavior of each $𝑢$ as ground truth and predict it based on all behaviors before.

> **UserBehavior**. UserBehavior是淘宝用户行为的一个子集，用于隐式反馈的推荐问题。每条记录包括用户ID、物品ID、物品类别ID、行为类型和时间戳。行为类型表示用户与物品的交互方式，包括点击、购买、将物品添加到购物车和将物品添加到收藏夹中。我们过滤掉一些稀疏性较高的用户，并保留至少有10个行为的用户。假设用户 $𝑢$ 的行为序列为 $(𝑏_{𝑢_1} , . . . , 𝑏_{𝑢_𝑘} , . . . , 𝑏_{𝑢_𝑛} )$，任务是基于之前的行为来预测 $𝑏_{𝑢_{𝑘+1}}$。验证集和测试集分别由随机选择的10,000个用户的样本组成。我们将每个用户 $𝑢$ 的第$⌈𝑙_𝑢/2⌉$个行为（其中$𝑙_𝑢$表示用户$𝑢$的行为序列长度）作为真值，并基于之前的所有行为进行预测。

**Industrial Data of Taobao**. The industry dataset is collected from the traffic logs in the Taobao platform, which is organized similarly to UserBehavior but with more features and records. The features of the industry dataset are mainly constituted by user profile, user behavior sequence, and item attributes.

> **淘宝的行业数据集**。该行业数据集是从淘宝平台的流量日志中收集而来，其组织方式与UserBehavior类似，但具有更多的特征和记录。该行业数据集的特征主要由用户资料、用户行为序列和物品属性组成。

*5.1.2 Metrics*. We use recall-all@𝑀,recall-retrieval@𝑀,recall-Δ@𝑀, coverage@𝑀 to evaluate the effectiveness of our proposed method. In general, for a user $𝑢$, the recall can be defined as
$$
\operatorname{recall}\left(\mathcal{P}_u, \mathcal{G}_u\right) @ M(u)=\frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|}
$$
where $\mathcal{P}_u\left(\left|\mathcal{P}_u\right|=M\right)$ denotes the set of retrieved items and $\mathcal{G}_𝑢$ denotes the set of ground truths.

The capacity of a trained scoring model $𝑠(𝑣, 𝑢)$ is assessed by exhaustively evaluating the corpus $\mathcal{V}$ for $𝑢 ∈ \mathcal{U}$, namely
$$
\text { recall-all@M(u)= recall }\left(\mathcal{B}_u, \mathcal{G}_u\right) @ M(u)
$$
where $\mathcal{B}_u=\operatorname{argTopk}_{v \in \mathcal{V}} s_u\left(\mathbf{e}_v\right)\left(\left|\mathcal{B}_u\right|=M\right)$ is the set of precisely top-$𝑘$ scored items that can be produced by brute-force scanning.

Suppose that we traverse the graph-based index by $𝑠(𝑣, 𝑢)$ and retrieve relevant items $R_𝑢 (|R_𝑢 | = 𝑀|)$ for each user $𝑢$, the retrieval recall then can be evaluated by,
$$
\text { recall-retrieval@M(u)= recall }\left(\mathcal{R}_u, \mathcal{G}_u\right) @ M(u)
$$

> *5.1.2 评价指标*。我们使用 recall-all@𝑀，recall-retrieval@𝑀，recall-Δ@𝑀 和 coverage@𝑀 来评估我们所提出方法的有效性。通常情况下，对于用户$𝑢$，可以定义召回率如下：
> $$
>  \text { recall }\left(\mathcal{P}_u, \mathcal{G}_u\right) @ M(u)=\frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|} 
> $$
> 其中 $\mathcal{P}_u\left(\left|\mathcal{P}_u\right|=M\right)$ 表示检索到的物品集合，$\mathcal{G}_𝑢$ 表示真实物品集合。 通过对语料库 $\mathcal{V}$ 进行完全评估来评估训练得分模型 $𝑠(𝑣, 𝑢)$ 的能力，即
> $$
> \text { recall-all@M(u)= recall }\left(\mathcal{B}_u, \mathcal{G}_u\right) @ M(u)
> $$
> 其中 $\mathcal{B}_u=\operatorname{argTopk}_{v \in \mathcal{V}} s_u\left(\mathbf{e}_v\right)\left(\left|\mathcal{B}_u\right|=M\right)$ 是通过暴力扫描生成的精确前 $𝑘$ 个得分最高的物品集合。
>
>  假设我们通过 $𝑠(𝑣, 𝑢)$ 遍历基于图的索引，并为每个用户 $𝑢$ 检索相关物品 $R_𝑢 (|R_𝑢 | = 𝑀|)$，则检索召回率可以通过以下方式进行评估： 
> $$
> \text { recall-retrieval@M(u)= recall }\left(\mathcal{R}_u, \mathcal{G}_u\right) @ M(u)
> $$

Correspondingly, the retrieval loss in terms of recall introduced by graph-based index can be defined as,
$$
\text { recall- } \Delta @ M(u)=\frac{\text { recall-all@ } M-\text { recall-retrieval@ } M}{\text { recall-all@ } M}
$$
Furthermore, we make use of coverage@𝑀(𝑢) to describe the discrepancy between the brute-force scanning and the retrieval. Formally,
$$
\text { coverage@M(u) }=\frac{\left|\mathcal{R}_u \cap \mathcal{B}_u\right|}{\left|\mathcal{B}_u\right|}
$$
From now on, we refer to the retrieval quality as the consistency between the items from retrieval and those from the brute-force, measured by recall-Δ@𝑀 and coverage@M.

Finally, we take the average over each $𝑢$ to obtain the final metrics, where $𝑢$ is from the testing set.

> 相应地，基于图索引引入的检索损失可以定义为召回率的差异： 
> $$
>  \text{recall-}\Delta @ M(u) = \frac{\text{recall-all@} M - \text{recall-retrieval@} M}{\text{recall-all@} M}
> $$
>  此外，我们使用 coverage@𝑀(𝑢) 来描述暴力扫描和检索之间的差异。具体地说，
> $$
>  \text{coverage@M(u)} = \frac{\left|\mathcal{R}_u \cap \mathcal{B}_u\right|}{\left|\mathcal{B}_u\right|} 
> $$
> 从现在开始，我们将检索质量指标称为检索结果与暴力扫描结果之间的一致性，通过 recall-Δ@𝑀 和 coverage@M 进行衡量。 最后，我们对每个 $𝑢$ 求平均以得到最终的评价指标，其中 $𝑢$ 来自测试集。

*5.1.3 Model architecture*. The model architecture (denoted as DNN w/ attention) is illustrated in Figure 1, which contains user network, target attention network, item network, and score network. More details are in Appendix. To measure the model capacity and retrieval performance of different model structures, we also conduct experiments on the following model structures: 1) DNN w/o attention, which replaces the target attention network with a simple sum-pooling over the embeddings of user behavior sequence; 2) two-sided, which only consists of user embedding (the concatenation of the output of user network and the sum-pooling over the embeddings of user behavior sequence) and item embedding, and calculate the user-item preference score by inner product.

> *5.1.3 模型结构*。模型结构（标记为带有注意力的DNN）如图1所示，包括用户网络、目标注意力网络、物品网络和评分网络。更多细节请参见附录。为了衡量不同模型结构的模型容量和检索性能，我们还对以下模型结构进行了实验：1）无注意力的DNN（DNN w/o attention），它将目标注意力网络替换为对用户行为序列嵌入进行简单求和池化；2）双边模型（two-sided），它仅包含用户嵌入（用户网络输出和用户行为序列嵌入求和的串联）和物品 embedding，并通过内积计算用户-物品偏好评分。

*5.1.4 Implementation details*. Given the dataset and model structure, we train the model with the loss function defined in Equation (3). Adam optimizer with learning rate 3e-3 is adopted to minimize the loss. The $𝜖$ of FGSM is set to 1e-2 for the industry dataset and 3e-4 for the UserBehavior dataset. We optimize the models by NCE and assign each label from the true distribution with 19 and 199 labels from noise distribution for the industry dataset and UserBehavior dataset respectively.

> *5.1.4 实现细节*。给定数据集和模型结构，我们使用方程（3）中定义的损失函数对模型进行训练。采用学习率为3e-3的Adam优化器来最小化损失。对于工业数据集，FGSM的𝜖设置为1e-2，对于UserBehavior数据集，𝜖设置为3e-4。我们通过NCE对模型进行优化，并将来自真实分布的每个标签分配为工业数据集和UserBehavior数据集分别从噪声分布中抽取的19个和199个标签。

After training, we extract the item feature after the item network for all valid items to build the HNSW graph. The standard index build algorithm [18] is used, the number of established connections is set to 32, size of the dynamic candidate list in the graph construction stage is set to 40.

> 训练完成后，我们提取所有有效 item 的 item 网络之后的 item 特征，用于构建HNSW图。使用标准的索引构建算法[18]，建立的连接数量设置为32，在图构建阶段动态候选列表的大小设置为40。

In the retrieval stage, we exhaustively calculate the scores of items in layer 2, which consists of millesimal items of entire vocabulary and can be scored in one batch efficiently. Then top-k relevant items, with$ k=𝑒 𝑓_2$, are retrieved as enter points for the following retrieval. The default retrieval parameter is set as $\{𝑒𝑓_2, 𝑒𝑓_1, 𝑒𝑓_0\} = \{100, 200, 400\}$, $\{𝑇_2,𝑇_1,𝑇_0\} = \{1, 1, 3\}$, described in Algorithms 1 and 2. Without further claim, we report top-200 ($𝑀 = 200$) metrics for final retrieved items. 

All the hyper-parameter are determined by cross-validation.

> 在检索阶段，我们对第二层中的项目进行全面的分数计算。第二层由整个词汇表的千分之一的项目组成，并且可以高效地以一个批次进行评分。然后，检索出前 $k$ 个相关项目作为下一步检索的入口点，其中 $k=𝑒 𝑓_2$。默认的检索参数设置为$\{𝑒𝑓_2, 𝑒𝑓_1, 𝑒𝑓_0\} = \{100, 200, 400\}$，$\{𝑇_2,𝑇_1,𝑇_0\} = \{1, 1, 3\}$，详见算法1和算法2的描述。如果没有进一步声明，我们将报告最终检索到的项目的top-200 ($𝑀 = 200$)指标。 
>
> 所有超参数都是通过交叉验证确定的。

### 5.2 Results

*5.2.1 Comparison to Baselines.* We compare with the baseline method SL2G, which directly leverages HNSW-retrieval with the deep model in the HNSW graph constructed by $l2$ distance among $e_𝑣$ . The comparison results of different methods are shown in Figure 3. Each $x$-axis stands for the ratio of the number of traversed items to $|\mathcal{V}|$ for reaching the final candidates.

First of all, NANN achieves great improvements on recall and coverage in comparison with SL2G across different numbers of traversed items for the two datasets. Especially, NANN outperforms SL2G by a larger margin when we evaluate a smaller portion of items to reach the final items.

Second, NANN performs on par with its brute-force counterpart by much fewer computations. Especially, NANN hardly plagues retrieval quality and achieves 0.60% recall-Δ and 99.0% coverage with default retrieval parameter when applied to industrial data of Taobao. Moreover, the model capacity and robustness indicated by recall-all can also benefit from the defense against moderate adversarial attacks.

Finally, NANN can rapidly converge, in terms of traversed items, to a promising retrieval performance. As described by the curvatures of Figure 3, only 1% ~2% of V need to be evaluated to reach a satisfying retrieval quality for the two datasets.

> **5.2.1 对比基线** 我们与基线方法SL2G进行比较，SL2G直接利用HNSW图中由$e_𝑣$之间的$l2$距离构建的深度模型进行HNSW检索。不同方法的比较结果如图3所示。每个 $x$ 轴代表达到最终候选项所需遍历的 item 数与$|\mathcal{V}|$之比。 
>
> 首先，在两个数据集中，与SL2G相比，NANN在不同数量的遍历项目上实现了召回率和覆盖率的显著提升。特别是，当我们评估较少部分的 item 以达到最终 item 时，NANN的优势更加明显。 
>
> 其次，NANN凭借更少的计算量与其暴力搜索对应的方法相媲美。尤其是，当应用于淘宝的工业数据时，NANN以默认检索参数实现了0.60%的召回率-Δ和99.0%的覆盖率，并且几乎不会影响检索质量。
>
> 此外，通过抵御中等级别的对抗攻击，模型的容量和鲁棒性（由recall-all指标表示）也会受益。
>
> 最后，NANN可以快速收敛到令人满意的检索性能。如图3的曲率所示，在两个数据集中，只需评估1%~2%的V即可达到令人满意的检索质量。

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure3.png)**Figure 3**: Results of our proposed NANN and SL2G on Industry and UserBehavior dataset.

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure4.png)**Figure 4**: The effect of adversarial gradient training on Industry and UserBehavior dataset.

*5.2.2 Beam-retrieval vs HNSW-retrieval*. Figure 3 demonstrate the recall and coverage for the Beam-retrieval (the “NANN” curve) and the original HNSW-retrieval (the “NANN-HNSW” curve) respectively. As shown in these figures, Algorithm 2 outperforms the HNSW-retrieval version in two ways: 1) it performs consistently better across different numbers of traversed items; 2) it converges to the promising retrieval quality more rapidly. Moreover, as shown in Figure 7, the while-loop of HNSW-retrieval results in redundant rounds of neighborhood propagation in the ground layer which is unnecessary for recall and coverage.

> **5.2.2 Beam-retrieval vs HNSW-retrieval** 图3展示了Beam-retrieval（“NANN”曲线）和原始HNSW-retrieval（“NANN-HNSW”曲线）的召回率和覆盖率。从这些图中可以看出，算法2在两个方面优于HNSW-retrieval版本：1）在不同数量的遍历item上表现更好；2）它更快地收敛到令人满意的检索质量。此外，如图7所示，HNSW-retrieval的while循环导致了在底层进行冗余的邻域传播，这对于召回率和覆盖率而言是不必要的。 

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Table3.png)

**Table 3**: Results of different model architectures on industry and UserBehavior dataset.

![Figure5](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure5.png)

**Figure 5**: Neighborhood propagation in ground layer.

![Figure6](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure6.png)

**Figure 6**: Reaction to small perturbations on the Industry dataset.

![Figure7](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure7.png)

**Figure 7**: Sensitivity Analysis for 𝜖 in FGSM and top-k of ground layer in Beam-retrieval on the Industry dataset.



**5.2.3 Effectiveness of adversarial gradient training**. In Figure 4, we traverse the similarity graph with Beam-retrieval and demonstrate the effectiveness of the defense against adversarial attacks. We observe that NANN is constantly superior to the model without adversarial training across the different degrees of traversal.

> **5.2.3 对抗性梯度训练的有效性**。图4中，我们使用Beam-retrieval遍历相似性图，并展示了对抗性攻击防御的有效性。我们观察到，在不同程度的遍历中，NANN始终优于没有进行对抗性训练的模型。

We also investigate the effects of FGSM on different model architectures. As indicated by recall-all and recall-Δ in Table 3, we empirically show that more complex models usually generalize better and achieve higher performances but may deteriorate the retrieval quality. Based on this observation, we claim that the growing discrepancy between recall-all and recall-retrieval may stem from the higher heterogeneity between similarity measures, and thus exploit adversarial training to mitigate the discrepancy. The default retrieval parameter is used for the comparison. As shown in Table 3, the performance of all model architectures ranging from simple to complex can benefit from the adversarial training; FGSM can greatly improve the retrieval quality, especially for more complex models.

> 我们还研究了FGSM对不同模型架构的影响。如表3所示的recall-all和recall-Δ，我们经验证明，更复杂的模型通常具有更好的泛化性能，但可能会降低检索质量。基于这一观察，我们认为recall-all和recall-retrieval之间日益增大的差距可能源自相似度度量之间的更高异质性，因此利用对抗性训练来减少差距。比较中使用了默认的检索参数。如表3所示，从简单到复杂的所有模型架构的性能都可以从对抗性训练中受益；FGSM可以显著提高检索质量，尤其是对于更复杂的模型。

**5.2.4 Analysis for adversarial gradient training**. Figure 6 shows the reaction of model to adversarial attack after model training. We define the $Δ$ of the adversarial attack as $\epsilon \cdot \operatorname{rand}(-1,1)$ akin to FGSM and compare the robustness of different models by visualizing $|𝑠_𝑢 (e_𝑣 )−𝑠_𝑢 (e_𝑣+Δ)|$. Figure 6 is the histogram of $\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$ where $v \in \operatorname{argTopk}_{v \in \mathcal{V}} s_u\left(\mathbf{e}_v\right)$. As demonstrated in Figure 6, the retrieval quality empirically correlates to the robustness of model when faced with adversarial attack: 1) the greater right-skewed distribution of $\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$ for model without attention demonstrates its superior robustness to model with attention, which is consistent with their recall-Δ in Table 3; 2) the retrieval quality of model with attention can be significantly improved by FGSM, and meanwhile its distribution of $\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$ become more skewed to the right with adversarial training.

> **5.2.4 对对抗性梯度训练的分析**。图6展示了模型在经过训练后对对抗攻击的反应。我们将对抗攻击的$Δ$定义为$\epsilon \cdot \operatorname{rand}(-1,1)$，类似于FGSM，并通过可视化$|𝑠_𝑢 (e_𝑣 )−𝑠_𝑢 (e_𝑣+Δ)|$来比较不同模型的鲁棒性。图6是$\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$的直方图，其中$v \in \operatorname{argTopk}_{v \in \mathcal{V}} s_u\left(\mathbf{e}_v\right)$。如图6所示，当面对对抗性攻击时，检索质量与模型的鲁棒性有经验证明相关：1）没有注意力机制的模型中$\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$的右偏分布更大，表明其对带有注意力机制的模型具有更好的鲁棒性，这与表3中的recall-Δ结果一致；2）通过FGSM可以显著提高带有注意力机制的模型的检索质量，并且通过对抗性训练，$\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$的分布向右偏斜更多。

**5.2.5 Sensitivity analysis.** Magnitude of $𝜖$. Figure 7(a) shows the correlation between $𝜖$ and retrieval quality measured by coverage. In general, the retrieval quality is positively correlated with the magnitude of $𝜖$. Besides, adversarial attacks can be beneficial to the overall performance measured by recall-all with a mild magnitude of $𝜖$, but harmful when $𝜖$ gets excessively large. Hence, the magnitude of $𝜖$ plays an important role in the balance between retrieval quality and overall performance.

Different top-k. Figure 7(b) shows the effects of our proposed method on different $𝑘$ for the final top-k retrieved items in Algorithm 1. NANN performs consistently well across different $𝑘$. The retrieval quality can be still guaranteed despite retrieving with larger $𝑘$. Therefore, our method is insensitive to $𝑘$ in general.

> **5.2.5 敏感性分析** $𝜖$ 的大小。图7(a)展示了$𝜖$和以覆盖率衡量的检索质量之间的相关性。一般来说，检索质量与$𝜖$的大小呈正相关。此外，在$𝜖$适度大小范围内，对抗性攻击可以对以recall-all衡量的整体性能有益，但当$𝜖$过大时，会产生不良影响。因此，$𝜖$的大小在平衡检索质量和整体性能之间起着重要作用。
>
> **不同的top-k值**。图7(b)展示了我们提出的方法对于算法1中最终检索到的前k个项目中不同k值的影响。NANN在不同的k值下表现依然良好。即使是使用较大的k值进行检索，也可以保证检索质量。因此，我们的方法对k值不敏感。

### 5.3 Online Results

Our proposed method is evaluated with real traffic in the Taobao display advertising platform. The online A/B experiments are conducted on main commercial pages within Taobao App, such as the "Guess What You Like" page, and last more than one month. The online baseline is the latest TDM method with Bayes optimality under beam search [33–35]. For a fair comparison, we only substitute TDM, one of the channels in the candidate generation stage, with NANN and maintain other factors like the number of candidate items that delivered to the ranking stage unchanged. Two common metrics for online advertising are adopted to measure online performance: Click-through Rate (CTR) and Revenue per Mille (RPM).
$$
\mathrm{CTR}=\frac{\# \text { of clicks }}{\# \text { of impressions }}, \mathrm{RPM}=\frac{\text { Ad revenue }}{\# \text { of impressions }} \times 1000
$$

> 我们在淘宝展示广告平台的实际流量上评估了我们提出的方法。在线A/B实验在淘宝App的主要商业页面上进行，例如“猜你喜欢”页面，并持续了一个多月。在线基准是最新的TDM方法，采用基于 beam search 的贝叶斯优化[33-35]。为了公平比较，我们只将候选生成阶段中的一个通道TDM替换为NANN，并保持其他因素（如传递到排序阶段的候选项数量）不变。采用了两个常见的在线广告指标来衡量在线性能：点击率（CTR）和千次展示收入（RPM）。
> $$
> \mathrm{CTR}=\frac{\# \text { of clicks }}{\# \text { of impressions }}, \mathrm{RPM}=\frac{\text { Ad revenue }}{\# \text { of impressions }} \times 1000
> $$

NANN significantly contributes up to 2.4% CTR and 3.1% RPM promotion compared with TDM, which demonstrates the effectiveness of our method in both user experience and business benefit.

Moreover, the efficient implementation of NANN introduced in Section 4 facilitates us to benefit from NANN without sacrificing the RT and QPS of online inference. In production, NANN meets the performance benchmark displayed in Table 1. Now, NANN has been fully deployed and provides the online retrieval service entirely in the Taobao display advertising platform.

> 与TDM相比，NANN在点击率（CTR）和千次展示收入（RPM）方面显著提升了2.4%和3.1%，这证明了我们方法在用户体验和商业效益方面的有效性。
>
> 此外，在第4节介绍的高效实现使我们能够从NANN中受益，而不会牺牲在线推理的响应时间和每秒请求数。在生产环境中，NANN满足了表1中显示的性能基准。目前，NANN已经完全部署，并在淘宝展示广告平台上提供在线检索服务。

## 6 CONCLUSION

In recent years, there has been a tendency to tackle the large-scale retrieval problem with deep neural networks. However, these methods usually suffer from the additional training budget and difficulties in using side information from target items because of the learnable index. We propose a lightweight approach to integrating post-training graph-based index with the arbitrarily advanced model. We present both heuristic and learning-based methods to ensure the retrieval quality: 1) our proposed Beam-retrieval can significantly outperform the existing search on graph method under the same amount of computation; 2) we inventively introduce adversarial attack into large-scale retrieval problems to benefit both the retrieval quality and model robustness. Extensive experimental results have already validated the effectiveness of our proposed method. In addition, we summarize in detail the hands-on practices of deploying NANN in Taobao display advertising where NANN has already brought considerable improvements in user experience and commercial revenues. We hope that our work can be broadly applicable to domains beyond recommender system such as web search and content-based image retrieval. In the future, we hope to further uncover the underlying mechanisms that govern the applicability of adversarial attacks to large-scale retrieval problems.

> 近年来，使用深度神经网络解决大规模检索问题的趋势日益增多。然而，这些方法通常在额外的训练成本和使用目标项的辅助信息方面存在困难，因为可学习索引的存在。我们提出了一种轻量级的方法，将事后训练的基于图的索引与任意先进的模型集成在一起。为了保证检索质量，我们提出了启发式方法和基于学习的方法：1）我们提出的Beam-retrieval在相同计算量下明显优于现有的图搜索方法；2）我们在大规模检索问题中创造性地引入对抗性攻击，以改善检索质量和模型的鲁棒性。广泛的实验证明了我们提出方法的有效性。此外，我们详细总结了在淘宝展示广告中部署NANN的实践经验，在用户体验和商业收入方面取得了显著的改进。我们希望我们的工作不仅适用于推荐系统等领域，也可以广泛应用于网页搜索和基于内容的图像检索等其他领域。未来，我们希望进一步揭示对抗性攻击对大规模检索问题适用性的潜在机制。

## 7 ACKNOWLEDGEMENTS

We sincerely appreciate Huihui Dong, Zhi Kou, Jingwei Zhuo, Xiang Li and Xiaoqiang Zhu for their assistance with the preliminary research. We thank Yu Zhang, Ziru Xu, Jin Li for their insightful suggestions and discussions. We thank Kaixu Ren, Yuanxing Zhang, Siran Yang, Huimin Yi, Yue Song, Linhao Wang, Bochao Liu, Haiping Huang, Guan Wang, Peng Sun and Di Zhang for implementing the key components of the training and serving infrastructure.

## REFERENCES

[1] Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, et al. 2016. Tensorflow: Large-scale machine learning on heterogeneous distributed systems. arXiv preprint arXiv:1603.04467 (2016). 

[2] Yoram Bachrach, Yehuda Finkelstein, Ran Gilad-Bachrach, Liran Katzir, Noam Koenigstein, Nir Nice, and Ulrich Paquet. 2014. Speeding up the xbox recommender system using a euclidean transformation for inner-product spaces. In Proceedings of the 8th ACM Conference on Recommender systems. 257–264. 

[3] Lawrence Cayton. 2008. Fast nearest neighbor retrieval for bregman divergences. In Proceedings of the 25th international conference on Machine learning. 112–119. 

[4] Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina. 2019. Entropy-sgd: Biasing gradient descent into wide valleys. Journal of Statistical Mechanics: Theory and Experiment 2019, 12 (2019), 124018. 

[5] Ryan R Curtin and Parikshit Ram. 2014. Dual-tree fast exact max-kernel search. Statistical Analysis and Data Mining: The ASA Data Science Journal 7, 4 (2014), 229–253. 

[6] Ryan R Curtin, Parikshit Ram, and Alexander G Gray. 2013. Fast exact max-kernel search. In Proceedings of the 2013 SIAM International Conference on Data Mining. SIAM, 1–9. 

[7] Guillaume Desjardins, Karen Simonyan, Razvan Pascanu, et al. 2015. Natural neural networks. Advances in neural information processing systems 28 (2015).

[8] Weihao Gao, Xiangjun Fan, Chong Wang, Jiankai Sun, Kai Jia, Wenzhi Xiao, Ruofan Ding, Xingyan Bin, Hui Yang, and Xiaobing Liu. 2020. Deep Retrieval:Learning A Retrievable Structure for Large-Scale Recommendations. arXiv preprint arXiv:2007.07203 (2020). 

[9] Ian J Goodfellow, Jonathon Shlens, and Christian Szegedy. 2014. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572 (2014). 

[10] Michael Gutmann and Aapo Hyvärinen. 2010. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 297–304. 

[11] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th international conference on world wide web. 173–182. 

[12] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Flat minima. Neural computation 9, 1 (1997), 1–42. 

[13] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. 2020. Embeddingbased retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2553–2561. 

[14] Eric Jang, Shixiang Gu, and Ben Poole. 2016. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144 (2016). 

[15] Chao Li, Zhiyuan Liu, Mengmeng Wu, Yuchi Xu, Huan Zhao, Pipei Huang, Guoliang Kang, Qiwei Chen, Wei Li, and Dik Lun Lee. 2019. Multi-interest network with dynamic routing for recommendation at Tmall. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2615–2623. 

[16] Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, and Tom Goldstein. 2017. Visualizing the loss landscape of neural nets. arXiv preprint arXiv:1712.09913 (2017). 

[17] Yury Malkov, Alexander Ponomarenko, Andrey Logvinov, and Vladimir Krylov. 2014. Approximate nearest neighbor algorithm based on navigable small world graphs. Information Systems 45 (2014), 61–68. 

[18] Yu A Malkov and Dmitry A Yashunin. 2018. Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE transactions on pattern analysis and machine intelligence 42, 4 (2018), 824–836. 

[19] Stanislav Morozov and Artem Babenko. 2019. Relevance Proximity Graphs for Fast Relevance Retrieval. arXiv preprint arXiv:1908.06887 (2019). 

[20] Gonzalo Navarro. 2002. Searching in metric spaces by spatial approximation. The VLDB Journal 11, 1 (2002), 28–46. 

[21] Qi Pi, Guorui Zhou, Yujing Zhang, Zhe Wang, Lejian Ren, Ying Fan, Xiaoqiang Zhu, and Kun Gai. 2020. Search-based user interest modeling with lifelong sequential behavior data for click-through rate prediction. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management. 2685–2692. 

[22] Parikshit Ram and Alexander G Gray. 2012. Maximum inner-product search using cone trees. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining. 931–939. 

[23] Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. 2019. Adversarial training for free! arXiv preprint arXiv:1904.12843 (2019). 

[24] Anshumali Shrivastava and Ping Li. 2014. Asymmetric LSH (ALSH) for sublinear time maximum inner product search (MIPS). arXiv preprint arXiv:1405.5869 (2014). 

[25] Anshumali Shrivastava and Ping Li. 2015. Asymmetric minwise hashing for indexing binary inner products and set containment. In Proceedings of the 24th international conference on world wide web. 981–991. 

[26] Ashish Shrivastava, Tomas Pfister, Oncel Tuzel, Joshua Susskind, Wenda Wang, and Russell Webb. 2017. Learning from simulated and unsupervised images through adversarial training. In Proceedings of the IEEE conference on computer vision and pattern recognition. 2107–2116. 

[27] Ayan Sinha, Zhao Chen, Vijay Badrinarayanan, and Andrew Rabinovich. 2018. Gradient adversarial training of neural networks. arXiv preprint arXiv:1806.08028 (2018). 

[28] Shulong Tan, Zhixin Zhou, Zhaozhuo Xu, and Ping Li. 2020. Fast item ranking under neural network based measures. In Proceedings of the 13th International Conference on Web Search and Data Mining. 591–599. 

[29] Yisen Wang, Xingjun Ma, James Bailey, Jinfeng Yi, Bowen Zhou, and Quanquan Gu. 2021. On the convergence and robustness of adversarial training. arXiv preprint arXiv:2112.08304 (2021). 

[30] Zhewei Yao, Amir Gholami, Qi Lei, Kurt Keutzer, and Michael W Mahoney. 2018. Hessian-based analysis of large batch training and robustness to adversaries. Advances in Neural Information Processing Systems 31 (2018). 

[31] Xiaoyong Yuan, Pan He, Qile Zhu, and Xiaolin Li. 2019. Adversarial examples: Attacks and defenses for deep learning. IEEE transactions on neural networks and learning systems 30, 9 (2019), 2805–2824. 

[32] Guorui Zhou, Xiaoqiang Zhu, Chenru Song, Ying Fan, Han Zhu, Xiao Ma, Yanghui Yan, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1059–1068.

[33] Han Zhu, Daqing Chang, Ziru Xu, Pengye Zhang, Xiang Li, Jie He, Han Li, Jian Xu, and Kun Gai. 2019. Joint optimization of tree-based index and deep model for recommender systems. Advances in Neural Information Processing Systems 32 (2019). 

[34] Han Zhu, Xiang Li, Pengye Zhang, Guozheng Li, Jie He, Han Li, and Kun Gai. 2018. Learning tree-based deep model for recommender systems. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1079–1088. 

[35] Jingwei Zhuo, Ziru Xu, Wei Dai, Han Zhu, Han Li, Jian Xu, and Kun Gai. 2020. Learning optimal tree models under beam search. In International Conference on Machine Learning. PMLR, 11650–11659.