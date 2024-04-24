# Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations

## ABSTRACT

在大规模推荐中，一个核心问题是准确高效地检索出最相关的候选项，最好能够在亚线性时间内实现。以前的方法主要是基于两步骤过程：首先学习内积模型，然后使用一些近似最近邻（ANN）搜索算法来找到top候选项。在本文中，我们提出了Deep Retrieval（DR），通过直接使用 user-item 交互数据（例如点击数据），而不依赖ANN算法中的欧式空间假设，来学习一个可检索的结构。DR的结构将所有候选项编码成离散的潜在空间。这些候选项的潜在编码是模型参数，并与其他神经网络参数一起学习以最大化相同的目标函数。在学习了模型之后，对该结构进行 beam search 以检索top候选项进行重新排序。从经验上讲，我们首先证明了DR具有亚线性的计算复杂度，可以在两个公共数据集上实现几乎与蛮力基线相同的准确性。此外，我们还展示了在实际的生产推荐系统中，部署的DR方法在参与度指标方面明显优于经过调优的ANN基线。据我们所知，DR是首批在工业级推荐系统中成功部署的非ANN算法，可应用于数亿个物品的规模。

## 1 INTRODUCTION

在过去几十年中，推荐系统在各种商业应用中取得了巨大成功。这些系统的目标是根据用户特征和历史行为从一个语料库中返回相关的物品。其中一种早期成功的技术是协同过滤（Collaborative Filtering，CF），它基于一个简单的想法：相似的用户可能喜欢相似的物品。基于物品的协同过滤（Item-CF）[26]通过考虑物品与物品之间的相似性扩展了这个想法，为亚马逊的推荐系统[17]奠定了基础。

在互联网时代，流行内容平台上的用户和物品数量迅速增长到数千万甚至数亿。可扩展性、效率和准确性都是现代推荐系统设计中面临的挑战性问题。最近，基于向量的检索方法得到了广泛应用。其主要思想是将 user 和 item 向量化到潜在向量空间中(embedding)，并使用向量的内积来表示用户和物品之间的偏好。代表性的向量 embedding 方法包括矩阵分解（Matrix Factorization，MF）[16, 22]、因子分解机（Factorization Machines，FM）[25]、DeepFM [7]、Field-aware FM（FFM）[14]等。然而，当 item 数量较大时，对所有 item 进行内积的蛮力计算成本是不可能的。因此，在语料库过大时，通常会使用近似最近邻（Approximate Nearest Neighbors，ANN）或最大内积搜索（Maximum Inner Product Search，MIPS）算法来检索 top 相关的物品。高效的MIPS或ANN算法包括基于树的算法[11, 23]、局部敏感哈希（Locality Sensitive Hashing，LSH）[27, 28]、量化积（Product Quantization，PQ）[6, 13]、层次可导航小世界图（Hierarchical Navigable Small World Graphs，HNSW）[18]等。

尽管向量为基础的 ANN 或 MIPS 方法在实际应用中取得了成功，但它们有两个主要的缺点：

1. user 和 item embdding 的内积结构可能不足以捕捉 user-item 交互的复杂结构[10]。
2. ANN 或 MIPS 是为了近似学习到的内积模型而设计的，并没有直接针对 user-item 交互数据进行优化。

作为一种解决方案，也提出了基于树的模型[31–33]来解决这些问题。其中一个潜在的问题是这些方法中每个 item 被映射到树的一个叶子节点，使得树结构本身难以学习。叶子级别上可用的数据可能有限，并且可能无法提供足够的信号来学习具有数亿个 item 的推荐系统中更细粒度的良好树结构。因此，在大规模推荐领域仍然非常需要一个高效且易于学习的检索系统。

在本文中，我们提出了Deep Retrieval（DR），以端到端的方式从数据中学习可检索结构。DR使用如图1a所示的 $K × D$ 矩阵进行索引，我们介绍了不同 $d ∈ \{1, ..., D\}$ 之间的代码的相互依赖关系。在这个结构中，我们将路径 $c$ 定义为正向索引遍历矩阵列的过程。每个路径的长度为 $D$，索引值范围为 $\{1, 2, ..., K\}$。有 $K^D$ 种可能的路径，每个路径可以解释为 item 的一个聚类。我们在给定用户输入的情况下学习了路径的概率分布，并与物品到路径的映射一起进行训练。在服务阶段，我们使用 beam search 来检索最可能的路径和与这些路径相关联的物品。设计DR结构有两个主要特点。 

1. DR的目的是“检索”，而不是排序。由于DR的结构中没有“叶子节点”，路径内的物品在检索目的上是无法区分的，从而减轻了对这些物品进行排序时的数据稀缺问题。DR可以轻松扩展到数亿个 item。 
2. 每个 item 被设计为可以通过多个路径进行索引，这意味着两个 item 可以共享一些路径，但在其他路径上有所不同。这种设计可以通过我们后面将展示的 DR 模型的概率公式自然实现。item 和路径之间的这种多对多编码方案与早期基于树结构的设计中使用的一对一映射方法有很大的区别。 在训练过程中，item 路径也是模型参数，并且使用期望最大化（Expectation-Maximization，EM）类型的算法[4]与结构模型的其他神经网络参数一起进行学习。

**相关研究。** 在这里，我们简要回顾了一些相关的研究，并讨论它们与 DR 的联系。在众多的大规模推荐算法中，与我们最接近的工作是基于树的方法，包括基于树的深度模型（TDM）[32]，JTM [31]，OTM [33]和AttentionXML [30]。基于树的方法将每个 item 映射到树状结构模型中的一个叶节点，并同时学习树结构和模型参数的目标函数。正如上面所讨论的，为数亿个 item 学习一个好的树结构可能很困难。

另一条研究线路尝试在离散潜空间中对候选 item 进行编码。Vector Quantised-Variational AutoEncoder（VQVAE）[29]使用类似 $𝐾 × 𝐷$ 的 embedding 空间来编码项目，但更注重对图像或音频等数据进行编码，而DR则侧重于基于 user-item 交互对 item 进行聚类。HashRec [15] 和 Merged-Averaged Classifiers via Hashing [20] 在大型推荐系统中使用多索引哈希函数来编码候选 item。与HashRec和MACH相比，DR使用了更复杂的结构模型，其中各层之间存在依赖关系，并采用了基于beam search的不同推理算法。

在极端分类中，已经使用了更复杂的结构模型。LTLS [12] 和 W-LTLS [5] 将有向无环图（DAG）用作结构模型。概率分类器链 [3] 也被用作多标签分类的结构模型。需要注意的是，这些极端分类应用中的标签数最多为几万个，而 DR 可应用于包含数亿个 item 的工业推荐系统。

**组织结构**。本文的剩余部分按以下方式组织。

在第2节中，我们详细描述了结构模型及其训练目标函数。然后，我们介绍了一种用于在推断阶段找到候选路径的 beam search 算法。

第3节介绍了EM算法，用于同时训练神经网络参数和 item 的路径。

在第4节中，我们首先展示了DR在两个公开数据集MovieLens-20M1和Amazon books2上的性能。实验结果显示，DR可以以次线性计算复杂度几乎达到暴力搜索的准确性。

此外，在第5节中，我们测试了DR在一个拥有数亿 user 和 item 的实时生产推荐系统上的性能，并展示它在A/B测试中在参与度指标方面明显优于经过调优的ANN基准线。

最后，在第6节中，我们总结了本文并讨论了几个可能的未来研究方向。

## 2 DEEP RETRIEVAL: LEARNING A RETRIEVABLE STRUCTURE FROM DATA

在这一部分中，我们详细介绍DR。首先，我们建立了它的基本的概率公式。然后，我们将其扩展为一种多路径机制，使得DR能够捕捉到物品的多方面特性。接着，我们介绍了一种惩罚设计，以防止将 item 分配给不同路径时出现崩溃。接下来，我们描述了一种用于检索的 beam search 算法。最后，我们介绍了DR与重排模型的多任务联合训练过程。

### 2.1 The DR Model

**基本模型**。基本的 DR 模型由 $D$ 层组成，每一层有 $K$ 个节点。在每一层中，我们使用一个多层感知机（MLP）和 $K$ 类的 softmax 函数，输出对其 $K$ 个节点的分布。设 $\mathcal{V} = \{1, . . . ,𝑉 \}$ 为所有物品的标签，$𝜋 : \mathcal{V} → [𝐾]^𝐷$ 为物品到路径的映射。这里的路径是沿着矩阵列的正向索引遍历，如图1a所示。为简单起见，我们假设在本节中给出了映射 $𝜋$（我们将在第3节中介绍学习映射 $𝜋$ 的算法）。此外，我们暂时假设一个物品只能映射到一个路径，并将多路径设置留给下一节。

给定一个训练样本对 $(𝑥, 𝑦)$，表示 user $𝑥$ 和 item $𝑦$ 之间的正向交互（点击、转化、喜欢等），以及与物品 $𝑦$ 相关联的路径 $𝑐 = (𝑐_1, 𝑐_2, . . . , 𝑐_𝐷)$ ，即 $𝜋(𝑦) = 𝑐$，路径概率 $𝑝(𝑐|𝑥, 𝜃)$ 按照以下方式逐层构建（参见图1b的流程图）：

- 第一层，以用户 embedding $emb(𝑥)$ 作为输入，并基于参数 $𝜃_1$ 输出第一层中 $𝐾$ 个节点的概率 $𝑝(𝑐_1 |𝑥, 𝜃_1)$。 

- 第二层，我们将用户 $emb(𝑥)$ 和所有先前层级的 $emb(𝑐_{𝑑−1})$（称为路径 embedding）连接起来作为MLP的输入，根据参数 $𝜃_𝑑$ 在第 $𝑑$ 层输出关于该层 $𝐾$ 个节点的概率 $𝑝(𝑐_𝑑 |𝑥, 𝑐_1, . . . , 𝑐_{𝑑−1}, 𝜃_𝑑)$。

- 给定用户 $𝑥$ 的路径 $𝑐$ 的概率是所有层级输出概率的乘积：  
  $$
  p(c \mid x, \theta)=\prod_{d=1}^D p\left(c_d \mid x, c_1, \ldots, c_{d-1}, \theta_d\right)
  \\(1)
  $$

给定一组 $𝑁$ 个训练样本 $\left\{\left(x_i, y_i\right)\right\}_{i=1}^N$，结构模型的对数似然函数为： 
$$
Q_{\mathrm{str}}(\theta, \pi)=\sum_{i=1}^N \log p\left(\pi\left(y_i\right) \mid x_i, \theta\right)
$$
第 $𝑑$ 层的输入向量大小是 embedding 大小乘以 $𝑑$，输出向量大小为 $𝐾$。第 $𝑑$ 层的参数大小为 $Θ(𝐾𝑑)$。参数 $𝜃$ 包含所有层的参数 $𝜃_𝑑$ 以及路径 embedding。整个模型的参数数量在 $Θ(𝐾𝐷^2)$ 的数量级上，与当 $𝐷 ≥ 2$ 时可能的路径数 $𝐾^𝐷$ 相比显著较小。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Figure1.png)

**Figure 1**: (a) Consider a structure with width $𝐾 = 100$ and depth $𝐷 = 3$, and an item encoded by a path of [36, 27, 20]. This path denotes that the item is assigned to the (1, 36), (2, 27), (3, 20) indices of the $𝐾 × 𝐷$ matrix. In the figure, arrows with the same color form a path. Different paths could intersect with each other by sharing a common index in a layer. (b) Flow diagram of the process for constructing $𝑝(𝑐|𝑥, 𝜃)$, the probability of path $𝑐 = [𝑐_1, 𝑐_2, 𝑐_3]$ given input $𝑥$.

> 译注：一图胜千言。

**多路径扩展。** 在基于树的深度模型[31, 32]以及我们上面介绍的结构模型中，每个 item 仅属于一个簇/路径，限制了模型在表达真实数据中的多方面信息方面的能力。例如，与烤肉串相关的 item 可以属于“食物”簇。与花朵相关的 item 可以属于“礼物”簇。然而，与巧克力或蛋糕相关的 item 可以属于两个簇，以便推荐给对食物或礼物感兴趣的用户。在现实世界的推荐系统中，一个簇可能没有明确的含义，比如食物或礼物，但这个例子激发了我们将每个 item 分配给多个簇的想法。在 DR 中，我们允许将每个 item $y_i$ 分配给 $J$ 个不同的路径 $\{c_{i,1}, . . . , c_{i,J}\}$。令 $𝜋 : \mathcal{V} → [𝐾]^{𝐷×𝐽}$ 表示 item 到多个路径的映射。多路径结构目标可以直接定义为
$$
Q_{\mathrm{str}}(\theta, \pi)=\sum_{i=1}^N \log \left(\sum_{j=1}^J p\left(c_{i, j}=\pi_j\left(y_i\right) \mid x_i, \theta\right)\right)
$$
其中属于多个路径的概率是属于各个路径的概率的求和。

**路径大小的惩罚。** 直接优化关于 item 到路径映射 $𝜋$ 的目标函数 $Q_{str}(𝜃, 𝜋)$ 可能会导致将不同的 item 分配到相同的路径中（我们在实践中观察到了这一点）。在极端情况下，我们可以将所有 item 都分配到一个单独的路径中，并且给定任何用户 $𝑥$，观察到这个单一路径的概率为 $1$（如公式 1 所示）。

这是因为只有一个可供选择的路径。然而，这样做对于检索前几个候选 item 并没有帮助，因为 item 之间没有区分性。我们必须调整 $𝜋$ 的可能分布以确保多样性。我们引入以下被惩罚的似然函数： 
$$
Q_{\text{pen}}(\theta, \pi)=Q_{\text{str}}(\theta, \pi)-\alpha \cdot \sum_{c \in [K]^D} f(|c|)
$$
其中 $𝛼$ 是惩罚因子，$|𝑐|$ 表示路径 $𝑐$ 中分配的 item 数量，$𝑓$ 是一个递增且凸函数。一个二次函数 $𝑓 (|𝑐|) = |𝑐|^2 /2$ 控制了路径的平均大小，而高阶多项式则更加严厉地对较大的路径进行惩罚。在我们的实验中，我们使用了 $𝑓 (|𝑐|) = |𝑐|^4 /4$。

### 2.2 Beam Search for Inference

在推断阶段，我们希望从 DR 模型中检索 item，以用户 embedding 作为输入。为此，我们使用 beam search 算法 [24] 来检索最有可能的路径。在每一层中，该算法从前一层选择的节点的所有后继节点中选择 top $𝐵$ 个节点。最后，在最终层返回 $𝐵$ 条 top 路径。当 $𝐵 = 1$ 时，这变成了贪婪搜索。在每一层中，从 $𝐾 × 𝐵$ 个候选项中选择前 $𝐵$ 个的时间复杂度为 $𝑂(𝐾𝐵 log 𝐵)$。总体复杂度为 $𝑂(𝐷𝐾𝐵 log 𝐵)$，与 item 总数 $𝑉$ 相比是次线性的。我们将该过程总结如算法 1 所示。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Alg1.png)

### 2.3 Multi-task Learning and Reranking with Softmax Models

DR beam search 返回的 item 数量远小于总 item 数，但通常不够小以满足用户请求，因此我们需要对这些检索到的 item 进行排序。然而，由于DR中的每个路径可以包含多个 item，仅使用DR就很难区分这些 item。为了解决这个问题，我们选择通过联合训练DR模型和重新排序器来处理： 
$$
Q_{\text{softmax}}=\sum_{i=1}^N \log p_{\text{softmax}}\left(y=y_i \mid x_i\right) 
$$
其中 $𝑝(𝑦 = 𝑦_𝑖 |𝑥_𝑖)$ 是一个输出大小为 $𝑉$ 的 softmax 模型。该模型使用采样的 softmax 算法进行训练。最终的目标函数由 $Q = Q_{pen} + Q_{softmax}$ 给出。在进行 beam search 以获取一组候选项后，我们对这些候选项进行重新排序以获得最终的 top 候选项。在这里，我们使用一个简单的 softmax 模型，但我们当然可以用更复杂的模型替换它，以获得更好的排序性能。

## 3 LEARNING WITH THE EM ALGORITHM

在前面的部分中，我们介绍了DR中的结构模型及其需要优化的目标。这个目标是关于神经网络参数 $𝜃$ 连续的，可以通过任何基于梯度的优化器进行优化。然而，涉及到 item 到路径映射 $𝜋$ 的目标是离散的，不能通过基于梯度的优化器进行优化。由于这个映射作为 item 的“潜在聚类”，这激发了我们使用EM风格的算法来联合优化映射和其他参数的想法。

通常来说，DR的EM风格算法可以总结如下。我们从随机初始化映射 $𝜋$ 和其他参数开始。然后在第 $𝑡$ 个迭代中， 

1. E步：对于固定的映射 $𝜋^{(𝑡−1)}$，使用基于梯度的优化器优化参数 $𝜃$，以最大化结构目标函数 $Q_{pen}(𝜃, 𝜋^{(𝑡−1)})$。 

2. M步：更新映射 $𝜋 (𝑡)$，最大化相同的结构目标函数 $Q_{pen}(𝜃, 𝜋)$。 由于E步与任何标准的随机优化类似，我们将在这里重点关注M步。为了简单起见，我们首先考虑没有惩罚项的目标函数 $Q_{str}$。给定一个user-item的训练对 $(𝑥_𝑖 , 𝑦_𝑖)$，令与该 item 相关的路径集合为 $\{𝜋_1 (𝑦_𝑖), . . . , 𝜋_𝐽 (𝑦_𝑖)\}$。

   对于固定的 $𝜃$，我们可以将 $Q_{str}(𝜃, 𝜋)$ 重新写为：
   $$
   \begin{aligned} Q_{\mathrm{str}}(\theta, \pi) & =\sum_{i=1}^N \log \left(\sum_{j=1}^J p\left(\pi_j\left(y_i\right) \mid x_i, \theta\right)\right) \\ & =\sum_{v=1}^V\left(\sum_{i: y_i=v} \log \left(\sum_{j=1}^J p\left(\pi_j(v) \mid x_i, \theta\right)\right)\right) \end{aligned}
   $$
   其中外部求和遍历所有 item $𝑣 ∈ \mathcal{V}$，内部求和遍历训练集中 item $𝑣$ 的所有出现。我们现在考虑最大化目标函数关于所有可能映射 $𝜋$ 的取值。然而，对于一个 item $𝑣$，有 $𝐾^𝐷$ 种可能的路径，因此无法枚举所有 $𝜋_𝑗 (𝑣)$。为了进一步简化问题，我们使用一个上界 $\sum_{i=1}^N \log p_i \leq N\left(\log \sum_{i=1}^N p_i-\log N\right)$ 来得到： 
   $$
   \begin{aligned} & Q_{\text {str }}(\theta, \pi) \leq \bar{Q}_{\text {str }}(\theta, \pi) \\ & =\sum_{v=1}^V\left(N_v \log \left(\sum_{j=1}^J \sum_{i: y_i=v} p\left(\pi_j(v) \mid x_i, \theta\right)\right)-\log N_v\right) \end{aligned}
   $$
   其中 $N_v=\sum_{i=1}^N \mathbb{I}\left[i: y_i=v\right]$ 表示训练集中 item $𝑣$ 出现的次数，该次数与映射无关。我们定义以下得分函数： 
   $$
   s[v, c] \triangleq \sum_{i: y_i=v} p\left(c \mid x_i, \theta\right)
   $$
   直观地说，$𝑠[𝑣, 𝑐]$ 可以理解为将 item $𝑣$ 分配给路径 $𝑐$ 的累积重要性得分。在实践中，不可能保留所有得分，因为可能的路径数 $𝑐$ 是指数级别的，所以我们只通过 beam search 保留一部分具有最高得分的 $𝑆$ 条路径，并将其余的得分设置为 $0$。

**关于使用流式训练估计 $𝑠[𝑣, 𝑐]$ 的说明。** 在实际生产的推荐系统中，由于持续的大量的数据，我们通常需要使用流式训练——仅处理按时间戳排序的数据。因此，基本思想是跟踪一个包含前 $𝑆$ 个重要路径的列表。对于item $𝑣$，假设我们已经记录了得分列表 $𝑠[𝑣, 𝑐_{1:𝑆} ]$。在获取到包含 $𝑣$ 的新的得分列表 $𝑠^′ [𝑣, 𝑐^′_{1:𝑆}]$ 后，我们按如下方式更新得分列表 $𝑠[𝑣, 𝑐_{1:𝑆}]$： 

(1) 设 $min\_score = min_𝑖 \ 𝑠[𝑣, 𝑐_𝑖]$。 

(2) 创建并集集合 $A=c_{1: S} \cup c_{1: S}^{\prime}$ 。 

(3) 对于每个 $c \in A$：  	

​    (a) 如果 $c \in c_{1: S}$ 并且 $𝑐 ∈ 𝑐^′_{1:𝑆}$ ，则设置 $𝑠[𝑣, 𝑐] ← 𝜂 \ 𝑠[𝑣, 𝑐] + 𝑠^′[𝑣, 𝑐]$。     

​    (b) 如果 $𝑐 ∉ 𝑐_{1:𝑆}$ 并且 $𝑐 ∈ 𝑐^′_{1:𝑆}$ ，则设置 $𝑠[𝑣, 𝑐] ← 𝜂$  $min\_score+𝑠^′ [𝑣, 𝑐]$。     

​    (c) 如果 $𝑐 ∈ 𝑐_{1:𝑆}$ 并且 $𝑐 ∉ 𝑐^′_{1:𝑆}$ ，则设置 $𝑠[𝑣, 𝑐] ← 𝜂 \ 𝑠[𝑣, 𝑐]$。 

(4) 从 $𝑐 ∈ 𝐴$ 中选择 $𝑆$ 个最大值的 $𝑠[𝑣, 𝑐]$，形成新的得分向量 $𝑠[𝑣, 𝑐_{1:𝑆} ]$。

在这里，$𝜂$ 是一个衰减因子，用于考虑流式估计，我们在实验中使用 $𝜂 = 0.999$。如图2所示，如果路径（红色）同时出现在记录的列表和新列表中，我们通过折扣滚动求和来更新得分。如果路径（绿色）只出现在记录的列表中而不在新列表中，我们简单地对得分进行折扣处理。如果路径（蓝色）只出现在新列表中而不在记录的列表中，我们在记录的列表中将其得分设为 $min\_score$ 而不是 $0$。这种方法增加了以流式方式探索新路径的可能性，我们发现这对于方法的良好运作非常重要。

使用坐标下降法解决 M 步骤。给定估计的得分 $𝑠[𝑣, 𝑐]$，现在考虑带惩罚项的目标函数 $Q_{pen}$。这导致在 M 步骤中出现以下代理函数， 
$$
\begin{array}{r}
\arg \max _{\left\{\pi_j(v)\right\}_{j=1}^J} \sum_{v=1}^V\left(N_v \log \left(\sum_{j=1}^J s\left[v, \pi_j(v)\right]\right)-\log N_v\right) \\
-\alpha \cdot \sum_{c \in[K]^D} f(|c|) .
\end{array}
$$
注意，对于这个最大化问题没有闭式解。因此，我们利用坐标下降算法，在固定其他 item 的分配的同时，优化 item $𝑣$ 的路径分配。注意到 item $- \log 𝑁_𝑣$ 与 $𝜋_𝑗 (𝑣)$ 无关，因此可以忽略。与 item $𝑣$ 相关的部分目标函数可以简化为 
$$
\arg \max _{\left\{\pi_j(v)\right\}_{j=1}^J} N_v \log \left(\sum_{j=1}^J s\left[v, \pi_j(v)\right]\right)-\alpha \sum_{j=1}^J f\left(\left|\pi_j(v)\right|\right) .
$$
现在我们逐步选择路径分配 $\{𝜋_1 (𝑣), . . . , 𝜋_𝑗 (𝑣)\}$。在第 $𝑖$ 步，通过选择 $𝑐 = 𝜋_𝑖(𝑣)$，目标函数的增益如下： 
$$
\begin{array}{r}
N_v\left(\log \left(\sum_{j=1}^{i-1} s\left[v, \pi_j(v)\right]+s[v, c]\right)-\log \left(\sum_{j=1}^{i-1} s\left[v, \pi_j(v)\right]\right)\right) \\
-\alpha(f(|c|+1)-f(|c|))
\end{array}
$$
通过跟踪部分和 $\sum_{j=1}^{i-1} s\left[v, \pi_j(v)\right]$ 和路径大小 $|𝑐|$，我们贪心地选择具有最大增益的路径。坐标下降算法的详细过程见算法 2。在实践中，进行三到五次迭代足以确保算法收敛。时间复杂度与词汇量 $𝑉$、路径的多样性 $𝐽$ 以及候选路径数 $𝑆$ 线性增长。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Figure2.png)

**Figure 2**: Illustration of the score updating process. $𝑚𝑖𝑛(𝑠)$ is used to increase the exploration of new paths in training. Blocks with the same colors indicate the same paths.

## 4 EXPERIMENTS ON PUBLIC DATASETS

在本节中，我们研究了 DR 在两个公开的推荐数据集上的性能：MovieLens-20M [8] 和 Amazon books [9, 19]。我们将 DR 的性能与暴力算法以及其他几种推荐基线模型进行比较，包括基于树的模型 TDM [32] 和 JTM [31]。在本节末尾，我们还研究了 DR 中重要超参数的作用。

### 4.1 Datasets and Metrics

**MovieLens-20M**. 这个数据集包含了一个名为 MovieLens 的电影推荐服务中的评分和自由文本标记活动。我们使用了 20M 子集，这些子集基于1995年至2015年间 138,493 位用户的行为创建而成。每个用户-电影交互包含一个用户 ID、一个电影 ID、一个介于 1.0 到 5.0 之间的评分，以及一个时间戳。为了进行公平比较，我们完全按照与 TDM 相同的数据预处理过程进行操作。我们只保留评分高于或等于 4.0 的记录，并且只保留至少有十条评价的用户。经过预处理后，数据集包含 129,797 个用户，20,709 部电影和 9,939,873 条交互。然后，我们随机抽样 1,000 个用户及其对应的记录构建验证集，另外 1,000 个用户构建测试集，其他用户构建训练集。对于每个用户，根据时间戳，前一半的评价被用作历史行为特征，后一半的评价被用作待预测的真实值。

**Amazon books**. 这个数据集包含了来自亚马逊的图书用户评论，其中每个用户-图书的交互包含一个用户 ID、一个 item ID 和相应的时间戳。与 MovieLens-20M 类似，我们按照 JTM 的预处理过程进行相同的操作。数据集包含 294,739 个用户，1,477,922 个物品和 8,654,619 条交互。请注意，与 MovieLens-20M 相比，Amazon books 数据集具有更多的物品，但交互稀疏得多。我们随机抽样了 5,000 个用户及其对应的记录作为测试集，另外 5,000 个用户作为验证集，其他用户作为训练集。行为特征和真实值的构建过程与 MovieLens-20M 中相同。

**评估指标**。我们使用 精准率、召回率和 F-metric作为每个算法的性能评估指标。这些指标是针对每个用户单独计算的，并在用户之间无权重地平均，与 TDM 和 JTM 的设置相同。我们在 MovieLens-20M 和 Amazon books 中分别为每个用户检索前 10 个和 200 个 item 来计算这些指标。

**模型和训练**。由于数据集的划分方式使得训练集、验证集和测试集中的用户不相交，我们舍弃了用户 ID，只将行为序列作为 DR 的输入。如果行为序列的长度超过 69，则将其截断到长度为 69，并使用占位符符号填充长度不足的部分。我们使用具有 GRU 的循环神经网络将行为序列投影到固定维度的 embedding 中，作为 DR 的输入。我们采用了多任务学习框架，并通过 softmax 重新排序器对召回路径中的物品进行重新排名。我们在初始的两个 epoch 中同时训练 DR 和 softmax 的 embedding 层，然后固定 softmax 的 embedding 层并再训练 DR 的 embedding 层两个 epoch。这么做的原因是为了防止 softmax 模型过拟合。在推断阶段，由于路径大小的差异，从 beam search 中检索的 item 数量并不固定，但方差并不大。根据经验，我们控制 beam 大小，使得来自 beam search 的 item 数量是最后检索 item 数量的 5 到 10 倍。

![Alg2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Alg2.png)

**Algorithm 2**: Coordinate descent algorithm for penalized path assignment.

我们将DR算法与以下算法进行性能比较：Item-CF [26]、YouTube-product-DNN [2]、TDM和JTM。我们直接使用了TDM和JTM论文中的Item-CF、YouTube DNN、TDM和JTM的数据，以便进行公平比较。在所提出的不同TDM变体中，我们选择了表现最好的那个。

关于JTM的结果只适用于Amazon图书。我们还将DR与暴力检索算法进行比较，该算法直接计算用户 embedding 和所有在softmax模型中学到的 item embedding 之间的内积，并返回前 $K$ 个项目。在实际的大型推荐系统中，暴力算法通常在计算上是禁止的，但对于基于内积的模型，它可以作为小型数据集的上界。

表1显示了DR在MovieLens-20M数据集上与其他算法和暴力检索的性能比较情况。表2则展示了Amazon图书数据集的结果。对于DR和暴力检索，我们独立地训练同样的模型5次，并计算每个指标的均值和标准差。结果如下所示.

- DR在包括TDM和JTM在内的其他方法中表现更好。
- DR的性能与暴力检索方法非常接近，可以将其视为Deep FM和HNSW等基于向量的方法性能的上限。然而，在Amazon图书数据集中，DR的推理速度比暴力检索快4倍（见表3）。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Table1.png)

**Table 1**: Comparison of precision@10, recall@10 and Fmeasure@10 for DR, brute-force retrieval and other recommendation algorithms on MovieLens-20M.

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Table2.png)

**Table 2**: Comparison of precision@200, recall@200 and Fmeasure@200 for DR, brute-force retrieval and other recommendation algorithms on Amazon Books.

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Table3.png)

**Table 3**: Comparison of inference time for DR and bruteforce retrieval on Amazon Books.

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Table4.png)

**Table 4**: Comparison of performance for different model depth $𝐷$ on Amazon Books

![Table5](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Table5.png)

**Table 5**: Relationship between the size of the path with most items (top path size) and penalty factor 𝛼.

### 4.3 Sensitivity of Hyperparameters

DR引入了一些关键的超参数，这些超参数可能会对性能产生重大影响，包括结构模型的宽度 $𝐾$，模型的深度 $𝐷$，路径的数量 $𝐽$，beam大小 $𝐵$ 和惩罚因子 $𝛼$。在MovieLens-20M实验中，我们选择了$𝐾 = 50$，$𝐷 = 3$，$𝐵 = 25$，$𝐽 = 3$和$𝛼 = 3 × 10^{−5}$。在Amazon图书实验中，我们选择了$𝐾 = 100$，$𝐷 = 3$，$𝐵 = 50$，$𝐽 = 3$和$𝛼 = 3 × 10^{−7}$。使用Amazon图书数据集，我们展示了这些超参数的作用，并观察它们如何影响性能。在图3中，我们展示了当这些超参数变化时，召回率@200的变化情况。在变化一个超参数时，我们保持其他超参数的值不变。图4和图5展示了Precision@200和F-measure@200的类似趋势。

- **模型的宽度 $𝐾$** 控制着模型的整体容量。如果 $𝐾$ 较小，所有 item 的聚类数量也较小；如果 $𝐾$ 较大，则训练和推理阶段的时间复杂度会与 $𝐾$ 线性增长。此外，较大的 $𝐾$ 可能会增加过拟合的可能性。应根据语料库的大小选择适当的 $𝐾$。
- **模型的深度 $𝐷$**。使用 $𝐷 = 1$ 显然不是一个好主意，因为它无法捕捉层间的依赖关系。在表4中，我们调查了Amazon图书数据集上 $𝐷 = 2, 3$ 和 $4$ 的结果，并得出以下结论：  
  - 使用相同的 $𝐾$，$𝐷 = 2$ 会降低性能；
  - 使用相同的 $𝐾$，$𝐷 = 4$ 不会改善性能；
  - 具有相同可能路径数的模型（$𝐾 = 1000$，$𝐷 = 2$ 和 $𝐾 = 100$，$𝐷 = 3$）具有类似的性能。然而，参数数量的量级是 $𝐾𝐷^2$，因此更深的模型可以以更少的参数实现相同的性能。作为模型性能和内存使用之间的权衡，我们在所有实验中选择了 $𝐷 = 3$。
- **路径数量 $𝐽$** 能够表达候选 item 的多方面信息。当 $𝐽 = 1$ 时，性能最差，并随着 $𝐽$ 的增加而增加。较大的 $𝐽$ 可能不会影响性能，但训练的时间复杂度会与 $𝐽$ 线性增长。在实践中，建议选择介于3和5之间的 $𝐽$。 
- **波束大小 $𝐵$** 控制要回溯的候选路径数量。较大的 $𝐵$ 导致更好的性能，但推理阶段的计算也更重。需要注意的是，当 $𝐵 = 1$ 时，贪婪搜索是一种特殊情况，其性能比具有较大 $𝐵$ 的 beam serach 差。
- **惩罚因子 $𝛼$** 控制每条路径中的 item 数。当 $𝛼$ 落入某个范围时，可以达到最佳性能。从表5中，我们得出结论，较小的 $𝛼$ 导致路径尺寸较大，从而在重新排序阶段增加了计算量。波束大小 $𝐵$ 和惩罚因子 $𝛼$ 应适当选择，以权衡模型性能和推理速度。 

总体来说，我们可以看到DR对超参数相当稳定，因为有很大的超参数范围可以实现接近最优的性能。

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Figure3.png)

**Figure 3**: Relationship between recall@200 in Amazon Books experiment and model width 𝐾, number of paths 𝐽, beam size 𝐵 and penalty factor 𝛼, respectively.

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Figure4.png)

**Figure 4**: Relationship between precision@200 in Amazon Books experiment and model width 𝐾, number of paths 𝐽, beam size 𝐵 and penalty factor 𝛼, respectively.

![Figure5](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Figure5.png)

**Figure 5**: Relationship between F-measure@200 in Amazon Books experiment and model width 𝐾, number of paths 𝐽, beam size 𝐵 and penalty factor 𝛼, respectively.

![Figure6](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Deep Retrieval Learning A Retrievable Structure for Large-Scale Recommendations/Figure6.png)

**Figure 6**: An illustrative diagram to show how to use DR in an industrial recommendation system.

## 5 LIVE EXPERIMENTS

在本节中，我们展示了 DR 在实际生产系统中的工作原理。据我们所知，DR是第一个成功应用于工业推荐系统规模达数亿个 item 的非深度学习算法之一。

尽管上一节中公共数据集上的结果揭示了DR模型的基本行为，但更重要的是要了解它是否能改善工业环境中真实推荐系统的用户体验。这些系统由于数据量庞大（用户生成内容，UGC）和动态性质而变得更加复杂和难以改进——几乎每秒都会有新的 item 上传。

由于我们系统中的 item 数量通常达到数亿级别，因此进行全面细粒度的排序是一项困难的任务。因此，工业规模的推荐系统通常由几个阶段组成，如YouTube推荐论文[2]所示。至少，系统包括两个阶段：候选生成和细粒度排序。在这里，细粒度排序阶段通常使用更复杂的模型，并且具有候选生成阶段计算上难以处理的其他特征。因此，细粒度排序通常仅处理每个用户请求的数百个 item。DR作为候选生成组件之一起作用。图6显示了这个过程的示意图。在实践中，同时使用多个候选生成源以产生多样化的候选集，供细粒度排序阶段使用。

我们的实验平台是行业内最大的之一，拥有数亿用户和 item。在我们的系统中，item 是短视频。对于这个特定的DR实验，标签是根据用户是否观看完整视频来生成的。基线候选生成方法使用Fieldaware Factorization Machine (FFM)[14]学习 user 和 item的 embedding，并使用HNSW [18]进行近似最近邻搜索。该基线经过广泛调优，以最大程度地提高平台上的用户体验。与在DR的公共数据集上使用softmax进行重新排序略有不同，我们在这里使用的重新排序模型是逻辑回归模型。之前在生产中使用softmax的尝试并没有带来有意义的改进。重新排序模型的架构对HNSW和DR都是相同的。

我们在AB测试中报告了DR的性能结果，如表6所示。与基线模型相比，DR在视频完成率、应用程序浏览时间和第二天留存等关键指标上都取得了显著的增益。所有的改进都具有统计学意义。这可能是因为DR的端到端训练直接从 user-item 交互中将 user embedding 和可检索结构的学习对齐在同一个目标函数下。因此，DR的聚类包含了更多与候选 item 相关的用户行为信息。我们还发现，DR对于不太受欢迎的视频或不太受欢迎的创作者更友好，并且AB测试显示更多这样的视频被推荐给最终用户。这对于平台的创作者生态系统是有益的。我们认为原因如下：在DR结构的每条路径中，item 是不能区分的，只要它们与热门 item 具有一些相似行为，不太受欢迎的 item 就可以被检索出来。最后但并非最不重要的是，DR自然适用于流式训练，并且与HNSW相比，构建用于检索的结构所需的时间要少得多，因为DR的Mstep中没有计算任何 user 或 item 的 embedding。使用多线程CPU实现处理总 item 约需要10分钟。在像我们这样的工业推荐系统中，这些改进是相当大的。这个实验展示了DR在大规模推荐系统中的优势。

## 6 CONCLUSION AND DISCUSSION

在本论文中，我们提出了Deep Retrieval（DR），这是一个可端到端学习的用于大规模推荐系统的结构模型。DR使用EM风格的算法来共同学习模型参数和 item 路径。实验证明，在两个公开的推荐数据集以及拥有数亿 user 和 item 的实际生产环境中，与蛮力基线相比，DR表现出色。基于当前模型设计，存在几个未来研究方向。

首先，目前的结构模型仅基于用户侧信息定义路径上的概率分布。一个有用的想法是如何更直接地将 item 侧信息纳入DR模型中。这可能通过利用额外的 item 特征来增强模型性能。

其次，在结构模型中，我们只考虑了 user 和 item 之间的正向交互，如点击、转化或喜欢。未来的工作还应考虑负向交互，如非点击、不喜欢或取消关注，以改进模型性能。通过纳入这些负向交互，可以更全面地理解用户偏好，并进一步提高模型的效果。

最后，我们目前使用简单的点积类型模型作为重新排序器，而我们计划在未来使用更复杂的模型。通过使用更复杂的重新排序模型，系统可以捕捉到 user 和 item 交互中更细微的差异和模式，从而提高推荐性能。

这些未来研究方向有望推动DR的能力，并提高其在大规模推荐系统中的效果。