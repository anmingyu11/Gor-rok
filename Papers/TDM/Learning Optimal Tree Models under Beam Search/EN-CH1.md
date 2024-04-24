# Learning Optimal Tree Models under Beam Search

## Abstract

Retrieving relevant targets from an extremely large target set under computational limits is a common challenge for information retrieval and recommendation systems. Tree models, which formulate targets as leaves of a tree with trainable node-wise scorers, have attracted a lot of interests in tackling this challenge due to their logarithmic computational complexity in both training and testing. Tree-based deep models (TDMs) and probabilistic label trees (PLTs) are two representative kinds of them. Though achieving many practical successes, existing tree models suffer from the training-testing discrepancy, where the retrieval performance deterioration caused by beam search in testing is not considered in training. This leads to an intrinsic gap between the most relevant targets and those retrieved by beam search with even the optimally trained node-wise scorers. We take a first step towards understanding and analyzing this problem theoretically, and develop the concept of Bayes optimality under beam search and calibration under beam search as general analyzing tools for this purpose. Moreover, to eliminate the discrepancy, we propose a novel algorithm for learning optimal tree models under beam search. Experiments on both synthetic and real data verify the rationality of our theoretical analysis and demonstrate the superiority of our algorithm compared to state-of-the-art methods.

> 在计算资源受限的情况下从一个极大的目标数据集合中检索相关目标是信息检索和推荐系统面临的共同挑战。树模型将目标表达为可训练的非叶子结点和叶子结点组成的打分器，由于其在训练和测试中的对数计算复杂度，因此吸引了很多关注来应对这一挑战。TDMs 和 PLTs 是其中两种典型的模型。尽管取得了许多实际上的成功，但现有的树模型存在训练-测试差异的问题，在训练过程中未考虑测试中由于 beam search 导致的检索能力下降。这导致了最相关目标与通过 beam search 检索到的目标之间存在固有差距，即使使用最优的节点评分器也无法消除此差距。我们首次迈出了理论上理解和分析这个问题的一步，并提出了 beam search 下的贝叶斯最优性和 beam search 的校准概念作为通用的分析工具。此外，为了消除这种差异，我们提出了一种学习 beam search 下最优树模型的新算法。在合成数据和真实数据上的实验证实了我们理论分析的合理性，并证明了我们的算法相比最先进的方法的优越性。

## 1. Introduction

Extremely large-scale retrieval problems prevail in modern industrial applications of information retrieval and recommendation systems. For example, in online advertising systems, several advertisements need to be retrieved from a target set containing tens of millions of advertisements and presented to a user in tens of milliseconds. The limits of computational resources and response time make models, whose computational complexity scales linearly with the size of target set, become unacceptable in practice. 

> 在现代信息检索和推荐系统的工业应用中，广泛存在超大规模检索问题。例如，在在线广告系统中，需要从一个包含数千万个广告的目标集合中检索出多个广告，并在几十毫秒内展示给用户。计算资源和响应时间的限制使得那些计算复杂度与目标集合大小成线性关系的模型在实践中变得不可接受。

Tree models are of special interest to solve these problems because of their ability in achieving logarithmic complexity in both training and testing. Tree-based deep models (TDMs) (Zhu et al., 2018; 2019; You et al., 2019) and Probabilistic label trees (PLTs) (Jasinska et al., 2016; Prabhu et al., 2018; Wydmuch et al., 2018) are two representative kinds of tree models. These models introduce a tree hierarchy in which each leaf node corresponds to a target and each non-leaf node defines a pseudo target for measuring the existence of relevant targets on the subtree rooted at it. Each node is also associated with a node-wise scorer which is trained to estimate the probability that the corresponding (pseudo) target is relevant. To achieve logarithmic training complexity, a subsampling method is leveraged to select logarithmic number of nodes on which the scorers are trained for each training instance. In testing, beam search is usually used to retrieve relevant targets in logarithmic complexity. 

> 树模型对于解决这些问题特别有意义，因为它们在训练和测试中都可以达到对数复杂度。基于树结构的深度模型（TDMs）和概率标签树（PLTs）是两种有代表性的树模型。这些模型引入了一个树层次结构，其中每个叶节点对应一个目标，每个非叶节点定义了一个伪目标，用于衡量其子树上存在相关目标的程度。每个节点还与一个节点打分器相关联，该打分器被训练以估计相应（伪）目标的相关概率。为了实现对数级的训练复杂度，使用下采样方法选择要对每个训练实例选择对数个数的节点来训练打分器。在测试中，通常使用 beam search 来以对数复杂度检索相关目标。

As a greedy method, beam search only expands parts of nodes with larger scores while pruning other nodes. This character achieves logarithmic computational complexity but may result in deteriorating retrieval performance if ancestor nodes of the most relevant targets are pruned. An ideal tree model should guarantee no performance deterioration when its node-wise scorers are leveraged for beam search. However, existing tree models ignore this and treat training as a separated task to testing: (1) Node-wise scorers are trained as probability estimators of pseudo targets which are not designed for optimal retrieval; (2) They are also trained on subsampled nodes which are different to those queried by beam search in testing. Such discrepancy makes even the optimal node-wise scorers w.r.t. training loss can lead to suboptimal retrieval results when they are used in testing to retrieve relevant targets via beam search. To the best of our knowledge, there is little work discussing this problem either theoretically or experimentally. 

> 作为一种贪心法，beam search 仅展开得分较高的部分节点，同时剪枝其他节点。这种性质确保了对数级的计算复杂度，但如果最相关目标的祖先节点被修剪，则可能导致检索性能下降。理想的树模型应该在使用节点评分器进行beam search 时不会导致性能下降。
>
> 然而，现有的树模型忽略了这一点，并将训练视为与测试分离的任务：（1）节点评分器被训练为估计非最优检索的伪目标的概率；（2）它们还在与测试中的 beam search 不同的子采样节点上进行训练。这种差异使得即使针对训练损失而言最优的节点评分器在测试中通过 beam search 检索相关目标时也可能导致次优的结果。据我们所知，目前很少有基于理论或实验证明这个问题的工作。

We take a first step towards understanding and resolving the training-testing discrepancy on tree models. To analyze this formally, we develop the concept of Bayes optimality under beam search and calibration under beam search as the optimality measure of tree models and corresponding training loss, respectively. Both of them serve as general analyzing tools for tree models. Based on these concepts, we show that neither TDMs nor PLTs are optimal, and derive a sufficient condition for the existence of optimal tree models as well. We also propose a novel algorithm for learning such an optimal tree model. Our algorithm consists of a beam search aware subsampling method and an optimal retrieval based definition of pseudo targets, both of which resolve the training-testing discrepancy. Experiments on synthetic and real data not only verify the rationality of our newly proposed concepts in measuring the optimality of tree models, but also demonstrate the superiority of our algorithm compared to existing state-of-the-art methods.

> 我们迈出了解决树模型的训练和测试差异的第一步。为了形式化地分析这个问题，我们引入了 beam search 下的贝叶斯最优性和校准作为树模型和相应训练损失的最优度量。这两个概念都可以作为树模型的通用分析工具。基于这些概念，我们证明了 TDMs 和 PLTs 都不是最优的，并推导出最优树模型存在的充分条件。我们还提出了一种新的算法来学习这样一个最优的树模型。我们的算法包括一个考虑束搜索的下采样方法和一种基于最优检索的伪目标定义，这两种方法解决了训练和测试之间的差异。在合成数据和真实数据上的实验不仅验证了我们新提出的概念在衡量树模型最优性方面的合理性，还证明了我们的算法相对于现有最先进方法的优越性。

## 2.Related Work

**Tree Models**: Research on tree models1 has mainly focused on formulating node-wise scorers and the tree structure. For node-wise scorers, linear models are widely adopted (Jasinska et al., 2016; Wydmuch et al., 2018; Prabhu et al., 2018), while deep models (Zhu et al., 2018; 2019; You et al., 2019) become popular recently. For the tree structure, apart from the random tree (Jasinska et al., 2016), recent works propose to learn it either via hierarchical clustering over targets (Wydmuch et al., 2018; Prabhu et al., 2018; Khandagale et al., 2019) or under a joint optimization framework with node-wise scorers (Zhu et al., 2019). Without dependence on specific formulations of node-wise scorers or the tree structure, our theoretical findings and proposed training algorithm are general and applicable to these advances. 

**Bayes Optimality and Calibration**: Bayes optimality and calibration have been extensively investigated on flat models (Lapin et al., 2017; Menon et al., 2019; Yang & Koyejo, 2019), and they have also been used to measure the performance of tree models on hierarchical probability estimation (Wydmuch et al., 2018). However, there is a gap between the performance on hierarchical probability estimation and that on retrieving relevant targets, since the former ignores beam search and corresponding performance deterioration. As a result, how to measure the retrieval performance of tree models formally remains an open question. We fill this void by developing the concept of Bayes optimality under beam search and calibration under beam search. 

**Beam Search in Training**: Formulating beam search into training to resolve the training-testing discrepancy is not a new idea. It has been extensively investigated on structured prediction models for problems like machine translation and speech recognition (Daume III & Marcu ´ , 2005; Xu & Fern, 2007; Ross et al., 2011; Wiseman & Rush, 2016; Goyal et al., 2018; Negrinho et al., 2018). Though performance deterioration caused by beam search has been analyzed empirically (Cohen & Beck, 2019), it still lacks a theoretical understanding. Besides, little effort has been made to understand and resolve the training-testing discrepancy on tree models. We take a first step towards studying these problems both theoretically and experimentally.

> **树模型**：对树模型的研究主要集中在节点评分器和树结构的建模上。对于节点评分器，广泛采用线性模型，而近年来深度模型变得流行起来。对于树结构，除了随机树，最近的研究提出通过层次聚类目标或在节点评分器的联合优化框架下学习树结构。不依赖于特定的节点评分器或树结构的具体形式，我们的理论发现和提出的训练算法是通用的，并适用于这些进展。
>
> **贝叶斯最优性和校准**：贝叶斯最优性和校准在一般模型上得到了广泛的研究（，并且它们也被用于衡量树模型在层次概率估计上的性能（Wydmuch等，2018）。然而，在层次概率估计和相关目标检索之间存在差距，因为前者忽略了 beam search 和相应的性能恶化。因此，如何形式上衡量树模型的目标检索性能仍然是一个开放的问题。我们通过在 beam search 下发展贝叶斯最优性和校准的概念来填补这一空白。
>
> **训练中的波束搜索**：将 beam search 纳入训练以解决训练-测试差异并不是一个新思路。它已经在结构化预测模型（如机器翻译和语音识别）上进行了广泛的研究。尽管通过波束搜索导致的性能恶化已经经验性地进行了分析，但仍然缺乏理论上的理解。此外，在理解和解决树模型上的训练-测试差异方面几乎没有任何进展。我们首先从理论和实验两方面对这些问题进行研究。

## 3. Preliminaries

### 3.1. Problem Definition

Suppose $\mathcal{I} = \{1, ..., M\}$ with $M \gg 1$ is a target set and $\mathcal{X}$ is an observation space, we denote an instance as $(\mathbf{x}, \mathcal{I}_x)$, implying that an observation $\mathbf{x} ∈ \mathcal{X}$ is associated with a subset of relevant targets $\mathcal{I}_\mathbf{x} ⊂ \mathcal{I}$, which usually satisfies $|\mathcal{I}_\mathbf{x}| \ll M$. For notation simplicity, we introduce a binary vector $\mathbf{y} ∈ \mathcal{Y} = \{0, 1\}^M$ as an alternative representation for $\mathcal{I}_x$, where $y_j = 1$ implies $j ∈ \mathcal{I}_\mathbf{x}$ and vice versa. As a result, an instance can also be denoted as $(\mathbf{x}, \mathbf{y}) ∈ \mathcal{X} × \mathcal{Y}$. 

> 假设 $\mathcal{I} = \{1, ..., M\}$，其中 $M \gg 1$ 是目标集合，$\mathcal{X}$ 是观测空间。我们将一个实例表示为 $(\mathbf{x}, \mathcal{I}_\mathbf{x})$，表示观测 $\mathbf{x} \in \mathcal{X}$ 与相关目标的子集 $\mathcal{I}_\mathbf{x} \subset \mathcal{I}$ 相关联，通常满足 $|\mathcal{I}_\mathbf{x}| \ll M$。为了简化表示，我们引入二进制向量$\mathbf{y} \in \mathcal{Y} = \{0, 1\}^M$ 作为 $\mathcal{I}_x$ 的替代表示，其中 $y_j = 1$ 意味着 $j \in \mathcal{I}_\mathbf{x}$，反之亦然。因此，一个实例也可以表示为 $(\mathbf{x}, \mathbf{y}) \in \mathcal{X} \times \mathcal{Y}$。

Let $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ be a probability density function for data which is unknown in practice, we slightly abuse notations by regarding an instance $(\mathbf{x}, \mathbf{y})$ as either the random variable pair w.r.t. $p(\mathbf{x}, \mathbf{y})$ or a sample of $p(\mathbf{x}, \mathbf{y})$. We also assume the training dataset $\mathcal{D}_{t r}$ and the testing dataset $\mathcal{D}_{t e}$ to be the sets containing i.i.d. samples of $p(\mathbf{x}, \mathbf{y})$. Since $\mathbf{y}$ is a binary vector, we use the simplified notation $\eta_j(\mathbf{x})=p\left(y_j=1 \mid \mathbf{x}\right)$ for any $j \in \mathcal{I}$ in the rest of this paper. 

> 设 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 是一个在实践中未知的数据概率密度函数，我们稍微滥用符号，将实例$(\mathbf{x}, \mathbf{y})$视为关于$p(\mathbf{x}, \mathbf{y})$的随机变量对或$p(\mathbf{x}, \mathbf{y})$的一个样本。我们还假设训练数据集$\mathcal{D}_{tr}$和测试数据集$\mathcal{D}_{te}$是包含$p(\mathbf{x}, \mathbf{y})$的独立同分布样本的集合。由于$\mathbf{y}$是一个二进制向量，在本文的其余部分中，我们使用简化的记法$\eta_j(\mathbf{x})=p(y_j=1 \mid \mathbf{x})$来表示任意$j \in \mathcal{I}$。

Given these notations, the extremely large-scale retrieval problem is defined as to learn a model $\mathcal{M}$ such that its retrieved subset for any $\mathbf{x} \sim p(\mathbf{x})$, denoted by either $\hat{\mathcal{I}}_{\mathbf{x}}$ or $\hat{\mathbf{y}}$, is as close as $\mathbf{y} \sim p(\mathbf{y} \mid \mathbf{x})$ according to some performance metrics. Since $p(\mathbf{x}, \mathbf{y})$ is unknown in practice, such a model is usually learnt as an estimator of $p(\mathbf{y} \mid \mathbf{x})$ on $D_{tr}$ and its retrieval performance is evaluated on $D_{te}$.

>在这些符号的基础上，极大规模的检索问题被定义为学习一个模型$\mathcal{M}$，使得对于任意$\mathbf{x} \sim p(\mathbf{x})$，其检索到的子集，用 $\hat{\mathcal{I}}_{\mathbf{x}}$ 或 $\hat{\mathbf{y}}$ 表示，根据某些性能度量指标与 $\mathbf{y} \sim p(\mathbf{y} \mid \mathbf{x})$ 尽可能接近。由于实际中 $p(\mathbf{x}, \mathbf{y})$ 是未知的，因此这样的模型通常作为 $p(\mathbf{y} \mid \mathbf{x})$ 的估计器在 $D_{tr}$ 上进行学习，并在 $D_{te}$ 上评估其检索性能。

### 3.2. Tree Models

Suppose $\mathcal{T}$ is a $b$-arity tree with height $H$, we regard the node at the $0$-th level as the root and nodes at the $H$-th level as leaves. Formally, we denote the node set at $h$-th level as $\mathcal{N}_h$ and the node set of $\mathcal{T}$ as $\mathcal{N}=\bigcup_{h=0}^H \mathcal{N}_h$. For each node $n ∈ \mathcal{N}$ , we denote its parent as $\rho(n) \in \mathcal{N}$ , its children set as $\mathcal{C}(n) \subset \mathcal{N}$, the path from the root to it as $Path(n)$, and the set of leaves on its subtree as $\mathcal{L}(n)$.

> 假设 $\mathcal{T}$ 是一个高度为 $H$ 的 $b$ 元树，我们将第 $0$ 层的节点视为根节点，第 $H$ 层的节点视为叶子节点。形式上，我们将第 $h$ 层的节点集合表示为 $\mathcal{N}_h$，将 $\mathcal{T}$ 的节点集合表示为 $\mathcal{N}=\bigcup_{h=0}^H \mathcal{N}_h$。对于每个节点$n \in \mathcal{N}$，我们用 $\rho(n) \in \mathcal{N}$ 表示其父节点，用 $\mathcal{C}(n) \subset \mathcal{N}$ 表示其子节点集合，用 $Path(n)$ 表示从根节点到 $n$ 的路径，用 $\mathcal{L}(n)$ 表示其子树上的叶子节点集合。

Tree models formulate the target set $\mathcal{I}$ as leaves of $\mathcal{T}$ through a bijective mapping $π : \mathcal{N}_H → \mathcal{I}$, which implies $H = O(log_bM)$. For any instance $(\mathbf{x}, \mathbf{y})$, each node $n ∈ \mathcal{N}$ is defined with a pseudo target $z_n ∈ \{0, 1\}$ to measure the existence of relevant targets on the subtree of $n$, i.e.,
$$
z_n=\mathbb{I}\left(\sum_{n^{\prime} \in \mathcal{L}(n)} y_{\pi\left(n^{\prime}\right)} \geq 1\right)
\\(1)
$$
which satisfies $z_n = y_{π(n)}$ for $n ∈ \mathcal{N}_H$.

> 通过一个双射映射 $π : \mathcal{N}_H → \mathcal{I}$，树模型将目标集合 $\mathcal{I}$ 表示为 $\mathcal{T}$ 的叶子节点，这意味着$H = O(log_bM)$。对于任意实例 $(\mathbf{x}, \mathbf{y})$，每个节点 $n ∈ \mathcal{N}$ 都定义了一个伪目标 $z_n ∈ \{0, 1\}$ 来衡量在 $n$ 的子树上是否存在相关目标，即， 
> $$
> z_n=\mathbb{I}\left(\sum_{n^{\prime} \in \mathcal{L}(n)} y_{\pi\left(n^{\prime}\right)} \geq 1\right)
> \\(1)
> $$
> 其中对于 $n ∈ \mathcal{N}_H$，满足 $z_n = y_{π(n)}$ 。

By doing so, tree models transform the original problem of estimating $p(y_j |\mathbf{x})$ to a series of hierarchical subproblems of estimating $p(z_n|\mathbf{x})$ on $n ∈ Path(π^{−1}(j))$. They introduce the node-wise scorer $g : \mathcal{X} × \mathcal{N} → \mathbb{R}$ to build such a node-wise estimator for each $n ∈ \mathcal{N}$ , which is denoted as $p_g(z_n|\mathbf{x})$ to distinguish from the unknown distribution $p(z_n|\mathbf{x})$. In the rest of this paper, we denote a tree model as $M(\mathcal{T} , g)$ to highlight its dependence on $\mathcal{T}$ and $g$.

> 通过这样做，树模型将原始的估计问题 $p(y_j |\mathbf{x})$ 转化为一系列层次子问题，即在 $n ∈ Path(π^{−1}(j))$ 上估计 $p(z_n|\mathbf{x})$。它们引入了逐节点评分函数 $g : \mathcal{X} × \mathcal{N} → \mathbb{R}$ 来为每个 $n ∈ \mathcal{N}$ 构建一个逐节点的估计器，该估计器用 $p_g(z_n|\mathbf{x})$ 表示，以区别于未知的分布 $p(z_n|\mathbf{x})$。在本文的其余部分，我们将树模型表示为 $M(\mathcal{T} , g)$，以强调其对 $\mathcal{T}$ 和 $g$ 的依赖关系。

#### 3.2.1. TRAINING OF TREE MODELS

The training loss of tree models can be written as $\operatorname{argmin}_g \sum_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}_{t r}} L(\mathbf{y}, \mathbf{g}(\mathbf{x}))$ , where
$$
L(\mathbf{y}, \mathbf{g}(\mathbf{x}))=\sum_{h=1}^H \sum_{n \in \mathcal{S}_h(\mathbf{y})} \ell_{\mathrm{BCE}}\left(z_n, g(\mathbf{x}, n)\right)
\\(2)
$$
In Eq. (2), $\mathbf{g}(\mathbf{x})$ is a vectorized representation of $\{g(\mathbf{x}, n) : n ∈ \mathcal{N} \}$ (e.g., level-order traversal), $l_{BCE}(z, g) = −z log(1 + exp(−g)) − (1 − z) log(1 + exp(g))$ is the binary cross entropy loss and $\mathcal{S}_h(y) ⊂ \mathcal{N}_h$ is the set of subsampled nodes at $h$-th level for an instance $(\mathbf{x}, \mathbf{y})$. Let $C = max_h |\mathcal{S}_h(y)|$, the training complexity is $O(HbC)$ per instance, which is logarithmic to the target set size $M$.

As two representatives of tree models, PLTs and TDMs adopt different ways3 to build $p_g$ and $S_h(y)$.

> 树模型的训练损失可以表示为 $\operatorname{argmin}_g \sum_{(\mathbf{x}, \mathbf{y}) \sim \mathcal{D}_{t r}} L(\mathbf{y}, \mathbf{g}(\mathbf{x}))$，其中 
> $$
> L(\mathbf{y}, \mathbf{g}(\mathbf{x}))=\sum_{h=1}^H \sum_{n \in \mathcal{S}_h(\mathbf{y})} \ell_{\mathrm{BCE}}\left(z_n, g(\mathbf{x}, n)\right)
> \\(2)
> $$
> 在公式 (2) 中，$\mathbf{g}(\mathbf{x})$ 是 $\{g(\mathbf{x}, n) : n ∈ \mathcal{N} \}$ 的向量化表示（例如，按层次遍历），$l_{BCE}(z, g) = −z log(1 + exp(−g)) − (1 − z) log(1 + exp(g))$ 是二分类的交叉熵损失函数，$\mathcal{S}_h(y) ⊂ \mathcal{N}_h$ 是实例 $(\mathbf{x}, \mathbf{y})$ 在第 $h$ 层的下采样节点集合。令 $C = max_h |\mathcal{S}_h(y)|$，训练复杂度是每个实例的 $O(HbC)$，这对于目标集大小 $M$ 是对数级别的。 作为树模型的两个代表，PLTs 和 TDMs 采用不同的方式构建 $p_g$ 和 $S_h(y)$。

**PLTs**: Since $p\left(z_n \mid \mathbf{x}\right)$ can be decomposed as $p\left(z_n=1 \mid \mathbf{x}\right)= \prod_{n^{\prime} \in \operatorname{Path}(n)} p\left(z_{n^{\prime}}=1 \mid z_{\rho\left(n^{\prime}\right)}=1, \mathbf{x}\right)$ according to Eq. (1), $p_g\left(z_n \mid \mathbf{x}\right)$ is decomposed accordingly via 
$$
p_g\left(z_{n^{\prime}} \mid z_{\rho\left(n^{\prime}\right)}=\right. 1, \mathbf{x})=1 /\left(1+\exp \left(-\left(2 z_{n^{\prime}}-1\right) g\left(\mathbf{x}, n^{\prime}\right)\right)\right)
$$
As a result, only nodes with  $z_{\rho(n)}=1$ are trained, which produces $\mathcal{S}_h(\mathbf{y})=\left\{n: z_{\rho(n)}=1, n \in \mathcal{N}_h\right\}$

**TDMs**: Unlike PLTs, $p(z_n|\mathbf{x})$ is estimated directly via $p_g\left(z_n \mid \mathbf{x}\right)=1 /\left(1+\exp \left(-\left(2 z_n-1\right) g(\mathbf{x}, n)\right)\right)$. Besides, the subsample set4 is chosen as $\mathcal{S}_h(\mathbf{y})=\mathcal{S}_h^{+}(\mathbf{y}) \bigcup \mathcal{S}_h^{-}(\mathbf{y})$ where $\mathcal{S}_h^{+}(\mathbf{y})=\left\{n: z_n=1, n \in \mathcal{N}_h\right\}$ and $\mathcal{S}_h^{-}(\mathbf{y})$ contains several random samples over $\mathcal{N}_h \backslash \mathcal{S}_h^{+}(\mathbf{y})$.

> PLTs（Path Learning Trees）: 根据公式 (1)，$p\left(z_n \mid \mathbf{x}\right)$ 可以分解为 $p\left(z_n=1 \mid \mathbf{x}\right)= \prod_{n^{\prime} \in \operatorname{Path}(n)} p\left(z_{n^{\prime}}=1 \mid z_{\rho\left(n^{\prime}\right)}=1, \mathbf{x}\right)$，因此，通过以下方式对 $p_g\left(z_n \mid \mathbf{x}\right)$ 进行相应的分解：
> $$
> p_g\left(z_{n^{\prime}} \mid z_{\rho\left(n^{\prime}\right)}=\right. 1, \mathbf{x})=1 /\left(1+\exp \left(-\left(2 z_{n^{\prime}}-1\right) g\left(\mathbf{x}, n^{\prime}\right)\right)\right)
> $$
> 结果是只有当 $z_{\rho(n)}=1$ 时，才会进行训练，从而产生 $\mathcal{S}_h(\mathbf{y})=\left\{n: z_{\rho(n)}=1, n \in \mathcal{N}_h\right\}$。 
>
> TDMs（Tree Dropout Models）: 与 PLTs 不同，通过 $p_g\left(z_n \mid \mathbf{x}\right)=1 /\left(1+\exp \left(-\left(2 z_n-1\right) g(\mathbf{x}, n)\right)\right)$ 直接估计 $p(z_n|\mathbf{x})$。此外，子采样集合选择为 $\mathcal{S}_h(\mathbf{y})=\mathcal{S}_h^{+}(\mathbf{y}) \bigcup \mathcal{S}_h^{-}(\mathbf{y})$，其中 $\mathcal{S}_h^{+}(\mathbf{y})=\left\{n: z_n=1, n \in \mathcal{N}_h\right\}$，而 $\mathcal{S}_h^{-}(\mathbf{y})$ 包含从 $\mathcal{N}_h \backslash \mathcal{S}_h^{+}(\mathbf{y})$ 中随机抽取的若干样本。

#### 3.2.2. TESTING OF TREE MODELS

For any testing instance $(\mathbf{x}, \mathbf{y})$, let $\mathcal{B}_h(\mathbf{x})$ denote the node set at $h$-th level retrieved by beam search and $k=\left|\mathcal{B}_h(\mathbf{x})\right|$ denote the beam size, the beam search process is defined as 
$$
\mathcal{B}_h(\mathbf{x}) \in \underset{n \in \tilde{\mathcal{B}}_h(\mathbf{x})}{\arg \operatorname{Topk}} p_g\left(z_n=1 \mid \mathbf{x}\right)
\\(3)
$$
where $\tilde{\mathcal{B}}_h(\mathbf{x})=\bigcup_{n^{\prime} \in \mathcal{B}_{h-1}(\mathbf{x})} \mathcal{C}\left(n^{\prime}\right)$.

> 对于任意的测试实例 $(\mathbf{x}, \mathbf{y})$，假设 $\mathcal{B}_h(\mathbf{x})$ 表示通过 Beam Search 检索得到的第 $h$ 层的节点集合，并且 $k=\left|\mathcal{B}_h(\mathbf{x})\right|$ 表示 Beam 大小（即保留的节点数），则 Beam Search 过程可以定义为： 
> $$
> \mathcal{B}_h(\mathbf{x}) \in \underset{n \in \tilde{\mathcal{B}}_h(\mathbf{x})}{\arg \operatorname{Topk}} p_g\left(z_n=1 \mid \mathbf{x}\right)
> \\(3)
> $$
>  其中 $\tilde{\mathcal{B}}_h(\mathbf{x})=\bigcup_{n^{\prime} \in \mathcal{B}_{h-1}(\mathbf{x})} \mathcal{C}\left(n^{\prime}\right)$，表示通过将上一层的节点 $n^{\prime}$ 映射到其对应的子节点集合 $\mathcal{C}\left(n^{\prime}\right)$，构建当前层的候选节点集合。然后，从这个候选节点集合中选择概率 $p_g\left(z_n=1 \mid \mathbf{x}\right)$ 最高的前 $k$ 个节点作为当前层的节点集合 $\mathcal{B}_h(\mathbf{x})$。

By applying Eq. (3) recursively until $h = H$, beam search retrieves the set containing $k$ leaf nodes, denoted by $\mathcal{B}_H(\mathbf{x})$. Let $m \leq k$ denote the number of targets to be retrieved, the retrieved target subset can be denoted as
$$
\hat{\mathcal{I}}_{\mathbf{x}}=\left\{\pi(n): n \in \mathcal{B}_H^{(m)}(\mathbf{x})\right\}
\\
(4)
$$

> 通过将公式 (3) 递归应用直到 $h = H$，Beam Search 将检索出包含 $k$ 个叶节点的集合，记为 $\mathcal{B}_H(\mathbf{x})$。假设 $m \leq k$ 表示要检索的目标数量，则检索得到的目标子集可以表示为 
> $$
> \hat{\mathcal{I}}_{\mathbf{x}}=\left\{\pi(n): n \in \mathcal{B}_H^{(m)}(\mathbf{x})\right\} \\ (4)
> $$
> 其中 $\mathcal{B}_H^{(m)}(\mathbf{x})$ 表示从 $\mathcal{B}_H(\mathbf{x})$ 中选择概率最高的 $m$ 个节点，并通过函数 $\pi(\cdot)$ 将这些节点映射到对应的目标。

where  $\mathcal{B}_H^{(m)}(\mathbf{x}) \in \operatorname{argTopm}_{n \in \mathcal{B}_H(\mathbf{x})} p_g\left(z_n=1 \mid \mathbf{x}\right)$ denote the subset of $\mathcal{B}_H(\mathbf{x})$ with top-m scored nodes according to $p_g\left(z_n=1 \mid \mathbf{x}\right)$. Since Eq. (3) only traverses at most $b k$ nodes and generating $\mathcal{B}_H(\mathbf{x})$ needs computing Eq. (3) for H times, the testing complexity is $\mathrm{O}(H b k)$ per instance, which is also logarithmic to $M$.

To evaluate the retrieval performance of $\mathcal{M}(\mathcal{T}, g)$on the testing dataset $D_{te}$, Precision@m, Recall@m and Fmeasure@m are widely adopted. we define them as the average of Eq. (5), Eq. (6) and Eq. (7) over $D_{te}$ respectively, where
$$
\mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{1}{m} \sum_{j \in \hat{\mathcal{I}}_{\mathbf{x}}} y_j
\\(5)
$$

$$
\mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{1}{\left|\mathcal{I}_{\mathbf{x}}\right|} \sum_{j \in \hat{\mathcal{I}}_{\mathbf{x}}} y_j
\\(6)
$$

$$
\mathrm{F} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{2 \cdot \mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y}) \cdot \mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})}{\mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})+\mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})}
\\(7)
$$

> 其中 $\mathcal{B}_H^{(m)}(\mathbf{x}) \in \operatorname{argTopm}_{n \in \mathcal{B}_H(\mathbf{x})} p_g\left(z_n=1 \mid \mathbf{x}\right)$ 表示根据 $p_g\left(z_n=1 \mid \mathbf{x}\right)$ 得分，从 $\mathcal{B}_H(\mathbf{x})$ 中选择得分最高的 $m$ 个节点构成的子集。由于公式 (3) 最多遍历 $b k$ 个节点，并且生成 $\mathcal{B}_H(\mathbf{x})$ 需要计算公式 (3) $H$ 次，每个实例的测试复杂度为 $\mathrm{O}(H b k)$，这也是对 $M$ 取对数的。 为了评估 $\mathcal{M}(\mathcal{T}, g)$ 在测试数据集 $D_{te}$ 上的检索性能，通常使用 Precision@m、Recall@m 和 F-measure@m 进行评估。我们将它们定义为分别在 $D_{te}$ 上对公式 (5)、公式 (6) 和公式 (7) 求平均值，其中 
>
> 
> $$
> \mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{1}{m} \sum_{j \in \hat{\mathcal{I}}_{\mathbf{x}}} y_j \\(5)
> $$
>
> $$
> \mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{1}{\left|\mathcal{I}_{\mathbf{x}}\right|} \sum_{j \in \hat{\mathcal{I}}_{\mathbf{x}}} y_j \\(6)
> $$
>
> $$
> \mathrm{F} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})=\frac{2 \cdot \mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y}) \cdot \mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})}{\mathrm{P} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})+\mathrm{R} @ m(\mathcal{M} ; \mathbf{x}, \mathbf{y})} \\(7)
> $$

## 4. Main Contributions

Our main contributions can be divided into three parts: (1) We highlight the existence of the training-testing discrepancy on tree models, and provide an intuitive explanation of its negative effects on retrieval performance; (2) We develop the concept of Bayes optimality under beam search and calibration under beam search to formalize this intuitive explanation; (3) We propose a novel algorithm for learning tree models that are Bayes optimal under beam search.

> 我们的主要贡献可以分为三个部分：
>
> 1. 我们强调了树模型中存在的训练-测试差异，并对其对检索性能的负面影响提供了直观的解释；
> 2. 我们提出了在 Beam Search 下的贝叶斯最优性和在 Beam Search 下的校准的概念，以形式化这种直观解释；
> 3. 我们提出了一种新颖的算法，用于学习在 Beam Search 下是贝叶斯最优的树模型。

### 4.1. Understanding the Training-Testing Discrepancy on Tree Models

According to Eq. (2), the training of $g(\mathbf{x}, n)$ depends on two factors: the subsample set $\mathcal{S}_h(\mathbf{y})$ and the pseudo target $z_n$.

We can show that both factors relate to the training-testing discrepancy on existing tree models.

First, according to Eq. (3), the nodes at $h$-th level on which $g(\mathbf{x}, n)$ is queried in testing can be denoted as $\tilde{\mathcal{B}}_h(\mathbf{x})$, which implies a self-dependency of $g(\mathbf{x}, n)$, i.e., nodes on which $g(\mathbf{x}, n)$ is queried at $h$-th level depends on $g(\mathbf{x}, n)$ queried at $(h − 1)$-th level. However,$\mathcal{S}_h(\mathbf{y})$, the nodes at $h$-th level on which $g(\mathbf{x}, n)$ is trained, is generated according to ground truth targets $\mathbf{y}$ via Eq. (1). Figure 1(a) and Figure 1(b) demonstrate such a difference: Node 7 and 8 (blue nodes) are traversed by beam search, but they are not in $\mathcal{S}_h(\mathbf{y})$ of PLTs and may not be in $\mathcal{S}_h(\mathbf{y})$ of TDMs according to $\mathcal{S}_h^{+}(\mathbf{y})$ (red nodes). As a result, $g(\mathbf{x}, n)$is trained without considering such a self-dependency on itself when it is used for retrieving relevant targets via beam search. This discrepancy results in that $g(\mathbf{x}, n)$ trained well does not perform well in testing.

> 根据公式 (2)，$g(\mathbf{x}, n)$ 的训练取决于两个因素：子样本集 $\mathcal{S}_h(\mathbf{y})$ 和伪目标 $z_n$。
>
> 我们可以证明这两个因素与现有树模型中的训练-测试差异有关。
>
> 首先，根据公式 (3)，在测试集中查询 $g(\mathbf{x}, n)$ 的第 $h$ 层节点可以表示为 $\tilde{\mathcal{B}}_h(\mathbf{x})$，这意味着 $g(\mathbf{x}, n)$ 存在自依赖性，即在第 $h$ 层查询 $g(\mathbf{x}, n)$ 的节点取决于在第 $(h − 1)$ 层查询 $g(\mathbf{x}, n)$ 的节点。然而，$\mathcal{S}_h(\mathbf{y})$，即训练 $g(\mathbf{x}, n)$ 的第 $h$ 层节点，是通过公式 (1) 根据真实目标 $\mathbf{y}$ 生成的。图 1(a) 和图 1(b) 展示了这种差异：节点 7 和 8（蓝色节点）由 Beam Search 遍历，但它们不在 PLT 的 $\mathcal{S}_h(\mathbf{y})$ 中，也可能不在 TDM 的 $\mathcal{S}_h(\mathbf{y})$ 中，根据 $\mathcal{S}_h^{+}(\mathbf{y})$（红色节点）。因此，当使用 $g(\mathbf{x}, n)$ 通过 Beam Search 检索相关目标时，它的训练并没有考虑到这种自依赖性。这种差异导致了即使 $g(\mathbf{x}, n)$ 训练得很好，在测试中表现也不好。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Figure1.png)

Figure 1. An overview of the training-testing discrepancy on a tree model $\mathcal{M}(\mathcal{T}, g)$. (a) The assignment of pseudo targets on existing tree models, where red nodes correspond to  $z_n=1$ defined in Eq. (1). (b) Beam search process, where targets mapping blue nodes at 3-th level (i.e., leaf nodes) are regarded as the retrieval results of $\mathcal{M}$. (c) The assignment of optimal pseudo targets based on the ground truth distribution $\eta_j(\mathbf{x})=p\left(y_j=1 \mid \mathbf{x}\right)$, where green nodes correspond to $z_n^*=1$ defined in Eq. (13).

> 图1. 树模型 $\mathcal{M}(\mathcal{T}, g)$ 上的训练-测试差异概述。 (a) 对现有树模型上伪目标的分配，其中红色节点对应于公式 (1) 中定义的 $z_n=1$。 (b) Beam Search 过程，其中将第 3 层的蓝色节点（即叶节点）映射为 $\mathcal{M}$ 的检索结果。 (c) 基于真实分布 $\eta_j(\mathbf{x})=p\left(y_j=1 \mid \mathbf{x}\right)$ 的最优伪目标分配，其中绿色节点对应于公式 (13) 中定义的 $z_n^*=1$。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Table1.png)

Table 1. Results for the toy experiment with $M = 1000$, $b = 2$. The reported number is $\left(\sum_{j \in \mathcal{I}^{(k)}} \eta_j-\sum_{j \in \hat{\mathcal{I}}} \eta_j\right) / k$ which is averaged over 100 runs with random initialization over $\mathcal{T}$ and $η_j$ .

> 表1. 在 $M = 1000$、$b = 2$ 的玩具实验中的结果。报告的数字是 $\left(\sum_{j \in \mathcal{I}^{(k)}} \eta_j-\sum_{j \in \hat{\mathcal{I}}} \eta_j\right) / k$，在对 $\mathcal{T}$ 和 $η_j$ 进行随机初始化后，进行了100次运行并进行了平均。

Second, $z_n$ defined in Eq. (1) does not guarantee beam search w.r.t. $p_g(z_n = 1|\mathbf{x})$ has no performance deterioration, i.e., retrieving the most relevant targets. To see this, we design a toy example by ignoring $\mathbf{x}$ and defining the data distribution to be $p(\mathbf{y})=\prod_{j=1}^M p\left(y_j\right)$, whose marginal probability $\eta_j=p\left(y_j=1\right)$ is sampled from a uniform distribution in [0, 1]. As a result, we denote the training dataset as $\mathcal{D}_{t r}=\left\{\mathbf{y}^{(i)}\right\}_{i=1}^N$  and the pseudo target for instance $\mathbf{y}^{(i)}$ on node $n$ as $z^{(i)}_n$ . For $\mathcal{M}(\mathcal{T}, g)$, we assume $\mathcal{T}$ is randomly built and estimate $p\left(z_n=1\right)$ directly via $p_g\left(z_n=1\right)=\sum_{i=1}^N z_n^{(i)} / N$ without the need to specify $g$, since there is no observation $\mathbf{x}$. Beam search with beam size $k$ is applied on $\mathcal{M}$ to retrieve the target subset whose size is $m = k$ as well, denoted by $\hat{\mathcal{I}}=\left\{\pi(n): n \in \mathcal{B}_H\right\}$. Since $p(\mathbf{y})$ is known in this toy example, we need no testing set $\mathcal{D}_{te}$ and evaluate the retrieval performance directly via the regret $\left(\sum_{j \in \mathcal{I}^{(k)}} \eta_j-\sum_{j \in \hat{\mathcal{I}}} \eta_j\right) / k$, where $\mathcal{I}^{(k)} \in \operatorname{argTopk}_{j \in \mathcal{I}} \eta_j$ denotes the top-$k$ targets according to $η_j$ . As a special case of Eq. (10), this metric quantifies the suboptimality of $\mathcal{M}$ and we’ll discuss it formally later.

> 第二，根据公式（1）定义的 $z_n$ 不能保证针对 $p_g(z_n = 1|\mathbf{x})$ 的 beam search 不会出现性能下降，即无法检索到最相关的目标。为了看清这一点，我们设计了一个简单的例子，忽略 $\mathbf{x}$ 并将数据分布定义为 $p(\mathbf{y})=\prod_{j=1}^M p\left(y_j\right)$，其中边缘概率 $\eta_j=p\left(y_j=1\right)$ 是从 $[0, 1]$ 的均匀分布中采样得到的。因此，我们将训练数据集表示为 $\mathcal{D}_{tr}=\left\{\mathbf{y}^{(i)}\right\}_{i=1}^N$，对于节点 $n$ 上的实例 $\mathbf{y}^{(i)}$，伪目标被表示为 $z^{(i)}_n$。对于$\mathcal{M}(\mathcal{T}, g)$，我们假设 $\mathcal{T}$ 是随机构建的，并通过 $p_g\left(z_n=1\right)=\sum_{i=1}^N z_n^{(i)} / N$ 直接估计 $p\left(z_n=1\right)$，而无需指定 $g$，因为没有观察到 $\mathbf{x}$。在 $\mathcal{M}$ 上应用大小为 $k$ 的beam sarch，以检索目标子集，其大小也为$m=k$，表示为$\hat{\mathcal{I}}=\left\{\pi(n): n \in \mathcal{B}_H\right\}$。由于在这个简单的例子中 $p(\mathbf{y})$ 是已知的，我们不需要测试集 $\mathcal{D}_{te}$，而是直接通过 $\left(\sum_{j \in \mathcal{I}^{(k)}} \eta_j-\sum_{j \in \hat{\mathcal{I}}} \eta_j\right) / k$ 来评估检索性能，其中 $\mathcal{I}^{(k)} \in \operatorname{argTopk}_{j \in \mathcal{I}} \eta_j$ 表示根据 $η_j$ 获取前 $k$ 个目标的集合。作为公式（10）的一种特殊情况，该度量指标衡量了 $\mathcal{M}$ 的次优性，我们将在后面正式讨论它。

### 4.2. Bayes Optimality and Calibration under Beam Search

In Sec. 4.1, we discuss the existence of the training-testing discrepancy on tree models and provide a toy example to explain its effect. Without loss of generality, we formalize this discussion with Precision@m as the retrieval performance metric in this subsection.

The first question is, what does “optimal” mean for tree models with respect to their retrieval performance. In fact, the answer has been partially revealed by the toy example in Sec. 4.1, and we give a formal definition as follows:

**Definition 1** (Bayes Optimality under Beam Search). Given the beam size $k$ and the data distribution $p : \mathcal{X} × \mathcal{Y} → \mathbb{R}^+$, a tree model $\mathcal{M}(\mathcal{T} , g)$ is called top-$k$ Bayes optimal under beam search if
$$
\left\{\pi(n): n \in \mathcal{B}_H(\mathbf{x})\right\} \in \underset{j \in \mathcal{I}}{\operatorname{argTopk}} \eta_j(\mathbf{x}) \\ (8)
$$
holds for any $\mathbf{x} ∈ \mathcal{X} . \mathcal{M}(\mathcal{T} , g)$ is called Bayes optimal under beams search if Eq. (8) holds for any $\mathbf{x} ∈ \mathcal{X}$ and $1 ≤ k ≤ M$.

> 在4.1节中，我们讨论了树模型中训练和测试之间差异的存在，并提供了一个简单的例子来解释其影响。为了方便起见，我们在本小节中将此讨论形式化为以Precision@m为检索性能度量。 
>
> 首先要解决的问题是，对于树模型而言，与其检索性能相关的“最优”是什么意思。实际上，在4.1节的例子中部分揭示了答案，我们给出以下正式定义： 
>
> **定义1**（束搜索下的贝叶斯最优性）。给定束大小 $k$ 和数据分布 $p: \mathcal{X} × \mathcal{Y} → \mathbb{R}^+$，如果对于任意 $\mathbf{x} ∈ \mathcal{X}$ 都有
> $$
> \left\{\pi(n): n \in \mathcal{B}_H(\mathbf{x})\right\} \in \underset{j \in \mathcal{I}}{\operatorname{argTopk}} \eta_j(\mathbf{x}) \\ (8)
> $$
> 则称树模型 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是 top-$k$ 贝叶斯最优的。如果对于任意 $\mathbf{x} ∈ \mathcal{X}$ 和 $1 ≤ k ≤ M$ 都满足公式（8），则称 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是贝叶斯最优的。

Given Definition 1, we can derive a sufficient condition for the existence of such an optimal tree model as follows7 :

Proposition 1 (Sufficient Condition for Bayes Optimality under Beam Search). Given the beam size $k$, the data distribution $p : \mathcal{X} × \mathcal{Y} → \mathbb{R}^+$, the tree $\mathcal{T}$ and
$$
p^*\left(z_n \mid \mathbf{x}\right)= \begin{cases}\max _{n^{\prime} \in \mathcal{L}(n)} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}), & z_n=1 \\ 1-\max _{n^{\prime} \in \mathcal{L}(n)} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}), & z_n=0\end{cases}
\\(9)
$$
a tree model $\mathcal{M}(\mathcal{T} , g)$ is top-$m$ Bayes optimal under beam search for any $m ≤ k$, if $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$ holds for any $\mathbf{x} ∈ \mathcal{X}$ and $n \in \bigcup_{h=1}^H \tilde{\mathcal{B}}_h(\mathbf{x})$ . $\mathcal{M}(\mathcal{T} , g)$ is Bayes optimal under beam search, if $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$ holds for any $\mathbf{x} \in \mathcal{X}$ and $n ∈ \mathcal{N}$ . 

> 根据定义1，我们可以推导出一个充分条件，来确保存在这样一个最优的树模型，如下所示： 定义1（ beam search 下贝叶斯最优性的充分条件）。给定 beam 大小 $k$，数据分布 $p : \mathcal{X} × \mathcal{Y} → \mathbb{R}^+$ ，树 $\mathcal{T}$ 和 
> $$
> p^*\left(z_n \mid \mathbf{x}\right)= \begin{cases}\max _{n^{\prime} \in \mathcal{L}(n)} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}), & z_n=1 \\ 1-\max _{n^{\prime} \in \mathcal{L}(n)} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}), & z_n=0\end{cases} \\(9)
> $$
> 对于任意 $m ≤ k$，如果对于任意 $\mathbf{x} ∈ \mathcal{X}$ 和 $n \in \bigcup_{h=1}^H \tilde{\mathcal{B}}_h(\mathbf{x})$ 都满足 $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$，则树模型 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是 top-$m$ 贝叶斯最优的。如果对于任意 $\mathbf{x} \in \mathcal{X}$ 和 $n ∈ \mathcal{N}$ 都满足$p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$，则 $\mathcal{M}(\mathcal{T} , g)$ 在 beam search 下是贝叶斯最优的。

Proposition 1 shows one case of what an optimal tree model should be, but it does not resolve all the problems, since both learning and evaluating a tree model require a quantitative measure of its suboptimality. Notice that Eq. (8) implies that
$$
\mathbb{E}_{p(\mathbf{x})}\left[\sum_{j \in \mathcal{I}_{\mathbf{x}}^{(k)}} \eta_j(\mathbf{x})\right]=\mathbb{E}_{p(\mathbf{x})}\left[\sum_{n \in \mathcal{B}_H(\mathbf{x})} \eta_{\pi(n)}(\mathbf{x})\right]
$$
where $\mathcal{I}_{\mathbf{x}}^{(k)}=\operatorname{argTopk}_{j \in \mathcal{I}} \eta_j(\mathbf{x})$ denotes the top-$k$ targets according to the ground truth $η_j (x)$. The deviation of such an equation can be used as a suboptimality measure of $\mathcal{M}$. Formally, we define it to be the regret w.r.t. Precision@k and denote it as $\operatorname{reg}_{p @ k}(\mathcal{M})$.

 This is a special case when $m = k$ for a more general definition $\operatorname{reg}_{p @ k}(\mathcal{M})$ =
$$
\mathbb{E}_{p(\mathbf{x})}\left[\frac{1}{m}\left(\sum_{j \in \mathcal{I}_{\mathbf{x}}^{(m)}} \eta_j(\mathbf{x})-\sum_{n \in \mathcal{B}_H^{(m)}(\mathbf{x})} \eta_{\pi(n)}(\mathbf{x})\right)\right]
\\(10)
$$
where $\mathcal{I}_{\mathbf{x}}^{(m)}=\operatorname{argTopm}_{j \in \mathcal{I}} \eta_j(\mathbf{x})$.

> 定义1展示了最优的树模型应该满足的一个情况，但它并没有解决所有问题，因为学习和评估树模型都需要对其次优性进行定量衡量。
>
> 注意，公式（8）意味着
> $$
> \mathbb{E}_{p(\mathbf{x})}\left[\sum_{j \in \mathcal{I}_{\mathbf{x}}^{(k)}} \eta_j(\mathbf{x})\right]=\mathbb{E}_{p(\mathbf{x})}\left[\sum_{n \in \mathcal{B}_H(\mathbf{x})} \eta_{\pi(n)}(\mathbf{x})\right]
> $$
> 其中
>
> $\mathcal{I}_{\mathbf{x}}^{(k)}=\operatorname{argTopk}_{j \in \mathcal{I}} \eta_j(\mathbf{x})$ 表示根据真实值 $η_j (x)$ 获取前 $k$ 个目标。这个方程的偏差可以用作 $\mathcal{M}$ 的次优性度量。正式地，我们将其定义为关于Precision@k的遗憾度，并将其表示为 $\operatorname{reg}_{p @ k}(\mathcal{M})$ 。 这是当 $m = k$ 时的一种特殊情况，对于更一般的定义 $\operatorname{reg}_{p @ k}(\mathcal{M})$，有 
> $$
> \mathbb{E}_{p(\mathbf{x})}\left[\frac{1}{m}\left(\sum_{j \in \mathcal{I}_{\mathbf{x}}^{(m)}} \eta_j(\mathbf{x})-\sum_{n \in \mathcal{B}_H^{(m)}(\mathbf{x})} \eta_{\pi(n)}(\mathbf{x})\right)\right] \\(10)
> $$
> 其中 $\mathcal{I}_{\mathbf{x}}^{(m)}=\operatorname{argTopm}_{j \in \mathcal{I}} \eta_j(\mathbf{x})$。

Though $\operatorname{reg}_{p @ k}(\mathcal{M})$ seems an ideal suboptimality measure, finding its minimizer is hard due to the existence of a series of nested non-differentiable argTopk operators. Therefore, finding a surrogate loss for $\operatorname{reg}_{p @ k}(\mathcal{M})$ such that its minimizer is still an optimal tree model becomes very important. To distinguish such a surrogate loss, we introduce the concept of calibration under beam search as follows:

**Definition 2** (Calibration under Beam Search). Given a tree model $\mathcal{M}(\mathcal{T}, g)$, a loss function$L:\{0,1\}^M \times \mathbb{R}^{|\mathcal{N}|} \rightarrow \mathbb{R}$ is called top-$k$ calibrated under beam search if
$$
\underset{a}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}[L(\mathbf{y}, \mathbf{g}(\mathbf{x}))] \subset \underset{a}{\operatorname{argmin}} \operatorname{reg}_{p @ k}(\mathcal{M})
\\(11)
$$
holds for any distribution $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$. $L$ is called calibrated under beam search if Eq. (11) holds for any $1 ≤ k ≤ M$.

Definition 2 shows a tree model $\mathcal{M}(\mathcal{T}, g)$ with $g$ minimizing a non-calibrated loss is not Bayes optimal under beam search in general. Recall that Proposition 1 shows that for any $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ and any $\mathcal{T}$ , the minimizer of $\operatorname{reg}_{p @ k}(\mathcal{M})$ always exists, which satisfies $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$ and achieves $\operatorname{reg}_{p @ k}(\mathcal{M})=0$. Therefore, the suboptimality of TDMs and PLTs can be proved by showing the minimizer of their training loss does not guarantee $\operatorname{reg}_{p @ k}(\mathcal{M})=0$ in general. This can be proved by finding a counterexample and the toy experiment shown in Table 1 meets this requirement. As a result, we have

Proposition 2. Eq. (2) with zn defined in Eq. (1) is not calibrated under beam search in general.

> 尽管 $\operatorname{reg}_{p @ k}(\mathcal{M})$ 似乎是一个理想的次优性度量，但由于存在一系列嵌套的、不可微分的 argTopk 运算符，找到它的优化器是非常困难的。因此，找到一个替代损失函数 $\operatorname{reg}_{p @ k}(\mathcal{M})$，使得其最小化器仍然是一个最优的树模型，变得非常重要。为了区分这样的替代损失函数，我们引入了 beam search 下的校准概念，如下所示： 
>
> **定义2**（beam search 下的校准性）。给定树模型 $\mathcal{M}(\mathcal{T}, g)$，一个损失函数 $L:\{0,1\}^M \times \mathbb{R}^{|\mathcal{N}|} \rightarrow \mathbb{R}$ 被称为在 beam search 下 top-$k$ 校准的，如果对于任意分布 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 都有 
> $$
>  \underset{a}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}[L(\mathbf{y}, \mathbf{g}(\mathbf{x}))] \subset \underset{a}{\operatorname{argmin}} \operatorname{reg}_{p @ k}(\mathcal{M}) \\(11) 
> $$
> 对于任意 $1 ≤ k ≤ M$，如果公式（11）对于任意 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 都成立，则 $L$ 在 beam search下被称为校准的。
>
> 定义2展示了一般情况下，具有最小化非校准损失的树模型 $\mathcal{M}(\mathcal{T}, g)$ 在 beam search 下不是贝叶斯最优的。回顾命题1表明，对于任何 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{+}$ 和任何 $\mathcal{T}$，$\operatorname{reg}_{p @ k}(\mathcal{M})$ 的最小化器总是存在，并且满足 $p_g\left(z_n \mid \mathbf{x}\right)=p^*\left(z_n \mid \mathbf{x}\right)$ 以及 $\operatorname{reg}_{p @ k}(\mathcal{M})=0$。因此，通过证明训练损失的最小化器通常不能保证 $\operatorname{reg}_{p @ k}(\mathcal{M})=0$，可以证明 TDMs 和 PLTs 的次优性。通过找到一个反例，以及在表1中展示的简单实验，我们满足了这个要求。因此，我们得到以下结论：
>
> 命题2. 公式(2)中定义的 $z_n$ 在一般情况下不是 beam search 下的校准的。

### 4.3. Learning Optimal Tree Models under Beam Search

Given the discussion in Sec. 4.2, we need a new surrogate loss function such that its minimizer corresponds to the tree model which is Bayes optimal under beam search. According to Definition 1, when the retrieval performance is measured by Precision@m, requiring a model to be top-m Bayes optimal under beam search will be enough. Proposition 1 provides a natural surrogate loss to achieve this purpose with beam size $k ≥ m$, i.e.,
$$
g \in \underset{g}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x})}\left[\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h(\mathbf{x})} \operatorname{KL}\left(p^*\left(z_n \mid \mathbf{x}\right) \| p_g\left(z_n \mid \mathbf{x}\right)\right)\right]
\\(12)
$$
where we follow the TDM style and assume $p_g\left(z_n \mid \mathbf{x}\right)=1 /\left(1+\exp \left(-\left(2 z_n-1\right) g(\mathbf{x}, n)\right)\right)$

Unlike Eq. (2), Eq. (12) uses nodes in $\tilde{\mathcal{B}}_h(\mathbf{x})$ instead of $\mathcal{S}_h^{+}(\mathbf{y})$ for training and introduces a different definition of pseudo targets compared to Eq. (1). Let $z_n^* \sim p^*\left(z_n \mid \mathbf{x}\right)$ denote the corresponding pseudo target, we have 
$$
z_n^*=y_{\pi\left(n^{\prime}\right)}, n^{\prime} \in \underset{n^{\prime} \in \mathcal{L}(n)}{\operatorname{argmax}} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x})
\\(13)
$$
Notice that for $n \in \mathcal{N}_H$ , $z_n^*=y_{\pi(n)}$ as well as $z_n$ in Eq. (1). To distinguish $z_n^*$ from $z_n$ , we call it the optimal pseudo target since it corresponds to the optimal tree model.Given this definition, Eq. (12) can be rewritten as $\operatorname{argmin}_g \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}\left[L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))\right]$ where
$$
L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))=\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h(\mathbf{x})} \ell_{\mathrm{BCE}}\left(z_n^*, g(\mathbf{x}, n)\right)
\\(14)
$$

> 根据第4.2节的讨论，我们需要一个新的替代损失函数，使得它的最小化器对应于在 beam search 下是贝叶斯最优的树模型。根据定义1，当使用Precision@m来衡量检索性能时，要求一个模型在 beam search 下是top-m的贝叶斯最优将足够。定义1提供了一个自然的替代损失函数来实现这个目的，其中 beam search 的大小$k ≥ m$，即
> $$
> g \in \underset{g}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x})}\left[\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h(\mathbf{x})} \operatorname{KL}\left(p^*\left(z_n \mid \mathbf{x}\right) \| p_g\left(z_n \mid \mathbf{x}\right)\right)\right] \\(12) 
> $$
> 其中我们遵循 TDM 的风格，并假设 $p_g\left(z_n \mid \mathbf{x}\right)=1 /\left(1+\exp \left(-\left(2 z_n-1\right) g(\mathbf{x}, n)\right)\right)$。 
>
> 与公式（2）不同，公式（12）在训练中使用 $\tilde{\mathcal{B}}_h(\mathbf{x})$ 中的节点，而不是 $\mathcal{S}_h^{+}(\mathbf{y})$，并且相对于公式（1），引入了不同的伪目标定义。设 $z_n^* \sim p^*\left(z_n \mid \mathbf{x}\right)$ 表示相应的伪目标，我们有 
> $$
> z_n^*=y_{\pi\left(n^{\prime}\right)}, n^{\prime} \in \underset{n^{\prime} \in \mathcal{L}(n)}{\operatorname{argmax}} \eta_{\pi\left(n^{\prime}\right)}(\mathbf{x}) \\(13) 
> $$
> 注意，对于 $n \in \mathcal{N}_H$，$z_n^*=y_{\pi(n)}$以及公式（1）中的 $z_n$ 也是一样的。为了区分 $z_n^*$ 和 $z_n$，我们将其称为最优伪目标，因为它对应于最优的树模型。
>
> 根据这个定义，公式（12）可以重写为 $\operatorname{argmin}_g \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}\left[L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))\right]$，其中 
> $$
> L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))=\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h(\mathbf{x})} \ell_{\mathrm{BCE}}\left(z_n^*, g(\mathbf{x}, n)\right) \\(14)
> $$
> 

Notice that in Eq. (14) we assign a subscript $p$ to highlight the dependence of $z^∗_n$ on $η_j (x)$, which implies that Eq. (14) is calibrated under beam search in the sense that its formulation depends on $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{\top}$.

Figure 1 provides a concrete example for the difference between $z^∗_n$ and $z_n$. Not all ancestor nodes of a relevant target $y_j = 1$ are regarded as relevant nodes according to $z^∗_n$ : Node 1 and 6 (red nodes in Figure 1(a)) are assigned with $z_n = 1$ but with $z^∗_n = 0$ (green nodes in Figure 1(c)). The reason is that among targets on the subtree rooted at these nodes, the irrelevant target has a higher $η_j (x)$ compared to the relevant target, i.e., $\eta_7(\mathbf{x})=0.5>\eta_8(\mathbf{x})=0.4$ and $\eta_1(\mathbf{x})=0.8>\eta_3(\mathbf{x})=0.7$, which leads $z^∗_n$ to be 0.

However, it is impossible to minimize Eq. (14) directly, since $η_j (\mathbf{x})$ is unknown in practice. As a result, we need to find an approximation of $z^∗_n$ without the dependence on $η_j (\mathbf{x})$. Suppose $g(\mathbf{x}, n)$ is parameterized with trainable parameters $\boldsymbol{\theta} \in \boldsymbol{\Theta}$, we use the notation $g_{\boldsymbol{\theta}}(\mathbf{x}, n)$,$p_{g_{\boldsymbol{\theta}}}(\mathbf{x})$ and$\mathcal{B}_h(\mathbf{x} ; \boldsymbol{\theta})$ to highlight their dependence on $θ$. A natural choice is to replace $\eta_{\pi\left(n^{\prime}\right)}(\mathbf{x})$ in Eq. (13) with $p_{gθ} (z_{n'} = 1|\mathbf{x})$. However, this formulation is still impractical since the computational complexity of traversing $\mathcal{L}(n)$ for each $n \in \mathcal{B}_h(\mathbf{x} ; \boldsymbol{\theta})$ is unacceptable. Thanks to the tree structure, we can approximate $z_n^*$ with $$\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})$$, which is constructed in a recursive manner for $n \in \mathcal{N} \backslash \mathcal{N}_H$ as
$$
\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})=\hat{z}_{n^{\prime}}(\mathbf{x} ; \boldsymbol{\theta}), n^{\prime} \in \underset{n^{\prime} \in \mathcal{C}(n)}{\operatorname{argmax}} p_{g_{\boldsymbol{\theta}}}\left(z_{n^{\prime}}=1 \mid \mathbf{x}\right),
\\(15)
$$
and is set directly as $\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})=y_{\pi(n)}$ for $n \in \mathcal{N}_H$.

> 注意，在公式（14）中，我们为 $z^*_n$ 赋予了下标 $p$ 以突出其对 $η_j (x)$ 的依赖关系，这意味着公式（14）在 beam search 下是校准的，即其形成取决于 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^{\top}$。
>
> 图1提供了 $z^*_n$ 和 $z_n$ 之间差异的具体示例。并非所有相关目标 $y_j = 1$ 的祖先节点都被视为相关节点，根据 $z^*_n$ ：节点1和6（图1（a）中的红色节点）被赋予 $z_n = 1$ 但 $z^*_n = 0$（图1（c）中的绿色节点）。原因是在以这些节点为根的子树上，不相关目标相对于相关目标具有更高的 $η_j (x)$，即 $\eta_7(\mathbf{x})=0.5>\eta_8(\mathbf{x})=0.4$ 和 $\eta_1(\mathbf{x})=0.8>\eta_3(\mathbf{x})=0.7$，这导致 $z^*_n$ 为0。 然而，直接最小化公式（14）是不可能的，因为在实践中无法得知 $η_j (\mathbf{x})$。因此，我们需要找到一个近似的 $z^*_n$，使其不依赖于 $η_j (\mathbf{x})$。假设 $g(\mathbf{x}, n)$ 的参数化使用可训练的参数 $\boldsymbol{\theta} \in \boldsymbol{\Theta}$，我们使用记号 $g_{\boldsymbol{\theta}}(\mathbf{x}, n)$，$p_{g_{\boldsymbol{\theta}}}(\mathbf{x})$ 和 $\mathcal{B}_h(\mathbf{x} ; \boldsymbol{\theta})$ 来强调它们对 $θ$ 的依赖性。一个自然的选择是用 $p_{gθ} (z_{n'} = 1|\mathbf{x})$ 来替换公式（13）中的 $\eta_{\pi\left(n^{\prime}\right)}(\mathbf{x})$。然而，这个表达式仍然不实际，因为对于每个 $n \in \mathcal{B}_h(\mathbf{x} ; \boldsymbol{\theta})$，遍历 $\mathcal{L}(n)$ 的计算复杂度是不可接受的。由于树结构的存在，我们可以用 $$\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})$$ 近似表示 $z_n^*$，对于 $n \in \mathcal{N} \backslash \mathcal{N}_H$，它以递归的方式构造如下： 
> $$
>  \hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})=\hat{z}_{n^{\prime}}(\mathbf{x} ; \boldsymbol{\theta}), n^{\prime} \in \underset{n^{\prime} \in \mathcal{C}(n)}{\operatorname{argmax}} p_{g_{\boldsymbol{\theta}}}\left(z_{n^{\prime}}=1 \mid \mathbf{x}\right), \\(15) 
> $$
> 对于 $n \in \mathcal{N}_H$，直接设定为 $\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})=y_{\pi(n)}$。

By doing so, we remove the dependence on unknown $η_j (x)$. But minimizing Eq. (14) when replacing $z^*_n$ with $\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ is still not an easy task since the parameter $θ$ affects $\tilde{\mathcal{B}}_h(\mathbf{x} ; \boldsymbol{\theta})$ ,$\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ and $g_{\boldsymbol{\theta}}(\mathbf{x}, n)$: Gradient with respect to $θ$ cannot be computed directly due to the non-differentiability of the argTopk operator in $\tilde{\mathcal{B}}_h(\mathbf{x} ; \boldsymbol{\theta})$ and the argmax operator in $\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})$. To get a differentiable loss function, we propose to replace $L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))$ defined in Eq. (14) with
$$
L_{\boldsymbol{\theta}_t}(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})=\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)} \ell_{\mathrm{BCE}}\left(\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right), g_{\boldsymbol{\theta}}(\mathbf{x}, n)\right)
\\(16)
$$
where $θ_t$ denotes the fixed parameter, which can be the parameter of the last iteration in a gradient based algorithm. Given the discussion above, we propose a novel algorithm for learning such a tree model as Algorithm 1.

> 这样做可以消除对未知 $η_j (x)$ 的依赖性。但是，当用 $\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ 替换 $z^*_n$ 时，最小化公式（14）仍然不是一个简单的任务，因为参数 $θ$ 会影响 $\tilde{\mathcal{B}}_h(\mathbf{x} ; \boldsymbol{\theta}) $、$\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ 和 $g_{\boldsymbol{\theta}}(\mathbf{x}, n)$ ：由于 $\tilde{\mathcal{B}}_h(\mathbf{x} ; \boldsymbol{\theta})$ 中的argTopk操作符以及 $\hat{z}_n(\mathbf{x} ; \boldsymbol{\theta})$ 中的argmax操作符的不可微性，无法直接计算关于 $θ$ 的梯度。为了得到一个可微的损失函数，我们建议用公式（14）中定义的 $L_p(\mathbf{y}, \mathbf{g}(\mathbf{x}))$ 替换为 
> $$
> L_{\boldsymbol{\theta}_t}(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})=\sum_{h=1}^H \sum_{n \in \tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)} \ell_{\mathrm{BCE}}\left(\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right), g_{\boldsymbol{\theta}}(\mathbf{x}, n)\right) \\(16)
> $$
> 其中 $θ_t$ 表示固定的参数，可以是梯度下降算法中的最后一次迭代的参数。根据上述讨论，我们提出了一种用于学习这样一个树模型的新算法，如算法1所示。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Optimal Tree Models under Beam Search/Alg1.png)

As is analyzed in the supplementary materials, the training complexity of Algorithm 1 is $\mathrm{O}\left(H b k+H b\left|\mathcal{I}_{\mathbf{x}}\right|\right)$ per instance, which is still logarithmic to $M$. Besides, for the tree model trained according to Algorithm 1, its testing complexity is $O(Hbk)$ per instance as that in Sec. 3.2.2, since Algorithm 1 does not alter beam search in testing. 

Now, the remaining question is, since introducing several approximations into Eq. (16), does it still have the nice property to achieve Bayes optimality under beam search? We provide an answer8 as follows:

Proposition 3 (Practical Algorithm). Suppose $\mathcal{G}=\left\{g_{\boldsymbol{\theta}}:\right.\boldsymbol{\theta} \in \boldsymbol{\Theta}\}$ has enough capacity and $L_{\boldsymbol{\theta}_t}^*(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})=$
$$
\sum_{h=1}^H \sum_{n \in \mathcal{N}_h} w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right) \ell_{\mathrm{BCE}}\left(\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right), g_{\boldsymbol{\theta}}(\mathbf{x}, n)\right)
\\(17)
$$
where $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)>0$. For any probability $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^+$, if there exists $\boldsymbol{\theta}_t \in \boldsymbol{\Theta}$ such that
$$
\boldsymbol{\theta}_t \in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}\left[L_{\boldsymbol{\theta}_t}^{\boldsymbol{*}}(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})\right]
\\(18)
$$
the corresponding tree model $\mathcal{M}\left(\mathcal{T}, g_{\boldsymbol{\theta}_t}\right)$ is Bayes optimal under beam search.

Proposition 3 shows that replacing $z^∗_n$ with $\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ and introducing the fixed parameter $θ_t$ does not affect the optimality of $\mathcal{M}\left(\mathcal{T}, g_{\boldsymbol{\theta}_t}\right)$ on Eq. (17). However, Eq. (16) does not have such a guarantee, since the summation over $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ corresponds to the summation over $\mathcal{N}_h$ with weight $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)=\mathbb{I}\left(n \in \mathcal{B}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)\right)$ and thus violating the restriction that $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)>0$. This problem can be solved by introducing randomness into Eq. (16) such that each n ∈ Nh has a non-zero $n \in \mathcal{N}_h$ in expectation. Examples include adding random samples of $\mathcal{N}_h$ into the summation in Eq. (16) or leveraging stochastic beam search (Kool et al., 2019) to generate $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$. Nevertheless, in experiments we find these strategies do not greatly affect the performance, and thus we still use Eq. (16).

>根据附录中的分析，算法1的训练复杂度为每个实例 $\mathrm{O}\left(H b k+H b\left|\mathcal{I}_{\mathbf{x}}\right|\right)$，仍然对 $M$ 是对数复杂度。此外，根据算法1训练的树模型，在测试时的复杂度为每个实例 $O(Hbk)$，与第3.2.2节中相同，因为算法1在测试中没有改变 beam search 的方式。
>
>现在，剩下的问题是，由于在公式（16）中引入了几个近似，它是否仍然具有在 beam search 下实现贝叶斯最优性的良好性质？我们给出如下答案： 
>
>命题3（实用算法）。假设 $\mathcal{G}=\left\{g_{\boldsymbol{\theta}}:\right.\boldsymbol{\theta} \in \boldsymbol{\Theta}\}$ 具有足够的容量，并且 $L_{\boldsymbol{\theta}_t}^*(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})=$ 
>$$
> \sum_{h=1}^H \sum_{n \in \mathcal{N}_h} w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right) \ell_{\mathrm{BCE}}\left(\hat{z}_n\left(\mathbf{x} ; \boldsymbol{\theta}_t\right), g_{\boldsymbol{\theta}}(\mathbf{x}, n)\right) \\(17) 
>$$
>其中 $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)>0$。对于任意概率 $p: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^+$，如果存在 $\boldsymbol{\theta}_t \in \boldsymbol{\Theta}$ 使得 
>$$
>\boldsymbol{\theta}_t \in \underset{\boldsymbol{\theta} \in \boldsymbol{\Theta}}{\operatorname{argmin}} \mathbb{E}_{p(\mathbf{x}, \mathbf{y})}\left[L_{\boldsymbol{\theta}_t}^{\boldsymbol{*}}(\mathbf{y}, \mathbf{g}(\mathbf{x}) ; \boldsymbol{\theta})\right] \\(18) 
>$$
>则相应的树模型 $\mathcal{M}\left(\mathcal{T}, g_{\boldsymbol{\theta}_t}\right)$ 在 beam search 下是贝叶斯最优的。 
>
>命题3表明，用 $\hat{z}_n(\mathbf{x}, \boldsymbol{\theta})$ 替换 $z^*_n$ 并引入固定参数 $θ_t$ 不会影响公式（17）中的 $\mathcal{M}\left(\mathcal{T}, g_{\boldsymbol{\theta}_t}\right)$ 的最优性。然而，公式（16）没有这样的保证，因为对 $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$ 的求和对应于在权重 $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)=\mathbb{I}\left(n \in \mathcal{B}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)\right)$ 下对 $\mathcal{N}_h$ 的求和，从而违反了 $w_n\left(\mathbf{x}, \mathbf{y} ; \boldsymbol{\theta}_t\right)>0$ 的限制。这个问题可以通过在公式（16）中引入随机性来解决，使得每个 $n \in \mathcal{N}_h$ 在期望情况下都有一个非零值。例如，将 $\mathcal{N}_h$ 的随机样本添加到公式（16）的求和中，或者利用随机 beam search (Kool等，2019)来生成 $\tilde{\mathcal{B}}_h\left(\mathbf{x} ; \boldsymbol{\theta}_t\right)$。然而，在实验中我们发现这些策略并不会对性能产生很大影响，因此我们仍然使用公式（16）。

