# Joint Optimization of Tree-based Index and Deep Model for Recommender Systems



## Abstract

Large-scale industrial recommender systems are usually confronted with computational problems due to the enormous corpus size. To retrieve and recommend the most relevant items to users under response time limits, resorting to an efficient index structure is an effective and practical solution. The previous work Tree-based Deep Model (TDM) [34] greatly improves recommendation accuracy using tree index. By indexing items in a tree hierarchy and training a user-node preference prediction model satisfying a max-heap like property in the tree, TDM provides logarithmic computational complexity w.r.t. the corpus size, enabling the use of arbitrary advanced models in candidate retrieval and recommendation.

> 在大规模的工业推荐系统中，由于庞大的语料库大小，通常会遇到计算问题。为了在响应时间限制下检索和推荐最相关的物品给用户，使用高效的索引结构是一种有效且实用的解决方案。之前的工作"Tree-based Deep Model (TDM)" [34]通过使用树索引极大地提升了推荐准确性。通过在树的层次结构中对物品进行索引，并训练一个满足树中类似最大堆属性的 user-node 偏好预测模型，TDM相对于语料库大小具有对数级的计算复杂度，从而可以在候选检索和推荐中使用任意先进的模型。

In tree-based recommendation methods, the quality of both the tree index and the user-node preference prediction model determines the recommendation accuracy for the most part. We argue that the learning of tree index and preference model has interdependence. Our purpose, in this paper, is to develop a method to jointly learn the index structure and user preference prediction model. In our proposed joint optimization framework, the learning of index and user preference prediction model are carried out under a unified performance measure. Besides, we come up with a novel hierarchical user preference representation utilizing the tree index hierarchy. Experimental evaluations with two large-scale real-world datasets show that the proposed method improves recommendation accuracy significantly. Online A/B test results at a display advertising platform also demonstrate the effectiveness of the proposed method in production environments.

> 在基于树的推荐方法中，树索引和 user-node 偏好预测模型的质量主要决定了推荐准确性。我们认为，树索引和偏好模型的学习具有相互依赖性。因此，本文旨在开发一种方法来联合学习索引结构和用户偏好预测模型。在我们提出的联合优化框架中，索引和用户偏好预测模型的学习是在统一的评估度量下进行的。此外，我们还利用树索引层次结构提出了一种新的分层用户偏好表示方法。通过对两个大规模真实世界数据集进行实验评估，结果显示所提出的方法显著提高了推荐准确性。而在一个展示广告平台上进行的在线A/B测试结果也证明了该方法在生产环境中的有效性。
>

## 1 Introduction

Recommendation problem is basically to retrieve a set of most relevant or preferred items for each user request from the entire corpus. In the practice of large-scale recommendation, the algorithm design should strike a balance between accuracy and efficiency. In corpus with tens or hundreds of millions of items, methods that need to linearly scan each item’s preference score for each single user request are not computationally tractable. To solve the problem, index structure is commonly used to accelerate the retrieval process. In early recommender systems, item-based collaborative filtering (Item-CF) along with the inverted index is a popular solution to overcome the calculation barrier [18]. However, the scope of candidate set is limited, because only those items similar to user’s historical behaviors can be ultimately recommended.

> 推荐问题基本上是从整个语料库中检索出一组最相关或首选的物品，以满足每个用户请求。在大规模推荐实践中，算法设计需要在准确性和效率之间取得平衡。在包含数千万或上亿个物品的语料库中，需要线性扫描每个用户请求的每个物品的偏好分数的方法在计算上是不可行的。为了解决这个问题，通常使用索引结构来加速检索过程。在早期的推荐系统中，基于物品的协同过滤(Item-CF)结合倒排索引是克服计算障碍的流行解决方案[18]。然而，候选集的范围是有限的，因为只有那些与用户历史行为相似的物品最终可以被推荐。

In recent days, vector representation learning methods [27, 16, 26, 5, 22, 2] have been actively researched. This kind of methods can learn user and item’s vector representations, the inner-product of which represents user-item preference. For systems that use vector representation based methods, the recommendation set generation is equivalent to the k-nearest neighbor (kNN) search problem. Quantization-based index [19, 14] for approximate kNN search is widely adopted to accelerate the retrieval process. However, in the above solution, the vector representation learning and the kNN search index construction are optimized towards different objectives individually. The objective divergence leads to suboptimal vector representations and index structure [4]. An even more important problem is that the dependence on vector kNN search index requires an inner-product form of user preference modeling, which limits the model capability [10]. Models like Deep Interest Network [32], Deep Interest Evolution Network [31] and xDeepFM [17], which have been proven to be effective in user preference prediction, could not be used to generate candidates in recommendation.

> 近年来，向量表示学习方法[27, 16, 26, 5, 22, 2]得到了广泛研究。这种方法可以学习用户和物品的向量表示，其内积表示用户对物品的偏好。对于使用基于向量表示的方法的系统，推荐集生成等价于 $k$ 最近邻(kNN)搜索问题。基于量化的索引[19, 14]用于近似 kNN 搜索已被广泛采用以加速检索过程。然而，在上述解决方案中，向量表示学习和 kNN 搜索索引构建分别针对不同的目标进行优化。这种目标的差异导致了次优的向量表示和索引结构[4]。更重要的问题是，对向量 kNN 搜索索引的依赖需要使用内积形式的用户偏好建模，这限制了模型的能力[10]。像 Deep Interest Network [32]、Deep Interest Evolution Network [31] 和 xDeepFM [17]这样在用户偏好预测方面已被证明有效的模型无法用于生成推荐候选集。

In order to break the inner-product form limitation and make arbitrary advanced user preference models computationally tractable to retrieve candidates from the entire corpus, the previous work Tree-based Deep Model (TDM) [34] creatively uses tree structure as index and greatly improves the recommendation accuracy. TDM uses a tree index to organize items, and each leaf node in the tree corresponds to an item. Like a max-heap, TDM assumes that each user-node preference equals to the largest one among the user’s preference over all children of this node. In the training stage, a user-node preference prediction model is trained to fit the max-heap like preference distribution. Unlike vector kNN search based methods where the index structure requires an inner-product form of user preference modeling, there is no restriction on the form of preference model in TDM. And in prediction, preference scores given by the trained model are used to perform layer-wise beam search in the tree index to retrieve the candidate items. The time complexity of beam search in tree index is logarithmic w.r.t. the corpus size and no restriction is imposed on the model structure, which is a prerequisite to make advanced user preference models feasible to retrieve candidates in recommendation.

> 为了突破内积形式的限制，并使任意先进的用户偏好模型能够在整个语料库中进行候选项检索，之前的工作"Tree-based Deep Model (TDM)" [34]创造性地使用树结构作为索引，极大地提高了推荐准确性。TDM使用树索引来组织物品，树中的每个叶节点对应一个物品。类似于最大堆，TDM假设每个用户-节点偏好等于用户对该节点的所有子节点偏好中的最大值。在训练阶段，会训练一个用户-节点偏好预测模型来拟合最大堆形式的偏好分布。与基于向量 k 最近邻（kNN）搜索的方法不同，在TDM中，索引结构并不要求用户偏好建模为内积形式。因此，对偏好模型的形式没有限制。在预测阶段，训练得到的模型给出的偏好分数用于在树索引中进行逐层的beam search，以检索候选项。树索引中 beam search 的时间复杂度与语料库大小呈对数关系，并且对模型结构没有限制，这是使先进的用户偏好模型在推荐中检索候选项可行的先决条件。

The index structure plays different roles in kNN search based methods and tree-based methods. In kNN search based methods, the user and item’s vector representations are learnt first, and the vector search index is built then. While in tree-based methods, the tree index’s hierarchy also affects the retrieval model training. Therefore, how to learn the tree index and user preference model jointly is an important problem. Tree-based method is also an active research topic in literature of extreme classification [29, 1, 24, 11, 8, 25], which is sometimes considered the same as recommendation [12, 25]. In the existing tree-based methods, the tree structure is learnt for a better hierarchy in the sample or label space. However, the objective of sample or label partitioning task in the tree learning stage is not fully consistent with the ultimate target, i.e., accurate recommendation.The inconsistency between objectives of index learning and prediction model training leads the overall system to a suboptimal status. To address this challenge and facilitate better cooperation of tree index and user preference prediction model, we focus on developing a way to simultaneously learn the tree index and user preference prediction model by optimizing a unified performance measure.

> 在基于kNN搜索的方法和 Tree-based的方法中，索引结构扮演着不同的角色。在基于kNN搜索的方法中，首先学习用户和物品的向量表示，然后构建向量搜索索引。而在基于树的方法中，树索引的层次结构也会影响检索模型的训练。因此，如何同时学习树索引和用户偏好模型是一个重要的问题。树结构方法也是极端分类文献中的一个研究热点[29, 1, 24, 11, 8, 25]，有时被认为与推荐系统相同[12, 25]。在现有的基于树的方法中，树结构是为了在样本或标签空间中获得更好的层次结构而学习的。然而，在树学习阶段，样本或标签分区任务的目标与准确的推荐目标并不完全一致。索引学习和预测模型训练目标的不一致性导致整个系统处于次优状态。为了解决这个挑战，促进树索引和用户偏好预测模型的更好协作，我们专注于开发一种同时学习树索引和用户偏好预测模型的方法，通过优化统一的评估度量。

The main contributions of this paper are: 1) We propose a joint optimization framework to learn the tree index and user preference prediction model in tree-based recommendation, where a unified performance measure, i.e., the accuracy of user preference prediction is optimized; 2) We demon- strate that the proposed tree learning algorithm is equivalent to the weighted maximum matching problem of bipartite graph, and give an approximate algorithm to learn the tree; 3) We propose a novel method that makes better use of tree index to generate hierarchical user representation, which can help learn more accurate user preference prediction model; 4) We show that both the tree index learning and hierarchical user representation can improve recommendation accuracy, and these two modules can even mutually improve each other to achieve more significant performance promotion.

> 本文的主要贡献如下：
>
> 1. 提出了一种联合优化框架，学习基于树的推荐中的树索引和用户偏好预测模型，优化了统一的评估度量，即用户偏好预测的准确性；
>2. 我们证明了所提出的树学习算法等价于二部图的加权最大匹配问题，并给出了一个近似算法来学习树结构；
> 3. 我们提出了一种新颖的方法，更好地利用树索引生成分层用户表示，可以帮助学习更准确的用户偏好预测模型；
> 4. 我们验证了树索引学习和分层用户表示都可以提高推荐准确性，这两个模块甚至可以相互改进，从而实现更显著的性能提升。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Figure1.png)

**Figure 1: Tree-based deep recommendation model. (a) User preference prediction model. We firstly hierarchically abstract the user behaviors with nodes in corresponding levels. Then the abstract user behaviors and the target node together with the other feature such as the user profile are used as the input of the model. (b) Tree hierarchy. Each item is firstly assigned to a different leaf node with a projection function $π(·)$. In retrieval stage, items that assigned to the red nodes in the leaf level are selected as the candidate set.**

> 图1：基于树的深度推荐模型。 (a) 用户偏好预测模型。我们首先使用相应层级的节点对用户行为进行层次抽象。然后，将抽象的用户行为和目标节点以及其他特征如用户画像一起作为模型的输入。(b) 树结构。每个 item 首先通过投影函数 $π(·)$分配给不同的叶子节点。在检索阶段，选择红色的叶子结点是候选集。

## 2 Joint Optimization of Tree-based Index and Deep Model

In this section, we firstly give a brief review of TDM [34] to make this paper self-contained. Then we propose the joint learning framework of the tree-based index and deep model. In the last subsection, we specify the hierarchical user preference representation used in model training.

> 本节首先对 TDM [34]进行简要回顾，以使本文自洽。然后，我们提出 Tree-based 索引和深度模型的联合学习框架。在最后一个子节中，我们详细说明了模型训练中使用的层次化用户偏好表示方法。

### 2.1 Tree-based Deep Recommendation Model

In recommender systems with large-scale corpus, how to retrieve candidates effectively and efficiently is a challenging problem. TDM uses a tree as index and proposes a max-heap like probability formulation in the tree, where the user preference for each non-leaf node $n$ in level $l$ is derived as:
$$
p^{(l)}(n \mid u)=\frac{\max _{n_c \in\left\{n^{\prime} s \text { children in level } l+1\right\}} p^{(l+1)}\left(n_c \mid u\right)}{\alpha^{(l)}}
\\
(1)
$$
where $p^{(l)}(n|u)$ is the ground truth probability that the user $u$ prefers the node $n$. $\alpha^{(l)}$ is a level normalization term. The above formulation means that the ground truth user-node probability on a node equals to the maximum user-node probability of its children divided by a normalization term. Therefore, the top-k nodes in level l must be contained in the children of top-k nodes in level $l − 1$, and the retrieval for top-k leaf items can be restricted to recursive top-k nodes retrieval top-down in each level without losing the accuracy. Based on this, TDM turns the recommendation task into a hierarchical retrieval problem, where the candidate items are selected gradually from coarse to fine. The candidate generating process of TDM is shown in Fig 1.

> 在具有大规模语料库的推荐系统中，如何高效地检索候选项是一个具有挑战性的问题。TDM 使用树作为索引，并在树中提出了类似最大堆的概率公式。对于 $l$ 层的非叶节点 $n$，其用户偏好概率由以下公式计算得到：
> $$
> p^{(l)}(n \mid u)=\frac{\max _{n_c \in\left\{n^{\prime} s \text { children in level } l+1\right\}} p^{(l+1)}\left(n_c \mid u\right)}{\alpha^{(l)}}
> \\
> (1)
> $$
> 其中 $p^{(l)}(n|u)$ 是用户 $u$ 对节点 $n$ 偏好的 ground truth。$\alpha^{(l)}$ 是一个层的归一化项。上述公式的意思是节点上的真实用户-节点概率等于其子节点的最大用户-节点概率除以归一化项。因此，$l$ 层中的前 $k$ 个节点必须包含在层级 $l-1$ 中前 $k$ 个节点的子节点中，而且可以通过逐层自上而下地进行检索来限制检索top $k$ 个叶子item，而不会失去准确性。基于这一点，TDM 将推荐任务转化为一个分层检索问题，其中候选项从粗到细逐步选择。TDM 的候选生成过程如图1所示。

Each item is firstly assigned to a leaf node in the tree hierarchy $\mathcal{T}$ . A layer-wise beam search strategy is carried out as shown in Fig1(b). For level $l$, only the children of nodes with top-k probabilities in level $l − 1$ are scored and sorted to pick $k$ candidate nodes in level $l$. This process continues until $k$ leaf items are reached. User features combined with the candidate node are used as the input of the prediction model $\mathcal{M}$ (e.g. fully-connected networks) to get the preference probability, as shown in Fig 1(a). With tree index, the overall retrieval complexity for a user request is reduced from linear to logarithmic w.r.t. the corpus size, and there is no restriction on the preference model structure. This makes TDM break the inner-product form of user preference modeling restriction brought by vector kNN search index and enable arbitrary advanced deep models to retrieve candidates from the entire corpus, which greatly raises the recommendation accuracy.

> 每个 item 首先分配到树 $\mathcal{T}$ 中的一个叶节点。如图1(b)所示，采用一种 层次化的 beam search 策略。对于层级 $l$，仅对于上一层级 $l-1$ 中分数排名前 $k$ 的节点的子节点进行评分和排序，以选择层级 $l$ 中的 $k$ 个候选节点。该过程持续进行，直到达到 $k$ 个叶子 item 为止。将用户特征与候选节点结合起来，作为预测模型 $\mathcal{M}$（例如全连接网络）的输入，可以得到偏好概率，如图1(a)所示。通过树索引，相对于语料库的大小，用户请求的整体检索复杂度从线性降低为对数级别，并且不限制偏好模型的结构。这使得 TDM 打破了由向量 kNN 搜索索引带来的用户偏好建模形式的限制，并使任意高级深度模型能够从整个语料库中检索候选项，从而大大提高了推荐准确性。

### 2.2 Joint Optimization Framework

Derive the training set that has $n$ samples as ${(u^{(i)} , c^{(i)})}^n_{i=1}$ , in which the $i$ - th pair $(u^{(i)} , c^{(i)})$ means the user $u^{(i)}$ is interested in the target item $c^{(i)}$ . For $(u^{(i)} , c^{(i)})$, tree hierarchy $\mathcal{T}$ determines the path that prediction model $\mathcal{M}$ should select to achieve $c^{(i)}$ for $u^{(i)}$. We propose to jointly learn $\mathcal{M}$ and $\mathcal{T}$ under a global loss function. As we will see in experiments, jointly optimizing  $\mathcal{M}$ and $\mathcal{T}$  could improve the ultimate recommendation accuracy.

> 我们得到一个包含 $n$ 个样本的训练集，表示为 ${(u^{(i)}, c^{(i)})}^n_{i=1}$，其中第 $i$ 对 $(u^{(i)}, c^{(i)})$ 表示用户 $u^{(i)}$ 对目标 item $c^{(i)}$感兴趣。对于 $(u^{(i)}, c^{(i)})$，树 $\mathcal{T}$ 确定了预测模型 $\mathcal{M}$ 应选择的路径，以实现用户 $u^{(i)}$ 对 $c^{(i)}$ 的预测。我们提出在全局损失函数下联合学习 $\mathcal{M}$ 和 $\mathcal{T}$。正如我们将在实验中看到的，联合优化 $\mathcal{M}$ 和 $\mathcal{T}$ 可以提高最终的推荐准确性。

Given a user-item pair $(u, c)$, denote $p (π(c)|u; π)$ as user $u$’s preference probability over leaf node $π(c)$ where $π(·)$ is a projection function that projects an item to a leaf node in $\mathcal{T}$ . Note that $π(·)$ completely determines the tree hierarchy $\mathcal{T}$ , as shown in Fig 1(b). And optimizing $\mathcal{T}$ is actually optimizing $π(·)$. The model $\mathcal{M}$ estimates the user-node preference $\hat{p}(π(c)|u;θ,π)$, given $θ$ as model parameters. If the pair $(u, c)$ is a positive sample, we have the ground truth preference  $p (π(c)|u; π) = 1$ following the multi-class setting [5, 2]. According to the max-heap property, the user preference probability of all $π(c)$’s ancestor nodes, i.e., ${p(b_j(\pi(c))|u;\pi)}^{l_{max}}_{j=0}$ should also be $1$, in which $b_j(·)$ is the projection from a node to its ancestor node in level $j$ and $l_{max}$ is the max level in $\mathcal{T}$ . To fit such a user-node preference distribution, the global loss function is formulated as ：
$$
\mathcal{L}(\theta, \pi)=-\sum_{i=1}^n \sum_{j=0}^{l_{\max }} \log \hat{p}\left(b_j\left(\pi\left(c^{(i)}\right)\right) \mid u^{(i)} ; \theta, \pi\right)
\\ (2)
$$
where we sum up the negative logarithm of predicted user-node preference probability on all the positive training samples and their ancestor user-node pairs as the global empirical loss.

> 给定一个user-item 对 $(u, c)$，我们将 $p(π(c)|u; π)$ 表示为用户 $u$ 对叶子节点 $π(c)$ 的偏好概率，其中 $π(·)$ 是一个映射函数，将 item 映射到树 $\mathcal{T}$ 中的叶节点。请注意，$π(·)$ 完全确定了树 $\mathcal{T}$，如图1(b)所示。优化 $\mathcal{T}$ 实际上就是优化 $π(·)$。模型 $\mathcal{M}$ 根据参数 $θ$ 估计用户-节点偏好概率 $\hat{p}(π(c)|u;θ,π)$。如果 user-item 对 $(u, c)$ 是正样本，则多分类中我们有真实的偏好概率 $p(π(c)|u; π) = 1$ 。
>
> 根据最大堆的性质，所有 $π(c)$ 的祖先节点的用户偏好概率，即 ${p(b_j(\pi(c))|u;\pi)}^{l_{max}}_{j=0}$，也应该为 $1$，其中 $b_j(·)$ 是从一个节点映射到其在第 $j$ 级的祖先节点的投影函数，$l_{max}$ 是 $\mathcal{T}$ 中的最大层级。为了拟合这样的 user-item 偏好分布，我们可以制定全局损失函数： 
> $$
> \mathcal{L}(\theta, \pi)=-\sum_{i=1}^n \sum_{j=0}^{l_{\max }} \log \hat{p}\left(b_j\left(\pi\left(c^{(i)}\right)\right) \mid u^{(i)} ; \theta, \pi\right)
> \\ (2)
> $$
> 我们将对所有正训练样本及其祖先 user-node 对的预测 user-node 偏好概率的负对数之和作为全局经验损失。

Optimizing $π(·)$ is a combinational optimization problem, which can hardly be simultaneously optimized with $θ$ using gradient-based algorithms. To conquer this, we propose a joint learning framework as shown in Algorithm 1. It alternatively optimizes the loss function (2) with respect to the user preference model and the tree hierarchy. The consistency of the training loss in model training and tree learning promotes the convergence of the framework. Actually, Algorithm 1 surely converges if both the model training and tree learning decrease the value of (2) since ${L(θ_t,π_t)}$ is a decreasing sequence and lower bounded by $0$. In model training, $min_θL(θ, π)$ is to learn a user-node preference model for all levels, which can be solved by popular optimization algorithms for neural networks such as SGD[3], Adam[15]. In the normalized user preference setting [5, 2], since the number of nodes increases exponentially with the node level, Noise-contrastive estimation[7] is an alternative to estimate $\hat{p}(b_j(π(c))|u;θ,π)$ to avoid calculating the normalization term by sampling strategy. The task of tree learning is to solve $max_π −L(θ, π)$ given $θ$. $max_π −L(θ, π)$ equals to the maximum weighted matching problem of bipartite graph that consists of items in the corpus $\mathcal{C}$ and the leaf nodes of $\mathcal{T}$. The detailed proof is shown in the supplementary material.

> 为了克服优化 $π(·)$ 的问题，我们提出了一个联合学习框架，如算法1所示。它交替地优化损失函数(2)关于用户偏好模型和树形层次结构。模型训练和树形学习中训练损失的一致性促进了框架的收敛。实际上，如果模型训练和树形学习都减小了(2)的值，那么算法1肯定会收敛，因为 ${\mathcal{L}(θ_t,π_t)}$ 是一个递减的序列，并且下界为0。在模型训练中，$min_θ\mathcal{L}(θ, π)$ 是为所有层级学习用户-节点偏好模型，可以通过流行的神经网络优化算法（如SGD[3]、Adam[15]）来解决。在归一化用户偏好（模型）配置中[5, 2]，由于节点数量随着节点层级的增加呈指数增长，为了避免通过采样策略计算归一化项，可以使用NCE噪声对比估计[7]来估计 $\hat{p}(b_j(π(c))|u;θ,π)$。树形学习的任务是给定 $θ$ 解决 $max_π −L(θ, π)$。$max_π −L(θ, π)$ 等于由语料库 $\mathcal{C}$ 中的 item 和 $\mathcal{T}$ 的叶节点组成的二分图的最大加权匹配问题。详细的证明在补充材料中给出。 

Traditional algorithms for assignment problem such as the classic Hungarian algorithm are hard to apply for large corpus because of their high complexities. Even for the naive greedy algorithm that greedily chooses the unassigned edge with the largest weight, a big weight matrix needs to be computed and stored in advance, which is not acceptable. To conquer this issue, we propose a segmented tree learning algorithm.

> 传统的指派问题算法，如经典的匈牙利算法，在处理大规模语料库时往往计算复杂度过高。即使对于朴素的贪心算法，它会选择具有最大权重的未分配边缘，但需要提前计算和存储一个大的权重矩阵，这是不可接受的。为了解决这个问题，我们提出了一种分段树学习算法。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Alg1.png)

Instead of assigning items directly to leaf nodes, we achieve this step-by-step from the root node to the leaf level. Given a projection $π$ and the $k$-th item $c_k$ in the corpus, denote
$$
\mathcal{L}_{c_k}^{s, e}(\pi)=\sum_{(u, c) \in \mathcal{A}_k} \sum_{j=s}^e \log \hat{p}\left(b_j(\pi(c)) \mid u ; \theta, \pi\right)
$$
where $A_k = \{(u^{(i)} , c^{(i)}) | c^{(i)} = c_k\}^n_{i=1}$ is the set of training samples whose target item is $c_k$ , $s$ and $e$ are the start and end level respectively. We firstly maximize $\sum_{k=1}^{|\mathcal{C}|} \mathcal{L}_{c_k}^{1, d}(\pi)$ w.r.t. $π$, which is equivalent to assign all the items to nodes in level $d$. For a complete binary tree $\mathcal{T}$ with max level $l_{max}$, each node in level $d$ is assigned with no more than $2^{l_{max}−d}$ items. This is also a maximum matching problem which can be efficiently solved by a greedy algorithm, since the number of possible locations for each item is largely decreased if $d$ is well chosen (e.g. for $d = 7$, the number is $2^d = 128$). Denote the found optimal projection in this step as $π^∗$. Then, we successively maximize$\sum_{k=1}^{|\mathcal{C}|} \mathcal{L}_{c_k}^{d+1,2 d}(\pi)$ under the constraint that $\forall c \in \mathcal{C}, b_d(\pi(c))=b_d\left(\pi^*(c)\right)$ , which means keeping each item’s corresponding ancestor node in level $d$ unchanged. The recursion stops until each item is assigned to a leaf node. The proposed algorithm is detailed in Algorithm 2.

> 我们不直接将 item 分配给叶节点，而是从根节点逐步完成此过程，直到达到叶。给定一个映射函数 $π$ 和语料库中的第 $k$ 个 item $c_k$，定义
> $$
> \mathcal{L}_{c_k}^{s, e}(\pi)=\sum_{(u, c) \in \mathcal{A}_k} \sum_{j=s}^e \log \hat{p}\left(b_j(\pi(c)) \mid u ; \theta, \pi\right)
> $$
> 其中，$A_k = \{(u^{(i)} , c^{(i)}) | c^{(i)} = c_k\}^n_{i=1}$ 是目标 item 为 $c_k$ 的训练样本集合，$s$ 和 $e$ 分别是起始层和结束层。首先，我们针对 $π$ 最大化 $\sum_{k=1}^{|\mathcal{C}|} \mathcal{L}_{c_k}^{1, d}(\pi)$ ，这等价于将所有 item 分配给 $d$ 层的节点。对于具有最大层 $l_{max}$ 的完全二叉树 $\mathcal{T}$，第 $d$ 层的每个节点最多分配 $2^{l_{max}−d}$ 个 item。这也是一个最大匹配问题，可以通过贪心法高效解决，因为如果选择得当（例如对于 $d=7$，可能的位置数为 $2^d=128$），则每个 item 的可能位置大大减少。将在此步骤中找到的最优投影表示为 $π^∗$。
>
> 然后，我们依次在约束条件 $\forall c \in \mathcal{C}, b_d(\pi(c))=b_d\left(\pi^*(c)\right)$ 下最大化 $\sum_{k=1}^{|\mathcal{C}|} \mathcal{L}_{c_k}^{d+1,2 d}(\pi)$ ，这意味着保持每个 item 对应的祖先节点在第 $d$ 层不变。递归停止直到每个 item 都分配给一个叶节点。算法细节见算法2。

In line 5 of Algorithm 2, we use a greedy algorithm with rebalance strategy to solve the sub-problem. Each item $c \in \mathcal{C}_{n_i}$ is firstly assigned to the child of $n_i$ in level $l$ with largest weight $\mathcal{L}_c^{l-d+1, l}(\cdot)$ Then, a rebalance process is applied to ensure that each child is assigned with no more than $2^{l_{max−l}}$ items. The detailed implementation of Algorithm 2 is given in the supplementary material.

> 在算法2的第5行，我们使用一种贪婪算法和平衡策略来解决子问题。首先，将每个item $c \in \mathcal{C}_{n_i}$ 分配给在层级 $l$ 中具有最大权重 $\mathcal{L}_c^{l-d+1, l}(\cdot)$ 的节点 $n_i$ 的子节点。 然后，应用一个平衡过程以确保每个子节点被分配的项数不超过 $2^{l_{max−l}}$。其中，$l_{max}$ 是树中允许的最大层级。 算法2的详细实现可以在附录材料中找到。它提供了如何使用贪婪算法和平衡策略解决子问题的逐步描述。为了获取更具体的信息和深入理解，建议参考附录材料。

![Alg2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Alg2.png)

### 2.3 Hierarchical User Preference Representation

As shown in Section 2.1, TDM is a hierarchical retrieval model to generate the candidate items hierarchically from coarse to fine. In retrieval, a layer-wise top-down beam search is carried out through the tree index by the user preference prediction model $\mathcal{M}$. Therefore, $\mathcal{M}$′s task in each level are heterogeneous. Based on this, a level-specific input of $\mathcal{M}$ is necessary to raise the recommendation accuracy.

> 如第2.1节所示，TDM是一种分层检索模型，可以从粗到细地生成候选项。在检索过程中，通过用户偏好预测模型 $\mathcal{M}$ 进行逐层自顶向下的 beam search，并通过树索引获得候选项。因此，$\mathcal{M}$ 在每个层级上的任务是异构的。基于此，为了提高推荐准确性，需要为$\mathcal{M}$ 提供特定于层级的输入。

A series of related work [30, 6, 18, 16, 32, 33, 34] has shown that the user’s historical be- haviors play a key role in predicting the user’s interests. However, in our tree-based approach we could even enlarge this key role in a novel and effective way. Given a user behavior se- quence $c = \{c_1, c_2, · · · , c_m\}$ where $c_i$ is the $i$-th item the user interacts, we propose to use $c^l = \{b_l(π(c_1)), b_l(π(c_2)), · · · , b_l(π(c_m))\}$ as user’s behavior feature in level $l$. $c^l$ together with the target node and other possible features such as user profile are used as the input of $\mathcal{M}$ in level $l$ to predict the user-node preference, as shown in Fig 1(a). In addition, since each node or item is a one-hot ID feature, we follow the common way to embed them into continuous feature space. In this way, the ancestor nodes of items the user interacts are used as the hierarchical user preference representation. Generally, the hierarchical representation brings two main benefits:

1. **Level independence**. As in the common way, sharing item embeddings between different levels will bring noises in training the user preference prediction model $\mathcal{M}$, because the targets differ for different levels. An explicit solution is to attach an item with an independent embedding for each level. However, this will greatly increase the number of parameters and make the system hard to optimize and apply. The proposed hierarchical representation uses node embeddings in the corresponding level as the input of $\mathcal{M}$, which achieves level independence in training without increasing the number of parameters.
2. **Precise description**. $\mathcal{M}$ generates the candidate items hierarchically through the tree. With the increase of retrieval level, the candidate nodes in each level describe the ultimate recommended items from coarse to fine until the leaf level is reached. The proposed hierarchical user preference representation grasps the nature of the retrieval process and gives a precise description of user behaviors with nodes in corresponding level, which promotes the predictability of user preference by reducing the confusion brought by too detailed or coarse description. For example, $\mathcal{M}$’s task in upper levels is to coarsely select a candidate set and the user behaviors are also coarsely described with homogeneous node embeddings in the same upper levels in training and prediction.

Experimental study in both Section 3 and the supplementary material will show the significant effectiveness of the proposed hierarchical representation.

> 一系列相关工作[30, 6, 18, 16, 32, 33, 34]表明，用户的历史行为在预测用户兴趣方面起着关键作用。然而，在我们的 Tree-based 的方法中，我们可以以一种新颖有效的方式扩大这一关键作用。给定用户行为序列$c = \{c_1, c_2, · · · , c_m\}$，其中 $c_i$ 是用户交互的第 $i$ 个item，我们建议在 $l$ 层中使用 $c^l = \{b_l(π(c_1)), b_l(π(c_2)), · · · , b_l(π(c_m))\}$ 作为用户在该层的行为特征。如图1(a)所示，$c^l$ 与目标节点以及其他可能的特征（如用户画像）一起作为 $l$ 层中 $\mathcal{M}$ 的输入，用于预测用户-节点偏好。此外，由于每个节点或 item 都是一种 one-hot 特征，我们遵循常见的方法将它们 embedding 到连续特征空间中。通过这种方式，用户交互的 item 的祖先节点被用作分层用户偏好表示。
>
>  总体上，分层表示带来了两个主要好处： 
>
> 1. 层级独立性。在常规方法中，不同层之间共享的 item embedding 会在训练用户偏好预测模型 $\mathcal{M}$ 时引入噪声，因为不同层具有不同的目标。一种显式解决方案是为每个层的 item 附加一个独立的 embedding。然而，这样会大大增加参数数量，使系统难以优化和应用。所提出的分层表示使用相应层级的节点 embedding 作为 $\mathcal{M}$ 的输入，在训练中实现了层级独立性而无需增加参数数量。 
>2. 精确描述。$\mathcal{M}$ 通过树结构分层生成候选项。随着检索的层级的增加，每个级别中的候选节点描述了从粗到细的最终推荐项，直到达到叶节点级别。所提出的分层用户偏好表示把握了检索过程的本质，并使用相应层级的节点精确描述用户行为，从而通过减少过于详细或过于粗略的描述来提高用户偏好的可预测性。例如，在上层中，$\mathcal{M}$ 的任务是粗略选择候选集，并且在训练和预测中使用相同上层的同质节点 embedding 对用户行为进行粗略描述。 
> 
>第3节和补充材料中的实验研究将展示所提出的分层表示的显著有效性。

## 3 Experimental Study

We study the performance of the proposed method both offline and online in this section. We firstly compare the overall performance of the proposed method with other baselines. Then we conduct experiments to verify the contribution of each part and convergence of the framework. At last, we show the performance of the proposed method in an online display advertising platform with real traffic.

> 在本节中，我们对所提出的方法进行了离线和线上性能研究。首先，我们将所提出方法的整体性能与其他基线方法进行了比较。然后，我们进行实验来验证每个部分的贡献以及框架的收敛性。最后，我们展示了所提出方法在具有真实流量的在线展示广告平台上的表现。

### 3.1 Experiment Setup

The offline experiments are conducted with two large-scale real-world datasets: 1) Amazon Books3[20, 9], a user-book review dataset made up of product reviews from Amazon. Here we use its largest subset Books; 2) UserBehavior4[34], a subset of Taobao user behavior data. These two datasets both contain millions of items and the data is organized in user-item interaction form: each user-item interaction consists of user ID, item ID, category ID and timestamp. For the above two datasets, only users with no less than 10 interactions are kept.

> 离线实验使用了两个大规模真实世界数据集进行，分别是：
>
> 1. Amazon Books3[20, 9]：这是一个由亚马逊的产品评论组成的用户-图书评论数据集。我们使用其中最大的子集 Books 进行实验。
> 2. UserBehavior4[34]：这是淘宝用户行为数据的一个子集。这两个数据集都包含数百万个物品，并且数据以 user-item 交互的形式组织：每个 user-item 交互包括用户ID、物品ID、类别ID和时间戳。对于上述两个数据集，我们只保留至少有10个交互的用户。

To evaluate the performance of the proposed framework, we compare the following methods:

- **Item-CF**[28] is a basic collaborative filtering method and is widely used for personalized recommendation especially for large-scale corpus [18].
- **YouTube product-DNN** [5] is a practical method used in YouTube video recommendation. It’s the representative work of vector kNN search based methods. The inner-product of the learnt user and item’s vector representation reflects the preference. And we use the exact kNN search to retrieve candidates in prediction.
- **HSM**[21] is the hierarchical softmax model.It adopts the multiplication of layer-wise conditional  probabilities to get the normalized item preference probability.
- **TDM** [34] is the tree-based deep model for recommendation. It enables arbitrary advanced mod- els to retrieve user interests using the tree index. We use the proposed basic DNN version of TDM without tree learning and attention.
- **DNN** is a variant of TDM without tree index. The only difference is that it directly learns a user- item preference model and linearly scan all items to retrieve the top-k candidates in prediction. It’s computationally intractable in online system but a strong baseline in offline comparison.
- **JTM** is the proposed joint learning framework of the tree index and user preference prediction model. JTM-J and JTM-H are two variants. JTM-J jointly optimizes the tree index and user preference prediction model without the proposed hierarchical representation in Section 2.3. And JTM-H adopts the hierarchical representation but use the fixed initial tree index without tree learning.

> 为了评估所提出的框架的性能，我们与以下方法进行比较：
>
> - **Item-CF**[28]：这是一种基础的协同过滤方法，广泛用于个性化推荐，特别是在大规模语料库中[18]。
> - **YouTube product-DNN**[5]：这是YouTube视频推荐中使用的一种实用方法。它是基于向量kNN搜索的方法的代表性工作。学习到的用户和物品向量表示的内积反映了用户的偏好。在预测中，我们使用精确的kNN搜索来检索候选项。
>- **HSM**[21]：这是层次softmax模型。它采用层级条件概率的乘积来获得归一化的物品偏好概率。
> - **TDM**[34]：这是一种基于树的深度推荐模型。它使得任意高级模型可以使用树索引来检索用户兴趣。我们使用了TDM的基本DNN版本，没有进行树学习和注意力机制。
>- **DNN**：这是一个没有树索引的 TDM 变体。唯一的区别在于，它直接学习用户-物品偏好模型，并在预测中线性扫描所有物品以获取前 K 个候选项。在在线系统中，这样的计算是不可行的，但在离线比较中是一个强大的基准方法。
> - **JTM**：这是树索引和用户偏好预测模型的联合学习框架。JTM-J 和 JTM-H 是两个变体。JTM-J 同时优化树索引和用户偏好预测模型，但没有使用第2.3节中提出的分层表示。JTM-H采用分层表示，但使用固定的初始树索引，没有进行树学习。

Following TDM [34], we split users into training, validation and testing sets disjointly. Each user- item interaction in training set is a training sample, and the user’s behaviors before the interaction are the corresponding features. For each user in validation and testing set, we take the first half of behaviors along the time line as known features and the latter half as ground truth.

> 根据TDM [34]的方法，我们将用户分为互不相交的训练集、验证集和测试集。训练集中的每个 user-item 交互都被视为一个训练样本，而交互之前的用户行为则作为对应的特征。对于验证集和测试集中的每个用户，我们将时间线上的前半部分行为作为已知特征，后半部分作为真实值。

Taking advantage of TDM’s open source work5, we implement all methods in Alibaba’s deep learn- ing platform X-DeepLearning (XDL). HSM, DNN and JTM adopt the same user preference predic- tion model with TDM. We deploy negative sampling for all methods except Item-CF and use the same negative sampling ratio. 100 negative items in Amazon Books and 200 in UserBehavior are sampled for each training sample. HSM, TDM and JTM require an initial tree in advance of training process. Following TDM, we use category information to initialize the tree structure where items from the same category aggregate in the leaf level. More details and codes about data pre-processing and training are listed in the supplementary material.

> 利用TDM的开源工作[5]，我们在阿里巴巴的深度学习平台 X-DeepLearning (XDL) 上实现了所有方法。HSM、DNN 和 JTM 采用与 TDM 相同的用户偏好预测模型。除了 Item-CF 外，我们对所有方法都采用了负采样，并使用相同的负采样比例。对于每个训练样本，在Amazon Books中采样了100个负样本，在UserBehavior中采样了200个负样本。HSM、TDM 和 JTM 在训练过程中需要预先初始化一颗树。遵循 TDM 的方法，我们使用类别信息来初始化树结构，其中同一类别的物品在叶节点级别聚合。有关数据预处理和训练的更多详细信息和代码，请参考补充材料。 Precision（精确率）、Recall（召回率）和 F-Measure（F值）是三个常用的评估指标，我们使用它们来评估不同方法的性能。

Precision, Recall and F-Measure are three general metrics and we use them to evaluate the perfor- mance of different methods. For a user $u$, suppose $\mathcal{P}_u(|\mathcal{P}_u| = \mathcal{M})$ is the recalled set and $\mathcal{G}_u$ is the ground truth set. The equations of three metrics are
$$
\text { Precision@M(u)= } \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{P}_u\right|}, \text { Recall@M(u)= } \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|}
$$

$$
\text { F-Measure@ } M(u)=\frac{2 * \text { Precision@ } M(u) * \text { Recall@ } M(u)}{\text { Precision@ } M(u)+\text { Recall@ } M(u)}
$$

The results of each metric are averaged across all users in the testing set, and the listed values are the average of five different runs.

> 对于用户 $u$，假设 $\mathcal{P}_u(|\mathcal{P}_u| = \mathcal{M})$ 表示召回的集合，$\mathcal{G}_u$ 表示真实值集合。这三个指标的计算公式如下： 
>$$
> \text { Precision@M(u)= } \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{P}_u\right|}, \text { Recall@M(u)= } \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|}
>\\
> \text { F-Measure@ } M(u)=\frac{2 * \text { Precision@ } M(u) * \text { Recall@ } M(u)}{\text { Precision@ } M(u)+\text { Recall@ } M(u)}
> $$
> 每个指标的结果都是在测试集中所有用户上进行平均，并且列出的数值是五次不同运行的平均值。

### 3.2 Comparison Results

Table 1 exhibits the results of all methods in two datasets. It clearly shows that our proposed JTM outperforms other baselines in all metrics. Compared with the previous best model DNN in two datasets, JTM achieves 45.3% and 9.4% recall lift in Amazon Books and UserBehavior respectively.

Table 1: Comparison results of different methods in Amazon Books and UserBehavior ($M$ = 200).

> 表格1展示了两个数据集中所有方法的结果。它清楚地显示了我们提出的 JTM 在所有指标上优于其他基线模型。与两个数据集中之前最好的模型DNN相比，JTM在Amazon Books和UserBehavior数据集中分别实现了45.3%和9.4%的召回率提升。
>
> Table1：Amazon Books和UserBehavior数据集中不同方法的对比结果（$M$ = 200）。

![table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/table1.png)

As mentioned before, though computationally intractable in online system, DNN is a significantly strong baseline for offline comparison. Comparison results of DNN and other methods give insights in many aspects.

> 正如前面提到的，在在线系统中，虽然DNN在计算上难以处理，但它是离线比较中一个非常强大的基线模型。DNN和其他方法的比较结果在许多方面都提供了有益的启示。

Firstly, gap between YouTube product-DNN and DNN shows the limitation of inner-product form. The only difference between these two methods is that YouTube product-DNN uses the inner- product of user and item’s vectors to calculate the preference score, while DNN uses a fully- connected network. Such a change brings apparent improvement, which verifies the effectiveness of advanced neural network over inner-product form.

> 首先，YouTube product-DNN和DNN之间的差距显示了内积形式的局限性。这两种方法之间唯一的区别在于，YouTube product-DNN使用用户和物品向量的内积来计算偏好分数，而DNN使用一个全连接网络。这样的改变带来了明显的改进，验证了先进神经网络在内积形式上的有效性。

Next, TDM performs worse than DNN with an ordinary but not optimized tree hierarchy. Tree hi- erarchy takes effect in both training and prediction process. User-node samples are generated along the tree to fit max-heap like preference distribution, and layer-wise beam search is deployed in the tree index when prediction. Without a well-defined tree hierarchy, user preference prediction model may converge to a suboptimal version with confused generated samples, and it’s possible to lose targets in the non-leaf levels so that inaccurate candidate sets may be returned. Especially in sparse dataset like Amazon Books, learnt embedding of each node in tree hierarchy is not distinguishable enough so that TDM doesn’t perform well than other baselines. This phenomenon illustrates the influence of tree and necessity of tree learning. Additionally, HSM gets much worse results than TDM. This result is consistent with that reported in TDM[34]. When dealing with large corpus, as a result of layer-wise probability multiplication and beam search, HSM cannot guarantee the final recalled set to be optimal.

> 接下来，TDM 表现不如具有普通但未经优化的树层次结构的DNN。树层次结构在训练和预测过程中都发挥作用。用户节点样本沿着树生成，以适应最大堆样式的偏好分布，并且在预测时使用树索引进行逐层波束搜索。如果没有定义良好的树层次结构，用户偏好预测模型可能会收敛到一个次优版本，生成的样本可能会混淆，并且可能会在非叶层级丢失目标，从而返回不准确的候选集。特别是在像 Amazon Books 这样的稀疏数据集中，树层次结构中每个节点的学习 embedding 不够可区分，因此 TDM 的表现不如其他基准模型。这种现象说明了树的影响和树学习的必要性。另外，HSM 的结果比 TDM 差得多。这个结果与 TDM[34]中报告的结果一致。在处理大规模语料库时，由于逐层概率相乘和 beam search 的结果，HSM 不能保证最终的召回集是最优的。

By joint learning of tree index and user preference model, JTM outperforms DNN on all met- rics in two datasets with much lower retrieval complexity. More precise user preference prediction model and better tree hierarchy are obtained in JTM, which leads a better item set selection. Hierarchical user preference representation alleviates the data sparsity problem in upper levels, be- cause the feature space of user behavior feature is much smaller while having the same number of samples. And it helps model training in a layer-wise way to reduce the propagation of noises between levels. Besides, tree hierarchy learning makes similar items aggregate in the leaf level, so that the internal level models can get training samples with more consistent and unambiguous distribution. Benefited from the above two reasons, JTM provides better results than DNN.

> 通过树索引和用户偏好模型的联合学习，JTM 在两个数据集上在所有指标上均优于 DNN，并且检索复杂度更低。JTM 获得了更精确的用户偏好预测模型和更好的树层次结构，从而导致更好的物品集选择。层次化的用户偏好表示减轻了上层数据稀疏性的问题，因为用户行为特征的特征空间要小得多，而样本数却相同。它以逐层方式帮助模型训练，降低了噪声在不同层级之间的传播。此外，树层次结构的学习使得相似的物品聚集在叶节点级别，这样内部层级模型可以获得更一致和明确分布的训练样本。受益于以上两个原因，JTM提供了比DNN更好的结果。

Results in Table 1 under the dash line indicate the contribution of each part and their joint perfor- mance in JTM. Take the recall metric as an example. Compared to TDM in UserBehavior, tree learning and hierarchical representation of user preference brings 0.88% and 2.09% absolute gain separately. Furthermore, 3.87% absolute recall promotion is achieved by the corporation of both optimizations under a unified objective. Similar gain is observed in Amazon Books. The above results clearly show the effectiveness of hierarchical representation and tree learning, as well as the joint learning framework.

> 表1中虚线下方的结果显示了JTM中每个部分的贡献及其联合性能。以召回率为例。与UserBehavior中的TDM相比，树学习和用户偏好的层次化表示分别带来了0.88%和2.09%的绝对增益。此外，在统一目标下，这两个优化的结合使召回率提升了3.87%。在Amazon Books中也观察到类似的增益。以上结果清楚地展示了层次化表示和树学习以及联合学习框架的有效性。

Convergence of Iterative Joint Learning Tree hierarchy determines sample generation and search path. A suitable tree would benefit model training and inference a great deal. Fig 2 gives the comparison of clustering-based tree learning algorithm proposed in TDM [34] and our proposed joint learning approach. For fairness, two methods both adopt hierarchical user representation.

> 迭代联合学习的收敛性树层次结构决定了样本生成和搜索路径。一个合适的树结构将极大地有利于模型的训练和推断。图2比较了 TDM [34]中提出的基于聚类的树学习算法和我们提出的联合学习方法。为了公正起见，这两种方法都采用了层次化用户表示。

Since the proposed tree learning algorithm has the same objective with the user preference prediction model, it has two merits from the results: 1) It can converge to an optimal tree stably; 2) The final recommendation accuracy is higher than the clustering-based method. From Fig 2, we can see that results increase iteratively on all three metrics. Besides, the model stably converges in both datasets, while clustering-based approach ultimately overfits. The above results demonstrate the effectiveness and convergence of iterative joint learning empirically. Some careful readers might have noticed that the clustering algorithm outperforms JTM in the first few iterations. The reason is that the tree learning algorithm in JTM involves a *lazy strategy*, i.e., try to reduce the degree of tree structure change in each iteration (details are given in the supplementary material).

> 由于所提出的树学习算法与用户偏好预测模型具有相同的目标，在结果上它具有两个优点：
>
> 1. 它可以稳定地收敛到最优的树；
>2. 最终的推荐准确率高于基于聚类的方法。从图2中可以看出，所有三个指标的结果都在迭代过程中逐渐增加。此外，该模型在两个数据集中都稳定地收敛，而基于聚类的方法最终过拟合。以上结果从实证上证明了迭代联合学习的有效性和收敛性。
> 
> 一些仔细的读者可能已经注意到，在前几次迭代中，聚类算法的表现优于 JTM。原因是 JTM 中的树学习算法采用了一种懒惰策略，即每次迭代尽量降低树结构的改变程度（详细信息请参见补充材料）。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Figure2.png)

Figure 2: Results of iterative joint learning in two datasets (M = 200). 2(a), 2(b), 2(c) are results in Amazon Books and 2(d), 2(e), 2(f) shows the performance in UserBehavior. The horizontal axis of each figure represents the number of iterations.

> 图2：两个数据集（$M = 200$）中迭代联合学习的结果。2(a)、2(b)、2(c)是Amazon Books的结果，2(d)、2(e)、2(f)展示了UserBehavior的性能。每个图的横轴表示迭代次数。

### 3.3 Online Results

We also evaluate the proposed JTM in production environments: the display advertising scenario of *Guess What You Like* column of Taobao App Homepage. We use click-through rate (CTR) and revenue per mille (RPM) to measure the performance, which are the key performance indicators.

The definitions are:
$$
\mathrm{CTR}=\frac{\# \text { of clicks }}{\# \text { of impressions }}, \mathrm{RPM}=\frac{\text { Ad revenue }}{\# \text { of impressions }} * 1000
$$
> 我们还在生产环境中评估了提出的 JTM 模型：淘宝App首页的“猜你喜欢”栏目的展示广告场景。我们使用点击率（CTR）和千次展示收入（RPM）来衡量性能，这些是关键绩效指标。
>
> 定义如下：
> $$
> \mathrm{CTR}=\frac{\# \text { of clicks }}{\# \text { of impressions }}, \mathrm{RPM}=\frac{\text { Ad revenue }}{\# \text { of impressions }} * 1000
> $$

In the platform, advertisers bid on plenty of granularities like ad clusters, items, shops, etc. Several simultaneously running recommendation approaches in all granularities produce candidate sets and the combination of them are passed to subsequent stages, like CTR prediction [32, 31, 23], ranking [33, 13], etc. The comparison baseline is such a combination of all running recommendation meth- ods. To assess the effectiveness of JTM, we deploy JTM to replace Item-CF, which is one of the major candidate-generation approaches in granularity of items in the platform. TDM is evaluated in the same way as JTM. The corpus to deal with contains tens of millions of items. Each com- parison bucket has 2% of the online traffic, which is big enough considering the overall page view request amount. Table 2 lists the promotion of the two main online metrics. 11.3% growth on CTR exhibits that more precise items have been recommended with JTM. As for RPM, it has a 12.9% improvement, indicating JTM can bring more income for the platform.

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/table2.png)

> 在该平台上，广告主对广告簇、物品、店铺等多个细分领域进行竞价。所有细分领域中同时运行的推荐方法会生成候选集，并将它们的组合传递给后续阶段，例如 CTR 预测[32, 31, 23]、排序[33, 13]等。比较基准是所有运行中推荐方法的组合。为了评估 JTM 的有效性，我们部署 JTM 来替代平台上物品细分领域的主要候选生成方法之一，即 Item-CF。TDM 的评估方式与 JTM 相同。处理的数据集包含数千万个物品。每个桶有2%的在线流量，考虑到整体页面访问请求量，这已经足够大了。Table2列出了这两个主要在线指标的提升情况。CTR增长了11.3%，说明 JTM 能够推荐更准确的物品。而 RPM 则提高了12.9%，表明 JTM 可以为平台带来更多收入。

## 4 Conclusion

Recommender system plays a key role in various kinds of applications such as video streaming and e-commerce. In this paper, we address an important problem in large-scale recommendation, i.e., how to optimize user representation, user preference prediction and the index structure under a global objective. To the best of our knowledge, JTM is the first work that proposes a unified framework to integrate the optimization of these three key factors. A joint learning approach of the tree index and user preference prediction model is introduced in this framework. The tree index and deep model are alternatively optimized under a global loss function with a novel hierarchical user representation based on the tree index. Both online and offline experimental results show the advantages of the proposed framework over other large-scale recommendation models.

> 推荐系统在视频流媒体和电子商务等各种应用中起着关键作用。本文针对大规模推荐中的一个重要问题进行了研究，即如何在全局目标下优化用户表示、用户偏好预测和索引结构。据我们所知，JTM 是第一个提出了统一框架来整合这三个关键因素优化的工作。该框架引入了树索引和用户偏好预测模型的联合学习方法。根据树索引建立了一种新颖的层次化用户表示，并在全局损失函数下交替优化树索引和深度模型。在线和离线实验结果都显示了所提出框架在大规模推荐模型中的优势。

Acknowledgements

We deeply appreciate Jingwei Zhuo, Mingsheng Long, Jin Li for their helpful suggestions and dis- cussions. Thank Huimin Yi, Yang Zheng and Xianteng Wu for implementing the key components of the training and inference platform. Thank Yin Yang, Liming Duan, Yao Xu, Guan Wang and Yue Gao for necessary supports about online serving.

References

1. [1]  R. Agrawal, A. Gupta, Y. Prabhu, and M. Varma. Multi-label learning with millions of labels: recom- mending advertiser bid phrases for web pages. In *WWW*, pages 13–24, 2013.

2. [2]  A.Beutel,P.Covington,S.Jain,C.Xu,J.Li,V.Gatto,andE.H.Chi.Latentcross:Makinguseofcontext in recurrent recommender systems. In *WSDM*, pages 46–54, 2018.

3. [3]  L.Bottou.Large-scalemachinelearningwithstochasticgradientdescent.In*COMPSTAT*,pages177–186. 2010.

4. [4]  Y. Cao, M. Long, J. Wang, H. Zhu, and Q. Wen. Deep quantization network for efficient image retrieval. In *AAAI*, pages 3457–3463, 2016.

5. [5]  P. Covington, J. Adams, and E. Sargin. Deep neural networks for youtube recommendations. In *RecSys*, pages 191–198, 2016.

6. [6]  J. Davidson, B. Liebald, J. Liu, P. Nandy, T. V. Vleet, U. Gargi, S. Gupta, Y. He, M. Lambert, B. Liv- ingston, and D. Sampath. The youtube video recommendation system. In *RecSys*, pages 293–296, 2010.

7. [7]  M. Gutmann and A. Hyva ̈rinen. Noise-contrastive estimation: A new estimation principle for unnormal-

   ized statistical models. In *AISTATS*, pages 297–304, 2010.

1. [8]  L. Han, Y. Huang, and T. Zhang. Candidates vs. noises estimation for large multi-class classification problem. In *ICML*, pages 1885–1894, 2018.

2. [9]  R. He and J. McAuley. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In *WWW*, pages 507–517, 2016.

3. [10]  X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T. Chua. Neural collaborative filtering. In *WWW*, pages 173–182, 2017.

4. [11]  H. D. III, N. Karampatziakis, J. Langford, and P. Mineiro. Logarithmic time one-against-some. In *ICML*, pages 923–932, 2017.

5. [12]  H. Jain, Y. Prabhu, and M. Varma. Extreme multi-label loss functions for recommendation, tagging, ranking & other missing label applications. In *KDD*, pages 935–944, 2016.

6. [13]  J. Jin, C. Song, H. Li, K. Gai, J. Wang, and W. Zhang. Real-time bidding with multi-agent reinforcement learning in display advertising. In *CIKM*, pages 2193–2201, 2018.

7. [14]  J. Johnson, M. Douze, and H. Je ́gou. Billion-scale similarity search with gpus. *arXiv preprint arXiv:1702.08734*, 2017.

8. [15]  D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. In *ICLR*, 2015.

9. [16]  Y. Koren, R. M. Bell, and C. Volinsky. Matrix factorization techniques for recommender systems. *IEEEComputer*, 42(8):30–37, 2009.

10. [17]  J. Lian, X. Zhou, F. Zhang, Z. Chen, X. Xie, and G. Sun. xdeepfm: Combining explicit and implicit

    feature interactions for recommender systems. In *KDD*, pages 1754–1763, 2018.

11. [18]  G. Linden, B. Smith, and J. York. Amazon.com recommendations: Item-to-item collaborative filtering.

    *IEEE Internet Computing*, 7(1):76–80, 2003.

12. [19]  T.Liu,A.W.Moore,A.G.Gray,andK.Yang.Aninvestigationofpracticalapproximatenearestneighbor

    algorithms. In *NeurIPS*, pages 825–832, 2004.

13. [20]  J. J. McAuley, C. Targett, Q. Shi, and A. van den Hengel. Image-based recommendations on styles and

    substitutes. In *SIGIR*, pages 43–52, 2015.

14. [21]  F. Morin and Y. Bengio. Hierarchical probabilistic neural network language model. In *AISTATS*, 2005.

15. [22]  S. Okura, Y. Tagami, S. Ono, and A. Tajima. Embedding-based news recommendation for millions of

    users. In *KDD*, pages 1933–1942, 2017.

16. [23]  Q. Pi, W. Bian, G. Zhou, X. Zhu, and K. Gai. Practice on long sequential user behavior modeling for

    click-through rate prediction. In *KDD*, pages 2671–2679, 2019.

17. [24]  Y. Prabhu and M. Varma. Fastxml: a fast, accurate and stable tree-classifier for extreme multi-label

    learning. In *KDD*, pages 263–272, 2014.

18. [25]  Y. Prabhu, A. Kag, S. Harsola, R. Agrawal, and M. Varma. Parabel: Partitioned label trees for extreme

    classification with application to dynamic search advertising. In *WWW*, pages 993–1002, 2018.

19. [26]  S. Rendle. Factorization machines. In *ICDM*, pages 995–1000, 2010.

20. [27]  R. Salakhutdinov and A. Mnih. Probabilistic matrix factorization. In *NeurIPS*, pages 1257–1264, 2007.

21. [28]  B.M.Sarwar,G.Karypis,J.A.Konstan,andJ.Riedl.Item-basedcollaborativefilteringrecommendation

    algorithms. In *WWW*, pages 285–295, 2001.

22. [29]  J. Weston, A. Makadia, and H. Yee. Label partitioning for sublinear ranking. In *ICML*, pages 181–189,

    2013.

23. [30]  S.Zhang,L.Yao,andA.Sun.Deeplearningbasedrecommendersystem:Asurveyandnewperspectives.

    *arXiv preprint arXiv:1707.07435*, 2017.

24. [31]  G. Zhou, N. Mou, Y. Fan, Q. Pi, W. Bian, C. Zhou, X. Zhu, and K. Gai. Deep interest evolution network

    for click-through rate prediction. *arXiv preprint arXiv:1809.03672*, 2018.

25. [32]  G.Zhou,X.Zhu,C.Song,Y.Fan,H.Zhu,X.Ma,Y.Yan,J.Jin,H.Li,andK.Gai.Deepinterestnetwork

    for click-through rate prediction. In *KDD*, pages 1059–1068, 2018.

26. [33]  H. Zhu, J. Jin, C. Tan, F. Pan, Y. Zeng, H. Li, and K. Gai. Optimized cost per click in taobao display

    advertising. In *KDD*, pages 2191–2200, 2017.

27. [34]  H.Zhu,X.Li,P.Zhang,G.Li,J.He,H.Li,andK.Gai.Learningtree-baseddeepmodelforrecommender

    systems. In *KDD*, pages 1079–1088, 2018.