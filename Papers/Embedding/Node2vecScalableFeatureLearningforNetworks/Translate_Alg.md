> Aditya Grover，斯坦福

# node2vec: Scalable Feature Learning for Networks

## Abstract

Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks.

Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node’s network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations.

We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.

> 网络中节点和边的预测任务需要仔细研究学习算法所使用的工程特性。最近在更广泛的表征学习领域的研究已经在通过学习特征本身来自动预测方面取得了重大进展。然而，现有的特征学习方法不足以捕捉网络中观察到的连接模式的多样性。
>
> 这里，我们提出了 node2vec ，一个用于学习网络中节点的连续特征表示的算法框架。在 node2vec 中，我们学习了节点到低维特征空间的映射，该映射最大程度地保留了节点的网络邻域。我们定义了一个灵活的节点网络邻域概念，设计了一个有偏随机游走程序，有效地探索了不同的邻域。我们的算法推广了以前基于网络邻域的刚性概念的工作，我们认为在探索邻域时增加的灵活性是学习更丰富表示的关键。
>
> 我们证明了 node2vec 在几个不同领域的真实网络中的多标签分类和链接预测方面比现有最先进的技术更有效。综上所述，我们的工作代表了一种在复杂网络中有效学习与任务无关的最先进表示的新方法。

## 1. Introduction

Many important tasks in network analysis involve predictions over nodes and edges. In a typical node classification task, we are interested in predicting the most probable labels of nodes in a network [33]. For example, in a social network, we might be interested in predicting interests of users, or in a protein-protein interaction network we might be interested in predicting functional labels of proteins [25, 37]. Similarly, in link prediction, we wish to predict whether a pair of nodes in a network should have an edge connecting them [18]. Link prediction is useful in a wide variety of domains; for instance, in genomics, it helps us discover novel interactions between genes, and in social networks, it can identify real-world friends [2, 34].

Any supervised machine learning algorithm requires a set of informative, discriminating, and independent features. In prediction problems on networks this means that one has to construct a feature vector representation for the nodes and edges. A typical solution involves hand-engineering domain-specific features based on expert knowledge. Even if one discounts the tedious effort required for feature engineering, such features are usually designed for specific tasks and do not generalize across different prediction tasks.

> 网络分析中的许多重要任务都涉及到对节点和边的预测。在典型的节点分类任务中，我们感兴趣的是预测网络中节点最可能的标签[33]。例如，在社交网络中，我们可能对预测用户的兴趣感兴趣，或者在蛋白质-蛋白质相互作用网络中，我们可能对预测蛋白质的功能标签感兴趣[25，37]。类似地，在链接预测中，我们希望预测网络中的一对节点是否应该具有连接它们的边[18]。链接预测在很多领域都很有用；例如，在基因组学中，它可以帮助我们发现基因之间的新交互作用；在社交网络中，它可以识别现实世界中的朋友[2，34]。
>
> 任何有监督的机器学习算法都需要一组信息丰富，具有区别性和独立性的特征。在网络的预测问题中，这意味着必须为节点和边构造特征的向量表示。一个典型的解决方案涉及基于专家知识手工设计特定领域的特征。即使不考虑特征工程所需的繁琐工作，这些特征通常是为特定任务设计的，不会在不同的预测任务中推广。

However, current techniques fail to satisfactorily define and optimize a reasonable objective required for scalable unsupervised feature learning in networks. Classic approaches based on linear and non-linear dimensionality reduction techniques such as Principal Component Analysis, Multi-Dimensional Scaling and their extensions [3, 27, 30, 35] optimize an objective that transforms a representative data matrix of the network such that it maximizes the variance of the data representation. Consequently, these approaches invariably involve eigendecomposition of the appropriate data matrix which is expensive for large real-world networks. Moreover, the resulting latent representations give poor performance on various prediction tasks over networks.

Alternatively, we can design an objective that seeks to preserve local neighborhoods of nodes. The objective can be efficiently optimized using stochastic gradient descent (SGD) akin to backpropogation on just single hidden-layer feedforward neural networks. Recent attempts in this direction [24, 28] propose efficient algorithms but rely on a rigid notion of a network neighborhood, which results in these approaches being largely insensitive to connectivity patterns unique to networks. Specifically, nodes in networks could be organized based on communities they belong to (i.e., homophily); in other cases, the organization could be based on the structural roles of nodes in the network (i.e., structural equivalence) [7, 10, 36]. For instance, in Figure 1, we observe nodes u and s1 belonging to the same tightly knit community of nodes, while the nodes u and s6 in the two distinct communities share the same structural role of a hub node. Real-world networks commonly exhibit a mixture of such equivalences. Thus, it is essential to allow for a flexible algorithm that can learn node representations obeying both principles: ability to learn representations that embed nodes from the same network community closely together, as well as to learn representations where nodes that share similar roles have similar embeddings. This would allow feature learning algorithms to generalize across a wide variety of domains and prediction tasks.

![Fig1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Node2vecScalableFeatureLearningforNetworks/Fig1.png)

**Figure 1: BFS and DFS search strategies from node $u$ $(k = 3)$.**

> 然而，现有的技术不能令人满意地定义和优化网络中可扩展的无监督特征学习所需的合理目标。基于线性和非线性降维技术的经典方法，例如主成分分析、多维缩放及其扩展[3、27、30、35]，通过使数据表示的方差最大化转换代表网络的数据矩阵。因此，这些方法总是涉及一些特征分解，这对于现实世界的大型网络来说代价是十分高的。此外，由此产生的潜在表示在网络上的各种预测任务中的性能较差。
>
> 或者，我们可以设计一个目标，以保留节点的本地邻域。可以使用类似于仅在单个隐藏层的前馈神经网络上进行反向传播的随机梯度下降（SGD）来有效地优化目标。在这个方向上的最新尝试[24，28]提出了有效的算法，但依赖于网络邻域的严格概念，这导致这些方法对网络独有的连接模式不敏感。具体来说，网络中的节点可以基于它们所属的社区（即同质性）进行组织；在其他情况下，组织可以基于网络中节点的结构角色（即结构等效性）[7、10、36]。例如，在图1中，我们观察到节点 $u$ 和 $s_1$属于同一紧密结合的节点社区，而两个不同社区中的节点 $u$ 和 $s_6$ 共享相同的中枢节点的结构角色。现实世界的网络通常会表现出与上述等价的的混合体。因此，必须允许一种灵活的算法，该算法可以学习符合这两个原则的节点表示：能够学习将来自同一网络社区的节点紧密j结合在一起的表示，以及能够学习到有相似角色的节点具有相似 embedding 的表示。这将允许特征学习算法在广泛的领域和预测任务中推广。

#### Present work

We propose node2vec, a semi-supervised algorithm for scalable feature learning in networks. We optimize a custom graph-based objective function using SGD motivated by prior work on natural language processing [21]. Intuitively, our approach returns feature representations that maximize the likelihood of preserving network neighborhoods of nodes in a d-dimensional feature space. We use a 2nd order random walk approach to generate (sample) network neighborhoods for nodes.

Our key contribution is in defining a flexible notion of a node’s network neighborhood. By choosing an appropriate notion of a neighborhood, node2vec can learn representations that organize nodes based on their network roles and/or communities they belong to. We achieve this by developing a family of biased random walks, which efficiently explore diverse neighborhoods of a given node. The resulting algorithm is flexible, giving us control over the search space through tunable parameters, in contrast to rigid search procedures in prior work [24, 28]. Consequently, our method generalizes prior work and can model the full spectrum of equivalences observed in networks. The parameters governing our search strategy have an intuitive interpretation and bias the walk towards different network exploration strategies. These parameters can also be learned directly using a tiny fraction of labeled data in a semisupervised fashion.

We also show how feature representations of individual nodes can be extended to pairs of nodes (i.e., edges). In order to generate feature representations of edges, we compose the learned feature representations of the individual nodes using simple binary operators. This compositionality lends node2vec to prediction tasks involving nodes as well as edges.

> 我们提出了一种网络中可扩展特征学习的半监督学习 node2vec。我们使用SGD优化基于图的自定义目标函数，灵感来自于先前在自然语言处理方面的工作[21]。直观地说，我们的方法返回的特征表示最大限度地保持了 $d$ 维特征空间中节邻域。我们使用二阶随机游走方法为节点生成(样本)网络邻域。我们的主要贡献在于定义了灵活的节点网络邻域概念。通过选择适当的邻域概念，node2vec可以学习根据节点的网络角色和/或它们所属的社区来组织节点的表示。我们通过开发一系列有偏的随机游走来实现这一目标，它可以有效地探索给定节点的不同邻域。结果算法是灵活的，使我们能够通过可调参数来控制搜索空间，而不是以前工作中严格的搜索过程[24，28]。因此，我们的方法可以概括先前的工作，并且可以对网络中观察到的等效范围进行建模。控制我们搜索策略的参数有一个直观的解释，并使我们倾向于不同的网络探索策略。这些参数还可以通过半监督方式使用一小部分标记数据直接学习。
>
> 我们还将展示如何将单个节点的特征表示扩展到成对的节点（即边）。 为了生成边的特征表示，我们使用简单的二元运算组合学习到的单个节点的特征表示。 这种组合性使n ode2vec 可以进行涉及节点以及边缘的预测任务。

Our experiments focus on two common prediction tasks in networks: a multi-label classification task, where every node is assigned one or more class labels and a link prediction task, where we predict the existence of an edge given a pair of nodes. We contrast the performance of node2vec with state-of-the-art feature learning algorithms [24, 28]. We experiment with several real-world networks from diverse domains, such as social networks, information networks, as well as networks from systems biology. Experiments demonstrate that node2vec outperforms state-of-the-art methods by up to 26.7% on multi-label classification and up to 12.6% on link prediction. The algorithm shows competitive performance with even 10% labeled data and is also robust to perturbations in the form of noisy or missing edges. Computationally, the major phases of node2vec are trivially parallelizable, and it can scale to large networks with millions of nodes in a few hours.

> 我们的实验集中在网络中两个常见的预测任务上：
>
> - 一个是多标签分类任务，每个节点被分配一个或多个类别标签；
> - 另一个是链接预测任务，在这个任务中，我们在给定一对节点的情况下预测一条边的存在。
>
> 我们将 node2vec 的表现与最先进的特征学习算法[24，28]进行了对比。我们用来自不同领域的几个真实网络进行实验，比如社会网络、信息网络以及来自系统生物学的网络。实验表明，node2vec在多标签分类和链接预测上的表现分别比现有方法高出26.7%和12.6%。该算法在 10% 标记数据的情况下的表现依然很强，并且对噪声或缺失边缘的干扰也具有鲁棒性。在计算上，node2vec 的主要阶段可以轻松并行化，并且可以在几个小时内扩展到具有数百万个节点的大型网络。

Overall our paper makes the following contributions: 

1. We propose node2vec, an efficient scalable algorithm for feature learning in networks that efficiently optimizes a novel network-aware, neighborhood preserving objective using SGD. 
2. We show how node2vec is in accordance with established principles in network science, providing flexibility in discovering representations conforming to different equivalences.
3. We extend node2vec and other feature learning methods based on neighborhood preserving objectives, from nodes to pairs of nodes for edge-based prediction tasks. 
4. We empirically evaluate node2vec for multi-label classification and link prediction on several real-world datasets. 

The rest of the paper is structured as follows. In Section 2, we briefly survey related work in feature learning for networks. We present the technical details for feature learning using node2vec in Section 3. In Section 4, we empirically evaluate node2vec on prediction tasks over nodes and edges on various real-world networks and assess the parameter sensitivity, perturbation analysis, and scalability aspects of our algorithm. We conclude with a discussion of the node2vec framework and highlight some promising directions for future work in Section 5. Datasets and a reference implementation of node2vec are available on the project page: http://snap.stanford.edu/node2vec.

> 综上所述，本文主要做了以下几个方面的工作：
>
> 1. 提出了 node2vec，这是一种在网络中进行特征学习的高效可扩展算法，它利用 SGD 有效地优化了一种新颖的网络感知的邻域保留目标。
> 2. 我们展示了node2vec如何符合网络科学中既定的原则，为发现符合不同等价关系的表示形式提供了灵活性。
> 3. 对于基于边(从节点到节点对)的预测任务，我们扩展了 node2vec 和其他基于邻域保留目标的特征学习方法。
> 4. 我们在几个真实数据集上对 node2vec 的多标签分类和链接预测进行了实证评估。
>
> 论文的其余部分结构如下。
>
> 在第二节中，我们简要综述了网络特征学习的相关工作。
>
> 在第三节中，我们给出了使用 node2vec 进行特征学习的技术细节。
>
> 在第四节中，我们对 node2vec 在各种真实网络的节点和边上的预测任务进行了实证评估，并评估了算法的参数敏感度、扰动分析和可扩展性等方面。
>
> 最后，我们讨论了 node2vec 框架，并强调了第 5 节中未来工作的一些有希望的方向。在项目页面上提供了 node2vec 的数据集和参考实现：http://snap.stanford.edu/node2vec.

## 2. RELATED WORK

Feature engineering has been extensively studied by the machine learning community under various headings. In networks, the conventional paradigm for generating features for nodes is based on feature extraction techniques which typically involve some seed hand-crafted features based on network properties [8, 11]. In contrast, our goal is to automate the whole process by casting feature extraction as a representation learning problem in which case we do not require any hand-engineered features.

Unsupervised feature learning approaches typically exploit the spectral properties of various matrix representations of graphs, especially the Laplacian and the adjacency matrices. Under this linear algebra perspective, these methods can be viewed as dimensionality reduction techniques. Several linear (e.g., PCA) and non-linear (e.g., IsoMap) dimensionality reduction techniques have been proposed [3, 27, 30, 35]. These methods suffer from both computational and statistical performance drawbacks. In terms of computational efficiency, eigendecomposition of a data matrix is expensive unless the solution quality is significantly compromised with approximations, and hence, these methods are hard to scale to large networks. Secondly, these methods optimize for objectives that are not robust to the diverse patterns observed in networks (such as homophily and structural equivalence) and make assumptions about the relationship between the underlying network structure and the prediction task. For instance, spectral clustering makes a strong homophily assumption that graph cuts will be useful for classification [29]. Such assumptions are reasonable in many scenarios, but unsatisfactory in effectively generalizing across diverse networks.

> 机器学习社区已在各种名义下对特征工程进行了广泛的研究。 在网络中，用于生成节点特征的常规范例是基于特征提取技术的，该技术通常包含一些基于网络属性的种子手撕的特征[8、11]。相反，我们的目标是通过将特征提取转换为表示学习问题来使整个过程自动化，在这种情况下，我们不需要任何手工设计的特征。
>
> 无监督特征学习方法通常利用图的各种矩阵表示的谱特性，特别是拉普拉斯矩阵和邻接矩阵。在线性代数的角度来看，这些方法可以看作是降维技术。已经提出了几种线性(例如，PCA)和非线性(例如，ISOMAP)降维技术[3，27，30，35]。这些方法在计算和统计性能上都存在缺陷。在计算效率方面，矩阵的特征分解是代价高昂的，除非通过近似方法，但这样会显著降低解的质量，因此，这些方法很难扩展到大型网络。其次，这些方法的优化目标对从网络中观察到的不同模式(例如同质性和结构等价性)并不健壮，并对底层网络结构和预测任务之间的关系进行假设。例如，谱聚类做了一个很强的同质性假设，即图割将对分类有用[29]。这些假设在许多情况下都是合理的，但在有效地推广到不同的网络时并不令人满意。

Recent advancements in representational learning for natural language processing opened new ways for feature learning of discrete objects such as words. In particular, the Skip-gram model [21] aims to learn continuous feature representations for words by optimizing a neighborhood preserving likelihood objective. The algorithm proceeds as follows: It scans over the words of a document, and for every word it aims to embed it such that the word’s features can predict nearby words (i.e., words inside some context window). The word feature representations are learned by optmizing the likelihood objective using SGD with negative sampling [22]. The Skip-gram objective is based on the distributional hypothesis which states that words in similar contexts tend to have similar meanings [9]. That is, similar words tend to appear in similar word neighborhoods.

Inspired by the Skip-gram model, recent research established an analogy for networks by representing a network as a “document” [24, 28]. The same way as a document is an ordered sequence of words, one could sample sequences of nodes from the underlying network and turn a network into a ordered sequence of nodes. However, there are many possible sampling strategies for nodes, resulting in different learned feature representations. In fact, as we shall show, there is no clear winning sampling strategy that works across all networks and all prediction tasks. This is a major shortcoming of prior work which fail to offer any flexibility in sampling of nodes from a network [24, 28]. Our algorithm node2vec overcomes this limitation by designing a flexible objective that is not tied to a particular sampling strategy and provides parameters to tune the explored search space (see Section 3).

Finally, for both node and edge based prediction tasks, there is a body of recent work for supervised feature learning based on existing and novel graph-specific deep network architectures [15, 16, 17, 31, 39]. These architectures directly minimize the loss function for a downstream prediction task using several layers of non-linear transformations which results in high accuracy, but at the cost of scalability due to high training time requirements.

> 用于自然语言处理的表示学习的最新进展为诸如单词等离散对象的特征学习开辟了新的途径。特别地，Skip-gram模型[21]旨在通过优化邻域保留的似然目标来学习单词的连续特征表示。该算法如下进行：它扫描文档中的单词，并针对每个单词将其嵌入，以便单词的特征可以预测附近的单词(即，某个上下文窗口中的单词)。通过使用具有负采样的 SGD 来优化似然目标来学习单词特征表示[22]。skip-gram 的目标是基于分布假设，该假设指出，在相似的上下文中的单词往往具有相似的含义[9]。也就是说，相似的单词往往出现在相似的单词邻域中。
>
> 受Skip-gram模型的启发，最近的研究为网络建立了一个类比，将网络表示为一份“文档”[24，28]。与文档相同的方法是按顺序排列的单词序列，可以从底层网络中采样节点序列，然后将网络转变为按有序的节点序列。然而，节点有许多可能的采样策略，导致学习到不同的特征表示。事实上，正如我们将要展示的那样，没有明确的获胜采样策略可在所有网络和所有预测任务中使用。 这是现有工作的主要缺点，现有工作不能提供来自网络[24，28]的节点采样的任何灵活性。 我们的算法node2vec通过设计一个与特定采样策略无关的灵活目标来克服了这一限制，并提供了用于调整探索的搜索空间的参数（请参见第3节）。
>
> 最后，对于基于节点和基于边的预测任务，基于现有的和新的特定于图的深度网络结构的有监督的特征学习有大量的最新工作[15，16，17，31，39]。这些结构使用几层非线性变换直接最小化下游预测任务的损失函数，这样精度很高，代价是计算时间更高和可扩展性。

## 3. Feature Learning Framework

We formulate feature learning in networks as a maximum likelihood optimization problem. Let $G = (V , E )$ be a given network.Our analysis is general and applies to any (un)directed, (un)weighted network. Let $f$ : $V → R^d$ be the mapping function from nodes to feature representaions we aim to learn for a downstream prediction task. Here $d$ is a parameter specifying the number of dimensions of our feature representation. Equivalently, $f$ is a matrix of size $|V | \times d$ parameters. For every source node $u \in V$ , we define $N_s (u) ⊂ V$ as a *network neighborhood* of node $u$ generated through a neighborhood sampling strategy $S$. We proceed by extending the Skip-gram architecture to networks [21, 24]. We seek to optimize the following objective function, which maximizes the log-probability of observing a network neighborhood $N_S(u)$ for a node u conditioned on its feature representation, given by $f$:

$$
\mathop{max}\limits_{f} \sum_{u \in V}log \ Pr(N_s(u)|f(u)) 
\qquad (1)
$$

> 我们将网络中的特征学习形式化为最大似然优化问题。 令 $ G =（V，E）$ 为给定网络。我们的分析是一般性的，适用于任何图(有向/无向)，无权/加权 网络。 令 $f : V→\mathbb R ^ d $是我们要为下游预测任务学习的从 Node 到特征表示的映射函数。这里 $ d $ 表示用于特征表示所需的维度数量的参数。 同样，$ f $ 是大小为 $| V |\times d $ 的参数矩阵。 对于每个源节点 $u \in V$ ，我们将 $N_s (u) \subset V$ 定义为通过邻域采样策略 $S$ 生成的节点 $u$ 的网络邻域。我们将 Skip-gram扩展到网络[21，24]。 我们优化以下目标函数，该函数使节点 $u$ 在其特征表示条件下观察网络邻域 $N_S(u)$ 的对数概率最大化，由 $f$ 给出:
>
> $$
>\mathop{max}\limits_{f} \sum_{u \in V}log \ Pr(N_s(u)|f(u)) 
> \qquad (1)
>$$

In order to make the optimization problem tractable, we make two standard assumptions:

-  Conditional independence. 

  We factorize the likelihood by assuming that the likelihood of observing a neighborhood node is independent of observing any other  neighborhood node given the feature representation of the source:

$$
Pr(N_s(u)|f(u)) = \prod_{n_i \in N_S(u)}Pr(n_i|f(u))
$$

- Symmetry in feature space.  

  A source node and neighborhood node have a symmetric effect over each other in feature space. Accordingly, we model the conditional likelihood of every source-neighborhood node pair as a softmax unit parametrized by a dot product of their features:
  $$
  Pr(n_i|f(u)) = \frac{exp(f(n_i) \cdot f(u))}{\sum_{v \in V}exp(f(v) \cdot f(u))}
  $$

With the above assumptions,  the objective in Eq. 1 simplifies to:
$$
\mathop{max}\limits_{f} \sum_{u \in V}[-logZ_u + \sum_{n_i \in N_S(u)}f(n_i)f(u)] \qquad(2)
$$
The per-node partition function, $Z_u = \sum_{u \in V}exp(f(u)f(v))$ , is expensive to compute for large networks and we approximate it using negative sampling [22]. We optimize Eq.  2 using stochastic gradient ascent over the model parameters defining the features f .

Feature learning methods based on the Skip-gram architecture have been originally developed in the context of natural language [21]. Given the linear nature of text, the notion of a neighborhood can be naturally defined using a sliding window over consecutive words. Networks, however, are not linear, and thus a richer notion of a neighborhood is needed. To resolve this issue, we propose a randomized procedure that samples many different neighborhoods of a given source node $u$. The neighborhoods $N_S(u)$ are not restricted to just immediate neighbors but can have vastly different structures depending on the sampling strategy $S$.

> 为了使优化问题易于处理，我们做两个标准假设:
>
> - 条件独立假设。
>
>   我们通过假设观察一个邻域节点的可能性独立于观察任何其他邻域节点来对似然进行因式分解，给定源的特征表示:
>   $$
>   Pr(N_s(u)|f(u)) = \prod_{n_i \in N_S(u)}Pr(n_i|f(u))
>   $$
>
> - 特征空间的对称性。
>
>   源节点和邻域节点在特征空间中具有对称效应。因此，我们将每个 源-邻域节点对的条件似然建模为一个 softmax 单元，由其特征的点积参数化:
>   $$
>   Pr(n_i|f(u)) = \frac{exp(f(n_i) \cdot f(u))}{\sum_{v \in V}exp(f(v) \cdot f(u))}
>   $$
>   通过以上假设，将 Eq. 1 中的目标简化为:
>   $$
>   \mathop{max}\limits_{f} \sum_{u \in V}[-logZ_u + \sum_{n_i \in N_S(u)}f(n_i)f(u)] \qquad(2)
>   $$
>   对于大型网络，每个节点的 partition function $Z_u = \sum_{u \in V}exp(f(u)f(V))$ 计算起来代价昂贵，我们使用 Negative Sampling[22] 来近似它。我们在定义特征 $f$ 的模型参数上使用随机梯度上升来优化公式(2)。
>
>   基于 skip-gram 的特征学习方法最初是在自然语言处理[21]中开发的。鉴于文本的线性特性，可以使用连续单词上的滑动窗口自然地定义邻域的概念。然而，网络不是线性的，因此需要更丰富的邻域概念。为了解决这个问题，我们提出了一个随机过程，即对给定源节点$u$ 的许多不同邻域进行采样。
>
>   邻域 $N_S(u)$ 不限于直接相连的邻域，不同的采样策略 $S$ 可以在结构上有很大差别。

## 3.1 Classic search strategies

We view the problem of sampling neighborhoods of a source node as a form of local search. Figure 1 shows a graph, where given a source node $u$ we aim to generate (sample) its neighborhood $N_S(u)$.

Importantly, to be able to fairly compare different sampling strategies $S$, we shall constrain the size of the neighborhood set $N_S$ to $k$ nodes and then sample multiple sets for a single node $u$. Generally, there are two extreme sampling strategies for generating neighborhood $set(s)$ $N_S$ of  $k$ nodes：

- **Breadth-first Sampling (BFS)** The neighborhood $N_S$ is restricted to nodes which are immediate neighbors of the source. For example, in Figure 1 for a neighborhood of size $k = 3$, BFS samples nodes $s_1, s_2, s_3$.
- **Depth-first Sampling (DFS)** The neighborhood consists of nodes sequentially sampled at increasing distances from the source node. In Figure 1, DFS samples $s_4, s_5, s_6$.

The breadth-first and depth-first sampling represent extreme scenarios in terms of the search space they explore leading to interesting implications on the learned representations.

> 我们把采样源节点邻域的问题看作是一种局部搜索的形式。图1显示了给定一个源节点 $u$，生成 (示例) 它的邻域$N_S(u)$。
>
> 重要的是，为了能够公平地比较不同的采样策略，我们将限制邻域集合 $N_s$ 的大小为 $k$ 个节点，然后对单个节点采样多个集合。一般来说，产生 $k$ 个节点的邻域 $set(s)$ $N_S$的极端采样策略有两种：
>
> - **宽度优先采样(BFS)** 邻域 $N_S$ 被限制为与源相邻的节点。例如，在图1中，对于大小 $k = 3$ 的邻域，BFS抽取节点 $s1, s2, s3$ 。
> - **深度优先采样(DFS)** 邻域由在距离源节点越来越远的地方顺序抽样的节点组成。在图1中，DFS对 $s4、s5、s6$ 进行了采样。
>
> 广度优先采样和深度优先采样代表了他们探索的搜索空间方面的极端情况，从而对表示学习产生了有趣的影响。

In particular, prediction tasks on nodes in networks often shuttle between two kinds of similarities: homophily and structural equivalence [12]. Under the homophily hypothesis [7, 36] nodes that are highly interconnected and belong to similar network clusters or communities should be embedded closely together (e.g., nodes $s_1$ and $u$ in Figure 1 belong to the same network community). In contrast, under the structural equivalence hypothesis [10] nodes that have similar structural roles in networks should be embedded closely together (*e.g.*, nodes $u$ and $s_6$ in Figure 1 act as hubs of their corresponding communities). Importantly, unlike homophily, structural equivalence does not emphasize connectivity; nodes could be far apart in the network and still have the same structural role. In realworld, these equivalence notions are not exclusive; networks commonly exhibit both behaviors where some nodes exhibit homophily while others reflect structural equivalence.

We observe that BFS and DFS strategies play a key role in producing representations that reflect either of the above equivalences. In particular, the neighborhoods sampled by BFS lead to embeddings that correspond closely to structural equivalence. Intuitively, we note that in order to ascertain structural equivalence, it is often sufficient to characterize the local neighborhoods accurately. For example, structural equivalence based on network roles such as bridges and hubs can be inferred just by observing the immediate neighborhoods of each node. By restricting search to nearby nodes, BFS achieves this characterization and obtains a microscopic view of the neighborhood of every node. Additionally, in BFS, nodes in the sampled neighborhoods tend to repeat many times. This is also important as it reduces the variance in characterizing the distribution of 1-hop nodes with respect the source node. However, a very small portion of the graph is explored for any given $k$.

The opposite is true for DFS which can explore larger parts of the network as it can move further away from the source node $u$ (with sample size $k$ being fixed). In DFS, the sampled nodes more accurately reflect a macroview of the neighborhood which is essential in inferring communities based on homophily. However, the issue with DFS is that it is important to not only infer which node-to-node dependencies exist in a network, but also to characterize the exact nature of these dependencies. This is hard given we have a constrain on the sample size and a large neighborhood to explore, resulting in high variance. Secondly, moving to much greater depths leads to complex dependencies since a sampled node may be far from the source and potentially less representative.

>特别地，网络节点上的预测任务经常在同质性(homophily)和结构等价性(structural equivalence)[12]这两种相似性之间穿梭。**根据同质假设[7，36]，高度互连并属于相似网络集群或社区的节点应紧密地嵌入在一起(例如，图1中的节点 $s_1$ 和 $u$ 属于同一网络社区(network comminity))。****相比之下，在结构等价假设下[10]，在网络中具有相似结构角色的节点应紧密地嵌入在一起（例如，图1中的节点 $u$ 和 $s_6$ 充当其相应社区的hubs(集线器)）。重要的是，与同质性不同，结构等价并不强调连通性；节点可能在网络中相距很远，但仍然具有相同的结构角色。**在现实世界中，这些对等概念不是排他性的。 网络通常表现出两种行为，其中某些节点表现出同质性，而另一些则反映出结构等价。
>
>我们观察到，BFS 和 DFS 策略在产生反映上述等价关系的表示形式中起着关键作用。 具体地说，由 BFS 采样的邻域导致了与结构等价性密切对应的 embedding。 直观来讲，我们注意到，为了确定结构等价性，准确地描述局部邻域关系通常就足够了。
>
>例如，基于网络角色(如 bridge 和 hub)的结构等价性可以仅通过观察每个节点的邻域来推断。通过将搜索限制在邻域节点，BFS 实现了这一特征，并获得了每个节点的邻域的微观视图。此外，在 BFS 中，采样邻域中的节点往往会重复多次。这很重要，因为它减少了 表示 1-hop 节点相对于源节点的分布时的差异。 但是，对于给定的 $ k $，只探索了图中很小的一部分。
>
>对于 DFS，情况恰恰相反，它可以探索网络的更大部分，因为它可以远离源节点 $u$（样本大小 $k$ 固定）。 在 DFS 中，采样的节点可以更准确地反映邻域的宏观视图，这对于基于同质性推断社区至关重要。
>
>然而，DFS的问题在于，不仅要推断网络中存在哪些节点到节点的依赖关系，而且要确定这些依赖关系的确切性质，这一点很重要。这是很难的，因为我们对样本大小有限制，而且要探索的邻域很大，导致了很高的方差。其次，移动到更深的位置会导致复杂的依赖关系，因为采样的节点可能离源很远，可能不太具有代表性。

## 3.2 node2vec

Building on the above observations, we design a flexible neighborhood sampling strategy which allows us to smoothly interpolate between BFS and DFS. We achieve this by developing a flexible biased random walk procedure that can explore neighborhoods in a BFS as well as DFS fashion.

> 在上述观察的基础上，我们设计了一种灵活的邻域采样策略，使我们能够在 BFS 和 DFS 之间平滑地进行插值。我们通过开发一种灵活的有偏的随机游走程序来实现这一点，该程序可以以 BFS 和 DFS 的方式探索社区。

#### 3.2.1 Random Walks

Formally, given a source node $u$, we simulate a random walk of fixed length $l$. Let $c_i$ denote the $i$th node in the walk, starting with $c_0$ = $u$. Nodes $c_i$ are generated by the following distribution:
$$
P(c_i=x|c_{i-1}=v)= \begin{cases}
\frac{\pi_{vx}}{Z},\quad if(v,x) \in E \\
0, \quad otherwise
\end{cases}
$$
where $\pi_{vx}$ is the unnormalized transition probability between nodes $v$ and $x$, and $Z$ is the normalizing constant.

> 形式化地讲：给定一个源节点 $u$，我们模拟长度固定为 $l$ 的随机游走。$c_i$ 表示遍历中的第 $i$ 个节点，从 $c_0 = u$ 开始。节点 $c_i$ 由以下分布生成:
>$$
> P(c_i=x|c_{i-1}=v)= \begin{cases}
>\frac{\pi_{vx}}{Z},\quad if(v,x) \in E \\
> 0, \quad otherwise
>\end{cases}
> $$
> 其中 $\pi_{vx}$ 是节点 $v$ 和 $x$ 之间的非标准化转移概率，$Z$是标准化常数。

#### 3.2.2 Search bias α

The simplest way to bias our random walks would be to sample the next node based on the static edge weights $w_{vx}$ i.e., $π_{vx} = w_{vx}$. (In case of unweighted graphs $w_{vx}$ = 1.) However, this does not allow us to account for the network structure and guide our search procedure to explore different types of network neighborhoods. Additionally, unlike BFS and DFS which are extreme sampling paradigms suited for structural equivalence and homophily respectively, our random walks should accommodate for the fact that these notions of equivalence are not competing or exclusive, and real-world networks commonly exhibit a mixture of both.

We define a 2nd order random walk with two parameters $p$ and $q$ which guide the walk: Consider a random walk that just traversed edge $(t, v)$ and now resides at node $v$ (Figure 2). The walk now needs to decide on the next step so it evaluates the transition probabilities $\pi_{vx}$ on edges $ (v, x) $ leading from $v$. We set the unnormalized transition probability to $\pi_{vx} = \alpha_{pq}(t, x) · w_{vx}$, where and $d_{tx}$ denotes the shortest path distance between nodes $t$ and $x$. Note that $d_{tx}$ must be one of $\{ 0, 1, 2 \}$, and hence, the two parameters are necessary and sufficient to guide the walk.

Intuitively, parameters $p$ and $q$ control how fast the walk explores and leaves the neighborhood of starting node $u$. In particular, the parameters allow our search procedure to (approximately) interpolate between BFS and DFS and thereby reflect an affinity for different notions of node equivalences.
$$
\alpha_{pq}(t,x) = \begin{cases}
\frac{1}{p} \quad if \ d_{tx} = 0\\
1 \quad if \ d_{tx} = 1\\
\frac{1}{q} \quad if \ d_{tx} = 2
\end{cases}
$$

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Node2vecScalableFeatureLearningforNetworks/Fig2.png)

**Figure 2: Illustration of the random walk procedure in node2vec. The walk just transitioned from $t$ to $v$ and is now evaluating its next step out of node $v$. Edge labels indicate search biases $\alpha$.**

> 使我们的随机游走成为有偏随机游走的最简单的方法是基于边的静态权值 $w_{vx}$ 对下一个节点进行采样，即，$v_x = w_{vx}$。(对于无权图，$w_{vx} = 1$)
>
> 然而，这并不允许我们解释网络结构，并指导我们的搜索过程探索不同类型的网络社区。此外，不像 BFS 和 DFS 是分别适用于结构等价和同质性的极端抽样范例，我们的随机游走应该适应这样一个事实，即这些等价的概念不是竞争的或排他性的，而现实网络通常表现出两者的混合。
>
> 我们定义一个带有两个参数 $p$ 和 $q$ 的二阶随机游走，以指导游走：考虑一个仅已经游走了边 $(t，v)$ 并现在位于节点 $v$ 处的随机游走（图2）。 游走现在需要决定下一步，以便评估从 $v$ 开始的边 $(v，x)$ 上的转移概率 $\pi_{vx}$。我们将未归一化的转移概率设置为 $\pi_{vx} = \alpha_{pq}(t,x)·w_{vx}$，其中 $d_{tx}$ 表示节点 $t$ 和 $x$ 之间的最短路径距离。请注意，$d_{tx}$ 必须是 $\{0，1，2\}$ 之一，因此，这两个参数对于引导游走是必要且足够的。 
>
> 直观地，参数 $p$ 和 $q$ 控制游走探索并离开起始节点 $u$ 的邻域的速度。 特别是，这些参数允许我们的搜索过程在 BFS 和 DFS 之间（大约）插值，从而反映对节点等效性不同概念的相似性。
> $$
> \alpha_{pq}(t,x) = \begin{cases}
> \frac{1}{p} \quad if \ d_{tx} = 0\\
> 1 \quad if \ d_{tx} = 1\\
> \frac{1}{q} \quad if \ d_{tx} = 2
> \end{cases}
> $$

#### Return parameter, $p$​.

Parameter $p$​ controls the likelihood of immediately revisiting a node in the walk. Setting it to a high value $(> max(q, 1))$ ensures that we are less likely to sample an already visited node in the following two steps (unless the next node in the walk had no other neighbor). This strategy encourages moderate exploration and avoids 2-hop redundancy in sampling. On the other hand, if $p$ is low $(< min(q, 1))$, it would lead the walk to backtrack a step (Figure 2) and this would keep the walk “local” close to the starting node $u$​.

>  参数$p$控制在游走中立即重走已走过的节点的可能性。 将其设置为较高的值$(>max(q，1))$可确保我们在接下来两步中减小已访问的节点进行采样的可能（除非遍历中的下一个节点没有其他邻居）。 这种策略鼓励适度的探索，并避免采样中冗余的 2-hop 。 另一方面，如果 $p$ 较小 $（<min（q，1））$ ，它将导致游走回溯一步（图2），这将使游走“局部”靠近起始节点 $u$。

#### In-out parameter, q.

Parameter $q$ allows the search to differentiate between “inward” and “outward” nodes. Going back to Figure 2, if $q > 1$, the random walk is biased towards nodes close to node $t$. Such walks obtain a local view of the underlying graph with respect to the start node in the walk and approximate BFS behavior in the sense that our samples comprise of nodes within a small locality. In contrast, if $q < 1$, the walk is more inclined to visit nodes which are further away from the node $t$. Such behavior is reflective of DFS which encourages outward exploration. However, an essential difference here is that we achieve DFS-like exploration within the random walk framework. Hence, the sampled nodes are not at strictly increasing distances from a given source node $u$, but in turn, we benefit from tractable preprocessing and superior sampling efficiency of random walks. Note that by setting $π_{v,x}$ to be a function of the preceeding node in the walk $t$, the random walks are 2nd order Markovian.

>参数 $q$ 允许搜索区分“向内”和“向外”节点。 见图2，如果 $q>1$，则随机游走偏向靠近节点 $t$ 的节点。
>
>这样的遍历获得了关于遍历中开始节点的的局部图，并且从我们的样本由小局部内的节点组成的意义上来说，获得了近似的BFS行为。
>
>相反，如果 $q <1$，则游走更倾向于访问距离节点 $t$ 较远的节点。 这种行为反映了 DFS，它鼓励向更远探索。 但是，这里的本质区别是我们在随机游走框架内实现了类似 DFS(search) 的探索。 因此，被采样的节点与给定源节点 $u$ 的距离并不是严格增加的，但反过来，我们受益于易于处理的预处理和随机游走的高采样效率。 注意，通过将 $\pi_{v,x}$ 设置为游走 $t$ 中先前节点的函数，随机游走是 2阶 马尔可夫。

#### Benefits of random walks. 

There are several benefits of random walks over pure BFS/DFS approaches. Random walks are computationally efficient in terms of both space and time requirements. The space complexity to store the immediate neighbors of every node in the graph is $O(|E|)$. For $2^{nd}$ order random walks, it is helpful to store the interconnections between the neighbors of every node, which incurs a space complexity of $O(a^2 |V |)$ where $a$ is the average degree of the graph and is usually small for realworld networks. 

The other key advantage of random walks over classic search-based sampling strategies is its time complexity. In particular, by imposing graph connectivity in the sample generation process, random walks provide a convenient mechanism to increase the effective sampling rate by reusing samples across different source nodes. By simulating a random walk of length $l > k$ we can generate $k$ samples for $l − k$ nodes at once due to the Markovian nature of the random walk. Hence, our effective complexity is $O(\frac{l}{k(l−k)})$  per sample. 

For example, in Figure 1 we sample a random walk $\{u, s4, s5, s6, s8, s9\}$ of length $l = 6$, which results in $N_S(u) = \{s_4, s_5, s_6\}$, $N_S(s_4) = \{s_5, s_6, s_8\}$ and $N_S(s_5) = \{s_6, s_8, s_9\}$. Note that sample reuse can introduce some bias in the overall procedure. However, we observe that it greatly improves the efficiency.

> 与纯 BFS/DFS 方法相比，随机游走有几个好处。就空间和时间要求而言，随机游走的计算效率很高。存储图中每个节点的近邻的空间复杂度为 $O(|E|)$ 。对于二阶随机游走，存储每个节点的邻居之间的互连是有帮助的，这会导致空间复杂度为 $O(a^2 | V |)$，其中 $a$ 是图的平均度， 在真实世界的网络通常较小。
>
> 与传统的基于搜索的抽样策略相比，随机游走的另一个关键优势是它的时间复杂度。特别是，通过在样本生成过程中引入图的连通性，随机游走提供了一种方便的机制，通过在不同源节点间重用样本来提高有效采样率。
>
> 通过模拟一个长度为 $l > k$ 的随机行走，由于随机游走的 Markovian 性质，我们可以为 $l - k$ 个节点一次生成 $k$ 个样本。因此,我们有效的复杂度是每个样本 $O(\frac{l}{k(l−k)})$。
>
> 例如，在图1中，我们对长度 $l = 6$ 的随机游走 $\{u, s4, s5, s6, s8, s9\}$ 进行采样，得到 $N_S(u) = \{s_4, s_5, s_6\}$， $N_S(s_4) = \{s_5, s_6, s_8\}$ 和 $N_S(s_5) = \{ s_6, s_8, s_9 \}$。注意，样本重用可能会在整个过程中引入一些偏差。然而，我们观察到它极大地提高了效率。

### **3.2.3 The node2vec algorithm**

The pseudocode for node2vec, is given in Algorithm 1. In any random walk, there is an implicit bias due to the choice of the start node $u$. Since we learn representations for all nodes, we offset this bias by simulating $r$ random walks of fixed length $l$ starting from every node. At every step of the walk, sampling is done based on the transition probabilities $\pi_{vx}$.

The transition probabilities $\pi_{vx}$ for the 2nd order Markov chain can be precomputed and hence, sampling of nodes while simulating the random walk can be done efficiently in $O(1)$  time using alias sampling. 

The three phases of node2vec, i.e., preprocessing to compute transition probabilities, random walk simulations and optimization using SGD, are executed sequentially. Each phase is parallelizable and executed asynchronously, contributing to the overall scalability of node2vec. 

node2vec is available at: http://snap.stanford.edu/node2v

>算法 1 给出了 node2vec 的伪代码。在任意随机游走中，由于起始节点 $u$ 的选择，都会存在隐性偏差。由于我们学习了所有节点的表示，因此我们通过模拟从每个节点开始的固定长度为 $l$ 的 $r$ 个随机游走来抵消此偏差。在游走的每一步，采样都是基于转移概率 $\pi_{vx}$。
>
>可以预先计算二阶马尔可夫链的转移概率 $\pi_{vx}$，因此，可以使用 Alias sampling 在 $O(1)$ 内高效地完成模拟随机游走时的节点采样。
>
>node2vec 的三个阶段依次执行，即转移概率预计算、随机游走模拟和使用SGD 进行优化。每个阶段都是可并行的，并异步执行，这有助于 node2vec的整体可扩展性。
>
>node2vec的下载地址是：http://snap.stanford.edu/node2v

## 3.3 Learning edge features

The node2vec algorithm provides a semi-supervised method to learn rich feature representations for nodes in a network. However, we are often interested in prediction tasks involving pairs of nodes instead of individual nodes. For instance, in link prediction, we predict whether a link exists between two nodes  in a network. Since our random walks are naturally based on the connectivity struct e between nodes in the underlying network, we extend them to pairs of nodes using a bootstrapping approach over the feature representations of the individual nodes.

Given two nodes $u$ and $v$, we define a binary operator $\circ$ over the corresponding feature vectors $f(u)$ and $f(v)$ in order to generate a representation $g(u, v)$ such that  $g : V \times V \rightarrow R^{d^{'}}$ where $d^{'}$ is the representation size for the pair $(u, v)$. We want our operators to be generally defined for any pair of nodes, even if an edge does not exist between the pair since doing so makes the representations useful for link prediction where our test set contains both true and false edges (i.e., do not  exist). We consider several choices for the operator $\circ$ such that $d^{'} =  d$ which are summarized in Table 1.

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/Node2vecScalableFeatureLearningforNetworks/Table1.png)

**Table 1: Choice of binary operators $\circ$ for learning edge features. The definitions correspond to the ith component of $g(u, v)$.**

> Node2vec算法提供了一种半监督方法来学习网络中节点的丰富特征表示。
> 然而，我们通常对涉及成对节点而不是单个节点的预测任务感兴趣。
> 例如，在链接预测中，我们预测网络中的两个节点之间是否存在链路。
> 由于我们的随机游走天生基于底层网络中节点之间的连通性结构，因此我们使用 bootstrapping 方法将其扩展到成对的节点，而不是单个节点的特征表示。
>
> 给定两个节点 $u$ 和 $v$ ，我们在相应的特征向量 $f(u)$ 和 $f(v)$ 上定义二元运算$\circ$，以便生成表示 $g(u，v)$，使得 $g : V \times V \rightarrow R^{d^{'}}$ 其中 $d^{'}$ 是对 $(u，v)$ 的表示维度。我们希望为任何一对节点定义我们的运算符，即使在这对节点之间不存在边，因为这样做使得表示对于我们的测试集既包含真边又包含假边(即，不存在)的链接预测是有用的。我们考虑运算符 $\circ$ 的几种选择，使 $d^{'}=d$，如表1所示。3

