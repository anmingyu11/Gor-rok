> bperozzi，ralrfou，skiena，纽约州立大学石溪分校

# DeepWalk: Online Learning of Social Representations

## ABSTRACT

We present DeepWalk, a novel approach for learning latent representations of vertices in a network. These latent representations encode social relations in a continuous vector space, which is easily exploited by statistical models. DeepWalk generalizes recent advancements in language modeling and unsupervised feature learning (or deep learning) from sequences of words to graphs.

DeepWalk uses local information obtained from truncated random walks to learn latent representations by treating walks as the equivalent of sentences. We demonstrate DeepWalk’s latent representations on several multi-label network classification tasks for social networks such as BlogCatalog, Flickr, and YouTube. Our results show that DeepWalk outperforms challenging baselines which are allowed a global view of the network, especially in the presence of missing information. DeepWalk’s representations can provide F1 scores up to 10% higher than competing methods when labeled data is sparse. In some experiments, DeepWalk’s representations are able to outperform all baseline methods while using 60% less training data.

DeepWalk is also scalable. It is an online learning algorithm which builds useful incremental results, and is trivially parallelizable. These qualities make it suitable for a broad class of real world applications such as network classification, and anomaly detection.

> 我们提出了一种新的学习网络中顶点的潜在表示的方法 DeepWalk。这些的 表示将社会关系编码在一个连续的向量空间中，这很容易被统计学习模型所学习。DeepWalk 通过将单词序列转化成图推动了语言模型和无监督特征学习(或深度学习)的最新改进。
>
> DeepWalk 通过使用截断 random walk 获得的局部信息，将 walk 等同于句子来学习潜在表示。我们展示了 DeepWalk 在 BlogCatalog、Flickr 和YouTube等社交网络的几个多标签 network 分类任务上的潜在表示。我们的研究结果表明，DeepWalk 优于具有挑战性的 baseline，该 baseline 允许对 network 进行全局观察，特别是在存在缺失信息的情况下。当标签数据稀疏时，DeepWalk 的表示法可以提供比对照方法高出 10% 的 F1-score。在一些实验中，DeepWalk 的表示能够在使用少于 60% 的训练数据的情况下优于所有 baseline 方法。
>
> DeepWalk 也是可扩展的。 它是一种在线学习算法，可以建立有用的增量结果，并且可以并行化。 这些性质使其适合各种实际应用，例如网络分类和异常检测。

## 1. INTRODUCTION

The sparsity of a network representation is both a strength and a weakness. Sparsity enables the design of efficient discrete algorithms, but can make it harder to generalize in statistical learning. Machine learning applications in networks (such as network classification [16, 37], content recommendation [12], anomaly detection [6], and missing link prediction [23]) must be able to deal with this sparsity in order to survive.

In this paper we introduce deep learning (unsupervised feature learning) [3] techniques, which have proven successful in natural language processing, into network analysis for the first time. We develop an algorithm (DeepWalk) that learns social representations of a graph’s vertices, by modeling a stream of short random walks. Social representations are latent features of the vertices that capture neighborhood similarity and community membership. These latent representations encode social relations in a continuous vector space with a relatively small number of dimensions. DeepWalk generalizes neural language models to process a special language composed of a set of randomly-generated walks. These neural language models have been used to capture the semantic and syntactic structure of human language [7], and even logical analogies [29].

DeepWalk takes a graph as input and produces a latent representation as an output. The result of applying our method to the well-studied Karate network is shown in Figure 1. The graph, as typically presented by force-directed layouts, is shown in Figure 1a. Figure 1b shows the output of our method with 2 latent dimensions. Beyond the striking similarity, we note that linearly separable portions of (1b) correspond to clusters found through modularity maximization in the input graph (1a) (shown as vertex colors).

To demonstrate DeepWalk’s potential in real world scenarios, we evaluate its performance on challenging multi-label network classification problems in large heterogeneous graphs. In the relational classification problem, the links between feature vectors violate the traditional i.i.d. assumption. Techniques to address this problem typically use approximate inference techniques [32] to leverage the dependency information to improve classification results. We distance ourselves from these approaches by learning label-independent representations of the graph. Our representation quality is not influenced by the choice of labeled vertices, so they can be shared among tasks.

DeepWalk outperforms other latent representation methods for creating social dimensions [39, 41], especially when labeled nodes are scarce. Strong performance with our representations is possible with very simple linear classifiers (e.g. logistic regression). Our representations are general, and can be combined with any classification method (including iterative inference methods). DeepWalk achieves all of that while being an online algorithm that is trivially parallelizable. Our contributions are as follows:

- We introduce deep learning as a tool to analyze graphs, to build robust representations that are suitable for statistical modeling. DeepWalk learns structural regularities present within short random walks.
- We extensively evaluate our representations on multilabel classification tasks on several social networks. We show significantly increased classification performance in the presence of label sparsity, getting improvements 5%-10% of Micro F1, on the sparsest problems we consider. In some cases, DeepWalk’s representations can outperform its competitors even when given 60% less training data. 
- We demonstrate the scalability of our algorithm by building representations of web-scale graphs, (such as YouTube) using a parallel implementation. Moreover, we describe the minimal changes necessary to build a streaming version of our approach.

The rest of the paper is arranged as follows. In Sections 2 and 3, we discuss the problem formulation of classification in data networks, and how it relates to our work. In Section 4 we present DeepWalk, our approach for Social Representation Learning. We outline ours experiments in Section 5, and present their results in Section 6. We close with a discussion of related work in Section 7, and our conclusions.

![Fig1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Fig1.png)

**Figure 1: Our proposed method learns a latent space representation of social interactions in $\mathbb{R}^d$ . The learned representation encodes community structure so it can be easily exploited by standard classification methods. Here, our method is used on Zachary’s Karate network [44] to generate a latent representation in $\mathbb{R}^2$ . Note the correspondence between community structure in the input graph and the embedding. Vertex colors represent a modularity-based clustering of the input graph.**

> network 表示的稀疏性既是优点也是缺点。稀疏性可以设计高效的离散算法，但也使其难以在统计学习中推广。机器学习在 network 中的应用(如 network 分类[16,37]、内容推荐[12]、异常检测[6]和缺失链接预测[23])必须能够处理这种稀疏性才能生存。
>
> 在本文中，我们首次将深度学习(无监督特征学习)[3]技术引入到 network 分析中，这些技术在 NLP 中已经被证明是成功的。我们开发了一个算法(DeepWalk)，它通过模拟一系列短的随机游走来学习图的顶点的 social 表示。social 表示是顶点的潜在特征，可以捕捉邻域相似性和社区成员关系。这些潜在表示将社会关系编码在维数相对较少的连续向量空间中。DeepWalk 将神经语言模型推广到处理由一组由随机游走生成的 walk 组成的特殊语言。这些神经语言模型已经被用来捕捉人类语言的语义和语法结构[7]，甚至逻辑类比[29]。
>
> DeepWalk 将图作为输入，并生成潜在表示作为输出。将我们的方法应用于经过充分研究的 Karate network 的结果如图1所示。图1a显示了常见的由 force-directed layouts 表示的图。图1b显示了具有 2个潜在维度的方法的输出。除了惊人的相似性之外，我们注意到(1b)的线性可分离部分对应于在输入图(1a)中通过 modularity maximization 找到的簇(显示为顶点颜色)。
>
> 为了展示 DeepWalk 在现实世界场景中的潜力，我们评估了它在大型异构图中挑战多标签 network 分类问题上的表现。在关系分类问题中，特征向量之间的联系违背了传统的 i.i.d.假设。解决这个问题的技术通常使用近似推理技术[32]来利用依赖信息来改善分类结果。我们通过学习图的标签无关表示来远离这些方法。我们的表示质量不受顶点的标签的影响，因此它们可以在任务之间共享。
>
> DeepWalk 在创建 Social 维度方面优于其他潜在表示方法[39，41]，特别是在 labeld 节点稀缺的情况下。使用非常简单的线性分类器(例如，Logistic回归)，我们的表示就有可能实现出色的性能。我们的表示是通用的，可以与任何分类方法(包括迭代推理方法)相结合。DeepWalk实现了所有这一切，同时也是一种可以并行化的在线算法。我们的贡献如下：
>
> - 我们引入深度学习作为分析 graph 的工具，以构建适合统计学习建模的健壮表示。DeepWalk 学习短随机游走中存在的结构规律性。
> - 我们在几个社交网络上广泛评估了我们在多标签分类任务上的表示。在标签稀疏的情况下，我们表现出显著的分类性能提高，在我们考虑的最稀疏的问题上，得到了 Micro F1 的 5% - 10% 的改进。在某些情况下，即使训练数据少于 60%，DeepWalk 的表现也能超过竞争对手。
> - 我们通过使用并行实现构建 web-scale 图的表示(如YouTube)来演示算法的可扩展性。此外，我们还描述了构建我们的方法的 stream 版本所需的最小更改。
>
> 论文的其余部分安排如下。在第 2 节和第 3 节中，我们讨论 network 中分类的问题表述，以及它与我们的工作的关系。在第四节中，我们介绍了 Social 表示学习方法 DeepWalk。第 5 节概述了我们的实验，并在第 6 节介绍了结果。最后，我们在第 7 节讨论了相关工作，并得出了我们的结论。

## 2. PROBLEM DEFINITION

We consider the problem of classifying members of a social network into one or more categories. Let $G = (V, E)$, where $V$ represent the members of the network, $E$ are their connections, $E \subseteq (V \times V )$, and $G_L = (V, E, X, Y)$ is a partially labeled social network, with attributes $X \in \mathbb{R}^{|V |×S}$ where $S$ is the size of the feature space for each attribute vector, and $Y \in R^{|V |×|Y|}$ , $Y$ is the set of labels.

In a traditional machine learning classification setting, we aim to learn a hypothesis $H$ that maps elements of $X$ to the labels set $\mathcal{Y}$. In our case, we can utilize the significant information about the dependence of the examples embedded in the structure of $G$ to achieve superior performance.

In the literature, this is known as the relational classification (or the collective classification problem [37]). Traditional approaches to relational classification pose the problem as an inference in an undirected Markov network, and then use iterative approximate inference algorithms (such as the iterative classification algorithm [32], Gibbs Sampling [15], or label relaxation [19]) to compute the posterior distribution of labels given the network structure.

We propose a different approach to capture the network topology information. Instead of mixing the label space as part of the feature space, we propose an unsupervised method which learns features that capture the graph structure independent of the labels’ distribution.

This separation between the structural representation and the labeling task avoids cascading errors, which can occur in iterative methods [34]. Moreover, the same representation can be used for multiple classification problems concerning that network.

Our goal is to learn $X_E \in R^{|V|×d}$ , where $d$ is small number of latent dimensions. These low-dimensional representations are distributed; meaning each social phenomena is expressed by a subset of the dimensions and each dimension contributes to a subset of the social concepts expressed by the space.

Using these structural features, we will augment the attributes space to help the classification decision. These features are general, and can be used with any classification algorithm (including iterative methods). However, we believe that the greatest utility of these features is their easy integration with simple machine learning algorithms. They scale appropriately in real-world networks, as we will show in Section 6.

> 考虑将一个社会网络中的成员划分为一个或多个类别的问题。
>
> 设 $G=(V，E)$，其中 $V$ 代表 network 成员，$E$ 是他们的连接，$E \subseteq (V \times V)$，$G_L=(V，E，X，Y)$ 是一个部分标记了的社交网络，属性$X\in\mathbb{R}^{|V|×S}$ 其中 $S$ 是每个属性向量的特征空间大小，$Y\in R^{|V| \times |\mathcal{Y}|}$ , $\mathcal{Y}$ 是标签集。
>
> 在传统的机器学习分类设置中，我们的目标是学习一个假设 $H$，它将 $X$ 的元素映射到 $\mathcal{Y}$ 的标签集。在我们的例子中，我们可以利用嵌入在 $G$ 结构中的例子上的的重要依赖信息来实现卓越的表现。
>
> 在文献中，这被称为关系分类（或集体分类问题[37]）。关系分类的传统方法将问题视为无向马尔可夫网络中的推理，然后使用迭代近似推理算法（如迭代分类算法[32]、吉布斯采样[15]或标签松弛[19]）来计算给定 network 的标签的后验分布。
>
> 我们提出了一种不同的方法来捕获 network 的拓扑信息。我们提出了一种无监督方法，该方法学习捕获与标签分布无关的图结构的特征，而不是将标签空间混合为特征空间的一部分。
>
> 结构表示和标记任务之间的这种分离避免了迭代方法[34]中可能出现的 cascading 错误。此外，相同的表示可以用于与该 network 相关的多个分类问题。
>
> 我们的目标是学习 $X_E \in R^{|V|×d} $ ，其中 $d$ 是少量的潜在维度。这些低维表示是分布式的；这意味着每个 social phenomena 都由维度的子集来表示，每个维度都贡献了空间所表达的社会概念的子集。
>
> 使用这些结构特征，我们将扩大属性空间以帮助分类决策。这些特征是通用的，可以用于任何分类算法 (包括迭代方法)。然而，我们认为这些特征的最大效用是它们易于与简单的机器学习算法集成。正如我们将在第 6 节中介绍的那样，它们在真实 network 中进行了适当的扩展。

## 3. LEARNING SOCIAL REPRESENTATIONS

We seek to learn social representations with the following characteristics: 

- Adaptability - Real social networks are constantly evolving; new social relations should not require repeating the learning process all over again.
- Community aware - The distance between latent dimensions should represent a metric for evaluating social similarity between the corresponding members of the network. This allows generalization in networks with homophily. 
- Low dimensional - When labeled data is scarce low dimensional models generalize better, and speed up convergence and inference. 
- Continuous - We require latent representations to model partial community membership in continuous space. In addition to providing a nuanced view of community membership, a continuous representation has smooth decision boundaries between communities which allows more robust classification. 

Our method satisfies these requirements by learning representation for vertices from a stream of short random walks, using optimization techniques originally designed for language modeling. Here, we review the basics of both random walks and language modeling, and describe how their combination satisfies our requirements.

> 我们试图学习具有以下特点的 social 表示：
>
> - Adaptability：真正的社交网络是不断发展的；新的社会关系不应该要求从头再来一遍的学习过程。
> - Community aware：潜在维度之间的距离应该代表一个衡量标准，用于评估网络中相应成员之间的 social 相似性。这允许在具有同质性的网络中进行泛化。
> - Low dimensional：当标签数据稀缺时，低维模型可以更好地泛化，并加快收敛和推理。
> - Continuous：我们需要潜在的表示来对连续空间中的部分 social member 进行建模。 除了提供 social member 的细微差别外，连续表示还具有 social 之间的平滑决策边界，从而可以进行更可靠的分类。
>
> 我们的方法通过使用最初为语言建模设计的优化技术，从短随机游走 stream 中学习顶点的表示来满足这些要求。在这里，我们回顾了随机游走和语言建模的基础知识，并描述了它们的组合如何满足我们的要求。

## 3.1 Random Walks

We denote a random walk rooted at vertex $v_i$ as $\mathcal{W}_{v_i}$ . It is a stochastic process with random variables $\mathcal{W}^1_{v_i} , \mathcal{W}^2_{v_i} , \cdots , \mathcal{W}^k_{v_i}$ such that $\mathcal{W}^{k+1}_{v_i}$ is a vertex chosen at random from the neighbors of vertex $v_k$. Random walks have been used as a similarity measure for a variety of problems in content recommendation [12] and community detection [2]. They are also the foundation of a class of output sensitive algorithms which use them to compute local community structure information in time sublinear to the size of the input graph [38].

It is this connection to local structure that motivates us to use a stream of short random walks as our basic tool for extracting information from a network. In addition to capturing community information, using random walks as the basis for our algorithm gives us two other desirable properties. First, local exploration is easy to parallelize. Several random walkers (in different threads, processes, or machines) can simultaneously explore different parts of the same graph. Secondly, relying on information obtained from short random walks make it possible to accommodate small changes in the graph structure without the need for global recomputation. We can iteratively update the learned model with new random walks from the changed region in time sub-linear to the entire graph.

> 我们将 root 是顶点 $v_i$ 的随机游走表示为 $\mathcal{W}_{v_i}$。它是一个包含随机变量 $\mathcal{W}^1_{v_i}， \cdots， \mathcal{W}^2_{v_i} $ 的随机过程，$\mathcal{W}^{k+1}_{v_i}$ 是从顶点 $v_k$ 的邻域中随机选择的顶点(注：这里应该是 $v_i$ 吧？)。在内容推荐[12]和社区检测[2]中，许多问题都使用了随机游走作为相似度量。它们也是一类输出敏感算法的基础，该算法使用与输入图的大小成线性关系的时间复杂度计算 local community 结构信息[38]。
>
> 正是这种与本地结构的联系促使我们使用短随机游走流作为我们从网络中提取信息的基本工具。 除了捕获 community 信息之外，使用随机游走作为我们算法的基础还为我们提供了两个理想的属特性。 首先，局部搜索很容易并行化。 几个随机游走（位于不同的线程，进程或机器中）可以同时浏览同一图的不同部分。 其次，依靠从短随机游走中获得的信息，可以在不需要全局重新计算的情况下适应图结构中的微小变化。 我们可以用新的随机游走迭代更新已学习到的模型时间复杂度同样是图大小的次线性。

## 3.2 Connection: Power laws

Having chosen online random walks as our primitive for capturing graph structure, we now need a suitable method to capture this information. If the degree distribution of a connected graph follows a power law (i.e. scale-free), we observe that the frequency which vertices appear in the short random walks will also follow a power-law distribution.

Word frequency in natural language follows a similar distribution, and techniques from language modeling account for this distributional behavior. To emphasize this similarity we show two different power-law distributions in Figure 2. The first comes from a series of short random walks on a scale-free graph, and the second comes from the text of 100,000 articles from the English Wikipedia.

A core contribution of our work is the idea that techniques which have been used to model natural language (where the symbol frequency follows a power law distribution (or Zipf ’s law)) can be re-purposed to model community structure in networks. We spend the rest of this section reviewing the growing work in language modeling, and transforming it to learn representations of vertices which satisfy our criteria.

> 在选择了在线随机游走作为我们捕获图结构的基元之后，我们现在需要一种合适的方法来捕获这些信息。如果连通图的度分布遵循幂律（i.e. scale-free），我们观察到短随机游游走出现顶点的频率也将遵循幂律分布。
>
> 自然语言中的词频遵循类似的分布，语言建模技术解释了这种分布行为。为了强调这种相似性，我们在图2中显示了两种不同的幂律分布。第一种来自无标度图上的一系列短期随机游走，第二种来自英文维基百科(Wikipedia)100,000篇文章的文本。
>
> 我们工作的一个核心贡献是，用于建模自然语言（其中符号频率遵循幂律分布（或 Zipf's定律））的技术可以被重新用于建模 network 中的 social comunity。我们在本节的其余部分回顾了语言建模方面不断进步的工作，并将其转换为学习满足我们标准的顶点表示。

![Fig2](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Fig2.png)

**Figure 2: The distribution of vertices appearing in short random walks (2a) follows a power-law, much like the distribution of words in natural language (2b).**

## 3.3 Language Modeling

The goal of language modeling is to estimate the likelihood of a specific sequence of words appearing in a corpus. More formally, given a sequence of words $W_1^n = (w_0, w_1, \cdots , w_n)$, where $w_i \in \mathcal{V}$ ( $\mathcal{V}$ is the vocabulary), we would like to maximize the $Pr(w_n|w_0, w_1, \cdots , w_{n−1})$ over all the training corpus. Recent work in representation learning has focused on using probabilistic neural networks to build general representations of words which extend the scope of language modeling beyond its original goals.

In this work, we present a generalization of language modeling to explore the graph through a stream of short random walks. These walks can be thought of as short sentences and phrases in a special language; the direct analog is to estimate the likelihood of observing vertex $v_i$ given all the previous vertices visited so far in the random walk, i.e.
$$
Pr(v_i | (v_1, v_2, \cdots , v_{i−1}))
\qquad (1)
$$
Our goal is to learn a latent representation, not only a probability distribution of node co-occurrences, and so we introduce a mapping function $\Phi$. This mapping $\Phi$ represents the latent social representation associated with each vertex $v$ in the graph. (In practice, we represent $\Phi$ by a $|V|× d$ matrix of free parameters, which will serve later on as our $X_E$). The problem then, is to estimate the likelihood:
$$
Pr(v_i|(\Phi(v_1),\Phi(v_2),\cdots,\Phi(v_{i-1})))
\qquad (2)
$$
However, as the walk length grows, computing this conditional probability becomes unfeasible.

A recent relaxation in language modeling [27, 28] turns the prediction problem on its head. First, instead of using the context to predict a missing word, it uses one word to predict the context. Secondly, the context is composed of the words appearing to both the right and left of the given word. Finally, it removes the ordering constraint on the problem, instead, requiring the model to maximize the probability of any word appearing in the context without the knowledge of its offset from the given word. In terms of vertex representation modeling, this yields the optimization problem:
$$
\mathop{minimize}_{\Phi}− log \Pr(\{v_{i−w}, \cdots , v_{i+w}\} \backslash v_i | \Phi(v_i))
\qquad (3)
$$
We find these relaxations are particularly desirable for social representation learning. First, the order independence assumption better captures a sense of ‘nearness’ that is provided by random walks. Moreover, this relaxation is quite useful for speeding up the training time by building small models as one vertex is given at a time.

Solving the optimization problem from Eq. 3 builds representations that capture the shared similarities in local graph structure between vertices. Vertices which have similar neighborhoods will acquire similar representations (encoding co-citation similarity), allowing generalization on machine learning tasks.

By combining both truncated random walks and language models we formulate a method which satisfies all of our desired properties. This method generates representations of social networks that are low-dimensional, and exist in a continuous vector space. Its representations encode latent forms of community membership, and because the method outputs useful intermediate representations, it can adapt to changing network topology.

> 语言建模的目标是估计特定单词序列出现在语料库中的可能性。更正式地说，给定一个单词序列 $W_1^n=（w_0，w_1，\cdots，w_n) $, 其中 $w_i\in\mathcal{V}$  ($\mathcal{V}$ 是词汇），我们希望在所有训练语料中最大化 $Pr(w_n|w_0，w_1，\cdots，w_{n-1}) $ 。最近在表示学习方面的工作专注在使用概率神经网络来构建单词的一般表示，这将语言建模的范围扩展到其原始目标之外。
>
> 在这项工作中，我们提出了一种语言建模的推广，通过一系列短随机游走来探索图。这些游走可以被认为是特殊语言中的短句和短语；直接模拟是根据随机游走中迄今为止访问的所有以前的顶点来估计观察顶点 $v_i$ 的可能性，即
> $$
> Pr(v_i | (v_1, v_2, \cdots , v_{i−1}))
> \qquad (1)
> $$
> 我们的目标是学习一个潜在表示，而不仅仅是节点共现的概率分布，因此我们引入了一个映射函数 $\Phi$ 。这个映射 $\Phi$ 代表与图中每个顶点 $v$ 相关的潜在社会表示。（在实践中，我们用一个自由参数的 $|V|×d$ 矩阵来表示 $\Phi$ ，这个矩阵将在以后用作我们的 $X_E$）。那么，问题是估计似然：
> $$
> Pr(v_i|(\Phi(v_1),\Phi(v_2),\cdots,\Phi(v_{i-1})))
> \qquad (2)
> $$
> 然而，随着游走长度的增长，计算这种条件概率变得不可行。
>
> 最近在语言建模 [27,28] 中的一个松弛条件使预测问题迎刃而解。首先，它不使用上下文来预测一个缺失的单词，而是使用一个单词来预测上下文。其次，上下文由给定单词的左右两侧出现的单词组成。最后，它移除了问题的顺序约束，而是要求模型在不知道其与给定单词偏移的情况下，最大化上下文中出现任何单词的概率。就顶点表示建模而言，这产生了优化问题：
> $$
> \mathop{minimize}_{\Phi}− log \Pr(\{v_{i−w}, \cdots , v_{i+w}\} \backslash v_i | \Phi(v_i))
> \qquad (3)
> $$
> 我们发现这些松弛条件特别适合于 social 表示学习。首先，顺序独立假设更好地捕捉了随机游走提供的“接近”感。此外，这种松弛对于通过构建小模型来加快训练时间非常有用，因为每次给出一个顶点。
>
> 从 Eq 3 中求解优化问题建立了捕获顶点间局部图结构中共享相似性的表示。具有相似邻域的顶点将获得相似的表示(编码 co-citation 相似性)，允许在机器学习任务中泛化。
>
> 通过结合 truncate random walk 和 language model，我们生成了一种满足我们所有需求的方法。这种方法生成了存在于连续向量空间中的低维的表示。它的表示编码了社区成员的潜在形式，因为这种方法输出了使用的的中间表示，所以它可以适应不断变化的 network 拓扑结构。

## 4. METHOD

In this section we discuss the main components of our algorithm. We also present several variants of our approach and discuss their merits.

> 在本节中，我们讨论算法的主要组成部分。我们还介绍了我们方法的几个变体，并讨论了它们的优点。

## 4.1 Overview

As in any language modeling algorithm, the only required input is a corpus and a vocabulary $\mathcal{V}$. DeepWalk considers a set of short truncated random walks its own corpus, and the graph vertices as its own vocabulary ($\mathcal{V} = V$ ). While it is beneficial to know $V$ and the frequency distribution of vertices in the random walks ahead of the training, it is not necessary for the algorithm to work as we will show in 4.2.2.

> 与任何语言建模算法一样，唯一需要的输入是语料库和词汇表 $\mathcal{V}$ 。DeepWalk认为一组被截断的短随机游走是它自己的语料库，图的顶点是它自己的词汇表( $\mathcal{V}=V$ )。虽然在训练之前知道 $V$ 和随机游走中顶点的频率分布是有益的，但算法没有必要像我们在4.2.2中所展示的那样工作。

## 4.2 Algorithm: DeepWalk

The algorithm consists of two main components; first a random walk generator, and second, an update procedure. The random walk generator takes a graph $G$ and samples uniformly a random vertex $v_i$ as the root of the random walk $\mathcal{W}_{v_i}$. A walk samples uniformly from the neighbors of the last vertex visited until the maximum length $(t)$ is reached. While we set the length of our random walks in the experiments to be fixed, there is no restriction for the random walks to be of the same length. These walks could have restarts (i.e. a teleport probability of returning back to their root), but our preliminary results did not show any advantage of using restarts. In practice, our implementation specifies a number of random walks $\gamma$ of length $t$ to start at each vertex.

Lines 3-9 in Algorithm 1 shows the core of our approach. The outer loop specifies the number of times, $\gamma$, which we should start random walks at each vertex. We think of each iteration as making a ‘pass’ over the data and sample one walk per node during this pass. At the start of each pass we generate a random ordering to traverse the vertices. This is not strictly required, but is well-known to speed up the convergence of stochastic gradient descent.

In the inner loop, we iterate over all the vertices of the graph. For each vertex $v_i$ we generate a random walk $|W_{v_i} | = t$, and then use it to update our representations (Line 7). We use the SkipGram algorithm [27] to update these representations in accordance with our objective function in Eq. 3.

![Algorithm 1](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Alg1.png)

> 该算法由两个主要部分组成;
>
> 第一个是随机游走生成器，第二个是更新过程。
>
> 随机游走生成器对图 $G$ 均匀抽样随机顶点 $v_i$ 作为随机游走 $ \mathcal{W}_{v_i}$ 的根。从访问的最后一个顶点的邻居均匀采样，直到达到最大长度 $(t)$。虽然我们在实验中设定了随机游走的长度是固定的，但是没有限制随机游走的长度是相同的。这些游走可以重新启动(即传送回其 root 的概率)，但我们的初步结果并未显示使用重新启动有任何优势。在实践中，我们的实现指定了从每个顶点开始 $\gamma$ 次，长度为 $t$ 的随机游走。
>
> 算法1 中的第 3-9 行显示了我们方法的核心。外循环指定了我们应该在每个顶点进行随机游走的次数 $\gamma$。我们认为每次迭代都是对数据进行“遍历”，并在此遍历期间对每个节点进行一次遍历。在每次遍历开始时，我们都会生成一个遍历顶点的随机顺序。这并不是严格要求，但众所周知，这可以加快随机梯度下降的收敛速度。
>
> 在内循环中，我们迭代图的所有顶点。对于每个顶点 $v_i$，我们生成一个随机游走 $|W_{v_i} | = t$，然后用它来更新我们的表示（第7行）。我们使用SkipGram 算法[27]根据我们在 Eq.3 中的目标函数来更新这些表示。

#### 4.2.1 SkipGram

SkipGram is a language model that maximizes the cooccurrence probability among the words that appear within a window, $w$, in a sentence. It approximates the conditional probability in Equation 3 using an independence assumption as the following
$$
Pr(\{v_{i−w}, · · · , v_{i+w}\} \backslash v_i | \Phi(v_i)) =
\prod_{j=i-w \\ j \ne i}^{i+w} Pr(v_j | \Phi(v_i))
\qquad (4)
$$
Algorithm 2 iterates over all possible collocations in random walk that appear within the window $w$ (lines 1-2). For each, we map each vertex $v_j$ to its current representation vector $\Phi(v_j) \in R^d$ (See Figure 3b). Given the representation of $v_j$ , we would like to maximize the probability of its neighbors in the walk (line 3). We can learn such a posterior distribution using several choices of classifiers. For example, modeling the previous problem using logistic regression would result in a huge number of labels (that is equal to $|V|$) which could be in millions or billions. Such models require vast computational resources which could span a whole cluster of computers [4]. To avoid this necessity and speed up the training time, we instead use the Hierarchical Softmax [30, 31] to approximate the probability distribution.

![Algorithm 2](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Alg2.png)

> SkipGram 是一个语言模型，它最大限度地提高了句子中出现在窗口 $w$ 中的单词之间的共现概率。它使用如下独立假设来近似 等式3 中的条件概率
> $$
> Pr(\{v_{i−w}, · · · , v_{i+w}\} \backslash v_i | \Phi(v_i)) =
> \prod_{j=i-w \\ j \ne i}^{i+w} Pr(v_j | \Phi(v_i))
> \qquad (4)
> $$
> 算法 2 迭代随机游走中在窗口 $w$ (第1-2行) 内的所有可能的搭配。对于每一个，我们将每个顶点 $v_j$ 映射到其当前表示向量 $\Phi(v_j)\in R^d$ （见图3b）。给定 $v_j$ 的表示，我们希望最大化其邻居在游走中的概率（第 3 行）。我们可以使用几种分类器选择来学习这样的后验分布。例如，使用逻辑回归对前面的问题进行建模会产生大量的标签(即等于 $|V|$ )，可能以百万或数十亿计。这样的模型需要大量的计算资源，这可能会跨越整个计算机集群[4]。为了避免这种必要性并加快训练时间，我们使用 Hierarchical Softmax[30,31]来近似概率分布。

#### 4.2.2 Hierarchical Softmax

Given that $u_k \in V$ , calculating $Pr(u_k | \Phi(v_j ))$ in line 3 is not feasible. Computing the partition function (normalization factor) is expensive, so instead we will factorize the conditional probability using Hierarchical softmax. We assign the vertices to the leaves of a binary tree, turning the prediction problem into maximizing the probability of a specific path in the hierarchy (See Figure 3c). If the path to vertex $u_k$ is identified by a sequence of tree nodes $(b_0, b_1, \cdots , b_{\lceil log \ |V| \rceil}), (b_0 = root, b_{\lceil log \ |V|\rceil} = u_k)$ then
$$
Pr(u_k| \Phi(v_j)) = \prod_{l=1}^{\lceil log \ |V|\rceil}
Pr(b_l| \Phi(v_j))
\qquad (5)
$$
Now, $Pr(b_l | \Phi(v_j))$ could be modeled by a binary classifier that is assigned to the parent of the node $b_l$ as Equation 6 shows,
$$
Pr(b_l | \Psi(v_j)) = 1/(1 + e^{− \Phi(v_j) \cdot \Psi(b_l)})
\qquad (6)
$$
where $\Psi(b_l) \in R^d$ is the representation assigned to tree node $b_l$’s parent. This reduces the computational complexity of calculating $Pr(u_k | \Phi(v_j ))$ from $O(|V|)$ to $O(log \ |V |)$.

We can speed up the training process further, by assigning shorter paths to the frequent vertices in the random walks. Huffman coding is used to reduce the access time of frequent elements in the tree.

![](/Users/helloword/Anmingyu/Gor-rok/Papers/Embedding/DeepWalkOnlineLearningofSocialRepresentations/Fig3.png)

**Figure 3: Overview of DeepWalk. We slide a window of length $2w + 1$ over the random walk $\mathcal{W}_{v_4}$ , mapping the central vertex $v_1$ to its representation $\Phi(v_1)$. Hierarchical Softmax factors out $Pr(v_3 | \Phi(v_1))$ and $Pr(v_5 | \Phi(v_1))$ over sequences of probability distributions corresponding to the paths starting at the root and ending at $v_3$ and $v_5$. The representation $\Phi$ is updated to maximize the probability of $v_1$ co-occurring with its context $\{v_3, v_5\}$.**

> 给定 $u_k\in V$，在第 3 行计算 $Pr(u_k|\Phi（v_j))$ 是不可行的。计算partition function (归一化因子) 是昂贵的，所以我们将使用 Hierarchical softmax分解条件概率。我们将顶点分配给二叉树的叶子，将预测问题转化为 Hierarchical 中特定路径的概率最大化（参见图3 c）。如果到顶点 $u_k$ 的路径由一系列树节点$ (b_0，b_1，\cdots，b_{\lceil log\|V|\rceil}), (b_0=root，b_{\lceil log\|V|\rceil}=u_k)$标识，那么
> $$
> Pr(u_k| \Phi(v_j)) = \prod_{l=1}^{\lceil log \ |V|\rceil}
> Pr(b_l| \Phi(v_j))
> \qquad (5)
> $$
> 现在，$Pr(b_l | \Phi(v_j))$ 可以由赋值给节点父节点 $b_l$ 的二元分类器建模，如式6所示，
> $$
> Pr(b_l | \Psi(v_j)) = 1/(1 + e^{− \Phi(v_j) \cdot \Psi(b_l)})
> \qquad (6)
> $$
> 其中，$\Psi(b_l) \in R^d$ 是分配给树节点 $b_l$ 的父节点的表示。这将计算 $Pr(u_k|\Phi(V_J))$ 的计算复杂度从 $O(|V|)$ 降低到 $O(log|V|)$。
>
> 通过给随机游走中的频繁顶点分配更短的路径来进一步加快训练过程。霍夫曼编码用于减少树中频繁元素的访问时间。

#### 4.2.3 Optimization

The model parameter set is $\theta = \{\Phi, \Psi\}$ where the size of each is $O(d|V|)$. Stochastic gradient descent (SGD) [5] is used to optimize these parameters (Line 4, Algorithm 2). The derivatives are estimated using the back-propagation algorithm. The learning rate $\alpha$ for SGD is initially set to 2.5% at the beginning of the training and then decreased linearly with the number of vertices that are seen so far.

> 模型参数为 $\theta=\{\Phi，\Psi\}$ ，其中每个参数的大小为 $O(d|V|)$。随机梯度下降(SGD)[5]用于优化这些参数(第4行，算法2)。使用反向传播算法计算导数。
> SGD的学习率 $\alpha$ 在训练开始时最初设置为 2.5%，然后随着到目前为止更新的顶点数线性下降。

## 4.3 Parallelizability

As shown in Figure 2 the frequency distribution of vertices in random walks of social network and words in a language both follow a power law. This results in a long tail of infrequent vertices, therefore, the updates that affect $\Phi$ will be sparse in nature. This allows us to use asynchronous version of stochastic gradient descent (ASGD), in the multi-worker case. Given that our updates are sparse and we do not acquire a lock to access the model shared parameters, ASGD will achieve an optimal rate of convergence [36]. While we run experiments on one machine using multiple threads, it has been demonstrated that this technique is highly scalable, and can be used in very large scale machine learning [9]. Figure 4 presents the effects of parallelizing DeepWalk. It shows the speed up in processing BlogCatalog and Flickr networks is consistent as we increase the number of workers to 8 (Figure 4a). It also shows that there is no loss of predictive performance relative to the running DeepWalk serially (Figure 4b).

> 如 图2 所示，社交网络随机游走中顶点的频率分布和语言中的单词都遵循幂律。这导致不频繁顶点的长尾，因此，影响 $\Phi$ 的更新本质上是稀疏的。这允许我们在多 worker 的情况下使用异步版本的随机梯度下降（ASGD）。鉴于我们的更新是稀疏的，并且我们没有获得访问模型共享参数的锁，ASGD将实现最佳收敛速度[36]。当我们在一台机器上使用多个线程运行实验时，已经证明这种技术具有高度的可扩展性，可以用于非常大规模的机器学习[9]。图4展示了并行化 DeepWalk 的效果。它显示了当我们将 worker 数量增加到 8 个时，处理 BlogCatalog 和 Flickr 网络的速度是一致的（图4 a）。它还表明，相对于连续运行的 DeepWalk ，预测性能没有损失（图4b）。

## 4.4 Algorithm Variants

Here we discuss some variants of our proposed method, which we believe may be of interest.

> 在这里，我们讨论了我们提出的方法的一些变体，我们认为这些变体可能会很有趣。

#### 4.4.1 Streaming

One interesting variant of this method is a streaming approach, which could be implemented without knowledge of the entire graph. In this variant small walks from the graph are passed directly to the representation learning code, and the model is updated directly. Some modifications to the learning process will also be necessary. First, using a decaying learning rate may no longer be desirable as it assumes the knowledge of the total corpus size. Instead, we can initialize the learning rate $\alpha$ to a small constant value. This will take longer to learn, but may be worth it in some applications. Second, we cannot necessarily build a tree of parameters any more. If the cardinality of $V$ is known (or can be bounded), we can build the Hierarchical Softmax tree for that maximum value. Vertices can be assigned to one of the remaining leaves when they are first seen. If we have the ability to estimate the vertex frequency a priori, we can also still use Huffman coding to decrease frequent element access times.

> 此方法的一个有趣变体是 streaming 方法，它可以在不了解整个图的情况下实现。在这个变型中，来自图的小游走被直接传递给表示学习代码，并且模型被直接更新。对学习过程进行一些修改也是必要的。首先，使用递减的学习率可能不再是可取的，因为它假设知道总的语料库大小。相反，我们可以将学习率 $\alpha$ 初始化为一个较小的常数。这将需要更长的时间来学习，但在某些应用程序中可能是值得的。其次，我们不一定再构建参数树。如果 $V$ 的基数是已知的(或者可以是有界的)，我们可以为该最大值构建 Hierarchical Softmax 树。第一次看到顶点时，可以将它们指定给剩余的树叶之一。如果我们有能力先验估计顶点频率，我们仍然可以使用霍夫曼编码来减少频繁的元素访问次数。

#### 4.4.2 Non-random walks

Some graphs are created as a by-product of agents interacting with a sequence of elements (e.g. users’ navigation of pages on a website). When a graph is created by such a stream of non-random walks, we can use this process to feed the modeling phase directly. Graphs sampled in this way will not only capture information related to network structure, but also to the frequency at which paths are traversed.

In our view, this variant also encompasses language modeling. Sentences can be viewed as purposed walks through an appropriately designed language network, and language models like SkipGram are designed to capture this behavior.

This approach can be combined with the streaming variant (Section 4.4.1) to train features on a continually evolving network without ever explicitly constructing the entire graph. Maintaining representations with this technique could enable web-scale classification without the hassles of dealing with a web-scale graph.

> 有些图是 agents 与一系列元素交互的副产品(例如，用户在网站上的页面导航)。当一个图由这样一个非随机游走流创建时，我们可以使用这个过程来直接提供建模阶段。以这种方式采样的图不仅可以捕获与网络结构有关的信息，还可以捕获哪个路径遍历的频率。
>
> 在我们看来，这种变体还包括语言建模。句子可以被看作是经过一个适当设计的 language network 的有目的游走，而像 SkipGram 等语言模型就是为了捕捉这种行为而设计的。
>
> 这种方法可以与 stream 变体 (第4.4.1节) 相结合，在不断进化的网络上训练特征，而无需明确构造整个图。使用这种技术维护表示可以实现 web-scale 的分类，而无需处理 web-scale 图的麻烦。
>

