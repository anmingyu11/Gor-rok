# metapath2vec: Scalable Representation Learning for Heterogeneous Networks

## ABSTRACT

We study the problem of representation learning in heterogeneous networks. Its unique challenges come from the existence of multiple types of nodes and links, which limit the feasibility of the conventional network embedding techniques. We develop two scalable representation learning models, namely metapath2vec and metapath2vec++. The metapath2vec model formalizes meta-pathbased random walks to construct the heterogeneous neighborhood of a node and then leverages a heterogeneous skip-gram model to perform node embeddings. The metapath2vec++ model further enables the simultaneous modeling of structural and semantic correlations in heterogeneous networks. Extensive experiments show that metapath2vec and metapath2vec++ are able to not only outperform state-of-the-art embedding models in various heterogeneous network mining tasks, such as node classication, clustering, and similarity search, but also discern the structural and semantic correlations between diverse network objects.

> 我们研究了异构网络中的表示学习问题。其独特的挑战来自于存在多种类型的节点和链接，这限制了传统网络嵌入技术的可行性。我们开发了两个可扩展的表示学习模型，即 metapath2vec 和 metapath2vec++。
>
> metapath2vec 模型通过基于元路径的随机游走，构建节点的异构邻域，然后利用异构的 skip-gram 模型进行节点嵌入。metapath2vec++ 模型则进一步支持在异构网络中同时建模结构和语义相关性。
>
> 广泛的实验表明，metapath2vec 和 metapath2vec++ 不仅能在各种异构网络挖掘任务（如节点分类、聚类和相似度搜索）中优于最先进的嵌入模型，还能识别不同网络对象之间的结构和语义相关性。

## 1 INTRODUCTION

Neural network-based learning models can represent latent embeddings that capture the internal relations of rich, complex data across various modalities, such as image, audio, and language [15]. Social and information networks are similarly rich and complex data that encode the dynamics and types of human interactions, and are similarly amenable to representation learning using neural networks. In particular, by mapping the way that people choose friends and maintain connections as a “social language,” recent advances in natural language processing (NLP) [3] can be naturally applied to network representation learning, most notably the group of NLP models known as word2vec [17, 18]. A number of recent research publications have proposed word2vec-based network representation learning frameworks, such as DeepWalk [22], LINE [30], and node2vec [8]. Instead of handcraed network feature design, these representation learning methods enable the automatic discovery of useful and meaningful (latent) features from the “raw networks.”

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Figure1.png)

> 基于神经网络的学习模型确实能够表示潜在嵌入，这些嵌入能够捕获图像、音频和语言等不同形式数据的内在关系。同样地，社会和信息网络也蕴含了丰富而复杂的数据，它们反映了人类互动的动态和类型，并且也适用于通过神经网络进行表示学习。
>
> 特别是，通过将人们选择朋友和维持联系的方式视为一种“社交语言”，自然语言处理的最新进展可以自然地应用于网络表示学习。在自然语言处理领域，最著名的模型之一就是word2vec，它能够从大量文本数据中学习词的嵌入表示。类似地，我们可以将这种技术应用于网络数据，从而学习网络中节点或边的嵌入表示。
>
> 近年来，确实有一些研究提出了基于word2vec的网络表示学习框架，如DeepWalk、LINE和node2vec等。这些方法的核心思想是通过在网络上进行随机游走，生成节点序列，然后利用word2vec等技术从这些序列中学习节点的嵌入表示。
>
> 与手工设计网络特征的方法相比，这些表示学习方法能够从“原始网络”中自动发现有用和有意义的潜在特征。这些特征可以进一步用于各种网络分析任务，如节点分类、链接预测、社区发现等。
>
> 总的来说，基于神经网络的学习模型在网络表示学习方面展现出了强大的能力，它们能够自动地从原始网络中提取有用的特征，为后续的网络分析和应用提供了便利。

However, these work has thus far focused on representation learning for homogeneous networks—representative of singular type of nodes and relationships. Yet a large number of social and information networks are heterogeneous in nature, involving diversity of node types and/or relationships between nodes [25]. ese heterogeneous networks present unique challenges that cannot be handled by representation learning models that are specically designed for homogeneous networks. Take, for example, a heterogeneous academic network: How do we eectively preserve the concept of “word-context” among multiple types of nodes, e.g., authors, papers, venues, organizations, etc.? Can random walks, such those used in DeepWalk and node2vec, be applied to networks of multiple types of nodes? Can we directly apply homogeneous network-oriented embedding architectures (e.g., skip-gram) to heterogeneous networks?

> 然而，目前这些工作主要集中在同质网络的表示学习上，即代表单一类型的节点和关系。但实际上，大量的社交和信息网络本质上是异质的，涉及多种节点类型和/或节点之间的关系。这些异质网络带来了独特的挑战，这是那些专门为同质网络设计的表示学习模型所无法处理的。
>
> 以异质学术网络为例，我们如何在多种类型的节点（如作者、论文、会议场地、组织等）之间有效地保留“词-上下文”的概念？DeepWalk和node2vec中使用的随机游走能否应用于具有多种类型节点的网络？我们是否可以直接将面向同质网络的嵌入架构（如skip-gram）应用于异质网络？

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Table1.png)

By solving these challenges, the latent heterogeneous network embeddings can be further applied to various network mining tasks, such as node classication [13], clustering [27, 28], and similarity search [26, 35]. In contrast to conventional meta-path-based methods [25], the advantage of latent-space representation learning lies in its ability to model similarities between nodes without connected meta-paths. For example, if authors have never published papers in the same venue—imagine one publishes 10 papers all in NIPS and the other has 10 publications all in ICML; their “APCPA”-based PathSim similarity [26] would be zero—this will be naturally overcome by network representation learning.

> 通过解决上述挑战，我们可以得到潜在的异质网络嵌入，这些嵌入可以进一步应用于各种网络挖掘任务，如节点分类、聚类和相似性搜索。与传统的基于元路径的方法相比，潜在空间表示学习的优势在于它能够在没有连接元路径的情况下对节点之间的相似性进行建模。
>
> 例如，如果两位作者从未在同一会议上发表过论文——想象一下，一位作者在NIPS上发表了10篇论文，而另一位作者在ICML上发表了10篇论文；那么他们基于“APCPA”的PathSim相似性将为零——这可以通过网络表示学习自然地克服。网络表示学习能够捕获节点之间的潜在关系，即使这些节点在显式的网络结构中并没有直接相连。

**Contributions.** We formalize the heterogeneous network representation learning problem, where the objective is to simultaneously learn the low-dimensional and latent embeddings for multiple types of nodes. We present the metapath2vec and its extension metapath2vec++ frameworks. The goal of metapath2vec is to maximize the likelihood of preserving both the structures and semantics of a given heterogeneous network. In metapath2vec, we first propose meta-path [25] based random walks in heterogeneous networks to generate heterogeneous neighborhoods with network semantics for various types of nodes. Second, we extend the skip-gram model [18] to facilitate the modeling of geographically and semantically close nodes. Finally, we develop a heterogeneous negative sampling-based method, referred to as metapath2vec++, that enables the accurate and effcient prediction of a node’s heterogeneous neighborhood.

> **贡献**。我们规范了异质网络表示学习问题，其目标是同时学习多种类型节点的低维和潜在嵌入。我们提出了metapath2vec及其扩展metapath2vec++框架。metapath2vec的目标是最大限度地保留给定异质网络的结构和语义。
>
> 在metapath2vec中，我们首先提出了基于元路径的随机游走，在异质网络中为各种类型的节点生成具有网络语义的异质邻域。其次，我们扩展了skip-gram模型，以促进对地理和语义上相近节点的建模。最后，我们开发了一种基于异质负采样的方法，称为metapath2vec++，它能够准确有效地预测节点的异质邻域。

The proposed metapath2vec and metapath2vec++ models are different from conventional network embedding models, which focus on homogeneous networks [8, 22, 30]. Specifically, conventional models suffer from the identical treatment of different types of nodes and relations, leading to the production of indistinguishable representations for heterogeneous nodes—as evident through our evaluation. Further, the metapath2vec and metapath2vec++ models also dffier from the Predictive Text Embedding (PTE) model [29] in several ways. First, PTE is a semi-supervised learning model that incorporates label information for text data. Second, the heterogeneity in PTE comes from the text network wherein a link connects two words, a word and its document, and a word and its label. Essentially, the raw input of PTE is words and its output is the embedding of each word, rather than multiple types of objects.

> 提出的metapath2vec和metapath2vec++模型与传统的网络嵌入模型不同，后者主要关注同质网络。具体来说，传统模型对不同类型的节点和关系进行相同的处理，导致产生无法区分的异质节点表示，这从我们的评估中可以明显看出。
>
> 此外，metapath2vec 和 metapath2vec++ 模型与预测性文本嵌入（PTE）模型在几个方面也有所不同。首先，PTE是一种半监督学习模型，它结合了文本数据的标签信息。其次，PTE的异质性来自于文本网络，其中链接连接两个单词、一个单词及其文档以及一个单词及其标签。本质上，PTE的原始输入是单词，其输出是每个单词的嵌入，而不是多种类型的对象。
>
> 相比之下，我们的metapath2vec和metapath2vec++模型专为异质网络设计，能够处理多种类型的节点和关系，并生成具有区分度的节点嵌入。这些嵌入捕获了节点的结构和语义信息，为后续的网络挖掘任务提供了丰富的特征表示。此外，我们的模型不需要标签信息，因此可以在无监督的环境下进行学习。
>
> 总的来说，我们的模型在处理异质网络方面具有独特的优势，能够为复杂网络的分析和挖掘提供新的视角和方法。这些贡献使得我们的模型在网络表示学习领域具有重要的理论和实践价值。

We summarize the differences of these methods in Table 1, which lists their input to learning algorithms, as well as the top-five similarity search results in the DBIS network for the same two queries used in [26] (see Section 4 for details). By modeling the hetero- geneous neighborhood and further leveraging the heterogeneous negative sampling technique, metapath2vec++ is able to achieve the best top-five similar results for both types of queries. Figure 1 shows the visualization of the 2D projections of the learned embeddings for 16 CS conferences and corresponding high-profile researchers in each field. Remarkably, we find that metapath2vec++ is capable of automatically organizing these two types of nodes and implicitly learning the internal relationships between them, suggested by the similar directions and distances of the arrows connecting each pair. For example, it learns J. Dean → OSDI and C. D. Manning → ACL. metapath2vec is also able to group each author-conference pair closely, such as R. E. Tarjan and FOCS. All of these properties are not discoverable from conventional network embedding models.

To summarize, our work makes the following contributions:

1. Formalizes the problem of heterogeneous network representation learning and identifies its unique challenges resulting from network heterogeneity.
2. Develops effective and efficient network embedding frame- works, metapath2vec & metapath2vec++, for preserving both structural and semantic correlations of heterogeneous networks.
3. Throughextensiveexperiments,demonstratestheefficacyand scalability of the presented methods in various heterogeneous network mining tasks, such as node classification (achieving relative improvements of 35–319% over benchmarks) and node clustering (achieving relative gains of 13–16% over baselines).
4. Demonstrates the automatic discovery of internal semantic relationships between different types of nodes in heterogeneous networks by metapath2vec & metapath2vec++, not discoverable by existing work.

> 我们总结了这些方法在表1中的区别，该表列出了它们的学习算法输入，以及在DBIS网络中对于与[26]中使用的相同的两个查询的前五个相似度搜索结果（详见第4节）。通过对异质邻域的建模，并进一步利用异质负采样技术，metapath2vec++能够为两种类型的查询获得最好的前五个相似结果。
>
> 图1显示了16个计算机科学会议和每个领域相应的高知名度研究人员的二维投影的可视化。值得注意的是，我们发现metapath2vec++能够自动组织这两种类型的节点，并通过连接每对的箭头的相似方向和距离来隐式地学习它们之间的内部关系。例如，它学习了J. Dean与OSDI以及C.D. Manning与ACL的关系。metapath2vec还能够将每对作者-会议紧密地组合在一起，例如R.E. Tarjan和FOCS。所有这些特性都是传统网络嵌入模型无法发现的。
>
> 综上所述，我们的工作做出了以下贡献：
>
> 1. 规范了异质网络表示学习的问题，并确定了由网络异质性带来的独特挑战。
> 2. 开发了有效且高效的网络嵌入框架metapath2vec和metapath2vec++，以保留异质网络的结构和语义相关性。
> 3. 通过广泛的实验，证明了所提出的方法在各种异质网络挖掘任务中的有效性和可扩展性，如节点分类（相对于基准测试提高了35-319%）和节点聚类（相对于基线提高了13-16%）。
> 4. 演示了metapath2vec和metapath2vec++在异质网络中自动发现不同类型节点之间的内部语义关系，这是现有工作无法发现的。



## 2 PROBLEM DEFINITION

We formalize the representation learning problem in heterogeneous networks, which was first briefly introduced in [21]. In specific, we leverage the definition of heterogeneous networks in [25, 27] and present the learning problem with its inputs and outputs.

> 我们规范了异质网络中的表示学习问题，该问题首先在[21]中进行了简要介绍。具体来说，我们利用[25, 27]中异质网络的定义，并展示了具有输入和输出的学习问题。

*Definition2.1*. **A Heterogeneous Network** is define defined as a graph $G = (V , E,T )$ in which each node $v$ and each link $e$ are associated with their mapping functions $\phi(v): V \rightarrow T_V$ and $\varphi(e): E \rightarrow T_E$ , respectively. $T_V$ and $T_E$ denote the sets of object and relation types, where $|T_V| + |T_E| > 2$.

For example, one can represent the academic network in Figure 2(a) with authors (A), papers (P), venues (V), organizations (O) as nodes, wherein edges indicate the coauthor (A–A), publish (A–P,P–V), affiliation (O–A) relationships. By considering a heteroge- neous network as input, we formalize the problem of heterogeneous network representation learning as follows.

> **定义2.1**：**异质网络**被定义为一个图$G = (V, E, T)$，其中每个节点$v$和每条边$e$分别与其映射函数$\phi(v): V \rightarrow T_V$和$\varphi(e): E \rightarrow T_E$相关联。$T_V$和$T_E$分别表示对象和关系类型的集合，其中$|T_V| + |T_E| > 2$。
>
> 例如，在图2(a)的学术网络中，可以用作者(A)、论文(P)、场所(V)、组织(O)作为节点来表示，其中边表示合著者(A–A)、出版(A–P, P–V)、隶属(O–A)关系。通过考虑异质网络作为输入，我们将异质网络表示学习的问题形式化如下。
>

The output of the problem is the low-dimensional matrix $X$, with the $v$ th row—a d-dimensional vector $X_v$—corresponding to the representation of node $v$. Notice that, although there are different types of nodes in $V$ , their representations are mapped into the same latent space. The learned node representations can benefit various heterogeneous network mining tasks. For example, the embedding vector of each node can be used as the feature input of node classification, clustering, and similarity search tasks.

The main challenge of this problem comes from the network heterogeneity, wherein it is difficult to directly apply homogeneous language and network embedding methods. The premise of network embedding models is to preserve the proximity between a node and its neighborhood (context) [8, 22, 30]. In a heterogeneous environment, how do we define and model this ‘node–neighborhood’ concept? Furthermore, how do we optimize the embedding models that effectively maintain the structures and semantics of multiple types of nodes and relations?

> 该问题的输出是低维矩阵$X$，其中第 $v$ 行——一个 $d$ 维向量 $X_v$ ——对应于节点 $v$ 的表示。请注意，尽管 $V$ 中有不同类型的节点，但它们的表示都被映射到相同的潜在空间中。学习到的节点表示可以有益于各种异质网络挖掘任务。例如，每个节点的嵌入向量可以用作节点分类、聚类和相似性搜索任务的特征输入。
>
> 这个问题的主要挑战来自网络的异质性，在这种情况下，很难直接应用同质的语言和网络嵌入方法。网络嵌入模型的前提是保持节点与其邻域（上下文）之间的邻近性[8, 22, 30]。在异质环境中，我们如何定义和建模这个“节点-邻域”的概念？此外，我们如何优化嵌入模型，以有效地维护多种类型和关系的结构和语义？
>

## 3 THE METAPATH2VEC FRAMEWORK

We present a general framework, metapath2vec, which is capable of learning desirable node representations in heterogeneous net- works. The objective of metapath2vec is to maximize the network probability in consideration of multiple types of nodes and edges.

> 我们提出了一个通用框架，即metapath2vec，它能够学习异质网络中理想的节点表示。metapath2vec的目标是在考虑多种类型的节点和边的情况下，最大化网络概率。

## 3.1 Homogeneous Network Embedding

We, first, briefly introduce the word2vec model and its application to homogeneous network embedding tasks. Given a text corpus, Mikolov et al. proposed word2vec to learn the distributed representations of words in a corpus [17, 18]. Inspired by it, DeepWalk [22] and node2vec [8] aim to map the word-context concept in a text corpus into a network. Both methods leverage random walks to achieve this and utilize the skip-gram model to learn the representation of a node that facilitates the prediction of its structural context—local neighborhoods—in a homogeneous network. Usually, given a network $G = (V , E)$, the objective is to maximize the network probability in terms of local structures [8, 18, 22], that is:
$$
\arg \max _\theta \prod_{v \in V} \prod_{c \in N(v)} p(c \mid v ; \theta)
$$
where $N(v)$ istheneighborhoodofnodevinthenetworkG,which can be defined in different ways such as $v$’s one-hop neighbors, and $p (c |v ; θ )$ defines the conditional probability of having a context node $c$ given a node $v$.

> 首先，我们简要介绍一下word2vec模型及其在同质网络嵌入任务中的应用。给定一个文本语料库，Mikolov等人提出了word2vec来学习语料库中单词的分布式表示[17, 18]。受其启发，DeepWalk[22]和node2vec[8]旨在将文本语料库中的词-上下文概念映射到网络中。这两种方法都利用随机游走来实现这一点，并利用skip-gram模型来学习节点的表示，该表示有助于预测同质网络中节点的结构上下文（即局部邻域）。通常，给定一个网络 $G = (V , E)$，目标是根据局部结构最大化网络概率[8, 18, 22]，即：
>
> $$
> \arg \max _\theta \prod_{v \in V} \prod_{c \in N(v)} p(c \mid v ; \theta)
> $$
>
> 其中，$N(v)$是网络$G$中节点$v$的邻域，这可以以不同的方式定义，例如$v$的一跳邻居，而$p(c \mid v ; \theta)$定义了给定节点$v$时具有上下文节点$c$的条件概率。
>
> 这个公式试图找到一组参数 $\theta$，使得对于网络中的每个节点 $v$，以及其邻域 $N(v)$ 中的每个上下文节点 $c$，条件概率 $p(c \mid v ; \theta)$ 的乘积最大化。换句话说，它试图学习节点的表示，这些表示能够最好地预测其局部网络结构。
>

#### 3.2 Heterogeneous Network Embedding:metapath2vec

To model the heterogeneous neighborhood of a node, metapath2vec introduces the heterogeneous skip-gram model. To incorporate the heterogeneous network structures into skip-gram, we propose meta-path-based random walks in heterogeneous networks.

**Heterogeneous Skip-Gram**. In metapath2vec, we enable skip- gram to learn effective node representations for a heterogeneous network $G = (V ,E,T )$ with $|T_V | > 1$ by maximizing the probability of having the heterogeneous context $N_t(v)$ , $t \in T_V$ given a node $v$:
$$
\arg \max _\theta \sum_{v \in V} \sum_{t \in T_V} \sum_{c_t \in N_t(v)} \log p\left(c_t \mid v ; \theta\right)
$$
where $N_t (v)$ denotes $v$’s neighborhood with the $t^{th}$ type of nodes and $p(c_t |v;θ)$ is commonly defined as a softmax function [3, 7, 18, 24], that is: $
p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u \in V} e^{X_u \cdot X_v}}
$, where $X_v$ is the $v^{th}$ row of $X$, representing the embedding vector for node $v$. For illustration, consider the academic network in Figure 2(a), the neighborhood of one author node a (e.g., a , a & a ), venues (e.g., ACL & KDD), organizations (CMU

& MIT), as well as papers (e.g., p & p ).

> 为了模拟节点的异质邻域，metapath2vec 引入了异质 skip-gram 模型。为了将异质网络结构纳入skip-gram中，我们提出了基于元路径的异质网络随机游走。
>
> **异质Skip-Gram**。在metapath2vec中，我们通过最大化给定节点 $v$ 时具有异质上下文 $N_t(v)$，$t \in T_V$的概率，使skip-gram能够为异质网络$G = (V, E, T)$学习有效的节点表示，其中$|T_V| > 1$：
>$$
> \arg \max _\theta \sum_{v \in V} \sum_{t \in T_V} \sum_{c_t \in N_t(v)} \log p(c_t \mid v ; \theta)
> $$
> 
>其中，$N_t(v)$表示节点$v$与第$t$类型节点的邻域，而$p(c_t | v; \theta)$通常定义为softmax函数[3, 7, 18, 24]，即：
> 
>$$
> p(c_t \mid v ; \theta) = \frac{e^{X_{c_t} \cdot X_v}}{\sum_{u \in V} e^{X_u \cdot X_v}}
> $$
> 
>其中，$X_v$是矩阵$X$的第$v$行，代表节点$v$的嵌入向量。为了说明，请考虑图2(a)中的学术网络，一个作者节点a（例如，a2, a3 & a5）的邻域包括会场（例如，ACL & KDD）、组织（CMU & MIT）以及论文（例如，p2 & p3）。
> 
>在这个框架下，我们可以通过预定义的元路径来指导随机游走，生成反映特定语义关系的节点序列。然后，利用异质skip-gram模型学习这些节点序列的嵌入表示，以捕获网络中的复杂关系和语义信息。通过这种方式，metapath2vec能够有效地处理异质网络中的表示学习问题，并为网络中的每个节点生成有意义的低维表示。这些表示可以进一步用于各种网络分析任务，如节点分类、链接预测和社区检测等。
> 
>需要注意的是，在计算softmax函数时，分母中的求和操作是针对所有节点进行的，这在实际应用中可能会导致计算量非常大。因此，通常会采用一些技巧来近似计算softmax，例如负采样或层次softmax等，以提高计算效率。

To achieve efficient optimization, Mikolov et al. introduced negative sampling [18], in which a relatively small set of words (nodes) are sampled from the corpus (network) for the construction of soft-max. We leverage the same technique for metapath2vec. Given a negative sample size $M$, Eq. 2 is updated as follows: $\log \sigma\left(X_{c_t} \cdot X_v\right)+ \sum_{m=1}^M \mathbb{E}_{u^m \sim P(u)}\left[\log \sigma\left(-X_{u^m} \cdot X_v\right)\right]$, where $\sigma(x)=\frac{1}{1+e^{-x}}$ and $P(u)$ is the pre-defined distribution from which a negative node $u^m$ is drew from for $M$ times. metapath2vec builds the the node frequency distribution by viewing different types of nodes homogeneously and draw (negative) nodes regardless of their types.

> 为了实现高效的优化，Mikolov等人引入了负采样方法[18]，其中从语料库（网络）中抽取一个相对较小的词（节点）集合来构建softmax。我们对metapath2vec采用了相同的技术。给定负样本大小$M$，方程2更新如下：
>
> $$\log \sigma(X_{c_t} \cdot X_v) + \sum_{m=1}^M \mathbb{E}_{u^m \sim P(u)}\left[\log \sigma(-X_{u^m} \cdot X_v)\right]$$
>
> 其中，$\sigma(x) = \frac{1}{1+e^{-x}}$，$P(u)$是预定义的分布，从中抽取负节点$u^m$共$M$次。
>
> 在metapath2vec中，我们通过将不同类型的节点视为同质来构建节点频率分布，并无论节点类型如何都进行（负）节点抽取。
>

**Meta-Path-Based Random Walks**. How to effectively transform the structure of a network into skip-gram? In DeepWalk [22] and node2vec [8], this is achieved by incorporating the node paths traversed by random walkers over a network into the neighborhood function.

Naturally, we can put random walkers in a heterogeneous network to generate paths of multiple types of nodes. At step $i$ , the transition probability $p(v^{i+1}|v^i)$ is denoted as the normalized probability distributed over the neighbors of $v^i$ by ignoring their node types. The generated paths can be then used as the input of node2vec and DeepWalk. However, Sun et al. demonstrated that heterogeneous random walks are biased to highly visible types of nodes—those with a dominant number of paths—and concentrated nodes—those with a governing percentage of paths pointing to a small set of nodes [26].

> **基于元路径的随机游走**。如何有效地将网络结构转化为skip-gram？在DeepWalk[22]和node2vec[8]中，这是通过将网络上随机游走者遍历的节点路径纳入邻域函数来实现的。
>
> 当然，我们可以在异质网络上放置随机游走者来生成多种类型的节点路径。在第$i$步，转移概率$p(v^{i+1}|v^i)$表示为忽略节点类型后，$v^i$的邻居上分布的归一化概率。然后，生成的路径可以作为node2vec和DeepWalk的输入。然而，Sun等人证明，异质随机游走偏向于高度可见的节点类型——即那些具有主导数量路径的节点——以及集中的节点——即那些指向一小部分节点的主要路径百分比的节点[26]。
>

In light of these issues, we design meta-path-based random walks to generate paths that are able to capture both the semantic and structural correlations between different types of nodes, facilitating the transformation of heterogeneous network structures into metapath2vec’s skip-gram.

> 鉴于这些问题，我们设计了基于元路径的随机游走，以生成能够捕获不同类型节点之间的语义和结构相关性的路径，从而促进将异构网络结构转化为metapath2vec的skip-gram。

Formally, a meta-path scheme $\mathcal{P}$ is defined as a path that is denoted in the form of $V_1 \xrightarrow{R_1} V_2 \xrightarrow{R_2} \cdots V_t \xrightarrow{R_t} V_{t+1} \cdots \xrightarrow{R_{l-1}} V_l$ , where in$R=R_1 \circ R_2 \circ \cdots \circ R_{l-1}$ denes the composite relations between node types $V_1$ and $V_l$ [25]. Take Figure 2(a) as an example, a meta-path “APA” represents the coauthor relationships on a paper (P) between two authors (A), and “APVPA” represents two authors (A) publish papers (P) in the same venue (V). Previous work has shown that many data mining tasks in heterogeneous information networks can benet from the modeling of meta-paths [6, 25, 27].

> 形式化地说，一个元路径方案 $\mathcal{P}$ 被定义为一条路径，其形式表示为
> $$
> V_1 \xrightarrow{R_1} V_2 \xrightarrow{R_2} \cdots V_t \xrightarrow{R_t} V_{t+1} \cdots \xrightarrow{R_{l-1}} V_l
> $$
> 其中 $R=R_1 \circ R_2 \circ \cdots \circ R_{l-1}$ 定义了节点类型 $V_1$ 和 $V_l$ 之间的复合关系[25]。以图2(a)为例，一个元路径“APA”表示两位作者（A）之间在同一篇论文（P）上的合著关系，而“APVPA”表示两位作者（A）在同一地点（V）发表了论文（P）。以前的工作已经表明，异构信息网络中的许多数据挖掘任务都可以从元路径建模中受益[6, 25, 27]。
>

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Figure2.png)

> **图2说明**：
>
> 图2展示了一个异构学术网络的示例以及用于嵌入这个网络的metapath2vec和metapath2vec++的skip-gram架构。
>
> (a) 部分展示了异构学术网络的一个实例。其中，黄色虚线表示合著者关系，红色虚线表示引用关系。
>
> (b) 部分展示了在预测节点 $a4$ 时 metapath2vec 所使用的 skip-gram 架构。如果忽略节点类型，这个架构与node2vec中的架构是相同的。$|V|=12$ 表示(a)中异构学术网络的节点数量，而 $a4$ 的邻居节点被设定为包括$CMU, a2, a3, a5, p2, p3, ACL, \& KDD$，因此 $k = 8$（即 $a4$ 的上下文节点数量）。
>
> (c) 部分展示了 metapath2vec++ 中使用的异构 skip-gram 架构。与在输出层为所有类型的邻居节点设置一组多项式分布不同，它为 $a4$ 的邻居中的每种类型的节点指定了一组多项式分布。$V_t$ 表示一种特定类型的 $t$ 节点，而 $V = V_V ∪ V_A ∪ V_O ∪ V_P$ 表示所有类型的节点集合。$k_t$ 指定了特定类型的邻居节点的大小，而$k = k_V + k_A + k_O + k_P$ 表示所有类型的邻居节点的总数。
>
> **计算与理解**：
>
> 在这个例子中，没有具体的数学计算，但我们可以从图中读取和理解网络结构和skip-gram模型的工作原理。在(b)和(c)中，我们可以看到skip-gram模型是如何根据中心节点（在这里是$a4$）来预测其上下文节点的。在metapath2vec中，我们不考虑节点的类型，而在metapath2vec++中，我们为每种类型的节点分别进行预测。
>
> 需要注意的是，图2主要是为了说明metapath2vec和metapath2vec++在处理异构网络时的不同方式，而不是提供具体的数学计算或数据。通过比较(b)和(c)，我们可以清楚地看到两种方法在处理节点类型时的差异。

Here we show how to use meta-paths to guide heterogeneous random walkers. Given a heterogeneous network $G = (V ,E,T)$ and  a meta-path scheme 
$$
\mathcal{P}: V_1 \xrightarrow{R_1} V_2 \xrightarrow{R_2} \cdots V_t \xrightarrow{R_t} V_{t+1} \cdots \xrightarrow{R_{l-1}} V_l
$$
the transition probability at step $i$ is defined as follows:
$$
p\left(v^{i+1} \mid v_t^i, \mathcal{P}\right)=\left\{\begin{array}{cl}
\frac{1}{\left|N_{t+1}\left(v_t^i\right)\right|} & \left(v^{i+1}, v_t^i\right) \in E, \phi\left(v^{i+1}\right)=t+1 \\
0 & \left(v^{i+1}, v_t^i\right) \in E, \phi\left(v^{i+1}\right) \neq t+1 \\
0 & \left(v^{i+1}, v_t^i\right) \notin E
\end{array}\right.
$$
where $v_t^i \in V_t$ and $N_{t+1}(v_t^i)$ denote the $V_{t+1}$ type of neighborhood of node $v_t^i$ . In other words, $v^{i+1} \in V_{t+1}$, that is, the flow of the walker is conditioned on the pre-defined meta-path $\mathcal{P}$. In addition, meta-paths are commonly used in a symmetric way, that is, its first node type $V_1$ is the same with the last one $V_l$ [25, 26, 28], facilitating its recursive guidance for random walkers, i.e.,
$$
p\left(v^{i+1} \mid v_t^i\right)=p\left(v^{i+1} \mid v_1^i\right), \text { if } t=l
$$
The meta-path-based random walk strategy ensures that the semantic relationships between different types of nodes can be properly incorporated into skip-gram. For example, in a traditional random walk procedure, in Figure 2(a), the next step of a walker on node $a4$ transitioned from node CMU can be all types of nodes surrounding it—$a2, a3, a5, p2, p3$, and CMU. However, under the meta-path scheme ‘OAPVPAO’, for example, the walker is biased towards paper nodes (P) given its previous step on an organization node CMU (O), following the semantics of this path.

> 这里展示了如何使用元路径来指导异构随机游走者。给定一个异构网络 $G = (V, E, T)$ 和一个元路径方案：
>
> $$
> \mathcal{P}: V_1 \xrightarrow{R_1} V_2 \xrightarrow{R_2} \cdots V_t \xrightarrow{R_t} V_{t+1} \cdots \xrightarrow{R_{l-1}} V_l
> $$
>
> 在步骤 $i$ 的转移概率定义如下：
>
> $$
> p\left(v^{i+1} \mid v_t^i, \mathcal{P}\right)=\left\{\begin{array}{cl}
> \frac{1}{\left|N_{t+1}\left(v_t^i\right)\right|} & \left(v^{i+1}, v_t^i\right) \in E, \phi\left(v^{i+1}\right)=t+1 \\
> 0 & \left(v^{i+1}, v_t^i\right) \in E, \phi\left(v^{i+1}\right) \neq t+1 \\
> 0 & \left(v^{i+1}, v_t^i\right) \notin E
> \end{array}\right.
> $$
>
> 其中，$v_t^i \in V_t$，而 $N_{t+1}(v_t^i)$ 表示节点 $v_t^i$ 的 $V_{t+1}$ 类型的邻居。换句话说，$v^{i+1} \in V_{t+1}$，即游走者的流动是基于预定义的元路径 $\mathcal{P}$ 的。此外，元路径通常以对称的方式使用，即其第一个节点类型$V_1$与最后一个节点类型$V_l$相同，这有助于其对随机游走者进行递归指导，即：
>
> $$
> p\left(v^{i+1} \mid v_t^i\right)=p\left(v^{i+1} \mid v_1^i\right), \text { if } t=l
> $$
>
> 基于元路径的随机游走策略确保了不同类型节点之间的语义关系可以被适当地纳入skip-gram模型中。例如，在传统的随机游走过程中（如图2(a)所示），位于节点$a4$上的游走者从CMU节点转移时，其下一步可以是围绕它的所有类型的节点——$a2, a3, a5, p2, p3$ 和CMU。然而，在元路径方案‘OAPVPAO’下，例如，给定游走者在上一步位于组织节点CMU（O）上，它会被偏向论文节点（P），遵循这条路径的语义。
>
> **我的理解**：
>
> 这段文字主要描述了如何在异构网络中使用元路径来指导随机游走。异构网络中的节点和边可以有多种类型，而元路径则定义了一种在网络中移动的特定路径，该路径遵循一定的节点和边的类型序列。通过这种方式，可以更有针对性地探索网络中的结构和关系。

#### 3.3 metapath2vec++

metapath2vec distinguishes the context nodes of nodev conditioned on their types when constructing its neighborhood function $N_t (v)$ in Eq. 2. However, it ignores the node type information in somax.

In other words, in order to infer the specic type of context ct in $N_t (v)$ given a node $v$, metapath2vec actually encourages all types of negative samples, including nodes of the same type $t$ as well as the other types in the heterogeneous network.

> metapath2vec在构建其邻域函数 $N_t(v)$ 时，会根据节点的类型来区分节点 $v$ 的上下文节点。这意味着，在给定一个节点 $v$ 时，metapath2vec会考虑与其关联的特定类型的上下文节点 。然而，在处理softmax函数时，metapath2vec却忽略了节点类型的信息。
>
> 换句话说，当给定节点 $v$ 并试图推断其邻域 $N_t(v)$ 中上下文节点$c_t$的具体类型时，metapath2vec实际上会鼓励所有类型的负样本，这包括与节点 $v$ 具有相同类型$t$的节点，以及异构网络中其他类型的节点。这可能导致在预测节点 $v$ 的上下文时，模型不会充分利用节点的类型信息，从而影响预测的准确性。
>

**Heterogeneous negative sampling**. We further propose the metapath2vec++ framework, in which the softmax function is normalized with respect to the node type of the context $c_t$ . Specifically, $p(c_t |v;θ)$ is adjusted to the specific node type $t$, that is,
$$
p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u_t \in V_t} e^{X_{u_t} \cdot X_v}}
$$
where $V_t$ is the node set of type $t$ in the network. In doing so, metapath2vec++ specifies one set of multinomial distributions for each type of neighborhood in the output layer of the skip-gram model. Recall that in metapath2vec and node2vec / DeepWalk, the dimension of the output multinomial distributions is equal to the number of nodes in the network. However, in metapath2vec++’s skip-gram, the multinomial distribution dimension for type $t$ nodes is determined by the number of $t$-type nodes. A clear illustration can be seen in Figure 2(c). For example, given the target node $a4$​ in the input layer, metapath2vec++ outputs four sets of multinomial distributions, each corresponding to one type of neighbors—venues V , authors A, organizations O , and papers P .

> **异构负采样**。我们进一步提出了metapath2vec++框架，该框架的 softmax 函数针对上下文 $c_t$ 的节点类型进行了归一化。具体来说，$p(c_t |v;θ)$ 被调整为针对特定节点类型 $t$，即：
> $$
> p\left(c_t \mid v ; \theta\right)=\frac{e^{X_{c_t} \cdot X_v}}{\sum_{u_t \in V_t} e^{X_{u_t} \cdot X_v}}
> $$
>
> 其中 $V_t$ 是网络中类型为 $t$ 的节点集合。通过这种方式，metapath2vec++在skip-gram模型的输出层为每个类型的邻域指定了一组多项分布。回想一下，在 metapath2vec 以及 node2vec/DeepWalk 中，输出多项分布的维度等于网络中的节点数。然而，在metapath2vec++的skip-gram中，类型为 $t$ 的节点的多项分布维度由 $t$ 类型节点的数量确定。图2(c)给出了一个清晰的说明。例如，给定输入层中的目标节点 $a4$，metapath2vec++ 会输出四组多项分布，每组分别对应一种类型的邻居——场所V、作者A、组织O和论文P。
>
> 通过这种方法，metapath2vec++能够更准确地捕捉和建模异构网络中节点之间复杂而多样的关系，特别是在节点类型丰富多样的场景下。这种改进使得metapath2vec++在处理异构网络嵌入任务时能够取得更好的性能，特别是在需要区分不同节点类型及其上下文信息的场景中。

Inspired by PTE [29], the sampling distribution is also specied by the node type of the neighbor $c_t$ that is targeted to predict, i.e., $P_t(·)$. therefore, we have the following objective:
$$
O(\mathbf{X})=\log \sigma\left(X_{c_t} \cdot X_v\right)+\sum_{m=1}^M \mathbb{E}_{u_t^m \sim P_t\left(u_t\right)}\left[\log \sigma\left(-X_{u_t^m} \cdot X_v\right)\right]
$$
whose gradients are derived as follows:
$$
\begin{aligned}
& \frac{\partial O(\mathbf{X})}{\partial X_{u_t^m}}=\left(\sigma\left(X_{u_t^m} \cdot X_v-\mathrm{I}_{c_t}\left[u_t^m\right]\right)\right) X_v \\
& \frac{\partial O(\mathbf{X})}{\partial X_v}=\sum_{m=0}^M\left(\sigma\left(X_{u_t^m} \cdot X_v-\mathbb{I}_{c_t}\left[u_t^m\right]\right)\right) X_{u_t^m}
\end{aligned}
$$
where $I_{c_t} [u^m_t ]$ is an indicator function to indicate whether $u^m_t$ is the neighborhood context node $c_t$ and when $m = 0, u^0_t = c_t$ . The model is optimized by using stochastic gradient descent algorithm. The pseudo code of metapath2vec++ is listed in Algorithm 1.

> 受PTE [29]的启发，采样分布也是由目标预测邻居$c_t$的节点类型指定的，即$P_t(·)$。因此，我们有以下目标函数：
>
> $$
> O(\mathbf{X})=\log \sigma\left(X_{c_t} \cdot X_v\right)+\sum_{m=1}^M \mathbb{E}_{u_t^m \sim P_t\left(u_t\right)}\left[\log \sigma\left(-X_{u_t^m} \cdot X_v\right)\right]
> $$
>
> 其梯度计算如下：
>
> $$
> \begin{aligned}
> & \frac{\partial O(\mathbf{X})}{\partial X_{u_t^m}}=\left(\sigma\left(X_{u_t^m} \cdot X_v-\mathrm{I}_{c_t}\left[u_t^m\right]\right)\right) X_v \\
> & \frac{\partial O(\mathbf{X})}{\partial X_v}=\sum_{m=0}^M\left(\sigma\left(X_{u_t^m} \cdot X_v-\mathbb{I}_{c_t}\left[u_t^m\right]\right)\right) X_{u_t^m}
> \end{aligned}
> $$
>
> 其中$I_{c_t} [u^m_t ]$是一个指示函数，用于指示$u^m_t$是否是邻域上下文节点$c_t$。当$m = 0$时，$u^0_t = c_t$。模型使用随机梯度下降算法进行优化。metapath2vec++的伪代码如算法1所示。
>
> 算法1中详细描述了metapath2vec++的训练过程，包括节点采样、上下文构建以及模型参数更新等步骤。通过这种方式，metapath2vec++能够更好地捕捉异构网络中节点间的复杂关系，并生成有区分度的节点嵌入表示，从而有助于后续的网络分析和挖掘任务。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Alg1.png)

## 4 EXPERIMENTS

In this section, we demonstrate the efficacy and efficiency of the presented metapath2vec and metapath2vec++ frameworks for het- erogeneous network representation learning.

**Data**. We use two heterogeneous networks, including the AMiner Computer Science (CS) dataset [31] and the Database and Infor- mation Systems (DBIS) dataset [26]. Both datasets and code are publicly available . This AMiner CS dataset consists of 9,323,739 computer scientists and 3,194,405 papers from 3,883 computer sci- ence venues—both conferences and journals—held until 2016. We construct a heterogeneous collaboration network, in which there are three types of nodes: authors, papers, and venues. The links represent different types of relationships among three sets of nodes— such as collaboration relationships on a paper.

The DBIS dataset was constructed and used by Sun et al. [26]. It covers 464 venues, their top-5000 authors, and corresponding 72,902 publications. We also construct the heterogeneous collaboration networks from DBIS wherein a link may connect two authors, one author and one paper, as well as one paper and one venue.

>  在本节中，我们将展示所介绍的metapath2vec和metapath2vec++框架在异构网络表示学习中的有效性和效率。
>
> **数据**。我们使用了两个异构网络，包括AMiner计算机科学（CS）数据集[31]和数据库与信息系统（DBIS）数据集[26]。这两个数据集和代码都是公开的。这个AMiner CS数据集包含了来自3883个计算机科学场所（包括会议和期刊）的9,323,739名计算机科学家和3,194,405篇论文，这些场所直到2016年都在举办活动。我们构建了一个异构协作网络，其中有三种类型的节点：作者、论文和场所。链接表示三组节点之间的不同类型的关系，例如论文上的协作关系。
>
> DBIS数据集由Sun等人构建和使用[26]。它涵盖了464个场所、他们的前5000名作者以及相应的72,902篇出版物。我们还从DBIS构建了异构协作网络，其中链接可能连接两个作者、一个作者和一篇论文，以及一篇论文和一个场所。

#### 4.1 Experimental Setup

We compare metapath2vec and metapath2vec++ with several recent network representation learning methods:

1. DeepWalk [22] / node2vec [8]: With the same random walk path input (p=1 & q=1 in node2vec), we find that the choice be- tween hierarchical softmax (DeepWalk) and negative sampling (node2vec) techniques does not yield significant differences. Therefore we use p=1 and q=1 [8] in node2vec for comparison.

2. LINE[30]:We use the advanced version of LINE by considering both the 1st- and 2nd-order of node proximity;

3. PTE[29]:We construct three bipartite heterogeneous networks (author–author, author–venue, venue–venue) and restrain it as an unsupervised embedding method;

4. Spectral Clustering [33] / Graph Factorization [2]: With the same treatment to these methods in node2vec [8], we exclude them from our comparison, as previous studies have demon- strated that they are outperformed by DeepWalk and LINE.

   For all embedding methods, we use the same parameters listed

below. In addition, we also vary each of them and fix the others for examining the parameter sensitivity of the proposed methods.

1. The number of walks per node w: 1000;
2. The walk length l : 100;
3. The vector dimension d: 128 (LINE: 128 for each order); 
4. The neighborhood size k : 7;
5. The size of negative samples: 5.

For metapath2vec and metapath2vec++, we also need to specify the meta-path scheme to guide random walks. We surveyed most of the meta-path-based work and found that the most commonly and effectively used meta-path schemes in heterogeneous academic networks are “APA” and “APVPA” [12, 25–27]. Notice that “APA” denotes the coauthor semantic, that is, the traditional (homoge- neous) collaboration links / relationships. “APVPA” represents the heterogeneous semantic of authors publishing papers at the same venues. Our empirical results also show that this simple meta-path scheme “APVPA” can lead to node embeddings that can be general- ized to diverse heterogeneous academic mining tasks, suggesting its applicability to potential applications for academic search services.

We evaluate the quality of the latent representations learned by different methods over three classical heterogeneous network mining tasks, including multi-class node classification [13], node clustering [27], and similarity search [26]. In addition, we also use the embedding projector in TensorFlow [1] to visualize the node embeddings learned from the heterogeneous academic networks.

> 我们将metapath2vec和metapath2vec++与几种最新的网络表示学习方法进行比较：
>
> 1. DeepWalk[22]/node2vec[8]：在相同的随机游走路径输入（node2vec 中的 p=1 & q=1）下，我们发现层次softmax（DeepWalk）和负采样（node2vec）技术之间的选择并没有产生显著差异。因此，我们在node2vec中使用p=1和q=1[8]进行比较。
>
> 2. LINE[30]：我们考虑节点邻近度的1阶和2阶，使用LINE的高级版本；
>
> 3. PTE[29]：我们构建三个二分异构网络（作者-作者，作者-场所，场所-场所），并将其限制为一种无监督嵌入方法；
>
> 4. 谱聚类[33]/图分解[2]：与node2vec[8]中对这些方法的相同处理一样，我们将其排除在我们的比较之外，因为先前的研究已经证明它们被DeepWalk和LINE超越。
>
> 对于所有嵌入方法，我们使用下面列出的相同参数。此外，我们还改变每个参数并固定其他参数，以检查所提出方法的参数敏感性。
>
> 1. 每个节点的游走次数 w：1000；
>
> 2. 游走长度 l：100；
>
> 3. 向量维度 d：128（LINE：每个阶数为128）；
>
> 4. 邻域大小 k：7；
>
> 5. 负样本的大小：5。
>
> 对于metapath2vec和metapath2vec++，我们还需要指定元路径方案来指导随机游走。我们调查了大多数基于元路径的工作，发现在异构学术网络中最常用且有效的元路径方案是“APA”和“APVPA”[12, 25–27]。请注意，“APA”表示合著者语义，即传统的（同质）协作链接/关系。“APVPA”表示作者在相同场所发表论文的异构语义。我们的实证结果也表明，这种简单的元路径方案“APVPA”可以引导节点嵌入，这些嵌入可以推广到各种异构学术挖掘任务，表明其适用于学术搜索服务的潜在应用。
>
> 我们评估了不同方法学习到的潜在表示在三个经典异构网络挖掘任务上的质量，包括多类节点分类[13]、节点聚类[27]和相似性搜索[26]。此外，我们还使用TensorFlow[1]中的嵌入投影仪来可视化从异构学术网络中学到的节点嵌入。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Table2.png)

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Table3.png)

#### 4.2 Multi-Class Classification

For the classication task, we use third-party labels to determine the class of each node. First, we match the eight categories2 of venues in Google Scholar3 with those in AMiner data. Among all of the 160 venues (20 per category × 8 categories), 133 of them are successfully matched and labeled correspondingly (Most of unmatched venues are pre-print venues, such as arXiv). Second, for each author who published in these 133 venues, his / her label is assigned to the category with the majority of his / her publications, and a tie is resolved by random selection among the possible categories; 246,678 authors are labeled with research category.

Note that the node representations are learned from the full dataset. e embeddings of above labeled nodes are then used as the input to a logistic regression classier. In the classication experiments, we vary the size of the training set from 5% to 90% and the remaining nodes for testing. We repeat each prediction experiment ten times and report the average performance in terms of both Macro-F1 and Micro-F1 scores.

**Results**. Tables 2 and 3 list the eight-class classication results. Overall, the proposed metapath2vec and metapath2vec++ models consistently and signicantly outperform all baselines in terms of both metrics. When predicting for the venue category, the advantage of both metapath2vec and metapath2vec++ are particular strong given a small size of training data. Given 5% of nodes as training data, for example, metapath2vec and metapath2vec++ achieve 0.08–0.23 (relatively 35–319%) improvements in terms of MacroF1 and 0.13–0.26 (relatively 39–145%) gains in terms of Micro-F1 over DeepWalk / node2vec, LINE, and PTE. When predicting for authors’ categories, the performance of each method is relatively stable when varying the train-test split. e constant gain achieved by the proposed methods is around 2-3% over LINE and PTE, and ∼20% over DeepWalk / node2vec.

In summary, metapath2vec and metapath2vec++ learn signi- cantly beer heterogeneous node embeddings than current stateof-the-art methods, as measured by multi-class classication performance. e advantage of the proposed methods lies in their proper consideration and accommodation of the network heterogeneity challenge—the existence of multiple types of nodes and relations.

**Parameter sensitivity**. In skip-gram-based representation learning models, there exist several common parameters (see Section 4.1). We conduct a sensitivity analysis of metapath2vec++ to these parameters. Figure 3 shows the classication results as a function of one chosen parameter when the others are controlled for. In general, we find that in Figures 3(a) and 3(b) the number of walks w rooting from each node and the length l of each walk are positive to the author classification performance, while they are surprisingly inconsequential for inferring venue nodes’ categories as measured by Macro-F1 and Micro-F1 scores. The increase of author clas- sification performance converges as w and l reach around 1000 and 100, respectively. Similarly, Figures 3(c) and 3(d) suggest that the number of embedding dimensions d and neighborhood size k are again of relatively little relevance to the predictive task for venues, and k on the other hand is positively crucial to determine the class of a venue. However, the descending lines as the increase of k for author classifications imply that a smaller neighborhood size actually produces the best embeddings for separating authors. This finding differs from those in a homogeneous environment [8], wherein the neighborhood size generally shows a positive effect on node classification.

According to the analysis, metapath2vec++ is not strictly sen- sitive to these parameters and is able to reach high performance under a cost-effective parameter choice (the smaller, the more ef- ficient). In addition, our results also indicate that those common parameters show different functions for heterogeneous network embedding with those in homogeneous network cases, demonstrat- ing the request of different ideas and solutions for heterogeneous network representation learning.

> 对于分类任务，我们使用第三方标签来确定每个节点的类别。首先，我们将Google Scholar3中的8个类别2的场所与AMiner数据中的场所进行匹配。在所有160个场所（每类20个×8类）中，有133个成功匹配并相应标记（大多数未匹配的场所是预印本场所，如arXiv）。其次，对于在这些133个场所发表文章的每位作者，其标签被分配给包含其大多数出版物的类别，如果有并列的情况，则通过从可能的类别中随机选择来解决；共有246,678位作者被标记为研究类别。
>
> 请注意，节点表示是从完整的数据集中学习的。上述标记节点的嵌入随后被用作逻辑回归分类器的输入。在分类实验中，我们将训练集的大小从5%变化到90%，其余的节点用于测试。我们重复每个预测实验十次，并报告在Macro-F1和Micro-F1得分方面的平均性能。
>
> **结果**。表2和表3列出了八类分类结果。总体而言，所提出的metapath2vec和metapath2vec++模型在两项指标上始终且显著优于所有基线。当预测场所类别时，给定小规模的训练数据，metapath2vec和metapath2vec++的优势特别强。例如，给定5%的节点作为训练数据，metapath2vec和metapath2vec++在Macro-F1方面实现了0.08–0.23（相对35–319%）的改进，在Micro-F1方面实现了0.13–0.26（相对39–145%）的增益，超过了DeepWalk/node2vec、LINE和PTE。当预测作者类别时，随着训练-测试分割的变化，每种方法的性能相对稳定。所提出的方法的恒定增益大约比LINE和PTE高2-3%，比DeepWalk/node2vec高约20%。
>
> 总之，与当前最先进的方法相比，metapath2vec和metapath2vec++学习到了显著更好的异构节点嵌入，这是通过多类分类性能来衡量的。所提出方法的优势在于它们适当地考虑和解决了网络异质性挑战——多种类型的节点和关系的存在。
>
> **参数敏感性**。在基于skip-gram的表示学习模型中，存在几个常见参数（见第4.1节）。我们对metapath2vec++的这些参数进行了敏感性分析。图3显示了当其他参数得到控制时，分类结果作为所选参数的函数。总的来说，我们发现，在图3(a)和图3(b)中，从每个节点开始的游走次数w和每个游走的长度l对作者分类性能有正面影响，但令人惊讶的是，它们对推断场所节点类别的预测任务几乎没有影响，这是通过Macro-F1和Micro-F1得分来衡量的。随着w和l分别增加到约1000和100，作者分类性能的增加趋于收敛。类似地，图3(c)和图3(d)表明，嵌入维度d的数量和邻域大小k对场所的预测任务相对不太相关，而k对确定场所的类别则至关重要。然而，随着k的增加，作者分类的下降线表明，较小的邻域大小实际上产生了最好的嵌入，用于分离作者。这一发现与同质环境中的发现不同[8]，其中邻域大小通常对节点分类有正面影响。
>
> 根据分析，metapath2vec++对这些参数并不严格敏感，并能够在成本效益高的参数选择下达到高性能（越小，效率越高）。此外，我们的结果还表明，这些常见参数在异构网络嵌入和同质网络情况下的功能不同，表明异构网络表示学习需要不同的想法和解决方案。

#### 4.3 Node Clustering

We illustrate how the latent representations learned by embed- ding methods can help the node clustering task in heterogeneous networks. We employ the same eight-category author and venue nodes used in the classification task above. The learned embeddings by each method is input to a clustering model. Here we leverage the k-means algorithm to cluster the data and evaluate the cluster- ing results in terms of normalized mutual information (NMI) [26]. In addition, we also report metapath2vec++’s sensitivity with re- spect to different parameter choices. All clustering experiments are conducted 10 times and the average performance is reported.

**Results**. Table 4 shows the node clustering results as measured by NMI in the AMiner CS data. Overall, the table demonstrates that metapath2vec and metapath2vec++ outperform all the compar- ative methods. When clustering for venues, the task is trivial as evident from the high NMI scores produced by most of the methods: metapath2vec, metapath2vec++, LINE, and PTE. Nevertheless, the proposed two methods outperform LINE and PTE by 2–3%. The author clustering task is more challenging than the venue case, and the gain obtained by metapath2vec and metapath2vec++ over the best baselines (LINE and PTE) is more significant—around 13–16%.

In summary, metapath2vec and metapath2vec++ generate more appropriate embeddings for different types of nodes in the network than comparative baselines, suggesting their ability to capture and incorporate the underlying structural and semantic relationships between various types of nodes in heterogeneous networks.

**Parametersensitivity**. Followingthesameexperimentalproce- dure in classification, we study the parameter sensitivity of meta- path2vec++ as measured by the clustering performance. Figure 4 shows the clustering performance as a function of each of the four parameters when fixing the other three. From Figures 4(a) and 4(b), we can observe that the balance between computational cost (a small w and l in x-axis) and efficacy (a high NMI in y-axis) can be achieved at around w = 800∼1000 and l = 100 for the clustering of both authors and venues. Further, different from the positive effect of increasing w and l on author clustering, d and k are negatively correlated with the author clustering performance, as observed from Figures 4(c) and 4(d). Similarly, the venue clustering performance also shows an descending trend with an increasing d, while on the other hand, we observe a first-increasing and then-decreasing NMI line when k is increased. Both figures together imply that d = 128 and k = 7 are capable of embedding heterogeneous nodes into latent space for promising clustering outcome.

> 我们展示了嵌入方法学习的潜在表示如何帮助异构网络中的节点聚类任务。我们使用了与上述分类任务中相同的八类作者和场所节点。每种方法学习的嵌入被输入到聚类模型中。在这里，我们利用k-means算法对数据进行聚类，并根据归一化互信息（NMI）[26]评估聚类结果。此外，我们还报告了metapath2vec++对不同参数选择的敏感性。所有聚类实验均进行10次，并报告平均性能。
>
> **结果**。表4显示了AMiner CS数据中通过NMI衡量的节点聚类结果。总体而言，该表表明metapath2vec和metapath2vec++在所有比较方法中表现最佳。当对场所进行聚类时，任务很简单，从大多数方法产生的高NMI得分中可以看出：metapath2vec、metapath2vec++、LINE和PTE。尽管如此，所提出的两种方法在场所聚类方面比LINE和PTE高出2–3%。作者聚类任务比场所情况更具挑战性，metapath2vec和metapath2vec++在最佳基线（LINE和PTE）上的增益更为显著——大约13–16%。
>
> 综上所述，metapath2vec和metapath2vec++为网络中的不同类型节点生成了比比较基线更合适的嵌入，表明它们能够捕获和融入异构网络中各种类型节点之间的潜在结构和语义关系。
>
> **参数敏感性**。遵循分类中的相同实验程序，我们研究了通过聚类性能衡量的metapath2vec++的参数敏感性。图4显示了当固定其他三个参数时，聚类性能作为四个参数中每一个的函数。从图4(a)和图4(b)中，我们可以观察到，对于作者和场所的聚类，在计算成本（x轴上的小w和l）和效率（y轴上的高NMI）之间的平衡可以在w=800∼1000和l=100左右实现。此外，与增加w和l对作者聚类的正面影响不同，d和k与作者聚类性能呈负相关，如图4(c)和图4(d)所示。类似地，随着d的增加，场所聚类性能也显示出下降趋势，而另一方面，当k增加时，我们观察到NMI线首先增加然后下降。两图共同表明，d=128和k=7能够将异构节点嵌入到潜在空间中，以获得有希望的聚类结果。

#### 4.4  Case Study: Similarity Search

We conduct two case studies to demonstrate the efficacy of our methods. We select 16 top CS conferences from the corresponding sub-fields in the AMiner CS data and another 5 from the DBIS data. This results in a total of 21 query nodes. We use cosine similarity to determine the distance (similarity) between the query node and the remaining others.

Table 5 lists the top ten similar results for querying the 16 leading conferences in corresponding computer science sub-fields. One can observe that for the query “ACL”, for example, metapath2vec++ returns venues with the same focus—natural language processing, such as EMNLP (1st ), NAACL (2nd ), Computational Linguistics (3r d ), CoNLL (4t h ), COLING (5t h ), and so on. Similar performance can be also achieved when querying the other conferences from various fields. More surprisingly, we find that in most cases, the top three results cover venues with similar prestige to the query one, such as STOC to FOCS in theory, OSDI to SOSP in system, HPCA to ISCA in architecture, CCS to S&P in security, CSCW to CHI in human-computer interaction, EMNLP to ACL in NLP, ICML to NIPS in machine learning, WSDM to WWW in Web, AAAI to IJCAI in artificial intelligence, PVLDB to SIGMOD in database, etc. Similar results can also be observed in Tables 6 and 1, which show the similarity search results for the DBIS network.

> 我们进行了两个案例研究，以证明我们方法的有效性。我们从AMiner CS数据中选择了16个顶级计算机科学会议，并从DBIS数据中选择了另外5个。这导致总共有21个查询节点。我们使用余弦相似度来确定查询节点与其他节点之间的距离（相似度）。
>
> 表5列出了在相应计算机科学子字段中查询16个领先会议的前十个相似结果。可以观察到，例如，对于“ACL”查询，metapath2vec++返回了具有相同焦点的场所——自然语言处理，如EMNLP（第1名）、NAACL（第2名）、计算语言学（第3名）、CoNLL（第4名）、COLING（第5名）等。当从各个字段查询其他会议时，也可以实现类似的性能。更令人惊讶的是，我们发现，在大多数情况下，前三名结果涵盖了与查询节点相似声望的场所，例如在理论领域中的STOC到FOCS，系统领域中的OSDI到SOSP，架构领域中的HPCA到ISCA，安全领域中的CCS到S&P，人机交互领域中的CSCW到CHI，NLP领域中的EMNLP到ACL，机器学习领域中的ICML到NIPS，Web领域中的WSDM到WWW，人工智能领域中的AAAI到IJCAI，数据库领域中的PVLDB到SIGMOD等。在表6和表1中也可以观察到类似的结果，它们显示了DBIS网络的相似搜索结果。

#### 4.5 Case Study: Visualization

We employ the TensorFlow embedding projector to further visualize the low-dimensional node representations learned by embedding models. First, we project multiple types of nodes—16 top CS confer- ences and corresponding top-profile authors—into the same space in Figure 1. From Figure 1(d), we can clearly see that metapath2vec++ is able to automatically organize these two types of nodes and im- plicitly learn the internal relationships between them, indicated by the similar directions and distances of the arrows connecting each pair of them, such as J. Dean → OSDI, C. D. Manning → ACL, R. E. Tarjan → FOCS, M. I. Jordan → NIPS, and so on. In addition, these two types of nodes are clearly located in two separate and straight columns. Neither of these two results can be made by the recent network embedding models in Figures 1(a) and 1(b).

As to metapath2vec, instead of separating the two types of nodes into two columns, it is capable of grouping each pair of one venue and its corresponding author closely, such as R. E. Tarjan and FOCS, H. Jensen and SIGGRAPH, H. Ishli and CHI, R. Agrawal and SIG- MOD, etc. Together, both models arrange nodes from similar fields close to each other and dissimilar ones distant from each other, such as the “Core CS” cluster of systems (OSDI), networking (SIGCOMM), security (S&P), and architecture (ISCA), as well as the “Big AI” clus- ter of data mining (KDD), information retrieval (SIGIR), artificial intelligence (AI), machine learning (NIPS), NLP (ACL), and vision (CVPR). These groupings are also reflected by their corresponding author nodes.

Second, Figure 5 visualizes the latent vectors—learned by meta- path2vec++—of 48 venues used in similarity search of Section 4.4, three each from 16 sub-elds. We can see that conferences from the same domain are geographically grouped to each other and each group is well separated from others, further demonstrating the embedding ability of metapath2vec++. In addition, similar to the observation in Figure 1, we can also notice that the heterogeneous embeddings are able to unveil the similarities across dierent domains, including the “Core CS” sub-eld cluster at the boom right and the “Big AI” sub-eld cluster at the top right.

Thus, Figures 1 and 5 intuitively demonstrate metapath2vec++’s novel capability to discover, model, and capture the underlying structural and semantic relationships between multiple types of nodes in heterogeneous networks.

> 我们使用TensorFlow嵌入投影仪进一步可视化嵌入模型学习的低维节点表示。首先，我们将多种类型的节点——16个顶级计算机科学会议和相应的高知名度作者——投影到图1中的同一空间中。从图1(d)中，我们可以清楚地看到metapath2vec++能够自动组织这两种类型的节点，并隐式地学习它们之间的内部关系，这由连接每对节点的箭头的相似方向和距离表示，例如J. Dean → OSDI，C. D. Manning → ACL，R. E. Tarjan → FOCS，M. I. Jordan → NIPS等。此外，这两种类型的节点清晰地位于两个分开且笔直的列中。这两个结果都无法通过图1(a)和图1(b)中的最近网络嵌入模型获得。
>
> 至于metapath2vec，它并没有将两种类型的节点分成两列，而是能够将每个场所及其对应的作者紧密地组合在一起，例如R. E. Tarjan和FOCS，H. Jensen和SIGGRAPH，H. Ishli和CHI，R. Agrawal和SIG-MOD等。总的来说，两种模型都将来自相似领域的节点排列在一起，而将来自不同领域的节点分开，例如系统（OSDI）、网络（SIGCOMM）、安全（S&P）和架构（ISCA）的“核心计算机科学”集群，以及数据挖掘（KDD）、信息检索（SIGIR）、人工智能（AI）、机器学习（NIPS）、NLP（ACL）和视觉（CVPR）的“大数据智能”集群。这些分组也反映在其相应的作者节点上。
>
> 其次，图5可视化了在4.4节的相似度搜索中使用的48个场所的潜在向量，每个子领域各有3个。我们可以看到，来自同一领域的会议在地理上相互分组，并且每个组都与其他组很好地分开，进一步证明了metapath2vec++的嵌入能力。此外，类似于图1中的观察，我们还可以注意到，异构嵌入能够揭示不同领域之间的相似性，包括右下角的“核心计算机科学”子领域集群和右上角的“大数据智能”子领域集群。
>
> 因此，图1和图5直观地展示了metapath2vec++在发现、建模和捕获异构网络中多种类型节点之间的潜在结构和语义关系方面的新颖能力。

![Figure456](/Users/anmingyu/Github/Gor-rok/Papers/Embedding/metapath2vec/Figure456.png)

#### 4.6 Scalability

In the era of big (network) data, it is necessary to demonstrate the scalability of the proposed network embedding models. e metapath2vec and metapath2vec++ methods can be parallelized by using the same mechanism as word2vec and node2vec [8, 18]. All codes are implemented in C and C++ and our experiments are conducted in a computing server with ad 12 (48) core 2.3 GHz Intel Xeon CPUs E7-4850. We run experiments on the AMiner CS data with the default parameters with dierent number of threads, i.e., 1, 2, 4, 8, 16, 24, 32, 40, each of them utilizing one CPU core.

Figure 6 shows the speedup of metapath2vec & metapath2vec++ over the single-threaded case. Optimal speedup performance is denoted by the dashed y = x line, which represents perfect distribution and execution of computation across all CPU cores. In general, we nd that both methods achieve acceptable sublinear speedups as both lines are close to the optimal line. In specic, they can reach 11–12× speedup with 16 cores and 24–32× speedup with 40 cores used. By using 40 cores, metapath2vec++’s learning process costs only 9 minutes for embedding the full AMiner CS network, which is composed of over 9 million authors with 3 million papers published in more than 3800 venues. Overall, the proposed metapath2vec and metapath2vec++ models are ecient and scalable for large-scale heterogeneous networks with millions of nodes.

> 在大数据（网络）时代，有必要展示所提出的网络嵌入模型的可扩展性。metapath2vec和metapath2vec++方法可以使用与word2vec和node2vec[8,18]相同的机制进行并行化。所有代码都是用C和C++实现的，我们的实验是在一台具有12（48）核心2.3 GHz Intel Xeon CPUs E7-4850的计算服务器上进行的。我们在AMiner CS数据上运行实验，使用默认参数和不同的线程数，即1、2、4、8、16、24、32、40，每个线程使用一个CPU核心。
>
> 图6显示了metapath2vec和metapath2vec++相对于单线程情况的加速比。最佳加速性能由虚线y=x表示，这代表在所有CPU核心之间完美分配和执行计算。总的来说，我们发现两种方法都实现了可接受的亚线性加速，因为两条线都接近最优线。具体来说，它们可以在使用16个核心时达到11-12倍的加速，在使用40个核心时达到24-32倍的加速。通过使用40个核心，metapath2vec++的学习过程仅花费9分钟来嵌入完整的AMiner CS网络，该网络由超过900万作者组成，他们在超过3800个场所发表了300万篇论文。总的来说，所提出的metapath2vec和metapath2vec++模型对于具有数百万节点的大规模异构网络是高效且可扩展的。

## 5 RELATED WORK

Network representation learning can be traced back to the usage of latent factor models for network analysis and graph mining tasks [10, 34], such as the application of factorization models for rec- ommendation systems [14, 16], node classification [32], relational mining [19], and role discovery [9]. This rich line of research focuses on factorizing the matrix/tensor format (e.g., the adjacency matrix) of a network, generating latent-dimension features for nodes or edges in this network. However, the computational cost of decom- posing a large-scale matrix/tensor is usually very expensive, and also suffers from its statistical performance drawback [8], making it neither practical nor effective for addressing tasks in big networks.

With the advent of deep learning techniques, significant effort has been devoted to designing neural network-based representa- tion learning models. For example, Mikolov et al. proposed the word2vec framework—a two-layer neural network—to learn the distributed representations of words in natural language [17, 18]. Building on word2vec, Perozzi et al. suggested that the “context” of a node can be denoted by their co-occurrence in a random walk path [22]. Formally, they put random walkers over networks to record their walking paths, each of which is composed of a chain of nodes that could be considered as a “sentence” of words in a text corpus. More recently, in order to diversify the neighborhood of a node, Grover & Leskovec presented biased random walkers—a mixture of breadth-first and width-first search procedures—over networks to produce paths of nodes [8]. With node paths gener- ated, both works leveraged the skip-gram architecture in word2vec to model the structural correlations between nodes in a path. In addition, several other methods have been proposed for learning representations in networks [4, 5, 11, 20, 23]. In particular, to learn network embeddings, Tang et al. decomposed a node’s context into first-order (friends) and second-order (friends’ friends) prox- imity [30], which was further developed into a semi-supervised model PTE for embedding text data [29].

Our work furthers this direction of investigation by designing the metapath2vec and metapath2vec++ models to capture hetero- geneous structural and semantic correlations exhibited from large- scale networks with multiple types of nodes, which can not be handled by previous models, and applying these models to a variety of network mining tasks.

> 网络表示学习可以追溯到潜在因子模型在网络分析和图形挖掘任务中的使用[10, 34]，例如将分解模型应用于推荐系统[14, 16]、节点分类[32]、关系挖掘[19]和角色发现[9]。这一丰富的研究线路专注于分解网络的矩阵/张量格式（例如，邻接矩阵），为该网络中的节点或边生成潜在维度特征。然而，分解大规模矩阵/张量的计算成本通常非常高，而且其统计性能也存在缺陷[8]，使得它既不实用也不有效，无法解决大网络中的任务。
>
> 随着深度学习技术的出现，已经投入了大量精力来设计基于神经网络的表示学习模型。例如，Mikolov等人提出了word2vec框架——一个两层神经网络——来学习自然语言中的词的分布式表示[17, 18]。在word2vec的基础上，Perozzi等人提出节点的“上下文”可以通过它们在随机游走路径中的共现来表示[22]。正式地，他们将随机游走者放在网络上以记录它们的游走路径，每条路径都由一系列节点组成，这些节点可以被视为文本语料库中的“句子”中的词。最近，为了多样化节点的邻域，Grover和Leskovec提出了偏置随机游走者——一种混合的广度优先和宽度优先搜索程序——在网络上产生节点路径[8]。在生成节点路径后，这两项工作都利用了word2vec中的skip-gram架构来建模路径中节点之间的结构相关性。此外，还提出了几种其他方法来学习网络中的表示[4, 5, 11, 20, 23]。特别是，为了学习网络嵌入，Tang等人将一个节点的上下文分解为一阶（朋友）和二阶（朋友的朋友）邻近度[30]，这进一步发展为半监督模型PTE，用于嵌入文本数据[29]。
>
> 我们的工作通过设计metapath2vec和metapath2vec++模型来进一步推进这一研究方向，以捕获具有多种类型节点的大规模网络所表现出的异构结构和语义相关性，这是以前的模型无法处理的，并将这些模型应用于各种网络挖掘任务。

## 6 CONCLUSION

In this work, we formally define the representation learning prob- lem in heterogeneous networks in which there exist diverse types of nodes and links. To address the network heterogeneity chal- lenge, we propose the metapath2vec and metapath2vec++ meth- ods. We develop the meta-path-guided random walk strategy in a heterogeneous network, which is capable of capturing both the structural and semantic correlations of differently typed nodes and relations. To leverage this method, we formalize the heterogeneous neighborhood function of a node, enabling the skip-gram-based maximization of the network probability in the context of multiple types of nodes. Finally, we achieve effective and efficient optimiza- tion by presenting a heterogeneous negative sampling technique. Extensive experiments demonstrate that the latent feature repre- sentations learned by metapath2vec and metapath2vec++ are able to improve various heterogeneous network mining tasks, such as sim- ilarity search, node classification, and clustering. Our results can be naturally applied to real-world applications in heterogeneous academic networks, such as author, venue, and paper search in academic search services.

Future work includes various optimizations and improvements. For example, 1) the metapath2vec and metapath2vec++ models, as is also the case with DeepWalk and node2vec, face the challenge of large intermediate output data when sampling a network into a huge pile of paths, and thus identifying and optimizing the sampling space is an important direction; 2) as is also the case with all meta- path-based heterogeneous network mining methods, metapath2vec and metapath2vec++ can be further improved by the automatic learning of meaningful meta-paths; 3) extending the models to incorporate the dynamics of evolving heterogeneous networks; and 4) generalizing the models for different genres of heterogeneous networks.

> 在这项工作中，我们正式定义了异构网络中的表示学习问题，其中存在多种类型的节点和链接。为了应对网络异质性挑战，我们提出了metapath2vec和metapath2vec++方法。我们在异构网络中开发了基于元路径指导的随机游走策略，该策略能够捕获不同类型节点和关系的结构和语义相关性。为了利用这种方法，我们形式化了节点的异构邻域函数，使得在多种类型节点的上下文中最大化网络概率成为可能。最后，我们通过提出一种异构负采样技术实现了有效且高效的优化。广泛的实验表明，metapath2vec和metapath2vec++学习的潜在特征表示能够改善各种异构网络挖掘任务，如相似性搜索、节点分类和聚类。我们的结果可以自然地应用于异构学术网络中的现实世界应用，如学术搜索服务中的作者、场所和论文搜索。
>
> 未来的工作包括各种优化和改进。例如，1）metapath2vec和metapath2vec++模型，与DeepWalk和node2vec一样，面临着将网络采样到大量路径时产生大量中间输出数据的挑战，因此识别和优化采样空间是一个重要方向；2）与所有基于元路径的异构网络挖掘方法一样，metapath2vec和metapath2vec++可以通过自动学习有意义的元路径来进一步改进；3）扩展模型以融入演化异构网络的动态；4）将模型泛化到不同类型的异构网络。