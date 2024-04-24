## Learning Tree-based Deep Model for Recommender Systems

### ABSTRACT

Model-based methods for recommender systems have been studied extensively in recent years. In systems with large corpus, however, the calculation cost for the learnt model to predict all useritem preferences is tremendous, which makes full corpus retrieval extremely difficult. To overcome the calculation barriers, models such as matrix factorization resort to inner product form (i.e., model user-item preference as the inner product of user, item latent factors) and indexes to facilitate efficient approximate k-nearest neighbor searches. However, it still remains challenging to incorporate more expressive interaction forms between user and item features, e.g., interactions through deep neural networks, because of the calculation cost.

> 近年来，model-based 的推荐系统方法得到了广泛研究。然而，在大规模数据集的系统中，学习模型以预测所有 user-item 偏好所需的计算成本非常高，这使得全量数据集检索变得极为困难。为了克服算力障碍，模型如矩阵分解，采用内积形式来建模 user-item 偏好（即将 user 和 item 的潜在因子进行内积运算），并利用索引来实现高效的近似的 k-nn 搜索。然而，由于计算成本的问题，要想将更具表达力的 user 和 item 特征之间的交互形式（如深度神经网络）纳入模型仍然具有挑战性。

In this paper, we focus on the problem of introducing arbitrary advanced models to recommender systems with large corpus. We propose a novel tree-based method which can provide logarithmic complexity w.r.t. corpus size even with more expressive models such as deep neural networks. Our main idea is to predict user interests from coarse to fine by traversing tree nodes in a top-down fashion and making decisions for each user-node pair. We also show that the tree structure can be jointly learnt towards better compatibility with users’ interest distribution and hence facilitate both training and prediction. Experimental evaluations with two large-scale real-world datasets show that the proposed method significantly outperforms traditional methods. Online A/B test results in Taobao display advertising platform also demonstrate the effectiveness of the proposed method in production environments.

> 本文着重解决了将任意高级模型引入到具有大规模数据集的推荐系统中的问题。我们提出了一种新的Tree-based方法，即使使用更具表达力的模型（如深度神经网络），也能（相对于数据集大小）保持对数复杂度。我们的主要思想是通过自上而下地遍历树节点，并为每个user-item组合做出决策，从粗粒度到细粒度预测用户兴趣。我们还证明树结构可以 jointly learn，以更好地与用户的兴趣分布相适应，从而提升训练和预测两方面的性能。通过对两个大规模实际数据集的实验评估，我们发现新方案明显优于传统方案。此外，在淘宝广告平台进行的线上A/B实验结果也证明了新方案在实际生产环境中的有效性。

Recommendation has been widely used by various kinds of content providers. Personalized recommendation method, based on the intuition that users’ interests can be inferred from their historical behaviors or other users with similar preference, has been proven to be effective in YouTube [7] and Amazon [22].

> 推荐系统已被各种内容提供商广泛使用。个性化推荐方法基于这样的认识：用户的兴趣可以从他们的历史行为或具有相似偏好的其他用户中进行推断，这在YouTube [7]和Amazon [22]中已被证明是有效的。

Designing such a recommendation model to predict the best candidate set from the entire corpus for each user has many challenges. In systems with enormous corpus, some well-performed recommendation algorithms may fail to predict from the entire corpus. The linear prediction complexity w.r.t. the corpus size is unacceptable. Deploying such large-scale recommender system requires the amount of calculation to predict for each single user be limited. And besides preciseness, the novelty of recommended items should also be responsible for user experience. Results that only contain homogeneous items with user’s historical behaviors are not expected.

> 设计一个能够从整个数据集中预测出最佳候选集合的推荐模型其中存在许多挑战。在拥有庞大数据集的系统中，一些表现良好的推荐算法可能无法对整个数据集进行预测。与数据集大小成线性关系的预测复杂度是不可接受的。部署这样的大规模推荐系统需要对每个用户预测的计算量有所限制。用户体验上，推荐的 item 的新颖性也很重要。仅包含用户历史行为中的同质化item的结果是不被期望的。

To reduce the amount of calculation and handle enormous corpus, memory-based collaborative filtering methods are widely deployed in industry [22]. As a representative method in collaborative filtering family, item-based collaborative filtering [31] can recommend from very large corpus with relatively much fewer computations, depending on the precalculated similarity between item pairs and using user’s historical behaviors as triggers to recall those most similar items. However, there exists restriction on the scope of candidate set, i.e., not all items but only items similar to the triggers can be ultimately recommended. This intuition prevents the recommender system from jumping out of historical behavior to explore potential user interests, which limits the accuracy of recalled results. And in practice the recommendation novelty is also criticized. Another way to reduce calculation is making coarse-grained recommendation. For example, the system recommends a small number of item categories for users and picks out all corresponding items, with a following ranking stage. However, for large corpus, the calculation problem is still not solved. If the category number is large, the category recommendation itself also meets the calculation barrier. If not, some categories will inevitably include too many items, making the following ranking calculation impracticable. Besides, the used categories are usually not designed for recommendation problem, which can seriously harm the recommendation accuracy.  

> 为了减少计算量并处理庞大的数据集，内存型协同过滤方法被广泛应用于工业界 [22]。作为协同过滤家族中的代表性方法，基于 item 的协同过滤 [31] 可以从非常庞大的数据集中进行推荐，并且相对较少的计算量，这取决于预先计算的 item-pair 之间的相似度，并使用用户的历史行为作为 trigger 来寻找那些最相似的 item。然而，候选集的范围存在限制，即最终只能推荐与 trigger 相似的 item，而不是所有的 item。这样阻止了推荐系统从用户历史行为中跳出来探索潜在的用户兴趣，这限制了被推荐结果的准确性（译注：原文 which limits the accuracy of recalled results）。实际上也限制了新颖性。
>
> 另一种减少计算量的方法是进行粗粒度的推荐。例如，系统为用户推荐少量item类别，然后从类别中挑选出所有相应的item，在随后的排序阶段对它们进行排序。然而，对于大规模数据集，算力问题仍然没有解决。如果类别数量很多，那么推荐类别本身也会遇到算力障碍。如果类别数量较少，某些类别将不可避免地包含太多的物品，使得随后的排序计算变得不切实际。此外，所使用的类别通常并非专为推荐问题设计，这可能严重损害推荐的准确性。

However, the inner product interaction form between user and item’s vector representations severely limits model’s capability. There exist many other kinds of more expressive interaction forms, for example, cross-product features between user’s historical behaviors and candidate items are widely used in click-through rate prediction [5]. Recent work [13] proposes a neural collaborative filtering method, where a neural network instead of inner product is used to model the interaction between user and item’s vector representations. The work’s experimental results prove that a multi-layer feed-forward neural network performs better than the fixed inner product manner. Deep interest network [34] points out that user in- terests are diverse, and an attention like network structure can generate varying user vectors according to different candidate items. Beyond the above works, other methods like product neural network [27] have also proven the effectiveness of advanced neural networks. However, as these kinds of models can not be regulated to inner product form between user and item vectors to utilize efficient approximate kNN search, they can not be used to recall candidates in large-scale recommender systems. How to overcome the calculation barrier to make arbitrary advanced neural networks feasible in large-scale recommendation is a problem. 

> 然而，user 和 item 向量表示之间的内积交互形式严重限制了模型的能力。存在许多其他更具表达力的交互形式，例如，用户历史行为和候选item之间的交叉特征在点击率预测中被广泛使用[5]。最近的研究[13]提出了一种神经协同过滤方法，其中使用神经网络而不是内积来建模用户和物品向量表示之间的交互。该研究的实验结果证明，多层前馈神经网络的表现优于固定的内积方式。deep interest network [34]指出用户的兴趣是多样的，并且类似注意力的网络结构可以根据不同的候选 item 生成不同的用户向量。除了上述工作，其他方法如产品 product neural network[27]也证明了先进神经网络的有效性。然而，由于这些模型无法将 user和 item 向量调整为内积形式以利用高效的近似k最近邻搜索，它们不能用于在大规模推荐系统中检索候选 item。如何克服算力瓶颈，使任意先进神经网络在大规模推荐中成为可能，是一个问题。

To address the challenges above, we propose a novel tree-based deep recommendation model (TDM) in this paper. Tree and tree-based methods are researched in multiclass classification problem [1–3, 6, 15, 26, 32], where tree is usually used to partition the sample or label space to reduce calculation cost. However, researchers seldom set foot in the context of recommender systems using tree structure as an index for retrieval. Actually, hierarchical structure of information ubiquitously exists in many domains. For example, in E-commerce scenario, iPhone is the fine-grained item while smartphone is the coarse-grained concept to which iPhone belongs. The proposed TDM method leverages this hierarchy of information and turns recommendation problem into a series of hierarchical classification problems. By solving the problem from easy to difficult, TDM can improve both accuracy and efficiency.

> 为了解决上述挑战，本文提出了一种新颖的基于树的深度推荐模型（TDM）。在多分类问题中，研究人员通常使用树来划分样本或标签空间以降低计算成本。然而，研究人员很少在使用树结构作为检索索引的推荐系统上进行探索。实际上，信息的层次结构普遍存在于许多领域。例如，在电子商务场景中，iPhone是精细粒度的物品，而智能手机是iPhone所属的粗粒度概念。所提出的TDM方法利用信息的这种层次结构，将推荐问题转化为一系列的层次分类问题。通过从易到难解决问题，TDM可以提高准确性和效果。

The main contributions of our paper can be summarized as follows:

- To our best knowledge, TDM is the first method that makes arbitrary advanced models possible in generating recommendations from large corpus. Benefiting from hierarchical tree search, TDM achieves logarithmic amount of calculation w.r.t. corpus size when making prediction. 
- TDM can help find novel but effective recommendation results more precisely, because the entire corpus is explored and more effective deep models also can help find potential interests. 
- Besides more advanced models, TDM also promotes recommendation accuracy by hierarchical search, which divides a large problem into smaller ones and solves them successively from easy to difficult. 
- As a kind of index, the tree structure can also be learnt towards optimal hierarchy of items and concepts for more effective retrieval, which in turn facilitates the model training. We employ a tree learning method that allows joint training of neural network and the tree structure. 
- We conduct extensive experiments on two large-scale real- world datasets, which show that TDM outperforms existing methods significantly. 

> 本篇论文的主要贡献总结如下：
>
> - 据我们所知，TDM 是第一种使任意高级模型能够从大型数据集生成推荐结果的方法。通过利用层次树搜索，TDM在进行预测时的计算量与数据集大小成对数关系。
> - TDM能够更精确地找到新颖且有效的推荐结果，因为它能够探索整个数据集，并且更有效的深度模型也能帮助发现潜在兴趣。
> - 除了使用更先进的模型，TDM还通过层次搜索提高了推荐准确性，将一个大问题划分为多个小问题，并逐步从易到难解决这些问题。
> - 作为一种索引，树结构可以学习到 item 和 concepts 的最优层次结构，从而更有效的检索，进而促使模型训练。我们采用了一种树学习方法，可以联合训练神经网络和树结构。
> - 我们在两个大规模真实世界数据集上进行了大量实验，结果表明TDM明显优于现有方法。

It’s worth mentioning that tree-based approach is also researched in language model work hierarchical softmax [24], but it’s different from the proposed TDM not only in motivation but also in formulation. In next-word prediction problem, conventional softmax has to calculate the normalization term to get any single word’s probability, which is very time-consuming. Hierarchical softmax uses tree structure, and next-word’s probability is converted to the product of node probabilities along the tree path. Such formulation reduces the computation complexity of next-word’s probability to logarithmic magnitude w.r.t. the corpus size. However, in recommendation problem, the goal is to search the entire corpus for those most preferred items, which is a retrieval problem. In hierarchical softmax tree, the optimum of parent nodes can not guarantee that the optimal low level nodes are in their descendants, and all items still need to be traversed to find the optimal one. Thus, it’s not suitable for such a retrieval problem. To address the retrieval problem, we propose a max-heap like tree formulation and introduce deep neural networks to model the tree, which forms an efficient method for large-scale recommendation. The following sections will show its difference in formulation and its superiority in performance. In addition, hierarchical softmax adopts a single hidden layer network for a specific natural language processing problem, while the proposed TDM method is practicable to engage any neural network structures. 

> 值得一提的是，树结构方法在语言模型工作中也有研究，如 hierarchical softmax [24]，但它与提出的TDM不仅在动机上有所区别，而且在表述方式上也存在差异。在下一个单词预测问题中，传统的softmax需要计算归一化项才能得到单词的概率，这非常耗时。hierarchical softmax 使用树结构，将下一个单词的概率转换为沿树路径的节点概率的乘积。这种表述方式将下一个单词的概率的计算复杂度降低到与语料库大小成对数量级的程度。然而，在推荐问题中，目标是在整个数据集中搜索那些最受喜爱的物品，这是一个检索问题。在 hierarchical softmax树中，父节点的最优性不能保证最优的低层节点在它们的后代中，仍然需要遍历所有item 来找到最优解。因此，它不适用于这样的检索问题。为了解决检索问题，我们提出了类似于最大堆的树形表述，并引入深度神经网络来建模该树结构，从而形成了一种适用于大规模推荐的高效方法。接下来的部分将展示它在表述方式上的差异以及在性能上的优越性。此外，hierarchical softmax 为特定的自然语言处理问题采用了单隐藏层网络，而所提出的TDM方法适用于任何神经网络结构。

The proposed tree-based model is a universal solution for all kinds of online content providers. The remainder of this paper is organized as follows: In Section 2, we’ll introduce the system architecture of Taobao display advertising to show the position of the proposed method. Section 3 will give a detailed introduction and formalization of the proposed tree-based deep model. And the following Section 4 will describe how the tree-based model serves online. Experimental results on large-scale benchmark dataset and Taobao advertising dataset are shown in Section 5. At last, Section 6 gives our work a conclusion. 

> 提出的 Tree-based 的模型是一种适用于各种在线内容提供商的通用解决方案。本文的剩余部分组织如下：在第2节中，我们将介绍淘宝展示广告系统的体系结构，以展示所提方法所作用的位置。第3节将对提出的基于树的深度模型进行详细的介绍和形式化描述。接下来的第4节将描述基于树的模型如何服务于在线推荐。第5节展示了在大规模基准数据集和淘宝广告数据集上的实验结果。最后，第6节对我们的工作进行总结。

## 2 SYSTEM ARCHITECTURE

In this section, we introduce the architecture of Taobao display advertising recommender system as Figure 1. After receiving page view request from a user, the system uses user features, context features and item features as input to generate a relatively much smaller set (usually hundreds) of candidate items from the entire corpus (hundreds of millions) in the matching server. The tree-based recommendation model takes effort in this stage and shrinks the size of candidate set by several orders of magnitude.

> 在本节中，我们介绍了淘宝展示广告推荐系统的架构，如图1所示。当用户发送页面浏览请求后，系统将使用用户特征、context特征和商品特征作为输入，在召回(matching)服务器中从整个数据集（数亿个商品）中生成一个相对较小的候选商品集合（通常为数百个）。Tree-based 推荐模型在这一阶段发挥了作用，将候选集的大小缩减了几个数量级。

With hundreds of candidate items, the real-time prediction server uses more expressive but also more time consuming models [11, 34] to predict indicators like click-through rate or conversion rate. And after ranking by strategy [17, 35], several items are ultimately impressed to user.

> 有了数百个候选商品，实时预测服务器使用更加表达能力但也更耗时的模型[11, 34]来预测诸如点击率或转化率等指标。经过策略性的排序[17, 35]之后，最终会向用户展示几个商品。

As aforementioned, the proposed recommendation model aims to construct a candidate set with hundreds of items. This stage is essential and also difficult. Whether the user is interested in the generated candidates gives an upper bound of the impression quality. How to draw candidates from the entire corpus weighing efficiency and effectiveness is a problem.

> 正如前面提到的，所提出的推荐模型旨在构建一个包含数百个商品的候选集合。这个阶段既至关重要又具有挑战性。用户是否对生成的候选商品感兴趣可以界定印象质量的上限。如何在考虑效率和有效性的情况下选取候选商品是一个问题。

![](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure1.png)

**Figure 1**: The system architecture of Taobao display advertising recommender system.

## 3 TREE-BASED DEEP MODEL

In this part, we first introduce the tree structure used in our tree- based model to give an overall conception. Secondly, we introduce hierarchical softmax [24] to show why its formulation is not suitable for recommendation. After that, we give a novel max-heap like tree formulation and show how to train the tree-based model. Then, the deep neural network architecture is introduced. At last, we show how to construct and learn the tree used in the tree-based model.

> 在本部分中，我们首先介绍了树模型中使用的树结构，以便给出一个整体概念。其次，我们介绍了分层Softmax[24]，以说明为什么它的公式不适用于推荐系统。接下来，我们提出了一种新颖的类似于最大堆的树形表达方式，并展示了如何训练基于 tree-based 的模型。然后，介绍了深度神经网络的架构。最后，我们展示了如何构建和学习用于 Tree-based 模型的树结构。

### 3.1 Tree for Recommendation

A recommendation tree consists of a set of nodes $N$ , where 
$$
N = \{n_1, n_2, \cdots n_{|N|}\}
$$
 represents $|N|$ individual non-leaf or leaf nodes. Each node in $N$ except the root node has one parent and an arbitrary number of children. Specifically, each item $c_i$ in the corpus $C$ corresponds to one and only one leaf node in the tree, and those non-leaf nodes are coarse-grained concepts. Without loss of generality, we suppose that node $n_1$ is always the root node. An example tree is illustrated in the right bottom corner of Figure 2, in which each circle represents a node and the number of node is its index in tree. The tree has 8 leaf nodes in total, each of which corresponds to an item in the corpus. It’s worth mentioning that though the given example is a complete binary tree, we don’t impose complete and binary as restrictions on the type of the tree in our model.

> 一个推荐树由一组节点N组成，其中 
> $$
> N = \{n_1, n_2, \cdots n_{|N|}\}
> $$
> 代表着 $|N|$ 个独立的非叶子节点或叶子节点。除了根节点之外，$N$ 中的每个节点都有一个父节点和任意数量的子节点。具体来说，数据集 C 中的每个商品 $c_i$ 对应于树中的一个且仅有一个叶子节点，而那些非叶子节点则代表粗粒度的概念。为了方便起见，我们假设节点 $n_1$ 始终是根节点。图2 右下角展示了一个示例树，其中每个圆圈代表一个节点，节点上的数字表示它在树中的索引。该树总共有8个叶子节点，每个叶子节点对应于语料库中的一个商品。值得一提的是，尽管给定的示例是一个完全二叉树，但我们在模型中不将 “完全“和”二叉“作为树类型的限制条件。

### 3.2 Related Work

With the tree structure, we firstly introduce the related work hierarchical softmax to help understand its difference with our TDM. In hierarchical softmax, each leaf node $n$ in tree has its unique encoding from the root to the node. For example, if we encode 1 as choosing the left branch and 0 as choosing the right branch, $n_9$’s encoding in tree in Figure 2 is 110 and $n_{15}$’s encoding is 000. Denote $b_j (n)$ as the encoding of node $n$ in level $j$. In hierarchical softmax’s formulation, the next-word’s probability given the context is derived as
$$
P(n \mid \text { context })=\prod_{j=1}^w P\left(b=b_j(n) \mid l_j(n), \text { context }\right)
\\(1)
$$
where $w$ is the length of leaf node $n$’s encoding, and $l_j(n)$ is $n$’s ancestor node in level $j$.

> 使用树结构，我们首先介绍了与我们的推荐树模型(TDM)不同的相关工作—— Hierarchical Softmax，以帮助理解二者的区别。在分层Softmax中，树中的每个叶子节点 $n$ 都有从根节点到该节点的唯一编码。例如，如果我们将选择左分支编码为1，选择右分支编码为0，在图2中树中 $n_9$ 的编码为 110，$n_{15}$ 的编码为 000。记$b_j(n)$ 为第 $j$ 级中节点 $n$ 的编码。在 Hierarchical Softmax的公式中，给定上下文，下一个词在树中的概率可以表示为：
> $$
> P(n \mid \text { context })=\prod_{j=1}^w P\left(b=b_j(n) \mid l_j(n), \text { context }\right)
> \\(1)
> $$
> 其中 $w$ 是叶子节点 $n$ 的编码长度，$l_j(n)$ 是节点 n 在第 j 级上的祖先节点。

In such a way, hierarchical softmax solves the probability calculation problem by avoiding the normalization term (each word in the corpus needs to be traversed) in conventional softmax. However, to find the most possible leaf, the model still has to traverse the entire corpus. Traversing each level’s most possible node top-down along the tree path can not guarantee to successfully retrieve the optimal leaf. Therefore, hierarchical softmax’s formulation is not suitable for large-scale retrieval problem. In addition, according to Equation 1, each non-leaf node in tree is trained as a binary classifier to discriminate between its two children nodes. But if two nodes are neighbors in the tree, they are probably to be similar. In recommendation scenario, it’s likely that user is interested in both two children. Hierarchical softmax’s model focuses on distinguishing optimal and suboptimal choices, which may lose the capability of discriminating from a global view. If greedy beam search is used to retrieve those most possible leaf nodes, once bad decisions are made in upper levels of the tree, the model may fail to find relatively better results among those low quality candidates in lower levels. YouTube’s work [7] also reports that they have tried hierarchical softmax to learn user and item embeddings, while it performs worse than sampled-softmax [16] manner.

Given that hierarchical softmax’s formulation is not suitable for large-scale recommendation, we propose a new tree model formu- lation in the following section.

> 通过避免传统Softmax中的归一化项（需要遍历语料库中的每个词），Hierarchical Softmax解决了概率计算问题。然而，为了找到最可能的叶子节点，模型仍然需要遍历整个语料库。沿着树路径自顶向下遍历每个层次上最可能的节点并不能保证成功地找到最优的叶子节点。因此，Hierarchical Softmax 的公式对于大规模检索问题并不合适。此外，根据 公式1，树中的每个非叶子节点被训练为一个二分类器，以区分其两个子节点。但是，如果两个节点在树中相邻，它们可能是相似的。在推荐场景中，用户可能对这两个子节点都感兴趣。Hierarchical Softmax的模型注重区分最优和次优选择，这可能会失去从全局视角区分的能力。如果使用贪心的 beam search 来检索那些最可能的叶子节点，一旦在树的较高层次做出了错误的决策，模型可能无法在较低层次的低质量候选集中找到相对更好的结果。YouTube的研究[7]也报告了他们尝试使用Hierarchical Softmax学习用户和商品嵌入，但效果不如采用采样Softmax [16]的方式。
>
> 鉴于 Hierarchical Softmax的公式不适用于大规模推荐，我们在接下来的部分提出了一种新的树模型公式。

### 3.3 Tree-based Model Formulation

To address the problem of efficient top-k retrieval of most preferred items, we propose a max-heap like tree probability formulation. Max-heap like tree is a tree structure where every non-leaf node $n$ in level $j$ satisfies the following equation for each user $u$:
$$
\begin{aligned}
& P^{(j)}(n \mid u)=\frac{\overset{max}{n_c \in\{n \text { 's children nodes in level } j+1}\}}{\alpha^{(j)}}^{p(j+1)(n_c|u)} \\
&
\end{aligned}
\\
(2)
$$
where $P^{(j)}(n|u)$ is the ground truth probability that user $u$ is interested in $n$. $\alpha^{(j)}$ is the layer-specific normalization term of level $j$ to ensure that the probability sum in the level equals to 1. Equation 2 says that a parent node’s ground truth preference equals to the maximum preference of its children nodes, divided by the normalization term. Note that we slightly abuse the notation and let $u$ denote a specific user state. In other words, a specific user state $u$ may transfer to another state $u'$  once the user has a new behavior.

> 为了解决高效检索用户最喜欢的商品的问题，我们提出了一种类似于最大堆的树形概率结构。类似于最大堆的树是一种树结构，在第 $j$ 层中的每个非叶子节点 $n$ 对于每个用户 $u$ 满足以下方程：
> $$
> \begin{aligned}
> & P^{(j)}(n \mid u)=\frac{\overset{max}{n_c \in\{n \text { 's children nodes in level } j+1}\}}{\alpha^{(j)}}^{p(j+1)(n_c|u)} \\
> &
> \end{aligned}
> \\
> (2)
> $$
> 其中 $P^{(j)}(n∣u)$ 是用户 $u$ 对 $n$ 感兴趣的真实概率。$\alpha(j)$是第 $j$ 层的归一化项，用于确保该层上的概率之和等于1。公式2 表示父节点的真实偏好等于其子节点中的最大偏好除以归一化项。请注意，我们稍微滥用了符号，将 $u$ 表示为特定的用户状态。换句话说，一旦用户有新的行为，特定的用户状态 $u$ 可能转变为另一个状态 $u′$。

The goal is to find $k$ leaf nodes with largest preference probabilities. Suppose that we have each node $n$‘s ground truth $P^{(j)}(n|u)$ in the tree, we can retrieve $k$ nodes with largest preference probabilities layer-wise, and only those children nodes of each level’s top $k$ need to be explored. In this way, top $k$ leaf nodes can be ultimately retrieved. Actually, we don’t need to know each tree node’s exact ground truth probability in the above retrieval process. What we need is the order of the probabilities in each level to help find the top $k$ nodes in the level. Based on this observation, we use user’s implicit feedback data and neural network to train each level’s discriminator that can tell the order of preference probabilities.

> 目标是找到具有最大偏好概率的 $k$ 个叶节点。假设我们在树中拥有每个节点 $n$ 的真实情况 $P^{(j)}(n|u)$ ，我们可以逐层检索具有最大偏好概率的 $k$ 个节点，并且只需要探索每一级的前 $k$ 个子节点。通过这种方式，可以最终获取到前 $k$ 个叶节点。实际上，在上述检索过程中，我们不需要知道每个树节点的准确真实概率。我们需要的是每个层级中概率的顺序，以帮助找到该层级中的前 $k$ 个节点。基于这一观察，我们使用用户的隐式反馈数据和神经网络来训练每个层级的判别器，以确定偏好概率的顺序。

Suppose that user $u$ has an interaction with leaf node $n_d$ , i.e., $n_d$ is a positive sample node for $u$. It means an order $P^{(m)}(n_d |u) > P^{(m)}(n_t |u)$, where $m$ is the level of leaves and $n_t$ is any other leaf node. In any level $j$, denote $l_j (n_d )$ as $n_d$ ’s ancestor in level $j$. According to the formulation of tree in Equation 2, we can derive that $P^{(j)}(l_j(n_d)|u) > P^{(j)}(n_q|u)$, where $n_q$ is any node in level $j$ except $l_j (n_d )$. In basis of the above analysis, we can use negative sampling [23] to train each level’s order discriminator. In detail, leaf node that have interaction with $u$, and its ancestor nodes constitute the set of positive samples in each level for $u$. And randomly selected nodes except positive ones in each level constitute the set of negative samples. Those green and red nodes in Figure 2 give examples for sampling. Suppose that given a user and its state, the target node is $n_{13}$. Then, $n_{13}$’s ancestors are positive samples, and those randomly sampled red nodes in each level are negative samples. These samples are then fed into binary probability models to get levels’ order discriminators. We use one global deep neural network binary model with different input for all levels’ order discriminators. Arbitrary advanced neural network can be adopted to improve model capability.

> 假设用户 $u$ 与叶节点 $n_d$ 进行了交互，即 $n_d$ 是 $u$ 的一个正样本节点。这意味着存在一个顺序 $P^{(m)}(n_d |u) > P^{(m)}(n_t |u)$，其中 $m$ 是叶节点的层级，$n_t$ 是任意其他叶节点。在任意层级 $j$ 中，将$l_j(n_d)$表示为 $j$ 层中 $n_d$ 的祖先 。根据 公式2 中树的构造，我们可以推导出 $P^{(j)}(l_j(n_d)|u) > P^{(j)}(n_q|u)$，其中 $n_q$ 是层级 $j$ 中除 $l_j{(n_d)}$ 之外的任意节点。基于以上分析，我们可以使用负采样[23]来训练每个层级的 order-discriminator(顺序判别器)。具体来说，与 $u$ 有交互的叶节点及其祖先节点构成了每个层级对于 $u$ 的正样本集合。而在每个层级中，随机选择除了正样本以外的节点作为负样本集合。图2 中的绿色和红色节点给出了采样示例。假设给定一个用户和其状态，目标节点是 $n_{13}$。那么，$n_{13}$ 的祖先节点是正样本，而在每个层级中随机选择的红色节点则是负样本。这些样本随后被送到二分类模型中，以获得各层级的顺序判别器。我们使用一个全局深度神经网络二分类模型，为所有层级的顺序判别器提供不同的输入。可以采用任意先进的神经网络来提高模型的能力。

Denote $\mathcal{Y}^+_u$ and $\mathcal{Y}^−_u$ as the set of positive and negative samples for $u$. The likelihood function is then derived as:
$$
\prod_u\left(\prod_{n \in \mathcal{Y}^+_u} P\left(\hat{y}_u(n)=1 \mid n, u\right) \prod_{n \in \mathcal{Y}^-_{u}} P\left(\hat{y}_u(n)=0 \mid n, u\right)\right) \text {, }
\\
(3)
$$
where $\hat{y}_u(n)$ is the predicted label of node $n$ given $u$. $P\left(\hat{y}_u(n) \mid n, u\right)$ is the output of binary probability model, taking user state $u$ and the sampled node $n$ as input. The corresponding loss function
$$
-\sum_u \sum_{n \in \mathcal{Y}_u^{+} \cup \mathcal{Y}_{\bar{u}}^{-}}
\\
y_u(n) \log P\left(\hat{y}_u(n)=1 \mid n, u\right)+\left(1-y_u(n)\right) \log P\left(\hat{y}_u(n)=0 \mid n, u\right)
\\
(4)
$$
where $y_u ^{(n)}$ is the ground truth label of node $n$ given $u$. Details about how to train the model according to the loss function are in Section 3.4.

> 将 $\mathcal{Y}^+_u$ 、 $\mathcal{Y}^−_u$ 表示为用户 $u$ 的正负样本集合. 似然函数如下:
> $$
> \prod_u\left(\prod_{n \in \mathcal{Y}^+_u} P\left(\hat{y}_u(n)=1 \mid n, u\right) \prod_{n \in \mathcal{Y}^-_{u}} P\left(\hat{y}_u(n)=0 \mid n, u\right)\right) \text {, }
> \\
> (3)
> $$
> 其中 $\hat{y}_u(n)$ 是给定 $u$ 节点 $n$ 的预测标签. $P\left(\hat{y}_u(n) \mid n, u\right)$ 是二分类模型的输出，以用户状态 $u$ 和采样节点 $n$ 作为输入. 其损失函数可以表示为：
> $$
> -\sum_u \sum_{n \in \mathcal{Y}_u^{+} \cup \mathcal{Y}_{\bar{u}}^{-}}
> \\
> y_u(n) \log P\left(\hat{y}_u(n)=1 \mid n, u\right)+\left(1-y_u(n)\right) \log P\left(\hat{y}_u(n)=0 \mid n, u\right)
> \\
> (4)
> $$
> 其中 $y_u^{(n)}$ 是给定用户 $u$ 的节点 $n$ 的真实标签。有关如何根据损失函数训练模型的详细信息，请参考第3.4节。

Note that the proposed sampling method is quite different from the underlying one in hierarchical softmax. Compared to the method used in hierarchical softmax which leads the model to distinguish optimal and suboptimal results, we randomly select negative samples in the same level for each positive node. Such method makes each level’s discriminator be an intralevel global one. Each level’s global discriminator can make precise decisions independently, with- out depending on the goodness of upper levels’ decisions. The global discriminating capability is very important for hierarchical recommendation approaches. It ensures that even if the model makes bad decision and low quality nodes leak into the candidate set in an upper-level, those relatively better nodes rather than very bad ones can be chosen by the model in the following levels.

> 需要注意的是，提出的采样方法与 hierarchical softmax 中的底层采样方法有很大的区别。与 hierarchical softmax 中的方法相比，该方法使模型能够区分最优和次优结果，我们在每个正样本节点的同一层级中随机选择负样本。这种方法使得每个层级的判别器成为一个层内全局的判别器。每个层级的全局判别器可以独立地做出精确的决策，而不依赖于上层决策的好坏（译注：树的父子节点之间其实本身就是一种强联系，独立地做出精确的决策这种描述很有问题。）。这种全局判别能力对于分层推荐方法非常重要。它确保即使在上层中模型做出了错误的决策并且低质量的节点泄漏到候选集中，模型在后续层级中仍然可以选择相对较好的节点，而不是非常差的节点。

Given a recommendation tree and an optimized model, the detailed hierarchical prediction algorithm is described in Algorithm 1. The retrieval process is layer-wise and top-down. Suppose that the desired candidate item number is $k$. For corpus $C$ with size $|C|$, traversing at most $2 \times k \times log|C|$ nodes can get the final recommendation set in a complete binary tree. The number of nodes need to be traversed is in a logarithmic relation w.r.t. corpus size, which makes advanced binary probability models possible to be employed.

> 给定一个推荐树和一个优化的模型，算法1描述了详细的分层预测算法。检索过程按层次和自顶向下进行。假设期望的候选项数量为 $k$。对于大小为$|C|$ 的数据集 $C$，最多遍历 $2 \times k \times log|C|$ 个节点即可在完全二叉树中得到最终的推荐集合。需要遍历的节点数量与语数据集大小呈对数关系，这使得可以使用先进的二分类概率模型来进行推荐。

Our proposed TDM method not only reduces the amount of calculation when making prediction, it also has potential to improve recommendation quality compared with brute-force search in all leaf nodes. Without the tree, training a model to find optimal items directly is a difficult problem because of the corpus size. Employing the tree hierarchy, a large-scale recommendation problem is divided into many smaller problems. There only exist a few nodes in high levels of the tree, thus the discrimination problem is easier. And decisions made by high levels refine the candidate set, which may help lower levels make better judgments. Experimental results in Section 5.4 will show that the proposed hierarchical retrieval approach performs better than direct brute-force search.

> 我们提出的 TDM 方法不仅减少了在进行预测时的计算量，而且与在所有叶节点上进行暴力搜索相比，还有改进推荐质量的潜力。没有树结构，直接训练一个模型来找到最优项是一个困难的问题，因为涉及到的数据集的规模很大。通过使用树形层次结构，一个大规模的推荐问题被划分为许多较小的子问题。树的高层级只存在少数节点，因此判别问题更容易解决。高层级的决策可以优化候选集合，从而帮助低层级做出更好的判断。第5.4节中的实验结果将显示，所提出的分层检索方法优于直接的暴力搜索方法。简而言之，我们的TDM方法降低了预测时的计算量，并通过树形层次结构将大规模的推荐问题转化为许多小问题。这种方法不仅减少了复杂度，还能提高推荐质量。实验结果表明，与直接暴力搜索相比，我们提出的分层检索方法表现更好。
>

### 3.4 The Deep Model

In the following part, we introduce the deep model we use. The entire model is illustrated in Figure 2. Inspired by the click-through rate prediction work [34], we learn low dimensional embeddings for each node in the tree, and use attention module to softly searching for related behaviors for better user representation. To exploit user behavior that contains timestamp information, we design the block-wise input layer to distinguish behaviors that lie in different time windows. The historical behaviors can be divided into different time windows along the timeline, and item embeddings in each time window is weighted averaged. Attention module and the following network greatly strengthen the model capability, and also make user’s preferences over candidate items can not be regulated to inner product form.

The embeddings of tree nodes and the tree structure itself are also parts of the model. To minimize Loss 4, the sampled nodes and the corresponding features are used to train the network. Note that we only illustrate the usage of user behavior feature in Figure 2 for briefness, while other features like user profile or contextual feature can be used with no obstacles in practice.

> 在接下来的部分，我们介绍了我们使用的深度模型。整个模型如图2所示。受点击率预测工作[34]的启发，我们为树中的每个节点学习低维度 embedding，并使用 attentionm模块 来软搜索相关行为以更好地表示用户。为了利用包含时间戳信息的用户行为，我们设计了分块输入层来区分位于不同时间窗口内的行为。历史行为可以沿时间轴划分为不同的时间窗口，每个时间窗口中的 item embedding 进行加权平均。注意力模块和后续的网络极大地增强了模型的能力，并且使得用户对 item 的偏好不能被限制为内积形式。
>
> 树节点的 embedding 和树结构本身也是模型的一部分。为了最小化 Loss 4，我们使用采样的节点和相应的特征来训练网络。需要注意的是，在 图2 中我们只简要展示了用户行为特征的使用，而实际上其他特征，如用户画像或上下文特征，也可以被无障碍地使用。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Alg1.png)

> 注：就是一个剪枝 bfs 的检索过程，bfs 剪枝检索，每一层取 topk，直到取完，计算量级较小，但可以遍历上百万甚至上亿节点的树，如果树是完全二叉树，一亿个叶子结点的话，大约有27层，其实计算量也不少了，但在实际场景中，比如用内容理解生成的类别作为 item 的类别，层数会急剧减小，10层就顶天了。

### 3.5 Tree Construction and Learning

The recommendation tree is a fundamental part of the tree-based deep recommendation model. Unlike multi-class and multi-label classification works [26, 32] where tree is used to partition samples or labels, our recommendation tree indexes items for retrieval. In hierarchical softmax [24], the word hierarchy is built according to expert knowledge from WordNet [21]. In the scenario of recommendation, not every corpus can provide specific expert knowledge. An intuitive alternation is to construct the tree using hierarchical clustering methods in basis of item concurrence or similarity drawn from the dataset. But the clustered tree may be quite imbalanced, which is detrimental for training and retrieval. Given pairwise item similarity, algorithm in [2] gives a way to split items into subsets recursively by spectral clustering [25]. However, spectral clustering is not scalable enough (cubic time complexity w.r.t. corpus size) for large-scale corpus. In this section, we focus on reasonable and feasible tree construction and learning approaches.

> 推荐树是基于树的深度推荐模型的基本组成部分。与多分类和多标签分类工作不同，在那些工作中树被用于对样本或标签进行分割，我们的推荐树用于索引检索项。在 hierarchical softmax 中，词的层次结构是根据WordNet的专业知识建立起来的。然而，在推荐场景中，并非每个语料库都能提供特定的专业知识。一种直观的替代方法是使用从数据集中得出的 item 共现性或相似性来构建树形结构。但是，聚类后的树可能会非常不平衡，这对训练和检索是有害的。根据 item 的两两相似度，[2]中的算法通过谱聚类将 item 递归地划分为子集。然而，谱聚类对于大规模语料库来说不够可扩展（与语料库大小相比具有立方时间复杂度）。在本节中，我们将重点关注合理和可行的树构建和学习方法。

**Tree initialization**. Since we suppose the tree to represent user interests’ hierarchical information, it’s natural to build the tree in a way that similar items are organized in close positions. Given that category information is extensive available in many domains, we intuitively come up with a method leveraging item’s category information to build the initial tree. Without loss of generality, we take binary tree as an example in this section.

Firstly, we sort all categories randomly, and place items belonging to the same category together in an intracategory random order. If an item belongs to more than one category, the item is assigned to a random one for uniqueness. In such way, we can get a list of ranked items.Secondly, those ranked items are halved to two equal parts recursively until the current set contains only one item, which could construct a near-complete binary tree top-down. The above kind of category-based initialization can get better hierarchy and results in our experiments than a complete random tree.

> **树的初始化**
>
> 由于我们假设树代表用户兴趣的层次信息，自然地，我们希望以相似 item 相互靠近的方式构建树。鉴于在许多领域中类别信息是广泛可用的，我们直观地提出一种利用 item 的类别信息来构建初始树的方法。在本节中，我们以二叉树为例进行说明。首先，我们随机对所有类别进行排序，并将属于同一类别的 item 按照类内随机顺序放在一起。如果一个 item 属于多个类别，则在这些类别中随机选择一个类别以确保唯一性。通过这种方式，我们可以得到一个经过排序的 item 列表。 其次，我们对这些排序的 item 进行递归的二分操作，直到当前集合只包含一个 item ，这样可以从上往下构建一个接近完全二叉树。与完全随机树相比，基于类别的初始化方法在实验中可以得到更好的层次结构和结果。
>
> 简而言之，我们通过利用 item 的类别信息来构建初始树。首先，将属于同一类别的 item 放在一起，并按类内随机顺序排列。然后，通过递归的二分操作构建接近完全二叉树。这种基于类别的初始化方法在实验中比完全随机树表现更好。

**Tree learning**. As a part of the model, each leaf node’s embedding can be learnt after model training. Then we use the learnt leaf nodes’ embedding vectors to cluster a new tree. Considering the corpus size, we use k-means clustering algorithm for its good scalability. At each step, items are clustered into two subsets according to their embedding vectors. Note that the two subsets are adjusted to equal for a more balanced tree. The recursion stops when only one item is left, and a binary tree could be constructed in such a top-down way. In our experiments, it takes about an hour to construct such a cluster tree when the corpus size is about 4 millions, using a single machine. Experimental results in Section 5 will show the effectiveness of the given tree learning algorithm.

The deep model and tree structure are learnt jointly in an alternative way: 1) Construct an initial tree and train the model till converging; 2) Learn to get a new tree structure in basis of trained leaf nodes’ embeddings; 3) Train the model again with the learnt new tree structure.

> **树的学习**
>
> 作为模型的一部分，在模型训练之后可以学习每个叶节点的 embedding。然后，我们使用学到的叶节点 embedding 向量来聚类一个新的树。考虑到语料库的规模，我们使用 k-means 聚类算法因为它具有良好的可扩展性。在每一步中，根据 item 的 embedding 向量将其聚类为两个子集。需要注意的是，为了得到更平衡的树，这两个子集会被调整为相等。当只剩下一个 item 时，递归停止，从而以自顶向下的方式构建二叉树。在我们的实验中，当语料库大小约为400万时，使用单台机器构建这样一个聚类树大约需要一个小时。第5节的实验结果将展示给定的树学习算法的有效性。 
>
> 深度模型和树结构是以交替的方式进行联合学习：
>
> 1. 构建初始树并训练模型直至收敛；
> 2. 基于训练后的叶节点 embedding 学习获得一个新的树结构；
> 3. 使用学到的新树结构重新训练模型；

![Fig2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure2.png)

**Figure 2: The tree-based deep model architecture. User behaviors are divided into different time windows according to the timestamp. In each time window, item embeddings are weighted averaged, and the weights come from activation units. Each time window’s output along with the candidate node’s embedding are concatenated as the neural network input. After three fully connected layers with PReLU [33] activation and batch normalization [14], a binary softmax is used to yield the probabil- ity whether the user is interested in the candidate node. Each item and its corresponding leaf node share the same embedding. All embeddings are randomly initialized.**

> **图2：基于树的深度模型架构。根据时间戳，用户行为被划分为不同的时间窗口。在每个时间窗口中，item embedding 进行加权平均，item 来自激活单元。每个时间窗口的输出以及候选节点的 embedding 被串联作为神经网络输入。通过三个全连接层（使用PReLU [33]激活函数和批归一化[14]）之后，使用 二分类 softmax 产生用户是否对候选节点感兴趣的概率。每个 item 及其对应的叶节点共享相同的 embedding。所有的 embedding 都是随机初始化的。**

## 4 ONLINE SERVING

Figure 3 illustrates the online serving system of the proposed method. Input feature assembling and item retrieval are split into two asynchronous stages. Each user behavior including click, purchase and adding item into shopping cart will strike the real-time feature server to assemble new input features. And once receiving page view request, the user targeting server will use the pre-assembled features to retrieve candidates from the tree. As described in Algorithm 1, the retrieval is layer-wise and the trained neural network is used to calculate the probability that whether a node is preferred given the input features.

> 图3 所示是所提方法的在线服务系统架构。输入的特征组装和 item 检索被分两个并发阶段。每个用户行为，包括点击、购买和将 item 添加到购物车中，都会触发实时特征服务器来组装新的输入特征。一旦接收到页面浏览请求，用户定向服务器将使用预充的特征从树中检索候选项。如算法1所描述的，检索是逐层进行的，并且使用训练好的神经网络来计算给定输入特征时节点是否首选的概率。

![Fig3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure3.png)

**Figure 3: The online serving system of the tree-based model, where user feature is assembled asynchronously.**

> **Figure 3: 基于树模型的在线服务系统，其中用户特征异步组装。**

## 5 EXPERIMENTAL STUDY

We study the performance of the proposed tree-based model in this section. Experimental results in MovieLens-20M [12] and Taobao advertising dataset called UserBehavior are presented. In the exper- iments, we compare the proposed method to other existing methods to show the effectiveness of the model, and empirical study results show how the tree-based model and tree learning algorithm work.

> 在本节中，我们对提出的基于树的模型进行了性能研究。我们展示了在MovieLens-20M [12]和称为UserBehavior的淘宝广告数据集中的实验结果。在实验中，我们将提出的方法与其他现有方法进行比较，以展示该模型的有效性，并通过实证研究结果展示基于树的模型和树学习算法的工作原理。

### 5.1 Datasets

The experiments are conducted in two large-scale real-world datasets with timestamps: 1) users’ movie viewing data from MovieLens [12]; 2) a user-item behavior dataset from Taobao called UserBe- havior. In more details:

> 实验在两个具有时间戳的大规模真实世界数据集上进行：1）来自MovieLens [12]的用户电影观看数据；2）来自淘宝的一个称为UserBehavior的用户-项目行为数据集。以下是更详细的说明：

**MovieLens-20M**: It contains user-movie ratings with timestamps in this dataset. As we deal with implicit feedback problem, the ratings are binarized by keeping the ratings of four or higher, which is a common way in other works [8, 20]. Besides, only the users who have watched at least 10 movies are kept. To create training, validation and testing sets, we randomly sample 1, 000 users as testing set and another 1, 000 users as validation set, while the rest users constitute the training set [8]. For validation and testing sets, the first half of user-movie views along the timeline is regarded as known behaviors to predict the latter half. 

**UserBehavior1**: This dataset is a subset of Taobao user behav- ior data. We randomly select about 1 million users who have be- haviors including click, purchase, adding item to shopping cart and item favoring during November 25 to December 03, 2017. The data is organized in a very similar form to MovieLens-20M, i.e., a user- item behavior consists of user ID, item ID, item’s category ID, behavior type and timestamp. As we do in MovieLens-20M, only the users who have at least 10 behaviors are kept. 10, 000 users are ran- domly selected as testing set and another randomly selected 10, 000 users are validation set. Items’ categories are from the bottom level of Taobao’s current commodity taxonomy. Table 1 summarizes the major dimensions of the above two datasets after preprocessing. 

> **MovieLens-20M数据集**：该数据集包含了用户对电影的评分和时间戳。由于我们处理的是隐性反馈问题，我们将评分进行二值化处理，保留4分及以上的评分，这是其他研究中常用的方式[8, 20]。此外，只保留至少观看了10部电影的用户。为了创建训练、验证和测试数据集，我们随机选择1,000个用户作为测试集，另外1,000个用户作为验证集，而剩下的用户构成训练集[8]。对于验证集和测试集，用户在时间线上的前一半电影观看行为被视为已知行为，用来预测后一半的行为。
>
> **UserBehavior1**：该数据集是淘宝用户行为数据的一个子集。我们随机选择了大约100万名用户，在2017年11月25日至12月3日期间具有点击、购买、将商品添加到购物车和收藏商品等行为。数据的组织形式与MovieLens-20M非常相似，即 user-item 行为由用户ID、item-ID、item-cate-ID、行为类型和时间戳组成。与MovieLens-20M一样，只保留至少有10个行为的用户。其中10,000个用户被随机选择作为测试集，另外随机选择的10,000个用户作为验证集。item 的类别来自淘宝当前商品分类法的底层。表1总结了经过预处理后上述两个数据集的主要维度。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table1.png)

**Table 1: Dimensions of the two datasets after preprocessing. One record is a user-item pair that represents user feedback.**

### 5.2 Metrics and Comparison Methods

To evaluate the effectiveness of different methods, we use Precision@M, Recall@M and F-Measure@M metrics [20]. Derive the recalled set of items for a user $u$ as $P_u (|P_u| = M)$ and the user’s ground truth set as $G_u$ . Precision@M and Recall@M are
$$
\operatorname{Precision@M(u)}=\frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{M}, \operatorname{Recall@M(u)=} \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|},
$$
and F-Measure@M is
$$
\text { F-Measure@M }(u)=\frac{2 * \operatorname{Precision} @ M(u) * \text { Recall@M }(u)}{\operatorname{Precision} @ M(u)+\operatorname{Recall@M(u)}} \text {. }
$$
As we emphasize, recommendation results’ novelty is responsible for user experience. Existing work [4] gives several approaches to measure the novelty of recommended list of items. Following one of its definition, the Novelty@M is defined as
$$
\text { Novelty@M(u)= } \frac{\left|\mathcal{P}_u\right \\  \mathcal{S}_u \mid}{M}
$$
where $S_u$ is the set of items that have interactions with user $u$ before recommending. User average of the above four metrics in testing set are used to compare the following methods:

- **FM**[28]. FM is a framework for factorization tasks. We use the implementation of FM provided by xLearn2 project.
- **BPR-MF**[29]. We use its matrix factorization form for implicit feedback recommendation. Implementation of BPR-MF provided by [10] is used.
- **Item-CF**[31].Item-basedcollaborativefilteringisoneofthe most widely used personalized recommendation method in production with large-scale corpus [22]. It’s also one of the major candidate generation approaches in Taobao. We use the implementation of item-CF provided by Alibaba machine learning platform.
- **YouTube product-DNN**[7] is the deep recommendation approach proposed by YouTube. Sampled-softmax [16] is employed in training, and the inner product of user and item’s embeddings reflects the preference. We implement YouTube product-DNN in Alibaba deep learning platform with the same input features with our proposed model. Ex- act kNN search in inner product space is adopted in predic- tion.
- **TDM attention-DNN** (tree-based deep model using atten- tion network) is our proposed method in Figure 2. The tree is initialized in the way described in Section 3.5 and keeps unchanged during the experiments. The implementation is available in GitHub3.

> 为了评估不同方法的有效性，我们使用Precision@M、Recall@M和F-Measure@M指标[20]。对于用户$u$，我们将其推荐集合定义为$P_u (|P_u|=M)$，将用户的真实集合定义为$G_u$。Precision@M和Recall@M的计算公式如下：
> $$
> \text{Precision@M(u)} = \frac{|P_u \cap G_u|}{M}, \quad \text{Recall@M(u)} = \frac{|P_u \cap G_u|}{|G_u|} 
> $$
>  F-Measure@M的计算公式如下： 
> $$
> \text{F-Measure@M(u)} = \frac{2 \times \text{Precision@M(u)} \times \text{Recall@M(u)}}{\text{Precision@M(u)} + \text{Recall@M(u)}}
> $$
>  
>
> 另外，推荐结果的新颖性对用户体验至关重要。现有研究[4]提供了几种衡量推荐列表新颖性的方法。根据其中一种定义，新颖性指标Novelty@M定义如下： 
> $$
>  \text{Novelty@M(u)} = \frac{|P_u \setminus S_u|}{M}
> $$
> 其中 $S_u$ 是在推荐之前与用户 $u$ 有过交互的 item 集合。 测试集中上述四个指标的用户平均值被用来比较以下方法： 
>
> - **FM**[28]：FM是一种用于因子分解任务的框架。我们使用 XLearn2 项目提供的FM实现。
> - **BPR-MF**[29]：我们使用其针对隐式反馈推荐的矩阵分解形式。使用[10]提供的BPR-MF实现。
> - **Item-CF**[31]：基于物品的协同过滤是生产中最常用的个性化推荐方法之一，适用于大规模语料库[22]。它也是淘宝主要的候选集生成方法之一。我们使用阿里巴巴机器学习平台提供的 item-CF 实现。
> - **YouTube product-DNN**[7]：这是YouTube提出的深度推荐方法。在训练中使用了采样 softmax [16]，用户和项目嵌入的内积反映了偏好。我们在阿里巴巴深度学习平台上使用与我们提出的模型相同的输入特征实现了YouTube product-DNN。预测时采用内积空间中的精确kNN搜索。
> - **TDM attention-DNN**（基于树的深度模型使用注意力网络）是我们在图2中提出的方法。树的初始化方式如第3.5节所述，在实验过程中保持不变。该实现可在GitHub3上获得。

For FM, BPR-MF and item-CF, we tune several most important hyper-parameters based on the validation set, i.e., the number of factors and iterations in FM and BPR-MF, the number of neighbors in item-CF. FM and BPR-MF require that the users in testing or validation set also have feedback in training set. Therefore, we add the first half of user-item interactions along the timeline in testing and validation set into the training set in both datasets. For YouTube product-DNN and TDM attention-DNN, the node embeddings’ di- mension is set to 24, because a higher dimension doesn’t perform significantly better in our experiments. The hidden unit numbers of three fully connected layers are 128, 64 and 24 respectively. According to the timestamp, user behaviors are divided into 10 time windows. In YouTube product-DNN and TDM attention-DNN, for each implicit feedback we randomly select 100 negative samples in MovieLens-20M and 600 negative samples in UserBehavior. Note that the negative sample number of TDM is the sum of all levels. And we sample more negatives for levels near to leaf.

> 对于 FM、BPR-MF 和 item-CF，我们根据验证集调整了几个最重要的超参数，即 FM 和 BPR-MF 中的因子数量和迭代次数，以及 item-CF 中的邻居数量。FM和BPR-MF要求测试集或验证集中的用户在训练集中也有反馈。因此，在两个数据集中，我们将测试集和验证集上时间线上的前一半用户-项目交互行为添加到训练集中。对于YouTube product-DNN和TDM attention-DNN，节点嵌入的维度设置为24，因为在我们的实验中，更高的维度并没有表现出显著提升。三个全连接层的隐藏单元数量分别为128、64和24。根据时间戳，将用户行为分为10个时间窗口。在YouTube product-DNN和TDM attention-DNN中，对于每个隐式反馈，我们在MovieLens-20M中随机选择100个负样本，在UserBehavior中随机选择600个负样本。请注意，TDM的负样本数量是所有级别的总和。我们会对靠近叶子的级别进行更多的负采样。

### 5.3 Comparison Results

The comparison results of different methods are shown in Table 2 above the dash line. Each metric is the average across all the users in testing set, and the presented values are the average across five different runs for methods with variance.

> 不同方法的比较结果显示在虚线上方的 Table 2中。每个度量指标是在测试集中所有用户的平均值，呈现的数值是具有方差的方法在五次不同运行中的平均值。

First, the results indicate that the proposed TDM attention-DNN outperforms all the baselines significantly in both datasets on most of the metrics. Comparing to the second best YouTube product-DNN approach, TDM attention-DNN achieves 21.1% and 42.6% improvements on recall metric in two datasets respectively without filtering. This result proves the effectiveness of advanced neural network and hierarchical tree search adopted by TDM attention-DNN. Among the methods that model user preference over items in inner product form, YouTube product-DNN outperforms BPR- MF and FM because of the usage of neural network. The widely used item-CF method gets worst novelty results, since it has strong memories about what the user has already interacted.

> 首先，结果表明，在大多数度量指标上，提出的 TDM attention-DNN 在两个数据集上都显著优于所有基线方法。与排名第二的 YouTube product-DNN 方法相比，TDM attetion-DNN 在两个数据集上的召回率指标上分别实现了 21.1% 和 42.6% 的改进，而且没有进行过滤（已交互的item）。这个结果证明了 TDM attetion-DNN 采用的先进神经网络和分层树搜索的有效性。在将 user-item 的偏好建模为内积形式的方法中，由于使用了神经网络，YouTube product-DNN 优于 BPR-MF 和 FM。而广泛使用的 item协同过滤（item-CF）方法则得到了最差的新颖性结果，因为它对用户已经交互过的物品有很强的记忆。

To improve the novelty, a common way in practice is to filter those interacted items in recommendation set [8, 20], i.e., only those novel items could be ultimately recommended. Thus, it’s more important to compare accuracy in a complete novel result set. In this experiment, the result set size will be complemented to required number $M$ if its size is smaller than $M$ after filtering. The bottom half of Table 2 shows that TDM attention-DNN outperforms all baselines in large margin as well after filtering interacted items.

> 为了提高新颖性，在实践中常见的一种方法是在推荐集中过滤那些已经交互过的 item[8, 20]，即只有那些新颖的 item 最终才能被推荐（只存在于ucf，其实没什么卵用）。因此，在一个完整的新颖结果集中比较准确性更为重要。在这个实验中，如果过滤后的结果集大小小于 $M$，将会补充到需要的数量 $M$。Table2 的下半部分显示，在过滤掉已交互 item 后，TDM attention-DNN 在各项指标上都显著优于所有基线方法。

To further evaluate the exploration ability of different methods, we do experiments by excluding those interacted categories from recommendation results. Results of each method are also complemented to satisfy the size requirement. Indeed, category-level novelty is currently the most important novelty metric in Taobao recommender system, as we want to reduce the amount of recommendations similar to user’s interacted items. Since MovieLens-20M has only 20 categories in total, these experiments are only conducted in UserBehavior dataset and results are shown in Table 3. Take the recall metric for example. We can observe that item- CF’s recall is only 1.06%, because its recommendation results can hardly jump out of user’s historical behaviors. YouTube product-DNN gets much better results compared to item-CF, since it can explore user’s potential interests from the entire corpus. The proposed TDM attention-DNN performs 34.3% better in recall than YouTube’s inner product manner. Such huge improvement is very meaningful for recommender systems, and it proves that more advanced model is an enormous difference for recommendation problem.

> 为了进一步评估不同方法的探索能力，我们通过排除推荐结果中与用户交互的类别来进行实验。每种方法的结果也被补充以满足大小要求。事实上，目前淘宝推荐系统中最重要的新颖度度量是基于类别的新颖度，因为我们希望减少与用户交互 item 相似的推荐。由于 MovieLens-20M 总共只有 20 个类别，这些实验仅在UserBehavior 数据集中进行，并且结果显示在表 3 中。以召回率指标为例，我们可以观察到 item 协同过滤（item-CF）的召回率仅为1.06%，因为其推荐结果很难跳出用户的历史行为。与 item-CF 相比，YouTube product-DNN的结果要好得多，因为它可以从整个语料库中探索用户的潜在兴趣。所提出的 TDM attention-DNN在召回率方面比YouTube的内积方式改进了34.3%。这样巨大的改进对于推荐系统来说非常重要，它证明了更先进的模型在推荐问题中有着巨大的差异。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table2.png)

**Table 2: The comparison results of different methods in MovieLens-20M and UserBehavior datasets. According to the different corpussize, metrics are evaluated @10 in MovieLens-20 and @200 in UserBehavior. In experiments of filtering interacted items, the recommendation results and ground truth only contain items that the user has not yet interacted with before.**

> **表2：MovieLens-20M 和 UserBehavior 数据集中不同方法的比较结果。根据不同的数据集大小，使用@10 对 MovieLens-20 进行评估，而对 UserBehavior 使用 @200 进行评估。在过滤已交互 item 的实验中，推荐结果和真实情况只包含用户之前尚未交互过的 item。**

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table3.png)

**Table 3: Results in UserBehavior dataset. Items belong to interacted categories are excluded from recommendation results and ground truth.**

> **表3：UserBehavior数据集中的结果。属于已交互类别的 item 被排除在推荐结果和真实情况之外。**

### 5.4 Empirical Analysis

Variants of TDM. To comprehend the proposed TDM method itself, we derive and evaluate several variants of TDM:

- TDM product-DNN. To find out whether advanced neural network can benefit the results in TDM, we test the variant TDM product-DNN. TDM product-DNN uses the same inner product manner as YouTube product-DNN. Specifically, the attention module in Figure 2 is removed, and the node embedding term is also removed from the network input. The inner product of node embedding and the third fully connected layer’s output (without PReLU and BN) along with a sigmoid activation constitute the new binary classifier. 
- TDM DNN. To further verify the improvements brought by attention module in TDM attention-DNN, we test the variant TDM DNN that only removes the activation unit, i.e., all items’ weights are 1.0 in Figure 2. 
- TDM attention-DNN-HS. As mentioned in Section 3, hierarchical softmax (HS) method [24] is not suitable for recommendation. We test the TDM attention-DNN-HS variant, i.e., use positive nodes’ neighbors as negative samples instead of randomly selected ones. Correspondingly, in retrieval of Algorithm 1, the ranking indicator changes from a single node’s $P\left(\hat{y}_u(n)=1 \mid n, u\right)$ to $\prod_{n^{\prime} \in \text { n's ancestors }} P\left(\hat{y}_u\left(n^{\prime}\right)=\right.\left.1 \mid n^{\prime}, u\right)$ Attention-DNN is used as the network structure.

> TDM的变体。为了理解所提出的TDM方法本身，我们推导并评估了几种TDM的变体： 
>
> - TDM product-DNN。为了确定先进的神经网络是否能够改善TDM的结果，我们测试了TDM product-DNN这个变体。TDM product-DNN使用与YouTube product-DNN相同的内积方式。具体来说，Figure2 中的 attention 模块被移除，并且 node embedding item 也从网络输入中移除。节点 embedding 和第三个全连接层输出（不包括PReLU和BN）之间的内积以及一个sigmoid激活函数构成了新的二分类器。 
> - TDM DNN。为了进一步验证TDM attention-DNN中注意力模块所带来的改进，我们测试了只移除激活单元的TDM DNN变体，即在图2中所有 item 的权重都是1.0。
> - TDM attention-DNN-HS。如第3节所述，hierarchical softmax（HS）方法[24]不适用于推荐任务。我们测试了 TDM attention-DNN-HS 这个变体，即将正样本节点的邻居作为负样本，而不是随机选择的样本。对应地，在算法1的检索中，排名指标由单个节点的 $P\left(\hat{y}_u(n)=1 \mid n, u\right)$ 更改为 $\prod_{n^{\prime} \in \text { n's ancestors }} P\left(\hat{y}_u\left(n^{\prime}\right)=\right.\left.1 \mid n^{\prime}, u\right)$。网络结构仍然使用 attention-DNN。

The experimental results of the above variants in both datasets are shown in Table 2 under the dash line. Comparing TDM attentionDNN to TDM DNN, the near 10% recall improvement in UserBehavior dataset indicates that the attention module takes impressive efforts. TDM product-DNN performs worse than TDM DNN and TDM attention-DNN, since the inner product manner is much less powerful than the neural network interaction form. These results prove that introducing advanced models in TDM can significantly improve the recommendation performance. Note that TDM attention-DNN-HS gets much worse results compared to TDM attentionDNN, since hierarchical softmax’s formulation doesn’t fit for recommendation problem

> 上述变体在两个数据集上的实验结果如 Table2 中的虚线下所示。将 TDM attentionDNN 与 TDM DNN 进行比较，在 UserBehavior 数据集中近 10% 的召回率改进表明，注意力模块起到了显著的作用。TDM product-DNN 的性能不如 TDM DNN 和 TDM attention-DNN，因为内积方式远没有神经网络交互形式强大。这些结果证明，引入先进的模型可以显著提高推荐性能。需要注意的是，与 TDM attentionDNN 相比，TDM attention-DNN-HS 的结果要差得多，因为 hierarchical softmax 的公式不适用于推荐问题。

**Role of the tree.** Tree is the key component of the proposed TDM method. It not only acts as an index used in retrieval, but also models the corpus in coarse-to-fine hierarchy. Section 3.3 mentioned that directly making fine-grained recommendation is more difficult than a hierarchical way. We conduct experiments to prove the point of view. Figure 4 illustrates the layer-wise Recall@200 of hierarchical tree search (Algorithm 1) and brute-force search (traverse all nodes in the corresponding level). The experiments are conducted in UserBehavior dataset with TDM product-DNN model, because it’s the only variant that is possible to employ brute- force search. Brute-force search slightly outperforms tree search in high levels (level 8, 9), since the node numbers there are small. Once the node number in a level grows, tree search gets better re- call results compared to brute-force search, because the tree search can exclude those low quality results in high levels, which reduces the difficulty of the problems in low levels. This result indicates that the hierarchy information contained in the tree structure can help improve recommendation preciseness.

> **树的作用。** 树是所提出的TDM方法的关键组成部分。它不仅作为检索中使用的索引，还以粗到细的层次结构对数据集进行建模。第3.3节提到，直接进行细粒度的推荐比采用分层方式更困难。我们进行了实验证明这个观点。Figure 4 展示了分层树搜索（算法1）和暴力搜索（遍历相应级别的所有节点）的逐层召回率@200。在 UserBehavior 数据集上使用 TDM product-DNN 模型进行了实验，因为它是唯一一个可以使用暴力搜索的变体。高层级（第8、9级）中，暴力搜索略优于树搜索，因为那里的节点数量较小。但是，一旦某个层级中的节点数量增加，与暴力搜索相比，树搜索可以排除高层中的低质量结果，从而降低了低层问题的难度，使得树搜索可以获得更好的召回结果。这个结果表明，树结构中包含的层次信息有助于提高推荐的准确性。

**Tree learning**. In Section 3.5, we propose the tree initialization and learning algorithms. Table 4 gives the comparison results between initial tree and learnt tree. From the results, we can observe that the trained model with learnt tree structure significantly outperforms the initial one. For example, the recall metric of learnt tree increases from 4.15% to 4.82% compared to initial tree in experiments of filtering interacted categories, which surpasses YouTube product-DNN’s 3.09% and item-CF’s 1.06% in very large margin. To further compare these two trees, we illustrate the test loss and recall curve of TDM attention-DNN method w.r.t. training iterations in Figure 5. From Figure 5(a), we can see that the learnt tree structure gets smaller test loss. And both Figure 5(a) and 5(b) indi- cate that the model converges to better results with learnt tree. The above results prove that the tree learning algorithm can improve the hierarchy of items, further to facilitate training and prediction.

> **树的学习**。在第3.5节中，我们提出了树的初始化和学习算法。表4 给出了初始树和学习树之间的比较结果。从结果可以看出，使用学习到的树结构的训练模型明显优于初始模型。例如，在筛选互动类别的实验中，学习到的树的召回率从 4.15% 增加到 4.82%，相比于初始树，大大超过了 YouTube product-DNN 的 3.09% 和item-CF 的 1.06%。为了进一步比较这两棵树，我们在图5中展示了 TDM attention-DNN 方法在训练迭代中的测试集损失和召回率曲线。从图5(a)可以看出，学习到的树结构具有更小的测试集损失。图5(a)和5(b)都表明，使用学习到的树模型收敛到更好的结果。上述结果证明，树的学习算法可以改善 item 的层次结构，进一步促进训练和预测的效果。

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure4.png)

**Figure 4: The results of layer-wise Recall@200 in UserBehavior dataset. The ground truth in testing set is traced back to each node’s ancestors, till the root node.**

> **图4：用户行为数据集中 layer-wise 的召回率（Recall@200）结果。测试集中的真实标签被追溯到每个节点的祖先节点，直至根节点。**

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table4.png)

**Table 4: Comparison results of different tree structures in UserBehavior dataset using TDM attention-DNN model (@200). Tree is initialized and learnt according to the algorithm described in Section 3.5.**

> Table4：使用 TDM attention-DNN模型（@200）在UserBehavior数据集中对不同树结构进行比较的结果。根据第3.5节中描述的算法初始化和学习了树。

### 5.5 Online Results

We evaluate the proposed TDM method in Taobao display advertising platform with real traffic. The experiments are conducted in Guess What You Like column of Taobao App Homepage. Two online metrics are used to measure the performance: click-through rate (CTR) and revenue per mille (RPM). Details are as follows:
$$
CTR=\frac{\# \ of \ clicks}{\# \ of \ impressions}, RPM=\frac{Ad \ revenue}{\# \ of \ impressions} * 1000
$$
> 我们使用真实流量在淘宝展示广告平台中评估了提出的TDM方法。实验是在淘宝App首页的“猜你喜欢”栏目中进行的。我们使用了两个在线指标来衡量性能：点击率（CTR）和每千次展示收入（RPM）。具体细节如下：
> $$
> CTR=\frac{\# \ of \ clicks}{\# \ of \ impressions}, RPM=\frac{Ad \ revenue}{\# \ of \ impressions} * 1000
> $$

In our advertising system, advertisers bid on some given ad clusters. There are about 1.4 million clusters and each ad cluster contains hundreds or thousands of similar ads. The experiments are conducted in the granularity of ad cluster to keep consistent with the existing system. The comparison method is mixture of logistic regression [9] that used to pick out superior results only from those interacted clusters, which is a strong baseline. Since there are many stages in the system like CTR prediction [11, 34] and ranking [35] as illustrated in Figure 1, deploying and evaluating the proposed TDM method online is a huge project, which involves the linkage and optimization of the whole system. We have finished the deployment of the first TDM DNN version so far and evaluated its improvements online. Each of the comparison buckets has 5% of all online traffic. It’s worth mentioning that there are several online simultaneously running recommendation methods. 

> 在我们的广告系统中，广告主对给定的广告聚类进行竞价。大约有140万个聚类，每个广告聚类包含数百或数千个相似的广告。为了与现有系统保持一致，实验是以广告聚类的粒度进行的。对照组是混合逻辑回归[9]，仅从那些互动过的聚类中选择出优秀的结果，这是一个强有力的 baseline。由于系统中有许多阶段，如CTR预测[11、34]和排序[35]，如图1所示，将提出的 TDM 方法部署和评估在线上是一个巨大的项目，涉及到整个系统的链接和优化。到目前为止，我们已经完成了第一个TDM DNN版本的部署，并在线上评估了其改进效果。每个小流量占据了所有在线流量的5%。值得一提的是，同时运行着几种在线推荐方法。

They take efforts in different point of views, and their recommendation results are merged together for the following stages. TDM only replaces the most effective one of them while keeping other modules unchanged. The average metric lift rates of the testing bucket with TDM are listed in Table 5.

> 它们从不同的角度生效，并将其推荐结果合并到后续阶段。TDM 只替换其中最有效的方法，而其他模块保持不变。小流量中使用 TDM 的平均指标提升率列在表5中。

![Table5](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table5.png)

**Table 5: Online results from Jan 22 to Jan 28, 2018 in Guess What You Like column of Taobao App Homepage.**

> Table5：淘宝App首页 “猜你喜欢” 栏目的2018年1月22日至1月28日期间的在线结果。

As shown in Table 5, the CTR of TDM method increases 2.1%. This improvement indicates that the proposed method can recall more accurate results for users. And on the other hand the RPM metric increases 6.4%, which means the TDM method can also bring more revenue for Taobao advertising platform. TDM has been deployed to serve major online traffic, we believe that the above improvement is only a preliminary result in a huge project, and there has room for further improvements.

> 如表5所示，TDM方法使点击率（CTR）提高了2.1%（译注：应该是相对值，绝对值岂不是上天了。）。这一改进表明所提出的方法能够为用户提供更准确的结果。另外，RPM指标增加了6.4%，这意味着TDM方法也可以为淘宝广告平台带来更多收益。TDM已经被部署用于处理主要的线上流量，我们相信上述改进只是一个庞大项目中的初步结果，并且还有进一步改进的空间。

**Prediction efficiency**. TDM makes advanced neural network feasible to interact user and items in large-scale recommendation, which opens a new perspective of view in recommender systems. It’s worth mentioning that though advanced neural networks need more calculation when inferring, but the complexity of a whole prediction process is no larger than $O (k ∗ log |C | ∗ t )$, where $k$ is the required results size, $|C|$ is the corpus size and $t$ is the complexity of network’s single feed-forward pass. This complexity upper bound is acceptable under current CPU/GPU hardware conditions, and user side’s features are shared across different nodes in one retrieval and some calculation could be shared according to model designs. In Taobao display advertising system, it actually takes the deployed TDM DNN model about 6 milliseconds to recommend once in average. Such running time is shorter than the following click-through rate prediction module, and is not the system’s bottleneck.

> **预测效率**。TDM 使得在大规模推荐系统中使用先进的神经网络与 user 和 item进行交互成为可能，为推荐系统带来了新的视角。值得一提的是，尽管高级神经网络在推断时需要更多计算，但整个预测过程的复杂度不超过 $O(k * log |C| * t)$ ，其中 $k$ 是所需结果的大小，$|C|$ 是语料库的大小，$t$ 是网络单次前向传递的复杂度。在当前的 CPU/GPU 硬件条件下，这种复杂度上限是可以接受的，并且用户侧的特征在一次检索中可以在不同类型的节点之间共享，并且根据模型设计，一些计算可以共享。在淘宝展示广告系统中，部署的TDM DNN模型平均每次推荐的运行时间约为6毫秒。这个运行时间比后续的点击率预测模块要短，并且不是系统的瓶颈。

## 6 CONCLUSION

We figure out the main challenge for model-based methods to generate recommendations from large-scale corpus, i.e., the amount of calculation problem when making prediction. A tree-based approach is proposed, where arbitrary advanced models can be employed in large-scale recommendation to infer user interests coarse-to-fine along the tree. Besides training the model, a tree structure learning approach is used, which proves that a better tree structure can lead to significantly better results. A possible future direction is to design more elaborate tree learning approaches. We conduct extensive experiments which validate the effectiveness of the proposed method, both in recommendation accuracy and novelty. In addition, empirical analysis showcases how and why the proposed method works. In Taobao display advertising platform, the proposed TDM method has been deployed in production, which improves both business benefits and user experience.

> 我们发现模型方法在从大规模语料库生成推荐时面临的主要挑战是计算量的问题。我们提出了一种基于树结构的方法，可以在大规模推荐中使用任意先进的模型来沿着树结构粗到细地推断用户的兴趣。除了训练模型外，还使用了一种树结构学习方法，证明了更好的树结构可以显著改善结果。一个可能的未来方向是设计更精细的树学习方法。我们进行了大量实验证明了所提出方法的有效性，无论是在推荐准确性还是新颖性方面都得到验证。此外，实证分析展示了该方法的工作原理和原因。在淘宝展示广告平台中，所提出的TDM方法已经在生产环境中部署，这既提高了商业效益，又改善了用户体验。

### ACKNOWLEDGEMENTS

We deeply appreciate Jian Xu, Chengru Song, Chuan Yu, Guorui Zhou and Yongliang Wang for their helpful suggestions and dis- cussions. Thank Huimin Yi, Yang Zheng, Zelin Hu, Sui Huang, Yin Yang and Bochao Liu for implementing the key components of the training and serving infrastructure. Thank Haiyang He, Yangyang Fu and Yang Wang for necessary engineering supports.

### REFERENCES

[1] RahulAgrawal,ArchitGupta,YashotejaPrabhu,andManikVarma.2013.Multi- label learning with millions of labels: Recommending advertiser bid phrases for web pages. In Proceedings of the 22nd international conference on World Wide Web. ACM, 13–24.

[2] Samy Bengio, Jason Weston, and David Grangier. 2010. Label embedding trees for large multi-class tasks. In International Conference on Neural Information Processing Systems. 163–171.

[3]  Alina Beygelzimer, John Langford, and Pradeep Ravikumar. 2007. Multiclass classification with filter trees. Gynecologic Oncology 105, 2 (2007), 312–320.

[4]  Pablo Castells, SaÃžl Vargas, and Jun Wang. 2011. Novelty and Diversity Metrics for Recommender Systems: Choice, Discovery and Relevance. In Proceedings of International Workshop on Diversity in Document Retrieval (2011), 29–37.

[5]  Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra, Hrishi Aradhye, Glen Anderson, Greg Corrado, Wei Chai, Mustafa Ispir, et al. 2016. Wide & deep learning for recommender systems. In Proceedings of the 1st Workshop on Deep Learning for Recommender Systems. ACM, 7–10.

[6]  Anna E Choromanska and John Langford. 2015. Logarithmic time online multi-class prediction. In Advances in Neural Information Processing Systems. 55–63.

[7]  Paul Covington, Jay Adams, and Emre Sargin. 2016. Deep Neural Networks for YouTube Recommendations. In ACM Conference on Recommender Systems. 191–198.

[8]  Robin Devooght and Hugues Bersini. 2016. Collaborative filtering with recurrent neural networks. arXiv preprint arXiv:1608.07400 (2016).

[9]  Kun Gai, Xiaoqiang Zhu, Han Li, Kai Liu, and Zhe Wang. 2017. Learning Piecewise Linear Models from Large Scale Data for Ad Click Prediction. arXiv preprintarXiv:1704.05194 (2017).

[10]  Zeno Gantner, Steffen Rendle, Christoph Freudenthaler, and Lars SchmidtThieme. 2011. MyMediaLite: A free recommender system library. In Proceedings of the fifth ACM conference on Recommender systems. ACM, 305–308.

[11]  Tiezheng Ge, Liqin Zhao, Guorui Zhou, Keyu Chen, Shuying Liu, Huiming Yi, Zelin Hu, Bochao Liu, Peng Sun, Haoyu Liu, et al. 2017. Image Matters: Jointly Train Advertising CTR Model with Image Representation of Ad and User Behavior. arXiv preprint arXiv:1711.06505 (2017).

[12]  FMaxwellHarperandJosephAKonstan.2016.Themovielensdatasets:History and context. ACM Transactions on Interactive Intelligent Systems 5, 4 (2016), 19.

[13]  Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web. 173–182.

[14]  Sergey Ioffe and Christian Szegedy. 2015. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In International conference on machine learning. 448–456.

[15]  HimanshuJain,YashotejaPrabhu,andManikVarma.2016.Extrememulti-label loss functions for recommendation, tagging, ranking & other missing label ap- plications. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 935–944.

[16]  Sébastien Jean, Kyunghyun Cho, Roland Memisevic, and Yoshua Bengio. 2014. On using very large target vocabulary for neural machine translation. arXiv preprint arXiv:1412.2007 (2014).

[17]  JunqiJin,ChengruSong,HanLi,KunGai,JunWang,andWeinanZhang.2018. Real-Time Bidding with Multi-Agent Reinforcement Learning in Display Adver- tising. arXiv preprint arXiv:1802.09756 (2018).

[18]  Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with GPUs. arXiv preprint arXiv:1702.08734 (2017).

[19]  YehudaKoren,RobertBell,andChrisVolinsky.2009.MatrixFactorizationTech- niques for Recommender Systems. Computer 42, 8 (2009), 30–37.

[20]  Dawen Liang, Jaan Altosaar, Laurent Charlin, and David M. Blei. 2016. Factor- ization Meets the Item Embedding:Regularizing Matrix Factorization with Item Co-occurrence. In ACM Conference on Recommender Systems. 59–66.

[21]  D. Lin. 1999. WordNet: An Electronic Lexical Database. Computational Linguis- tics 25, 2 (1999), 292–296.

[22]  Greg Linden, Brent Smith, and Jeremy York. 2003. Amazon.com recommenda- tions: Item-to-item collaborative filtering. IEEE Internet computing 7, 1 (2003), 76–80.

[23]  TomasMikolov,IlyaSutskever,KaiChen,GregCorrado,andJeffreyDean.2013. Distributed representations of words and phrases and their compositionality. In International Conference on Neural Information Processing Systems. 3111–3119.

[24]  Frederic Morin and Yoshua Bengio. 2005. Hierarchical probabilistic neural net- work language model. Aistats (2005).

[25]  Andrew Y. Ng, Michael I. Jordan, and Yair Weiss. 2001. On spectral clustering: analysis and an algorithm. In International Conference on Neural Information Pro- cessing Systems: Natural and Synthetic. 849–856.

[26]  Yashoteja Prabhu and Manik Varma. 2014. Fastxml: A fast, accurate and stable tree-classifier for extreme multi-label learning. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 263–272.

[27]  YanruQu,HanCai,KanRen,WeinanZhang,YongYu,YingWen,andJunWang. 2016. Product-based neural networks for user response prediction. In IEEE 16th International Conference on Data Mining. IEEE, 1149–1154.

[28]  Steffen Rendle. 2010. Factorization Machines. In IEEE International Conference on Data Mining. 995–1000.

[29]  Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt- Thieme. 2009. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the 25th conference on uncertainty in artificial intelligence. AUAI Press, 452–461.

[30]  Ruslan Salakhutdinov and Andriy Mnih. 2007. Probabilistic Matrix Factorization. In International Conference on Neural Information Processing Systems. 1257–1264.

[31]  Badrul Sarwar, George Karypis, Joseph Konstan, and John Riedl. 2001. Item- based collaborative filtering recommendation algorithms. In International Con- ference on World Wide Web. 285–295.

[32]  J. Weston, A. Makadia, and H. Yee. 2013. Label partitioning for sublinear ranking. In International Conference on Machine Learning. 181–189.

[33]  Bing Xu, Naiyan Wang, Tianqi Chen, and Mu Li. 2015. Empirical evaluation of rectified activations in convolutional network. arXiv:1505.00853 (2015).

[34]  Guorui Zhou, Chengru Song, Xiaoqiang Zhu, Xiao Ma, Yanghui Yan, Xingya Dai, Han Zhu, Junqi Jin, Han Li, and Kun Gai. 2018. Deep interest network for click-through rate prediction. In Proceedings of the 24th ACM SIGKDD Conference. ACM.

[35]  Han Zhu, Junqi Jin, Chang Tan, Fei Pan, Yifan Zeng, Han Li, and Kun Gai. 2017. Optimized Cost Per Click in Taobao Display Advertising. In Proceedings of the 23rd ACM SIGKDD Conference. ACM, 2191–2200.
