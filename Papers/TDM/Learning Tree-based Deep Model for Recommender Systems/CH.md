# Learning Tree-based Deep Model for Recommender Systems

### ABSTRACT

近年来，model-based 的推荐系统方法得到了广泛研究。然而，在大规模数据集的系统中，学习模型以预测所有 user-item 偏好所需的计算成本非常高，这使得全量数据集检索变得极为困难。为了克服算力障碍，模型如矩阵分解，采用内积形式来建模 user-item 偏好（即将 user 和 item 的潜在因子进行内积运算），并利用索引来实现高效的近似的 k-nn 搜索。然而，由于计算成本的问题，要想将更具表达力的 user 和 item 特征之间的交互形式（如深度神经网络）纳入模型仍然具有挑战性。

本文着重解决了将任意高级模型引入到具有大规模数据集的推荐系统中的问题。我们提出了一种新的Tree-based方法，即使使用更具表达力的模型（如深度神经网络），也能（相对于数据集大小）保持对数复杂度。我们的主要思想是通过自上而下地遍历树节点，并为每个user-item组合做出决策，从粗粒度到细粒度预测用户兴趣。我们还证明树结构可以 jointly learn，以更好地与用户的兴趣分布相适应，从而提升训练和预测两方面的性能。通过对两个大规模实际数据集的实验评估，我们发现新方案明显优于传统方案。此外，在淘宝广告平台进行的线上A/B实验结果也证明了新方案在实际生产环境中的有效性。

推荐系统已被各种内容提供商广泛使用。个性化推荐方法基于这样的认识：用户的兴趣可以从他们的历史行为或具有相似偏好的其他用户中进行推断，这在YouTube [7]和Amazon [22]中已被证明是有效的。

设计一个能够从整个数据集中预测出最佳候选集合的推荐模型其中存在许多挑战。在拥有庞大数据集的系统中，一些表现良好的推荐算法可能无法对整个数据集进行预测。与数据集大小成线性关系的预测复杂度是不可接受的。部署这样的大规模推荐系统需要对每个用户预测的计算量有所限制。用户体验上，推荐的 item 的新颖性也很重要。仅包含用户历史行为中的同质化item的结果是不被期望的。

为了减少计算量并处理庞大的数据集，内存型协同过滤方法被广泛应用于工业界 [22]。作为协同过滤家族中的代表性方法，基于 item 的协同过滤 [31] 可以从非常庞大的数据集中进行推荐，并且相对较少的计算量，这取决于预先计算的 item-pair 之间的相似度，并使用用户的历史行为作为 trigger 来寻找那些最相似的 item。然而，候选集的范围存在限制，即最终只能推荐与 trigger 相似的 item，而不是所有的 item。这样阻止了推荐系统从用户历史行为中跳出来探索潜在的用户兴趣，这限制了被推荐结果的准确性（译注：原文 which limits the accuracy of recalled results）。实际上也限制了新颖性。

另一种减少计算量的方法是进行粗粒度的推荐。例如，系统为用户推荐少量item类别，然后从类别中挑选出所有相应的item，在随后的排序阶段对它们进行排序。然而，对于大规模数据集，算力问题仍然没有解决。如果类别数量很多，那么推荐类别本身也会遇到算力障碍。如果类别数量较少，某些类别将不可避免地包含太多的物品，使得随后的排序计算变得不切实际。此外，所使用的类别通常并非专为推荐问题设计，这可能严重损害推荐的准确性。

然而，user 和 item 向量表示之间的内积交互形式严重限制了模型的能力。存在许多其他更具表达力的交互形式，例如，用户历史行为和候选 item 之间的交叉特征在点击率预测中被广泛使用[5]。最近的研究[13]提出了一种神经协同过滤方法，其中使用神经网络而不是内积来建模用户和物品向量表示之间的交互。该研究的实验结果证明，多层前馈神经网络的表现优于固定的内积方式。deep interest network [34]指出用户的兴趣是多样的，并且类似注意力的网络结构可以根据不同的候选 item 生成不同的用户向量。除了上述工作，其他方法如产品 product neural network[27]也证明了先进神经网络的有效性。然而，由于这些模型无法将 user和 item 向量调整为内积形式以利用高效的近似k最近邻搜索，它们不能用于在大规模推荐系统中检索候选 item。如何克服算力瓶颈，使任意先进神经网络在大规模推荐中成为可能，是一个问题。

为了解决上述挑战，本文提出了一种新颖的基于树的深度推荐模型（TDM）。在多分类问题中，研究人员通常使用树来划分样本或标签空间以降低计算成本。然而，研究人员很少在使用树结构作为检索索引的推荐系统上进行探索。实际上，信息的层次结构普遍存在于许多领域。例如，在电子商务场景中，iPhone是精细粒度的物品，而智能手机是iPhone所属的粗粒度概念。所提出的TDM方法利用信息的这种层次结构，将推荐问题转化为一系列的层次分类问题。通过从易到难解决问题，TDM可以提高准确性和效果。

本篇论文的主要贡献总结如下：

- 据我们所知，TDM 是第一种使任意高级模型能够从大型数据集生成推荐结果的方法。通过利用层次树搜索，TDM在进行预测时的计算量与数据集大小成对数关系。
- TDM能够更精确地找到新颖且有效的推荐结果，因为它能够探索整个数据集，并且更有效的深度模型也能帮助发现潜在兴趣。
- 除了使用更先进的模型，TDM还通过层次搜索提高了推荐准确性，将一个大问题划分为多个小问题，并逐步从易到难解决这些问题。
- 作为一种索引，树结构可以学习到 item 和 concepts 的最优层次结构，从而更有效的检索，进而促使模型训练。我们采用了一种树学习方法，可以联合训练神经网络和树结构。
- 我们在两个大规模真实世界数据集上进行了大量实验，结果表明TDM明显优于现有方法。

值得一提的是，树结构方法在语言模型工作中也有研究，如 hierarchical softmax [24]，但它与提出的TDM不仅在动机上有所区别，而且在表述方式上也存在差异。在下一个单词预测问题中，传统的softmax需要计算归一化项才能得到任何单词的概率，这非常耗时。hierarchical softmax 使用树结构，将下一个单词的概率转换为沿树路径的节点概率的乘积。这种表述方式将下一个单词的概率的计算复杂度降低到与语料库大小成对数量级的程度。然而，在推荐问题中，目标是在整个数据集中搜索那些最受喜爱的物品，这是一个检索问题。在 hierarchical softmax树中，父节点的最优性不能保证最优的低层节点在它们的后代中，仍然需要遍历所有item来找到最优解。因此，它不适用于这样的检索问题。为了解决检索问题，我们提出了类似于最大堆的树形结构，并引入深度神经网络来建模该树结构，从而形成了一种适用于大规模推荐的高效方法。接下来的部分将展示它在表述方式上的差异以及在性能上的优越性。此外，hierarchical softmax 为特定的自然语言处理问题采用了单隐藏层网络，而所提出的TDM方法适用于任何神经网络结构。

值得一提的是，树结构方法在语言模型工作中也有研究，如 hierarchical softmax [24]，但它与提出的TDM不仅在动机上有所区别，而且在表述方式上也存在差异。在下一个单词预测问题中，传统的softmax需要计算归一化项才能得到被预测单词的概率，这非常耗时。hierarchical softmax 使用树结构，将下一个单词的概率转换为沿树路径的节点概率的乘积。这种表述方式将下一个单词的概率的计算复杂度降低到与语料库大小成对数量级的程度。然而，在推荐问题中，目标是在整个数据集中搜索那些最受喜爱的物品，这是一个检索问题。在 hierarchical softmax 树中，父节点的最优性不能保证最优的低层节点在它们的后代中，仍然需要遍历所有item来找到最优解。

> 注：HS是单词查找树，词与词之间没有从粗到细的逻辑关系。

因此，它不适用于这样的检索问题。为了解决检索问题，我们提出了类似于最大堆的树结构，并引入深度神经网络来建模该树结构，从而形成了一种适用于大规模推荐的高效方法。接下来的部分将展示它在表述方式上的差异以及在性能上的优越性。此外，hierarchical softmax 为特定的自然语言处理问题采用了单隐藏层网络，而所提出的TDM方法适用于任何神经网络结构。

提出的 Tree-based 的模型是一种适用于各种在线内容提供商的通用解决方案。本文的剩余部分组织如下：在第2节中，我们将介绍淘宝展示广告系统的体系结构，以展示所提方法所作用的位置。第3节将对提出的基于树的深度模型进行详细的介绍和形式化描述。接下来的第4节将描述基于树的模型如何服务于在线推荐。第5节展示了在大规模基准数据集和淘宝广告数据集上的实验结果。最后，第6节对我们的工作进行总结。

## 2 SYSTEM ARCHITECTURE

在本节中，我们介绍了淘宝展示广告推荐系统的架构，如图1所示。当用户发送页面浏览请求后，系统将使用用户特征、context特征和商品特征作为输入，在召回(matching)服务器中从整个数据集（数亿个商品）中生成一个相对较小的候选商品集合（通常为数百个）。树结构的推荐模型在这一阶段发挥了作用，将候选集的大小缩减了几个数量级。

有了数百个候选商品，实时预测服务器使用更加表达能力但也更耗时的模型[11, 34]来预测诸如点击率或转化率等指标。经过策略性的排序[17, 35]之后，最终会向用户展示几个商品。

正如前面提到的，所提出的推荐模型旨在构建一个包含数百个商品的候选集合。这个阶段既至关重要又具有挑战性。用户是否对生成的候选商品感兴趣可以界定印象质量的上限。如何在考虑效率和有效性的情况下选取候选商品是一个问题。

![](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure1.png)

**Figure 1**: The system architecture of Taobao display advertising recommender system.

## 3 TREE-BASED DEEP MODEL

在本部分中，我们首先介绍了树模型中使用的树结构，以便给出一个整体概念。其次，我们介绍了分层Softmax[24]，以说明为什么它的公式不适用于推荐系统。接下来，我们提出了一种新颖的类似于最大堆的树结构表达方式，并展示了如何训练基于树的模型。然后，介绍了深度神经网络的架构。最后，我们展示了如何构建和学习用于树模型的树结构。

### 3.1 Tree for Recommendation

一个推荐树由一组节点 $N$ 组成，其中 
$$
N = \{n_1, n_2, \cdots n_{|N|}\}
$$
代表着 $|N|$ 个独立的非叶子节点或叶子节点。除了根节点之外，$N$ 中的每个节点都有一个父节点和任意数量的子节点。具体来说，数据集 $C$ 中的每个商品 $c_i$ 对应于树中的一个且仅有一个叶子节点，而那些非叶子节点则代表粗粒度的概念。为了方便起见，我们假设节点 $n_1$ 始终是根节点。图2 右下角展示了一个示例树，其中每个圆圈代表一个节点，节点上的数字表示它在树中的索引。该树总共有8个叶子节点，每个叶子节点对应于语料库中的一个商品。值得一提的是，尽管给定的示例是一个完全二叉树，但我们在模型中不将 “完全“和”二叉“作为树类型的限制条件。

### 3.2 Related Work

使用树结构，我们首先介绍了与我们的推荐树模型(TDM)不同的相关工作—— Hierarchical Softmax，以帮助理解二者的区别。在 分层Softmax 中，树中的每个叶子节点 $n$ 都有从根节点到该节点的唯一编码。例如，如果我们将选择左分支编码为1，选择右分支编码为0，在图2中树中 $n_9$ 的编码为 110，$n_{15}$ 的编码为 000。记 $b_j(n)$ 为第 $j$ 级中节点 $n$ 的编码。在 Hierarchical Softmax 的公式中，给定上下文，下一个词在树中的概率可以表示为：
$$
P(n \mid \text { context })=\prod_{j=1}^w P\left(b=b_j(n) \mid l_j(n), \text { context }\right)
\\(1)
$$
其中 $w$ 是叶子节点 $n$ 的编码长度，$l_j(n)$ 是节点 $n$ 在第 $j$ 级上的祖先节点。

通过避免计算传统 Softmax 中的归一化项（需要遍历语料库中的每个词），Hierarchical Softmax 解决了概率计算问题。然而，为了找到最可能的叶子节点，模型仍然需要遍历整个语料库。沿着树路径自顶向下遍历每个层次上最可能的节点并不能保证成功地找到最优的叶子节点。因此，Hierarchical Softmax 的公式对于大规模检索问题并不合适。此外，根据 公式1，树中的每个非叶子节点被训练为一个二分类器，以区分其两个子节点。但是，如果两个节点在树中相邻，它们可能是相似的。在推荐场景中，用户可能对这两个子节点都感兴趣。Hierarchical Softmax 的模型注重区分最优和次优选择，这可能会失去从全局视角区分的能力。如果使用贪心的 beam search 来检索那些最可能的叶子节点，一旦在树的较高层次做出了错误的决策，模型可能无法在较低层次的相对低质候选集中找到相对更好的结果。YouTube的研究[7]也报告了他们尝试使用 Hierarchical Softmax 学习 user 和 item embedding，但效果不如 sampled-softmax [16]的方式。

鉴于 Hierarchical Softmax 的公式不适用 大规模推荐，我们在接下来的部分提出了一种新的树模型公式。

### 3.3 Tree-based Model Formulation

为了解决高效检索用户最喜欢的商品的问题，我们提出了一种类似于最大堆的树形概率结构。类似于最大堆的树是一种树结构，在第 $j$ 层中的每个非叶子节点 $n$ 对于每个用户 $u$ 满足以下方程：
$$
\begin{aligned}
& P^{(j)}(n \mid u)=\frac{\overset{max}{n_c \in\{n \text { 's children nodes in level } j+1}\}}{\alpha^{(j)}}^{p(j+1)(n_c|u)} \\
&
\end{aligned}

\\(2)
$$
其中 $P^{(j)}(n∣u)$ 是用户 $u$ 对 $n$ 感兴趣的真实概率。$\alpha(j)$ 是第 $j$ 层的归一化项，用于确保该层上的概率之和等于1。公式2 表示父节点的真实偏好等于其子节点中的最大偏好除以归一化项。请注意，我们稍微滥用了符号，将 $u$ 表示为特定的用户状态。换句话说，一旦用户有新的行为，特定的用户状态 $u$ 可能转变为另一个状态 $u^′$。

目标是找到具有最大偏好概率的 $k$ 个叶节点。假设我们在树中拥有每个节点 $n$ 的真实情况 $P^{(j)}(n|u)$ ，我们可以逐层检索具有最大偏好概率的 $k$ 个节点，并且只需要探索每一级的前 $k$ 个子节点。通过这种方式，可以最终获取到前 $k$ 个叶节点。实际上，在上述检索过程中，我们不需要知道每个树节点的准确真实概率。我们需要的是每个层级中节点的概率大小顺序，以帮助找到该层级中的前 $k$ 个节点。基于这一观察，我们使用用户的隐式反馈数据和神经网络来训练每个层级的判别器，以确定偏好概率的顺序。

假设用户 $u$ 与叶节点 $n_d$ 进行了交互，即 $n_d$ 是 $u$ 的一个正样本节点。这意味着存在一个顺序 $P^{(m)}(n_d |u) > P^{(m)}(n_t |u)$，其中 $m$ 是叶节点的层级，$n_t$ 是任意其他叶节点。在任意层级 $j$ 中，将 $l_j(n_d)$ 表示为 $j$ 层中 $n_d$ 的祖先 。根据 公式2 中树的构造，我们可以推导出 $P^{(j)}(l_j(n_d)|u) > P^{(j)}(n_q|u)$，其中 $n_q$ 是层级 $j$ 中除 $l_j{(n_d)}$ 之外的任意节点。基于以上分析，我们可以使用负采样[23]来训练每个层级的 order-discriminator(顺序判别器)。具体来说，与 $u$ 有交互的叶节点及其祖先节点构成了每个层级对于 $u$ 的正样本集合。而在每个层级中，随机选择除了正样本以外的节点作为负样本集合。图2 中的绿色和红色节点给出了采样示例。假设给定一个用户和其状态，目标节点是 $n_{13}$。那么，$n_{13}$ 的祖先节点是正样本，而在每个层级中随机选择的红色节点则是负样本。这些样本随后被送到二分类模型中，以获得各层级的顺序判别器。我们使用一个全局深度神经网络二分类模型，为所有层级的顺序判别器提供不同的输入。可以采用任意先进的神经网络来提高模型的能力。

> 译注：非叶子节点的负样本采样有一些缺陷，不像普通模型的负样本采样，非叶子节点因为是log衰减，越靠近顶层，非叶节点叶点越少，但实际上在线上使用时 beam search上是取 topk而不是top1，这样的话训练和预测存在不一致性。

将 $\mathcal{Y}^+_u$ 、 $\mathcal{Y}^−_u$ 表示为用户 $u$ 的正负样本集合. 似然函数如下:
$$
\prod_u\left(\prod_{n \in \mathcal{Y}^+_u} P\left(\hat{y}_u(n)=1 \mid n, u\right) \prod_{n \in \mathcal{Y}^-_{u}} P\left(\hat{y}_u(n)=0 \mid n, u\right)\right) \text {, }
\\
(3)
$$
其中 $\hat{y}_u(n)$ 是给定 $u$ 节点 $n$ 的预测标签. $P\left(\hat{y}_u(n) \mid n, u\right)$ 是二分类模型的输出，以用户状态 $u$ 和采样节点 $n$ 作为输入. 

其损失函数可以表示为：
$$
-\sum_u \sum_{n \in \mathcal{Y}_u^{+} \cup \mathcal{Y}_{\bar{u}}^{-}}
\\
y_u(n) \log P\left(\hat{y}_u(n)=1 \mid n, u\right)+\left(1-y_u(n)\right) \log P\left(\hat{y}_u(n)=0 \mid n, u\right)
\\
(4)
$$
其中 $y_u^{(n)}$ 是给定用户 $u$ 的节点 $n$ 的真实标签。有关如何根据损失函数训练模型的详细信息，请参考第3.4节。

需要注意的是，提出的采样方法与 hierarchical softmax 中的底层采样方法有很大的区别。与 hierarchical softmax 中的方法相比，该方法使模型能够区分最优和次优结果，我们在每个正样本节点的同一层级中随机选择负样本。这种方法使得每个层级的判别器成为一个层内的全局判别器。每个层级的全局判别器可以独立地做出精确的决策，而不依赖于上层决策的好坏。这种全局判别能力对于分层推荐方法非常重要。它确保即使在上层中模型做出了错误的决策并且低质量的节点泄漏到候选集中，模型在后续层级中仍然可以选择相对较好的节点，而不是非常差的节点。

> 译注：树的父子节点之间其实本身就是一种强联系，独立地做出精确的决策这种描述很有问题。

给定一个推荐树和一个优化的模型，算法1描述了详细的分层预测算法。检索过程按层次和自顶向下进行。假设期望的候选项数量为 $k$。对于大小为$|C|$ 的数据集 $C$，最多遍历 $2 \times k \times log|C|$ 个节点即可在完全二叉树中得到最终的推荐集合。需要遍历的节点数量与语数据集大小呈对数关系，这使得计算成本上可以使用先进的二分类模型来进行推荐。

我们提出的 TDM 方法不仅减少了在进行预测时的计算量，而且与在所有叶节点上进行暴力搜索相比，还有改进推荐质量的潜力。没有树结构，直接训练一个模型来找到最优项是一个困难的问题，因为涉及到的数据集的规模很大。通过使用树结构，一个大规模的推荐问题被划分为许多较小的子问题。树的高层级只存在少数节点，因此判别问题更容易解决。高层级的决策可以优化候选集合，从而帮助低层级做出更好的判断。第5.4节中的实验结果将显示，所提出的分层检索方法优于直接的暴力搜索方法。简而言之，我们的TDM方法降低了预测时的计算量，并通过树形层次结构将大规模的推荐问题转化为许多小问题。这种方法不仅减少了复杂度，还能提高推荐质量。实验结果表明，与直接暴力搜索相比，我们提出的分层检索方法表现更好。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Alg1.png)

### 3.4 The Deep Model

在接下来的部分，我们介绍了我们使用的深度模型。整个模型如图2所示。受ctr预测工作[34]的启发，我们为树中的每个节点学习低维的 embedding，并使用 attention 模块 来软搜索相关行为以更好地表示用户。为了利用包含时间戳信息的用户行为，我们设计了分块输入层来区分位于不同时间窗口内的行为。历史行为可以沿时间轴划分为不同的时间窗口，每个时间窗口中的 item embedding 进行加权平均。注意力模块和后续的网络极大地增强了模型的能力，并且使得用户对 item 的偏好不能被限制为内积形式。

树节点的 embedding 和树结构本身也是模型的一部分。为了最小化 Loss 4，我们使用采样的节点和相应的特征来训练网络。需要注意的是，在 图2 中我们只简要展示了用户行为特征的使用，而实际上其他特征，如用户画像或上下文特征，也可以被无障碍地使用。

![Fig2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure2.png)

**Figure 2**: The tree-based deep model architecture. User behaviors are divided into different time windows according to the timestamp. In each time window, item embeddings are weighted averaged, and the weights come from activation units. Each time window’s output along with the candidate node’s embedding are concatenated as the neural network input. After three fully connected layers with PReLU [33] activation and batch normalization [14], a binary softmax is used to yield the probability whether the user is interested in the candidate node. Each item and its corresponding leaf node share the same embedding. All embeddings are randomly initialized.

### 3.5 Tree Construction and Learning

推荐树是基于树的深度推荐模型的基本组成部分。与多分类和多标签分类工作不同，在那些工作中树被用于对样本或标签进行分割，我们的推荐树用于索引检索项。在 hierarchical softmax 中，词的层次结构是根据 WordNet 的专业知识建立起来的。然而，在推荐场景中，并非每个语料库都能提供特定的专业知识。一种直观的替代方法是使用从数据集中得出的 item 共现性或相似性来构建树形结构。但是，聚类后的树可能会非常不平衡，这对训练和检索是有害的。根据 item 的两两相似度，[2]中的算法通过谱聚类将 item 递归地划分为子集。然而，谱聚类对于大规模语料库来说不够可扩展（与语料库大小相比具有立方时间复杂度）。在本节中，我们将重点关注合理和可行的树构建和学习方法。

**树的初始化**

由于我们假设树代表用户兴趣的层次信息，自然地，我们希望以相似 item 相互靠近的方式构建树。鉴于在许多领域中类别信息是广泛可用的，我们直观地提出一种利用 item 的类别信息来构建初始树的方法。在本节中，我们以二叉树为例进行说明。首先，我们随机对所有类别进行排序，并将属于同一类别的 item 按照类内随机顺序放在一起。如果一个 item 属于多个类别，则在这些类别中随机选择一个类别以确保唯一性。通过这种方式，我们可以得到一个经过排序的 item 列表。 其次，我们对这些排序的 item 进行递归的二分操作，直到当前集合只包含一个 item ，这样可以从上往下构建一个接近完全二叉树。与完全随机树相比，基于类别的初始化方法在实验中可以得到更好的层次结构和结果。

**树的学习**

作为模型的一部分，在模型训练之后可以学习每个叶节点的 embedding。然后，我们使用学到的叶节点 embedding 向量来聚类一个新的树。考虑到语料库的规模，我们使用 k-means 聚类算法因为它具有良好的可扩展性。在每一步中，根据 item 的 embedding 向量将其聚类为两个子集。需要注意的是，为了得到更平衡的树，这两个子集会被调整为相等。当只剩下一个 item 时，递归停止，从而以自顶向下的方式构建二叉树。在我们的实验中，当语料库大小约为400万时，使用单台机器构建这样一个聚类树大约需要一个小时。第5节的实验结果将展示给定的树学习算法的有效性。 

深度模型和树结构是以交替的方式进行联合学习：

1. 构建初始树并训练模型直至收敛；
2. 基于训练后的叶节点 embedding 学习获得一个新的树结构；
3. 使用学到的新树结构重新训练模型；

> 译注：此处有为了构建一个平衡的树而构建一个平衡的意思，完全二叉树个人感觉没有必要，现实生活中类别本身就是不均衡的。

## 4 ONLINE SERVING

图3 所示是所提方法的在线服务系统架构。输入的特征组装和 item 检索被分两个并发阶段。每个用户行为，包括点击、购买和将 item 添加到购物车中，都会触发实时特征服务器来组装新的输入特征。一旦接收到页面浏览请求，用户定向服务器将使用预充的特征从树中检索候选项。如算法1所描述的，检索是逐层进行的，并且使用训练好的神经网络来计算给定输入特征时节点是否首选的概率。

![Fig3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure3.png)

**Figure 3: The online serving system of the tree-based model, where user feature is assembled asynchronously.**

## 5 EXPERIMENTAL STUDY

在本节中，我们对提出的基于树的模型进行了性能研究。我们展示了在 MovieLens-20M [12]和称为UserBehavior 的淘宝广告数据集中的实验结果。在实验中，我们将提出的方法与其他现有方法进行比较，以展示该模型的有效性，并通过实验研究结果展示基于树的模型和树学习算法的工作原理。

### 5.1 Datasets

实验在两个具有时间戳的大规模真实世界数据集上进行：1）来自MovieLens [12]的用户电影观看数据；2）来自淘宝的一个称为UserBehavior的 user-item 行为数据集。以下是更详细的说明：

**MovieLens-20M**：该数据集包含了用户对电影的评分和时间戳。由于我们处理的是隐性反馈问题，我们将评分进行二值化处理，保留4分及以上的评分，这是其他研究中常用的方式[8, 20]。此外，只保留至少观看了10部电影的用户。为了创建训练、验证和测试数据集，我们随机选择1,000个用户作为测试集，另外1,000个用户作为验证集，而剩下的用户构成训练集[8]。对于验证集和测试集，用户在时间线上的前一半电影观看行为被视为已知行为，用来预测后一半的行为。

**UserBehavior1**：该数据集是淘宝用户行为数据的一个子集。我们随机选择了大约100万名用户，在2017年11月25日至12月3日期间具有点击、购买、将商品添加到购物车和收藏商品等行为。数据的组织形式与 MovieLens-20M 非常相似，即 user-item 行为由 user-ID、item-ID、item-cate-ID、行为类型和时间戳组成。与MovieLens-20M一样，只保留至少有10个行为的用户。其中10,000个用户被随机选择作为测试集，另外随机选择的10,000个用户作为验证集。item 的类别来自淘宝当前商品分类法的底层(次级类目)。表1总结了经过预处理后上述两个数据集的主要维度。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table1.png)

**Table 1**: Dimensions of the two datasets after preprocessing. One record is a user-item pair that represents user feedback.

### 5.2 Metrics and Comparison Methods

为了评估不同方法的有效性，我们使用 Precision@M、Recall@M 和 F-Measure@M 指标[20]。对于用户 $u$，我们将其召回集合定义为 $P_u (|P_u|=M)$，将用户的 ground truth 集合定义为 $G_u$。Precision@M 和 Recall@M 的计算公式如下：
$$
\text{Precision@M(u)} = \frac{|P_u \cap G_u|}{M}, \quad \text{Recall@M(u)} = \frac{|P_u \cap G_u|}{|G_u|}
$$
F-Measure@M 的计算公式如下： 
$$
\text{F-Measure@M(u)} = \frac{2 \times \text{Precision@M(u)} \times \text{Recall@M(u)}}{\text{Precision@M(u)} + \text{Recall@M(u)}}
$$


另外，推荐结果的新颖性对用户体验至关重要。现有研究[4]提供了几种衡量推荐列表新颖性的方法。根据其中一种定义，新颖性指标 Novelty@M 定义如下： 
$$
\text{Novelty@M(u)} = \frac{|P_u \setminus S_u|}{M}
$$
其中 $S_u$ 是在推荐之前与用户 $u$ 有过交互的 item 集合。 测试集中上述四个指标的用户平均值被用来比较以下方法： 

- **FM**[28]：FM是一种用于因子分解任务的框架。我们使用 xLearn 项目提供的FM实现。
- **BPR-MF**[29]：我们使用其针对隐式反馈推荐的矩阵分解形式。使用[10]提供的BPR-MF实现。
- **Item-CF**[31]：基于物品的协同过滤是生产中最常用的个性化推荐方法之一，适用于大规模语料库[22]。它也是淘宝主要的候选集生成方法之一。我们使用阿里巴巴机器学习平台提供的 item-CF 实现。
- **YouTube product-DNN**[7]：这是YouTube提出的深度推荐方法。在训练中使用了采样 softmax [16]，user-item 的内积反映了偏好。我们在阿里巴巴深度学习平台上使用与我们提出的模型相同的输入特征实现了YouTube product-DNN。预测时采用内积空间中的精确 kNN 搜索。
- **TDM attention-DNN**（基于树的深度模型使用注意力网络）是我们在图2中提出的方法。树的初始化方式如第3.5节所述，在实验过程中保持不变。该实现可在GitHub3上获得。

对于 FM、BPR-MF 和 item-CF，我们根据验证集调整了几个最重要的超参数，即 FM 和 BPR-MF 中的因子数量和迭代次数，以及 item-CF 中的邻居数量。FM 和 BPR-MF 要求测试集或验证集中的用户在训练集中也有反馈。因此，在两个数据集中，我们将测试集和验证集上时间线上的前一半 user-item 交互行为添加到训练集中。对于YouTube product-DNN 和 TDM attention-DNN，节点embedding 的维度设置为24，因为在我们的实验中，更高的维度并没有表现出显著提升。三个全连接层的隐藏单元数量分别为 128、64 和 24。根据时间戳，将用户行为分为10个时间窗口。在YouTube product-DNN和TDM attention-DNN中，对于每个隐式反馈，我们在MovieLens-20M 中随机选择 100 个负样本，在 UserBehavior 中随机选择600个负样本。请注意，TDM 的负样本数量是所有层级的总和。我们会对靠近叶子的层级进行更多的负采样。

## 5.3 Comparison Results

不同方法的比较结果显示在虚线上方的 Table 2中。每个度量指标是在测试集中所有用户的平均值，呈现的数值是具有方差的方法在五次不同运行中的平均值。

首先，结果表明，在大多数度量指标上，提出的 TDM attention-DNN 在两个数据集上都显著优于所有基线方法。与排名第二的 YouTube product-DNN 方法相比，TDM attetion-DNN 在两个数据集上的召回率指标上分别实现了 21.1% 和 42.6% 的改进，而且没有进行过滤（已交互的item）。这个结果证明了 TDM attetion-DNN 采用的先进神经网络和分层树搜索的有效性。在将 user-item 的偏好建模为内积形式的方法中，由于使用了神经网络，YouTube product-DNN 优于 BPR-MF 和 FM。而广泛使用的 item协同过滤（item-CF）方法则得到了最差的新颖性结果，因为它对用户已经交互过的 item 有很强的记忆。

为了提高新颖性，在实践中常见的一种方法是在推荐集中过滤那些已经交互过的 item[8, 20]，即只有那些新颖的 item 最终才能被推荐。因此，在一个完整的新颖结果集中比较准确性更为重要。在这个实验中，如果过滤后的结果集大小小于 $M$，将会补充到需要的数量 $M$。Table2 的下半部分显示，在过滤掉已交互 item 后，TDM attention-DNN 在各项指标上都显著优于所有基线方法。

为了进一步评估不同方法的探索能力，我们通过排除推荐结果中与用户交互的类别来进行实验。每种方法的结果也被补充以满足大小要求。事实上，目前淘宝推荐系统中最重要的新颖度度量是基于类别的新颖度，因为我们希望减少与用户交互 item 相似的推荐。由于 MovieLens-20M 总共只有 20 个类别，这些实验仅在 UserBehavior 数据集中进行，并且结果显示在表 3 中。以召回率指标为例，我们可以观察到 item 协同过滤（item-CF）的召回率仅为1.06%，因为其推荐结果很难跳出用户的历史行为。与 item-CF 相比，YouTube product-DNN 的结果要好得多，因为它可以从整个语料库中探索用户的潜在兴趣。所提出的 TDM attention-DNN 在召回率方面比YouTube的内积方式改进了34.3%。这样巨大的改进对于推荐系统来说非常重要，它证明了更先进的模型在推荐问题中有着巨大的差异。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table2.png)

**Table 2**: The comparison results of different methods in MovieLens-20M and UserBehavior datasets. According to the different corpussize, metrics are evaluated @10 in MovieLens-20 and @200 in UserBehavior. In experiments of filtering interacted items, the recommendation results and ground truth only contain items that the user has not yet interacted with before.

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table3.png)

**Table 3**: Results in UserBehavior dataset. Items belong to interacted categories are excluded from recommendation results and ground truth.

### 5.4 Empirical Analysis

TDM的变体。为了理解所提出的TDM方法本身，我们推导并评估了几种TDM的变体： 

- TDM product-DNN。为了确定先进的神经网络是否能够改善TDM的结果，我们测试了TDM product-DNN这个变体。TDM product-DNN使用与YouTube product-DNN相同的内积方式。具体来说，Figure2 中的 attention 模块被移除，并且 node embedding item 也从网络输入中移除。节点 embedding 和第三个全连接层输出（不包括PReLU和BN）之间的内积以及一个 sigmoid 激活函数构成了新的二分类器。 
- TDM DNN。为了进一步验证 TDM attention-DNN 中注意力模块所带来的改进，我们测试了只移除激活单元的TDM DNN变体，即在图2中所有 item 的权重都是1.0。
- TDM attention-DNN-HS。如第3节所述，hierarchical softmax（HS）方法[24]不适用于推荐任务。我们测试了 TDM attention-DNN-HS 这个变体，即将正样本节点的邻居作为负样本，而不是随机选择的样本。对应地，在算法1的检索中，排名指标由单个节点的 $P\left(\hat{y}_u(n)=1 \mid n, u\right)$ 更改为 $\prod_{n^{\prime} \in \text { n's ancestors }} P\left(\hat{y}_u\left(n^{\prime}\right)=\right.\left.1 \mid n^{\prime}, u\right)$。网络结构仍然使用 attention-DNN。

上述变体在两个数据集上的实验结果如 Table2 中的虚线下所示。将 TDM attentionDNN 与 TDM DNN 进行比较，在 UserBehavior 数据集中近 10% 的召回率改进表明，注意力模块起到了显著的作用。TDM product-DNN 的性能不如 TDM DNN 和 TDM attention-DNN，因为内积方式远没有神经网络交互形式强大。这些结果证明，引入先进的模型可以显著提高推荐性能。需要注意的是，与 TDM attentionDNN 相比，TDM attention-DNN-HS 的结果要差得多，因为 hierarchical softmax 的公式不适用于推荐问题。

**树的作用。** 树是所提出的TDM方法的关键组成部分。它不仅作为检索中使用的索引，还以粗到细的层次结构对数据集进行建模。第3.3节提到，直接进行细粒度的推荐比采用分层方式更困难。我们进行了实验证明这个观点。Figure 4 展示了分层树搜索（算法1）和暴力搜索（遍历相应级别的所有节点）的逐层召回率@200。在 UserBehavior 数据集上使用 TDM product-DNN 模型进行了实验，因为它是唯一一个可以使用暴力搜索的变体。高层级（第8、9级）中，暴力搜索略优于树搜索，因为那里的节点数量较小。但是，一旦某个层级中的节点数量增加，与暴力搜索相比，树搜索可以排除高层中的低质量结果，从而降低了低层问题的难度，使得树搜索可以获得更好的召回结果。这个结果表明，树结构中包含的层次信息有助于提高推荐的准确性。

**树的学习**。在第3.5节中，我们提出了树的初始化和学习算法。表4 给出了初始树和学习树之间的比较结果。从结果可以看出，使用学习到的树结构的训练模型明显优于初始模型。例如，在筛选互动类别的实验中，学习到的树的召回率从 4.15% 增加到 4.82%，相比于初始树，大大超过了 YouTube product-DNN 的 3.09% 和 item-CF 的 1.06%。为了进一步比较这两棵树，我们在图5中展示了 TDM attention-DNN 方法在训练迭代中的测试集损失和召回率曲线。从图5(a)可以看出，学习到的树结构具有更小的测试集损失。图5(a)和5(b)都表明，使用学习到的树模型收敛到更好的结果。上述结果证明，树的学习算法可以改善 item 的层次结构，进一步促进训练和预测的效果。

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Figure4.png)

**Figure 4**: The results of layer-wise Recall@200 in UserBehavior dataset. The ground truth in testing set is traced back to each node’s ancestors, till the root node.

![Table4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table4.png)

**Table 4**: Comparison results of different tree structures in UserBehavior dataset using TDM attention-DNN model (@200). Tree is initialized and learnt according to the algorithm described in Section 3.5.

### 5.5 Online Results

我们使用真实流量在淘宝展示广告平台中评估了提出的TDM方法。实验是在淘宝App首页的“猜你喜欢”栏目中进行的。我们使用了两个在线指标来衡量性能：点击率（CTR）和每千次展示收入（RPM）。具体细节如下：
$$
CTR=\frac{\# \ of \ clicks}{\# \ of \ impressions}, RPM=\frac{Ad \ revenue}{\# \ of \ impressions} * 1000
$$
在我们的广告系统中，广告主对给定的广告聚类进行竞价。大约有140万个聚类，每个广告聚类包含数百或数千个相似的广告。为了与现有系统保持一致，实验是以广告聚类的粒度进行的。对照组是混合逻辑回归[9]，仅从那些互动过的聚类中选择出优秀的结果，这是一个强有力的 baseline。由于系统中有许多阶段，如CTR预测[11、34]和排序[35]，如图1所示，将提出的 TDM 方法部署和评估在线上是一个巨大的项目，涉及到整个系统的链接和优化。到目前为止，我们已经完成了第一个TDM DNN版本的部署，并在线上评估了其改进效果。每个小流量占据了所有在线流量的5%。值得一提的是，同时运行着几种在线推荐方法。

它们从不同的角度生效，并将其推荐结果合并到后续阶段。TDM 只替换其中最有效的方法，而其他模块保持不变。小流量中使用 TDM 的平均指标提升率列在表5中。

![Table5](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Learning Tree-based Deep Model for Recommender Systems/Table5.png)

**Table 5**: Online results from Jan 22 to Jan 28, 2018 in Guess What You Like column of Taobao App Homepage.

如表5所示，TDM方法使点击率（CTR）提高了2.1%。这一改进表明所提出的方法能够为用户提供更准确的结果。另外，RPM指标增加了6.4%，这意味着TDM方法也可以为淘宝广告平台带来更多收益。TDM已经被部署到线上，我们相信上述改进只是一个庞大项目中的第一步，并且还有进一步改进的空间。

> 译注：ctr提升应该是应该是相对值，绝对值这实验岂不起飞了。

**预测效率**。TDM 使得在大规模推荐系统中使用先进的神经网络与 user 和 item进行交互成为可能，为推荐系统带来了新的视角。值得一提的是，尽管高级神经网络在推断时需要更多计算，但整个预测过程的复杂度不超过 $O(k * log |C| * t)$ ，其中 $k$ 是所需结果的大小，$|C|$ 是语料库的大小，$t$ 是网络单次前向传递的复杂度。在当前的 CPU/GPU 硬件条件下，这种复杂度上限是可以接受的，并且用户侧的特征在一次检索中可以在不同类型的节点之间共享，并且根据模型设计，一些计算可以共享。在淘宝展示广告系统中，部署的TDM DNN模型平均每次推荐的运行时间约为6毫秒。这个运行时间比后续的点击率预测模块要短，并且不是系统的瓶颈。

## 6 CONCLUSION

我们发现模型方法在从大规模语料库生成推荐时面临的主要挑战是计算量的问题。我们提出了一种基于树结构的方法，可以在大规模推荐中使用任意先进的模型来沿着树结构粗到细地推断用户的兴趣。除了训练模型外，还使用了一种树结构学习方法，证明了更好的树结构可以显著改善结果。一个可能的未来方向是设计更精细的树学习方法。我们进行了大量实验证明了所提出方法的有效性，无论是在推荐准确性还是新颖性方面都得到验证。此外，实证分析展示了该方法的工作原理和原因。在淘宝展示广告平台中，所提出的TDM方法已经在生产环境中部署，这既提高了商业效益，又改善了用户体验。