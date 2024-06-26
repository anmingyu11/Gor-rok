# Joint Optimization of Tree-based Index and Deep Model for Recommender Systems

## Abstract

在大规模的工业推荐系统中，由于庞大的语料库大小，通常会遇到计算问题。为了在响应时间限制下检索和推荐最相关的物品给用户，使用高效的索引结构是一种有效且实用的解决方案。之前的工作"Tree-based Deep Model (TDM)" [34]通过使用树索引极大地提升了推荐准确性。通过在树的层次结构中对物品进行索引，并训练一个满足树中类似最大堆属性的 user-node 偏好预测模型，TDM相对于语料库大小具有对数级的计算复杂度，从而可以在候选检索和推荐中使用任意先进的模型。

在基于树的推荐方法中，树索引和 user-node 偏好预测模型的质量主要决定了推荐准确性。我们认为，树索引和偏好模型的学习具有相互依赖性。因此，本文旨在开发一种方法来联合学习索引结构和用户偏好预测模型。在我们提出的联合优化框架中，索引和用户偏好预测模型的学习是在统一的评估度量下进行的。此外，我们还利用树索引层次结构提出了一种新的分层用户偏好表示方法。通过对两个大规模真实世界数据集进行实验评估，结果显示所提出的方法显著提高了推荐准确性。而在一个广告展示平台上进行的线上A/B实验结果也证明了该方法在生产环境中的有效性。

## 1 Introduction

推荐问题基本上是从整个语料库中检索出一组最相关或首选的物品，以满足每个用户请求。在大规模推荐实践中，算法设计需要在准确性和效率之间取得平衡。在包含数千万或上亿个物品的语料库中，需要线性扫描每个用户请求的每个物品的偏好分数的方法在计算上是不可行的。为了解决这个问题，通常使用索引结构来加速检索过程。在早期的推荐系统中，基于物品的协同过滤(Item-CF)结合倒排索引是克服计算障碍的流行解决方案[18]。然而，候选集的范围是有限的，因为只有那些与用户历史行为相似的物品最终可以被推荐。

近年来，向量表示学习方法[27, 16, 26, 5, 22, 2]得到了广泛研究。这种方法可以学习user和item的向量表示，其内积表示 user 对 item 的偏好。对于使用基于向量表示的方法的系统，推荐集生成等价于 $k$ 最近邻(kNN)搜索问题。基于量化的索引[19, 14]用于近似 kNN 搜索已被广泛采用以加速检索过程。然而，在上述解决方案中，向量表示学习和 kNN 搜索索引构建分别针对不同的目标进行优化。这种目标的差异导致了次优的向量表示和索引结构[4]。更重要的问题是，对向量 kNN 搜索索引的依赖需要使用内积形式的用户偏好建模，这限制了模型的能力[10]。像 Deep Interest Network [32]、Deep Interest Evolution Network [31] 和 xDeepFM [17]这样在用户偏好预测方面已被证明有效的模型无法用于生成推荐候选集。

为了突破内积形式的限制，并使任意先进的用户偏好模型能够在整个语料库中进行候选项检索，之前的工作"Tree-based Deep Model (TDM)" [34]创造性地使用树结构作为索引，极大地提高了推荐准确性。TDM使用树索引来组织物品，树中的每个叶节点对应一个item。类似于最大堆，TDM假设每个user-item偏好等于用户对该节点的所有子节点偏好中的最大值。在训练阶段，会训练一个user-node偏好预测模型来拟合最大堆形式的偏好分布。与基于向量 k 最近邻（kNN）搜索的方法不同，在 TDM 中，索引结构并不要求用户偏好建模为内积形式。因此，对偏好模型的形式没有限制。在预测阶段，训练得到的模型给出的偏好分数用于在树索引中进行逐层的 beam search，以检索候选项。树索引中 beam search 的时间复杂度与语料库大小呈对数关系，并且对模型结构没有限制，这是使先进的用户偏好模型在推荐中检索候选项可行的先决条件。

在基于kNN搜索的方法和 Tree-based的方法中，索引结构扮演着不同的角色。在基于kNN搜索的方法中，首先学习user和item的向量表示，然后构建向量搜索索引。而在基于树的方法中，树索引的层次结构也会影响检索模型的训练。因此，如何同时学习树索引和用户偏好模型是一个重要的问题。树结构方法也是极端分类文献中的一个研究热点[29, 1, 24, 11, 8, 25]，有时被认为与推荐系统相同[12, 25]。在现有的Tree-based的方法中，树结构是为了在样本或标签空间中获得更好的层次结构而学习的。然而，在树学习阶段，样本或标签分区任务的目标与准确的推荐目标并不完全一致。索引学习和预测模型训练目标的不一致性导致整个系统处于次优状态。为了解决这个挑战，促进树索引和用户偏好预测模型的更好协作，我们专注于开发一种同时学习树索引和用户偏好预测模型的方法，通过优化统一的评估度量。

本文的主要贡献如下：

1. 提出了一种联合优化框架，学习Tree-based的推荐中的树索引和用户偏好预测模型，优化了统一的评估度量，即用户偏好预测的准确性；
2. 我们证明了所提出的树学习算法等价于二部图的加权最大匹配问题，并给出了一个近似算法来学习树结构；
3. 我们提出了一种新颖的方法，更好地利用树索引生成分层用户表示，可以帮助学习更准确的用户偏好预测模型；
4. 我们验证了树索引学习和分层用户表示都可以提高推荐准确性，这两个模块甚至可以相互改进，从而实现更显著的性能提升。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Figure1.png)

**Figure 1**: Tree-based deep recommendation model. (a) User preference prediction model. We firstly hierarchically abstract the user behaviors with nodes in corresponding levels. Then the abstract user behaviors and the target node together with the other feature such as the user profile are used as the input of the model. (b) Tree hierarchy. Each item is firstly assigned to a different leaf node with a projection function $π(·)$. In retrieval stage, items that assigned to the red nodes in the leaf level are selected as the candidate set.

## 2 Joint Optimization of Tree-based Index and Deep Model

本节首先对 TDM [34]进行简要回顾，以使本文自洽。然后，我们提出 Tree-based 索引和深度模型的联合学习框架。在最后一个子节中，我们详细说明了模型训练中使用的层次化用户偏好表示方法。

### 2.1 Tree-based Deep Recommendation Model

在具有大规模语料库的推荐系统中，如何高效地检索候选集是一个具有挑战性的问题。TDM 使用树作为索引，并在树中提出了类似最大堆的概率公式。对于 $l$ 层的非叶节点 $n$，其用户偏好概率由以下公式计算得到：
$$
p^{(l)}(n \mid u)=\frac{\max _{n_c \in\left\{n^{\prime} s \text { children in level } l+1\right\}} p^{(l+1)}\left(n_c \mid u\right)}{\alpha^{(l)}}
\\
(1)
$$
其中 $p^{(l)}(n|u)$ 是用户 $u$ 对节点 $n$ 偏好的 ground truth。$\alpha^{(l)}$ 是一个层的归一化项。上述公式的意思是节点上的真实用户-节点概率等于其子节点的最大用户-节点概率除以归一化项。因此，$l$ 层中的前 $k$ 个节点必须包含在层级 $l-1$ 中前 $k$ 个节点的子节点中，而且可以通过逐层自上而下地进行检索来限制检索top $k$ 个叶子item，而不会失去准确性。基于这一点，TDM 将推荐任务转化为一个分层检索问题，其中候选项从粗到细逐步选择。TDM 的候选生成过程如图1所示。

每个 item 首先分配到树 $\mathcal{T}$ 中的一个叶节点。如图1(b)所示，采用一种层次化的 beam search 策略。对于层级 $l$，仅对于上一层级 $l-1$ 中分数排名前 $k$ 的节点的子节点进行评分和排序，以选择层级 $l$ 中的 $k$ 个候选节点。该过程持续进行，直到达到 $k$ 个叶子 item 为止。将用户特征与候选节点结合起来，作为预测模型 $\mathcal{M}$（例如全连接网络）的输入，可以得到偏好概率，如图1(a)所示。通过树索引，相对于语料库的大小，用户请求的整体检索复杂度从线性降低为对数级别，并且不限制偏好模型的结构。这使得 TDM 打破了由向量 kNN 搜索索引带来的用户偏好建模形式的限制，并使任意高级深度模型能够从整个语料库中检索候选项，从而大大提高了推荐准确性。

### 2.2 Joint Optimization Framework

我们得到一个包含 $n$ 个样本的训练集，表示为 ${(u^{(i)}, c^{(i)})}^n_{i=1}$，其中第 $i$ 对 $(u^{(i)}, c^{(i)})$ 表示用户 $u^{(i)}$ 对目标 item $c^{(i)}$ 感兴趣。对于 $(u^{(i)}, c^{(i)})$，树 $\mathcal{T}$ 确定了预测模型 $\mathcal{M}$ 应选择的路径，以实现用户 $u^{(i)}$ 对 $c^{(i)}$ 的预测。我们提出在全局损失函数下联合学习 $\mathcal{M}$ 和 $\mathcal{T}$。正如我们将在实验中看到的，联合优化 $\mathcal{M}$ 和 $\mathcal{T}$ 可以提高最终的推荐准确性。

给定一个user-item 对 $(u, c)$，我们将 $p(π(c)|u; π)$ 表示为用户 $u$ 对叶子节点 $π(c)$ 的偏好概率，其中 $π(·)$ 是一个映射函数，将 item 映射到树 $\mathcal{T}$ 中的叶节点。请注意，$π(·)$ 完全确定了树 $\mathcal{T}$，如图1(b)所示。优化 $\mathcal{T}$ 实际上就是优化 $π(·)$。模型 $\mathcal{M}$ 根据参数 $θ$ 估计 user-node 偏好概率 $\hat{p}(π(c)|u;θ,π)$。如果 user-item 对 $(u, c)$ 是正样本，则多分类中我们有真实的偏好概率 $p(π(c)|u; π) = 1$ 。根据最大堆的性质，所有 $π(c)$ 的祖先节点的用户偏好概率，即 ${p(b_j(\pi(c))|u;\pi)}^{l_{max}}_{j=0}$，也应该为 $1$，其中 $b_j(·)$ 是从一个节点映射到其在第 $j$ 级的祖先节点的投影函数，$l_{max}$ 是 $\mathcal{T}$ 中的最大层级。为了拟合这样的 user-item 偏好分布，我们可以制定全局损失函数： 

$$
\mathcal{L}(\theta, \pi)=-\sum_{i=1}^n \sum_{j=0}^{l_{\max }} \log \hat{p}\left(b_j\left(\pi\left(c^{(i)}\right)\right) \mid u^{(i)} ; \theta, \pi\right)
\\ (2)
$$
我们将对所有正训练样本及其祖先 user-node 对的预测 user-node 偏好概率的负对数之和作为全局经验损失。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Alg1.png)

为了克服优化 $π(·)$ 的问题，我们提出了一个联合学习框架，如算法1所示。它交替地优化损失函数(2)关于用户偏好模型和树形层次结构。模型训练和树形学习中训练损失的一致性促进了框架的收敛。实际上，如果模型训练和树形学习都减小了(2)的值，那么算法1肯定会收敛，因为 ${\mathcal{L}(θ_t,π_t)}$ 是一个递减的序列，并且下界为0。在模型训练中，$min_θ\mathcal{L}(θ, π)$ 是为所有层级学习user-node偏好模型，可以通过流行的神经网络优化算法（如SGD[3]、Adam[15]）来解决。在归一化用户偏好（模型）配置中[5, 2]，由于节点数量随着节点层级的增加呈指数增长，为了避免通过采样策略计算归一化项，可以使用NCE噪声对比估计[7]来估计 $\hat{p}(b_j(π(c))|u;θ,π)$。树形学习的任务是给定 $θ$ 解决 $max_π −L(θ, π)$。$max_π −L(θ, π)$ 等于由语料库 $\mathcal{C}$ 中的 item 和 $\mathcal{T}$ 的叶节点组成的二分图的最大加权匹配问题。详细的证明在补充材料中给出。 

传统的指派问题算法，如经典的匈牙利算法，在处理大规模语料库时往往计算复杂度过高。即使对于朴素的贪心算法，它会选择具有最大权重的未分配边缘，但需要提前计算和存储一个大的权重矩阵，这是不可接受的。为了解决这个问题，我们提出了一种分段树学习算法。

我们不直接将 item 分配给叶节点，而是从根到叶逐步完成此过程。给定一个映射函数 $π$ 和语料库中的第 $k$ 个 item $c_k$，定义
$$
\mathcal{L}_{c_k}^{s, e}(\pi)=\sum_{(u, c) \in \mathcal{A}_k} \sum_{j=s}^e \log \hat{p}\left(b_j(\pi(c)) \mid u ; \theta, \pi\right)
$$
其中，$A_k = \{(u^{(i)} , c^{(i)}) | c^{(i)} = c_k\}^n_{i=1}$ 是目标 item 为 $c_k$ 的训练样本集合，$s$ 和 $e$ 分别是起始层和结束层。首先，我们针对 $π$ 最大化 $\sum_{k=1}^{|\mathcal{C}|} \mathcal{L}_{c_k}^{1, d}(\pi)$ ，这等价于将所有 item 分配给 $d$ 层的节点。对于具有最大层 $l_{max}$ 的完全二叉树 $\mathcal{T}$，第 $d$ 层的每个节点最多分配 $2^{l_{max}−d}$ 个 item。这也是一个最大匹配问题，可以通过贪心法高效解决，因为如果选择得当（例如对于 $d=7$，可能的位置数为 $2^d=128$），则每个 item 的可能位置大大减少。将在此步骤中找到的最优投影表示为 $π^∗$。

然后，我们依次在约束条件 $\forall c \in \mathcal{C}, b_d(\pi(c))=b_d\left(\pi^*(c)\right)$ 下最大化 $\sum_{k=1}^{|\mathcal{C}|} \mathcal{L}_{c_k}^{d+1,2 d}(\pi)$ ，这意味着保持每个 item 对应的祖先节点在第 $d$ 层不变。递归停止直到每个 item 都分配给一个叶节点。算法细节见算法2。

在算法2的第5行，我们使用一种贪婪算法和平衡策略来解决子问题。首先，将每个item $c \in \mathcal{C}_{n_i}$ 分配给在层级 $l$ 中具有最大权重 $\mathcal{L}_c^{l-d+1, l}(\cdot)$ 的节点 $n_i$ 的子节点。 然后，应用一个平衡过程以确保每个子节点被分配的item数不超过 $2^{l_{max−l}}$。其中，$l_{max}$ 是树中允许的最大层级。 算法2的详细实现可以在附录材料中找到。它提供了如何使用贪婪算法和平衡策略解决子问题的逐步描述。为了获取更具体的信息和深入理解，建议参考附录材料。

> 译注：这块描述过于形式化，令人难懂，还是直接看代码的好：[jtm-github](https://github.com/massquantity/dismember/tree/main/jtm/src)

![Alg2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Alg2.png)

### 2.3 Hierarchical User Preference Representation

如第2.1节所示，TDM是一种分层检索模型，可以从粗到细地生成候选项。在检索过程中，通过用户偏好预测模型 $\mathcal{M}$ 进行逐层自顶向下的 beam search，并通过树索引获得候选项。因此，$\mathcal{M}$ 在每个层级上的任务是异构可并发的。基于此，为了提高推荐准确性，需要为$\mathcal{M}$ 提供特定于层级的输入。

一系列相关工作[30, 6, 18, 16, 32, 33, 34]表明，用户的历史行为在预测用户兴趣方面起着关键作用。然而，在我们的 Tree-based 的方法中，我们可以以一种新颖有效的方式扩大这一关键作用。给定用户行为序列$c = \{c_1, c_2, · · · , c_m\}$，其中 $c_i$ 是用户交互的第 $i$ 个item，我们建议在 $l$ 层中使用 $c^l = \{b_l(π(c_1)), b_l(π(c_2)), · · · , b_l(π(c_m))\}$ 作为用户在该层的行为特征。如图1(a)所示，$c^l$ 与目标节点以及其他可能的特征（如用户画像）一起作为 $l$ 层中 $\mathcal{M}$ 的输入，用于预测用户-节点偏好。此外，由于每个节点或 item 都是一种 one-hot 特征，我们遵循常见的方法将它们 embedding 到连续特征空间中。通过这种方式，用户交互的 item 的祖先节点被用作分层用户偏好表示。

总体上，分层表示带来了两个主要好处： 

1. 层级独立性。在常规方法中，不同层之间共享的 item embedding 会在训练用户偏好预测模型 $\mathcal{M}$ 时引入噪声，因为不同层具有不同的目标。一种显式解决方案是为每个层的 item 附加一个独立的 embedding。然而，这样会大大增加参数数量，使系统难以优化和应用。所提出的分层表示使用相应层级的节点 embedding 作为 $\mathcal{M}$ 的输入，在训练中实现了层级独立性而无需增加参数数量。 
2. 精确描述。$\mathcal{M}$ 通过树结构分层生成候选项。随着检索的层级的增加，每个级别中的候选节点描述了从粗到细的最终推荐项，直到达到叶节点级别。所提出的分层用户偏好表示把握了检索过程的本质，并使用相应层级的节点精确描述用户行为，从而通过减少过于详细或过于粗略的描述来提高用户偏好的可预测性。例如，在上层中，$\mathcal{M}$ 的任务是粗略选择候选集，并且在训练和预测中使用相同上层的同质节点 embedding 对用户行为进行粗略描述。 

第3节和补充材料中的实验研究将展示所提出的分层表示的显著有效性。

> 译注：这段的意思其实就是对于每个层级的预测，用户的行为序列特征中对应的item转化为对应的node作为输入去预测对应的node，保持特征和label的一致性。起了个名字叫 hierarchical user representation。

## 3 Experimental Study

1. Amazon Books3[20, 9]：这是一个由亚马逊的产品评论组成的用户-图书评论数据集。我们使用其中最大的子集 Books 进行实验。
2. UserBehavior4[34]：这是淘宝用户行为数据的一个子集。这两个数据集都包含数百万个物品，并且数据以 user-item 交互的形式组织：每个 user-item 交互包括用户ID、物品ID、类别ID和时间戳。对于上述两个数据集，我们只保留至少有10个交互的用户。

为了评估所提出的框架的性能，我们与以下方法进行比较：

- **Item-CF**[28]：这是一种基础的协同过滤方法，广泛用于个性化推荐，特别是在大规模语料库中[18]。
- **YouTube product-DNN**[5]：这是YouTube视频推荐中使用的一种实用方法。它是基于向量kNN搜索的方法的代表性工作。学习到的用户和物品向量表示的内积反映了用户的偏好。在预测中，我们使用精确的kNN搜索来检索候选项。
- **HSM**[21]：这是层次softmax模型。它采用层级条件概率的乘积来获得归一化的物品偏好概率。
- **TDM**[34]：这是一种基于树的深度推荐模型。它使得任意高级模型可以使用树索引来检索用户兴趣。我们使用了TDM的基本DNN版本，没有进行树学习和注意力机制。
- **DNN**：这是一个没有树索引的 TDM 变体。唯一的区别在于，它直接学习用户-物品偏好模型，并在预测中线性扫描所有物品以获取前 K 个候选项。在在线系统中，这样的计算是不可行的，但在离线比较中是一个强大的基准方法。
- **JTM**：这是树索引和用户偏好预测模型的联合学习框架。JTM-J 和 JTM-H 是两个变体。JTM-J 同时优化树索引和用户偏好预测模型，但没有使用第2.3节中提出的分层表示。JTM-H采用分层表示，但使用固定的初始树索引，没有进行树学习。

根据TDM [34]的方法，我们将用户分为互不相交的训练集、验证集和测试集。训练集中的每个 user-item 交互都被视为一个训练样本，而交互之前的用户行为则作为对应的特征。对于验证集和测试集中的每个用户，我们将时间线上的前半部分行为作为已知特征，后半部分作为真实值。

利用TDM的开源工作[5]，我们在阿里巴巴的深度学习平台 X-DeepLearning (XDL) 上实现了所有方法。HSM、DNN 和 JTM 采用与 TDM 相同的用户偏好预测模型。除了 Item-CF 外，我们对所有方法都采用了负采样，并使用相同的负采样比例。对于每个训练样本，在Amazon Books中采样了100个负样本，在UserBehavior中采样了200个负样本。HSM、TDM 和 JTM 在训练过程中需要预先初始化一颗树。遵循 TDM 的方法，我们使用类别信息来初始化树结构，其中同一类别的物品在叶节点级别聚合。有关数据预处理和训练的更多详细信息和代码，请参考补充材料。 Precision（精确率）、Recall（召回率）和 F-Measure（F值）是三个常用的评估指标，我们使用它们来评估不同方法的性能。

对于用户 $u$，假设 $\mathcal{P}_u(|\mathcal{P}_u| = \mathcal{M})$ 表示召回的集合，$\mathcal{G}_u$ 表示真实值集合。这三个指标的计算公式如下： 
$$
\text { Precision@M(u)= } \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{P}_u\right|}, \text { Recall@M(u)= } \frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|}
\\
\text { F-Measure@ } M(u)=\frac{2 * \text { Precision@ } M(u) * \text { Recall@ } M(u)}{\text { Precision@ } M(u)+\text { Recall@ } M(u)}
$$
每个指标的结果都是在测试集中所有用户上进行平均，并且列出的数值是五次不同运行的平均值。

### 3.2 Comparison Results

表格1展示了两个数据集中所有方法的结果。它清楚地显示了我们提出的 JTM 在所有指标上优于其他基线模型。与两个数据集中之前最好的模型DNN相比，JTM在Amazon Books和UserBehavior数据集中分别实现了45.3%和9.4%的召回率提升。

![table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/table1.png)

**Table 1**: Comparison results of different methods in Amazon Books and UserBehavior ($M$ = 200).

正如前面提到的，在在线系统中，虽然DNN在计算上难以处理，但它是离线比较中一个非常强大的基线模型。DNN和其他方法的比较结果在许多方面都提供了有益的启示。

首先，YouTube product-DNN和DNN之间的差距显示了内积形式的局限性。这两种方法之间唯一的区别在于，YouTube product-DNN使用用户和物品向量的内积来计算偏好分数，而DNN使用一个全连接网络。这样的改变带来了明显的改进，验证了先进神经网络在内积形式上的有效性。

接下来，TDM 表现不如具有普通但未经优化的树结构的DNN。树结构在训练和预测过程中都发挥作用。用户节点样本沿着树生成，以适应最大堆样式的偏好分布，并且在预测时使用树索引进行逐层 beam search。如果没有定义良好的树层次结构，用户偏好预测模型可能会收敛到一个次优版本，生成的样本可能会混淆，并且可能会在非叶层级丢失目标，从而返回不准确的候选集。特别是在像 Amazon Books 这样的稀疏数据集中，树层次结构中每个节点的学习 embedding 不够可区分，因此 TDM 的表现不如其他基准模型。这种现象说明了树的影响和树学习的必要性。另外，HSM 的结果比 TDM 差得多。这个结果与 TDM[34]中报告的结果一致。在处理大规模语料库时，由于逐层概率相乘和 beam search 的结果，HSM 不能保证最终的召回集是最优的。

通过树索引和用户偏好模型的联合学习，JTM 在两个数据集上在所有指标上均优于 DNN，并且检索复杂度更低。JTM 获得了更精确的用户偏好预测模型和更好的树层次结构，从而导致更好的物品集选择。层次化的用户偏好表示减轻了上层数据稀疏性的问题，因为用户行为特征的特征空间要小得多，而样本数却相同。它以逐层方式帮助模型训练，降低了噪声在不同层级之间的传播。此外，树层次结构的学习使得相似的物品聚集在叶节点级别，这样内部层级模型可以获得更一致和明确分布的训练样本。受益于以上两个原因，JTM提供了比DNN更好的结果。

表1中虚线下方的结果显示了JTM中每个部分的贡献及其联合性能。以召回率为例。与UserBehavior中的TDM相比，树学习和用户偏好的层次化表示分别带来了0.88%和2.09%的绝对增益。此外，在统一目标下，这两个优化的结合使召回率提升了3.87%。在Amazon Books中也观察到类似的增益。以上结果清楚地展示了层次化表示和树学习以及联合学习框架的有效性。

**迭代联合学习的收敛性** 树的层次结构决定了样本生成和搜索路径。一个合适的树对于模型的训练和推断都会有很大的好处。图2比较了在TDM [34]中提出的基于聚类的树学习算法和我们提出的联合学习方法。为了公平起见，两种方法都采用了分层用户表示。

由于所提出的树学习算法与用户偏好预测模型具有相同的目标，在结果上它具有两个优点：

1. 它可以稳定地收敛到最优的树；
2. 最终的推荐准确率高于基于聚类的方法。从图2中可以看出，所有三个指标的结果都在迭代过程中逐渐增加。此外，该模型在两个数据集中都稳定地收敛，而基于聚类的方法最终过拟合。以上结果从实证上证明了迭代联合学习的有效性和收敛性。

一些仔细的读者可能已经注意到，在前几次迭代中，聚类算法的表现优于 JTM。原因是 JTM 中的树学习算法采用了一种懒惰策略，即每次迭代尽量降低树结构的改变程度（详细信息请参见补充材料）。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Joint Optimization of Tree-based Index and Deep Model for Recommender Systems/Figure2.png)

**Figure 2**: Results of iterative joint learning in two datasets (M = 200). 2(a), 2(b), 2(c) are results in Amazon Books and 2(d), 2(e), 2(f) shows the performance in UserBehavior. The horizontal axis of each figure represents the number of iterations.

### 3.3 Online Results

我们还在生产环境中评估了提出的 JTM 模型：淘宝App首页的“猜你喜欢”栏目的展示广告场景。我们使用点击率（CTR）和千次展示收入（RPM）来衡量性能，这些是关键绩效指标。

定义如下：
$$
\mathrm{CTR}=\frac{\# \text { of clicks }}{\# \text { of impressions }}, \mathrm{RPM}=\frac{\text { Ad revenue }}{\# \text { of impressions }} * 1000
$$
在该平台上，广告主对广告簇、物品、店铺等多个细分领域进行竞价。所有细分领域中同时运行的推荐方法会生成候选集，并将它们的组合传递给后续阶段，例如 CTR 预测[32, 31, 23]、排序[33, 13]等。比较基准是所有运行中推荐方法的组合。为了评估 JTM 的有效性，我们部署 JTM 来替代平台上物品细分领域的主要候选生成方法之一，即 Item-CF。TDM 的评估方式与 JTM 相同。处理的数据集包含数千万个物品。每个桶有2%的在线流量，考虑到整体页面访问请求量，这已经足够大了。Table2 列出了这两个主要在线指标的提升情况。CTR增长了11.3%，说明 JTM 能够推荐更准确的物品。而 RPM 则提高了12.9%，表明 JTM 可以为平台带来更多收入。

## 4 Conclusion

推荐系统在视频流媒体和电子商务等各种应用中起着关键作用。本文针对大规模推荐中的一个重要问题进行了研究，即如何在全局目标下优化用户表示、用户偏好预测和索引结构。据我们所知，JTM 是第一个提出了统一框架来整合这三个关键因素优化的工作。该框架引入了树索引和用户偏好预测模型的联合学习方法。根据树索引建立了一种新颖的层次化用户表示，并在全局损失函数下交替优化树索引和深度模型。在线和离线实验结果都显示了所提出框架在大规模推荐模型中的优势。