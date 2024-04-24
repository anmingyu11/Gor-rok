# Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation

## ABSTRACT

长期以来，针对推荐系统的基于模型的方法已经得到广泛研究。现代推荐系统通常采用以下两种方法：1) 使用表示学习模型来 user-item 偏好，将其表示为 embedding 表示之间的距离；2) 使用基于 embedding 的近似最近邻（ANN）搜索来解决大规模数据集引入的效率问题。虽然 embedding 检索提供了高效的检索功能，但由于 user-item 偏好度量形式受限于 embedding 表示之间的距离，这也限制了模型的容量。然而，对于其他更精确的 user-item 偏好度量方法（例如直接从深度神经网络获得的偏好评分），由于缺乏高效的检索方法，计算上是难以处理的，并且对所有 user-item 对进行穷举搜索是不可行的。

在本文中，我们提出了一种新颖的方法，将近似最近邻（ANN）搜索扩展到任意匹配函数，例如深度神经网络。我们的主要思想是在从所有 item 构建的相似性图中，利用匹配函数进行贪婪搜索。为了解决图构建的相似性度量和 user-item 匹配函数异质的问题，我们提出了可插拔的对抗训练任务，以确保具有任意匹配函数的图搜索能够达到相当高的精度。在开源数据集和工业数据集上的实验结果表明了我们方法的有效性。该方法已经完全部署在淘宝展示广告平台，并带来了相当可观的广告收入增长。我们还总结了在部署过程中的详细经验。

在本文中，我们提出了一种新颖的方法，将近似最近邻（ANN）搜索扩展到任意匹配函数，例如深度神经网络。我们的主要思想是在从所有 item 构建的相似性图中，利用匹配函数进行贪婪搜索。为了解决图构建的相似性度量和 user-item 匹配函数异质的问题，我们提出了可插拔的对抗训练任务，以确保具有任意匹配函数的图搜索能够达到相当高的精度。在开源数据集和工业数据集上的实验结果表明了我们方法的有效性。该方法已经完全部署在淘宝展示广告平台，并带来了相当可观的广告收入增长。我们还总结了在部署过程中的详细经验。

## 1 INTRODUCTION

不断增长的可用信息量对现代推荐系统提出了巨大挑战。为了应对信息爆炸，现代推荐系统通常采用多阶段级联架构，主要包括候选生成和排序两个阶段。在候选生成阶段，也被称为匹配阶段，从一个非常庞大的语料库中检索出成千上万个目标，然后在排序阶段，根据用户的偏好对这些检索出的目标进行排序。值得注意的是，在实际系统中，由于计算资源和延迟的限制，无法通过顺序扫描整个语料库来解决候选生成问题，尤其是面对大规模的语料库时。

为了避免扫描整个语料库带来的计算成本过高，embedding-based的retrival（EBR）已经在推荐系统中盛行多年，因其简单和高效的特点而受到青睐[13, 15]。然而，EBR无法很好地建模 user-item 偏好的复杂结构。许多研究已经表明，更复杂的模型通常具有更好的泛化性能[11, 21, 32]。研究人员一直努力开发技术来解决使用更复杂模型进行大规模检索的问题。为了克服计算瓶颈并从任意先进的模型中受益，最近提出了通过索引来规范总计算成本的思想。这些方法[8, 33–35]通常具有可学习的索引，并遵循期望最大化（EM）类型的优化范式，在深度模型和索引之间交替更新。因此，深度模型和 beam search 可以以与语料库大小相关的次线性复杂度从大规模语料库中检索相关项。尽管这些端到端方法可以将深度模型引入到大规模检索中，但有两个方面不能忽视：1) 对于大规模数据，索引和模型的联合训练需要昂贵的训练预算，包括训练时间和计算资源；2) 索引结构的内部节点（如TDMs中的非叶节点[33–35]和DR中的路径节点[8]）使得难以利用 item 的附加信息。

本研究通过以轻量级的方式使用任意先进的模型来解决大规模检索问题，即神经网络近似最近邻搜索（NANN）。具体而言，我们利用深度模型作为 greedy walker，在模型训练后构建的相似性图中进行探索。采用解耦的范式可以大大释放端到端方法的联合训练预算。此外，深度模型遍历的相似性图不包含内部节点，这有助于利用候选项的附加信息。为了提高图搜索的效率和效果，在我们的NANN框架中创造性地提出了启发式检索方法和辅助训练任务。本文的主要贡献总结如下：

- 我们提出了一个统一且轻量级的框架，可以将任意先进的模型引入到大规模ANN检索中作为匹配函数。基本思想是利用匹配函数进行相似性图搜索。
- 为了使图搜索中的计算成本和延迟可控，我们提出了一种启发式检索方法称为Beam-retrieval，它可以在较少的计算量下获得更好的结果。我们还在模型训练中提出了一个辅助对抗任务，可以显著减轻相似性度量异质性的影响，并提高检索质量。
- 我们在公开可访问的基准数据集和实际行业数据集上进行了广泛的实验，结果表明所提出的NANN是在神经相似性度量下ANN搜索的优秀经验性解决方案。此外，NANN已经完全部署在淘宝展示广告平台上，并带来了3.1%的广告收入改善。
- 我们详细描述了NANN在淘宝展示广告平台上的实际部署经验。该部署及其相应的优化是基于Tensorflow框架[1]进行的。我们希望我们在开发这样一个轻量级但有效的大规模检索框架方面的经验能够有助于将NANN轻松应用于其他场景。

## 2 RELATED WORK

下面，让我们用 $\mathcal{V}$ 和 $\mathcal{U}$ 分别表示物品集合和用户集合。在推荐系统中，我们努力为每个用户 $𝑢 ∈ \mathcal{U}$ 从一个大规模的语料库 $\mathcal{V}$ 中检索一组相关的物品 $\mathcal{B}_u$。数学上，我们可以表示为： 
$$
\mathcal{B}_u=\underset{v \in \mathcal{V}}{\operatorname{argTopk}} s(v, u) \\ (1)
$$
其中 $𝑠(𝑣, 𝑢)$ 是相似度函数。该公式描述了推荐系统的目标，即根据相似度函数 $s(v, u)$，找出与用户 $u$ 相似度最高的前 $k$ 个物品，并将它们作为集合 $\mathcal{B}_u$ 返回。

**图搜索**。图搜索以其卓越的效率、性能和灵活性（在相似度函数方面）而广为人知，是最近邻搜索（NNS）的一种基本且强大的方法。图搜索的理论基础是由相似度函数 $𝑠(𝑣, 𝑢)$ 定义的 s-Delaunay图。之前的研究[20]表明，当 $𝑘 = 1$ 并且 $𝑢, 𝑣 ∈ \mathbb{R}_𝑑$ 时，通过从 $\mathcal{V}$ 构建的 s-Delaunay图上的某种贪婪行走，可以找到公式（1）的精确解，其中 $𝑠(𝑣, 𝑢) = −||𝑣 −𝑢||_2$ 。更一般地，许多现有工作试图将这个结论推广到非度量情况，如内积[2, 22, 24, 25]、Mercer核[5, 6]和Bregman散度[3]等。此外，研究人员还尝试近似 s-Delaunay图，因为使用大规模语料库构建完美的 s-Delaunay 图是不可行的。Navigable Small World (NSW) [17] 提出了一种极大地优化图的构建和搜索过程的方法。此外，Hierarchical NSW (HNSW) [18]通过逐步构建多层结构从邻近图中提供了最先进的NNS技术。我们的方法将使用HNSW算法，尽管其他基于图的NNS方法也可以工作。

**基于深度模型的检索**。近年来，基于模型，尤其是基于深度模型的方法在大规模检索中成为一个活跃的研究领域。在推荐系统中，许多工作致力于采用端到端的方式同时训练索引和深度模型。树结构方法，包括TDM [34]、JTM [33]和BSAT [35]，将索引构建为一棵树结构，并从粗到细建模用户兴趣。深度检索（DR）[8]使用可学习的路径对所有候选物品进行编码，并通过训练物品路径以及深度模型来最大化相同的目标函数。这些方法通过遍历索引来预测用户兴趣，并通过 beam search 实现了与语料库大小呈亚线性的计算复杂度。然而，这些方法通常需要额外的内部节点来参数化可学习的索引，这在利用物品的附加信息时会带来困难。此外，由于存在可学习的索引和EM类型的训练范式，这些端到端的方法还需支付额外的模型参数和训练时间。

**基于深度模型的图搜索** 一些工作已经尝试将相似度函数扩展到深度神经网络。与我们的工作最接近的是SL2G[28]，它通过l2距离构建索引图，并使用深度神经网络遍历训练后的图。然而，他们的方法只能推广到具有凸性或拟凸性的 $𝑠(𝑣, 𝑢)$。对于非凸相似度函数（深度神经网络的最常见情况），他们直接应用SL2G而没有进行调整。另一个工作[19]在 item 对之间定义了没有相似度的索引图。他们利用了相关的item 应该对于同一用户具有接近的 $𝑠(𝑣, 𝑢)$的思想，并通过子样本 $\left\{s(v, u_i) \mid j=1, \ldots, m\right\}$ 来表示候选item。然而，在实践中对于大规模语料库 $\mathcal{V}$ 来说，很难采样出一个代表性的集合。

## 3 METHODOLOGY

在本节中，我们首先在第3.1节中提供了关于EBR和基于模型的检索的总体框架，包括模型架构和训练范式。然后，我们分别在第3.2节和第3.3节介绍了所提出的NANN的相似性图构建和基于图的检索方法。在理解这些初步概念之后，我们在第3.4节中详细介绍可插入的对抗训练任务，并演示它如何消除图构建和基于模型匹配函数之间的相似性度量差距。

### 3.1 General Framework

#### 3.1.1 基于 embedding 的检索综述

我们提出的方法可以被普遍视为EBR框架的扩展，其中我们将简单的相似度度量推广到任意的神经网络模型中。因此，为了清晰起见，我们简要回顾一下EBR框架。EBR采用双塔模型架构，其中一侧用于编码用户的个人资料和行为序列，另一侧用于编码物品。数学上表示为：
$$
\mathbf{e}_u=\mathrm{NN}_u\left(\mathbf{f}_u\right), \mathbf{e}_v=\mathrm{NN}_v\left(\mathbf{f}_v\right) \\(2)
$$
其中两个深度神经网络 $NN_𝑢$ 和 $NN_𝑣$（即用户网络和物品网络）将$f_𝑢$和$f_𝑣$的输入编码为分别属于$\mathbb{R}^𝑑$的稠密向量 $e_𝑢$ 和 $e_𝑣$。用户-物品偏好形式可以由语义 embedding 的内积表示，即 $e^𝑇_𝑢e_𝑣$。由于计算整个大规模语料库的分区函数的困难性，通常使用基于候选采样的准则，如噪声对比估计（NCE）[10] Sampled-softmax [14]来训练EBR模型。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure1.png)

> **图1**: 通用框架。在左侧部分，深度模型包含三个基本分支，即用户网络、目标注意力网络和物品网络。用户网络负责对用户的画像进行建模。我们采用注意机制来灵活地学习用户行为序列和目标物料之间的交互。物料的信息通过物品网络来学习。在右侧部分，我们根据物料网络的输出 $e_𝑣$ 之间的 $𝑙_2$ 距离近似定义的Delaunay图，采用HNSW算法进行表示。请注意，在物品网络分支中，在线推断从$e_𝑣$开始。

#### 3.1.2 模型架构

与大规模检索中的传统EBR方法相比，NANN通过更复杂的架构，包括用户网络、目标注意力网络和物料网络，大大提升了模型容量，如图1所示。换句话说，我们用一个更普适和表达能力更强的 $𝑠(𝑣, 𝑢)$ 替代了内积 $e^𝑇_𝑢e_𝑣$。然而，基于深度神经网络的广义形式 $𝑠(𝑣, 𝑢)$ 也给我们带来了理论和实践上的挑战：1）如何合理地将基于图的检索推广到任意非线性和非凸的 $𝑠(𝑣, 𝑢)$ ；2）如何将基于图的检索与复杂的深度模型集成，并以轻量高效的方式部署计算密集型的检索框架。

#### 3.1.3 训练

与EBR相同，我们将计算复杂的问题转化为通过NCE估计二分类器的参数的问题。正样本来自用户 $𝑢$ 与物品 $𝑣$ 之间真实分布的交互，而负样本则从“噪声”分布 $𝑞(𝑣)$ 中抽取，例如在 $\mathcal{V}$ 上的单字分布。我们将相应的损失函数表示为 $L_{𝑁𝐶𝐸}$。此外，我们通过使用一个辅助任务，其损失由$L_{𝐴𝑈𝑋}$表示（详细内容见第3.4节），将基于图的检索推广到任意度量 $𝑠(𝑣, 𝑢)$ 上。因此，整体目标函数为： 
$$
\mathcal{L}_{\text {all }}=\mathcal{L}_{N C E}+\mathcal{L}_{A U X} \\(3)
$$

#### 3.1.4 在训练后的相似性图上进行搜索

基于预计算的 item embedding $e_𝑣$（从物品网络 $NN_𝑣$ 中提取）构建了基于图的索引。在预测阶段，我们以适应实际系统和任意 $𝑠(𝑣, 𝑢)$ 的方式遍历相似性图。

### 3.2 Graph Construction

最初，基于相似性图的搜索是针对度量空间提出的，并扩展到了对称非度量情景，例如Mercer核和最大内积搜索（MIPS）。通过利用凸性代替三角不等式，$𝑠(𝑣, 𝑢)$ 也可以推广到某些非对称情况，例如Bregman散度[3]。然而，对于任意的 $𝑠(𝑣, 𝑢)$，s-Delaunay图不能保证存在或唯一。此外，从大规模语料库构建这样的s-Delaunay图即使在精确和近似的情况下都具有计算上的限制。因此，我们采用SL2G的方法[28]，通过使用 item embedding $e_𝑣$ 来构建图索引。该图是由 $e_𝑣$ 之间的 $𝑙_2$ 距离定义的，并且与 $𝑠(𝑣, 𝑢)$ 无关。在实践中，我们直接构建HNSW图，据称这是一种适当的方法来近似定义在 $𝑙_2$ 距离上的Delaunay图。

### 3.3 Online Retrieval

我们通过在原始HNSW上增加beam search的方法，提出了一种称为Beam-retrieval的算法来处理生产环境下的在线检索。利用预计算的$e_𝑣$值，在线检索阶段可以表示为： 
$$
\mathcal{B}_u=\underset{v \in \mathcal{V}}{\operatorname{argTopk}} s_u\left(\mathbf{e}_v\right) \\(4) 
$$
其中，$𝑠_𝑢(.)$ 是实时计算的用户特定函数，而 $e_𝑣$ 是唯一关于 $𝑠_𝑢(.)$ 的可变量，当在基于图的索引上进行搜索时。

HNSW的搜索过程以逐层自顶向下的方式遍历一系列近邻图，如算法1所示。为了方便起见，将原始的HNSW检索算法称为HNSW-retrieval。在HNSW-retrieval中，采用简单的贪婪搜索，其中算法1中的 $𝑒𝑓_𝑙(𝑙 > 0)$ 被设置为顶层的值1，并且给 $𝑒𝑓_0$ 赋予较大的值，以保证在底层具有良好的检索性能。然而，在实际的大规模推荐系统中，HNSW-retrieval不足以应对大规模检索问题，因为它存在以下问题：1）HNSW-retrieval中的SEARCH − LAYER子程序使用while循环探索图，使得在线推断的计算和延迟无法控制；2）采用简单贪婪搜索进行遍历更容易陷入局部最优解，特别是在我们的情况下，$𝑠_𝑢 (e_𝑣 )$通常是非凸的。因此，我们根据算法2重新设计了HNSW-retrieval中的SEARCH − LAYER子程序。首先，我们使用for循环替代了while循环，以控制预测延迟和要评估的候选项$𝑣$的数量。尽管有 early stop 策略，但for循环仍然可以保证检索性能，如图7所示。其次，我们打破了$𝑒𝑓_𝑙 (𝑙 > 0)$的限制，并在顶层进行扩大以利用批量计算。使用多条路径进行遍历等效于在相似性图上进行beam search，这被证明比图中原始版本更高效，如图3所示。

![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Alg1.png)

![Alg2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Alg2.png)

### 3.4 Search with Arbitrary Neural Metric

#### 3.4.1 动机

在面对任意模型时，无法再利用三角不等式、对称性和凸性来验证使用 $𝑠_𝑢 (e_𝑣 )$ 进行相似性图搜索的合理性。实际上，当对 $e_𝑣$ 进行微小扰动时，$𝑠_𝑢 (e_𝑣 )$ 的反应是高度不确定的，例如，当稍微扰动 $e_𝑣$ 时，$𝑠_𝑢 (𝑒_𝑣 )$ 可能会剧烈波动。直观上，这种不确定性会严重影响检索性能，特别是当在图构建阶段（例如，$e_𝑣$ 之间的 $l_2$ 距离）和检索阶段（$𝑠_𝑢 (e_𝑣 )$）中使用的相似度度量非常异质时，如表 3 所示。而在本文中，我们展示了通过有意地偏置 $𝑠_𝑢 (e_𝑣 )$ 来避免与 $e_𝑣$ 相关的不确定性，从而可以在实证上增强检索性能。

我们的理念基于对于一个可微函数 $𝑓 : \mathbb{R}_𝑑 → \mathbb{R}$ 在某个特定方向上寻找局部最优解的类比。假设 $min −𝑠_𝑢 (𝑒_𝑣 )$ 的解是在 $𝑅^𝑑$ 中定义的任意向量，梯度下降和坐标下降是常用的方法来寻找局部最优解。我们声称图搜索类似于块坐标下降，其中更新方向由图结构和前 $k$ 个节点的过程决定，而不是由梯度决定。因此，根据上述，我们可以将 $𝑠_𝑢 (e_𝑣 )$ 相对于 $e_𝑣$ 的不确定性解释为类似于基于梯度优化中损失函数  flat/sharpness of loss landscape。虽然有争议，但普遍认为与“尖锐最小值”相比，“平坦最小值”通常具有更好的泛化性能 [16, 30]，因为它们对输入的微小扰动具有鲁棒性。早期的研究试图改变优化算法以偏好平坦最小值并找到“更好”的区域 [4, 7, 12]。受到这些工作的启发，我们利用对抗性训练 [9, 26, 27, 30, 31] 来减轻 $𝑠_𝑢 (e_𝑣 )$ 相对于 $e_𝑣$ 的不确定性，并提高任意 $𝑠_𝑢 (e_𝑣 )$ 对于 $e_𝑣$ 的鲁棒性。

#### 3.4.2 对抗梯度方法

一般而言，我们采用对抗梯度方法，并在基于端到端学习的方法中引入平坦性到 $𝑠_𝑢 (e_𝑣 )$ [30]。近年来，通过对抗性示例的防御来实现深度神经网络的鲁棒性已被广泛应用于各种计算机视觉任务[9, 26, 27, 31]。对抗性示例是指带有精心设计的扰动的正常输入，这些扰动通常对人类来说不可察觉，但可以恶意欺骗深度神经网络。我们工作中使用的对抗训练是最有效的方法之一[23, 29]，用于对抗性示例进行深度学习的防御。更具体地说，我们通过对对抗性扰动 $\tilde{\mathbf{e}_v}$ 进行训练，使得 $e_𝑣$ 关于 $𝑠_𝑢(e_𝑣)$ 的风景线变得平坦。

在我们的情况下，我们要最大化 $𝑠_𝑢 (e_𝑣)$ 的解决方案局限于语料库 $\mathcal{V}$。因此，我们主要关注每个 $e_𝑣$ 周围 $𝑠_𝑢 (.)$ 的风景线，而不是整体风景线。我们将训练目标以平坦性表示如下： 
$$
\begin{aligned} \mathcal{L}_{A U X} & =\sum_u \sum_{v \in \mathcal{Y}_u} s_u\left(\mathbf{e}_v\right) \log \frac{s_u\left(\mathbf{e}_v\right)}{s_u\left(\tilde{\mathbf{e}_v}\right)} \\ \tilde{\mathbf{e}_v} & =\mathbf{e}_v+\Delta \end{aligned} \\(5)
$$
其中$\mathcal{Y}_𝑢$根据NCE原理由每个$𝑢 ∈ \mathcal{U}$对应的真实分布和噪声分布的标签组成。

具体而言，我们使用快速梯度符号方法（FGSM）[9]生成对抗性示例，该方法计算扰动为： 
$$
\Delta=\epsilon \operatorname{sign}\left(\nabla_{\mathbf{e}_v} s_u\left(\mathbf{e}_v\right)\right) \\(6) 
$$
其中$\nabla_{\mathbf{e}_v} s_u\left(\mathbf{e}_v\right)$表示$s_u\left(\mathbf{e}_v\right)$相对于$e_𝑣$的梯度，可以通过反向传播轻松计算得到，并且扰动$Δ$的最大范数受到$𝜖$的限制。 简单来说，我们在搜索中实现了任意度量，而不利用$𝑠_𝑢 (e_𝑣)$的凸性。相反，我们的框架建立在$𝑠_𝑢 (e_𝑣)$相对于每个$e_𝑣$的平坦性上，这可以通过一个简单而有效的辅助任务来实现。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure2.png)

**Figure 2**: Online Serving System. The NANN service is provided by real-time prediction server, where graph-based index and deep neural network constitute a unified Tensorflow graph. The NANN service receives the user features from real-time feature server and output the retrieved candidate items to downstream task directly.

## 4 SYSTEM IMPLEMENTATION

图2展示了所提出方法的在线服务架构。一般来说，几乎任何现成的深度神经网络推理系统，例如TensorFlow Serving，都可以为NANN提供即开即用的服务。由于我们将基于图的索引与深度神经网络集成，并形成一个统一的Tensorflow图，因此该框架具有灵活的使用和维护性。因此，NANN的神经网络推理和基于图的检索可以作为一个统一的模块进行服务。

如算法1所描述的，在线推断主要由$𝑠_𝑢 (e_𝑣 )$的前向传播和图上的搜索组成，两者交替进行。对于在线计算，我们将图上的搜索放置在中央处理器（CPU）上以保持检索的灵活性，而将$𝑠_𝑢 (e_𝑣 )$的前向传播放置在图形处理器（GPU）上以提高效率。相应地，基于图的索引和预先计算的$e_𝑣$都表示为Tensorflow tensor，并缓存在CPU内存中。主机与设备之间的通信遵循最新的Peripheral Component Interconnect Express（PCIe）总线标准。这种设计可以在灵活性和效率之间取得平衡，同时只引入轻微的通信开销。

图表示为在线检索奠定了基础。在我们的实现中，首先对每个$𝑣 ∈ \mathcal{V}$进行串行编号，并分配一个唯一的标识符。然后，HNSW的层次结构由多个Tensorflow RaggedTensors表示。 

在这里，我们主要强调我们提出的方法的在线服务效率优化，这是基于Tensorflow框架实现的。

### 4.1 Mark with Bitmap

为了确保在线检索性能，重要的是在有限的邻域传播轮次内增加候选项的扩展范围，如算法2所示。因此，我们需要标记已访问的 item 并绕过它们以进一步遍历。由于 $𝑣$ 是连续编号的，位图的思想就浮现了出来。我们通过在Tensorflow框架中构建C++自定义操作（Ops）来实现位图过程。我们在表1中以每秒查询数（QPS）和响应时间（RT）（以毫秒为单位）总结了Bitmap Ops的性能。

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Table1.png)

我们使用生产环境中部署的 $𝑠_𝑢 (e_𝑣)$ 模型架构来测试 Bitmap Ops 的性能，其详细配置将在第5节中介绍。我们遍历了一个三层图形索引，其中$|\mathcal{V}|$等于1,300,000，并调整算法2中的参数来控制要评估的候选人数，大约为17,000个用于基准测试。正如表1所示，我们的自定义Bitmap Ops明显优于Tensorflow原始集合操作（Raw Set Ops）。

### 4.2 Dynamic Shape with XLA

XLA（加速线性代数）是一种针对线性代数的领域特定编译器，可以加速TensorFlow模型的执行。通过将单个TensorFlow操作融合成粗粒度的集群，XLA可以自动优化模型的执行速度和内存使用情况。在我们的模型中，借助XLA的帮助，我们实现了大约3倍的性能提升。然而，它要求计算图中的所有张量具有固定的形状，并且编译的代码是专门针对具体形状的。在我们的场景中，根据算法2中的每次邻居传播，待评估的未访问 item $|\mathcal{𝐶}|$ 的数量是动态的。因此，我们提出了一种“自动填充”策略，将动态形状（例如算法2中的 $|\mathcal{𝐶}|$）转换为预定义的固定形状。具体而言，我们预先设定了一个潜在形状网格，并使用XLA的即时编译（JIT）在这些预定义形状上生成编译代码，该编译由生产环境中的日志重放触发。对于在线推断，"自动填充"策略会将大小为 $|\mathcal{𝐶}|$ 的张量自动填充到网格上最接近的更大点，并通过XLA使用相应的编译代码高效执行，之后再将张量切片回其原始形状。简而言之，我们通过自动的“填充-切片”策略扩展了XLA对动态形状的支持能力。

## 5 EXPERIMENTS

在本节中，我们研究了所提出的方法的性能，并进行了相应的分析。除了与基准方法的比较外，由于直接相关工作的不足，我们更加重视NANN的检索性能及其对应的消融研究。我们在一个开源基准数据集和淘宝的一个行业数据集上进行实验，以展示所提出方法的有效性。我们观察到，我们所提出的方法可以显著优于基准方法，并且在计算量明显减少的情况下，几乎达到与其暴力搜索对应的检索性能。

### 5.1 Setup

#### 5.1.1 数据集

我们使用了两个大规模数据集进行实验：1) 一个来自淘宝的公开可访问的用户-物品行为数据集，称为 UserBehavior3；2) 一份来自淘宝流量日志的真实行业数据集。表2总结了这两个数据集的主要统计信息。

**UserBehavior**. UserBehavior是淘宝用户行为的一个子集，用于隐式反馈的推荐问题。每条记录包括user-ID、item-ID、item-cate-ID、行为类型和时间戳。行为类型表示 user 与 item 的交互方式，包括点击、购买、将物品添加到购物车和将物品添加到收藏夹中。我们过滤掉一些稀疏性较高的用户，并保留至少有10个行为的用户。假设用户 $𝑢$ 的行为序列为 $(𝑏_{𝑢_1} , . . . , 𝑏_{𝑢_𝑘} , . . . , 𝑏_{𝑢_𝑛} )$，任务是基于之前的行为来预测 $𝑏_{𝑢_{𝑘+1}}$。验证集和测试集分别由随机选择的10,000个用户的样本组成。我们将每个用户 $𝑢$ 的第$⌈𝑙_𝑢/2⌉$个行为（其中$𝑙_𝑢$表示用户$𝑢$的行为序列长度）作为真值，并基于之前的所有行为进行预测。

**淘宝的行业数据集**。该行业数据集是从淘宝平台的流量日志中收集而来，其组织方式与UserBehavior类似，但具有更多的特征和记录。该行业数据集的特征主要由用户资料、用户行为序列和物品属性组成。

#### 5.1.2 评价指标

我们使用 recall-all@𝑀，recall-retrieval@𝑀，recall-Δ@𝑀 和 coverage@𝑀 来评估我们所提出方法的有效性。通常情况下，对于用户$𝑢$，可以定义召回率如下：
$$
\text { recall }\left(\mathcal{P}_u, \mathcal{G}_u\right) @ M(u)=\frac{\left|\mathcal{P}_u \cap \mathcal{G}_u\right|}{\left|\mathcal{G}_u\right|}
$$
其中 $\mathcal{P}_u\left(\left|\mathcal{P}_u\right|=M\right)$ 表示检索到的物品集合，$\mathcal{G}_𝑢$ 表示真实物品集合。 通过对语料库 $\mathcal{V}$ 进行完全评估来评估训练得分模型 $𝑠(𝑣, 𝑢)$ 的能力，即
$$
\text { recall-all@M(u)= recall }\left(\mathcal{B}_u, \mathcal{G}_u\right) @ M(u)
$$
其中 $\mathcal{B}_u=\operatorname{argTopk}_{v \in \mathcal{V}} s_u\left(\mathbf{e}_v\right)\left(\left|\mathcal{B}_u\right|=M\right)$ 是通过暴力扫描生成的精确前 $𝑘$ 个得分最高的物品集合。

假设我们通过 $𝑠(𝑣, 𝑢)$ 遍历基于图的索引，并为每个用户 $𝑢$ 检索相关物品 $R_𝑢 (|R_𝑢 | = 𝑀|)$，则检索召回率可以通过以下方式进行评估： 
$$
\text { recall-retrieval@M(u)= recall }\left(\mathcal{R}_u, \mathcal{G}_u\right) @ M(u)
$$
相应地，基于图索引引入的检索损失可以定义为召回率的差异： 
$$
\text{recall-}\Delta @ M(u) = \frac{\text{recall-all@} M - \text{recall-retrieval@} M}{\text{recall-all@} M}
$$
此外，我们使用 coverage@𝑀(𝑢) 来描述暴力扫描和检索之间的差异。具体地说，
$$
\text{coverage@M(u)} = \frac{\left|\mathcal{R}_u \cap \mathcal{B}_u\right|}{\left|\mathcal{B}_u\right|} 
$$
从现在开始，我们将检索质量指标称为检索结果与暴力扫描结果之间的一致性，通过 recall-Δ@𝑀 和 coverage@M 进行衡量。 最后，我们对每个 $𝑢$ 求平均以得到最终的评价指标，其中 $𝑢$ 来自测试集。

#### 5.1.3 模型结构

模型结构（标记为带有注意力的DNN）如图1所示，包括用户网络、目标注意力网络、物品网络和评分网络。更多细节请参见附录。为了衡量不同模型结构的模型容量和检索性能，我们还对以下模型结构进行了实验：1）无注意力的DNN（DNN w/o attention），它将目标注意力网络替换为对用户行为序列嵌入进行简单求和池化；2）双边模型（two-sided），它仅包含用户embedding（用户网络输出和用户行为序列embedding求和的串联）和item embedding，并通过内积计算用户-物品偏好评分。

#### 5.1.4 实现细节

给定数据集和模型结构，我们使用方程（3）中定义的损失函数对模型进行训练。采用学习率为3e-3的Adam优化器来最小化损失。对于工业数据集，FGSM的 𝜖 设置为1e-2，对于UserBehavior数据集，𝜖 设置为3e-4。我们通过NCE对模型进行优化，并将来自真实分布的每个标签分配为工业数据集和UserBehavior数据集分别从噪声分布中抽取的19个和199个标签。

训练完成后，我们提取所有有效 item 的 item 网络之后的 item 特征，用于构建HNSW图。使用标准的索引构建算法[18]，建立的连接数量设置为32，在图构建阶段动态候选列表的大小设置为40。

在检索阶段，我们对第二层中的项目进行全面的分数计算。第二层由整个词汇表的千分之一的项目组成，并且可以高效地以一个批次进行评分。然后，检索出前 $k$ 个相关项目作为下一步检索的入口点，其中 $k=𝑒 𝑓_2$。默认的检索参数设置为$\{𝑒𝑓_2, 𝑒𝑓_1, 𝑒𝑓_0\} = \{100, 200, 400\}$，$\{𝑇_2,𝑇_1,𝑇_0\} = \{1, 1, 3\}$，详见算法1和算法2的描述。如果没有进一步声明，我们将报告最终检索到的项目的top-200 ($𝑀 = 200$)指标。 

所有超参数都是通过交叉验证确定的。

#### 5.2.1 对比基线

我们与基线方法SL2G进行比较，SL2G直接利用HNSW图中由$e_𝑣$之间的 $l2$ 距离构建的深度模型进行HNSW检索。不同方法的比较结果如图3所示。每个 $x$ 轴代表达到最终候选项所需遍历的 item 数与$|\mathcal{V}|$之比。 

首先，在两个数据集中，与SL2G相比，NANN在不同数量的遍历项目上实现了召回率和覆盖率的显著提升。特别是，当我们评估较少部分的 item 以达到最终 item 时，NANN的优势更加明显。 

其次，NANN凭借更少的计算量与其暴力搜索对应的方法相媲美。尤其是，当应用于淘宝的工业数据时，NANN以默认检索参数实现了0.60%的召回率-Δ和99.0%的覆盖率，并且几乎不会影响检索质量。

此外，通过抵御中等级别的对抗攻击，模型的容量和鲁棒性（由recall-all指标表示）也会受益。

最后，NANN可以快速收敛到令人满意的检索性能。如图3的曲率所示，在两个数据集中，只需评估1%~2%的V即可达到令人满意的检索质量。

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure3.png)**Figure 3**: Results of our proposed NANN and SL2G on Industry and UserBehavior dataset.

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure4.png)**Figure 4**: The effect of adversarial gradient training on Industry and UserBehavior dataset.

#### 5.2.2 Beam-retrieval vs HNSW-retrieval

图3展示了Beam-retrieval（“NANN”曲线）和原始HNSW-retrieval（“NANN-HNSW”曲线）的召回率和覆盖率。从这些图中可以看出，算法2在两个方面优于HNSW-retrieval版本：1）在不同数量的遍历item上表现更好；2）它更快地收敛到令人满意的检索质量。此外，如图7所示，HNSW-retrieval的while循环导致了在底层进行冗余的邻域传播，这对于召回率和覆盖率而言是不必要的。 

![Table3](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Table3.png)

**Table 3**: Results of different model architectures on industry and UserBehavior dataset.

![Figure5](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure5.png)

**Figure 5**: Neighborhood propagation in ground layer.

![Figure6](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure6.png)

**Figure 6**: Reaction to small perturbations on the Industry dataset.

![Figure7](/Users/anmingyu/Github/Gor-rok/Papers/TDM/Approximate Nearest Neighbor Search under Neural Similarity Metric for Large-Scale Recommendation/Figure7.png)

**Figure 7**: Sensitivity Analysis for 𝜖 in FGSM and top-k of ground layer in Beam-retrieval on the Industry dataset.

#### 5.2.3 对抗性梯度训练的有效性

图4中，我们使用Beam-retrieval遍历相似性图，并展示了对抗性攻击防御的有效性。我们观察到，在不同程度的遍历中，NANN始终优于没有进行对抗性训练的模型。

我们还研究了FGSM对不同模型架构的影响。如表3所示的recall-all和recall-Δ，我们经验证明，更复杂的模型通常具有更好的泛化性能，但可能会降低检索质量。基于这一观察，我们认为recall-all和recall-retrieval之间日益增大的差距可能源自相似度度量之间的更高异质性，因此利用对抗性训练来减少差距。比较中使用了默认的检索参数。如表3所示，从简单到复杂的所有模型架构的性能都可以从对抗性训练中受益；FGSM可以显著提高检索质量，尤其是对于更复杂的模型。

#### 5.2.4 对对抗性梯度训练的分析

图6展示了模型在经过训练后对对抗攻击的反应。我们将对抗攻击的$Δ$定义为$\epsilon \cdot \operatorname{rand}(-1,1)$，类似于FGSM，并通过可视化$|𝑠_𝑢 (e_𝑣 )−𝑠_𝑢 (e_𝑣+Δ)|$来比较不同模型的鲁棒性。图6是$\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$的直方图，其中$v \in \operatorname{argTopk}_{v \in \mathcal{V}} s_u\left(\mathbf{e}_v\right)$。如图6所示，当面对对抗性攻击时，检索质量与模型的鲁棒性有经验证明相关：1）没有注意力机制的模型中$\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$的右偏分布更大，表明其对带有注意力机制的模型具有更好的鲁棒性，这与表3中的recall-Δ结果一致；2）通过FGSM可以显著提高带有注意力机制的模型的检索质量，并且通过对抗性训练，$\left|s_u\left(\mathbf{e}_v\right)-s_u\left(\mathbf{e}_v+\Delta\right)\right|$的分布向右偏斜更多。

#### 5.2.5 敏感性分析

$𝜖$ 的大小。图7(a)展示了$𝜖$和以覆盖率衡量的检索质量之间的相关性。一般来说，检索质量与$𝜖$的大小呈正相关。此外，在$𝜖$适度大小范围内，对抗性攻击可以对以recall-all衡量的整体性能有益，但当$𝜖$过大时，会产生不良影响。因此，$𝜖$的大小在平衡检索质量和整体性能之间起着重要作用。

**不同的top-k值**。图7(b)展示了我们提出的方法对于算法1中最终检索到的前k个项目中不同k值的影响。NANN在不同的k值下表现依然良好。即使是使用较大的k值进行检索，也可以保证检索质量。因此，我们的方法对k值不敏感。

### 5.3 Online Results

我们在淘宝展示广告平台的实际流量上评估了我们提出的方法。在线A/B实验在淘宝App的主要商业页面上进行，例如“猜你喜欢”页面，并持续了一个多月。在线基准是最新的TDM方法，采用基于 beam search 的贝叶斯优化[33-35]。为了公平比较，我们只将候选生成阶段中的一个通道TDM替换为NANN，并保持其他因素（如传递到排序阶段的候选项数量）不变。采用了两个常见的在线广告指标来衡量在线性能：点击率（CTR）和千次展示收入（RPM）。
$$
\mathrm{CTR}=\frac{\# \text { of clicks }}{\# \text { of impressions }}, \mathrm{RPM}=\frac{\text { Ad revenue }}{\# \text { of impressions }} \times 1000
$$
与TDM相比，NANN在点击率（CTR）和千次展示收入（RPM）方面显著提升了2.4%和3.1%，这证明了我们方法在用户体验和商业效益方面的有效性。

此外，在第4节介绍的高效实现使我们能够从NANN中受益，而不会牺牲在线推理的响应时间和每秒请求数。在生产环境中，NANN满足了表1中显示的性能基准。目前，NANN已经完全部署，并在淘宝展示广告平台上提供在线检索服务。

## 6 CONCLUSION

近年来，使用深度神经网络解决大规模检索问题的趋势日益增多。然而，这些方法通常在额外的训练成本和使用目标项的辅助信息方面存在困难，因为可学习索引的存在。我们提出了一种轻量级的方法，将事后训练的基于图的索引与任意先进的模型集成在一起。为了保证检索质量，我们提出了启发式方法和基于学习的方法：1）我们提出的Beam-retrieval在相同计算量下明显优于现有的图搜索方法；2）我们在大规模检索问题中创造性地引入对抗性攻击，以改善检索质量和模型的鲁棒性。广泛的实验证明了我们提出方法的有效性。此外，我们详细总结了在淘宝展示广告中部署NANN的实践经验，在用户体验和商业收入方面取得了显著的改进。我们希望我们的工作不仅适用于推荐系统等领域，也可以广泛应用于网页搜索和基于内容的图像检索等其他领域。未来，我们希望进一步揭示对抗性攻击对大规模检索问题适用性的潜在机制。