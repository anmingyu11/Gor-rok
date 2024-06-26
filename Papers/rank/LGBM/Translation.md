> 北大微软研究院

# LightGBM: A Highly Efficient Gradient Boosting Decision Tree

## Abstract

Gradient Boosting Decision Tree (GBDT) is a popular machine learning algorithm, and has quite a few effective implementations such as XGBoost and pGBRT. Although many engineering optimizations have been adopted in these implementations, the efficiency and scalability are still unsatisfactory when the feature dimension is high and data size is large. A major reason is that for each feature, they need to scan all the data instances to estimate the information gain of all possible split points, which is very time consuming. To tackle this problem, we propose two novel techniques: Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). 

With GOSS, we exclude a significant proportion of data instances with small gradients, and only use the rest to estimate the information gain. We prove that, since the data instances with larger gradients play a more important role in the computation of information gain, GOSS can obtain quite accurate estimation of the information gain with a much smaller data size.

With EFB, we bundle mutually exclusive features (i.e., they rarely take nonzero values simultaneously), to reduce the number of features. We prove that finding the optimal bundling of exclusive features is NP-hard, but a greedy algorithm can achieve quite good approximation ratio (and thus can effectively reduce the number of features without hurting the accuracy of split point determination by much). 

We call our new GBDT implementation with GOSS and EFB LightGBM. Our experiments on multiple public datasets show that, LightGBM speeds up the training process of conventional GBDT by up to over 20 times while achieving almost the same accuracy.

> 梯度提升树(GBDT)是一种流行的机器学习算法，有 XGBoost 和 pGBRT 等多种有效的实现方法。虽然在这些实现中采取了许多工程优化措施，但在特征维数高、数据量大的情况下，效率和可扩展性仍然不能令人满意。一个主要的原因是，对于每个特征，他们需要扫描所有的数据实例来估计所有可能的分割点的信息增益，这非常耗时。为了解决这个问题，我们提出了两种新技术：基于梯度的单边采样 (GOSS) 和 互斥特征捆绑 (EFB)。
>
> 使用GOSS，我们排除了很大一部分具有小梯度的数据实例，而只使用其余的数据实例来估计信息增益。我们证明，具有较大梯度的数据在信息增益的计算中起着更重要的作用，因此利用GOSS方法计算得到的信息增益即使只用了较少数据，精度也非常高 。
>
> 使用EFB，我们捆绑互斥的特征(即，它们很少同时采用非零值)，以减少特征的数量。我们证明了找到最理想的特征捆绑的解法是 NP-hard 的，但 exact greedy 算法可以获得相当好近似效果(从而可以有效地减少特征的数量，而不会对分割点确定的准确性造成太大的影响)。我们使用 GOSS 和 EFB LightGBM 将我们的新 GBDT 实现称为 GBDT。我们在多个公开数据集上的实验表明，LightGBM 在获得几乎相同的准确率的情况下，将传统 GBDT 的训练过程加快了 20 倍以上。

## 1 Introduction

Gradient boosting decision tree (GBDT) [1] is a widely-used machine learning algorithm, due to its efficiency, accuracy, and interpretability. GBDT achieves state-of-the-art performances in many machine learning tasks, such as multi-class classification [2], click prediction [3], and learning to rank [4]. 

In recent years, with the emergence of big data (in terms of both the number of features and the number of instances), GBDT is facing new challenges, especially in the tradeoff between accuracy and efficiency. Conventional implementations of GBDT need to, for every feature, scan all the data instances to estimate the information gain of all the possible split points. Therefore, their computational complexities will be proportional to both the number of features and the number of instances. This makes these implementations very time consuming when handling big data.

To tackle this challenge, a straightforward idea is to reduce the number of data instances and the number of features. However, this turns out to be highly non-trivial. For example, it is unclear how to perform data sampling for GBDT. While there are some works that sample data according to their weights to speed up the training process of boosting [5, 6, 7], they cannot be directly applied to GBDT since there is no sample weight in GBDT at all. In this paper, we propose two novel techniques towards this goal, as elaborated below.

> 梯度提升树(GBDT)[1]是一种广泛使用的机器学习算法，具有高效、准确、可解释等优点。GBDT 在许多机器学习任务中达到了最先进的性能，例如多分类[2]、点击预测[3] 和 learning to rank [4]。
>
> 近年来，随着大数据的出现(无论是从特征数量还是实例数量)，GBDT 都面临着新的挑战，尤其是在精度和效率之间的权衡。对于每个特征，GBDT的常规实现需要扫描所有数据实例以估计所有可能的分割点的信息增益。因此，它们的计算复杂度将与特征数量和实例数量成正比。这使得在处理大数据时非常耗时。
>
> 为了应对这一挑战，最直截了当的办法是减少数据量、缩小特征维度。 然而，这个想法是非常有价值的。 例如，目前还不清楚如何为 GBDT 进行数据采样。 虽然有些研究根据数据权重对数据进行采样进而加速的训练过程，但由于 GBDT 中根本没有样本权重，因此无法直接应用于 GBDT。 在本文中，我们提出了两种新的技术来实现这一目的，下面详细叙述。

#### Gradient-based One-Side Sampling (GOSS). 

While there is no native weight for data instance in GBDT, we notice that data instances with different gradients play different roles in the computation of information gain. In particular, according to the definition of information gain, those instances with larger $gradients^1$ (i.e., under-trained instances) will contribute more to the information gain. Therefore, when down sampling the data instances, in order to retain the accuracy of information gain estimation, we should better keep those instances with large gradients (e.g., larger than a pre-defined threshold, or among the top percentiles), and only randomly drop those instances with small gradients. We prove that such a treatment can lead to a more accurate gain estimation than uniformly random sampling, with the same target sampling rate, especially when the value of information gain has a large range.

> 虽然 GBDT 中没有数据实例的固有权重，但我们注意到不同梯度的数据实例在信息增益计算中扮演着不同的角色。特别地，根据信息增益的定义，$gradients^1$ 较大的实例(即训练不足的实例)对信息增益的贡献更大。因此，在对数据实例进行下采样时，为了保持信息增益估计的准确性，最好保留那些梯度较大的实例(例如，大于预定义的阈值，或者在前几个百分位数之间)，而只随机丢弃那些梯度较小的实例。我们证明，在相同的采样率下，这种处理方法最终计算出的信息增益比均匀随机采样要准确，特别是当信息增益的值具有大范围时。

Exclusive Feature Bundling (EFB). 

Usually in real applications, although there are a large number of features, the feature space is quite sparse, which provides us a possibility of designing a nearly lossless approach to reduce the number of effective features. Specifically, in a sparse feature space, many features are (almost) exclusive, i.e., they rarely take nonzero values simultaneously. Examples include the one-hot features (e.g., one-hot word representation in text mining). We can safely bundle such exclusive features. To this end, we design an efficient algorithm by reducing the optimal bundling problem to a graph coloring problem (by taking features as vertices and adding edges for every two features if they are not mutually exclusive), and solving it by a greedy algorithm with a constant approximation ratio.

We call the new GBDT algorithm with GOSS and EFB LightGBM2 . Our experiments on multiple public datasets show that LightGBM can accelerate the training process by up to over 20 times while achieving almost the same accuracy.

> 通常在实际应用中，虽然存在大量的特征，但特征空间是相当稀疏的，这为我们设计一种几乎无损的方法来减少有效特征的数量提供了可能。具体地说，在稀疏特征空间中，许多特征(几乎)是互斥的，即它们很少同时取非零值。比如包括 one-hot 特征(例如，文本挖掘中的一次热词表示)。我们可以放心地捆绑这些互斥特征。为此，我们设计了一种高效的算法，将最优捆绑问题归结为图着色问题(以特征为顶点，在特征不互斥的情况下为每两个特征添加边)，并通过贪心法解决，恒定近似比。
>
> 我们称新的搭配 GOSS 和 EFB 算法的 GBDT 叫 LightGBM。我们在多个公开数据集上的实验表明，LightGBM 将训练速度加速了近 20 倍，同时能达到几乎相同的精度。
>
> 本文的其余部分整理如下。首先，我们在第 2 部分检查整理了 GBDT 算法及其相关工作。然后，我们在第 3、4 部分介绍了 GOSS 算法和 EFB 算法的细节。第 5 部分呈现了在公开数据集上的实验结果。最后，第 6 部分是结论。

## 2 Preliminaries

#### 2.1 GBDT and Its Complexity Analysis

GBDT is an ensemble model of decision trees, which are trained in sequence [1]. In each iteration, GBDT learns the decision trees by fitting the negative gradients (also known as residual errors).

The main cost in GBDT lies in learning the decision trees, and the most time-consuming part in learning a decision tree is to find the best split points. One of the most popular algorithms to find split points is the pre-sorted algorithm [8, 9], which enumerates all possible split points on the pre-sorted feature values. This algorithm is simple and can find the optimal split points, however, it is inefficient in both training speed and memory consumption. Another popular algorithm is the histogram-based algorithm [10, 11, 12], as shown in Alg. $1^3$ . Instead of finding the split points on the sorted feature values, histogram-based algorithm buckets continuous feature values into discrete bins and uses these bins to construct feature histograms during training. Since the histogram-based algorithm is more efficient in both memory consumption and training speed, we will develop our work on its basis.

As shown in Alg. 1, the histogram-based algorithm finds the best split points based on the feature histograms. It costs `O(#data × #feature)` for histogram building and `O(#bin × #feature)` for split point finding. Since `#bin` is usually much smaller than `#data`, histogram building will dominate the computational complexity. If we can reduce #data or #feature, we will be able to substantially speed up the training of GBDT.

> GBDT 是顺序训练的决策树集成模型[1]。在每次迭代中，GBDT通过拟合负梯度(也称为残差)来学习决策树。
>
> GBDT 的主要代价在于学习决策树，而学习决策树最耗时的部分是寻找最佳分割点。寻找分割点最流行的算法之一是预排序算法[8，9]，它遍历预排序特征值上所有可能的分割点。该算法简单，能找到最优分割点，但在训练速度和内存消耗方面效率较低。另一个流行的算法是基于直方图的算法[10，11，12]， Alg  1​。基于直方图的算法不是在排序的特征值上寻找分割点，而是将连续的特征值分成离散的桶，并在训练期间使用这些桶来构建特征直方图。由于基于直方图的算法在内存消耗和训练速度上都更有效，我们将在此基础上发展我们的工作。
>
> 如 Alg 1中所示。基于直方图的算法根据特征直方图寻找最佳分割点。
> 构建直方图需要 `O(#data×#Feature)` ，查找分割点需要 `O(#bin×#Feature)` 。由于 `#bin` 通常比 `#data` 小得多，所以直方图构建将计算复杂度的主导地位。如果我们能够减少 `#data` 或 `#Feature`，我们将能够大幅加快 GBDT 的训练。

#### 2.2 Related Work

There have been quite a few implementations of GBDT in the literature, including XGBoost [13], pGBRT [14], scikit-learn [15], and gbm in R [16] 4 . Scikit-learn and gbm in R implements the presorted algorithm, and pGBRT implements the histogram-based algorithm. XGBoost supports both the pre-sorted algorithm and histogram-based algorithm. As shown in [13], XGBoost outperforms the other tools. So, we use XGBoost as our baseline in the experiment section.

To reduce the size of the training data, a common approach is to down sample the data instances. For example, in [5], data instances are filtered if their weights are smaller than a fixed threshold. SGB [20] uses a random subset to train the weak learners in every iteration. In [6], the sampling ratio are dynamically adjusted in the training progress. However, all these works except SGB [20] are based on AdaBoost [21], and cannot be directly applied to GBDT since there are no native weights for data instances in GBDT. Though SGB can be applied to GBDT, it usually hurts accuracy and thus it is not a desirable choice.

Similarly, to reduce the number of features, it is natural to filter weak features [22, 23, 7, 24]. This is usually done by principle component analysis or projection pursuit. However, these approaches highly rely on the assumption that features contain significant redundancy, which might not always be true in practice (features are usually designed with their unique contributions and removing any of them may affect the training accuracy to some degree).

The large-scale datasets used in real applications are usually quite sparse. GBDT with the pre-sorted algorithm can reduce the training cost by ignoring the features with zero values [13]. However, GBDT with the histogram-based algorithm does not have efficient sparse optimization solutions. The reason is that the histogram-based algorithm needs to retrieve feature bin values (refer to Alg. 1) for each data instance no matter the feature value is zero or not. It is highly preferred that GBDT with the histogram-based algorithm can effectively leverage such sparse property.

To address the limitations of previous works, we propose two new novel techniques called Gradientbased One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB). More details will be introduced in the next sections.

![Algorithm1](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/ Alg1.png)

> 在文献中已经有很多关于 GBDT 的实现，包括 XGBoost、pGBRT、scikit-learn和 R 中的 GBM。R 中的 scikit-learn和 gbm 实现了预排序算法，PGBRT 实现了基于直方图的算法。XGBoot 同时支持预排序算法和基于直方图的算法。文献「3」中指出，XGBoost 优于其他工具。因此，我们在实验部分使用XGBoost作为我们的 baseline。
>
> 为了减少训练数据的大小，一种常见的方法是对数据降采样。例如，在 [5] 中，如果数据样本的权重小于固定阈值，则对数据样本进行过滤。SGB在每次迭代中使用随机子集来训练弱学习器。在 [6] 中，在训练过程中动态调整采样率。然而，除了 SGB 以外的所有这些工作都是基于 AdaBoost 的，并且不能直接应用于 GBDT，因为在 GBDT 中样本没有初始权重。虽然 SGB 可以应用于 GBDT，但通常会影响精度，因此不是一个理想的选择。
>
> 类似地，为了减少特征维度，过滤弱特征是很自然的想法。这通常是通过主成分分析或 projection pursuit 来实现的。然而，这些方法高度依赖于特征显著冗余的假设，这在实践中可能并不总是正确的（特征被设计出来通常会有自己独特的作用，去除它们中的任何一种可能在某种程度上影响训练精度）。
>
> 实际应用中的大规模数据集通常非常稀疏。使用预排序算法的 GBDT 可以通过忽略具有零值的特征来降低训练成本。然而，基于直方图的算法的 GBDT 针对稀疏问题没有有效的优化解。原因在于基于直方图的算法对于每个数据样本，无论特征值是否为零，都需要去检索特征直方图的值 (Alg.1)。最好是基于直方图的算法的GBDT也可以有效地利用这种稀疏性。
>
> 为了解决前人研究中的局限性，我们提出了两种新技术，称为基于梯度的单侧采样（GOSS）和互斥特征捆绑（EFB）。更多细节将在下一节中介绍。

>这里我来简单阐述一下作者所说的直方图算法。这算法其实很简单，和XGBoost中分割点的近似算法有点像，只不过XGBoost中是利用百分位数来粗略的卡分割位置，而直方图算法是直接把特征值进行分箱然后编号。箱子的长度是可以设定的，直方图算法就是把多个点放到一个箱子里算成一个点了，这样找分割位置的时候，原来预排序算法一步走一个点，现在一步走一个箱子，肯定是快了不少，内存消耗也会变小。
>
>作者原文中说的特征为稀疏矩阵的情况，预排序可以忽略为0的，这就相当于数据量急剧减小。但直方图算法有个装箱的过程，无论是是不是0你都得首先装箱，这就是作者说的此时效率低的原因。
>
>除此之外直方图算法还有一个做差加速的操作，听起来名字高大上，其实很弱智。这个操作是说一个节点的直方图可以用它的父母节点的直方图减去兄弟节点的直方图得到，这是显而易见的。
>
>原文链接：https://blog.csdn.net/zhaojc1995/article/details/88382424

## 3 Gradient-based One-Side Sampling

In this section, we propose a novel sampling method for GBDT that can achieve a good balance between reducing the number of data instances and keeping the accuracy for learned decision trees.

![Algorithm2](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Alg2.png)

> 在本节中，我们提出了一种新的 GBDT 采样方法，该方法可以良好的在减少数据实例的数量和保持学习的决策树的准确性之间进行权衡。

#### 3.1 Algorithm Description

In AdaBoost, the sample weight serves as a good indicator for the importance of data instances. However, in GBDT, there are no native sample weights, and thus the sampling methods proposed for AdaBoost cannot be directly applied. Fortunately, we notice that the gradient for each data instance in GBDT provides us with useful information for data sampling. That is, if an instance is associated with a small gradient, the training error for this instance is small and it is already well-trained. A straightforward idea is to discard those data instances with small gradients. However, the data distribution will be changed by doing so, which will hurt the accuracy of the learned model. To avoid this problem, we propose a new method called Gradient-based One-Side Sampling (GOSS).

GOSS keeps all the instances with large gradients and performs random sampling on the instances with small gradients. In order to compensate the influence to the data distribution, when computing the information gain, GOSS introduces a constant multiplier for the data instances with small gradients (see Alg. 2). Specifically, GOSS firstly sorts the data instances according to the absolute value of their gradients and selects the top $a×100\%$ instances. Then it randomly samples $b×100\%$ instances from the rest of the data. After that, GOSS amplifies the sampled data with small gradients by a constant $\frac{1}{a-b}$ when calculating the information gain. By doing so, we put more focus on the under-trained instances without changing the original data distribution by much.

> 在 AdaBoost 中，样本权重可以很好地指示数据实例的重要性。然而，在 GBDT 中，没有初始样本权重，因此不能直接应用为 AdaBoost 建议的采样方法。幸运的是，我们注意到 GBDT 中每个数据实例的梯度为数据采样提供了有用的信息。也就是说，如果一个实例与一个小梯度相关联，则该实例的训练误差很小，并且它已经经过了良好的训练。一个简单的想法是丢弃那些具有小梯度的数据实例。但是，这样做会改变数据的分布，从而影响学习模型的准确性。为了避免这个问题，我们提出了一种新的方法，称为基于梯度的单边采样(GOSS)。
>
> GOSS方法保留所有较大梯度的样本，并对剩下小梯度样本进行随机采样。为了补偿对数据分布的影响，在计算信息增益时令小梯度数据样本乘以一个常数（见Alg.2）。具体来说，GOSS 算法首先根据数据梯度的绝对值进行排序，排序后选择前 $a × 100 \%$ 的样本。然后从剩下的数据中随机抽取 $b × 100 \%$ 的样本。之后在计算信息增益时，通过一个常数 $\frac{1-a}{b}$ 来增大小梯度样本的权重。这样做我们就可以在不改变数据分布的同时把更多精力放在训练的没那么好的数据上。

## 3.2 Theoretical Analysis

GBDT uses decision trees to learn a function from the input space $\mathcal{X}^s$ to the gradient space $\mathcal{G}$ [1]. Suppose that we have a training set with $n$ i.i.d. instances $\{x_1,\cdots, x_n\}$, where each $x_i$ is a vector with dimension $s$ in space $\mathcal{X}^s$ . In each iteration of gradient boosting, the negative gradients of the loss function with respect to the output of the model are denoted as $\{g_1,\cdots, g_n\}$. The decision tree model splits each node at the most informative feature (with the largest information gain). For GBDT, the information gain is usually measured by the variance after splitting, which is defined as below.

**Definition 3.1** *Let $O$ be the training dataset on a fixed node of the decision tree. The variance gain of splitting feature $j$ at point $d$ for this node is defined as*
$$
\begin{array}{c}
V_{j \mid O}(d)=\frac{1}{n_{O}}\left(\frac{\left(\sum_{\left\{x_{i} \in O: x_{i j} \leq d\right\}} g_{i}\right)^{2}}{n_{l \mid O}^{j}(d)}+\frac{\left(\sum_{\left\{x_{i} \in O: x_{i j}>d\right\}} g_{i}\right)^{2}}{n_{r \mid O}^{j}(d)}\right) \\
\end{array}
$$
where 
$$
n_{O}=\sum I\left[x_{i} \in O\right], 
\\
n_{l \mid O}^{j}(d)=\sum I\left[x_{i} \in O: x_{i j} \leq d\right] 
\\
\text { and }
\\
n_{r \mid O}^{j}(d)=\sum I\left[x_{i} \in O: x_{i j}>d\right]
$$
For feature $j$, the decision tree algorithm selects $d^∗_j=argmax_dV_j(D)$ and calculates the largest gain $V_j(d_j^*)$.5  Then, the data are split according feature $j^*$ at point $d_{j_∗}$ into the left and right child nodes.

In our proposed GOSS method, first, we rank the training instances according to their absolute values of their gradients in the descending order; second, we keep the top-$a × 100\%$  instances with the larger gradients and get an instance subset $A$; then, for the remaining set $A^c$ consisting $(1 − a) × 100\%$ instances with smaller gradients, we further randomly sample a subset $B$ with size $b × |A^c|$; finally, we split the instances according to the estimated variance gain $\tilde{V}_{j}(d)$ over the subset $A ∪ B$, i.e.,
$$
\tilde{V}_{j}(d)=\frac{1}{n}\left(\frac{\left(\sum_{x_{i} \in A_{l}} g_{i}+\frac{1-a}{b} \sum_{x_{i} \in B_{l}} g_{i}\right)^{2}}{n_{l}^{j}(d)}+\frac{\left(\sum_{x_{i} \in A_{r}} g_{i}+\frac{1-a}{b} \sum_{x_{i} \in B_{r}} g_{i}\right)^{2}}{n_{r}^{j}(d)}\right),
$$
where $A_l = {x_i \in A : x_{ij} \le d}$ , $A_r = {x_i \in A : x_{ij} > d}$ , $B_l = {x_i \in B : x_{ij} \le d}$ , $B_r = {x_i \in B : x_{ij} > d}$, and the coefficient $\frac{1−a}{b}$ is used to normalize the sum of the gradients over $B$ back to the size of $A_c$ .

Thus, in GOSS, we use the estimated $\tilde{V}_{j}(d)$ over a smaller instance subset, instead of the accurate $\tilde{V}_{j}(d)$ over all the instances to determine the split point, and the computation cost can be largely reduced. More importantly, the following theorem indicates that GOSS will not lose much training accuracy and will outperform random sampling. Due to space restrictions, we leave the proof of the theorem to the supplementary materials.

> GBDT 使用决策树来学习出一个从输入空间 $\mathcal{X}^s$ 到梯度空间 $\mathcal{G}$ 的映射函数。假设我们有一个数据量为 $n$ 的训练集 $\{x_1,\cdots, x_n\}$，其中每个 $x_i$ 是空间 $\mathcal{X}^s$ 中一个维度为 $s$ 的向量。在每一次梯度提升迭代中，损失函数负梯度在当前模型输出的值表示为 $\{g_1,\cdots, g_n\}$。决策树模型在信息量最大的特征处进行分割（信息增益最大）。对于 GBDT，信息增益通常是通过分裂后的方差来度量的，其定义如下。
>
> **定义3.1** *设 $O$ 为决策树一个固定节点内的数据集。此节点处特征 $j$ 在 $d$ 分割点的方差增益定义为*：
> $$
> \begin{array}{c}
> V_{j \mid O}(d)=\frac{1}{n_{O}}\left(\frac{\left(\sum_{\left\{x_{i} \in O: x_{i j} \leq d\right\}} g_{i}\right)^{2}}{n_{l \mid O}^{j}(d)}+\frac{\left(\sum_{\left\{x_{i} \in O: x_{i j}>d\right\}} g_{i}\right)^{2}}{n_{r \mid O}^{j}(d)}\right) \\
> \end{array}
> $$
> 其中 
> $$
> n_{O}=\sum I\left[x_{i} \in O\right], 
> \\
> n_{l \mid O}^{j}(d)=\sum I\left[x_{i} \in O: x_{i j} \leq d\right] 
> \\
> \text { and }
> \\
> n_{r \mid O}^{j}(d)=\sum I\left[x_{i} \in O: x_{i j}>d\right]
> $$
> 对于特征 $j$，决策树算法选择 $d^∗_j=argmax_dV_j(D)$ 并计算最大增益 $V_j(d_j^*)$。然后，根据特征 $j^∗$ 在点 $d_{j^∗}$ 处将数据分割成左子节点和右子节点。
>
> 该方法首先根据训练实例的梯度绝对值对训练实例进行降序排序；然后保留梯度最大的 $a×100\%$ 实例，得到一个实例子集 $A$；然后，对于剩余的 $A^c$ 集合，包括 $(1−a)×100\%$ 个梯度较小的实例，从中随机采样一个大小为 $b×|A^c|$ 的子集 $B$；最后我们在并集 $A \cup B$ 上根据方差增益 $\tilde{V}_{j}(d)$ 来划分数据，即：
> $$
> \tilde{V}_{j}(d)=\frac{1}{n}\left(\frac{\left(\sum_{x_{i} \in A_{l}} g_{i}+\frac{1-a}{b} \sum_{x_{i} \in B_{l}} g_{i}\right)^{2}}{n_{l}^{j}(d)}+\frac{\left(\sum_{x_{i} \in A_{r}} g_{i}+\frac{1-a}{b} \sum_{x_{i} \in B_{r}} g_{i}\right)^{2}}{n_{r}^{j}(d)}\right),
> $$
> 其中，$A_l = {x_i \in A : x_{ij} \le d}$ , $A_r = {x_i \in A : x_{ij} > d}$ , $B_l = {x_i \in B : x_{ij} \le d}$ , $B_r = {x_i \in B : x_{ij} > d}$，并且系数 $\frac{1−a}{b}$ 用于将 $B$ 上的梯度和归一化回 $A_c$ 的大小。
>
> 因此，在 GOSS 中，我们在较小的实例子集上使用估计的 $\tilde{V}_{j}(d)$ 来确定分割点，而不是在所有实例上使用精确的 $\tilde{V}_{j}(d)$ 来确定分割点，从而可以大大降低计算成本。更重要的是，下面的定理表明，GOSS 将不会损失太多训练精度，并且将优于随机抽样。由于篇幅所限，我们把定理的证明留给补充材料。

**Theorem 3.2** *We denote the approximation error in GOSS as $\mathcal{E}(d)=\left|\tilde{V}_{j}(d)-V_{j}(d)\right|$ and $\bar{g}_{l}^{j}(d)=\frac{\sum_{x_{i} \in\left(A \cup A^{c}\right)_{l}}\left|g_{i}\right|}{n_{1}^{j}(d)}$ ，$\bar{g}_{r}^{j}(d)=\frac{\sum_{x_{i} \in\left(A \cup A^{c}\right)_{r}}\left|g_{i}\right|}{n_{r}^{j}(d)}$ . With probability at least $1 − δ$, we have*
$$
\mathcal{E}(d) \leq C_{a, b}^{2} \ln 1 / \delta \cdot \max \left\{\frac{1}{n_{l}^{j}(d)}, \frac{1}{n_{r}^{j}(d)}\right\}+2 D C_{a, b} \sqrt{\frac{\ln 1 / \delta}{n}},
\qquad(2)
$$
where $C_{a, b}=\frac{1-a}{\sqrt{b}}$ ， $max _{x_{i} \in A^{c}}\left|g_{i}\right|$，and $D=\max \left(\bar{g}_{l}^{j}(d), \bar{g}_{r}^{j}(d)\right)$.

According to the theorem, we have the following discussions: 

1. The asymptotic approximation ratio of GOSS is $\mathcal{O}\left(\frac{1}{n_{l}^{j}(d)}+\frac{1}{n_{r}^{j}(d)}+\frac{1}{\sqrt{n}}\right)$ . If the split is not too unbalanced (i.e., $n_{l}^{j}(d) \geq \mathcal{O}(\sqrt{n})$ and $n_{r}^{j}(d) \geq \mathcal{O}(\sqrt{n})$ ), the approximation error will be dominated by the second term of Ineq.(2) which decreases to $0$ in $O(\sqrt{n})$ with $n \rightarrow \infin$. That means when number of data is large, the approximation is quite accurate. 
2. Random sampling is a special case of GOSS with $a = 0$. In many cases, GOSS could outperform random sampling, under the condition $C_{0, \beta}>C_{a, \beta-a}$, which is equivalent to $\frac{\alpha_{a}}{\sqrt{\beta}}>\frac{1-a}{\sqrt{\beta-a}}$ with $\alpha_{a}=\max _{x_{i} \in A \cup A^{c}}\left|g_{i}\right| / \max _{x_{i} \in A^{c}}\left|g_{i}\right|$.

Next, we analyze the generalization performance in GOSS. We consider the generalization error in GOSS $\mathcal{E}_{g e n}^{G O S S}(d)=\left|\tilde{V}_{j}(d)-V_{*}(d)\right|$, which is the gap between the variance gain calculated by the sampled training instances in GOSS and the true variance gain for the underlying distribution. We have $\mathcal{E}_{\text {gen }}^{G O S S}(d) \leq\left|\tilde{V}_{j}(d)-V_{j}(d)\right|+\left|V_{j}(d)-V_{*}(d)\right| \triangleq \mathcal{E}_{G O S S}(d)+\mathcal{E}_{\text {gen }}(d)$. 

Thus, the generalization error with GOSS will be close to that calculated by using the full data instances if the GOSS approximation is accurate. On the other hand, sampling will increase the diversity of the base learners, which potentially help to improve the generalization performance [24].

>**定理 3.2** *因为概率至少为 $1-\delta$ 我们定义在 GOSS 中的近似误差为 $\mathcal{E}(d)=\left|\tilde{V}_{j}(d)-V_{j}(d)\right|$ 和 $\bar{g}_{l}^{j}(d)=\frac{\sum_{x_{i} \in\left(A \cup A^{c}\right)_{l}}\left|g_{i}\right|}{n_{1}^{j}(d)}$ ，$\bar{g}_{r}^{j}(d)=\frac{\sum_{x_{i} \in\left(A \cup A^{c}\right)_{r}}\left|g_{i}\right|}{n_{r}^{j}(d)}$ .  $1 − \delta$, 我们有*
>$$
>\mathcal{E}(d) \leq C_{a, b}^{2} \ln 1 / \delta \cdot \max \left\{\frac{1}{n_{l}^{j}(d)}, \frac{1}{n_{r}^{j}(d)}\right\}+2 D C_{a, b} \sqrt{\frac{\ln 1 / \delta}{n}},
>\qquad(2)
>$$
>其中 $C_{a, b}=\frac{1-a}{\sqrt{b}}$ ， $max _{x_{i} \in A^{c}}\left|g_{i}\right|$，和 $D=\max \left(\bar{g}_{l}^{j}(d), \bar{g}_{r}^{j}(d)\right)$.
>
>根据以上定理我们有以下讨论：
>
>1. GOSS 算法的渐进逼近率为 $\mathcal{O}\left(\frac{1}{n_{l}^{j}(d)}+\frac{1}{n_{r}^{j}(d)}+\frac{1}{\sqrt{n}}\right)$ 如果划分过于不平衡 ( $n_{l}^{j}(d) \geq \mathcal{O}(\sqrt{n})$ 和 $n_{r}^{j}(d) \geq \mathcal{O}(\sqrt{n})$ )，逼近误差将由公式 (2) 中的第二项主要决定，随着 $n$ 的增大会趋向于 $0$。这意味着当数据量大时，逼近是相当准确的。
>2. 随机采样是 GOSS 算法的一种特殊情况，也就是当 $a=0$ 时。在大多情况下 GOSS 算法都比随机采样表现出色，限定条件为 $C_{0, \beta}>C_{a, \beta-a}$  这相当于 $\frac{\alpha_{a}}{\sqrt{\beta}}>\frac{1-a}{\sqrt{\beta-a}}$ 以及 $\alpha_{a}=\max _{x_{i} \in A \cup A^{c}}\left|g_{i}\right| / \max _{x_{i} \in A^{c}}\left|g_{i}\right|$
>
>接下来，我们对 GOSS 的泛化性能进行了分析。我们考虑了GOSS $\mathcal{E}_{g e n}^{G O S S}(d)=\left|\tilde{V}_{j}(d)-V_{*}(d)\right|$ ，中的泛化误差，也就是采样后训练集的方差增益与原始分布真实方差增益之间的差值。我们有 $\mathcal{E}_{\text {gen }}^{G O S S}(d) \leq\left|\tilde{V}_{j}(d)-V_{j}(d)\right|+\left|V_{j}(d)-V_{*}(d)\right| \triangleq \mathcal{E}_{G O S S}(d)+\mathcal{E}_{\text {gen }}(d)$。
>
>因此，如果 GOSS 算法的近似是准确的，GOSS 的泛化误差接近基于全数据计算得到的泛化误差。从另一方面来说，采样将会提高基学习器的差异性，这样可以潜在地提高泛化能力。

## 4 Exclusive Feature Bundling

In this section, we propose a novel method to effectively reduce the number of features.

High-dimensional data are usually very sparse. The sparsity of the feature space provides us a possibility of designing a nearly lossless approach to reduce the number of features. Specifically, in a sparse feature space, many features are mutually exclusive, i.e., they never take nonzero values simultaneously. 

We can safely bundle exclusive features into a single feature (which we call an exclusive feature bundle). By a carefully designed feature scanning algorithm, we can build the same feature histograms from the feature bundles as those from individual features. 

In this way, the complexity of histogram building changes from O(#data × #feature) to O(#data × #bundle), while #bundle << #feature. Then we can significantly speed up the training of GBDT without hurting the accuracy. In the following, we will show how to achieve this in detail.

There are two issues to be addressed. The first one is to determine which features should be bundled together. The second is how to construct the bundle.

> 在这一部分中，我们提出了一种新的方法来有效地减少特征的数量。
>
> 高维数据通常非常稀疏。 特征空间的稀疏性给我们设计几乎无损方法以减少特征数量的可能性提供了可能。 具体而言，在稀疏的特征空间中，许多特征是互斥的，即它们永远不会同时采用非零值。
>
> 这样，直方图构建的复杂度从 O(#data × # feature) 变为 O(#data × #bundle)，而 #bunch<< #feature。这样就可以在不影响准确度的情况下，大大加快 GBDT 的训练速度。在下面，我们将详细介绍如何实现这一点。
>
> 有两个问题需要解决。
> 第一个是确定哪些 feature 应该 bundle。
> 第二个问题是如何构建 bundle。

**Theorem 4.1** *The problem of partitioning features into a smallest number of exclusive bundles is NP-hard.*

Proof: We will reduce the graph coloring problem [25] to our problem. Since graph coloring problem is NP-hard, we can then deduce our conclusion.

Given any instance $G = (V, E)$ of the graph coloring problem. We construct an instance of our problem as follows. Take each row of the incidence matrix of $G$ as a feature, and get an instance of our problem with $|V|$ features. It is easy to see that an exclusive bundle of features in our problem corresponds to a set of vertices with the same color, and vice versa.

For the first issue, we prove in Theorem 4.1 that it is NP-Hard to find the optimal bundling strategy, which indicates that it is impossible to find an exact solution within polynomial time. In order to find a good approximation algorithm, we first reduce the optimal bundling problem to the graph coloring problem by taking features as vertices and adding edges for every two features if they are not mutually exclusive, then we use a greedy algorithm which can produce reasonably good results (with a constant approximation ratio) for graph coloring to produce the bundles. Furthermore, we notice that there are usually quite a few features, although not 100% mutually exclusive, also rarely take nonzero values simultaneously. 

If our algorithm can allow a small fraction of conflicts, we can get an even smaller number of feature bundles and further improve the computational efficiency. By simple calculation, random polluting a small fraction of feature values will affect the training accuracy by at most $\mathcal{O}([(1 − \gamma)n]^{−2/3})$ (See Proposition 2.1 in the supplementary materials), where $\gamma$ is the maximal conflict rate in each bundle. So, if we choose a relatively small $\gamma$, we will be able to achieve a good balance between accuracy and efficiency.

> 定理 4.1：最优的特征捆绑是 NP-hard。
>
> 证明：我们将图着色问题简化为我们的问题。因为图着色问题是 NP-hard 的，这样可以推断出我们的结论。
>
> 给定图着色问题中的任意样本点 $G=(V,E)$。下面我们构建我们的问题中的一个实例。以 $G$ 的关联矩阵中的每一行作为特征，通过此得到我们的问题中的一个具有 $|V∣$ 个特征的实例。很容易看出，在我们的问题中一组互斥特征对应一组具有相同颜色的顶点，反之亦然。
>
> 对于第一个问题，我们在定理 4.1 中证明了寻找最优 bundle 策略是 NP-hard 的，这表明不可能在多项式时间内找到精确解。为了找到一个好的近似算法，我们首先将最优 bundle 问题归为图着色问题，将特征作为顶点，如果每两个特征不是互斥的，则为每两个特征添加边，然后使用贪婪算法来产生图着色的合理结果(具有恒定的逼近比)。此外，我们注意到通常有相当多的特征，虽然不是 100% 互斥的，但也很少同时采用非零值。
>
> 如果我们的算法可以允许很小一部分冲突，我们可以得到更少的特征 bundle，从而进一步提高计算效率。通过简单计算，随机污染一小部分特征值最多影响训练精度 $\mathcal{O}([(1 − \gamma)n]^{−2/3})$ (见补充资料中的命题2.1)，其中 $\gamma$ 是每个 bundle 中的最大冲突率。因此，如果我们选择一个相对较小的 $\gamma$，我们将能够在精度和效率之间取得很好的平衡。

Based on the above discussions, we design an algorithm for exclusive feature bundling as shown in Alg. 3. First, we construct a graph with weighted edges, whose weights correspond to the total conflicts between features. Second, we sort the features by their degrees in the graph in the descending order. Finally, we check each feature in the ordered list, and either assign it to an existing bundle with a small conflict (controlled by $\gamma$), or create a new bundle. The time complexity of Alg. 3 is $O(\#feature^2 )$ and it is processed only once before training. 

This complexity is acceptable when the number of features is not very large, but may still suffer if there are millions of features. To further improve the efficiency, we propose a more efficient ordering strategy without building the graph: ordering by the count of nonzero values, which is similar to ordering by degrees since more nonzero values usually leads to higher probability of conflicts. Since we only alter the ordering strategies in Alg. 3, the details of the new algorithm are omitted to avoid duplication.

For the second issues, we need a good way of merging the features in the same bundle in order to reduce the corresponding training complexity. The key is to ensure that the values of the original features can be identified from the feature bundles. Since the histogram-based algorithm stores discrete bins instead of continuous values of the features, we can construct a feature bundle by letting exclusive features reside in different bins. This can be done by adding offsets to the original values of the features. 

For example, suppose we have two features in a feature bundle. Originally, feature A takes value from [0, 10) and feature B takes value [0, 20). We then add an offset of 10 to the values of feature B so that the refined feature takes values from [10, 30). After that, it is safe to merge features A and B, and use a feature bundle with range [0, 30] to replace the original features A and B. The detailed algorithm is shown in Alg. 4.

EFB algorithm can bundle many exclusive features to the much fewer dense features, which can effectively avoid unnecessary computation for zero feature values. Actually, we can also optimize the basic histogram-based algorithm towards ignoring the zero feature values by using a table for each feature to record the data with nonzero values. By scanning the data in this table, the cost of histogram building for a feature will change from O(#data) to O(#non_zero_data). 

However, this method needs additional memory and computation cost to maintain these per-feature tables in the whole tree growth process. We implement this optimization in LightGBM as a basic function. Note, this optimization does not conflict with EFB since we can still use it when the bundles are sparse.

![Algorithm3](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Alg3.png)

![Algorithm4](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Alg4.png)

> 基于以上讨论，我们设计出一种针对互斥特征捆绑问题的算法，如 Alg.3 所示。首先，我们构建一个图，其边是带权的，权重由量特征之间的冲突率关联。其次，我们基于特征在图中的度数进行降序排序。最后，我们检查排序好的每个特征，要么将其分配给一个与其冲突率小（由 $\gamma$ 控制）的现有 bundle 特征，要么建立新的捆绑。算法 Alg.3 的时间复杂度为 $O(\#feature^2)$ 且在训练之前仅处理一次。
>
> 当特征数量没有那么大时这个复杂度是可以接受的，但如果面对几百万个特征就会受到很大的影响。为了大大提高效率，我们提出了一种不需要建图的更有效率的排序策略：依据非 $0$ 值的数量进行排序。这与依据度数排序是相似的，因为非 $0$ 值的数量多有几率产生更大的冲突。因为我们只改变了排序策略，所以省略新算法的细节以避免重复。
>
> 对于第二个问题，为了减少训练复杂度，我们需要一个好方法来合并应该 bundle 的两个特征。关键是必须保证能从 bundle 特征值中识别出原始特征的取值。因为基于直方图的算法存储离散的 bins 而不是连续的特征值，我们可以通过使互斥特征分别从属不同的 bins 来构造 bundle 特征。这可以通过给原始特征添加偏移量来完成。
>
> 举个例子，假设在一个 bundle 特征里有两个特征需要合并。开始 $A$ 特征的取值范围是 $[0,10)$，$B$ 特征的取值范围是 $[0,20)$。然后我们给特征 $B$ 添加一个 $10$ 的偏移量，那么特征 $B$ 的取值就变成了 $[10,30)$。之后我们就可以放心的合并特征 $A$ 和 $B$ 去代替原始的 $A$ 和 $B$，合并后的取值为 $[ 0 , 30 ]$ 。算法细节在 Alg.4 中体现。
>
> EFB 算法可以将很多互斥特征捆绑成少量的稠密特征，这样可以避免很多针对特征取值为 0 的不必要的计算。的确，我们可以优化基于直方图的算法，使用一张表来记录每个特征的非 0 取值。通过扫描这张表来建立直方图，每个特征建立直方图的复杂度就从 $O(\#data)$ 变成了 $O(\#{non\_zero\_data})$。然而，在整棵树的建立过程中都需要花费额外的内存和算力去保存和更新这张表。我们在 LightGBM 中实现了这种优化并将其作为一个基本特征。值得注意的是，这种优化方法与 EFB 算法并不是冲突的，因为当捆绑后的特征依然稀疏时我们仍可以使用它。
> 

## 5 Experiments

In this section, we report the experimental results regarding our proposed LightGBM algorithm. We use five different datasets which are all publicly available. The details of these datasets are listed in Table 1. Among them, the Microsoft Learning to Rank (LETOR) [26] dataset contains 30K web search queries. The features used in this dataset are mostly dense numerical features. The Allstate Insurance Claim [27] and the Flight Delay [28] datasets both contain a lot of one-hot coding features. And the last two datasets are from KDD CUP 2010 and KDD CUP 2012. We directly use the features used by the winning solution from NTU [29, 30, 31], which contains both dense and sparse features, and these two datasets are very large. These datasets are large, include both sparse and dense features, and cover many real-world tasks. Thus, we can use them to test our algorithm thoroughly.

Our experimental environment is a Linux server with two E5-2670 v3 CPUs (in total 24 cores) and 256GB memories. All experiments run with multi-threading and the number of threads is fixed to 16.

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Table1.png)

**Table 1: Datasets used in the experiments.**

> 在本节中，我们报告关于我们提出的 LightGBM 算法的实验结果。我们利用五个公开的数据集做了实验。数据集的细节在 Table 1 中列出。这些数据集中，微软的 Learning to Rank（LETOR）数据集包含了 30k 的网络调查问卷。这些数据集中的特征大都是数值型特征。Allstate Insurance Claim和 Flight Delay都含有大量的 one-hot 编码的特征。最后两个数据集来自 KDD CUP 2010 和 KDD CUP 2012。我们直接使用 NTU 的获胜的解决方案中的特征，其中同时包含稠密和稀疏特征且这两个数据集是非常大的。这些数据集都非常大，同时包含稠密和稀疏特征，并且涵盖很多实际中的任务需求。因此，我们可以利用他们彻底测试我们的算法。
>
> 我们的实验环境是一台 Linux 服务器，具有两个 E5-2670v3 CPU(总共24核)和 256 GB内存。所有实验都使用多线程运行，线程数固定为16。

#### 5.1 Overall Comparison

We present the overall comparisons in this subsection. XGBoost [13] and LightGBM without GOSS and EFB (called lgb_baselline) are used as baselines. For XGBoost, we used two versions, xgb_exa (pre-sorted algorithm) and xgb_his (histogram-based algorithm). For xgb_his, lgb_baseline, and LightGBM, we used the leaf-wise tree growth strategy [32]. For xgb_exa, since it only supports layer-wise growth strategy, we tuned the parameters for xgb_exa to let it grow similar trees like other methods. And we also tuned the parameters for all datasets towards a better balancing between speed and accuracy. We set $a = 0.05$, $b = 0.05$ for Allstate, KDD10 and KDD12, and set $a = 0.1$, $b = 0.1$ for Flight Delay and LETOR. We set $\gamma = 0$ in EFB. All algorithms are run for fixed iterations, and we get the accuracy results from the iteration with the best score.6

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Table2.png)

**Table 2: Overall training time cost comparison. LightGBM is lgb_baseline with GOSS and EFB. EFB_only is lgb_baseline with EFB. The values in the table are the average time cost (seconds) for training one iteration.**

![Table3](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Table3.png)

**Table 3: Overall accuracy comparison on test datasets. Use AUC for classification task and NDCG@10 for ranking task. SGB is lgb_baseline with Stochastic Gradient Boosting, and its sampling ratio is the same as LightGBM.**

> 我们在这一小节中介绍整体比较。XGBoost [13]和 不带 GOSS 和 EFB 的LightGBM (称为 lgb_baselline)被用作 baseline。对于XGBoost，我们使用了两个版本，xgb_exa (预排序算法)和 xgb_his(基于直方图的算法)。对于 xgb_his、lgb_baseline 和 LightGBM，我们使用 leaf-wise 生长策略[32]。对于 xgb_exa，因为它只支持 layer-wise 生长策略，所以我们对 xgb_exa 进行了调参，使其可以像其他方法一样生长类似的树。我们还调整了所有数据集的参数，以便在速度和准确性之间取得更好的平衡。我们为 Allstate、KDD10 和 KDD12 设置了 $a=0.05$ 、$b=0.05$，为 Flight Delay 和 LETOR 设置了$a=0.1$、$b=0.1$。我们在 EFB 中设置了 $\gamma=0$。所有算法都运行的迭代次数都是固定的，我们取最优迭代结果作为最终结果。

The training time and test accuracy are summarized in Table 2 and Table 3 respectively. From these results, we can see that LightGBM is the fastest while maintaining almost the same accuracy as baselines. The xgb_exa is based on the pre-sorted algorithm, which is quite slow comparing with histogram-base algorithms. By comparing with lgb_baseline, LightGBM speed up 21x, 6x, 1.6x, 14x and 13x respectively on the Allstate, Flight Delay, LETOR, KDD10 and KDD12 datasets. Since xgb_his is quite memory consuming, it cannot run successfully on KDD10 and KDD12 datasets due to out-of-memory. 

On the remaining datasets, LightGBM are all faster, up to 9x speed-up is achieved on the Allstate dataset. The speed-up is calculated based on training time per iteration since all algorithms converge after similar number of iterations. To demonstrate the overall training process, we also show the training curves based on wall clock time on Flight Delay and LETOR in the Fig. 1 and Fig. 2, respectively. To save space, we put the remaining training curves of the other datasets in the supplementary material.

On all datasets, LightGBM can achieve almost the same test accuracy as the baselines. This indicates that both GOSS and EFB will not hurt accuracy while bringing significant speed-up. It is consistent with our theoretical analysis in Sec. 3.2 and Sec. 4.

LightGBM achieves quite different speed-up ratios on these datasets. The overall speed-up comes from the combination of GOSS and EFB, we will break down the contribution and discuss the effectiveness of GOSS and EFB separately in the next sections.

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Fig1.png)

**Figure 1: Time-AUC curve on Flight Delay.**

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/LGBM/Fig2.png)

**Figure 2: Time-NDCG curve on LETOR.**

> 训练时间和测试精度在 Table 2 和 Table 3 中分别总结了。从这些结果中我们可以看出，LightGBM 在维持与 baseline 几乎相同的精度的同时是最快的。xgb_exa 基于预排序算法，相比基于直方图算法的慢了很多。通过与 lgb_baseline 对比，LightGBM 在 数据集 Allstate，Flight Delay，LETOR，KDD10 和 KDD12 中分别加速了 21x，6x，1.6x，14x 和 13x。由于 xgb_his 内存消耗非常大，因为内存不足无法成功在数据集 KDD10 和 KDD12 上跑通。在其余的数据集上，LightGBM 都更快，在数据集 Allstate 上达到了 9x的加速。加速的计算是基于每次训练迭代消耗的时间，因为所有算法经过差不多相同的迭代次数后都会收敛。为了展示整个训练过程，我们在 图1 和 图2 中分别展示了数据集 Flight Delay 和 LETOR 上基于时间的训练曲线。为了节省空间，我们把其他数据集的训练曲线放在补充材料中展示。
>
> 在所有的数据集上，LightGBM可以达到与基线几乎相同的精度。这说明GOSS 和 EFB 在显著加速的同时不会影响精度。这与我们在 3.2 和 4 中的理论分析是一致的。

#### 5.2 Analysis on GOSS

First, we study the speed-up ability of GOSS. From the comparison of LightGBM and EFB_only (LightGBM without GOSS) in Table 2, we can see that GOSS can bring nearly 2x speed-up by its own with using 10% - 20% data. GOSS can learn trees by only using the sampled data. However, it retains some computations on the full dataset, such as conducting the predictions and computing the gradients. Thus, we can find that the overall speed-up is not linearly correlated with the percentage of sampled data. However, the speed-up brought by GOSS is still very significant and the technique is universally applicable to different datasets.

Second, we evaluate the accuracy of GOSS by comparing with Stochastic Gradient Boosting (SGB) [20]. Without loss of generality, we use the LETOR dataset for the test. We tune the sampling ratio by choosing different a and b in GOSS, and use the same overall sampling ratio for SGB. We run these settings until convergence by using early stopping. The results are shown in Table 4. We can see the accuracy of GOSS is always better than SGB when using the same sampling ratio. These results are consistent with our discussions in Sec. 3.2. All the experiments demonstrate that GOSS is a more effective sampling method than stochastic sampling.

> 首先，我们研究 GOSS 的加速能力。从 表2 的 LightGBM 和EFB_only（LightGBM without GOSS），我们可以看出 GOSS 通过利用10%-20% 的数据可以带来 2x 的加速。GOSS 算法仅仅利用采样后的数据来学习决策树。然而，它保留了对整个数据集的一些计算，例如进行预测和计算梯度。因此，我们可以发现，总体加速与采样数据的百分比不是线性相关的。然而，GOSS 带来的加速仍然是非常显著的，并且该技术对不同的数据集具有普遍的适用性。
>
> 第二，通过与随机梯度增强算法(SGB)[20]的比较，对 GOSS 算法的准确性进行了评估。在不影响泛化性的情况下，我们使用 LETOR 数据集进行测试。我们通过在 GOSS 中选择不同的 $a$ 和 $b$ 来调整采样率，并对 SGB 使用相同的总体采样率。我们通过使用 early-stopping 来运行这些设置，直到收敛。结果如 表4 所示。我们可以看到，在使用相同的采样率时，GOSS 的精度总是好于 SGB。

#### 5.3 Analysis on EFB

We check the contribution of EFB to the speed-up by comparing lgb_baseline with EFB_only. The results are shown in Table 2. Here we do not allow the confliction in the bundle finding process (i.e., $\gamma = 0$).7 We find that EFB can help achieve significant speed-up on those large-scale datasets.

Please note lgb_baseline has been optimized for the sparse features, and EFB can still speed up the training by a large factor. It is because EFB merges many sparse features (both the one-hot coding features and implicitly exclusive features) into much fewer features. The basic sparse feature optimization is included in the bundling process. However, the EFB does not have the additional cost on maintaining nonzero data table for each feature in the tree learning process. What is more, since many previously isolated features are bundled together, it can increase spatial locality and improve cache hit rate significantly. Therefore, the overall improvement on efficiency is dramatic. With above analysis, EFB is a very effective algorithm to leverage sparse property in the histogram-based algorithm, and it can bring a significant speed-up for GBDT training process.

> 我们通过比较 lgb_baseline 和 EFB_only 来检查 EFB 对加速的贡献。结果如 表2 所示。在这里，这里我们不允许捆绑特征之间有冲突率（(即，$\gamma=0$ )7。我们发现 EFB 可以帮助在这些大规模数据集上实现显著的加速。
>
> 请注意 lgb_baseline已经对稀疏特征进行了优化，EFB算法仍然可以（在此基础上）极大的提高训练速度。这是因为 EFB 将许多稀疏特征（包括 one-hot 编码特征和 隐式互斥特征）合并成更少的特征。bundling 的过程中包含了基本的稀疏特征优化。然而，EFB不需要在树模型的学习过程中为每个特征记录非零数据表花费额外成本。更重要的是，由于许多先前独立的特征捆绑在一起，它可以增加空间局部性并显著提高缓存命中率。因此，整体效率的提高是显著的。通过以上分析，EFB 是一种非常有效的基于直方图算法的稀疏特征处理算法，可以为 GBDT 训练过程显著加速。

## 6 Conclusion

In this paper, we have proposed a novel GBDT algorithm called LightGBM, which contains two novel techniques: Gradient-based One-Side Sampling and Exclusive Feature Bundling to deal with large number of data instances and large number of features respectively. We have performed both theoretical analysis and experimental studies on these two techniques. The experimental results are consistent with the theory and show that with the help of GOSS and EFB, LightGBM can significantly outperform XGBoost and SGB in terms of computational speed and memory consumption. For the future work, we will study the optimal selection of $a$ and $b$ in Gradient-based One-Side Sampling and continue improving the performance of Exclusive Feature Bundling to deal with large number of features no matter they are sparse or not.

> 本文提出了一种新的 GBDT 算法 LightGBM，它包含两种新技术：基于梯度的单边采样和互斥特征捆绑，分别用于处理大量的数据实例和大量的特征。我们对这两种技术进行了理论分析和实验研究。实验结果与理论一致，表明在 GOSS 和 EFB 的帮助下，LightGBM 在计算速度和内存消耗方面明显优于 XGBoost 和 SGB。在未来的工作中，我们将研究基于梯度的单边采样中 $a$ 和 $b$ 的最优选择，并继续提高互斥特征捆绑的性能，以处理大量稀疏或非稀疏特征。

## References

 [1] Jerome H Friedman. Greedy function approximation: a gradient boosting machine. Annals of statistics, pages 1189–1232, 2001. 

[2] Ping Li. Robust logitboost and adaptive base class (abc) logitboost. arXiv preprint arXiv:1203.3491, 2012. 

[3] Matthew Richardson, Ewa Dominowska, and Robert Ragno. Predicting clicks: estimating the click-through rate for new ads. In Proceedings of the 16th international conference on World Wide Web, pages 521–530. ACM, 2007.

[4] Christopher JC Burges. From ranknet to lambdarank to lambdamart: An overview. Learning, 11(23-581):81, 2010. 

[5] Jerome Friedman, Trevor Hastie, Robert Tibshirani, et al. Additive logistic regression: a statistical view of boosting (with discussion and a rejoinder by the authors). The annals of statistics, 28(2):337–407, 2000. 

[6] Charles Dubout and François Fleuret. Boosting with maximum adaptive sampling. In Advances in Neural Information Processing Systems, pages 1332–1340, 2011. 

[7] Ron Appel, Thomas J Fuchs, Piotr Dollár, and Pietro Perona. Quickly boosting decision trees-pruning underachieving features early. In ICML (3), pages 594–602, 2013. 

[8] Manish Mehta, Rakesh Agrawal, and Jorma Rissanen. Sliq: A fast scalable classifier for data mining. In International Conference on Extending Database Technology, pages 18–32. Springer, 1996. 

[9] John Shafer, Rakesh Agrawal, and Manish Mehta. Sprint: A scalable parallel classi er for data mining. In Proc. 1996 Int. Conf. Very Large Data Bases, pages 544–555. Citeseer, 1996. 

[10] Sanjay Ranka and V Singh. Clouds: A decision tree classifier for large datasets. In Proceedings of the 4th Knowledge Discovery and Data Mining Conference, pages 2–8, 1998. 

[11] Ruoming Jin and Gagan Agrawal. Communication and memory efficient parallel decision tree construction. In Proceedings of the 2003 SIAM International Conference on Data Mining, pages 119–129. SIAM, 2003. 

[12] Ping Li, Christopher JC Burges, Qiang Wu, JC Platt, D Koller, Y Singer, and S Roweis. Mcrank: Learning to rank using multiple classification and gradient boosting. In NIPS, volume 7, pages 845–852, 2007. 

[13] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of the 22Nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM, 2016. 

[14] Stephen Tyree, Kilian Q Weinberger, Kunal Agrawal, and Jennifer Paykin. Parallel boosted regression trees for web search ranking. In Proceedings of the 20th international conference on World wide web, pages 387–396. ACM, 2011. 

[15] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. Journal of Machine Learning Research, 12(Oct):2825–2830, 2011. 

[16] Greg Ridgeway. Generalized boosted models: A guide to the gbm package. Update, 1(1):2007, 2007. 

[17] Huan Zhang, Si Si, and Cho-Jui Hsieh. Gpu-acceleration for large-scale tree boosting. arXiv preprint arXiv:1706.08359, 2017. 

[18] Rory Mitchell and Eibe Frank. Accelerating the xgboost algorithm using gpu computing. PeerJ Preprints, 5:e2911v1, 2017. 

[19] Qi Meng, Guolin Ke, Taifeng Wang, Wei Chen, Qiwei Ye, Zhi-Ming Ma, and Tieyan Liu. A communication-efficient parallel algorithm for decision tree. In Advances in Neural Information Processing Systems, pages 1271–1279, 2016. 

[20] Jerome H Friedman. Stochastic gradient boosting. Computational Statistics & Data Analysis, 38(4):367–378, 2002. 

[21] Michael Collins, Robert E Schapire, and Yoram Singer. Logistic regression, adaboost and bregman distances. Machine Learning, 48(1-3):253–285, 2002. 

[22] Ian Jolliffe. Principal component analysis. Wiley Online Library, 2002.

[23] Luis O Jimenez and David A Landgrebe. Hyperspectral data analysis and supervised feature reduction via projection pursuit. IEEE Transactions on Geoscience and Remote Sensing, 37(6):2653–2667, 1999. 

[24] Zhi-Hua Zhou. Ensemble methods: foundations and algorithms. CRC press, 2012. 

[25] Tommy R Jensen and Bjarne Toft. Graph coloring problems, volume 39. John Wiley & Sons, 2011. 

[26] Tao Qin and Tie-Yan Liu. Introducing LETOR 4.0 datasets. CoRR, abs/1306.2597, 2013. 

[27] Allstate claim data, https://www.kaggle.com/c/ClaimPredictionChallenge. 

[28] Flight delay data, https://github.com/szilard/benchm-ml#data. 

[29] Hsiang-Fu Yu, Hung-Yi Lo, Hsun-Ping Hsieh, Jing-Kai Lou, Todd G McKenzie, Jung-Wei Chou, Po-Han Chung, Chia-Hua Ho, Chun-Fu Chang, Yin-Hsuan Wei, et al. Feature engineering and classifier ensemble for kdd cup 2010. In KDD Cup, 2010. 

[30] Kuan-Wei Wu, Chun-Sung Ferng, Chia-Hua Ho, An-Chun Liang, Chun-Heng Huang, Wei-Yuan Shen, Jyun-Yu Jiang, Ming-Hao Yang, Ting-Wei Lin, Ching-Pei Lee, et al. A two-stage ensemble of diverse models for advertisement ranking in kdd cup 2012. In KDDCup, 2012. 

[31] Libsvm binary classification data, https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html. 

[32] Haijian Shi. Best-first decision tree learning. PhD thesis, The University of Waikato, 2007.