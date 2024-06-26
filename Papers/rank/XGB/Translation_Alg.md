> 华盛顿大学，陈天奇

# XGBoost: A Scalable Tree Boosting System

## ABSTRACT

Tree boosting is a highly effective and widely used machine learning method. In this paper, we describe a scalable endto-end tree boosting system called XGBoost, which is used widely by data scientists to achieve state-of-the-art results on many machine learning challenges. We propose a novel sparsity-aware algorithm for sparse data and weighted quantile sketch for approximate tree learning. More importantly, we provide insights on cache access patterns, data compression and sharding to build a scalable tree boosting system. By combining these insights, XGBoost scales beyond billions of examples using far fewer resources than existing systems.

> 提升树是一种高效、应用广泛的机器学习方法。在本文中，我们描述了一个可扩展的 end-to-end 提升树系统 XGBoost，它被数据科学家广泛使用，以在许多机器学习挑战中获得最先进的结果。我们提出了一种新的稀疏感知算法，用于稀疏数据和用于近似树学习的加权分位数草图。
>
> 更重要的是，我们提供了有关缓存访问模式、数据压缩和分片的见解，以构建可扩展的提升树系统。通过结合这些见解，XGBoost 使用比现有系统少得多的资源来扩展数十亿个示例。

## 1. INTRODUCTION

Machine learning and data-driven approaches are becoming very important in many areas. Smart spam classifiers protect our email by learning from massive amounts of spam data and user feedback; advertising systems learn to match the right ads with the right context; fraud detection systems protect banks from malicious attackers; anomaly event detection systems help experimental physicists to find events that lead to new physics. There are two important factors that drive these successful applications: usage of effective (statistical) models that capture the complex data dependencies and scalable learning systems that learn the model of interest from large datasets.

Among the machine learning methods used in practice, gradient tree boosting [10] 1 is one technique that shines in many applications. Tree boosting has been shown to give state-of-the-art results on many standard classification benchmarks [16]. LambdaMART [5], a variant of tree boosting for ranking, achieves state-of-the-art result for ranking problems. Besides being used as a stand-alone predictor, it is also incorporated into real-world production pipelines for ad click through rate prediction [15]. Finally, it is the defacto choice of ensemble method and is used in challenges such as the Netflix prize [3].

In this paper, we describe XGBoost, a scalable machine learning system for tree boosting. The system is available as an open source package2 . The impact of the system has been widely recognized in a number of machine learning and data mining challenges. Take the challenges hosted by the machine learning competition site Kaggle for example. Among the 29 challenge winning solutions 3 published at Kaggle’s blog during 2015, 17 solutions used XGBoost. Among these solutions, eight solely used XGBoost to train the model, while most others combined XGBoost with neural nets in ensembles. For comparison, the second most popular method, deep neural nets, was used in 11 solutions. The success of the system was also witnessed in KDDCup 2015, where XGBoost was used by every winning team in the top-10. Moreover, the winning teams reported that ensemble methods outperform a well-configured XGBoost by only a small amount [1].

> 机器学习和数据驱动的方法在许多领域变得非常重要。智能垃圾邮件分类器通过学习海量垃圾数据和用户反馈来保护我们的电子邮件；广告系统学习将正确的广告与正确的上下文相匹配；欺诈检测系统保护银行免受恶意攻击者的攻击；
>
> 驱动这些成功应用程序的两个重要因素是：使用有效的(统计)模型来捕获复杂的数据依赖关系，以及可扩展的学习系统从大数据集中学习模型。
>
> 在实际使用的机器学习方法中，梯度提升树[10]1是一种具有广泛应用前景的技术。提升树已被证明在许多标准分类基准上给出了最先进的结果[16]。
> LambdaMART[5]，是用于排序的提升树的变体，在排序问题上获得了最先进的结果。除了被用作独立的预测器外，它还被整合到现实世界的 pielines 中，用于广告点击率预测[15]。最后，它是 集成方法的 defacto 选择，并被用于挑战如Netflix奖的挑战[3]。
>
> 本文描述了 XGBoost，一个可扩展的提升树机器学习系统。该系统以开放源码软件包2的形式提供。该系统的影响在许多机器学习和数据挖掘挑战中得到了广泛认可。以机器学习竞赛网站 Kaggle 举办的挑战为例。在 2015 年发布在 Kaggle 博客上的 29 个挑战制胜解决方案 3 中，有 17 个解决方案使用了 XGBoost。在这些解决方案中，有 8 个只使用了 XGBoost 来训练模型，而其他大多数解决方案则将 XGBoost 与神经网络结合在一起进行集成。作为比较，第二种流行的方法，深度神经网络，共有 11 种解决方案使用。该系统的成功也在 2015 年的 KDDCup 比赛中得到了见证，前 10 名中的每一支获胜队伍都使用了 XGBoost。此外，获胜的团队报告说，集成方法的性能仅比配置良好的 XGBoost 好一点[1]。

These results demonstrate that our system gives state-of-the-art results on a wide range of problems. Examples of the problems in these winning solutions include: store sales prediction; high energy physics event classification; web text classification; customer behavior prediction; motion detection; ad click through rate prediction; malware classification; product categorization; hazard risk prediction; massive online course dropout rate prediction. While domain dependent data analysis and feature engineering play an important role in these solutions, the fact that XGBoost is the consensus choice of learner shows the impact and importance of our system and tree boosting.

The most important factor behind the success of XGBoost is its scalability in all scenarios. The system runs more than ten times faster than existing popular solutions on a single machine and scales to billions of examples in distributed or memory-limited settings. The scalability of XGBoost is due to several important systems and algorithmic optimizations. These innovations include: a novel tree learning algorithm is for handling sparse data; a theoretically justified weighted quantile sketch procedure enables handling instance weights in approximate tree learning. Parallel and distributed computing makes learning faster which enables quicker model exploration. More importantly, XGBoost exploits out-of-core computation and enables data scientists to process hundred millions of examples on a desktop. Finally, it is even more exciting to combine these techniques to make an end-to-end system that scales to even larger data with the least amount of cluster resources. The major contributions of this paper is listed as follows:

- We design and build a highly scalable end-to-end tree boosting system.
- We propose a theoretically justified weighted quantile sketch for efficient proposal calculation.
- We introduce a novel sparsity-aware algorithm for parallel tree learning.
- We propose an effective cache-aware block structure for out-of-core tree learning.

> 这些结果表明，我们的系统在广泛的问题上提供了最先进的结果。这些获奖解决方案中的解决的问题包括：商店销量预测；高能物理事件分类；网页文本分类；客户行为预测；运动检测；广告点击率预测；恶意软件分类；产品分类；危险风险预测；海量在线课程辍学率预测。虽然领域相关的数据分析和特征工程在这些解决方案中扮演着重要的角色，但 XGBoost 是学习者的共识选择，这一事实表明了我们的系统和提升树的影响和重要性。
>
> XGBoost 成功的最重要的因素是它在所有场景中的可扩展。该系统在一台机器上的运行速度比现有的流行解决方案快十倍以上，在分布式或内存有限的设置下可以扩展到数十亿个示例。XGBoost 的可扩展性得益于几个重要的系统和算法优化。这些创新包括：新的树学习算法用于处理稀疏数据；理论上合理的 weighted quantile sketch 过程使得能够在近似树学习中处理实例权重。并行和分布式计算使学习速度更快，从而能够更快地探索模型。更重要的是，XGBoost 利用了 out-of-core 计算，使数据科学家能够在终端上处理数亿个示例。最后，更令人兴奋的是，将这些技术结合起来形成一个end-to-end 系统，可以用最少的集群资源扩展到更大的数据。
>
> 本文的主要贡献如下：
>
> - 我们设计并构建了一个高度可扩展的端到端提升树系统。
> - 我们提出了一种理论上合理的 weighted quantile sketch，用于高效计算。
> - 我们提出了一种新的稀疏性感知并行树学习算法。
> - 我们提出了一种有效的缓存感知块结构，用于 out-of-core 学习。

While there are some existing works on parallel tree boosting [22, 23, 19], the directions such as out-of-core computation, cache-aware and sparsity-aware learning have not been explored. More importantly, an end-to-end system that combines all of these aspects gives a novel solution for real-world use-cases. This enables data scientists as well as researchers to build powerful variants of tree boosting algorithms [7, 8]. Besides these major contributions, we also make additional improvements in proposing a regularized learning objective, which we will include for completeness. 

The remainder of the paper is organized as follows. We will first review tree boosting and introduce a regularized objective in Sec. 2. We then describe the split finding methods in Sec. 3 as well as the system design in Sec. 4, including experimental results when relevant to provide quantitative support for each optimization we describe. Related work is discussed in Sec. 5. Detailed end-to-end evaluations are included in Sec. 6. Finally we conclude the paper in Sec. 7.

> 虽然已有一些关于并行提升树的工作[22，23，19]，但还没有探索到诸如 out-of-core 计算、缓存感知和稀疏感知学习等方向。更重要的是，结合了所有这些方面的 end-to-end 系统为现实世界的用例提供了一种新颖的解决方案。这使得数据科学家以及研究人员能够构建强大的提升树算法变种[7，8]。除了这些主要贡献之外，我们还在提出正则化学习目标方面做出了额外的改进，我们将把它包括进来，以确保完整性。论文的其余部分组织如下。W
>
> 论文的其余部分组织如下。我们将首先回顾树的提升，并在第二节引入一个正则化目标。然后，第三节描述分裂查找方法以及第四节中的系统设计，包括相关的实验结果，以便为我们描述的每一种优化提供定量支持。第五节讨论了相关工作。第六节包括了详细的 end-to-end 评估。最后，我们在第七节对论文进行了总结。

## 2. TREE BOOSTING IN A NUTSHELL（果壳中的提升树）

We review gradient tree boosting algorithms in this section. The derivation follows from the same idea in existing literatures in gradient boosting. Specicially the second order method is originated from Friedman et al. [12]. We make minor improvements in the reguralized objective, which were found helpful in practice.

> 在这一部分中，我们回顾了梯度提升树算法。这一推导源于现有文献中关于梯度提升的相同思想。具体地说，二阶方法起源于 Friedman 等人。
> [12]。我们在重新定位的目标上做了一些小的改进，这在实践中被发现是有帮助的。

#### 2.1 Regularized Learning Objective

For a given data set with $n$ examples and $m$ features $\mathcal{D}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}\left(|\mathcal{D}|=n, \mathbf{x}_{i} \in \mathbb{R}^{m}, y_{i} \in \mathbb{R}\right)$, a tree ensemble model (shown in Fig. 1) uses $K$ additive functions to predict the output.
$$
\hat{y}_{i}=\phi\left(\mathbf{x}_{i}\right)=\sum_{k=1}^{K} f_{k}\left(\mathbf{x}_{i}\right), \quad f_{k} \in \mathcal{F}
\qquad(1)
$$
where $ \mathcal{F}=\left\{f(\mathbf{x})=w_{q(\mathbf{x})}\right\}\left(q: \mathbb{R}^{m} \rightarrow T, w \in \mathbb{R}^{T}\right)$ is the space of regression trees (also known as CART). Here $q$ represents the structure of each tree that maps an example to the corresponding leaf index. $T$ is the number of leaves in the tree. Each $f_k$ corresponds to an independent tree structure $q$ and leaf weights $w$. Unlike decision trees, each regression tree contains a continuous score on each of the leaf, we use wi to represent score on $i$-th leaf. For a given example, we will use the decision rules in the trees (given by $q$) to classify it into the leaves and calculate the final prediction by summing up the score in the corresponding leaves (given by $w$). To learn the set of functions used in the model, we minimize the following regularized objective.

Here $l$ is a differentiable convex loss function that measures the difference between the prediction $\hat{y}_i$ and the target $y_i$. The second term $\Omega$ penalizes the complexity of the model (i.e., the regression tree functions). The additional regularization term helps to smooth the final learnt weights to avoid over-fitting. Intuitively, the regularized objective will tend to select a model employing simple and predictive functions. A similar regularization technique has been used in Regularized greedy forest (RGF) [25] model. Our objective and the corresponding learning algorithm is simpler than RGF and easier to parallelize. When the regularization parameter is set to zero, the objective falls back to the traditional gradient tree boosting.

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Fig1.png)

**Figure 1: Tree Ensemble Model. The final prediction for a given example is the sum of predictions from each tree.**

> 对于具有 $n$ 个示例和 $m$ 个特征 $\mathcal{D}=\left\{\left(\mathbf{x}_{i}, y_{i}\right)\right\}\left(|\mathcal{D}|=n, \mathbf{x}_{i} \in \mathbb{R}^{m}, y_{i} \in \mathbb{R}\right)$ 的给定数据集，集成的树模型(如图1所示)使用 $K$ 个函数加法来预测输出。
> $$
> \hat{y}_{i}=\phi\left(\mathbf{x}_{i}\right)=\sum_{k=1}^{K} f_{k}\left(\mathbf{x}_{i}\right), \quad f_{k} \in \mathcal{F}
> \qquad(1)
> $$
> 其中 $\mathcal{F}=\left\{f(\mathbf{x})=w_{q(\mathbf{x})}\right\}\left(q: \mathbb{R}^{m} \rightarrow T, w \in \mathbb{R}^{T}\right)$ 是回归树的空间(也称为CART)。这里 $q$ 表示将示例映射到相应叶索引的每棵树的结构。$T$是树中的叶数。每个 $f_k$ 对应于独立的树结构 $q$ 和叶权重 $w$。与决策树不同，每个回归树都包含每个叶的连续分数，我们使用 $w_i$ 表示第 $i$ 个叶的分数。对于给定的示例，我们将使用树中的决策规则(由 $q$ 给出)来将其分类为叶子，并通过将相应叶子(由 $w$ 给出)中的得分相加来计算最终预测。为了学习模型中使用的函数集，我们最小化以下带正则化目标。
> $$
> \begin{array}{l}
> \mathcal{L}(\phi)=\sum_{i} l\left(\hat{y}_{i}, y_{i}\right)+\sum_{k} \Omega\left(f_{k}\right) \\
> \text { where } \Omega(f)=\gamma T+\frac{1}{2} \lambda\|w\|^{2}
> \end{array}
> 
> \qquad(2)
> $$
> 这里，$l$ 是一个可微的凸函数，它度量预测 $\hat{y}_i$ 和目标 $y_i$ 之间的差异。第二个术语 $\Omega$ 对应于模型的复杂度的惩罚(即回归树函数)。附带的正则化项有助于平滑最终学习的权重，以避免过拟合。
>
> 直观地说，正则化目标将倾向于选择使用简单和泛化性好的函数的模型。在正则化贪婪森林(RGF)[25]模型中也使用了类似的正则化技术。我们的目标和相应的学习算法比RGF更简单，更容易并行化。当正则化参数设置为零时，目标退化为传统的梯度提升树。

#### 2.2 Gradient Tree Boosting

The tree ensemble model in Eq. (2) includes functions as parameters and cannot be optimized using traditional optimization methods in Euclidean space. Instead, the model is trained in an additive manner. Formally, let $\hat{y}^{(t)}_i$ be the prediction of the $i$-th instance at the $t$-th iteration, we will need to add $f_t$ to minimize the following objective.
$$
\mathcal{L}^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right)\right)+\Omega\left(f_{t}\right)
$$
This means we greedily add the $f_t$ that most improves our model according to Eq. (2). Second-order approximation can be used to quickly optimize the objective in the general setting [12].
$$
\mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
$$
where $g_{i}=\partial_{\hat{y}^{(t-1)}} l\left(y_{i}, \hat{y}^{(t-1)}\right)$ and $h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)$ are first and second order gradient statistics on the loss function. We can remove the constant terms to obtain the following simplified objective at step $t$.
$$
\tilde{\mathcal{L}}^{(t)}=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right) \qquad(3)
$$
Define $I_{j}=\left\{i \mid q\left(\mathbf{x}_{i}\right)=j\right\}$ as the instance set of leaf $j$. We can rewrite Eq (3) by expanding $\Omega$ as follows
$$
\begin{aligned}
\tilde{\mathcal{L}}^{(t)} &=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\
&=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T
\end{aligned}
\qquad (4)
$$
For a fixed structure $q(\mathbb{x})$, we can compute the optimal weight $w^∗_j$ of leaf $j$ by
$$
w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}
\qquad (5)
$$
and calculate the corresponding optimal value by
$$
\tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T
\qquad (6)
$$
Eq (6) can be used as a scoring function to measure the quality of a tree structure $q$. This score is like the impurity score for evaluating decision trees, except that it is derived for a wider range of objective functions. Fig. 2 illustrates how this score can be calculated.

Normally it is impossible to enumerate all the possible tree structures $q$. A greedy algorithm that starts from a single leaf and iteratively adds branches to the tree is used instead. Assume that $I_L$ and $I_R$ are the instance sets of left and right nodes after the split. Lettting $I = I_L ∪ I_R$, then the loss reduction after the split is given by
$$
\mathcal{L}_{\text {split }}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma
\qquad(7)
$$
This formula is usually used in practice for evaluating the split candidates.

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Fig2.png)

**Figure 2: Structure Score Calculation. We only need to sum up the gradient and second order gradient statistics on each leaf, then apply the scoring formula to get the quality score.**

> Eq.(2)中的集成树模型的参数包括函数，不能在欧式空间用传统的方法进行优化。取而代之的是，以加法模型的方式训练模型。形式上，假设 $\hat{y}^{(T)}_i$ 是第 $t$ 次迭代的第 $i$ 个实例的预测，我们将需要添加 $f_t$ 以最小化以下目标。
> $$
> \mathcal{L}^{(t)}=\sum_{i=1}^{n} l\left(y_{i}, \hat{y}_{i}^{(t-1)}+f_{t}\left(\mathbf{x}_{i}\right)\right)+\Omega\left(f_{t}\right)
> $$
> 这意味着我们根据 Eq.(2)以贪婪方式添加最能改善我们模型的 $f_t$。在一般设置[12]中，二阶近似可以用来快速优化目标。
> $$
> \mathcal{L}^{(t)} \simeq \sum_{i=1}^{n}\left[l\left(y_{i}, \hat{y}^{(t-1)}\right)+g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right)
> $$
> 其中 $g_{i}=\partial_{\hat{y}^{(t-1)}} l\left(y_{i}, \hat{y}^{(t-1)}\right)$ 和 $h_{i}=\partial_{\hat{y}^{(t-1)}}^{2} l\left(y_{i}, \hat{y}^{(t-1)}\right)$ 是损失的一阶和二阶梯度统计量。我们可以在步骤 $t$ 中去掉常数项以获得以下简化目标。
> $$
> \tilde{\mathcal{L}}^{(t)}=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\Omega\left(f_{t}\right) \qquad(3)
> $$
> 定义 $I_{j}=\left\{i \mid q\left(\mathbf{x}_{i}\right)=j\right\}$ 作为叶子节点 $j$ 的实例集。我们可以通过展开 $\Omega$ 来重写公式(3)，如下所示
> $$
> \begin{aligned}
> \tilde{\mathcal{L}}^{(t)} &=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\
> &=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T
> \end{aligned}
> \qquad (4)
> $$
> 对于固定结构 $q(\mathbb{x})$，我们可以计算叶子节点 $j$ 的最优权重 $w^∗_j$
> $$
> w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda}
> \qquad (5)
> $$
> 并通过以下方式计算相应的最优值
> $$
> \tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T
> \qquad (6)
> $$
> Eq(6) 可以作为一个评分函数来衡量树结构 $q$ 的质量。这个分数类似于用于评估决策树的 impurity 分数，只不过它是为更广泛的目标函数派生的。图2 说明了如何计算这个分数。
> $$
> \mathcal{L}_{\text {split }}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma
> \qquad(7)
> $$
> 这一公式在实践中通常用于评估分裂的候选。

#### 2.3 Shrinkage and Column Subsampling

Besides the regularized objective mentioned in Sec. 2.1, two additional techniques are used to further prevent overfitting. 

The first technique is shrinkage introduced by Friedman [11]. Shrinkage scales newly added weights by a factor $\eta$ after each step of tree boosting. Similar to a learning rate in tochastic optimization, shrinkage reduces the influence of each individual tree and leaves space for future trees to improve the model. 

The second technique is column (feature) subsampling. This technique is used in RandomForest [4,13], It is implemented in a commercial software TreeNet 4 for gradient boosting, but is not implemented in existing opensource packages. 

According to user feedback, using column sub-sampling prevents over-fitting even more so than the traditional row sub-sampling (which is also supported). The usage of column sub-samples also speeds up computations of the parallel algorithm described later.

> 除了 Sec.2.1 提到的带有正则化的目标函数之外。还使用了两种其他的技术来进一步防止过拟合。
>
> 第一种是 friedman 介绍的 shrinkage [11]。
>
> 在提升树的每一步，使用 $\eta$ 因子缩放新添加的权重。与随机优化中的学习率类似，shrinkage 减少了每棵树的影响，并为未来的树去改进模型留出了空间。
>
> 第二种是列(特征)子抽样。
>
> 这项技术在 RandomForest[4，13]中使用，它是在商业软件 TreeNet4 中实现的，用于梯度提升，但没有在现有的开源软件包中实现。
>
> 根据使用者反馈，使用列抽样比使用传统的行抽样(也支持)更能防止过拟合。列抽样的使用还加快了稍后描述的并行算法的计算速度。

## 3. SPLIT FINDING ALGORITHMS

#### 3.1 Basic Exact Greedy Algorithm

One of the key problems in tree learning is to find the best split as indicated by Eq (7). In order to do so, a split finding algorithm enumerates over all the possible splits on all the features. We call this the exact greedy algorithm. Most existing single machine tree boosting implementations, such as scikit-learn [20], R’s gbm [21] as well as the single machine version of XGBoost support the exact greedy algorithm. The exact greedy algorithm is shown in Alg. 1. 

It is computationally demanding to enumerate all the possible splits for continuous features. In order to do so efficiently, the algorithm must first sort the data according to feature values and visit the data in sorted order to accumulate the gradient statistics for the structure score in Eq (7).

![Algorithm1](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Alg1.png)

> 如公式 (7) 所示，树学习的关键问题之一是找到最佳分割点。为了做到这一点，分裂点查找算法枚举了所有特征上的所有可能的分割点。我们称之为exact greedy 算法。大多数现有的 single machine tree boosting 实现，如scikit-learn [20]、R的gbm[21]以及 XGBoost 的单机版本都支持 exact greedy 算法。exact greedy 如 ALG.1 所示。
>
> 对于连续的特征，需要计算所有可能的分割。为了提高效率，算法必须首先根据特征值对数据进行排序，然后根据公式 Eq (7) 计算出当前分割点的梯度统计量。

#### 3.2 Approximate Algorithm

The exact greedy algorithm is very powerful since it enumerates over all possible splitting points greedily. However, it is impossible to efficiently do so when the data does not fit entirely into memory. Same problem also arises in the distributed setting. To support effective gradient tree boosting in these two settings, an approximate algorithm is needed.

We summarize an approximate framework, which resembles the ideas proposed in past literatures [17, 2, 22], in Alg. 2. To summarize, the algorithm first proposes candidate splitting points according to percentiles of feature distribution (a specific criteria will be given in Sec. 3.3). The algorithm then maps the continuous features into buckets split by these candidate points, aggregates the statistics and finds the best solution among proposals based on the aggregated statistics.

> The exact greedy 算法是非常强大的，因为它贪婪地枚举了所有可能的分割点。然而，当数据不能完全读入内存时，这样做就不会很有效率。同样的问题也出现在分布式环境中。为了在这两种设置下支持有效的梯度树提升，需要一种近似算法。
>
> 我们在 ALG.2 中提出了一个近似的框架，它类似于过去文献[17，2，22]中提出的思想。综上所述，该算法首先根据特征分布的百分位数提出候选分割点(具体标准将在3.3节给出)。
>
> 然后，该算法将连续的特征映射到由这些候选分割点分出的桶中，计算出每个箱子中数据的统计量（注：这里的统计量指的是公式（7）中的 $g$ 和 $h $），然后根据统计量找到最佳的分割点。

There are two variants of the algorithm, depending on when the proposal is given. The global variant proposes all the candidate splits during the initial phase of tree construction, and uses the same proposals for split finding at all levels. The local variant re-proposes after each split. The global method requires less proposal steps than the local method. However, usually more candidate points are needed for the global proposal because candidates are not refined after each split. The local proposal refines the candidates after splits, and can potentially be more appropriate for deeper trees. A comparison of different algorithms on a Higgs boson dataset is given by Fig. 3. We find that the local proposal indeed requires fewer candidates. The global proposal can be as accurate as the local one given enough candidates.

Most existing approximate algorithms for distributed tree learning also follow this framework. Notably, it is also possible to directly construct approximate histograms of gradient statistics [22]. It is also possible to use other variants of binning strategies instead of quantile [17]. Quantile strategy benefit from being distributable and recomputable, which we will detail in next subsection. From Fig. 3, we also find that the quantile strategy can get the same accuracy as exact greedy given reasonable approximation level.

Our system efficiently supports exact greedy for the single machine setting, as well as approximate algorithm with both local and global proposal methods for all settings. Users can freely choose between the methods according to their needs.

![Figure3](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Fig3.png)

**Figure 3: Comparison of test AUC convergence on Higgs 10M dataset. The eps parameter corresponds to the accuracy of the approximate sketch. This roughly translates to 1 / eps buckets in the proposal. We find that local proposals require fewer buckets, because it refine split candidates.**

![Alg2](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Alg2.png)

> 该算法有两种变种，取决于分割的时间。
>
> 全局选择在树构造的初始阶段给出所有候选分裂点，并且在树的所有层中使用相同的分裂节点用于分裂。局部选择在分裂后重新给出分裂候选节点。
>
> 全局方法比局部方法需要更少的步骤。然而，通常在全局选择中需要更多的候选点，因为在每次分裂后候选节点没有被更新。局部选择在分裂后更新候选节点，并且可能更适合于深度更深的树。图3 给出了基于希格斯玻色子数据集的不同算法的比较。我们发现，本地变种确实需要更少的候选节点。当给出足够的候选节点，全局变种可以达到与本地变种一样的准确率。
>
> 现有的大多数分布式树模型学习的近似算法也遵循这一框架。值得注意的是，还可以直接构建梯度统计量的近似直方图[22]。也可以使用其他不同的分桶策略来代替分位数[17]。分位数策略受益于可分发和可重新计算，我们将在下一小节中详细说明这一点。从 图3 我们还发现，当设置合理的近似水平，分位数策略可以获得与 exact greedy 相同的精度。
>
> 我们的系统有效地支持单机环境下的 exact greedy，同时也支持近似算法的local 变种和 global 变种的所有设置。使用者可以根据自己的需要自由选择不同的方法。

#### 3.3 Weighted Quantile Sketch

One important step in the approximate algorithm is to propose candidate split points. Usually percentiles of a feature are used to make candidates distribute evenly on the data. Formally, let multi-set $\mathcal{D}_{k}=\left\{\left(x_{1 k}, h_{1}\right),\left(x_{2 k}, h_{2}\right) \cdots\left(x_{n k}, h_{n}\right)\right\}$ represent the $k$-th feature values and second order gradient statistics of each training instances. We can define a rank functions $r_{k}: \mathbb{R} \rightarrow[0,+\infty)$ as
$$
r_{k}(z)=\frac{1}{\sum_{(x, h) \in \mathcal{D}_{k}} h} \sum_{(x, h) \in \mathcal{D}_{k}, x<z} h \qquad(8)
$$
which represents the proportion of instances whose feature value $k$ is smaller than $z$. The goal is to find candidate split points $\{s_{k1}, s_{k2},\cdots,s_{kl}\}$, such that
$$
\left|r_{k}\left(s_{k, j}\right)-r_{k}\left(s_{k, j+1}\right)\right|<\epsilon, \quad s_{k 1}=\min _{i} \mathbf{x}_{i k}, s_{k l}=\max _{i} \mathbf{x}_{i k}
$$
Here $\epsilon$ is an approximation factor. Intuitively, this means that there is roughly $1/\epsilon$ candidate points. Here each data point is weighted by $h_i$. To see why $h_i$ represents the weight, we can rewrite Eq (3) as
$$
\sum_{i=1}^{n} \frac{1}{2} h_{i}\left(f_{t}\left(\mathbf{x}_{i}\right)-g_{i} / h_{i}\right)^{2}+\Omega\left(f_{t}\right)+\text { constant }
$$
which is exactly weighted squared loss with labels $g_i/h_i$  and weights $h_i$. For large datasets, it is non-trivial to find candidate splits that satisfy the criteria. When every instance has equal weights, an existing algorithm called quantile sketch [14, 24] solves the problem. However, there is no existing quantile sketch for the weighted datasets. Therefore, most existing approximate algorithms either resorted to sorting on a random subset of data which have a chance of failure or heuristics that do not have theoretical guarantee.

To solve this problem, we introduced a novel distributed weighted quantile sketch algorithm that can handle weighted data with a provable theoretical guarantee. The general idea is to propose a data structure that supports merge and prune operations, with each operation proven to maintain a certain accuracy level. A detailed description of the algorithm as well as proofs are given in the appendix.

> 近似算法中很重要的一步是列出候选的分割点。通常特征的百分位数作为候选分割点的分布会比较均匀。
>
> 形式化地讲，设多元集合 $\mathcal{D}_{k}=\left\{\left(x_{1 k}, h_{1}\right),\left(x_{2 k}, h_{2}\right) \cdots\left(x_{n k}, h_{n}\right)\right\}$ 表示样本的第 $k$ 个特征的取值和其二阶梯度统计量。我们可以定义一个排序函数 $r_{k}: \mathbb{R} \rightarrow[0,+\infty)$ :
> $$
> r_{k}(z)=\frac{1}{\sum_{(x, h) \in \mathcal{D}_{k}} h} \sum_{(x, h) \in \mathcal{D}_{k}, x<z} h \qquad(8)
> $$
> 上式表示样本中第 $k$个特征的取值小于 $z$ 的比例 (注：特征值小于 $z$ 的二阶梯度统计量的比例)。我们的目标是找到候选的分割节点 $\{s_{k1}, s_{k2},\cdots,s_{kl}\}$
> $$
> \left|r_{k}\left(s_{k, j}\right)-r_{k}\left(s_{k, j+1}\right)\right|<\epsilon, \quad s_{k 1}=\min _{i} \mathbf{x}_{i k}, s_{k l}=\max _{i} \mathbf{x}_{i k}
> \qquad(9)
> $$
> 这里 $\epsilon$ 是一个近似因子。直观地说，这意味着大约有 $1/\epsilon$ 候选点。这里，每个数据点由 $h_i$ 加权。要了解为什么 $h_i$ 表示权重，我们可以将公式(3)重写为
> $$
> \sum_{i=1}^{n} \frac{1}{2} h_{i}\left(f_{t}\left(\mathbf{x}_{i}\right)-g_{i} / h_{i}\right)^{2}+\Omega\left(f_{t}\right)+\text { constant }
> $$
> 这实际上是权值为 $h_i$，标签为 $g_i/h_i$ 的加权平方损失。对于大数据集来说，找到满足标准的候选分割点是非常不容易的。当每个实例具有相等的权重时，一个现存的叫 quantile sketch 的算法解决了这个问题。然而，对于加权的数据集没有现成的 quantile sketch 算法。因此，大部分现存的近似算法要么对可能失败的数据的随机子集进行排序，要么使用没有理论保证的启发式算法。
>
> 为了解决这个问题，我们引入了一种新的分布式加权 quantile sketch 算法，该算法可以处理加权数据，并且可以从理论上证明。通常的做法是提出一种支持 merge 和 prune 操作的数据结构，每个操作都是可以被证明保持一定准确度的。附录中给出了算法的详细描述以及证明。

#### 3.4 Sparsity-aware Split Finding

In many real-world problems, it is quite common for the input $\mathbb{x}$ to be sparse. There are multiple possible causes for sparsity: 

1. presence of missing values in the data;
2. frequent zero entries in the statistics; 
3. artifacts of feature engineering such as one-hot encoding.

It is important to make the algorithm aware of the sparsity pattern in the data. In order to do so, we propose to add a default direction in each tree node, which is shown in Fig. 4. When a value is missing in the sparse matrix $\mathbb{x}$, the instance is classified into the default direction. 

There are two choices of default direction in each branch. The optimal default directions are learnt from the data. The algorithm is shown in Alg. 3. The key improvement is to only visit the non-missing entries $I_k$. The presented algorithm treats the non-presence as a missing value and learns the best direction to handle missing values. 

The same algorithm can also be applied when the non-presence corresponds to a user specified value by limiting the enumeration only to consistent solutions.

To the best of our knowledge, most existing tree learning algorithms are either only optimized for dense data, or need specific procedures to handle limited cases such as categorical encoding. XGBoost handles all sparsity patterns in a unified way. More importantly, our method exploits the sparsity to make computation complexity linear to number of non-missing entries in the input. 

Fig. 5 shows the comparison of sparsity aware and a naive implementation on an Allstate-10K dataset (description of dataset given in Sec. 6). We find that the sparsity aware algorithm runs 50 times faster than the naive version. This confirms the importance of the sparsity aware algorithm.

![Figure5](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Fig5.png)

**Figure 5: Impact of the sparsity aware algorithm on Allstate-10K. The dataset is sparse mainly due to one-hot encoding. The sparsity aware algorithm is more than 50 times faster than the naive version that does not take sparsity into consideration.**

![Algorithm3](/Users/helloword/Anmingyu/Gor-rok/Papers/GBDT/XGB/Alg3.png)

> 在许多实际问题中，输入 $\mathbb{x}$ 稀疏是很常见的。稀疏有多种可能的原因：
>
> 1. 数据中存在缺失值；
> 2. 大量零值；
> 3. 特征工程，如 one-hot。
>
> 重要的是让算法感知数据中的稀疏模式。为此，我们建议在每个树节点中添加一个默认方向，如 图4 所示。当稀疏矩阵 $\mathbb{x}$ 中缺缺失时，实例被分类为默认方向。
>
> 每个分支中有两个默认方向选择。从数据中学习最佳默认方向。该算法在Alg.3. 中实现。关键改进是只访问非缺失的 item $I_k$。该算法将不存在视为缺失值，并学习最佳方向来处理缺失值。
>
> The same algorithm can also be applied when the non-presence corresponds to a user specified value by limiting the enumeration only to consistent solutions.
>
> 据我们所知，大多数现有的树学习算法要么只对连续数据进行优化，要么需要特定的过程来处理部分情况，例如类别编码。XGBoost 以统一的方式处理所有稀疏模式。更重要的是，我们的方法利用了稀疏性，使得计算的复杂度与输入中的非缺失数据的数量成线性关系。图5 显示了稀疏感知算法和一个常规算法在数据集 Allstate-10K（此数据集在第6部分描述）上的比较。我们发现稀疏感知算法的运行速度比常规版本快 50 倍。这证实了稀疏感知算法的重要性。
>

