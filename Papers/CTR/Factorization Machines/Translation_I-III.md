> Steffen Rendle 
>
> 作者是德国人，本文如果纯英文读起来会好一些，但是极其不好翻译。
>
> 大阪大学

# Factorization Machines

## Abstract

In this paper, we introduce Factorization Machines (FM) which are a new model class that combines the advantages of Support Vector Machines (SVM) with factorization models. Like SVMs, FMs are a general predictor working with any real valued feature vector. In contrast to SVMs, FMs model all interactions between variables using factorized parameters. Thus they are able to estimate interactions even in problems with huge sparsity (like recommender systems) where SVMs fail. We show that the model equation of FMs can be calculated in linear time and thus FMs can be optimized directly. So unlike nonlinear SVMs, a transformation in the dual form is not necessary and the model parameters can be estimated directly without the need of any support vector in the solution. We show the relationship to SVMs and the advantages of FMs for parameter estimation in sparse settings.

On the other hand there are many different factorization models like matrix factorization, parallel factor analysis or specialized models like SVD++, PITF or FPMC. The drawback of these models is that they are not applicable for general prediction tasks but work only with special input data. Furthermore their model equations and optimization algorithms are derived individually for each task. We show that FMs can mimic these models just by specifying the input data (i.e. the feature vectors). This makes FMs easily applicable even for users without expert knowledge in factorization models.

> 本文介绍了因子分解机(FM)，它是结合支持向量机(SVM)和因子分解模型优点的一种新的模型。与 SVMs 一样，FMs 也是处理任何实值特征向量的通用预测器。与 SVMs 不同，FMs 使用分解参数对变量之间的所有交互(注：特征交叉)进行建模。因此，即使在 SVMs 失败的巨大稀疏性问题(如推荐系统)中，他们也能够估计交互。结果表明，FMs 的模型可以在线性时间内求解，从而可以直接对 FMs 进行优化。因此，与非线性支持向量机不同的是，该方法不需要对偶变换，直接估计模型参数，不需要任何支持向量。我们展示了与支持向量机的关系，以及在稀疏环境下使用支持向量机进行参数估计的优势。
>
> 另一方面，有许多不同的因子分解模型，如矩阵因子分解、并行因子分析或专门的模型，如 SVD++、PITF 或 FPMC。这些模型的缺点是它们不适用于一般的预测任务，而只适用于特殊的输入数据。此外，还针对每个任务分别推导了它们的模型方程和优化算法。我们发现，FMs 可以通过指定输入数据(即特征向量)来模拟这些模型。这使得即使没有因子分解模型专业知识的使用者也可以很容易地使用。

## I. INTRODUCTION

Support Vector Machines are one of the most popular predictors in machine learning and data mining. Nevertheless in settings like collaborative filtering, SVMs play no important role and the best models are either direct applications of standard matrix/tensor factorization models like PARAFAC [1] or specialized models using factorized parameters [2], [3], [4]. In this paper, we show that the only reason why standard SVM predictors are not successful in these tasks is that they cannot learn reliable parameters (‘hyperplanes’) in complex (non-linear) kernel spaces under very sparse data. On the other hand, the drawback of tensor factorization models and even more for specialized factorization models is that (1) they are not applicable to standard prediction data (e.g. a real valued feature vector in $\mathbb{R}^n$.) and (2) that specialized models are usually derived individually for a specific task requiring effort in modelling and design of a learning algorithm.

In this paper, we introduce a new predictor, the Factorization Machine (FM), that is a general predictor like SVMs but is also able to estimate reliable parameters under very high sparsity. The factorization machine models all nested variable interactions (comparable to a polynomial kernel in SVM), but uses a factorized parametrization instead of a dense parametrization like in SVMs. We show that the model equation of FMs can be computed in linear time and that it depends only on a linear number of parameters. This allows direct optimization and storage of model parameters without the need of storing any training data (e.g. support vectors) for prediction. In contrast to this, non-linear SVMs are usually optimized in the dual form and computing a prediction (the model equation) depends on parts of the training data (the support vectors). We also show that FMs subsume many of the most successful approaches for the task of collaborative filtering including biased MF, SVD++ [2], PITF [3] and FPMC [4].

In total, the advantages of our proposed FM are: 

1. FMs allow parameter estimation under very sparse data where SVMs fail.
2. FMs have linear complexity, can be optimized in the primal and do not rely on support vectors like SVMs. We show that FMs scale to large datasets like Netflix with 100 millions of training instances. 
3. FMs are a general predictor that can work with any real valued feature vector. In contrast to this, other state-of-the-art factorization models work only on very restricted input data. We will show that just by defining the feature vectors of the input data, FMs can mimic state-of-the-art models like biased MF, SVD++, PITF or FPMC.

> 支持向量机是机器学习和数据挖掘中最常用的预测工具之一。然而，在协同过滤的设置下，支持向量机并没有发挥重要作用，最好的模型要么是直接应用标准的 矩阵/张量 因子分解模型，如PARAFAC[1]，要么是使用分解因子[2]、[3]、[4]的专用模型。在本文中，我们表明标准 支持向量机 预测器在这些任务中无法完成这些任务的的唯一原因是它们不能在非常稀疏的数据下在复杂(非线性)核空间中学习可靠的参数(“超平面”)。另一方面,张量分解模型的缺点和更专业的因子分解模型的缺点是：
>
> 1. 他们不适用标准预测数据(例如一个实值的特征向量 $\mathbb{R}^n$)。
> 2. 专业模型通常是为一个特定的任务单独设计的。
>
> 在本文中，我们介绍了一种新的预测器，即因子分解机(FM)，它与支持向量机一样是一种通用的预测器，但它能在 高稀疏度下估计可靠的参数。因子分解机对所有嵌套的变量交互进行建模(类似于支持向量机中的多项核)，但是使用因子分解的参数化而不是像支持向量机中那样密集的参数化。我们证明了 FMs 的模型方程可以在线性时间内计算出来，它仅取决于线性数量的参数。这允许直接优化和存储模型参数，而不需要存储任何用于预测的训练数据(例如支持向量)。与此相反，非线性支持向量机通常以对偶形式进行优化，计算预测(模型方程)取决于部分训练数据(支持向量)。我们还表明，FMs包含了许多最成功的协同过滤方法，包括有偏 MF、SVD++[2]、PITF[3] 和 FPMC[4]。
>
> 综上所述，我们提出的 FM 的优点有:
>
> 1. FMs 允许在支持向量机失效的非常稀疏数据下进行参数估计。
> 2. FMs 线性复杂度，可以在原始条件下进行优化，并且不依赖支持向量（如SVM）。我们证明了 FMs 可以扩展到像 Netflix 这样拥有1亿训练实例的大型数据集。
> 3. FMs 是一种适用于任何实值特征向量的通用预测器。与此形成对比的是，其他最先进的因式分解模型只适用于非常有限的输入数据。我们将展示，只需定义输入数据的特征向量，FMs 就可以模拟最先进的模型，如 biased MF、SVD++、PITF 或 FPMC。

## II. PREDICTION UNDER SPARSITY

The most common prediction task is to estimate a function $y: \mathbb{R}^n \rightarrow T$ from a real valued feature vector $x \in R^n$ to a target domain $T$ (e.g. $T = R$ for regression or $T = \{+, −\}$ for classification). In supervised settings, it is assumed that there is a training dataset $D = \{(\textbf{x}^{(1)}， y^{(1)})，(\textbf{x}^{(2)}， y^{(2)}) , \cdots \}$ of examples for the target function $y$ given. We also investigate the ranking task where the function $y$ with target $T = R$ can be used to score feature vectors x and sort them according to their score. Scoring functions can be learned with pairwise training data [5], where a feature tuple $(\textbf{x}(A)， \textbf{x}(B)) \in D$ means that $\textbf{x}(A)$ should be ranked higher than $\textbf{x}(B)$ . As the pairwise ranking relation is antisymmetric, it is sufficient to use only positive training instances.

In this paper, we deal with problems where $\textbf{x}$ is highly sparse, i.e. almost all of the elements $x_i$ of a vector $\textbf{x}$ are zero. Let $m(\textbf{x)}$ be the number of non-zero elements in the feature vector $\textbf{x}$ and $\bar{m}_D$ be the average number of non-zero elements $m(\textbf{x})$ of all vectors $\textbf{x} \in D$. Huge sparsity ($\bar{m}_D ≪ n$) appears in many real-world data like feature vectors of event transactions (e.g. purchases in recommender systems) or text analysis (e.g. bag of word approach). One reason for huge sparsity is that the underlying problem deals with large categorical variable domains.

> 最常见的预测任务是从实值特征向量 $x \in R^n$ 的估计函数 $y: \mathbb{R}^n \rightarrow T$ 到目标域 $T$ (例如，$T = R$ 表示回归，$T =\{+，−\}$ 表示分类)。
>
> 在有监督学习设置中，给定训练数据集 $D = \{(\textbf{x}^{(1)}， y^{(1)})，(\textbf{x}^{(2)}， y^{(2)}) , \cdots \}$ 个目标函数为 $y$ 的训练实例。
>
> 我们还研究了排序任务，其中函数 $y$ 与目标 $T = R$ 可以用来评分特征向量 $\textbf{x}$，并根据他们的分数排序。得分可通过 pair-wise 训练数据[5]可以学习得到，其中特征元组 $(\textbf{x}(A)， \textbf{x}(B)) \in D$ 意味着 $\textbf{x}(A)$ 的排名应该高于 $\textbf{x}(B)$。由于 pairwise 排序是反对称的，只使用正的训练实例就足够了。
>
> 本文讨论了 $\textbf{x}$ 高度稀疏的问题，即向量 $\textbf{x}$ 的几乎所有元素 $x_i$ 都为零。
> 设 $m(\textbf{x)}$ 是特征向量 $\textbf{x}$ 中非零元素的个数，$\bar{m}_D$ 是 $D$ 中所有向量 $\textbf{x}$ 的非零元素 $m(\textbf{x})$ 的平均数。巨大的稀疏性 ($\bar{m}_D≪n$) 出现在许多真实世界的数据中，如事件交易的特征向量(如推荐系统中的购买)或文本分析(如词袋方法)。高度稀疏的原因之一是潜在的问题涉及较大的类型变量。

#### Example 1 

Assume we have the transaction data of a movie review system. The system records which user $u \in U$ rates a movie (item) $i \in I$ at a certain time $t \in \mathbb{R}$ with a rating $r \in \{1, 2, 3, 4, 5\}$. Let the users $U$ and items $I$ be:
$$
\begin{align}
U &= \{Alice (A), Bob (B), Charlie (C), \cdots \}
\\
I &= \{Titanic (TI), Notting Hill (NH), Star Wars (SW),
Star Trek (ST), \cdots\}
\end{align}
$$
Let the observed data $S$ be:
$$
S = {(A, TI, 2010-1, 5),(A, NH, 2010-2, 3),(A, SW, 2010-4, 1),
\\
(B, SW, 2009-5, 4),(B, ST, 2009-8, 5),
\\
(C, TI, 2009-9, 1),(C, SW, 2009-12, 5)}
$$
An example for a prediction task using this data, is to estimate a function $\hat{y}$ that predicts the rating behaviour of a user for an item at a certain point in time.

> 假设我们有一个电影评论系统的数据。系统记录了 user $u \in U$ 对一个电影 (item) $i \in I$ 在时间 $t \in \mathbb{R}$ 的评分 $r \in \{1, 2, 3, 4, 5\}$. 
>
> 设代表 user 的 $U$ 和代表 item的 $I$:
> $$
> \begin{align}
> U &= \{Alice (A), Bob (B), Charlie (C), \cdots \}
> \\
> I &= \{Titanic (TI), Notting Hill (NH), Star Wars (SW),
> Star Trek (ST), \cdots\}
> \end{align}
> $$
> 设观测数据 $S$ 为：
> $$
> S = {(A, TI, 2010-1, 5),(A, NH, 2010-2, 3),(A, SW, 2010-4, 1),
> \\
> (B, SW, 2009-5, 4),(B, ST, 2009-8, 5),
> \\
> (C, TI, 2009-9, 1),(C, SW, 2009-12, 5)}
> $$
> 使用该数据的预测任务的一个例子是估计一个函数 $\hat{y}$，该函数预测用户在某一特定时间点对某电影的评分行为。

Figure 1 shows one example of how feature vectors can be created from $S$ for this task.1 Here, first there are $|U|$ binary indicator variables (blue) that represent the active user of a transaction – there is always exactly one active user in each transaction $(u, i, t, r) \in S$, e.g. user Alice in the first one $(x^{(1)}_A = 1)$. The next $|I|$ binary indicator variables (red) hold the active item – again there is always exactly one active item (e.g. $x^{(1)}_{TI} = 1$). The feature vectors in figure 1 also contain indicator variables (yellow) for all the other movies the user has ever rated. For each user, the variables are normalized such that they sum up to 1. E.g. Alice has rated Titanic, Notting Hill and Star Wars. Additionally the example contains a variable (green) holding the time in months starting from January, 2009. And finally the vector contains information of the last movie (brown) the user has rated before he/she rated the active one – e.g. for $\textbf{x}$ (2) , Alice rated Titanic before she rated Notting Hill. In section $V$, we show how factorization machines using such feature vectors as input data are related to specialized state-of-the-art factorization models.

We will use this example data throughout the paper for illustration. However please note that FMs are general predictors like SVMs and thus are applicable to any real valued feature vectors and are not restricted to recommender systems.

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Factorization Machines/Fig1.png)

**Fig. 1. Example for sparse real valued feature vectors x that are created from the transactions of example 1. Every row represents a feature vector $\textbf{x}^{(i)}$ with its corresponding target $y^{(i)}$ . The first 4 columns (blue) represent indicator variables for the active user; the next 5 (red) indicator variables for the active item. The next 5 columns (yellow) hold additional implicit indicators (i.e. other movies the user has rated). One feature (green) represents the time in months. The last 5 columns (brown) have indicators for the last movie the user has rated before the active one. The rightmost column is the target – here the rating.**

> 图1 显示了如何从 $S$ 为该任务创建特征向量的一个示例。
>
> 首先有 $|U|$ 个 one-hot变量(蓝色)表示事务(注：这里的事务即 transaction 指的是用户对电影的评分)的相关的用户，每个事务 $(u，i，t，r) \in S$ 中总是恰好有一个激活的用户，例如第一个事务 $(x^{(1)}_A=1)$ 中的用户 Alice。下一个 $|I|$ 二值变量(红色)保存 item，同样，始终只有一个激活的 item (例如$x^{(1)}_{TI}=1$)。
>
> 图1 中的特征向量还包含用户曾经评分过的所有其他电影的变量(黄色)。对于每个用户，变量被归一化，这样它们的总和就是 1。例如，爱丽丝给《泰坦尼克号》、《诺丁山》和《星球大战》打了分。此外，该示例还包含一个变量(绿色)， 保存从2009年1月开始至今的时间。
>
> 最后，向量包含用户在给活动电影打分之前打分的最后一部电影(棕色)的信息，如，$\textbf{x}$ (2)，爱丽丝在给《诺丁山》打分之前给《泰坦尼克号》打分。
>
> 在 $V$ 部分，我们展示了使用这样的特征向量作为输入数据的因子分解机是如何与专门的最先进的因子分解模型相关联的。我们将在整篇文章中使用此示例数据进行说明。然而，请注意，像支持向量机一样，FM 是通用预测器，因此适用于任何实值特征向量，并不局限于推荐系统。

## III. FACTORIZATION MACHINES (FM)

In this section, we introduce factorization machines. We discuss the model equation in detail and show shortly how to apply FMs to several prediction tasks.

> 在这一节中，我们将介绍因子分解机。我们详细讨论了模型方程，并简要地展示了如何将 FMs 应用于几个预测任务。

#### A. Factorization Machine Model

**1) Model Equation**：

The model equation for a factorization machine of degree $d = 2$ is defined as:
$$
\hat{y}(\textbf{x}) := w_0
+ \sum_{i=1}^{n} w_i x_i
+ \sum_{i=1}^{n}\sum_{j=i+1}^{n}<\textbf{v}_i,\textbf{v}_j>x_ix_j
\qquad(1)
$$
where the model parameters that have to be estimated are:
$$
w_0 \in \mathbb{R},\textbf{w} \in \mathbb{R}^n,\textbf{V} \in \mathbb{R}^{n \times k}
\qquad(2)
$$
And $<·, ·>$ is the dot product of two vectors of size $k$ :
$$
<\textbf{v}_i,\textbf{v}_j> := \sum_{f=1}^{k} v_{i,f}\cdot v_{j,f}
\qquad (3)
$$
A row $\textbf{v}_i$ within $\textbf{V}$ describes the $i$-th variable with $k$ factors. $k \in \mathbb{N}_0^+$ is a hyperparameter that defines the dimensionality of the factorization.

A 2-way FM (degree $d = 2$) captures all single and pairwise interactions between variables:

- $w_0$ is the global bias.
- $w_i$ models the strength of the $i$-th variable.
- $\hat{w}_{i,j} := <\textbf{v}_i,\textbf{v}_j>$ models the interaction between the $i$-th and $j$-th variable. Instead of using an own model parameter $w_{i,j} \in \mathbb{R}$ for each interaction, the FM models the interaction by factorizing it. We will see later on, that this is the key point which allows high quality parameter estimates of higher-order interactions ($d ≥ 2$) under sparsity.

> **1) 模型方程：** 阶数为 $d=2$ 的因子分解机的模型方程定义为：
> $$
> \hat{y}(\textbf{x}) := w_0
> + \sum_{i=1}^{n} w_i x_i
> + \sum_{i=1}^{n}\sum_{j=i+1}^{n}<\textbf{v}_i,\textbf{v}_j>x_ix_j
> \qquad(1)
> $$
> 必须要估计的模型参数是：
> $$
> w_0 \in \mathbb{R},\textbf{w} \in \mathbb{R}^n,\textbf{V} \in \mathbb{R}^{n \times k}
> $$
>  $<·, ·>$ 是长度为 $k$ 的两个向量的点积：
>
>  $\textbf{v}_i$ 是 $\textbf{V}$  中长度为 $k$ 的第 $i$ 行向量。$k \in \mathbb{N}_0^+$ 是定义因子分解维度的超参数.
>
> 2-way FM 捕获变量之间的所有单个和成对的交互作用：
>
> - $w_0$ 是全局偏置变量。
> - $w_i$ 对第 $i$ 个变量建模。
> - $\hat{w}_{i,j} := <\textbf{v}_i,\textbf{v}_j>$ 对 i 和 j 建模。FM 通过分解交互来建模，而不是使用自己的模型参数 $w_{i,j} \in \mathbb{R}$，稍后我们将看到，这是在稀疏条件下实现高阶相互作用 ($d≥2$) 的高质量参数估计的关键。

**2) Expressiveness: **

It is well known that for any positive definite matrix $\textbf{W}$, there exists a matrix $\textbf{V}$ such that $\textbf{W = V}\cdot\textbf{V}^t$ provided that $k$ is sufficiently large. This shows that a FM can express any interaction matrix $\textbf{W}$ if $k$ is chosen large enough. Nevertheless in sparse settings, typically a small $k$ should be chosen because there is not enough data to estimate complex interactions $\textbf{W}$. Restricting $k$ – and thus the expressiveness of the FM – leads to better generalization and thus improved interaction matrices under sparsity.

> 众所周知，对于任何正定矩阵 $\textbf{W}$，存在一个矩阵 $\textbf{V}$ 使得 $\textbf{W=V}\cdot\textbf{V}^t$，这表明，如果选择足够大的 $k$，FM 可以表示任何交互矩阵 $\textbf{W}$。然而，在稀疏设置中，通常应该选择较小的 $k$，因为没有足够的数据来估计复杂的交互矩阵 $\textbf{W}$。限制 $k$ 从而限制 FM 的表现力导致更好的泛化，从而改善稀疏的交互矩阵。限制 $k$，从而使 FM 的表达能力得到更好的推广，从而改善了稀疏条件下的交互矩阵。

**3) Parameter Estimation Under Sparsity** : 

In sparse settings, there is usually not enough data to estimate interactions between variables directly and independently. Factorization machines can estimate interactions even in these settings well because they break the independence of the interaction parameters by factorizing them. In general this means that the data for one interaction helps also to estimate the parameters for related interactions. We will make the idea more clear with an example from the data in figure 1. Assume we want to estimate the interaction between Alice (A) and Star Trek (ST) for predicting the target $y$ (here the rating). Obviously, there is no case $\textbf{x}$ in the training data where both variables $x_A$ and $x_{ST}$ are non-zero and thus a direct estimate would lead to no interaction ($w_{A,ST} = 0$). 

But with the factorized interaction parameters $<\textbf{v}_A, \textbf{v}_{ST}>$ we can estimate the interaction even in this case. First of all, Bob and Charlie will have similar factor vectors $\textbf{v}_B$ and $\textbf{v}_C$ because both have similar interactions with Star Wars ($\textbf{v}_{SW}$) for predicting ratings – i.e. $<\textbf{v}_B, \textbf{v}_{SW}>$ and $<\textbf{v}_C , \textbf{v}_{SW}>$ have to be similar. Alice ($\textbf{v}_A$) will have a different factor vector from Charlie ($\textbf{v}_C$ ) because she has different interactions with the factors of Titanic and Star Wars for predicting ratings. Next, the factor vectors of Star Trek are likely to be similar to the one of Star Wars because Bob has similar interactions for both movies for predicting $y$. In total, this means that the dot product (i.e. the interaction) of the factor vectors of Alice and Star Trek will be similar to the one of Alice and Star Wars – which also makes intuitively sense.

> 在稀疏设置下，通常没有足够的数据来直接独立地估计变量之间的交互。即使在这种条件下下，因子分解机器也可以很好地估计交互参数，因为它们通过因子分解打破了交互参数的独立性。一般来说，这意味着一个交互的数据也有助于估计相关交互的参数。我们将通过图1中的数据示例来说明这一点。假设我们要估计 Alice(A) 和 Star Trek(ST) 之间的交互，以预测目标 $y$ (这里是评分)。显然，在训练数据中不存在变量 $x_A$ 和 $x_{ST}$ 都非零的情况 $\textbf{x}$，因此直接估计会使两者没有交互参数，即 $w_{A,ST}=0$。
>
> 但是，使用因子分解后的交互参数 $<\textbf{v}_A， \textbf{v}_{ST}>$，即使在这种情况下，我们仍然可以估计这种情况下的交互参数。首先，Bob 和 Charlie 会有相似的因子向量 $\textbf{v}_B$ 和 $\textbf{v}_C$，因为在预测评分时他们都与《星球大战》($\textbf{v}_{SW}$)相似的作用，即 $<\textbf{v}_B, \textbf{v}_{SW}>$ 和 $<\textbf{v}_C， \textbf{v}_{SW}>$ 一定相似。Alice ($\textbf{v}_A$) 将拥有与 Charlie ($\textbf{v}_C$)不同的因子向量，因为在预测评分时她与《泰坦尼克号》和《星球大战》的因子有不同的交互作用。其次，《星际迷航》的因子向量可能与《星球大战》相似，因为在预测 $y$ 时鲍勃对两部电影的的交互相似。总而言之，这意味着 Alice 和《星际迷航》的因子向量的点积(即交互作用)将与 Alice 和 《星球大战》的一个相似——这在直觉上也是有道理的。

**4) Computation: **

Next, we show how to make FMs applicable from a computational point of view. The complexity of straight forward computation of eq. (1) is in $O(k \ n^2)$ because all pairwise interactions have to be computed. But with reformulating it drops to linear runtime.

Lemma 3.1: The model equation of a factorization machine (eq. (1)) can be computed in linear time $O(k \ n).$

Proof: Due to the factorization of the pairwise interactions, there is no model parameter that directly depends on two variables (e.g. a parameter with an index $(i, j)$). So the pairwise interactions can be reformulated:
$$
\begin{aligned}
& \sum_{i=1}^{n}\sum_{j=i+1}^{n} <\textbf{v}_i,\textbf{v}_j>x_ix_j
\\
&= \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n}<\textbf{v}_i,\textbf{v}_j>x_ix_j - \frac{1}{2}\sum_{i=1}^{n}<\textbf{v}_i,\textbf{v}_i>x_ix_i
\\
&= \frac{1}{2}(
\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{j,f}x_ix_j - 
\sum_{i-1}^{n}\sum_{j=1}^{k}v_{i,f}v_{i,f}x_ix_i
)
\\
&= \frac{1}{2} ((\sum_{i=1}^{n}v_{i,f}x_{i})(\sum_{j=1}^{n}v_{j,f}x_j) - \sum_{i=1}^{n}v_{i,f}^{2}x_{i}^{2})
\\
&= \frac{1}{2} \sum_{f=1}^{k}
((\sum_{i=1}^{n} v_{i,f}x_i)^2 - \sum_{i=1}^{n}v_{i,f}^{}x_{i}^2)
\end{aligned}
$$
This equation has only linear complexity in both $k$ and $n$ – i.e. its computation is in $O(k \ n)$.

Moreover, under sparsity most of the elements in $x$ are $0$ (i.e. $m(x)$ is small) and thus, the sums have only to be computed over the non-zero elements. Thus in sparse applications, the computation of the factorization machine is in $O(k \  \bar{m}_D)$ – e.g. $\bar{m}_D = 2$ for typical recommender systems like MF approaches (see section V-A).

> 接下来，我们将从计算的角度展示如何使 FMs 适用。eq.(1) 的直接计算复杂度是 $O(k \ n^2)$ ，因为所有的对之间的交互都必须计算。但是，通过重新表述，它将下降到线性时间。
>
> 引理 3.1：因式分解机的模型方程 ( 公式 (1) ) 可以在线性时间 $O(kn)$ 内计算。
>
> 证明：由于对两两交互的因子分解，没有模型参数直接依赖于两个变量(例如一个索引为 $(i,j)$ 的参数)。因此，两两交互可以重新表述为:
> $$
> \begin{aligned}
> & \sum_{i=1}^{n}\sum_{j=i+1}^{n} <\textbf{v}_i,\textbf{v}_j>x_ix_j
> \\
> &= \frac{1}{2} \sum_{i=1}^{n}\sum_{j=1}^{n}<\textbf{v}_i,\textbf{v}_j>x_ix_j - \frac{1}{2}\sum_{i=1}^{n}<\textbf{v}_i,\textbf{v}_i>x_ix_i
> \\
> &= \frac{1}{2}(
> \sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^{k}v_{i,f}v_{j,f}x_ix_j - 
> \sum_{i-1}^{n}\sum_{j=1}^{k}v_{i,f}v_{i,f}x_ix_i
> )
> \\
> &= \frac{1}{2} ((\sum_{i=1}^{n}v_{i,f}x_{i})(\sum_{j=1}^{n}v_{j,f}x_j) - \sum_{i=1}^{n}v_{i,f}^{2}x_{i}^{2})
> \\
> &= \frac{1}{2} \sum_{f=1}^{k}
> ((\sum_{i=1}^{n} v_{i,f}x_i)^2 - \sum_{i=1}^{n}v_{i,f}^{}x_{i}^2)
> \end{aligned}
> $$
> 时间复杂度 $O(k \ n)$ 。
>
> 此外，在稀疏情况下，$x$ 中的大多数元素都是 $0$ ( 即 $m(X)$ 很小 )，因此，只需在非零元素上求和即可。因此，在稀疏应用中，对于像 MF 方法这样的推荐系统，因子分解机的计算量是 $O(k \ \bar{m}_D)$。
>
> 例如 $\bar{m}_D=2$ ( 参见 V-A )。

#### B. Factorization Machines as Predictors

FM can be applied to a variety of prediction tasks. 

Among them are:

- **Regression**：$\hat{y}(x)$ can be used directly as the predictor and the optimization criterion is e.g. the minimal least square error on $D$.
- **Binary classification**：the sign of $\hat{y}(x)$ is used and the parameters are optimized for hinge loss or logit loss.
- **Ranking**：the vectors $\textbf{x}$ are ordered by the score of $\hat{y}(x)$ and optimization is done over pairs of instance vectors $(\textbf{x}(a) , \textbf{x}(b)) \in D$ with a pairwise classification loss (e.g. like in [5]). 

In all these cases, regularization terms like L2​ are usually added to the optimization objective to prevent overfitting.

> - **Regression**：$\hat{y}(x)$ 可以直接作为预测器，优化准则为 $D$ 上的最小 MSE。
> - **Binary classification**：用 $\hat{y}(x)$ ，针对 hinge loss 或 logit loss 进行参数优化。
> - **Ranking**：向量 $\textbf{x}$ 按 $\hat{y}(x)$ 的得分排序，对实例向量 $(\textbf{x}(a)， \textbf{x}(b)) \in D$ 的 pairwise 分类损失(如[5]) 进行优化。
>
> 在所有这些情况下，像 L2 这样的正则化项通常被添加到优化目标中以防止过拟合。

#### C. Learning Factorization Machines

As we have shown, FMs have a closed model equation that can be computed in linear time. Thus, the model parameters ($w_0$, $\textbf{w}$ and $\textbf{V}$) of FMs can be learned efficiently by gradient descent methods – e.g. stochastic gradient descent (SGD) – for a variety of losses, among them are square, logit or hinge loss. The gradient of the FM model is:
$$
\begin{equation}
f(x)
=\left\{
\begin{aligned}
&1 
&if \ \theta \ is \ w_0
\\
&x_i, 
&if \ \theta \ is \ w_i
\\
&x_i\sum_{j=1} ^{n}v_{j,f}x_j-v_{i,f}x_i^2,
&if \ \theta \ is \ v_{i,f}
\end{aligned}
\right.
\qquad (4)
\end{equation}
$$
The sum $\sum_{j=1}^{n}v_{j,f}x_j$ is independent of $i$ and thus can be precomputed (e.g. when computing $\hat{y}(x)$). In general, each gradient can be computed in constant time $O(1)$. And all parameter updates for a case $(\textbf{x}, y)$ can be done in $O(k \ n)$ – or $O(k \ m(\textbf{x}))$ under sparsity.

We provide a generic implementation, LIBFM2 , that uses SGD and supports both element-wise and pairwise losses.

> 正如我们所展示的，FM 有一个封闭的模型方程，计算复杂度为线性。因此，对于 square 、logit 和 hinge loss，可以通过梯度下降法(例如随机梯度下降法(SGD))有效地学习 FM 的模型参数($w_0$、$\textbf{w}$ 和 $\textbf{V}$)。FM模型的梯度为：
> $$
> \begin{equation}
> f(x)
> =\left\{
> \begin{aligned}
> &1 
> &if \ \theta \ is \ w_0
> \\
> &x_i, 
> &if \ \theta \ is \ w_i
> \\
> &x_i\sum_{j=1} ^{n}v_{j,f}x_j-v_{i,f}x_i^2,
> &if \ \theta \ is \ v_{i,f}
> \end{aligned}
> \right.
> \qquad (4)
> \end{equation}
> $$
> $\sum_{j=1}^{n}v_{j,f}x_j$ 独立于 $i$，因此可以预计算(例如计算 $\hat{y}(x)$ )。一般来说，每个梯度的计算时间为常数 $O(1)$。并且对于一个 case $(\textbf{x}， y)$ 的所有参数更新都可以在稀疏性下 $O(k \ n)$ -或 $O(k \ m(\textbf{x}))$ 中完成。
>
> 我们提供了一个实现 LIBFM2，它使用 SGD 并同时支持 element-wise  loss 和 pair-wise loss。

#### D. d-way Factorization Machine

The 2-way FM described so far can easily be generalized to a d-way FM:
$$
\hat{y}(x) := w_0 + \sum_{i=1}^{n} w_ix_i
\\
+ \sum_{l=2}^{d}\sum_{i_1=1}^{n}\cdots\sum_{i_l = i_{l-1}+1}^{n}
(\prod_{j=1}^{l}x_{i_j})(\sum_{f=1}^{k_l}\prod_{j=1}^{l}v_{i_{j,f}^{(l)}})
\qquad (5)
$$
where the interaction parameters for the $l$-th interaction are factorized by the PARAFAC model [1] with the model parameters:
$$
\textbf{V}^{(l)} \in \mathbb{R}^{n \times k_l},k_l \in \mathbb{N}_0^+
\qquad (6)
$$
The straight-forward complexity for computing eq. (5) is $O(k_d \ n^d)$. But with the same arguments as in lemma 3.1, one can show that it can be computed in linear time.

> 到目前为止描述的 2-way FM 可以容易地推广到 d-way FM：
> $$
> \hat{y}(x) := w_0 + \sum_{i=1}^{n} w_ix_i
> \\
> + \sum_{l=2}^{d}\sum_{i_1=1}^{n}\cdots\sum_{i_l = i_{l-1}+1}^{n}
> (\prod_{j=1}^{l}x_{i_j})(\sum_{f=1}^{k_l}\prod_{j=1}^{l}v_{i_{j,f}^{(l)}})
> \qquad (5)
> $$
> 其中，$l$ 次交互的交互参数由 PARAFAC 模型[1]分解：
> $$
> \textbf{V}^{(l)} \in \mathbb{R}^{n \times k_l},k_l \in \mathbb{N}_0^+
> \qquad (6)
> $$
> 公式 (5) 的时间复杂度是 $O(k_d \ n^d)$。
> 但是用与引理 3.1 相同的论据，可以证明它可以在线性时间内计算。

#### E. Summary

FMs model all possible interactions between values in the feature vector $\textbf{x}$ using factorized interactions instead of full parametrized ones. This has two main advantages:

1. The interactions between values can be estimated even under high sparsity. Especially, it is possible to generalize to unobserved interactions.
2. The number of parameters as well as the time for prediction and learning is linear. This makes direct optimization using SGD feasible and allows optimizing against a variety of loss functions.

In the remainder of this paper, we will show the relationships between factorization machines and support vector machines as well as matrix, tensor and specialized factorization models.

> FMs 对特征向量 $\textbf{x}$ 中，值之间的所有可能的交互进行建模，使用分解交互而不是完全参数化的交互。这有两个主要优势：
>
> 1. 即使在高稀疏情况下，也可以估计值之间的交互。特别是，它可以推广到未观察到的交叉。
> 2. 参数的数量以及预测和学习的时间是线性的。这使得使用 SGD 进行优化是可行的，并允许针对各种损失函数进行优化。
>
> 在本文的剩余部分，我们将展示 因子分解机 和 支持向量机 之间的关系，以及 矩阵、张量 和 专门的因子分解模型。

