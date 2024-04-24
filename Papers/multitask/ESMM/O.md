# Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate

## ABSTRACT

Estimating post-click conversion rate (CVR) accurately is crucial for ranking systems in industrial applications such as recommendation and advertising. Conventional CVR modeling applies popular deep learning methods and achieves state-of-the-art performance. However it encounters several task-specific problems in practice, making CVR modeling challenging. For example, conventional CVR models are trained with samples of clicked impressions while utilized to make inference on the entire space with samples of all impressions. This causes a sample selection bias problem. Besides, there exists an extreme data sparsity problem, making the model fitting rather difficult. In this paper, we model CVR in a brand-new perspective by making good use of sequential pattern of user actions, i.e., impression → click → conversion. The proposed Entire Space Multi-task Model (ESMM) can eliminate the two problems simultaneously by i) modeling CVR directly over the entire space, ii) employing a feature representation transfer learning strategy. Experiments on dataset gathered from traffic logs of Taobao’s recommender system demonstrate that ESMM significantly outperforms competitive methods. We also release a sampling version of this dataset to enable future research. To the best of our knowledge, this is the first public dataset which contains samples with sequential dependence of click and conversion labels for CVR modeling.

> 准确估计点击后的转化率(CVR)对于工业应用中的 rank 系统（如推荐和广告）至关重要。传统的CVR建模采用了流行的深度学习方法，并达到了最先进的性能。然而，在实际应用中，它遇到了几个与任务特定的问题，使得CVR建模具有挑战性。例如，传统的CVR模型是用点击的曝光样本进行训练的，而用于推断的是所有曝光样本的整体空间。这导致了样本选择偏差问题。此外，存在极端的数据稀疏性问题，使得模型拟合相当困难。在本文中，我们从全新的角度对CVR进行建模，充分利用用户行为的顺序模式，即展现→点击→转化。提出的 Entire Space Multi-task Model(ESMM)可以通过以下方式同时消除这两个问题：i)直接在整体空间上建模CVR，ii)采用特征表示迁移学习策略。在淘宝推荐系统日志数据集上进行的实验表明，ESMM显著优于对照方法。我们也发布了该数据集的采样版本，以便进行未来的研究。据我们所知，这是第一个包含点击和转化label的样本具有时序依赖性的公共数据集，用于CVR建模。

## 1 INTRODUCTION

Conversion rate (CVR) prediction is an essential task for ranking system in industrial applications, such as online advertising and recommendation etc. For example, predicted CVR is used in OCPC (optimized cost-per-click) advertising to adjust bid price per click to achieve a win-win of both platform and advertisers [4]. It is also an important factor in recommender systems to balance users’ click preference and purchase preference.

> 转化率（CVR）预测是工业应用中 rank 系统的一项基本任务，例如在线广告和推荐等。例如，预测的CVR被用于OCPC（optimized cost-per-click）广告中，以调整每次点击的出价，实现平台和广告主的双赢[4]。它也是推荐系统中平衡用户点击偏好和购买偏好的重要因素。通过准确地预测CVR，可以提高广告效果，增加广告主的投资回报率，并提升用户的购买体验。因此，CVR预测在工业应用中具有重要的作用，对于提高rank系统的效能和推动相关业务的发展都具有重要意义。

In this paper, we focus on the task of post-click CVR estimation. To simplify the discussion, we take the CVR modeling in recommender system in e-commerce site as an example. Given recommended items, users might click interested ones and further buy some of them. In other words, user actions follow a sequential pattern of $impression → click → conversion$. In this way, CVR modeling refers to the task of estimating the post-click conversion rate, i.e., $pCVR = p (conversion|click,impression)$.

> 在这篇论文中，我们关注于点击后的转化率（CVR）预估任务。为了简化讨论，我们以电商网站推荐系统中的CVR建模为例。在推荐系统中，用户会看到一些推荐的商品，可能会点击感兴趣的商品，并进一步购买其中的一些。换句话说，用户的行为遵循一种顺序模式，即 $浏览 → 点击 → 转化$。因此，CVR建模指的是估计点击后的转化率的任务，也就是 $pCVR = p (转化|点击,浏览)$。
>
> 这种顺序模式反映了用户与推荐系统交互的自然过程，也为我们的CVR建模提供了有价值的信息。通过理解和利用这种顺序模式，我们可以更准确地预测用户点击后的转化行为，进而优化推荐系统的性能和广告效果。因此，如何有效地建模和利用这种顺序模式，将是我们解决CVR估计问题的关键。

In general, conventional CVR modeling methods employ similar techniques developed in click-through rate (CTR) prediction task, for example, recently popular deep networks [2, 3]. However, there exist several task-specific problems, making CVR modeling challenging. Among them, we report two critical ones encountered in our real practice: i) sample selection bias (SSB) problem [12]. As illustrated in Fig.1, conventional CVR models are trained on dataset composed of clicked impressions, while are utilized to make inference on the entire space with samples of all impressions. SSB problem will hurt the generalization performance of trained models. ii) data sparsity (DS) problem. In practice, data gathered for training CVR model is generally much less than CTR task. Sparsity of training data makes CVR model fitting rather difficult.

> 一般来说，传统的转化率（CVR）建模方法采用了点击率（CTR）预测任务中开发的类似技术，例如最近流行的深度网络[2,3]。然而，存在一些特定于任务的问题，使得CVR建模具有挑战性。其中，我们在实践中遇到了两个关键问题：一）样本选择偏差（SSB）问题[12]。如图1所示，传统的CVR模型是在由点击过的展示（impressions）组成的数据集上进行训练的，然而却用于对所有展示样本的整个空间进行推断。SSB问题会损害训练模型的泛化性能。二）数据稀疏性（DS）问题。在实践中，为训练CVR模型而收集的数据通常远少于CTR任务。训练数据的稀疏性使得CVR模型拟合相当困难。

There are several studies trying to tackle these challenges. In [5], hierarchical estimators on different features are built and combined with a logistic regression model to solve DS problem. However, it relies on a priori knowledge to construct hierarchical structures, which is difficult to be applied in recommender systems with tens of millions of users and items. Oversampling method [11] copies rare class examples which helps lighten sparsity of data but is sensitive to sampling rates. All Missing As Negative (AMAN) applies random sampling strategy to select un-clicked impressions as negative examples [6]. It can eliminate the SSB problem to some degree by introducing unobserved examples, but results in a consistently underestimated prediction. Unbiased method [10] addresses SSB problem in CTR modeling by fitting the truly underlying distribution from observations via rejection sampling. However, it might encounter numerical instability when weighting samples by division of rejection probability. In all, neither SSB nor DS problem has been well addressed in the scenario of CVR modeling, and none of above methods exploits the information of sequential actions.

> 有多项研究试图解决这些挑战。在文献[5]中，研究者构建了基于不同特征的分层估计器，并将其与逻辑回归模型相结合以解决数据稀疏性问题。然而，这种方法依赖于先验知识来构建分层结构，这在拥有数千万用户和物品的推荐系统中很难应用。过采样方法[11]复制稀有类别的样本，这有助于减轻数据的稀疏性，但对采样率敏感。All Missing As Negative (AMAN)采用随机采样策略选择未点击的展示作为负样本[6]。通过引入未观察到的样本，它可以在一定程度上消除样本选择偏差问题，但会导致预测结果一直被低估。Unbiased方法[10]通过拒绝采样从观测结果中拟合真正的潜在分布，从而解决了点击率建模中的样本选择偏差问题。然而，当通过除以拒绝概率来对样本进行加权时，可能会遇到数值不稳定的问题。总的来说，在转化率建模的场景中，样本选择偏差和数据稀疏性问题都没有得到很好的解决，而且上述方法都没有利用顺序动作的信息。

In this paper, by making good use of sequential pattern of user actions, we propose a novel approach named Entire Space Multitask Model (ESMM), which is able to eliminate the SSB and DS problems simultaneously. In ESMM, two auxiliary tasks of predicting the post-view click-through rate (CTR) and post-view clickthrough&conversion rate (CTCVR) are introduced. Instead of training CVR model directly with samples of clicked impressions, ESMM treats pCVR as an intermediate variable which multiplied by pCTR equals to pCTCVR. Both pCTCVR and pCTR are estimated over the entire space with samples of all impressions, thus the derived pCVR is also applicable over the entire space. It indicates that SSB problem is eliminated. Besides, parameters of feature representation of CVR network is shared with CTR network. The latter one is trained with much richer samples. This kind of parameter transfer learning [7] helps to alleviate the DS trouble remarkablely.

> 在本文中，通过充分利用用户行为的顺序模式，我们提出了一种名为“全空间多任务模型（ESMM）”的新方法，该方法能够同时消除样本选择偏差（SSB）和数据稀疏性（DS）问题。在ESMM中，我们引入了预测后验点击率（CTR）和后验点击转化率（CTCVR）两个辅助任务。ESMM不直接使用点击样本训练CVR模型，而是将pCVR视为一个中间变量，该变量乘以pCTR等于pCTCVR。pCTCVR和pCTR都是使用所有样本在整个空间上进行估计的，因此推导出的pCVR也适用于整个空间。这表明消除了SSB问题。此外，CVR网络的特征表示参数与CTR网络共享。后者使用更丰富的样本进行训练。这种参数迁移学习[7]的方法显著地有助于缓解数据稀疏性（DS）问题。

For this work, we collect traffic logs from Taobao’s recommender system. The full dataset consists of 8.9 billions samples with sequential labels of click and conversion. Careful experiments are conducted. ESMM consistently outperforms competitive models, which demonstrate the effectiveness of the proposed approach. We also release our dataset1 for future research in this area.

> 对于这项工作，我们从淘宝的推荐系统中收集了流量日志。完整的数据集包含89亿个样本，带有点击和转化的顺序标签。我们进行了仔细的实验。ESMM始终优于竞争模型，这证明了所提出方法的有效性。我们还发布了我们的数据集1，以供未来在该领域进行研究。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/ESMM/Figure1.png)

## 2 THE PROPOSED APPROACH

#### 2.1 Notation

We assume the observed dataset to be $\mathcal{S}=\left.\left\{\left(\boldsymbol{x}_i, y_i \rightarrow z_i\right)\right\}\right|_{i=1} ^N$ , with sample $(x, y \rightarrow z)$ drawn from a distribution $D$ with domain $\mathcal{X} \times \mathcal{Y} \times \mathcal{Z}$, where $\mathcal{X}$ is feature space, $\mathcal{Y}$ and $\mathcal{Z}$ are label spaces, and $N$ is the total number of impressions. $x$ represents feature vector of observed impression, which is usually a high dimensional sparse vector with multi-fields [8], such as user field, item field etc. $y$ and $z$ are binary labels with $y = 1$ or $z = 1$ indicating whether click or conversion event occurs respectively. $y → z$ reveals the sequential dependence of click and conversion labels that there is always a preceding click when conversion event occurs.

> 我们假设观察到的数据集为 $\mathcal{S}=\left\{\left(\boldsymbol{x}_i, y_i \rightarrow z_i\right)\right\}_{i=1}^N$，其中样本 $(x, y \rightarrow z)$ 是从分布 $D$ 中抽取的，该分布的定义域为 $\mathcal{X} \times \mathcal{Y} \times \mathcal{Z}$。在这里，$\mathcal{X}$ 是特征空间，$\mathcal{Y}$ 和 $\mathcal{Z}$ 是标签空间，而 $N$ 是总的展示次数。$x$ 代表观察到的展示的特征向量，这通常是一个包含多个字段（如用户字段、物品字段等）的高维稀疏向量[8]。$y$ 和 $z$ 是二元标签，其中 $y = 1$ 或 $z = 1$ 分别表示是否发生了点击或转化事件。$y → z$ 揭示了点击和转化标签的顺序依赖性，即当转化事件发生时，前面总是有一个点击事件。
>

Post-click CVR modeling is to estimate the probability of $pCVR = p(z = 1|y = 1, x)$. Two associated probabilities are: post-view click-through rate (CTR) with $pCTR = p(z = 1|x)$ and post-view click&conversion rate (CTCVR) with $pCTCVR = p(y = 1, z = 1|x)$. Given impression $x$, these probabilities follow Eq.(1):
$$
\underbrace{p(y=1, z=1 \mid \boldsymbol{x})}_{p C T C V R}=\underbrace{p(y=1 \mid \boldsymbol{x})}_{p C T R} \times \underbrace{p(z=1 \mid y=1, \boldsymbol{x})}_{p C V R} .
$$

> 点击后转化率（CVR）建模是为了估计概率 $pCVR = p(z = 1|y = 1, x)$。两个相关的概率是：后视点击率（CTR），其概率为 $pCTR = p(z = 1|x)$，以及后视点击和转化率（CTCVR），其概率为 $pCTCVR = p(y = 1, z = 1|x)$。给定展示 $x$，这些概率遵循等式（1）：
>
> $$
> \underbrace{p(y=1, z=1 \mid \boldsymbol{x})}_{pCTCVR} = \underbrace{p(y=1 \mid \boldsymbol{x})}_{pCTR} \times \underbrace{p(z=1 \mid y=1, \boldsymbol{x})}_{pCVR} .
> $$
>
> **详细解释：**
>
> 这个等式描述了三个概率之间的关系：点击后转化率（CVR）、后视点击率（CTR）和后视点击及转化率（CTCVR）。给定一个展示 $x$（例如一个广告或一个商品的展示），我们可以根据这个等式计算出用户点击并进一步转化的联合概率。
>
> - $pCVR = p(z = 1|y = 1, x)$ 表示在给定展示 $x$ 并且用户已经点击（$y = 1$）的条件下，用户进一步转化的概率。
> - $pCTR = p(y = 1|x)$ 表示在给定展示 $x$ 的条件下，用户点击的概率。
> - $pCTCVR = p(y = 1, z = 1|x)$ 表示在给定展示 $x$ 的条件下，用户既点击又转化的联合概率。
>
> 等式（1）表明，CTCVR的概率可以分解为CTR和CVR的乘积。这是概率论中的基本规则，即联合概率可以分解为条件概率的乘积。这个等式在推荐系统和广告系统中非常重要，因为它允许我们从更容易观察和估计的量（CTR和CTCVR）来推导CVR，而CVR通常是直接估计较为困难的。

## 2.2 CVR Modeling and Challenges

Recently deep learning based methods have been proposed for CVR modeling, achieving state-of-the-art performance. Most of them follow a similar Embedding&MLP network architecture, as introduced in [3]. The left part of Fig.2 illustrates this kind of architecture, which we refer to as BASE model, for the sake of simplicity.

> 最近，基于深度学习的方法被提出用于CVR建模，并取得了最先进的性能。它们中的大多数都遵循类似的Embedding&MLP网络架构，如[3]中所介绍。为了简单起见，我们将图2左侧的这种架构称为BASE模型。

In brief, conventional CVR modeling methods directly estimate the post-click conversion rate $p(z = 1|y = 1, x)$. They train models with samples of clicked impressions, i.e.,$\mathcal{S}_c=\left.\left\{\left(\boldsymbol{x}_j, z_j\right) \mid y_j=1\right\}\right|_{j=1} ^M$ . $M$ is the number of clicks over all impressions. Obviously, $S_c$ is a subset of $S$. Note that in $S_c$ , (clicked) impressions without conversion are treated as negative samples and impressions with conversion (also clicked) as positive samples. In practice, CVR modeling encounters several task-specific problems, making it challenging.

> 简而言之，传统的CVR建模方法直接估计点击后的转化率 $p(z = 1|y = 1, x)$。他们使用点击过的展示样本来训练模型，即 $\mathcal{S}_c=\left\{\left(\boldsymbol{x}_j, z_j\right) \mid y_j=1\right\}_{j=1}^M$。$M$ 是所有展示中的点击数量。显然，$S_c$ 是 $S$ 的一个子集。注意，在 $S_c$ 中，没有转化的点击展示被视为负样本，而有转化的点击展示被视为正样本。在实践中，CVR建模遇到了几个特定于任务的问题，使其具有挑战性。
>
> **详细解释：**
>
> 传统的CVR建模方法主要关注于估计用户在点击某个项目后进一步转化的概率，即 $p(z = 1|y = 1, x)$。为了实现这一点，这些方法通常使用那些已经被点击过的展示作为训练数据，这些数据构成了集合 $S_c$。在这个集合中，只有那些被点击过的展示（$y_j=1$）被考虑在内，而未点击的展示则被忽略。
>
> 在 $S_c$ 集合中，进一步区分了两种类型的样本：
>
> 1. **负样本**：那些被点击了但没有进一步转化的展示。这些展示在CVR建模中通常被视为负样本，因为它们代表了用户点击了但没有产生购买或其他转化行为的情况。
> 2. **正样本**：那些既被点击了又产生了转化的展示。这些展示在CVR建模中被视为正样本，因为它们正是模型想要预测和优化的目标。
>
> 然而，在实际应用中，CVR建模面临着几个特定的问题和挑战。例如，数据稀疏性（因为转化事件相对较少）、样本选择偏差（由于只考虑了点击过的样本）以及不平衡的数据分布（转化和非转化的比例可能非常不平衡）等。这些问题都使得CVR建模成为一个具有挑战性的任务。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/ESMM/Figure2.png)

**Sample selection bias (SSB)** [12]. In fact, conventional CVR modeling makes an approximation of $p(z = 1|y = 1, x) ≈ q(z = 1|x_c )$ by introducing an auxiliary feature space $\mathcal{X}_c$ . $\mathcal{X}_c$ represents a limited2 space associated with $\mathcal{S}_c$ .$\forall x_c \in X_c$ there exists a pair $\left(x=x_c, y_x=1\right)$ where $x \in \mathcal{X}$ and $y_x$ is the click label of $x$. In this way, $q(z = 1|x_c )$ is trained over space $\mathcal{X}_c$ with clicked samples of $\mathcal{S}_c$ . At inference stage, the prediction of $p(z = 1|y = 1, x)$ over entire space $\mathcal{X}$ is calculated as $q(z = 1|x)$ under the assumption that for any pair of $(x,y_x = 1)$ where $x \in \mathcal{X}$, $x$ belongs to $\mathcal{X}_c$ . This assumption would be violated with a large probability as $\mathcal{X}_c$ is just a small part of entire space $\mathcal{X}$. It is affected heavily by the randomness of rarely occurred click event, whose probability varies over regions in space $\mathcal{X}$. Moreover, without enough observations in practice, space $\mathcal{X}_c$ may be quite different from $\mathcal{X}$. This would bring the drift of distribution of training samples from truly underling distribution and hurt the generalization performance for CVR modeling.

> 样本选择偏差（SSB）[12]。实际上，传统的CVR建模通过引入一个辅助特征空间 $\mathcal{X}_c$ 来近似$p(z = 1|y = 1, x) \approx q(z = 1|x_c )$。$\mathcal{X}_c$表示与$\mathcal{S}_c$相关的有限空间。对于所有$x_c \in X_c$，都存在一对$(x=x_c, y_x=1)$，其中$x \in \mathcal{X}$，$y_x$是$x$的点击标签。通过这种方式，$q(z = 1|x_c )$是在空间$\mathcal{X}_c$上使用$\mathcal{S}_c$的点击样本进行训练的。在推理阶段，对整个空间$\mathcal{X}$的$p(z = 1|y = 1, x)$的预测被计算为$q(z = 1|x)$，这是基于这样的假设：对于任何$(x,y_x = 1)$对，其中$x \in \mathcal{X}$，$x$属于$\mathcal{X}_c$。由于$\mathcal{X}_c$只是整个空间$\mathcal{X}$的一小部分，因此这个假设很有可能会被违反。它受到很少发生的点击事件的随机性的严重影响，其概率在空间$\mathcal{X}$的不同区域中有所不同。此外，如果实践中没有足够的观察，空间$\mathcal{X}_c$可能与$\mathcal{X}$大不相同。这将导致训练样本的分布与真正的潜在分布发生偏移，从而影响CVR建模的泛化性能。
>
> **详细解释：**
>
> 样本选择偏差（SSB）是CVR建模中的一个重要问题。由于CVR建模主要关注点击后的转化概率，因此训练数据通常仅限于那些已经被点击的样本（即$\mathcal{S}_c$）。这导致模型实际上是在一个受限的特征空间$\mathcal{X}_c$上进行训练的，而不是在整个可能的特征空间$\mathcal{X}$上。
>
> 1. **受限的特征空间**：$\mathcal{X}_c$是与点击样本集$\mathcal{S}_c$相关联的有限空间。这意味着模型只在这个有限的子空间内学习，而不是在整个输入空间$\mathcal{X}$内。
>
> 2. **分布漂移**：由于模型是在$\mathcal{X}_c$上训练的，但在整个$\mathcal{X}$上进行预测，因此存在分布漂移的风险。也就是说，训练数据的分布可能与测试数据的分布不一致，从而影响模型的泛化能力。
>
> 3. **随机性的影响**：点击事件本身是随机的，且在整个输入空间中的分布可能是不均匀的。这意味着在某些区域，点击事件可能非常罕见，导致模型在这些区域的预测能力受限。
>
> 4. **实践与理论的差异**：在实践中，由于数据的稀疏性和随机性，$\mathcal{X}_c$可能与整个输入空间$\mathcal{X}$存在显著差异。这种差异会导致模型在未见过的数据上表现不佳。
>
> 综上所述，样本选择偏差是CVR建模中一个需要认真考虑和解决的问题。它可能导致模型在训练数据和实际数据之间的分布不一致，从而影响模型的预测性能和泛化能力。

**Data sparsity (DS)**. Conventional methods train CVR model with clicked samples of $\mathcal{S}_c$ . The rare occurrence of click event causes training data for CVR modeling to be extremely sparse. Intuitively, it is generally 1-3 orders of magnitude less than the associated CTR task, which is trained on dataset of $\mathcal{S}$ with all impressions. Table 1 shows the statistics of our experimental datasets, where number of samples for CVR task is just 4% of that for CTR task.

> **数据稀疏性（DS）**。传统方法使用$\mathcal{S}_c$中的点击样本来训练CVR模型。点击事件的罕见性导致CVR建模的训练数据极其稀疏。直观地说，它通常比相关的CTR任务少1-3个数量级，CTR任务是在包含所有展示的$\mathcal{S}$数据集上训练的。表1显示了我们实验数据集的统计数据，其中CVR任务的样本数仅为CTR任务样本数的4%。
>
> **详细解释：**
>
> 在CVR建模中，数据稀疏性是一个普遍存在的问题。这是因为点击事件相对于所有的展示事件来说是非常罕见的，导致用于训练CVR模型的数据量非常有限。
>
> 1. **点击事件的罕见性**：在网络广告或电子商务环境中，用户点击广告或产品的比例通常很低。这意味着，与CTR（点击率）任务相比，CVR（转化率）任务可用的训练数据要少得多。CTR任务可以使用所有的展示数据进行训练，而CVR任务则仅限于那些被点击的数据。
>
> 2. **训练数据的稀疏性**：由于点击事件的罕见性，用于CVR建模的训练数据往往非常稀疏。这可能导致模型难以学习到有效的转化模式，因为可用的正面样本（即转化的点击）非常有限。
>
> 3. **与CTR任务的比较**：表1中提供的数据统计显示，CVR任务的样本数量仅为CTR任务样本数量的4%。这直观地说明了CVR建模面临的数据稀疏性问题的严重性。CTR任务可以使用所有的展示数据进行训练，因此其数据量通常比CVR任务大得多。
>
> 数据稀疏性对CVR模型的训练和性能有重大影响。由于可用的训练数据有限，模型可能难以准确预测转化率，尤其是在那些罕见或未见过的数据点上。因此，解决数据稀疏性问题是提高CVR模型性能的关键挑战之一。

It is worth mentioning that there exists other challenges for CVR modeling, e.g. delayed feedback [1]. This work does not focus on it. One reason is that the degree of conversion delay in our system is slightly acceptable. The other is that our approach can be combined with previous work [1] to handle it.

> 值得一提的是，CVR建模还存在其他挑战，例如延迟反馈[1]。但这项工作并不关注这一点。其中一个原因是，我们系统中的转化延迟程度是可以接受的。另一个原因是，我们的方法可以与之前的工作[1]相结合来处理这个问题。

## 2.3 Entire Space Multi-Task Model

The proposed ESMM is illustrated in Fig.2, which makes good use of the sequential pattern of user actions. Borrowing the idea from multi-task learning [9], ESMM introduces two auxiliary tasks of CTR and CTCVR and eliminates the aforementioned problems for CVR modeling simultaneously.

> 所提出的ESMM如图2所示，它充分利用了用户行为的顺序模式。借鉴多任务学习的思想[9]，ESMM引入了两个辅助任务CTR和CTCVR，并同时解决了CVR建模中上述提到的问题。

On the whole, ESMM simultaneously outputs pCTR, pCVR as well as pCTCVR w.r.t. a given impression. It mainly consists of two sub-networks: CVR network illustrated in the left part of Fig.2 and CTR network in the right part. Both CVR and CTR networks adopt the same structure as BASE model. CTCVR takes the product of outputs from CVR and CTR network as the output. There are some highlights in ESMM, which have notable effects on CVR modeling and distinguish ESMM from conventional methods.

> 总体来说，ESMM能同时输出给定展示的pCTR、pCVR和pCTCVR。它主要由两个子网络组成：图2左侧所示的CVR网络和右侧的CTR网络。CVR和CTR网络都采用与BASE模型相同的结构。CTCVR将CVR和CTR网络的输出乘积作为其输出。ESMM有一些亮点，对CVR建模有显著影响，并使ESMM与传统方法有所不同。
>
> **详细解释：**
>
> ESMM（Entire Space Multi-task Model）是一个多任务模型，旨在同时预测点击率（CTR）、转化率（CVR）和点击后的转化率（CTCVR）。以下是关于ESMM的一些关键点：
>
> 1. **模型输出**：对于给定的广告或产品展示，ESMM能够同时输出预测的CTR（pCTR）、预测的CVR（pCVR）和预测的CTCVR（pCTCVR）。
>
> 2. **模型结构**：ESMM由两个主要的子网络构成——CVR网络和CTR网络。这两个网络在图2中分别位于左侧和右侧。值得注意的是，这两个网络都采用了与BASE模型相同的结构，这意味着它们可以共享相同的特征提取和表示学习能力。
>
> 3. **CTCVR的计算**：在ESMM中，CTCVR是通过将CVR网络和CTR网络的输出相乘来得到的。这种计算方式直观地反映了用户行为的顺序性：首先用户需要点击广告或产品（由CTR网络预测），然后才有可能发生转化（由CVR网络预测）。因此，CTCVR实际上就是这两个事件的联合概率。
>
> 4. **模型亮点与影响**：
>    - **多任务学习**：通过同时学习CTR、CVR和CTCVR三个任务，ESMM能够更有效地利用数据，并捕获到任务之间的相关性。
>    - **解决数据稀疏性和样本选择偏差**：由于ESMM在整个曝光空间上进行训练，而不仅仅是在点击样本上，因此它能够有效地解决传统CVR建模中遇到的数据稀疏性和样本选择偏差问题。
>    - **共享底层结构**：CVR网络和CTR网络采用相同的底层结构，这有助于模型捕获到两个任务之间的共享特征和信息，从而提高预测性能。
>
> 综上所述，ESMM通过其独特的设计和多任务学习能力，在CVR建模方面取得了显著的改进，并区别于传统的建模方法。

**Modeling over entire space.** Eq.(1) gives us hints, which can be transformed into Eq.(2).
$$
p(z=1 \mid y=1, x)=\frac{p(y=1, z=1 \mid x)}{p(y=1 \mid x)}
$$
Here $p(y = 1, z = 1|x)$ and $p(y = 1|x)$ are modeled on dataset of $S$ with all impressions. Eq.(2) tells us that with estimation of pCTCVR and pCTR, pCVR can be derived over the entire input space $X$, which addresses the sample selection bias problem directly. This seems easy by estimating pCTR and pCTCVR with individually trained models separately and obtaining pCVR by Eq.(2), which we refer to as DIVISION for simplicity. However, pCTR is a small number practically, divided by which would arise numerical instability. ESMM avoids this with the multiplication form. In ESMM, pCVR is just an intermediate variable which is constrained by the equation of Eq.(1). pCTR and pCTCVR are the main factors ESMM actually estimated over entire space. The multiplication form enables the three associated and co-trained estimators to exploit the sequential pattern of data and communicate information with each other during training. Besides, it ensures the value of estimated pCVR to be in range of [0,1], which in DIVISION method might exceed 1.

> **在整个空间上建模**。 公式(1)给了我们一些提示，可以转化为公式(2)。
> $$
> p(z=1 \mid y=1, x)=\frac{p(y=1, z=1 \mid x)}{p(y=1 \mid x)}
> $$
> 在这里，$p(y = 1, z = 1|x)$ 和 $p(y = 1|x)$ 是在包含所有展示的数据集 $S$ 上进行建模的。方程(2)告诉我们，通过估计pCTCVR和pCTR，可以在整个输入空间 $X$ 上推导出pCVR，这直接解决了样本选择偏差的问题。这看起来很容易，只需通过单独训练的模型分别估计pCTR和pCTCVR，然后通过方程(2)获得pCVR，为简化起见，我们将其称为DIVISION方法。然而，实际上pCTR是一个很小的数，用它进行除法会产生数值不稳定性。ESMM通过乘法形式避免了这个问题。在ESMM中，pCVR只是一个受方程(1)约束的中间变量。pCTR和pCTCVR是ESMM在整个空间上实际估计的主要因素。乘法形式使这三个相关联且共同训练的估计器能够在训练过程中利用数据的顺序模式并相互传递信息。此外，它确保了估计的pCVR值在[0,1]范围内，而在DIVISION方法中，该值可能会超过1。
>
> **详细解释：**
>
> 这段文本详细描述了ESMM（Entire Space Multi-task Model）如何处理CVR（转化率）的估计，特别是与CTR（点击率）和CTCVR（点击后转化率）的关系。以下是几个关键点：
>
> 1. **DIVISION方法的局限性**：虽然可以通过分别估计pCTR和pCTCVR，并使用方程(2)来计算pCVR（这种方法被称为DIVISION），但这种方法存在数值不稳定性的问题，因为pCTR通常是一个很小的数，用它进行除法运算可能导致不准确的结果。
>
> 2. **ESMM的乘法形式**：为了避免DIVISION方法的问题，ESMM采用了乘法形式。在ESMM中，pCVR被视为一个中间变量，它受到方程(1)的约束。模型实际上主要估计的是pCTR和pCTCVR。这种乘法形式不仅使模型能够利用数据的顺序模式（点击->转化），还允许三个相关联的估计器（CTR、CVR、CTCVR）在训练过程中相互传递信息。
>
> 3. **数值范围的保证**：使用ESMM的乘法形式还确保了估计的pCVR值始终在[0,1]的范围内，这是有意义的，因为概率值应该在0到1之间。相比之下，使用DIVISION方法可能会导致pCVR的估计值超过1，这是不合理的。
>
> 总的来说，这段文本强调了ESMM在处理CVR估计时的优势，特别是与简单的DIVISION方法相比。ESMM通过乘法形式结合了CTR、CVR和CTCVR的估计，不仅提高了数值稳定性，还允许模型更好地利用数据中的顺序模式和相关性。

The loss function of ESMM is defined as Eq.(3). It consists of two loss terms from CTR and CTCVR tasks which are calculated over samples of all impressions, without using the loss of CVR task.
$$
\begin{aligned}
L\left(\theta_{c v r}, \theta_{c t r}\right) & =\sum_{i=1}^N l\left(y_i, f\left(\boldsymbol{x}_i ; \theta_{c t r}\right)\right) \\
& +\sum_{i=1}^N l\left(y_i \& z_i, f\left(\boldsymbol{x}_i ; \theta_{c t r}\right) \times f\left(\boldsymbol{x}_i ; \theta_{c v r}\right)\right),
\end{aligned}
$$
where $θ_{ctr}$ and $θ_{cvr}$ are the parameters of CTR and CVR networks and $l(·)$ is cross-entropy loss function. Mathematically, Eq.(3) decomposes $y → z$ into two parts3 : $y$ and $y\&z$​, which in fact makes use of the sequential dependence of click and conversion labels.

> **翻译**：
>
> ESMM的损失函数被定义为等式（3）。它由CTR和CTCVR任务中的两个损失项组成，这些损失项是在所有曝光样本上计算的，而没有使用CVR任务的损失。
>
> $$
> \begin{aligned}
> L\left(\theta_{cvr}, \theta_{ctr}\right) & =\sum_{i=1}^N l\left(y_i, f\left(\boldsymbol{x}_i; \theta_{ctr}\right)\right) \\
> & +\sum_{i=1}^N l\left(y_i \& z_i, f\left(\boldsymbol{x}_i; \theta_{ctr}\right) \times f\left(\boldsymbol{x}_i; \theta_{cvr}\right)\right),
> \end{aligned}
> $$
>
> 其中，$\theta_{ctr}$和$\theta_{cvr}$分别是CTR和CVR网络的参数，而$l(·)$是交叉熵损失函数。从数学上讲，等式（3）将$y → z$分解为两部分：$y$和$y\&z$，这实际上利用了点击和转化标签的顺序依赖性。
>
> **理解**：
>
> 这段论文描述了ESMM（全空间多任务模型）的损失函数。该函数的特点是将点击率（CTR）和点击后的转化率（CTCVR）两个任务的损失结合在一起，而没有直接使用转化率（CVR）的损失。这样做的目的是为了解决直接估算CVR时可能遇到的数据稀疏性和样本选择偏差问题。
>
> 具体来说，损失函数包含两部分：一部分是预测点击率的损失，另一部分是预测点击并购买转化率的损失。这两部分都是在所有曝光样本上进行计算的，而不仅仅是在点击样本上，从而减少了样本选择偏差。
>
> 此外，损失函数的设计还巧妙地利用了点击和转化之间的顺序依赖性。在电商或广告场景中，用户必须先点击广告或商品，然后才可能发生转化（如购买）。因此，通过将CTR和CTCVR的损失结合在一起，模型可以更好地学习到这种顺序依赖性，从而提高预测的准确性。
>
> 总的来说，ESMM的损失函数设计非常巧妙，既解决了数据稀疏性和样本选择偏差的问题，又充分利用了点击和转化之间的顺序依赖性。这使得ESMM在广告推荐和电商领域具有广泛的应用前景。

**Feature representation transfer. **As introduced in section 2.2, embedding layer maps large scale sparse inputs into low dimensional representation vectors. It contributes most of the parameters of deep network and learning of which needs huge volume of training samples. In ESMM, embedding dictionary of CVR network is shared with that of CTR network. It follows a feature representation transfer learning paradigm. Training samples with all impressions for CTR task is relatively much richer than CVR task. This parameter sharing mechanism enables CVR network in ESMM to learn from un-clicked impressions and provides great help for alleviating the data sparsity trouble.

Note that the sub-network in ESMM can be substituted with some recently developed models [2, 3], which might get better performance. Due to limited space, we omit it and focus on tackling challenges encountered in real practice for CVR modeling.

> **特征表示的迁移学习**。如2.2节所介绍，embedding layer 将大规模稀疏输入映射到低维表示向量。它贡献了深度网络中的大部分参数，并且其学习需要大量的训练样本。在ESMM中，CVR网络的 embedding 字典与CTR网络的 embedding 字典是共享的。这遵循了一种特征表示的迁移学习范式。CTR任务的全部曝光训练样本相对于CVR任务来说要丰富得多。这种参数共享机制使得ESMM中的CVR网络能够从未点击的曝光中学习，为缓解数据稀疏性问题提供了很大的帮助。
>
> 请注意，ESMM中的子网络可以替换为最近开发的一些模型[2, 3]，这可能会获得更好的性能。由于篇幅有限，我们省略了这一点，而专注于解决CVR建模实践中遇到的挑战。
>
> **理解**：
>
> 这段论文进一步解释了ESMM（全空间多任务模型）中特征表示的迁移学习策略。在深度网络中，嵌入层负责将高维、稀疏的输入数据转化为低维、密集的表示向量，这是深度学习模型处理大规模稀疏数据的关键步骤。然而，嵌入层包含大量的参数，需要丰富的训练样本来进行有效学习。
>
> 为了解决这个问题，ESMM采用了参数共享的策略，即CVR网络和CTR网络共享同一个嵌入字典。由于CTR任务的训练样本通常比CVR任务的样本丰富得多，这种共享机制使得CVR网络能够从未被点击的曝光样本中学习，从而缓解了数据稀疏性的问题。这是一种迁移学习的应用，即利用一个任务（CTR预测）的丰富数据来帮助另一个任务（CVR预测）的学习。
>
> 此外，作者还提到ESMM中的子网络可以替换为其他先进的模型以提高性能，但由于篇幅限制，他们选择专注于解决CVR建模中遇到的实际挑战。这表明ESMM框架具有一定的灵活性和可扩展性，可以根据具体的应用场景和需求进行定制和优化。

## 3 EXPERIMENTS

#### 3.1 Experimental Setup

**Datasets.** During our survey, no public datasets with sequential labels of click and conversion are found in CVR modeling area. To evaluate the proposed approach, we collect traffic logs from Taobao’s recommender system and release a 1% random sampling version of the whole dataset, whose size still reaches 38GB (without compression). In the rest of the paper, we refer to the released dataset as Public Dataset and the whole one as Product Dataset. Table 1 summarizes the statistics of the two datasets. Detailed descriptions can be found in the website of Public Dataset1 .

> **数据集**。在我们调查期间，发现在CVR建模领域没有带有点击和转化顺序标签的公开数据集。为了评估所提出的方法，我们从淘宝的推荐系统中收集了流量日志，并发布了整个数据集的1%随机抽样版本，其大小仍然达到了38GB（未压缩）。在本文的其余部分，我们将发布的数据集称为公开数据集，将完整的数据集称为产品数据集。表1总结了这两个数据集的统计数据。详细描述可以在公开数据集的网站上找到。

**Competitors.** We conduct experiments with several competitive methods on CVR modeling. (1) BASE is the baseline model introduced in section 2.2. (2) AMAN [6] applies negative sampling strategy and best results are reported with sampling rate searched in {10%, 20%, 50%, 100%}. (3) OVERSAMPLING [11] copies positive examples to reduce difficulty of training with sparse data, with sampling rate searched in {2, 3, 5, 10}. (4) UNBIAS follows [10] to fit the truly underlying distribution from observations via rejection sampling. pCTR is taken as the rejection probability. (5) DIVISION estimates pCTR and pCTCVR with individually trained CTR and CTCVR networks and calculates pCVR by Eq.(2). (6) ESMM-NS is a lite version of ESMM without sharing of embedding parameters.

The first four methods are different variations to model CVR directly based on state-of-the-art deep network. DIVISION, ESMMNS and ESMM share the same idea to model CVR over entire space which involve three networks of CVR, CTR and CTCVR. ESMM-NS and ESMM co-train the three networks and take the output from CVR network for model comparison. To be fair, all competitors including ESMM share the same network structure and hyper parameters with BASE model, which i) uses ReLU activation function, ii) sets the dimension of embedding vector to be 18, iii) sets dimensions of each layers in MLP network to be 360 × 200 × 80 × 2, iv) uses adam solver with parameter $β_1 = 0.9$, $β_2 = 0.999$, $ϵ = 10^{−8}$​ .

> **Competitors方法**。我们在CVR建模上使用了几种有竞争力的方法进行了实验。(1) BASE是2.2节中介绍的基线模型。(2) AMAN [6] 采用了负采样策略，并报告了在采样率为{10%, 20%, 50%, 100%}中搜索得到的最佳结果。(3) OVERSAMPLING [11] 通过复制正样本来降低稀疏数据训练的难度，采样率在{2, 3, 5, 10}中搜索。(4) UNBIAS遵循[10]的方法，通过拒绝采样从观测值中拟合真正的潜在分布。其中，pCTR被用作拒绝概率。(5) DIVISION使用单独训练的CTR和CTCVR网络来估计pCTR和pCTCVR，并通过等式(2)计算pCVR。(6) ESMM-NS是ESMM的一个简化版本，没有共享嵌入参数。
>
> 前四种方法是基于最先进的深度网络直接建模CVR的不同变体。DIVISION、ESMM-NS和ESMM都采用了在整个空间上建模CVR的相同思想，涉及CVR、CTR和CTCVR三个网络。ESMM-NS和ESMM共同训练这三个网络，并取CVR网络的输出进行模型比较。为了公平起见，包括ESMM在内的所有竞争方法都与BASE模型共享相同的网络结构和超参数，即(i) 使用ReLU激活函数，(ii) embedding向量的维度设置为18，(iii) MLP网络中各层的维度设置为360 × 200 × 80 × 2，(iv) 使用参数为 $β_1 = 0.9$, $β_2 = 0.999$, $ϵ = 10^{−8}$ 的 adam求解器。

**Metric.** The comparisons are made on two different tasks: (1) conventional CVR prediction task which estimates pCVR on dataset with clicked impressions, (2) CTCVR prediction task which estimates pCTCVR on dataset with all impressions. Task (2) aims to compare different CVR modeling methods over entire input space, which reflects the model performance corresponding to SSB problem. In CTCVR task, all models calculate pCTCVR by pCTR × pCVR, where: i) pCVR is estimated by each model respectively, ii) pCTR is estimated with a same independently trained CTR network (same structure and hyper parameters as BASE model). Both of the two tasks split the first 1/2 data in the time sequence to be training set while the rest to be test set. Area under the ROC curve (AUC) is adopted as performance metrics. All experiments are repeated 10 times and averaged results are reported.

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/multitask/ESMM/Table1.png)

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/multitask/ESMM/Table2.png)

> **评估指标**。我们在两个不同的任务上进行了比较：(1) 传统的CVR预测任务，该任务在包含已点击展示的数据集上估计pCVR，(2) CTCVR预测任务，该任务在包含所有展示的数据集上估计pCTCVR。任务(2)旨在比较整个输入空间上的不同CVR建模方法，这反映了与SSB问题相对应的模型性能。在CTCVR任务中，所有模型都通过pCTR × pCVR来计算pCTCVR，其中：i) pCVR由每个模型分别估计，ii) pCTR使用相同的独立训练的CTR网络进行估计（与BASE模型具有相同的结构和超参数）。这两个任务都将时间序列中的前1/2数据划分为训练集，其余数据划分为测试集。采用ROC曲线下的面积(AUC)作为性能指标。所有实验均重复10次，并报告平均结果。

## 3.2 Results on Public Dataset

Table 2 shows results of different models on public dataset. (1) Among all the three variations of BASE model, only AMAN performs a little worse on CVR task, which may be due to the sensitive of random sampling. OVERSAMPLING and UNBIAS show improvement over BASE model on both CVR and CTCVR tasks. (2) Both DIVISION and ESMM-NS estimate pCVR over entire space and achieve remarkable promotions over BASE model. Due to the avoidance of numerical instability, ESMM-NS performs better than DIVISION. (3) ESMM further improves ESMM-NS. By exploiting the sequential patten of user actions and learning from un-clicked data with transfer mechanism, ESMM provides an elegant solution for CVR modeling to eliminate SSB and DS problems simultaneously and beats all the competitors. Compared with BASE model, ESMM achieves absolute AUC gain of 2.56% on CVR task, which indicates its good generalization performance even for biased samples. On CTCVR task with full samples, it brings 3.25% AUC gain. These results validate the effectiveness of our modeling method.

> 表2显示了公共数据集上不同模型的结果。(1)
>
> 在BASE模型的所有三种变体中，只有AMAN在CVR任务上的表现稍差，这可能是由于随机采样的敏感性。OVERSAMPLING和UNBIAS在CVR和CTCVR任务上都表现出了相对于BASE模型的改进。
>
> (2) DIVISION和ESMM-NS都在整个空间上估计pCVR，并且与BASE模型相比取得了显著的提升。由于避免了数值不稳定性，ESMM-NS的表现优于DIVISION。
>
> (3) ESMM进一步改进了ESMM-NS。通过利用用户行为的顺序模式和从未点击的数据中学习迁移机制，ESMM为CVR建模提供了一个优雅的解决方案，以同时消除SSB和DS问题，并击败了所有竞争对手。与BASE模型相比，ESMM在CVR任务上实现了2.56%的绝对AUC增益，这表明它即使对于有偏样本也具有良好的泛化性能。在包含全样本的CTCVR任务上，它带来了3.25%的AUC增益。这些结果验证了我们建模方法的有效性。

## 3.3 Results on Product Dataset

We further evaluate ESMM on our product dataset with 8.9 billions of samples, two orders of magnitude larger than public one. To verify the impact of the volume of the training dataset, we conduct careful comparisons on this large scale datasets w.r.t. different sampling rates, as illustrated in Fig.3. First, all methods show improvement with the growth of volume of training samples. This indicates the influence of data sparsity. In all cases except AMAN on 1% sampling CVR task, BASE model is defeated. Second, ESMM-NS and ESMM outperform all competitors consistently w.r.t. different sampling rates. In particular, ESMM maintains a large margin of AUC promotion over all competitors on both CVR and CTCVR tasks. BASE model is the latest version which serves the main traffic in our real system. Trained with the whole dataset, ESMM achieves absolute AUC gain of 2.18% on CVR task and 2.32% on CTCVR task over BASE model. This is a significant improvement for industrial applications where 0.1% AUC gain is remarkable.

> 我们进一步在包含89亿样本的产品数据集上评估ESMM，该数据集的规模比公共数据集大两个数量级。为了验证训练数据集的体积对模型性能的影响，我们如图3所示，对不同采样率下的大规模数据集进行了仔细的比较。首先，随着训练样本量的增加，所有方法都表现出改进。这表明了数据稀疏性的影响。除了AMAN在1%采样的CVR任务上外，在所有情况下，BASE模型都被击败了。其次，ESMM-NS和ESMM在不同采样率下始终优于所有竞争对手。特别是，在CVR和CTCVR任务上，ESMM相对于所有竞争对手都保持了较大的AUC提升幅度。BASE模型是我们真实系统中服务主要流量的最新版本。使用整个数据集进行训练，与BASE模型相比，ESMM在CVR任务上实现了2.18%的绝对AUC增益，在CTCVR任务上实现了2.32%的绝对AUC增益。对于工业应用来说，这是一个显著的改进，因为0.1%的AUC增益都是非常显著的。

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/multitask/ESMM/Figure3.png)

## 4 CONCLUSIONS AND FUTURE WORK

In this paper, we propose a novel approach ESMM for CVR modeling task. ESMM makes good use of sequential patten of user actions. With the help of two auxiliary tasks of CTR and CTCVR, ESMM elegantly tackles challenges of sample selection bias and data sparsity for CVR modeling encountered in real practice. Experiments on real dataset demonstrate the superior performance of the proposed ESMM. This method can be easily generalized to user action prediction in scenario with sequential dependence. In the future, we intend to design global optimization models in applications with multistage actions like request → impression → click → conversion.

> 在本文中，我们提出了一种用于CVR建模任务的新方法ESMM。ESMM充分利用了用户行为的顺序模式。在CTR和CTCVR两个辅助任务的帮助下，ESMM巧妙地解决了CVR建模在实际应用中遇到的样本选择偏差和数据稀疏性的挑战。在实际数据集上的实验证明了所提出的ESMM的优越性能。该方法可以很容易地推广到具有顺序依赖性的场景中的用户行为预测。在未来，我们打算在具有多阶段动作的应用中设计全局优化模型，如请求→展示→点击→转化。