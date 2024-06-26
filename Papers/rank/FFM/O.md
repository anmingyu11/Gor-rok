# Field-aware Factorization Machines for CTR Prediction

## ABSTRACT

Click-through rate (CTR) prediction plays an important role in computational advertising. Models based on degree-2 polynomial mappings and factorization machines (FMs) are widely used for this task. Recently, a variant of FMs, field- aware factorization machines (FFMs), outperforms existing models in some world-wide CTR-prediction competitions. Based on our experiences in winning two of them, in this paper we establish FFMs as an effective method for clas- sifying large sparse data including those from CTR predic- tion. First, we propose efficient implementations for training FFMs. Then we comprehensively analyze FFMs and com- pare this approach with competing models. Experiments show that FFMs are very useful for certain classification problems. Finally, we have released a package of FFMs for public use.

> 点击率（CTR）预测在计算广告中起着重要作用。基于二次多项式映射和因子分解机（FMs）的模型被广泛用于此任务。最近，因子分解机的一种变体，即场感知因子分解机（FFMs），在一些全球性的CTR预测竞赛中表现优于现有模型。基于我们在其中两项竞赛中获胜的经验，本文确立了FFMs作为一种有效的分类大型稀疏数据的方法，包括来自CTR预测的数据。首先，我们提出了训练FFMs的高效实现方法。然后，我们对FFMs进行了全面分析，并与竞争模型进行了比较。实验表明，FFMs对于某些分类问题非常有用。最后，我们已经发布了一个FFMs软件包供公众使用。

## 1.INTRODUCTION

Click-through rate (CTR) prediction plays an important role in advertising industry [1, 2, 3]. Logistic regression is probably the most widely used model for this task [3]. Given a data set with m instances $\left(y_i, \boldsymbol{x}_i\right), i=1, \ldots, m$ , where $y_i$ is the label and $x_i$ is an $n$-dimensional feature vector, the model $w$ is obtained by solving the following optimization problem.
$$
\min _{\boldsymbol{w}} \frac{\lambda}{2}\|\boldsymbol{w}\|_2^2+\sum^m \log \left(1+\exp \left(-y_i \phi_{\mathrm{LM}}\left(\boldsymbol{w}, \boldsymbol{x}_i\right)\right)\right) .
$$
In problem (1), $λ$ is the regularization parameter, and in the loss function we consider the linear model:
$$
\phi_{\mathrm{LM}}(\boldsymbol{w}, \boldsymbol{x})=\boldsymbol{w} \cdot \boldsymbol{x} .
$$
Learning the effect of feature conjunctions seems to be crucial for CTR prediction; see, for example, [1]. Here, we consider an artificial data set in Table 1 to have a better understanding of feature conjunctions. An ad from Gucci has a particularly high CTR on Vogue. This information is however difficult for linear models to learn because they learn the two weights Gucci and Vogue separately. To ad- dress this problem, two models have been used to learn the effect of feature conjunction. The first model, degree-2 poly- nomial mappings (Poly2) [4, 5], learns a dedicate weight for each feature conjunction. The second model, factorization machines (FMs) [6], learns the effect of feature conjunction by factorizing it into a product of two latent vectors. We will discuss details about Poly2 and FMs in Section 2.

A variant of FM called pairwise interaction tensor factor- ization (PITF) [7] was proposed for personalized tag recom- mendation. In KDD Cup 2012, a generalization of PITF called “factor model” was proposed by “Team Opera Solutions” [8]. Because this term is too general and may easily be confused with factorization machines, we refer to it as “field-aware factorization machines” (FFMs) in this paper. The difference between PITF and FFM is that PITF con- siders three special fields including “user,”“item,” and “tag,” while FFM is more general. Because [8] is about the over- all solution for the competition, its discussion of FFM is limited. We can conclude the following results in [8]:

1. They use stochastic gradient method (SG) to solve the optimization problem. To avoid over-fitting, they only train with one epoch.
2. FFM performs the best among six models they tried.

In this paper, we aim to concretely establish FFM as an effective approach for CTR prediction. Our major results are as follows.

- Though FFM is shown to be effective in [8], this work may be the only published study of applying FFMs on CTR prediction problems. To further demonstrate the effectiveness of FFMs on CTR prediction, we present the use of FFM as our major model to win two world-wide CTR competitions hosted by Criteo and Avazu.
- We compare FFMs with two related models, Poly2 and FMs. We first discuss conceptually why FFMs might be better than them, and conduct experiments to see the difference in terms of accuracy and training time.
- We present techniques for training FFMs. They include an effective parallel optimization algorithm for FFMs and the use of early-stopping to avoid over-fitting.
- To make FFMs available for public use, we release an open source software.

This paper is organized as follows. Before we present FFMs and its implementation in Section 3, we discuss the two existing models Poly2 and FMs in Section 2. Experi- ments comparing FFMs with other models are in Section 4. Finally, conclusions and future directions are in Section 5.

> 点击率（CTR）预测在广告行业中扮演着重要角色。逻辑回归可能是此任务中使用最广泛的模型。给定一个包含 $m$ 个实例的数据集 $\left(y_i, \boldsymbol{x}_i\right), i=1, \ldots, m$ ，其中 $y_i$ 是标签，$x_i$ 是一个 $n$ 维特征向量，模型 $w$ 是通过解决以下优化问题获得的。
>
> $$
> \min _{\boldsymbol{w}} \frac{\lambda}{2}\|\boldsymbol{w}\|_2^2+\sum^m \log \left(1+\exp \left(-y_i \phi_{\mathrm{LM}}\left(\boldsymbol{w}, \boldsymbol{x}_i\right)\right)\right) .
> $$
>
> 在问题（1）中，$λ$是正则化参数，而在损失函数中我们考虑线性模型：
>
> $$
> \phi_{\mathrm{LM}}(\boldsymbol{w}, \boldsymbol{x})=\boldsymbol{w} \cdot \boldsymbol{x} .
> $$
>
> 学习特征结合的影响似乎对CTR预测至关重要。在这里，我们考虑表1中的人工数据集，以便更好地理解特征结合。Gucci的广告在Vogue上具有特别高的CTR。然而，线性模型很难学习到这些信息，因为它们分别学习Gucci和Vogue的两个权重。为了解决这个问题，已经使用了两个模型来学习特征结合的影响。第一个模型，二次多项式映射（Poly2），为每个特征结合学习一个专用权重。第二种模型，因子分解机（FMs），通过将特征结合分解为两个潜在向量的乘积来学习其影响。我们将在第2节中详细讨论Poly2和FMs。
>
> 提出了一种FM的变体，称为成对交互张量因子分解（PITF），用于个性化标签推荐。在KDD Cup 2012中，“Team Opera Solutions”提出了一种称为“因子模型”的PITF推广。因为这个术语太过笼统，并且很容易与因子分解机混淆，所以我们在本文中将其称为“场感知因子分解机”（FFMs）。PITF和FFM之间的区别在于，PITF考虑“用户”、“项目”和“标签”这三个特殊领域，而FFM则更为通用。[8]是关于比赛整体解决方案的，因此它对FFM的讨论是有限的。我们可以得出[8]中的以下结果：
>
> 1. 他们使用随机梯度方法（SG）来解决优化问题。为了避免过度拟合，他们只训练一个周期。
>
> 2. FFM在他们尝试的六种模型中表现最好。
>
> 在本文中，我们的目标是具体确定FFM作为CTR预测的有效方法。我们的主要结果如下：
>
> - 尽管[8]显示了FFM的有效性，但这项工作可能是唯一一项关于将FFMs应用于CTR预测问题的已发表研究。为了进一步证明FFMs在CTR预测上的有效性，我们展示了使用FFM作为我们的主要模型，赢得了Criteo和Avazu主办的两场全球性CTR比赛。
>
> - 我们将FFMs与两个相关模型Poly2和FMs进行了比较。我们首先从概念上讨论为什么FFMs可能比它们更好，并进行实验以观察准确性和训练时间方面的差异。
>
> - 我们展示了训练FFMs的技术。它们包括一种有效的FFMs并行优化算法和使用提前停止来避免过度拟合。
>
> - 为了使FFMs可供公众使用，我们发布了一个开源软件。
>
> 本文的组织结构如下。在我们在第3节中介绍FFMs及其实现之前，我们将在第2节中讨论现有的两个模型Poly2和FMs。比较FFMs与其他模型的实验见第4节。最后，结论和未来方向见第5节。
>
> （注：此段文字为对原文的理解和翻译，具体表达可能因语言和文化差异而略有不同。）

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Table1.png)

## 2.POLY2 AND FM

Chang et. al [4] have shown that a degree-2 polynomial mapping can often effectively capture the information of feature conjunctions. Further, they show that by applying a linear model on the explicit form of degree-2 mappings, the training and test time can be much faster than using kernel methods. This approach, referred to as Poly2, learns a weight for each feature pair:
$$
\phi_{\text {Poly } 2}(\boldsymbol{w}, \boldsymbol{x})=\sum_{j_1=1}^n \sum_{j_2=j_1+1}^n w_{h\left(j_1, j_2\right)} x_{j_1} x_{j_2},
$$
where $h(j_1 , j_2 )$ is a function encoding $j_1$ and $j_2$ into a natural number. The complexity of computing (2) is $O\left(\bar{n}^2\right)$, where $\bar{n}$ is the average number of non-zero elements per instance.

FMs proposed in [6] implicitly learn a latent vector for each feature. Each latent vector contains $k$ latent factors, where $k$ is a user-specified parameter. Then, the effect of feature conjunction is modelled by the inner product of two latent vectors:
$$
\phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})=\sum_{j_1=1}^n \sum_{j_2=j_1+1}^n\left(\boldsymbol{w}_{j_1} \cdot \boldsymbol{w}_{j_2}\right) x_{j_1} x_{j_2}
$$
The number of variables is $n × k$, so directly computing costs $O(\bar{n}^2k)$ time. Following [6], by re-writing (3) to
$$
\phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})=\frac{1}{2} \sum_{j=1}^n\left(\boldsymbol{s}-\boldsymbol{w}_j x_j\right) \cdot \boldsymbol{w}_j x_j
$$
where
$$
\boldsymbol{s}=\sum_{j^{\prime}=1}^n \boldsymbol{w}_{j^{\prime}} x_{j^{\prime}}
$$
the complexity is reduced to $O(\bar{n}k)$.

Rendle [6] explains why FMs can be better than Poly2 when the data set is sparse. Here we give a similar illus- tration using the data set in Table 1. For example, there is only one negative training data for the pair (ESPN, Adidas). For Poly2, a very negative weight wESPN,Adidas might be learned for this pair. For FMs, because the prediction of (ESPN, Adidas) is determined by wESPN · wAdidas, and be- cause wESPN and wAdidas are also learned from other pairs (e.g., (ESPN, Nike), (NBC, Adidas)), the prediction may be more accurate. Another example is that there is no training data for the pair (NBC, Gucci). For Poly2, the prediction on this pair is trivial, but for FMs, because wNBC and wGucci can be learned from other pairs, it is still possible to do meaningful prediction.

Note that in Poly2, the naive way to implement $h(j_1,j_2)$ is to consider every pair of features as a new feature [4].1 This approach requires the model as large as $O(n^2)$, which is usually impractical for CTR prediction because of very large n. Vowpal Wabbit (VW) [9], a widely used machine learning package, solves this problem by hashing $j_1$ and $j_2$.2 Our implementation is similar to VW’s approach. Specifically,
$$
h\left(j_1, j_2\right)=\left(\frac{1}{2}\left(j_1+j_2\right)\left(j_1+j_2+1\right)+j_2\right) \bmod B
$$
where the model size $B$ is a user-specified parameter.

In this paper, for the simplicity of formulations, we do not include linear terms and bias term. However, in Section 4, we include them for some experiments.

> Chang等人[4]已经证明，二次多项式映射通常可以有效地捕获特征组合的信息。此外，他们表明，通过对二次映射的显式形式应用线性模型，训练和测试时间可以比使用核方法快得多。这种方法被称为Poly2，它为每对特征学习一个权重：
>
> $$
> \phi_{\text {Poly } 2}(\boldsymbol{w}, \boldsymbol{x})=\sum_{j_1=1}^n \sum_{j_2=j_1+1}^n w_{h\left(j_1, j_2\right)} x_{j_1} x_{j_2},
> $$
>
> 其中 $h(j_1 , j_2)$ 是一个将 $j_1$ 和 $j_2$ 编码为自然数的函数。计算式(2)的复杂度为 $O\left(\bar{n}^2\right)$，其中 $\bar{n}$ 是每个实例中非零元素的平均数量。
>
> 在[6]中提出的因子分解机(FMs)隐式地为每个特征学习一个潜在向量。每个潜在向量包含 $k$ 个潜在因子，其中$k$ 是用户指定的参数。然后，通过两个潜在向量的内积来模拟特征组合的效果：
>
> $$
> \phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})=\sum_{j_1=1}^n \sum_{j_2=j_1+1}^n\left(\boldsymbol{w}_{j_1} \cdot \boldsymbol{w}_{j_2}\right) x_{j_1} x_{j_2}
> $$
>
> 变量的数量是 $n × k$，所以直接计算的时间复杂度是$O(\bar{n}^2k)$。根据[6]，通过将式(3)重写为：
>
> $$
> \phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})=\frac{1}{2} \sum_{j=1}^n\left(\boldsymbol{s}-\boldsymbol{w}_j x_j\right) \cdot \boldsymbol{w}_j x_j
> $$
>
> 其中，
>
> $$
> \boldsymbol{s}=\sum_{j^{\prime}=1}^n \boldsymbol{w}_{j^{\prime}} x_{j^{\prime}}
> $$
>
> 复杂度降低到$O(\bar{n}k)$。
>
> Rendle在[6]中解释了为什么当数据集稀疏时，因子分解机可能会优于Poly2。在这里，我们使用表1中的数据集给出了类似的说明。例如，(ESPN, Adidas)这对组合只有一个负训练数据。对于Poly2，可能会为这对组合学习到一个非常负的权重wESPN,Adidas。而对于因子分解机，因为(ESPN, Adidas)的预测是由wESPN · wAdidas决定的，并且wESPN和wAdidas也是从其他组合（例如，(ESPN, Nike), (NBC, Adidas)）中学习到的，所以预测可能会更准确。另一个例子是，(NBC, Gucci)这对组合没有训练数据。对于Poly2，这对组合的预测是微不足道的，但对于因子分解机，因为wNBC和wGucci可以从其他组合中学习，所以仍然有可能做出有意义的预测。
>
> 请注意，在Poly2中，实现 $h(j_1,j_2)$ 的简单方法是将每对特征视为新特征[4]。这种方法需要模型大小达到 $O(n^2)$，这对于点击率预测来说通常是不切实际的，因为 $n$ 非常大。广泛使用的机器学习包Vowpal Wabbit (VW)[9]通过哈希 $j_1$ 和 $j_2$ 解决了这个问题。我们的实现类似于VW的方法。具体来说，
>
> $$
> h\left(j_1, j_2\right)=\left(\frac{1}{2}\left(j_1+j_2\right)\left(j_1+j_2+1\right)+j_2\right) \bmod B
> $$
>
> 其中模型大小 $B$ 是用户指定的参数。
>
> 在本文中，为了简化公式，我们没有包括线性项和偏置项。然而，在实验部分（第4节）中，我们为某些实验包括了它们。

## 3. FFM

The idea of FFM originates from PITF [7] proposed for recommender systems with personalized tags. In PITF, they assume three available fields including User, Item, and Tag, and factorize (User, Item), (User, Tag), and (Item, Tag) in separate latent spaces. In [8], they generalize PITF for more fields (e.g., AdID, AdvertiserID, UserID, QueryID) and effectively apply it on CTR prediction. Because [7] aims at recommender systems and is limited to three specific fields (User, Item, and Tag), and [8] lacks detailed discussion on FFM, in this section we provide a more comprehensive study of FFMs on CTR prediction. For most CTR data sets like that in Table 1, “features” can be grouped into “fields.” In our example, three features ESPN, Vogue, and NBC, belong to the field Publisher, and the other three features Nike, Gucci, and Adidas, belong to the field Advertiser. FFM is a variant of FM that utilizes this information. To explain how FFM works, we consider the following new example:

Recall that for FMs, $\phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})$ is
$$
\boldsymbol{w}_{\text {ESPN }} \cdot \boldsymbol{w}_{\text {Nike }}+\boldsymbol{w}_{\text {ESPN }} \cdot \boldsymbol{w}_{\text {Male }}+\boldsymbol{w}_{\text {Nike }} \cdot \boldsymbol{w}_{\text {Male }}
$$
In FMs, every feature has only one latent vector to learn the latent effect with any other features. Take ESPN as an example, $w_{ESPN}$ is used to learn the latent effect with Nike ($w_{ESPN} · w_{Nike}$) and Male ($w_{ESPN} · w_{Male}$). However, because Nike and Male belong to different fields, the latent effects of (EPSN, Nike) and (EPSN, Male) may be different.

In FFMs, each feature has several latent vectors. Depending on the field of other features, one of them is used to do the inner product. In our example, $\phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})$ is
$$
\boldsymbol{w}_{\text {ESPN }, \mathrm{A}} \cdot \boldsymbol{w}_{\text {Nike }, \mathrm{P}}+\boldsymbol{w}_{\text {ESPN }, \mathrm{G}} \cdot \boldsymbol{w}_{\text {Male,P }}+\boldsymbol{w}_{\text {Nike }, \mathrm{G}} \cdot \boldsymbol{w}_{\text {Male, }, \mathrm{A}}
$$
We see that to learn the latent effect of (ESPN, NIKE), $w_{ESPN,A}$ is used because Nike belongs to the field Advertiser, and $w_{Nike,P}$ is used because ESPN belongs to the field Publisher. Again, to learn the latent effect of (EPSN, Male), $w_{ESPN,G}$ is used because Male belongs to the field Gender, and $w_{Male,P}$​ is used because ESPN belongs to the field Publisher. Mathematically,
$$
\phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})=\sum_{j_1=1}^n \sum_{j_2=j_1+1}^n\left(\boldsymbol{w}_{j_1, f_2} \cdot \boldsymbol{w}_{j_2, f_1}\right) x_{j_1} x_{j_2}
$$
where $f_1$ and $f_2$ are respectively the fields of $j_1$ and $j_2$. If $f$ is the number of fields, then the number of variables of FFMs is nfk, and the complexity to compute (4) is $O(\bar{n}^2_k)$. It is worth noting that in FFMs because each latent vector only needs to learn the effect with a specific field, usually
$$
k_{\mathrm{FFM}} \ll k_{\mathrm{FM}} .
$$
Table 2 compares the number of variables and the computational complexity of different models.

> FFM的思想源于为具有个性化标签的推荐系统提出的PITF [7]。在PITF中，他们假设有三个可用字段，包括用户、项目和标签，并在单独的潜在空间中对（用户，项目）、（用户，标签）和（项目，标签）进行因式分解。在[8]中，他们将PITF推广到更多字段（例如，AdID、AdvertiserID、UserID、QueryID），并将其有效地应用于点击率预测。由于[7]针对推荐系统，且仅限于三个特定字段（用户、项目和标签），而[8]缺乏对FFM的详细讨论，因此在本节中，我们将对FFM在点击率预测上进行更全面的研究。对于表1中的大多数点击率数据集，“特征”可以分组为“字段”。在我们的示例中，ESPN、Vogue和NBC这三个特征属于发布商字段，而Nike、Gucci和Adidas这三个特征属于广告商字段。FFM是利用这些信息的FM的一个变体。为了解释FFM的工作原理，我们考虑以下新示例：
>
> 
>
> 回顾一下，对于FMs，$\phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})$ 是
>
> $$
> \boldsymbol{w}_{\text {ESPN }} \cdot \boldsymbol{w}_{\text {Nike }}+\boldsymbol{w}_{\text {ESPN }} \cdot \boldsymbol{w}_{\text {Male }}+\boldsymbol{w}_{\text {Nike }} \cdot \boldsymbol{w}_{\text {Male }}
> $$
>
> 在FMs中，每个特征只有一个潜在向量来学习与其他任何特征的潜在影响。以ESPN为例，$w_{ESPN}$ 用于学习与Nike（$w_{ESPN} · w_{Nike}$）和Male（$w_{ESPN} · w_{Male}$）的潜在影响。然而，由于Nike和Male属于不同的字段，因此（EPSN，Nike）和（EPSN，Male）的潜在影响可能不同。
>
> 
>
> 在FFMs中，每个特征都有几个潜在向量。根据其他特征的字段，其中之一用于执行内积。在我们的示例中，$\phi_{\mathrm{FM}}(\boldsymbol{w}, \boldsymbol{x})$ 是
>
> $$
> \boldsymbol{w}_{\text {ESPN }, \mathrm{A}} \cdot \boldsymbol{w}_{\text {Nike }, \mathrm{P}}+\boldsymbol{w}_{\text {ESPN }, \mathrm{G}} \cdot \boldsymbol{w}_{\text {Male,P }}+\boldsymbol{w}_{\text {Nike }, \mathrm{G}} \cdot \boldsymbol{w}_{\text {Male, }, \mathrm{A}}
> $$
>
> 我们看到，为了学习（ESPN，NIKE）的潜在影响，使用了$w_{ESPN,A}$，因为Nike属于广告商字段，而使用了$w_{Nike,P}$，因为ESPN属于发布商字段。同样，为了学习（EPSN，Male）的潜在影响，使用了$w_{ESPN,G}$，因为Male属于性别字段，而使用了$w_{Male,P}$，因为ESPN属于发布商字段。数学上，
>
> $$
> \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})=\sum_{j_1=1}^n \sum_{j_2=j_1+1}^n\left(\boldsymbol{w}_{j_1, f_2} \cdot \boldsymbol{w}_{j_2, f_1}\right) x_{j_1} x_{j_2}
> $$
>
> 其中$f_1$和$f_2$分别是$j_1$和$j_2$的字段。如果$f$是字段的数量，那么FFMs的变量数量就是nfk，计算（4）的复杂度是$O(\bar{n}^2_k)$。值得注意的是，在FFMs中，由于每个潜在向量只需要学习与特定字段的效果，通常
>
> $$
> k_{\mathrm{FFM}} \ll k_{\mathrm{FM}} .
> $$
>
> 表2比较了不同模型的变量数量和计算复杂度。

![Table](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Table_100.png)

#### 3.1 Solving the Optimization Problem

The optimization problem is the same as (1) except that $\phi_{LM}(w,x)$ is replaced by $\phi_{FFM}(w,x)$. Following [7, 8], we use stochastic gradient methods (SG). Recently, some adaptive learning-rate schedules such as [10, 11] have been proposed to boost the training process of SG. We use AdaGrad [10] because [12] has shown its effectiveness on matrix factorization, which is a special case of FFMs.

At each step of SG a data point $(y,x)$ is sampled for updating $w_{j_1 ,f2}$ and $w_{j_2 ,f_1}$ in (4). Note that because $x$ is highly sparse in our application, we only update dimensions with non-zero values. First, the sub-gradients are
$$
\begin{aligned}
& \boldsymbol{g}_{j_1, f_2} \equiv \nabla_{\boldsymbol{w}_{j_1, f_2}} f(\boldsymbol{w})=\lambda \cdot \boldsymbol{w}_{j_1, f_2}+\kappa \cdot \boldsymbol{w}_{j_2, f_1} x_{j_1} x_{j_2}, \\
& \boldsymbol{g}_{j_2, f_1} \equiv \nabla_{\boldsymbol{w}_{j_2, f_1}} f(\boldsymbol{w})=\lambda \cdot \boldsymbol{w}_{j_2, f_1}+\kappa \cdot \boldsymbol{w}_{j_1, f_2} x_{j_1} x_{j_2},
\end{aligned}
$$
where
$$
\kappa=\frac{\partial \log \left(1+\exp \left(-y \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})\right)\right)}{\partial \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})}=\frac{-y}{1+\exp \left(y \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})\right)}
$$
Second, for each coordinate $d = 1, . . . , k$, the sum of squared gradient is accumulated:
$$
\begin{aligned}
& \left(G_{j_1, f_2}\right)_d \leftarrow\left(G_{j_1, f_2}\right)_d+\left(g_{j_1, f_2}\right)_d^2 \\
& \left(G_{j_2, f_1}\right)_d \leftarrow\left(G_{j_2, f_1}\right)_d+\left(g_{j_2, f_1}\right)_d^2
\end{aligned}
$$
Finally, $(w_{j_1},f_2)_d$ and $(w_{j_2},f_1)_d$​ are updated by:
$$
\begin{aligned}
& \left(w_{j_1, f_2}\right)_d \leftarrow\left(w_{j_1, f_2}\right)_d-\frac{\eta}{\sqrt{\left(G_{j_1, f_2}\right)_d}}\left(g_{j_1, f_2}\right)_d \\
& \left(w_{j_2, f_1}\right)_d \leftarrow\left(w_{j_2, f_1}\right)_d-\frac{\eta}{\sqrt{\left(G_{j_2, f_1}\right)_d}}\left(g_{j_2, f_1}\right)_d
\end{aligned}
$$
where $\eta$ is a user-specified learning rate. The initial values of $w$ are randomly sampled from a uniform distribution between $[0, 1/ √ k]$. The initial values of $G$ are set to one in order to prevent a large value of $(G_{j_1},f_2 )^{-\frac{1}{2}}_d$ . The overall procedure is presented in Algorithm 1. 

Empirically, we find that normalizing each instance to have the unit length makes the test accuracy slightly better and insensitive to parameters.

> **论文翻译：**
>
> 优化问题与（1）中的问题相同，只是将 $\phi_{LM}(w,x)$ 替换为 $\phi_{FFM}(w,x)$。根据文献 [7, 8]，我们使用了随机梯度方法（SG）。最近，有人提出了一些自适应学习率计划，例如 [10, 11]，以加速SG的训练过程。我们使用AdaGrad [10]，因为 [12] 显示了其在矩阵分解上的有效性，而矩阵分解是FFM的一个特例。
>
> 在SG的每一步中，都会采样一个数据点 $(y,x)$ 来更新（4）中的 $w_{j_1 ,f2}$ 和 $w_{j_2 ,f_1}$。请注意，由于在我们的应用中 $x$ 是高度稀疏的，因此我们只更新非零值的维度。首先，子梯度为：
>
> $$
> \begin{aligned}
> & \boldsymbol{g}_{j_1, f_2} \equiv \nabla_{\boldsymbol{w}_{j_1, f_2}} f(\boldsymbol{w})=\lambda \cdot \boldsymbol{w}_{j_1, f_2}+\kappa \cdot \boldsymbol{w}_{j_2, f_1} x_{j_1} x_{j_2}, \\
> & \boldsymbol{g}_{j_2, f_1} \equiv \nabla_{\boldsymbol{w}_{j_2, f_1}} f(\boldsymbol{w})=\lambda \cdot \boldsymbol{w}_{j_2, f_1}+\kappa \cdot \boldsymbol{w}_{j_1, f_2} x_{j_1} x_{j_2},
> \end{aligned}
> $$
>
> 其中
>
> $$
> \kappa=\frac{\partial \log \left(1+\exp \left(-y \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})\right)\right)}{\partial \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})}=\frac{-y}{1+\exp \left(y \phi_{\mathrm{FFM}}(\boldsymbol{w}, \boldsymbol{x})\right)}
> $$
>
> 接下来，对于每个坐标 $d = 1, . . . , k$，梯度平方的和被累积：
>
> $$
> \begin{aligned}
> & \left(G_{j_1, f_2}\right)_d \leftarrow\left(G_{j_1, f_2}\right)_d+\left(g_{j_1, f_2}\right)_d^2 \\
> & \left(G_{j_2, f_1}\right)_d \leftarrow\left(G_{j_2, f_1}\right)_d+\left(g_{j_2, f_1}\right)_d^2
> \end{aligned}
> $$
>
> 最后，$(w_{j_1},f_2)_d$ 和 $(w_{j_2},f_1)_d$ 被更新为：
>
> $$
> \begin{aligned}
> & \left(w_{j_1, f_2}\right)_d \leftarrow\left(w_{j_1, f_2}\right)_d-\frac{\eta}{\sqrt{\left(G_{j_1, f_2}\right)_d}}\left(g_{j_1, f_2}\right)_d \\
> & \left(w_{j_2, f_1}\right)_d \leftarrow\left(w_{j_2, f_1}\right)_d-\frac{\eta}{\sqrt{\left(G_{j_2, f_1}\right)_d}}\left(g_{j_2, f_1}\right)_d
> \end{aligned}
> $$
>
> 其中 $\eta$ 是用户指定的学习率。$w$ 的初始值是从 $[0, 1/ √ k]$ 之间的均匀分布中随机采样的。为了防止 $(G_{j_1},f_2 )^{-\frac{1}{2}}_d$ 出现较大值，$G$ 的初始值设为1。整个过程见算法1。
>
> 根据经验，我们发现将每个实例规范化为单位长度可以使测试精度略有提高，并且对参数不敏感。
>
> **理解：**
>
> 这段论文描述了一个使用FFM（Field-aware Factorization Machine，场感知分解机）的优化问题及其解决方案。FFM是FM（Factorization Machine）的一个扩展，能够更有效地捕捉特征之间的交互。论文中，作者使用了随机梯度方法（SG）进行优化，并采用AdaGrad学习率调整策略。
>
> 在每一步的SG过程中，都会采样一个数据点来更新模型的参数。由于数据的稀疏性，只更新那些非零值的特征对应的参数。然后，计算这些参数的梯度，并用AdaGrad方法来更新它们。AdaGrad方法能够自适应地调整每个参数的学习率，使得在训练过程中能够更好地优化模型。
>
> 此外，论文还提到了一种经验性的发现，即将每个实例规范化为单位长度可以提高测试精度，并且使得模型对参数的选择不那么敏感。这可能是因为规范化能够使得模型更好地处理不同尺度的特征，从而提高其泛化能力。



![Alg1](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Alg1.png)

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Table2.png)

#### 3.2 Parallelization on Shared-memory Systems

Modern computers are widely equipped with multi-core CPUs. If these cores are fully utilized, the training time can be significantly reduced. Many parallelization approaches for SG have been proposed. In this paper, we apply Hogwild! [13], which allows each thread to run independently without any locking. Specifically, the for loop at line 3 of Algorithm 1 is parallelized.

In Section 4.4 we run extensive experiments to investigate the effectiveness of parallelization.

> 现代计算机普遍配备了多核CPU。如果能充分利用这些核心，训练时间可以大大减少。目前，已经有许多针对随机梯度下降（SG）的并行化方法被提出。在本文中，我们采用了Hogwild!方法[13]，这种方法允许每个线程在没有任何锁定的情况下独立运行。具体来说，我们对算法1第3行的for循环进行了并行化处理。
>
> 在第4.4节中，我们进行了广泛的实验，以研究并行化的效果。

#### 3.3 Adding Field Information

Consider the widely used LIBSVM data format:
$$
label \ feat1:val1\ feat2:val2 \dots,
$$
where each (feat, val) pair indicates feature index and value. For FFMs, we extend the above format to
$$
label \ field1:feat1:val1 \ field2:feat2:val2 ·
$$
That is, we must assign the corresponding field to each feature. The assignment is easy on some kinds of features, but may not be possible for some others. We discuss this issue on three typical classes of features.

> 考虑广泛使用的LIBSVM数据格式：
>
> $$
> \text{label} \quad \text{feat1:val1} \quad \text{feat2:val2} \quad \dots,
> $$
>
> 其中每对(feat, val)表示特征索引和值。对于FFM，我们将上述格式扩展为
>
> $$
> \text{label} \quad \text{field1:feat1:val1} \quad \text{field2:feat2:val2} \quad \dots
> $$
>
> 也就是说，我们必须为每个特征分配相应的字段。对于某些类型的特征，这种分配是很容易的，但对于其他一些特征，可能无法实现。我们将在三种典型的特征类上讨论这个问题。

#### Categorical Features

For linear models, a categorical feature is commonly transformed to several binary features. For a data instance
$$
\text{Yes} \\ \text{P:ESPN} \\ \text{A:Nike} \\ \text{G:Male},
$$
we generate the following LIBSVM format.
$$
\text{Yes} \\
\text{P-ESPN:1} \\
\text{A-Nike:1} \\
\text{G-Male:1}
$$
Note that according to the number of possible values in a categorical feature, the same number of binary features are generated and every time only one of them has the value 1. In the LIBSVM format, features with zero values are not stored. We apply the same setting to all models, so in this paper, every categorical feature is transformed to several binary ones. To add the field information, we can consider each category as a field. Then the above instance becomes
$$
\text { Yes \  P:P-ESPN:1 \  A:A-Nike:1 \  G:G-Male:1. }
$$

> 对于线性模型，通常将分类特征转换为几个二进制特征。对于一个数据实例
>
> $$
> \text{Yes} \\ \text{P:ESPN} \\ \text{A:Nike} \\ \text{G:Male}
> $$
>
> 我们生成以下LIBSVM格式。
>
> $$
> \text{Yes} \\
> \text{P-ESPN:1} \\
> \text{A-Nike:1} \\
> \text{G-Male:1}
> $$
>
> 请注意，根据分类特征中可能的值的数量，会生成相同数量的二进制特征，并且每次只有一个特征的值为1。在LIBSVM格式中，值为零的特征不会被存储。我们将相同的设置应用于所有模型，因此在本文中，每个分类特征都被转换为几个二进制特征。为了添加字段信息，我们可以将每个类别视为一个字段。那么上述实例变为
>
> $$
> \text { Yes   P:P-ESPN:1   A:A-Nike:1   G:G-Male:1. }
> $$
> 

#### Numerical Features

Consider the following example to predict if a paper will be accepted by a conference. We use three numerical features “accept rate of the conference (AR),”“h-index of the author (Hidx),” and “number of citations of the author (Cite):”

![Diag2](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Diag2.png)

There are two possible ways to assign fields. A naive way is to treat each feature as a dummy field, so the generated data is:
$$
\text { Yes \  AR:AR:45.73 \  Hidx:Hidx:2 \  Cite:Cite:3 }
$$
However, the dummy fields may not be informative because they are merely duplicates of features.

Another possible way is to discretize each numerical feature to a categorical one. Then, we can use the same setting for categorical features to add field information. The generated data looks like:
$$
\text { Yes \  AR:45:1 \  Hidx:2:1 \  Cite:3:1 }
$$
where the AR feature is rounded to an integer. The main drawback is that usually it is not easy to determine the best discretization setting. For example, we may transform 45.73 to “45.7,” “45,” “40,” or even “int(log(45.73)).” In addition, we may lose some information after discretization.

> 考虑以下例子，用于预测一篇论文是否会被一个会议接受。我们使用了三个数值特征：“会议的接受率（AR）”，“作者的h指数（Hidx）”和“作者的引用次数（Cite）”。
>
> 有两种可能的方式来分配字段。一种简单的方式是将每个特征视为一个虚拟字段，所以生成的数据是：
>
> $$
> \text { Yes \  AR:AR:45.73 \  Hidx:Hidx:2 \  Cite:Cite:3 }
> $$
>
> 然而，虚拟字段可能并不具有信息量，因为它们仅仅是特征的重复。
>
> 另一种可能的方式是将每个数值特征离散化为一个分类特征。然后，我们可以使用与分类特征相同的设置来添加字段信息。生成的数据看起来像这样：
>
> $$
> \text { Yes \  AR:45:1 \  Hidx:2:1 \  Cite:3:1 }
> $$
>
> 其中AR特征被四舍五入为整数。主要的缺点是，通常不容易确定最佳的离散化设置。例如，我们可以将45.73转换为“45.7”，“45”，“40”，甚至“int(log(45.73))”。此外，离散化后可能会丢失一些信息。

#### Single-field Features

On some data sets, all features belong to a single field and hence it is meaningless to assign fields to features. Typically this situation happens on NLP data sets. Consider the following example of predicting if a sentence expresses a good mood or not:

In this example the only field is “sentence.” If we assign this field to all words, then FFMs is reduced to FMs. Readers may ask about assigning dummy fields as we do for numerical features. Recall that the model size of FFMs is $O(nfk)$. The use of dummy fields is impractical because $f = n$ and $n$​ is often huge.

> 在某些数据集上，所有特征都属于一个字段，因此为特征分配字段是没有意义的。这种情况通常发生在自然语言处理（NLP）的数据集上。以下是一个预测句子是否表达良好情绪的例子：
>
> 在这个例子中，唯一的字段是“句子”。如果我们把这个字段分配给所有的单词，那么场感知分解机（FFMs）就被简化为因子分解机（FMs）。读者可能会问，是否可以像处理数值特征那样分配虚拟字段。回顾一下，FFMs的模型大小是O(nfk)。使用虚拟字段是不切实际的，因为当f=n且n通常非常大时，模型会变得非常庞大和难以处理。

![Diag3](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Diag3.png)

## 4.EXPERIMENTS

In this section, we first provide the details about the experimental setting in Section 4.1. Then, we investigate the impact of parameters. We find that unlike LM or Poly2, FFM is sensitive to the number of epochs. Therefore, in Section 4.3, we discuss this issue in detail before proposing an early stopping trick. The speedup of parallelization is studied in Section 4.4.

After checking various properties of FFMs, in Sections 4.5- 4.6, we compare FFMs with other models including Poly2 and FMs. They are all implemented by the same SG method, so besides accuracy we can fairly compare their training time. Further in the comparison we include state-of-theart packages LIBLINEAR [14] and LIBFM [15] for training LM/Poly2 and FMs, respectively.

> 在本节中，我们首先在4.1节中提供了关于实验设置的详细信息。然后，我们研究参数的影响。我们发现，与LM或Poly2不同，FFM对迭代次数很敏感。因此，在4.3节中，我们在提出提前停止技巧之前详细讨论了这个问题。并行化的加速效果在4.4节中进行了研究。
>
> 在检查了FFMs的各种属性之后，在4.5-4.6节中，我们将FFMs与其他模型进行了比较，包括Poly2和FMs。它们都是使用相同的随机梯度（SG）方法实现的，因此除了准确性之外，我们还可以公平地比较它们的训练时间。在进一步的比较中，我们分别包括了用于训练LM/Poly2和FMs的最先进的软件包LIBLINEAR[14]和LIBFM[15]。

#### 4.1 Experiment Settings

#### Data Sets

We mainly consider two CTR sets Criteo and Avazu from Kaggle competitions,3 though in Section 4.6 more sets are considered. For feature engineering, we mainly apply our winning solution but remove complicated components.4 For example, our winning solution for Avazu includes the ensemble of 20 models, but here we only use the simplest one. For other details please check our experimental code. A hashing trick is applied to generate $10^6$​ features. The statistics of the two data sets are:

- Va: The validation set mentioned above. 
- Tr: The new training set after excluding the validation set from the original training data.
- TrVa: The original training set. 
- Te: The original test set. 

The labels are not released, so we must submit our prediction to the original evaluation systems to get the score. To avoid over-fitting the test set, the competition organizers divide this data set to two subsets “public set” on which the score is visible during the competition and “private set” on which the score is available after the end of competition. The final rank is determined by the private set.

For both data sets, the labels in the test sets are not publicly available, so we split the available data to two sets for training and validation. The data split follows from how test sets are obtained: For Criteo, the last 6,040,618 lines are used as the validation set; for Avazu, we select the last 4,218,938 lines. We use the following terms to represent different sets of a problem.

For example, CriteoVa means the validation set from Criteo.

![Diag4](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Diag4.png)

> 我们主要考虑Kaggle竞赛中的两个点击率（CTR）数据集：Criteo和Avazu，尽管在4.6节中会考虑更多的数据集。在特征工程方面，我们主要采用了我们获奖的解决方案，但去除了复杂的组件。例如，我们为Avazu设计的获奖解决方案包括了20个模型的集成，但在这里我们只使用最简单的模型。有关其他详细信息，请查看我们的实验代码。我们采用哈希技巧生成了$10^6$个特征。这两个数据集的统计信息如下：
>
> - Va：上述验证集。
> - Tr：从原始训练数据中排除验证集后得到的新训练集。
> - TrVa：原始训练集。
> - Te：原始测试集。
>
> 标签没有公开，因此我们必须将预测结果提交给原始评估系统以获取分数。为了避免过度拟合测试集，比赛组织者将数据集分为两个子集：“公开集”，在比赛期间可以看到其上的分数；“私有集”，比赛结束后可以看到其上的分数。最终排名由私有集确定。
>
> 对于这两个数据集，测试集中的标签并未公开，因此我们将可用数据分为训练和验证两组。数据分割遵循如何获得测试集：对于Criteo，最后6,040,618行用作验证集；对于Avazu，我们选择最后4,218,938行。我们使用以下术语来表示问题的不同集合。
>
> 例如，CriteoVa表示来自Criteo的验证集。

#### Evaluation

Depending on the model, we change $\phi(w, x)$ in (1) to $\phi LM(w, x)$, $\phi Poly2(w, x)$, $\phi FM(w, x)$, or $\phi FFM(w, x)$ introduced in Sections 1-3. For the evaluation criterion, we consider the logistic loss defined as
$$
\log \operatorname{loss}=\frac{1}{m} \sum_{i=1}^m \log \left(1+\exp \left(-y_i \phi\left(\boldsymbol{w}, \boldsymbol{x}_i\right)\right)\right)
$$
where $m$ is the number of test instances.

> 根据模型的不同，我们将(1)中的$\phi(w, x)$更改为前面1-3节中介绍的$\phi LM(w, x)$，$\phi Poly2(w, x)$，$\phi FM(w, x)$或$\phi FFM(w, x)$。对于评估标准，我们考虑定义为如下的逻辑损失：
>
> $$
> \log \operatorname{loss}=\frac{1}{m} \sum_{i=1}^m \log \left(1+\exp \left(-y_i \phi\left(\boldsymbol{w}, \boldsymbol{x}_i\right)\right)\right)
> $$
>
> 其中$m$是测试实例的数量。
>
> 这个逻辑损失函数是用于评估分类模型性能的一种常见指标，特别是在二分类问题中。在这个函数中，$\phi\left(\boldsymbol{w}, \boldsymbol{x}_i\right)$ 是模型对于第$i$个实例$\boldsymbol{x}_i$的预测值（通过模型的参数$\boldsymbol{w}$计算得出），而$y_i$是这个实例的真实标签（通常是+1或-1）。逻辑损失函数衡量了模型预测与实际标签之间的差异，损失值越低，说明模型的预测越准确。
>
> 注意，这里的逻辑损失函数实际上是对每个测试实例的损失进行平均。对于每个实例，损失是通过计算模型预测值与实际标签之间的差异，并通过逻辑函数将这种差异转换为一个介于0和正无穷之间的值。如果模型预测完全正确，那么损失将接近0；如果模型预测完全错误，那么损失将是一个较大的正数。通过这种方式，逻辑损失函数能够很好地反映模型的分类性能。

#### Implementation

We implement LMs, Poly2, FMs, and FFMs all in C++. For FMs and FFMs, we use SSE instructions to boost the efficiency of inner products. The parallelization discussed in Section 3.2 is implemented by OpenMP [16]. Our implementations include linear terms and bias term as they improve performance in some data sets. These terms should be used in general as we seldom see them to be harmful.

Note that for code extensibility the field information is stored regardless of the model used. For non-FFM models, the implementation may become slightly faster by a simpler data structure without field information, but our conclusions from experiments should remain the same.

> 我们用C++实现了LMs、Poly2、FMs和FFMs。对于FMs和FFMs，我们使用SSE指令来提高内积的效率。3.2节中讨论的并行化是通过OpenMP[16]实现的。我们的实现包括线性项和偏置项，因为它们可以提高某些数据集的性能。这些项通常应该使用，因为我们很少看到它们会产生负面影响。
>
> 请注意，为了代码的可扩展性，无论使用哪种模型，都会存储字段信息。对于非FFM模型，通过没有字段信息的更简单数据结构可能会使实现稍微快一些，但我们从实验中得出的结论应该保持不变。

### 4.2 Impact of Parameters

We conduct experiments to investigate the impact of k, λ, and η. The results can be found in Figure 1. Regarding the parameter k, results in Figure 1a show that it does not affect the logloss much. In Figure 1b, we present the relationship between λ and logloss. If λ is too large, the model is not able to achieve a good performance. On the contrary, with a small λ, the model gets better results, but it easily overfits the data. We observe that the training logloss keeps decreasing. For the parameter η, Figure 1c shows that if we apply a small η, FFMs will obtain its best performance slowly. However, with a large η, FFMs are able to quickly reduce the logloss, but then over-fitting occurs. From the results in Figures 1b and 1c, there is a need of early-stopping that will be discussed in Section 4.3.

> 我们进行实验以研究k、λ和η的影响。结果可以在图1中找到。关于参数k，图1a的结果表明它对logloss的影响不大。在图1b中，我们展示了λ和logloss之间的关系。如果λ太大，模型就无法取得良好的表现。相反，λ较小时，模型会得到更好的结果，但很容易过度拟合数据。我们观察到训练logloss持续下降。对于参数η，图1c表明，如果我们应用较小的η，FFMs将缓慢地达到其最佳性能。然而，当η较大时，FFMs能够快速降低logloss，但随后会发生过度拟合。从图1b和1c的结果来看，需要进行早期停止，这将在4.3节中讨论。

### 4.3 Early Stopping

Early stopping, which terminates the training process before reaching the best result on training data, can be used to avoid over-fitting for many machine learning problems [17, 18, 19]. For FFM, the strategy we use is: 

1. Split the data set into a training set and a validation set. 
2. At the end of each epoch, use the validation set to calculate the loss. 
3. If the loss goes up, record the number of epochs. Stop or go to step 4.
4. If needed, use the full data set to re-train a model with the number of epochs obtained in step 3.

A difficulty in applying early stopping is that the logloss is sensitive to the number of epochs. Then the best epoch on the validation set may not be the best one on the test set. We have tried other approaches to avoid the overfitting such as lazy update5 and ALS-based optimization methods. However, results are not as successful as that by early stopping of using a validation set.

> 早期停止是指在训练数据上达到最佳结果之前终止训练过程，它可用于避免许多机器学习问题的过度拟合[17, 18, 19]。对于FFM，我们使用的策略是：
>
> 1. 将数据集分为训练集和验证集。
>
> 2. 在每个周期结束时，使用验证集来计算损失。
>
> 3. 如果损失上升，记录周期数。停止或转到步骤4。
>
> 4. 如需必要，使用完整数据集和步骤3中获得的周期数重新训练模型。
>
> 应用早期停止的一个难点在于logloss对周期数很敏感。那么验证集上的最佳周期可能并不是测试集上的最佳周期。我们已尝试其他方法来避免过度拟合。

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Figure1.png)

#### 4.4 Speedup

Because the parallelization of SG may cause a different convergence behavior, we experiment with different numbers of threads in Figure 2. Results show that our parallelization still leads to similar convergence behavior. With this property we can define the speedup as:
$$
\frac{\text { Running time of one epoch with a single thread }}{\text { Running time of one epoch with multiple threads }} \text {. }
$$
The result in Figure 3 shows a good speedup when the number of threads is small. However, if many threads are used, the speedup does not improve much. An explanation is that if two or more threads attempt to access the same memory address, one must wait for its term. This kind of conflicts can happen more often when more threads are used.

> 由于随机梯度下降（SG）的并行化可能会导致不同的收敛行为，我们在图2中实验了不同数量的线程。结果表明，我们的并行化仍然导致了类似的收敛行为。有了这个性质，我们可以将加速比定义为：
>
> $$
> \frac{\text { 单线程一个时代的运行时间 }}{\text { 多线程一个时代的运行时间 }} \text {. }
> $$
>
> 图3的结果表明，当线程数量较少时，加速效果良好。然而，如果使用了许多线程，加速比并没有太大改善。一种解释是，如果两个或更多线程尝试访问相同的内存地址，则必须等待其结束。当使用更多线程时，这种冲突可能更频繁地发生。

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Figure2.png)

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/rank/FFM/Figure3.png)

#### 4.5 Comparison with LMs, Poly2, and FMs on Two CTR Competition Data Sets

To have a fair comparison, we implement the same SG method for LMs, Poly2, FMs, and FFMs. Further, we compare with two state-of-the-art packages:

- LIBLINEAR: a widely used package for linear models. For L2-regularized logistic regression, it implements two optimization methods: Newton method to solve the primal problem, and coordinate descent (CD) method to solve the dual problem. We used both for checking how optimization methods affect the performance; see the discussion in the end of this sub-section. Further, the existing Poly2 extension of LIBLINEAR does not support the hashing trick,6 so we conduct suitable modifications and denote it as LIBLINEAR-Hash in this paper
- LIBFM: As a widely used library for factorization machines, it supports three optimization approaches including stochastic gradient method (SG), alternating least squares (ALS), and Markov Chain Monte Carlo (MCMC). We tried all of them and found that ALS is significantly better than the other two in terms of logloss. Therefore, we consider ALS in our experiments.

For the parameters in all models, from a grid of points we select those that lead to the best performance on the validation sets. Every optimization algorithm needs a stopping condition; we use the default setting for Newton method and coordinate descent (CD) method by LIBLINEAR. For each of the other models, we need a validation set to check which iteration leads to the best validation score. After we obtain the best number of iterations, we re-train the model up to that iteration. Results on Criteo and Avazu with the list of parameters used can be found in Table 3. Clearly, FFMs outperform other models in terms of logloss, but it also requires longer training time than LMs and FMs. On the other end, though the logloss of LMs is worse than other models, it is significantly faster. These results show a clear trade-off between logloss and speed. Poly2 is the slowest among all models. The reason might be the expensive computation of (2). FM is a good balance between logloss and speed.

For LIBFM, it performs closely to our implementation of FMs in terms of logloss on Criteo. 7 However, we see that our implementation is significantly faster. We provide three possible reasons:

- The ALS algorithm used by LIBFM is more complicated than the SG algorithm we use. 
- We use an adaptive learning rate strategy in SG. 
- We use SSE instructions to boost inner product operations.

Because logistic regression is a convex problem, ideally, for either LM or Poly2, the three optimization methods (SG, Newton, and CD) should generate exactly the same model if they converge to the global optimum. However, practically results are slightly different. In particular, LM by SG is better than the two LIBLINEAR-based models on Avazu. In our implementation, LM via SG only loosely solves the optimization problem. Our experiments therefore indicate that the stopping condition of optimization methods can affect the performance of the resulting model even if the problem is convex.

> 对于L2正则化逻辑回归，它实现了两种优化方法：求解原问题的牛顿法和求解对偶问题的坐标下降（CD）法。我们使用了这两种方法来检查优化方法如何影响性能；请参见本子节末尾的讨论。此外，LIBLINEAR现有的Poly2扩展不支持哈希技巧，6因此我们进行了适当的修改，并在本文中将其表示为LIBLINEAR-Hash。
>
> - LIBFM：作为一个广泛使用的因子分解机库，它支持三种优化方法，包括随机梯度方法（SG）、交替最小二乘法（ALS）和马尔可夫链蒙特卡洛（MCMC）。我们尝试了所有方法，并发现就对数损失而言，ALS明显优于其他两种方法。因此，在我们的实验中，我们考虑了ALS。
>
> 对于所有模型中的参数，我们从一系列点中选择那些在验证集上表现最佳的参数。每种优化算法都需要一个停止条件；我们对牛顿法和坐标下降（CD）法使用LIBLINEAR的默认设置。对于其他每个模型，我们需要一个验证集来检查哪次迭代导致最佳的验证分数。在获得最佳迭代次数后，我们重新训练模型以达到该迭代次数。Criteo和Avazu的结果以及所使用的参数列表可以在表3中找到。显然，就对数损失而言，FFMs优于其他模型，但它也需要比LMs和FMs更长的训练时间。另一方面，尽管LMs的对数损失比其他模型差，但它的速度明显更快。这些结果表明了对数损失和速度之间的明显权衡。Poly2是所有模型中最慢的。原因可能是（2）的计算成本高昂。因子分解机（FM）在对数损失和速度之间达到了良好的平衡。
>
> 对于LIBFM，在Criteo的对数损失方面，它的表现与我们的因子分解机（FMs）实现非常接近。7但是，我们看到我们的实现明显更快。我们提供了三个可能的原因：
>
> - LIBFM使用的ALS算法比我们使用的SG算法更复杂。
>
> - 我们在SG中使用了自适应学习率策略。
>
> - 我们使用SSE指令来加速内积操作。
>
> 因为逻辑回归是一个凸问题，所以理想情况下，对于LM或Poly2，如果三种优化方法（SG、牛顿法和CD）收敛到全局最优，它们应该生成完全相同的模型。然而，实际上结果略有不同。特别是，在Avazu上，通过SG得到的LM优于两个基于LIBLINEAR的模型。在我们的实现中，通过SG得到的LM只是松散地解决了优化问题。因此，我们的实验表明，即使问题是凸的，优化方法的停止条件也会影响所得模型的性能。



# 总结

FFM（Field-aware Factorization Machine）的核心思想和实现方式可以归纳如下：

**核心思想：**

1. **学习特征间的交互作用**：FFM的核心思想是学习特征间的交互作用，特别是不同字段（field）间的复杂交互作用。这是FFM区别于其他模型的一个重要特点。

2. **字段感知的隐向量**：与FM（Factorization Machine）模型相比，FFM引入了“字段”的概念。在FFM中，每个特征不仅有一个对应的隐向量，而是针对其他每个字段都有一个特定的隐向量。这意味着，当某个特征与不同字段的特征进行交互时，它会使用不同的隐向量。这种“字段感知”的特性使得FFM能够更精细地捕捉特征之间的交互信息。

**实现方式：**

1. **特征嵌入**：首先，对每个字段（如用户ID、商品ID、类型等）学习嵌入（embedding），得到每个字段的低维向量表达。这些嵌入向量捕捉了字段的内在信息，并为后续的交互计算提供了基础。

2. **交互特征计算**：接下来，计算每个字段与其他所有字段的交互特征。在FFM中，当两个来自不同字段的特征进行交互时，它们会选择与对方字段相对应的隐向量进行点积运算，从而得到交互特征的值。这一过程能够捕捉字段之间的复杂交互关系。

3. **模型训练和预测**：通过优化一个损失函数（如逻辑回归损失），来训练FFM模型的参数，包括所有特征的嵌入向量和可能的偏置项。一旦模型训练完成，就可以使用这些学习到的参数来进行预测。在预测阶段，FFM会根据输入的特征向量和训练得到的参数计算出一个预测值，该值表示了某种目标事件发生的概率或程度（如用户点击广告的概率）。

需要注意的是，虽然FFM能够提供更精细的特征交互建模能力，但它也增加了模型的复杂性。由于每个特征都需要学习与其他每个字段相对应的隐向量，因此FFM的参数数量会显著增加。这可能导致模型训练需要更多的数据和计算资源，并可能增加过拟合的风险。在实际应用中，需要权衡模型的复杂性和性能之间的关系。



