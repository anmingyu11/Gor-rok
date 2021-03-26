# ITEM2VEC: NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING 

## ABSTRACT

Many Collaborative Filtering (CF) algorithms are itembased in the sense that they analyze item-item relations in order to produce item similarities. Recently, several works in the field of Natural Language Processing (NLP) suggested to learn a latent representation of words using neural embedding algorithms. Among them, the Skip-gram with Negative Sampling (SGNS), also known as word2vec, was shown to provide state-of-the-art results on various linguistics tasks. In this paper, we show that itembased CF can be cast in the same framework of neural word embedding. Inspired by SGNS, we describe a method we name item2vec for item-based CF that produces embedding for items in a latent space. The method is capable of inferring item-item relations even when user information is not available. We present experimental results that demonstrate the effectiveness of the item2vec method and show it is competitive with SVD. 

**Index terms** – skip-gram, word2vec, neural word embedding, collaborative filtering, item similarity, recommender systems, market basket analysis, itemitem collaborative filtering, item recommendations.

> 许多协同过滤(CF)算法都是基于 item 的，因为它们分析item - item 关系以产生 item 相似性。最近，自然语言处理(NLP)领域的一些研究工作提出了利用 neural embedding 算法学习单词的潜在表示。其中，Skip-gram with Negative sampling(SGNS)，也被称为 word2vec，在各种语言学任务中提供了最先进的结果。在本文中，我们证明了基于 item 的 CF 可以在相同的 neural word embedding 框架下进行转换。受SGNS的启发，我们描述了一种名为 item2vec 的方法，用基于 item 的 CF，生成了在潜在空间中为 item embedding 。即使在用户信息不可用的情况下，该方法也能够推断 item-item 关系。实验结果证明了 item2vec 方法的有效性，并表明该方法与 SVD 具有较好的竞争优势
>
> **索引词** - skip-gram, word2vec, neural word embedding, collaborative filtering, item similarity, recommender systems, market basket analysis, itemitem collaborative filtering, item recommendations.

## 1. INTRODUCTION AND RELATED WORK

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Fig1.png)

**Fig. 1. Recommendations in Windows 10 Store based on similar items to Need For Speed. **

Computing item similarities is a key building block in modern recommender systems. While many recommendation algorithms are focused on learning a low dimensional embedding of users and items simultaneously [1, 2, 3], computing item similarities is an end in itself. Item similarities are extensively used by online retailers for many different recommendation tasks. This paper deals with the overlooked task of learning item similarities by embedding items in a low dimensional space. 

Item-based similarities are used by online retailers for recommendations based on a single item. For example, in the Windows 10 App Store, the details page of each app or game includes a list of other similar apps titled “People also like”. This list can be extended to a full page recommendation list of items similar to the original app as shown in Fig. 1. Similar recommendation lists which are based merely on similarities to a single item exist in most online stores e.g., Amazon, Netflix, Google Play, iTunes store and many others.

The single item recommendations are different than the more “traditional” user-to-item recommendations because they are usually shown in the context of an explicit user interest in a specific item and in the context of an explicit user intent to purchase. Therefore, single item recommendations based on item similarities often have higher Click-Through Rates (CTR) than user-to-item recommendations and consequently responsible for a larger share of sales or revenue. 

Single item recommendations based on item similarities are used also for a variety of other recommendation tasks: In “candy rank” recommendations for similar items (usually of lower price) are suggested at the check-out page right before the payment. In “bundle” recommendations a set of several items are grouped and recommended together. Finally, item similarities are used in online stores for better exploration and discovery and improve the overall user experience. It is unlikely that a user-item CF method, that learns the connections between items implicitly by defining slack variables for users, would produce better item representations than a method that is optimized to learn the item relations directly. 

Item similarities are also at the heart of item-based CF algorithms that aim at learning the representation directly from the item-item relations [4, 5]. There are several scenarios where item-based CF methods are desired: in a large scale dataset, when the number of users is significantly larger than the number of items, the computational complexity of methods that model items solely is significantly lower than methods that model both users and items simultaneously. For example, online music services may have hundreds of millions of enrolled users with just tens of thousands of artists (items). 

In certain scenarios, the user-item relations are not available. For instance, a significant portion of today’s online shopping is done without an explicit user identification process. Instead, the available information is per session. Treating these sessions as “users” would be prohibitively expensive as well as less informative.

Recent progress in neural embedding methods for linguistic tasks have dramatically advanced state-of-the-art NLP capabilities [6, 7, 8, 12]. These methods attempt to map words and phrases to a low dimensional vector space that captures semantic relations between words. Specifically, Skip-gram with Negative Sampling (SGNS), known also as word2vec [8], set new records in various NLP tasks [7, 8] and its applications have been extended to other domains beyond NLP [9, 10]. 

In this paper, we propose to apply SGNS to itembased CF. Motivated by its great success in other domains, we suggest that SGNS with minor modifications may capture the relations between different items in collaborative filtering datasets. To this end, we propose a modified version of SGNS named item2vec. We show that item2vec can induce a similarity measure that is competitive with an itembased CF using SVD, while leaving the comparison to other more complex methods to a future research.

The rest of the paper is organized as follows: Section 2 overviews the SGNS method. Section 3 describes how to apply SGNS to item-based CF. In Section 4, we describe the experimental setup and present qualitative and quantitative results. 

> 计算 item 相似度是现代推荐系统中的关键组成部分。 虽然许多推荐算法专注于同时学习 user 和 item 的低维嵌入[1,2,3]，计算 item 相似性本身就是目的。 online retailers 广泛使用 item 相似性来完成许多不同的推荐任务。 本本文讨论了通过在低维空间中 embedding 项 item 来学习项目相似性这一被忽视的任务。
>
> 基于 item 的相似性被 online retailers 用于基于单个 item 的推荐。例如，在Windows 10 App Store中，每个 App 或 Game 的详细信息页面都包含一个名为“People also like”的其他类似应用列表。这个列表可以扩展为与原始 App 相似的完整页面推荐列表，如图1所示。类似的推荐列表仅仅基于与单个的 item 的相似性在大多数在线商店中存在，如 Amazon，Netflix，Google Play, iTunes store和许多其他的。
>
> 单个 item 推荐与更“传统”的 user-to-item 推荐不同，因为它们通常是在用户对特定 item 具有明确的兴趣的上下文中，以及在明确的 user 购买意图的上下文中显示的。 因此，基于 item 相似性的单个 item 建议通常比  user-to-item 的建议具有更高的点击率（CTR），因此在销售额或收入中所占的比例更高。
>
> 基于 item 相似度的单一 item 推荐也可用于其他各种推荐任务：在“candy rank”中，类似 item (通常价格较低) 的推荐会在付款前的结账页面上提出。在 “bundle” 推荐中，一组由几个 item 组成的 item被组合在一起进行推荐。最后，在 online store 中使用 item 相似性，以便更好地探索和发现，并改善整体用户体验。通过为 user 定义松弛变量 (在优化问题中，松弛变量是添加到不等式约束中以将其转换为等式的变量。) 来隐式学习 item 之间的联系的 user-to-item CF 方法，比经过优化以直接学习 item 关系的方法，不可能产生更好的 item 表示。
>
> item 相似性也是基于 item 的 CF 算法的核心，该算法旨在直接从 item - item 关系学习表示[4，5]。有几种场景需要基于 item 的 CF 方法：在大规模数据集中，当 user 数量明显大于 item 数量时，仅对 item 建模的方法的计算复杂度明显低于同时对 user 和 item 建模的方法。例如，在线音乐服务可能有数亿注册用户，只有数万名艺术家(item)。
>
> 在某些场景下，user-item 关系不可用。例如，今天的网上购物有很大一部分是在没有明确的用户识别过程的情况下完成的。相反，可用的信息是按 session 计算的。把这些 session 当作“user”对待，不仅代价高得令人望而却步，而且信息量也会更少。
>
> 近年来，用于语言任务的 neural embdding 方法的最新进展已极大地提高了NLP技术的先进水平[6、7、8、12]。这些方法试图将单词和短语映射到捕获单词之间语义关系的低维向量空间。其中，也被称为 word2vec [8]的S kip-gram with Negative Sampling (SGNS) 在各种自然语言处理任务中屡创新高[7,8]，其应用已经扩展到自然语言处理之外的其他领域[9,10]。
>
> 在本文中，我们建议将 SGNS 应用于基于 item 的CF中。由于 SGNS 在其他领域的巨大成功，我们建议 SGNS 稍加修改就可以捕获协同过滤数据集中不同 item 之间的关系。为此，我们提出了 SGNS 的一个修改版本，命名为 item2vec。我们表明，item2vec可以 induce 一个相似度度量，与使用SVD的item-based-CF相竞争。而将与其他更复杂方法的比较留到未来的研究中。
>
> 论文的其余部分组织如下：第 2 节概述了 SGNS 背景下的相关工作，第 3 节描述了如何将 SGNS 应用于 item-based CF，第4节描述了实验设置，并给出了定性和定量的 结果。

## 2. SKIP-GRAM WITH NEGATIVE SAMPLING

Skip-gram with negative sampling (SGNS) is a neural word embedding method that was introduced by Mikolov et. al in [8]. The method aims at finding words representation that captures the relation between a word to its surrounding words in a sentence. In the rest of this section, we provide a brief overview of the SGNS method. 

Given a sequence of words $(w_i)_{i=1}^{K}$ from a finite vocabulary $W = \{w_i\}_{i=1}^{W}$, the Skip-gram objective aims at maximizing the following term:
$$
\frac{1}{K} \sum_{i=1}^{K} \sum_{-c \le j \le c, j \ne 0} \ log \ p(w_{i+j} | w_i) 
\qquad (1)
$$
where $c$ is the context window size (that may depend on $w_i$ ) and $p(w_j | w_i)$ is the softmax function: 
$$
p(w_j|w_i) = \frac{exp(u_j^Tv_j)}{\sum_{k \in I_w}exp(u_i^Tv_k)} \qquad(2)
$$
where $u_i \in U(\subset \mathbb{R}^m)$ and $v_i \in V(\subset \mathbb{R}^m)$ are latent vectors that correspond to the target and context representations for the word $w_i \in W$ , respectively, $I_w \triangleq \{1,\dots,|W|\}$ and the parameter $m$ is chosen empirically and according to the size of the dataset. 

Using Eq. (2) is impractical due to the computational complexity of $\nabla p(w_j|w_i)$ , which is a linear function of the vocabulary size $W$ that is usually in size of $10^5 -10^6$.

Negative sampling comes to alleviate the above computational problem by the replacement of the softmax function from Eq.(2) with 
$$
p(w_j|w_i) = \sigma(u_i^Tv_j)\prod_{k=1}^{N}\sigma(-u_i^Tv_k)
$$
where $\sigma(x) = 1/1+exp(-x)$ , $N$ is a parameter that determines the number of negative examples to be drawn per a positive example. A negative word $w_i$ is sampled from the unigram distribution raised to the $3/4$rd power. This distribution was found to significantly outperform the unigram distribution, empirically [8]. 

In order to overcome the imbalance between rare and frequent words the following subsampling procedure is proposed [8]: Given the input word sequence, we discard each word $w$ with a probability $p(discard|w) = 1- \sqrt{\frac{\rho}{f(w)}}$ where $f(w)$ is the frequency of the word $w$ and $\rho$ is a prescribed threshold. This procedure was reported to accelerate the learning process and to improve the representation of rare words significantly [8]. 

Finally, $U$ and $V$ are estimated by applying stochastic gradient ascent with respect to the objective in Eq. (1). 

> Skip-gram with negative sampling(SGNS)是Mikolov等人提出的一种 neural word embedding 方法。该方法的目的是找到能够捕捉句子中词与其周围词之间关系的词表示。在本节的其余部分中，我们将简要概述SGNS方法。
>
> 给定来自有限大小的词典 $W=\{w_i\}_{i=1}^{W}$ 的词序列 $(w_i)_{i=1}^{K}$，skip-gram 目标旨在最大化：
> $$
> \frac{1}{K} \sum_{i=1}^{K} \sum_{-c \le j \le c, j \ne 0} \ log \ p(w_{i+j} | w_i) 
> \qquad (1)
> $$
> 其中 $c$ 是上下文窗口大小(可能取决于 $w_i$ )，$p(w_j|w_i)$ 是 softmax 函数：
> $$
> p(w_j|w_i) = \frac{exp(u_j^Tv_j)}{\sum_{k \in I_w}exp(u_i^Tv_k)} \qquad(2)
> $$
> 其中 $u_i \in U(\subset \mathbb{R}^m)$ 和 $v_i \in V(\subset \mathbb{R}^m)$ 是隐藏的向量，分别对应于与词 $w_i \in W$ 的目标与上下文表示, , $I_w \triangleq \{1,\dots,|W|\}$ ， 参数 $m$ 是根据数据集的大小根据经验选择的。
>
> 由于 $\nabla p(w_j|w_i)$的计算复杂度，使用 Eq.(2)是不切实际的，这是词典大小 $W$ 的线性函数，其大小通常为 $10^5 -10^6$。
>
> NS 通过可以通过将公式 (2) 中的 softmax 函数替换为
> $$
> p(w_j|w_i) = \sigma(u_i^Tv_j)\prod_{k=1}^{N}\sigma(-u_i^Tv_k)
> $$
> 来缓解上述计算问题
>
> 其中 $\sigma(x) = 1/1+exp(-x)$，$N$ 是确定对于每个正样本要负采样的样本个数的参数。负样本 $w_i$从 unigram 分布的 $3/4$ 次幂采样而来，经验上讲，这分布明显优于 unigram 分布。
>
> 为了克服罕见词和频繁词之间的不平衡，我们提出了[8]  subsampling 过程 : 给定输入词序列，我们以概率 $p(discard|w) = 1- \sqrt{\frac{\rho}{f(w)}}$ 丢弃每个词 $w$，其中 $f(w)$ 为单词 $w$的词频，$\rho$ 为规定的阈值。研究表明，该方法可以加快学习过程，显著提高罕见词的表示精度。
>
> 最后，通过对 Eq.(1) 中的目标应用随机梯度上升来求解 $U$ 和 $V$。

## 3. ITEM2VEC – SGNS FOR ITEM SIMILARITY

In the context of CF data, the items are given as user generated sets. Note that the information about the relation between a user and a set of items is not always available. For example, we might be given a dataset of orders that a store received, without the information about the user that made the order. In other words, there are scenarios where multiple sets of items might belong to the same user, but this information is not provided. In Section 4, we present experimental results that show that our method handles these scenarios as well.

We propose to apply SGNS to item-based CF. The application of SGNS to CF data is straightforward once we realize that a sequence of words is equivalent to a set or basket of items. Therefore, from now on, we will use the terms “word” and “item” interchangeably. 

By moving from sequences to sets, the spatial / time information is lost. We choose to discard this information, since in this paper, we assume a static environment where items that share the same set are considered similar, no matter in what order / time they were generated by the user. This assumption may not hold in other scenarios, but we keep the treatment of these scenarios out of scope of this paper. 

Since we ignore the spatial information, we treat each pair of items that share the same set as a positive example. This implies a window size that is determined from the set size. Specifically, for a given set of items, the objective from Eq. (1) is modified as follows: 

Another option is to keep the objective in Eq. (1) as is, and shuffle each set of items during runtime. In our experiments we observed that both options perform the same. 

The rest of the process remains identical to the method described in Section 2. We name the described method item2vec. 

In this work, we used $u_i$ as the final representation for the $i$-th item and the affinity between a pair of items is computed by the cosine similarity. Other options are to use $v_i$ , the additive composition, $u_i + v_i$ or the concatenation ${[u_i^Tv_i^T]}^T$ . Note that the last two options sometimes produce superior representation. 

> 在 CF 数据的上下文中，item 是作为用户生成的集合(注：浏览的商品集合)给出的。请注意，关于 user 和一组 item 之间关系的信息并不总是可用的。例如，我们可能会得到一个商店所接收到的订单数据集，而不包含下订单的用户的信息。换句话说，在某些情况下，多个 item 可能属于同一个用户，但却没有提供此信息。在 第 4 节中，我们给出了实验结果，这些结果表明我们的方法也可以处理这些场景。
>
> 我们建议将 SGNS 应用于 item-based-CF。一旦我们意识到一个单词序列等同于 item 集合，SGNS 应用于 CF 数据就变得很简单了。因此，从现在起，我们将交替使用 “word” 和 “item” 这两个术语。
>
> 从sequence 到 set，空间/时间信息就丢失了。我们选择丢弃这个信息，因为在本文中，我们假设在一个静态环境中，相同集合中的 item 被认为是相似的，无论用户以什么 顺序/时间 生成它们。这个假设在其他场景中可能不成立，但是我们不讨论这些场景的处理。
>
> 由于我们忽略了空间信息，所以我们将同一集合的每一对 item 作为一个正样本。这意味着窗口大小由集合大小决定。具体来说，对于给定的一组 item ，将式(1)中的目标修改如下:
> $$
> \frac{1}{K}\sum_{i=1}^{K}\sum_{j \ne i}^{K} \ log \ p(w_j | w_i).
> $$
> 另一个选择是保持 Eq.(1) 中的目标不变，并在运行时 shuffle 每一组项目。在我们的实验中，我们观察到两种选择的效果是相同的。
>
> 该流程的其余部分与第 2 节中描述的方法相同。我们将所描述的方法命名为 item2vec。
>
> 在这项工作中，我们使用 $u_i$ 作为第 $i$ -th item 的最终表示，并通过余弦相似度来计算一对 item 之间的 affinity。其他选项是使用 $v_i$ ，加法组合 $u_i + v_i$ 或串联 ${[u_i^T \ v_i^T]}^T$ 。请注意，后两个选项有时会产生高级表示。

## 4. EXPERIMENTAL SETUP AND RESULTS

In this section, we present an empirical evaluation of the item2vec method. We provide both qualitative and quantitative results depending whether a metadata about the items exists. As a baseline item-based CF algorithm we used item-item SVD. 

> 在本节中，我们对 item2vec 方法进行了实践评估。我们同时提供定性和定量结果，这取决于是否存在关于这些 item 的元数据。作为 item-based 的 CF baseline 算法，我们使用了 item - item SVD。

#### 4.1 Datasets

We evaluate the methods on two different datasets, both private. The first dataset is user-artist data that is retrieved from the Microsoft Xbox Music service. This dataset consist of 9M events. Each event consists of a user-artist relation, which means the user played a song by the specific artist. The dataset contains 732K users and 49K distinct artists.

The second dataset contains orders of products from Microsoft Store. An order is given by a basket of items without any information about the user that made it. Therefore, the information in this dataset is weaker in the sense that we cannot bind between users and items. The dataset consist of 379K orders (that contains more than a single item) and 1706 distinct items. 

> 我们在两个不同的私有数据集上评估这些方法。 第一个数据集是从Microsoft Xbox 音乐服务检索的 user-artist 数据。 该数据集包含 9M 个事件。 每个事件都包含 user-artist 的关系，这意味着用户播放了特定艺术家的歌曲。 数据集包含 732K user 和 49K 独立 artists。
>
> 第二个数据集包含 Microsoft Store 中产品的订单。 订单是由一个商品集合给出的(订单购物车？)，不包含下订单的用户的任何信息。 因此，此数据集中的信息较弱，因为我们无法将用户和商品绑定。 该数据集由 379K 订单（包含多个项目）和 1706 个不同商品组成。

#### 4.2 Systems and parameters 

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Fig2.png)

**Fig.2: t-SNE embedding for the item vectors produced by item2vec (a) and SVD (b). The items are colored according to a web retrieved genre metadata**

We applied item2vec to both datasets. The optimization is done by stochastic gradient decent. We ran the algorithm for $20$ epochs. We set the negative sampling value to $N =15$ for both datasets. The dimension parameter $m$ was set to $100$ and $40$ for the Music and Store datasets, respectively. We further applied subsampling with $\rho$ values of $10^{-5}$ and $10^{-3}$ to the Music and Store datasets, respectively. The reason we set different parameter values is due to different sizes of the datasets. 

We compare our method to a SVD based item-item similarity system. To this end, we apply SVD to a square matrix in size of number of items, where the $(i,j)$ entry contains the number of times $(w_i,w_j)$ appears as a positive pair in the dataset. Then, we normalized each entry according to the square root of the product of its row and column sums. Finally, the latent representation is given by the rows of $US^{1/2}$ , where $S$ is a diagonal matrix that its diagonal contains the top $m$ singular values and $U$ is a matrix that contains the corresponding left singular vectors as columns. The affinity between items is computed by cosine similarity of their representations. Throughout this section we name this method “SVD”.

> 我们将 item2vec 应用于这两个数据集。采用 SGD 求解。我们运行了 $20$ 个 epochs。我们将两个数据集的 negative sampling 值设置为 $N =15$ 。对于 Music 和 Store 数据集，维度参数 $m$ 分别设置为 $100$ 和 $40$。我们进一步对 Music 和 Store 数据集分别使用 $10^{-5}$ 和 $10^{-3}$ 的 subsampling。我们之所以设置不同的参数值，是因为数据集的大小不同。
>
> 我们将我们的方法与 SVD based 的 item-item 相似性进行比较。为此，我们将 SVD 应用于一个以 item 数量为大小的方阵，其中 $(i,j)$ 条目包含 $(w_i,w_j)$ 在数据集中作为一对出现的次数。然后，我们根据行和列总和的乘积的平方根归一化每个 item 。最后，由 $US^{1/2}$ 的行给出隐性表示，其中 $S$ 是对角线包含前 $m$ 奇异值的对角矩阵，$U$ 是相应的左奇异向量为列的矩阵。item 之间的 affinity 通过它们表示的余弦相似度来计算。在本节中，我们将此方法命名为“SVD”。

#### 4.3 Experiments and results 

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Table1.png)

**TABLE 1: INCONSISTENCIES BETWEEN GENRES FROM THE WEB CATALOG AND THE ITEM2VEC BASED KNN PREDICTIONS**

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Table2.png)

**TABLE 2: A COMPARISON BETWEEN SVD AND ITEM2VEC ON GENRE CLASSIFICATION TASK FOR VARIOUS SIZES OF TOP POPULAR ARTIST SETS**

![Table3](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Table3.png)

**TABLE 3: A QUALITATIVE COMPARISON BETWEEN ITEM2VEC AND SVD FOR SELECTED ITEMS FROM THE MUSIC DATASET**

![Table4](/Users/helloword/Anmingyu/Gor-rok/Papers/Item2vec/ITEM2VEC NEURAL ITEM EMBEDDING FOR COLLABORATIVE FILTERING /Table4.png)

**TABLE 4: A QUALITATIVE COMPARISON BETWEEN ITEM2VEC AND SVD FOR SELECTED ITEMS FROM THE STORE DATASET**

The music dataset does not provide genre metadata. Therefore, for each artist we retrieved the genre metadata from the web to form a genre-artist catalog. Then we used this catalog in order to visualize the relation between the learnt representation and the genres. This is motivated by the assumption that a useful representation would cluster artists according to their genre. To this end, we generated a subset that contains the top 100 popular artists per genre for the following distinct genres: 'R&B / Soul', 'Kids', 'Classical', 'Country', 'Electronic / Dance', 'Jazz', 'Latin', 'Hip Hop', 'Reggae / Dancehall', 'Rock', 'World', 'Christian / Gospel' and 'Blues / Folk'. We applied t-SNE [11] with a cosine kernel to reduce the dimensionality of the item vectors to 2. Then, we colored each artist point according to its genre. 

Figures 2(a) and 2(b) present the 2D embedding that was produced by t-SNE, for item2vec and SVD, respectively. As we can see, item2vec provides a better clustering. We further observe that some of the relatively homogenous areas in Fig. 2(a) are contaminated with items that are colored differently. We found out that many of these cases originate in artists that were mislabeled in the web or have a mixed genre. 

Table 1 presents several examples, where the genre associated with a given artist (according to metadata that we retrieved from the web) is inaccurate or at least inconsistent with Wikipedia. Therefore, we conclude that usage based models such as item2vec may be useful for the detection of mislabeled data and even provide a suggestion for the correct label using a simple $k$ nearest neighbor (KNN) classifier. 

In order to quantify the similarity quality, we tested the genre consistency between an item and its nearest neighbors. We do that by iterating over the top $q$ popular items (for various values of $q$ ) and check whether their genre is consistent with the genres of the $k$ nearest items that surround them. This is done by a simple majority voting. We ran the same experiment for different neighborhood sizes ( $k = 6, 8, 10, 12$ and $16$) and no significant change in the results was observed.

Table 2 presents the results obtained for $k = 8$ . We observe that item2vec is consistently better than the SVD model, where the gap between the two keeps growing as $q$ increases. This might imply that item2vec produces a better representation for less popular items than the one produced by SVD, which is unsurprising since item2vec subsamples popular items and samples the negative examples according to their popularity. 

We further validate this hypothesis by applying the same ‘genre consistency’ test to a subset of $10K$ unpopular items (the last row in Table 2). We define an unpopular item in case it has less than $15$ users that played its corresponding artist. The accuracy obtained by item2vec was $68\%$, compared to $58.4\%$ by SVD. 

Qualitative comparisons between item2vec and SVD are presented in Tables $3-4$ for Music and Store datasets, respectively. The tables present seed items and their $4$ nearest neighbors (in the latent space). The main advantage of this comparison is that it enables the inspection of item similarities in higher resolutions than genres. Moreover, since the Store dataset lacks any informative tags / labels, a qualitative evaluation is inevitable. We observe that for both datasets, item2vec provides lists that are better related to the seed item than the ones that are provided by SVD. Furthermore, we see that even though the Store dataset contains weaker information, item2vec manages to infer item relations quite well. 

> 音乐数据集不提供流派的元数据。因此，我们从网上检索每个艺术家的流派元数据，形成流派艺术家目录。然后我们使用这个目录来可视化学习到的表示和流派之间的关系。**这是基于这样一种假设 : 一种有用的表示方式会根据艺术家的流派将他们聚集在一起。** 
>
> 为此，我们生成了一个子集，包含了以下不同流派中每个流派的前 100 名流行艺术家: “R&B / Soul”、“Kids”、“Classical”、“Country”、“Electronic / Dance”、“Jazz”、“Latin”、“Hip Hop”、“Reggae / Dancehall”、“Rock”、“World”、“Christian / Gospel” 和 “Blues / Folk”。我们使用 cosine kernel 的 t-SNE[11]将 item 向量的维数降为 2。然后，我们根据流派为每个艺术家点上色。
>
> 图2(a)和图2(b) 分别表示 item2vec 和 SVD 的 t-SNE 生成的 2D embedding。如我们所见，item2vec 提供了更好的 聚类结果。我们进一步观察到，图2(a)中一些相对均匀的区域被不同颜色的物品污染。我们发现很多这样的案例都是源于那些在网络上被错误标记的艺术家或者是混合类型的艺术家（注：错误标注或本身属性）。
>
> 表1 给出了几个例子，其中与给定艺术家相关的类型(根据我们从web上检索到的元数据)是不准确的，或者至少与Wikipedia不一致的。**因此，我们得出结论，基于使用的模型 (如item2vec) 可能有助于检测错误标记的数据，甚至可以使用简单的 $k$ 最近邻(KNN)分类器提供正确标记的建议。**
>
> 为了量化相似性质量，我们测试了 item 和 它的最近邻之间的流派一致性。我们通过迭代 top $q$ 的流行 item(对于不同的 $q$ 值)，并检查它们的 流派 是否与它们周围最接近的 $k$个 item的流派一致。这是通过简单多数投票来完成的。我们对不同的 $k$ ($k = 6,8,10,12 $ 和 $16$)进行了相同的实验，结果没有观察到显著的变化。
>
> 表2 给出了 $k = 8$ 时的结果。我们观察到 item2vec 始终比 SVD 模型好，在 SVD 模型中，两者之间的差距随着 $q$ 的增加而不断扩大。这可能意味着，item2vec为不太流行的 item 生成了比SVD生成的更好的表示，这并不奇怪，因为 item2vec 对流行的项目进行了 subsample，并根据它们的受欢迎程度进行了 negative sample 采样。
>
> 我们进一步验证了这一假设，同样将 “流派一致性” 测试应用于 $10K$ 不太流行的 item 子集。(表2的最后一行)。我们定义了一个不太流行的 item。如果这些 item 对应的 user-artist 对的数量少于 $15$ 个，item2vec 得到的精度为 $68\%$ ，SVD得到的精度为 $58.4\%$ 。
>
> item2vec 和 SVD 的定性比较分别出现在表格$3-4$ for Music和Store数据集中。这些表显示种子 item和它们的 $4$ 个最近邻(在向量空间中)。这种比较的主要优势在于，它能够以更高粒度(而非流派类型)来查看 item 的相似性。此外，由于 Store 数据集缺乏任何有意义的 tags/labels，定性评估是不可避免的。我们观察到，对于这两个数据集，item2vec 提供的 lists 比 SVD 提供的 lists 更好地与种子 item 相关。此外，我们看到，即使 Store 数据集的信息较弱，item2vec 也能够很好地推断出商品关系。

## 5. CONCLUSION

In this paper, we proposed item2vec - a neural embedding algorithm for item-based collaborative filtering. item2vec is based on SGNS with minor modifications. 

We present both quantitative and qualitative evaluations that demonstrate the effectiveness of item2vec when compared to a SVD-based item similarity model. We observed that item2vec produces a better representation for items than the one obtained by the baseline SVD model, where the gap between the two becomes more significant for unpopular items. We explain this by the fact that item2vec employs negative sampling together with subsampling of popular items. 

In future we plan to investigate more complex CF models such as [1, 2, 3] and compare between them and item2vec. We will further explore Bayesian variants [12] of SG for the application of item similarity. 

> 在本文中，我们提出了item2vec -  a neural embedding algorithm for item-based collaborative filtering。item2vec 基于做了少量修改的SGNS。
>
> 我们同时提供了 定量 和 定性 评估，这些评估与 SVD-based 的item相似性模型相比证明了 item2vec 的有效性。 我们观察到 item2vec 为 item产生了比 baseline SVD 模型的更好的表示，对于不那么流行的 item 两者之间的差距更加显著。 我们用 item2vec 使用 negative sampling 和对 popular item 进行subsampling 的事实来解释了这一点。
>
> 在未来，我们计划研究更复杂的CF模型，如[1,2,3]，并将其与 item2vec 进行比较。我们将进一步探索 SG 的贝叶斯变量(注：?)[12]用于item相似性。

## 6. REFERENCES

[1] Paquet, U., Koenigstein, N. (2013, May). One-class collaborative filtering with random graphs. In Proceedings of the 22nd international conference on World Wide Web (pp. 999-1008). 

[2] Koren Y, Bell R, Volinsky C. Matrix factorization techniques for recommender systems. Computer. 2009 Aug 1(8):30-7. 

[3] Salakhutdinov R, Mnih A. Bayesian probabilistic matrix factorization using Markov chain Monte Carlo. In Proceedings ICML 2008 Jul 5 (pp. 880-887). 

[4] Sarwar B, Karypis G, Konstan J, Riedl J. Item-based collaborative filtering recommendation algorithms. In Proceedings WWW 2001 Apr 1 (pp. 285-295). ACM. 

[5] Linden G, Smith B, York J. Amazon.com recommendations: Item-to-item collaborative filtering. Internet Computing, IEEE. 2003 Jan;7(1):76-80. 

[6] Collobert R, Weston J. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of ICML 2008 Jul 5 (pp. 160-167). 

[7] Mnih A, Hinton GE. A scalable hierarchical distributed language model. In Proceedings of NIPS 2009 (pp. 1081- 1088). 

[8] Mikolov T, Sutskever I, Chen K, Corrado GS, Dean J. Distributed representations of words and phrases and their compositionality. In Proceedings of NIPS 2013 (pp. 3111- 3119). 

[9] Frome A, Corrado GS, Shlens J, Bengio S, Dean J, Mikolov T. Devise: A deep visual-semantic embedding model. Proceedings of NIPS 2013 (pp. 2121-2129). 

[10] Lazaridou A, Pham NT, Baroni M. Combining language and vision with a multimodal skip-gram model. arXiv preprint arXiv:1501.02598. 2015 Jan 12. 

[11] Van der Maaten, L., & Hinton, G. Visualizing data using t-SNE. Journal of Machine Learning Research, (2008) 9(2579-2605), 85. 

[12] Barkan O. Bayesian neural word embedding. arXiv preprint arXiv: 1603.06571. 2015. 