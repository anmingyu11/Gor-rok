# Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

## ABSTRACT

Recommender systems (RSs) have been the most important technology for increasing the business in Taobao, the largest online consumer-to-consumer (C2C) platform in China. 

There are three major challenges facing RS in Taobao: scalability, sparsity and cold start. In this paper, we present our technical solutions to address these three challenges. The methods are based on a wellknown graph embedding framework. 

We first construct an item graph from users’ behavior history, and learn the embeddings of all items in the graph. The item embeddings are employed to compute pairwise similarities between all items, which are then used in the recommendation process. To alleviate the sparsity and cold start problems, side information is incorporated into the graph embedding framework. 

We propose two aggregation methods to integrate the embeddings of items and the corresponding side information. Experimental results from offline experiments show that methods incorporating side information are superior to those that do not. Further, we describe the platform upon which the embedding methods are deployed and the workflow to process the billion-scale data in Taobao. Using A/B test, we show that the online Click-Through-Rates (CTRs) are improved comparing to the previous collaborative filtering based methods widely used in Taobao, further demonstrating the effectiveness and feasibility of our proposed methods in Taobao’s live production environment.



> 推荐系统(RSs)已经成为淘宝(中国最大的在线消费者对消费者(C2C)平台)业务增长的最重要的技术。
>
> 淘宝的RS面临三大挑战: 可扩展性、稀疏性和冷启动。在本文中，我们提出了解决这三个挑战的技术解决方案。这些方法是基于一个著名的GraphEmbedding框架。
>
> 我们首先从用户的行为历史记录中构建一个以item为节点的图，并学习所有item在图中的Embedding。使用item embedding计算所有item之间的两两相似度，然后将其用于推荐过程中。为了缓解稀疏性和冷启动问题，将side information 整合到 graph embedding 框架中。

## 1 INTRODUCTION

Internet technology has been continuously reshaping the business landscape, and online businesses are everywhere nowadays. Alibaba, the largest provider of online business in China, makes it possible for people or companies all over the world to do business online. With one billion users, the Gross Merchandise Volume (GMV) of Alibaba in 2017 is 3,767 billion Yuan and the revenue in 2017 is 158 billion Yuan. In the famous Double-Eleven Day, the largest online shopping festival in China, in 2017, the total amount of transactions was around 168 billion Yuan. Among all kinds of online platforms in Alibaba, Taobao1 , the largest online consumerto-consumer (C2C) platform, stands out by contributing 75% of the total traffic in Alibaba E-commerce. 

>互联网技术一直在不断改变着商业格局，如今电子商务无处不在。阿里巴巴是中国最大的电子商务供应商，它为世界各地的人或公司在网上做生意提供了可能。阿里巴巴拥有10亿用户，2017年的商品交易总额(GMV)为3767亿元人民币，2017年的收入为1580亿元人民币。在著名的双十一，中国最大的网购节日，在2017年，总交易额约1680亿元。在阿里巴巴的各类网络平台中，淘宝网作为最大的C2C网络平台脱颖而出，为阿里巴巴电子商务贡献了75%的总流量。

With one billion users and two billion items, i.e., commodities, in Taobao, the most critical problem is how to help users find the needed and interesting items quickly. To achieve this goal, recommendation, which aims at providing users with interesting items based on their preferences, becomes the key technology in Taobao. For example, the homepage on Mobile Taobao App (see Figure 1), which are generated based on users’ past behaviors with recommendation techniques, contributes 40% of the total recommending traffic. 

Furthermore, recommendation contributes the majority of both revenues and traffic in Taobao. In short, recommendation has become the vital engine of GMV and revenues of Taobao and Alibaba. Despite the success of various recommendation methods in academia and industry, e.g., collaborative filtering (CF) [9, 11, 16], content-based methods [2], and deep learning based methods [5, 6, 22], the problems facing these methods become more severe in Taobao because of the billion-scale of users and items.

> 淘宝有10亿用户和20亿item(即商品)，最关键的问题是如何帮助用户快速找到他们所需要的和感兴趣的商品。为了实现这一目标，淘宝的核心技术是推荐，即根据用户的喜好为用户提供感兴趣的商品。例如，移动淘宝App的首页(见图1)是根据用户过去的推荐行为，通过推荐技术生成的，占推荐总流量的40%。
>
> 此外，推荐是淘宝收入和流量的主要来源。总之，推荐已经成为淘宝和阿里巴巴GMV和收入的重要引擎。尽管学术界和工业界的各种推荐方法都很成功，例如协同过滤(CF)[9,11,16]、基于内容的推荐方法[2]、基于深度学习的推荐方法[5,6,22]，但由于淘宝有数十亿规模的用户和商品，这些方法面临的问题变得更加严重。

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig1.png)

**Figure 1: The areas highlighted with dashed rectangles are personalized for one billion users in Taobao. Attractive images and textual descriptions are also generated for better user experience. Note they are on Mobile Taobao App homepage, which contributes 40% of the total recommending traffic.**

There are three major technical challenges facing RS in Taobao:

- Scalability: Despite the fact that many existing recommendation approaches work well on smaller scale datasets, i.e., millions of users and items, they fail on the much larger scale dataset in Taobao, i.e., one billion users and two billion items. 
- Sparsity: Due to the fact that users tend to interact with only a small number of items, it is extremely difficult to train an accurate recommending model, especially for users or items with quite a small number of interactions. It is usually referred to as the “sparsity” problem. 
- Cold Start: In Taobao, millions of new items are continuously uploaded each hour. There are no user behaviors for these items. It is challenging to process these items or predict the preferences of users for these items, which is the so-called “cold start” problem.

To address these challenges in Taobao, we design a two-stage recommending framework in Taobao’s technology platform. The first stage is matching, and the second is ranking. In the matching stage, we generate a candidate set of similar items for each item users have interacted with, and then in the ranking stage, we train a deep neural net model, which ranks the candidate items for each user according to his or her preferences. Due to the aforementioned challenges, in both stages we have to face different unique problems. Besides, the goal of each stage is different, leading to separate technical solutions

> RS在淘宝面临三大技术挑战:
>
> - 可扩展性 : 尽管许多现有的推荐方法在较小规模的数据集(即数百万用户和商品)上工作得很好，但它们在淘宝上更大的数据集(即10亿用户和20亿商品)上却失败了。
> - 稀疏性 : 由于用户往往只与少量物品进行互动，因此很难训练出准确的推荐模型，特别是对于互动次数较少的用户或物品。它通常被称为“稀疏性”问题。
> - 冷启动 : 在淘宝，每小时不断有数百万条新商品被上传。这些item没有用户行为。处理这些item或预测用户对这些item的偏好是具有挑战性的，这就是所谓的“冷启动”问题。
>
> 为了解决淘宝的这些挑战，我们在淘宝的技术平台上设计了一个两阶段的推荐框架。第一阶段是召回，第二阶段是排序。在召回阶段，我们为每个用户交互过的物品生成一个相似物品的候选集，然后在排序阶段，我们训练一个深度神经网络模型，该模型根据用户的偏好对每个用户的候选物品进行排序。由于上述的挑战，我们在两个阶段都必须面对不同的独特问题。此外，每个阶段的目标是不同的，导致了不同的技术解决方案

In this paper, we focus on how to address the challenges in the matching stage, where the core task is the computation of pairwise similarities between all items based on users’ behaviors. After the pairwise similarities of items are obtained, we can generate a candidate set of items for further personalization in the ranking stage. 

To achieve this, we propose to construct an item graph from users’ behavior history and then apply the state-of-art graph embedding methods [8, 15, 17] to learn the embedding of each item, dubbed Base Graph Embedding (BGE). In this way, we can generate the candidate set of items based on the similarities computed from the dot product of the embedding vectors of items. 

Note that in previous works, CF based methods are used to compute these similarities. However, CF based methods only consider the co-occurrence of items in users’ behavior history [9, 11, 16]. 

In our work, using random walk in the item graph, we can capture higher-order similarities between items. Thus, it is superior to CF based methods. However, it’s still a challenge to learn accurate embeddings of items with few or even no interactions. To alleviate this problem, we propose to use side information to enhance the embedding procedure, dubbed Graph Embedding with Side information (GES). 

For example, items belong to the same category or brand should be closer in the embedding space. In this way, we can obtain accurate embeddings of items with few or even no interactions. 

However, in Taobao, there are hundreds of types of side information, like category, brand, or price, etc., and it is intuitive that different side information should contribute differently to learning the embeddings of items. Thus, we further propose a weighting mechanism when learning the embedding with side information, dubbed Enhanced Graph Embedding with Side information (EGES).

> 本文主要研究如何解决召回阶段的挑战，其核心任务是根据用户的行为计算所有item之间的两两相似度。在获得item的两两相似度后，我们可以在排序阶段生成商品候选集，用于进一步个性化。
>
> 为此，我们提出从用户的历史行为中构建一个item图，然后应用目前最先进的GraphEmbedding方法[8,15,17]来学习每个item的embedding，称为Base Graph Embedding(BGE)。这样，我们就可以根据item embedding的向量的点积计算出的相似度来生成item的候选集。
>
> 注意，在以前的工作中，基于CF的方法被用来计算这些相似性。然而，基于CF的方法只考虑用户行为历史中item的共现[9,11,16]。
>
> 在我们的工作中，使用item图中的随机游走，我们可以获取物品之间的高阶相似度。因此，它优于基于CF的方法。然而，在item只有很少甚至没有交互的情况下，学习如何准确的学习item embedding仍然是一个挑战。为了缓解这一问题，我们提出利用side information对embedding过程进行改进，称为 Graph Embedding with Side Information(GES)。
>
> 例如，属于同一类别或同一品牌的物品应该在embedding空间中靠得更近。这样，在很少甚至没有交互作用的情况下，我们可以获得精确的 item 嵌入。
>
> 而在淘宝中，有上百种类型的side information，如类别、品牌、价格等，不同的side information对物品嵌入学习的贡献是不同的，这是很直观的。因此，当学习带有side infomation 的 embedding时，进一步提出了一种加权机制，称为 Enhanced Graph Embedding with Side information (EGES)。

In summary, there are three important parts in the matching stage: 

1. Based on years of practical experience in Taobao, we design an effective heuristic method to construct the item graph from the behavior history of one billion users in Taobao. 
2. We propose three embedding methods, BGE, GES, and EGES, to learn embeddings of two billion items in Taobao. We conduct offline experiments to demonstrate the effectiveness of GES and EGES comparing to BGE and other embedding methods.
3. To deploy the proposed methods for billion-scale users and items in Taobao, we build the graph embedding systems on the XTensorflow (XTF) platform constructed by our team. We show that the proposed framework significantly improves recommending performance on the Mobile Taobao App, while satisfying the demand of training efficiency and instant response of service even on the Double-Eleven Day.

The rest of the paper is organized as follows. In Section 2, we elaborate on the three proposed embedding methods. Offline and online experimental results are presented in Section 3. We introduce the deployment of the system in Taobao in Section 4, and review the related work in Section 5. We conclude our work in Section 6.

> 总之，在召回阶段有三个重要部分：
>
> 1. 基于多年在淘宝上的实践经验，我们设计了一种有效的启发式方法，根据淘宝上十亿用户的历史行为来构建item graph。
> 2. 我们提出了BGE、GES和 EGES 三种嵌入方法，学习20亿件商品在淘宝上的嵌入。我们进行了离线实验，以证明GES和EGES与BGE和其他embedding方法相比的有效性。
> 3. 为了将本文提出的方法部署到10亿规模的淘宝用户和商品上，我们在自己团队搭建的XTensorflow (XTF)平台上构建了graph embedding系统。实验结果表明，该框架显著提高了移动淘宝App上的推荐效果，同时即使在双十一当天也能满足训练效率和服务及时响应的需求。
>
> 本文的其余部分安排如下。在第 2 节中，我们详细介绍了三种建议的嵌入方法。 第 3 部分介绍了离线和线上实验结果。第 4 部分介绍了在淘宝中的系统部署，第 5 部分介绍了相关工作。第 6 部分总结了我们的工作。

## 2 FRAMEWORK

In this section, we first introduce the basics of graph embedding, and then elaborate on how we construct the item graph from users’ behavior history. Finally, we study the proposed methods to learn the embeddings of items in Taobao.

> 在本节中，我们首先介绍graph embedding 的基础知识，然后详细说明如何从用户的历史行为中构建商品图。最后，我们研究提出的淘宝中商品embedding的学习方法。

#### 2.1 Preliminaries

In this section, we give an overview of graph embedding and one of the most popular methods, DeepWalk [15], based on which we propose our graph embedding methods in the matching stage. Given a graph $G = (V, E)$, where $V$ and $E$ represent the node set and the edge set, respectively. Graph embedding is to learn a low-dimensional representation for each node $v \in V$ in the space $R^d$ , where $d \ll |V|$. In other words, our goal is to learn a mapping function $\Phi : V \to R^d$ , i.e., representing each node in $V$ as a d-dimensional vector.

In [13, 14], word2vec was proposed to learn the embedding of each word in a corpus. Inspired by word2vec, Perozzi et al. proposed DeepWalk to learn the embedding of each node in a graph [15]. They first generate sequences of nodes by running random walk in the graph, and then apply the Skip-Gram algorithm to learn the representation of each node in the graph. To preserve the topological structure of the graph, they need to solve the following optimization problem:
$$
\mathop{minimize}\limits_{\Phi} \sum_{v \in V}\sum_{c \in N(v)} -logPr(c|\Phi(v)) \quad (1)
$$
where $N(v)$ is the neighborhood of node $v$, which can be defined as nodes within one or two hops from $v$.$Pr(c|\Phi(v))$ defines the conditional probability of having a context node c given a node v. 

In the rest of this section, we first present how we construct the item graph from users’ behaviors, and then propose the graph embedding methods based on DeepWalk for generating lowdimensional representation for two billion items in Taobao.

> 在本节中，我们将概述 Graph embedding 方法，以及目前最流行的方法之一DeepWalk[15]，并在此基础上提出了匹配阶段的图嵌入方法。给定图$G = (V,E)$，其中 $V$ 和 $E$ 分别代表点集和边集。Graph Embedding 目的是去学习空间地位表示。换句话说，我们的目标是学习映射函数 $\Phi : V  \to R^d$ ，即将 $V$ 中的每个节点表示为一个 $d$ 维向量。
>
> 在[13,14]中，word2vec被提出用来学习每个单词在语料库中的嵌入。受word2vec的启发，Perozzi 等人提出了 DeepWalk 来学习在一个[15]图中每个节点的 embedding。他们首先通过在图中运行随机游走生成节点序列，然后应用 Skip-Gram算法学习图中每个节点的表示。为了保留图的拓扑结构，他们需要解决以下优化问题:
> $$
> \mathop{minimize}\limits_{\Phi} \sum_{v \in V}\sum_{c \in N(v)} -log \ Pr(c|\Phi(v)) \quad (1)
> $$
> 其中 $N(v)$ 是节点 $v$ 的邻域，可以将其定义为距离 $v$ 一或两跳内的节点。 $Pr(c | \Phi(v))$ 定义了给定节点 $v$ 的上下文节点 $c$ 的条件概率。
>
> 在本节的其余部分，我们首先介绍如何根据用户的行为构造商品图，然后提出基于 DeepWalk 的Graph Embedding方法，以生成淘宝中20亿个商品的低维表示。

#### 2.2 Construction of Item Graph from Users’ Behaviors

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig2.png)

**Figure 2: Overview of graph embedding in Taobao: (a) Users’ behavior sequences: One session for user u1, two sessions for user u2 and u3; these sequences are used to construct the item graph; (b) The weighted directed item graph $G = (V, E)$; (c) The sequences generated by random walk in the item graph; (d) Embedding with Skip-Gram.**

In this section, we elaborate on the construction of the item graph from users’ behaviors. In reality, a user’s behaviors in Taobao tend to be sequential as shown in Figure 2 (a). Previous CF based methods only consider the co-occurrence of items, but ignore the sequential information, which can reflect users’ preferences more precisely. 

However, it is not possible to use the whole history of a user because 

1. The computational and space cost will be too expensive with so many entries; 
2. A user’s interests tend to drift with time. Therefore, in practice, we set a time window and only choose users’ behaviors within the window. This is called session-based users’ behaviors. Empirically, the duration of the time window is one hour.

After we obtain the session-based users’ behaviors, two items are connected by a directed edge if they occur consecutively, e.g., in Figure 2 (b) item D and item A are connected because user u1 accessed item D and A consecutively as shown in Figure 2 (a). 

> 在本节中，我们将用户的行为详细说明商品图的构造。实际上，用户在淘宝上的行为往往是顺序的，如图2 (a)所示。以往基于CF的方法只考虑商品的共现，忽略了顺序信息，因为顺序信息能更准确地反映用户的偏好。
>
> 但是，不可能使用用户的整个历史记录，因为:
>
> 1. 巨大的样本数量将造成计算成本和空间成本将过于昂贵
> 2. 用户的兴趣会随着时间而变化。因此在实践中，我们设置了一个时间窗口，只在窗口内选择用户的行为。这称为基于session的用户行为。根据经验，时间窗口的持续时间为1小时。
>
> 在获得基于session的用户行为后，如果两个商品连续发生，则通过有向边连接，如图2 (b)中，商品D和商品A之所以连接，是因为用户 u1 连续访问了商品D和商品A，如图2 (a)所示。

By utilizing the collaborative behaviors of all users in Taobao, we assign a weight to each edge $e_{ij}$ based on the total number of occurrences of the two connected items in all users’ behaviors. Specifically, the weight of the edge is equal to the frequency of item i transiting to item j in the whole users’ behavior history. In this way, the constructed item graph can represent the similarity between different items based on all of the users’ behaviors in Taobao.

In practice, before we extract users’ behavior sequences, we need to filter out invalid data and abnormal behaviors to eliminate noise for our proposed methods. Currently, the following behaviors are regarded as noise in our system:

- If the duration of the stay after a click is less than one second, the click may be unintentional and needs to be removed. 
- There are some “over-active” users in Taobao, who are actually spam users. According to our long-term observations in Taobao, if a single user bought 1,000 items or his/her total number of clicks is larger than 3,500 in less than three months, it is very likely that the user is a spam user. We need to filter out the behaviors of these users.
- Retailers in Taobao keep updating the details of a commodity. In the extreme case, a commodity can become a totally different item for the same identifier in Taobao after a long sequence of updates. Thus, we remove the item related to the identifier

> 我们利用淘宝内所有用户的协同行为，根据两个关联商品在所有用户行为中出现的总数，为每条边$e_{ij}$分配一个权重。具体来说，边的权重等于项目 $i$ 在整个用户行为历史中转换到项目 $j$ 的频率。这样，所构建的商品图可以根据淘宝上所有用户的行为来表示不同商品之间的相似度。
>
> 在实际应用中，在提取用户行为序列之前，我们需要对所提出方法中的无效数据和异常行为进行过滤以消除噪声。目前，以下行为在我们的系统中被认为是噪声:
>
> - 如果点击后停留的时间少于一秒，点击可能是无意的，需要删除。
> - 淘宝上有一些“过度活跃”的用户，他们实际上是垃圾用户。根据我们在淘宝长期的观察，如果单个用户在不到三个月的时间内购买了1000件商品或总点击量大于3500次，则该用户极有可能是垃圾用户。我们需要过滤掉这些用户的行为。
> - 淘宝上的零售商不断更新商品的详细信息。 在极端情况下，经过长时间的更新，商品可能在淘宝中成为完全相同的标识符的完全不同的商品。 因此，我们删除了与（上述）标识符所关联的商品。

## 2.3 Base Graph Embedding

After we obtain the weighted directed item graph, denoted as $ G = (V, E) $,  we adopt DeepWalk to learn the embedding of each node in $G$. Let $M$ denote the adjacency matrix of $G$ and $M_{ij}$ the weight of the edge from node $i$ pointing to node $j$. We first generate node sequences based on random walk and then run the Skip-Gram algorithm on the sequences. The transition probability of random walk is defined as
$$
P(v_j|v_i) = \begin{cases}
\frac{M_{ij}}{\sum_\limits{j \in N_+(v_i)}M_{ij}}, \quad v_j \in N_+(v_i) \\
0, \quad e_{ij} \notin \varepsilon 
\end{cases} 
\quad (2)
$$
where $N_+(v_i)$ represents the set of outlink neighbors, i.e. there are edges from $v_i$ pointing to all of the nodes in $N_+(v_i)$. By running random walk, we can generate a number of sequences as shown in Figure 2 (c). 

Then we apply the Skip-Gram algorithm [13, 14] to learn the embeddings,  which maximizes the co-occurrence probability of two nodes in the obtained sequences. This leads to the following optimization problem:
$$
\mathop{minimize}\limits_{\Phi} -log \ Pr(\{ v_{i-w},\cdots,v_{i+w}\} \backslash v_i | \Phi(v_i)) \quad(3)
$$
where $w$ is the window size of the context nodes in the sequences. Using the independence assumption, we have
$$
Pr(\{v_{i-w},\cdots,v_{i+w}\} \backslash v_i|\Phi(v_i)) = \prod_{j=i-w,j \ne i}^{i+w} Pr(v_j | \Phi(v_i)) \quad (4)
$$
Applying negative sampling [13, 14], Eq. (3) can be transformed into
$$
\mathop{minimize}\limits_{\Phi}\ log\ \sigma(\Phi(v_j)^T\Phi(v_i)) + \sum\limits_{t \in N(v_i)'} log\ \sigma(-\Phi(v_t)^T\Phi(v_i)) \quad(5)
$$
where $ N(v_i)'$ is the negative samples for $v_i$ , and $\sigma()$ is the sigmoid function$\sigma(x) = \frac{1}{1+e^{-x}}$. Empirically,  the larger$ |N(v_i)′|$ is, the better the obtained results.

> 在得到记为 $G = (V, E) $ 的加权有向项图后，我们采用DeepWalk来学习每个节点在 $G$ 中的嵌入。设 $M$ 表示 $G$ 和 $M_{ij}$ 的邻接矩阵，即 $i$ 到 $j$ 的边的权值。首先基于随机游走生成节点序列，然后在序列上运行Skip-Gram算法。定义随机游走的转移概率为:
> $$
> P(v_j|v_i) = \begin{cases}
> \frac{M_{ij}}{\sum_\limits{j \in N_+(v_i)}M_{ij}}, \quad v_j \in N_+(v_i) \\
> 0, \quad e_{ij} \notin \varepsilon 
> \end{cases}
> $$
> 其中$N_+(v_i)$表示一组外向临近节点（出度的指向节点）的集合，即$v_i$中的边指向$N_+(v_i)$中的所有节点。通过随机游走，我们可以生成一些序列，如图2 (c)所示。
>
> 然后应用Skip-Gram算法[13,14]来学习embeddings，使得到的序列中两个节点的共现概率最大化。这就导致了以下优化问题:
> $$
> \mathop{minimize}\limits_{\Phi} -log \ Pr(\{ v_{i-w},\cdots,v_{i+w}\} \backslash v_i | \Phi(v_i))
> $$
> 其中 $w$ 是序列中上下文节点的窗口大小。根据独立性假设，我们有
> $$
> Pr(\{v_{i-w},\cdots,v_{i+w}\} \backslash v_i|\Phi(v_i)) = \prod_{j=i-w,j \ne i}^{i+w} Pr(v_j | \Phi(v_i))
> $$
> 应用 Negtive Sampling [13,14]，可将式(3)转化为
> $$
> \mathop{minimize}\limits_{\Phi}\ log\ \sigma(\Phi(v_j)^T\Phi(v_i)) + \sum\limits_{t \in N(v_i)'} log\ \sigma(-\Phi(v_t)^T\Phi(v_i)) \quad(5)
> $$
> 

