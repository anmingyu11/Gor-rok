# Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba

## ABSTRACT

Recommender systems (RSs) have been the most important technology for increasing the business in Taobao, the largest online consumer-to-consumer (C2C) platform in China. 

There are three major challenges facing RS in Taobao: scalability, sparsity and cold start. In this paper, we present our technical solutions to address these three challenges. The methods are based on a wellknown graph embedding framework. 

We first construct an item graph from users’ behavior history, and learn the embeddings of all items in the graph. The item embeddings are employed to compute pairwise similarities between all items, which are then used in the recommendation process. To alleviate the sparsity and cold start problems, side information is incorporated into the graph embedding framework. 

We propose two aggregation methods to integrate the embeddings of items and the corresponding side information. Experimental results from offline experiments show that methods incorporating side information are superior to those that do not. Further, we describe the platform upon which the embedding methods are deployed and the workflow to process the billion-scale data in Taobao. Using A/B test, we show that the online Click-Through-Rates (CTRs) are improved comparing to the previous collaborative filtering based methods widely used in Taobao, further demonstrating the effectiveness and feasibility of our proposed methods in Taobao’s live production environment.

> 推荐系统(RSs)已经成为淘宝(中国最大的在线消费者对消费者(C2C)平台)业务增长的最重要的技术。
>
> 淘宝的RS面临三大挑战: 可扩展性、稀疏性和冷启动。在本文中，我们提出了解决这三个挑战的技术解决方案。这些方法是基于一个著名的 graph embedding 框架。
>
> 我们首先从用户的行为历史记录中构建一个以 item为节点的图，并学习所有 item 在图中的 embedding。使用 item embedding 计算所有 item之间的两两相似度，然后将其用于推荐过程中。为了缓解稀疏性和冷启动问题，将side information 整合到 graph embedding 框架中。

## 1 INTRODUCTION

Internet technology has been continuously reshaping the business landscape, and online businesses are everywhere nowadays. Alibaba, the largest provider of online business in China, makes it possible for people or companies all over the world to do business online. With one billion users, the Gross Merchandise Volume (GMV) of Alibaba in 2017 is 3,767 billion Yuan and the revenue in 2017 is 158 billion Yuan. In the famous Double-Eleven Day, the largest online shopping festival in China, in 2017, the total amount of transactions was around 168 billion Yuan. Among all kinds of online platforms in Alibaba, Taobao1 , the largest online consumer-to-consumer (C2C) platform, stands out by contributing 75% of the total traffic in Alibaba E-commerce. 

>互联网技术一直在不断改变着商业格局，如今电子商务无处不在。阿里巴巴是中国最大的电子商务供应商，为世界各地的人或公司在网上做生意提供了可能。阿里巴巴拥有10亿用户，2017年的商品交易总额(GMV)为3767亿元人民币，2017年的收入为1580亿元人民币。在著名的双十一，中国最大的网购节日，在2017年，总交易额约1680亿元。在阿里巴巴的各类网络平台中，淘宝网作为最大的C2C网络平台脱颖而出，为阿里巴巴电子商务贡献了75%的总流量。

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

To address these challenges in Taobao, we design a two-stage recommending framework in Taobao’s technology platform. The first stage is matching, and the second is ranking. In the matching stage, we generate a candidate set of similar items for each item users have interacted with, and then in the ranking stage, we train a deep neural net model, which ranks the candidate items for each user according to his or her preferences. Due to the aforementioned challenges, in both stages we have to face different unique problems. Besides, the goal of each stage is different, leading to separate technical solutions.

> RS在淘宝面临三大技术挑战:
>
> - 可扩展性 : 尽管许多现有的推荐方法在较小规模的数据集(即数百万用户和商品)上工作得很好，但它们在淘宝上更大的数据集(即10亿用户和20亿商品)上却失败了。
> - 稀疏性 : 由于 user 往往只与少量 item 进行互动，因此很难训练出准确的推荐模型，特别是对于互动次数较少的 user 或 item 。它通常被称为“稀疏性”问题。
> - 冷启动 : 在淘宝，每小时不断有数百万条新 item 被上传。这些 item 没有用户行为。处理这些 item 或预测用户对这些 item 的偏好是具有挑战性的，这就是所谓的“冷启动”问题。
>
> 为了解决淘宝的这些挑战，我们在淘宝的技术平台上设计了一个两阶段的推荐框架。第一阶段是召回(matching)，第二阶段是排序(ranking)。在matching 阶段，我们为每个 user 交互过的 item 生成一个相似 item 的候选集，然后在排序阶段，我们训练一个深度神经网络模型，该模型根据 user 的偏好对每个 user 的候选 item 进行排序。由于上述的挑战，我们在两个阶段都必须面对不同的独特问题。此外，每个阶段的目标是不同的，导致了不同的技术解决方案。

In this paper, we focus on how to address the challenges in the matching stage, where the core task is the computation of pairwise similarities between all items based on users’ behaviors. After the pairwise similarities of items are obtained, we can generate a candidate set of items for further personalization in the ranking stage. 

To achieve this, we propose to construct an item graph from users’ behavior history and then apply the state-of-art graph embedding methods [8, 15, 17] to learn the embedding of each item, dubbed Base Graph Embedding (BGE). In this way, we can generate the candidate set of items based on the similarities computed from the dot product of the embedding vectors of items. 

Note that in previous works, CF based methods are used to compute these similarities. However, CF based methods only consider the co-occurrence of items in users’ behavior history [9, 11, 16]. 

In our work, using random walk in the item graph, we can capture higher-order similarities between items. Thus, it is superior to CF based methods. However, it’s still a challenge to learn accurate embeddings of items with few or even no interactions. To alleviate this problem, we propose to use side information to enhance the embedding procedure, dubbed Graph Embedding with Side information (GES). 

For example, items belong to the same category or brand should be closer in the embedding space. In this way, we can obtain accurate embeddings of items with few or even no interactions. 

However, in Taobao, there are hundreds of types of side information, like category, brand, or price, etc., and it is intuitive that different side information should contribute differently to learning the embeddings of items. Thus, we further propose a weighting mechanism when learning the embedding with side information, dubbed Enhanced Graph Embedding with Side information (EGES).

> 本文主要研究如何解决 matching 阶段的挑战，其核心任务是根据 user 的行为计算所有 item 之间的两两相似度。在获得 item 的两两相似度后，我们可以在 ranking 阶段生成 item 候选集，用于进一步个性化。
>
> 为此，我们提出从 user 的历史行为中构建一个 item graph，然后应用目前最先进的 graph embedding 方法[8,15,17]来学习每个 item 的 embedding ，称为Base Graph Embedding(BGE)。这样，我们就可以根据 item embedding 的向量的点积计算出的相似度来生成 item 的候选集。
>
> 注意，在以前的工作中，基于CF的方法被用来计算这些相似性。然而，基于 CF 的方法只考虑用户行为历史中 item 的共现[9,11,16]。
>
> 在我们的工作中，使用 item graph 中的 random walk ，我们可以获取 item 之间的高阶相似度。因此，它优于CF-based 的方法。然而，在 item 交互很少甚至没有交互的情况下，学习如何准确的学习 item embedding 仍然是一个挑战。为了缓解这一问题，我们提出利用 side information对 embedding 过程进行改进，称为 Graph Embedding with Side Information(GES)。
>
> 例如，属于同一类别或同一品牌的物品应该在 embedding 空间中靠得更近。这样，在很少甚至没有交互作用的情况下，我们可以获得精确的 item  embedding。
>
> 而在淘宝中，有上百种类型的 side information ，如类别、品牌、价格等，不同的 side information 对 item embedding 学习的贡献是不同的，这是很直观的。因此，当学习带有 side infomation 的 embedding时，进一步提出了一种加权机制，称为 Enhanced Graph Embedding with Side information (EGES)。

In summary, there are three important parts in the matching stage: 

1. Based on years of practical experience in Taobao, we design an effective heuristic method to construct the item graph from the behavior history of one billion users in Taobao. 
2. We propose three embedding methods, BGE, GES, and EGES, to learn embeddings of two billion items in Taobao. We conduct offline experiments to demonstrate the effectiveness of GES and EGES comparing to BGE and other embedding methods.
3. To deploy the proposed methods for billion-scale users and items in Taobao, we build the graph embedding systems on the XTensorflow (XTF) platform constructed by our team. We show that the proposed framework significantly improves recommending performance on the Mobile Taobao App, while satisfying the demand of training efficiency and instant response of service even on the Double-Eleven Day.

The rest of the paper is organized as follows. In Section 2, we elaborate on the three proposed embedding methods. Offline and online experimental results are presented in Section 3. We introduce the deployment of the system in Taobao in Section 4, and review the related work in Section 5. We conclude our work in Section 6.

> 总之，在召回阶段有三个重要部分：
>
> 1. 基于多年在淘宝上的实践经验，我们设计了一种有效的启发式方法，根据淘宝上十亿 user 的历史行为来构建 item graph。
> 2. 我们提出了 BGE、GES和 EGES 三种 embedding 方法，学习20亿 item 在淘宝上的 embedding。我们进行了离线实验，以证明GES和EGES与BGE和其他 embedding 方法相比的有效性。
> 3. 为了将本文提出的方法部署到10亿规模的淘宝 user 和 item 上，我们在自己团队搭建的XTensorflow (XTF)平台上构建了 graph embedding 系统。实验结果表明，该框架显著提高了移动淘宝App上的推荐效果，同时即使在双十一当天也能满足训练效率和服务及时响应的需求。
>
> 本文的其余部分安排如下。在第 2 节中，我们详细介绍了三种建议的 embedding 方法。 第 3 部分介绍了离线和线上实验结果。第 4 部分介绍了在淘宝中的系统部署，第 5 部分介绍了相关工作。第 6 部分总结了我们的工作。

## 2 FRAMEWORK

In this section, we first introduce the basics of graph embedding, and then elaborate on how we construct the item graph from users’ behavior history. Finally, we study the proposed methods to learn the embeddings of items in Taobao.

> 在本节中，我们首先介绍 graph embedding 的基础知识，然后详细说明如何从 user 的历史行为中构建 item graph。最后，我们研究提出的淘宝中 item  embedding 的学习方法。

### 2.1 Preliminaries

In this section, we give an overview of graph embedding and one of the most popular methods, DeepWalk [15], based on which we propose our graph embedding methods in the matching stage. Given a graph $G = (V, E)$, where $V$ and $E$ represent the node set and the edge set, respectively. Graph embedding is to learn a low-dimensional representation for each node $v \in V$ in the space $R^d$ , where $d \ll |V|$. In other words, our goal is to learn a mapping function $\Phi : V \to R^d$ , i.e., representing each node in $V$ as a $d$-dimensional vector.

In [13, 14], word2vec was proposed to learn the embedding of each word in a corpus. Inspired by word2vec, Perozzi et al. proposed DeepWalk to learn the embedding of each node in a graph [15]. They first generate sequences of nodes by running random walk in the graph, and then apply the Skip-Gram algorithm to learn the representation of each node in the graph. To preserve the topological structure of the graph, they need to solve the following optimization problem:
$$
\mathop{minimize}\limits_{\Phi} \sum_{v \in V}\sum_{c \in N(v)} -logPr(c|\Phi(v)) \quad (1)
$$
where $N(v)$ is the neighborhood of node $v$ , which can be defined as nodes within one or two hops from $v$.  $Pr(c|\Phi(v))$ defines the conditional probability of having a context node $c$ given a node $v$. 

In the rest of this section, we first present how we construct the item graph from users’ behaviors, and then propose the graph embedding methods based on DeepWalk for generating low-dimensional representation for two billion items in Taobao.

> 在本节中，我们将概述 graph embedding 方法，以及目前最流行的方法之一DeepWalk[15]，并在此基础上提出了 matching 阶段的 graph embedding 方法。给定 $G = (V,E)$，其中 $V$ 和 $E$ 分别代表点集和边集。Graph Embedding 的目的是去学习节点 $v \in V$ 的低维表示。换句话说，我们的目标是学习映射函数 $\Phi : V  \to R^d$ ，即将 $V$ 中的每个节点 $v$ 表示为一个 $d$ 维向量。
>
> 在[13,14]中，word2vec被提出用来学习每个单词在语料库中的 embedding。受 word2vec 的启发，Perozzi 等人提出了 DeepWalk 来学习在一个[15] graph 中每个节点的 embedding。他们首先通过在 graph 中运行 random walk 生成节点序列，然后应用 Skip-Gram算法学习 graph 中每个节点的表示。为了保留图的拓扑结构，他们需要解决以下优化问题:
> $$
> \mathop{minimize}\limits_{\Phi} \sum_{v \in V}\sum_{c \in N(v)} -log \ Pr(c|\Phi(v)) \quad (1)
> $$
> 其中 $N(v)$ 是节点 $v$ 的邻域，可以将其定义为距离 $v$ 一或两跳内的节点。 $Pr(c | \Phi(v))$ 定义了给定节点 $v$ 的上下文节点 $c$ 的条件概率。
>
> 在本节的其余部分，我们首先介绍如何根据 user 的行为构造 item graph，然后提出基于 DeepWalk 的 Graph Embedding方法，以生成淘宝中20亿个 item 的低维表示。

### 2.2 Construction of Item Graph from Users’ Behaviors

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig2.png)

**Figure 2: Overview of graph embedding in Taobao: (a) Users’ behavior sequences: One session for user u1, two sessions for user u2 and u3; these sequences are used to construct the item graph; (b) The weighted directed item graph $G = (V, E)$; (c) The sequences generated by random walk in the item graph; (d) Embedding with Skip-Gram.**

In this section, we elaborate on the construction of the item graph from users’ behaviors. In reality, a user’s behaviors in Taobao tend to be sequential as shown in Figure 2 (a). Previous CF based methods only consider the co-occurrence of items, but ignore the sequential information, which can reflect users’ preferences more precisely. 

However, it is not possible to use the whole history of a user because 

1. The computational and space cost will be too expensive with so many entries; 
2. A user’s interests tend to drift with time. Therefore, in practice, we set a time window and only choose users’ behaviors within the window. This is called session-based users’ behaviors. Empirically, the duration of the time window is one hour.

After we obtain the session-based users’ behaviors, two items are connected by a directed edge if they occur consecutively, e.g., in Figure 2 (b) item D and item A are connected because user u1 accessed item D and A consecutively as shown in Figure 2 (a). 

> 在本节中，我们将 user 的行为详细说明 item graph 的构造。实际上，user 在淘宝上的行为往往是顺序的，如图2 (a)所示。以往 CF-based 的方法只考虑 item 的共现，忽略了顺序信息，因为顺序信息能更准确地反映 user 的偏好。
>
> 但是，不可能使用 user 的整个历史记录，因为:
>
> 1. 巨大的样本数量将造成计算成本和空间成本将过于昂贵。
> 2. 用户的兴趣会随着时间而变化。因此在实践中，我们设置了一个时间窗口，只在窗口内选择用户的行为。这称为基于session的用户行为。根据经验，时间窗口的持续时间为1小时。
>
> 在获得基于session的用户行为后，如果两个 item 在行为中是连续的，则通过有向边连接，如图2 (b)中，item D 和 item A 之所以连接，是因为用户 u1 连续访问了 item D 和 item A，如图2 (a)所示。

By utilizing the collaborative behaviors of all users in Taobao, we assign a weight to each edge $e_{ij}$ based on the total number of occurrences of the two connected items in all users’ behaviors. Specifically, the weight of the edge is equal to the frequency of item i transiting to item j in the whole users’ behavior history. In this way, the constructed item graph can represent the similarity between different items based on all of the users’ behaviors in Taobao.

In practice, before we extract users’ behavior sequences, we need to filter out invalid data and abnormal behaviors to eliminate noise for our proposed methods. Currently, the following behaviors are regarded as noise in our system:

- If the duration of the stay after a click is less than one second, the click may be unintentional and needs to be removed. 
- There are some “over-active” users in Taobao, who are actually spam users. According to our long-term observations in Taobao, if a single user bought 1,000 items or his/her total number of clicks is larger than 3,500 in less than three months, it is very likely that the user is a spam user. We need to filter out the behaviors of these users.
- Retailers in Taobao keep updating the details of a commodity. In the extreme case, a commodity can become a totally different item for the same identifier in Taobao after a long sequence of updates. Thus, we remove the item related to the identifier

> 我们利用淘宝内所有 user 的协同行为，根据两个关联 item 在所有 user 行为中出现的总数，为每条边$e_{ij}$分配一个权重。具体来说，边的权重等于 item  $i$ 在整个 user 行为历史中转换到 item $j$ 的频率。这样，所构建的 item graph 可以根据淘宝上所有 user 的行为来表示不同 item 之间的相似度。
>
> 在实际应用中，在提取 user 行为序列之前，我们需要对所提出方法中的无效数据和异常行为进行过滤以消除噪声。目前，以下行为在我们的系统中被认为是噪声:
>
> - 如果点击后停留的时间少于一秒，点击可能是无意的，需要删除。
> - 淘宝上有一些“过度活跃”的 user ，他们实际上是 spam user。根据我们在淘宝长期的观察，如果单个 user 在不到三个月的时间内购买了1000件 item 或总点击量大于3500次，则该 user 极有可能是 spam user。我们需要过滤掉这些 user 的行为。
> - 淘宝上的零售商不断更新 item 的详细信息。 在极端情况下，经过长时间的更新 , item 可能在淘宝中成为完全相同的标识符的完全不同的 item 。 因此，我们删除了与（上述）标识符所关联的 item。

### 2.3 Base Graph Embedding

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
where $ N(v_i)'$ is the negative samples for $v_i$ , and $\sigma()$ is the sigmoid function $\sigma(x) = \frac{1}{1+e^{-x}}$ . Empirically,  the larger $ |N(v_i)′|$  is, the better the obtained results.

> 在得到记为 $G = (V, E) $ 的加权有向项图后，我们采用DeepWalk来学习每个节点在 $G$ 中的嵌入。设 $M$ 表示 $G$ 和 $M_{ij}$ 的邻接矩阵，即 $i$ 到 $j$ 的边的权值。首先基于 random walk 生成节点序列，然后在序列上运行 Skip-Gram 算法。定义 random walk 的转移概率为:
> $$
> P(v_j|v_i) = \begin{cases}
> \frac{M_{ij}}{\sum_\limits{j \in N_+(v_i)}M_{ij}}, \quad v_j \in N_+(v_i) \\
> 0, \quad e_{ij} \notin \varepsilon 
> \end{cases} 
> \quad (2)
> $$
> 其中 $N_+(v_i)$ 表示一组外向临近节点（出度的指向节点）的集合，即 $v_i$ 中的边指向 $N_+(v_i)$ 中的所有节点。通过 random walk，我们可以生成一些序列，如图2 (c)所示。
>
> 然后应用 Skip-Gram 算法[13,14]来学习 embeddings ，使得到的序列中两个节点的共现概率最大化。这就有了以下优化问题：
> $$
> \mathop{minimize}\limits_{\Phi} -log \ Pr(\{ v_{i-w},\cdots,v_{i+w}\} \backslash v_i | \Phi(v_i)) \quad(3)
> $$
> 其中 $w$ 是序列中上下文节点的窗口大小。根据独立性假设，我们有
> $$
> Pr(\{v_{i-w},\cdots,v_{i+w}\} \backslash v_i|\Phi(v_i)) = \prod_{j=i-w,j \ne i}^{i+w} Pr(v_j | \Phi(v_i)) \quad (4)
> $$
> 应用 Negtive Sampling [13,14]，可将式(3)转化为
> $$
> \mathop{minimize}\limits_{\Phi}\ log\ \sigma(\Phi(v_j)^T\Phi(v_i)) + \sum\limits_{t \in N(v_i)'} log\ \sigma(-\Phi(v_t)^T\Phi(v_i)) \quad(5)
> $$
> 其中 $ N(v_i)'$ 是对于 $v_i$ 的 negative samples，$\sigma()$ 是 sigmoid 函数 $\sigma(x) = \frac{1}{1+e^{-x}}$ 。从经验上看，$ |N(v_i) ' |$ 越大，得到的结果越好。

### 2.4 Graph Embedding with Side Information

By applying the embedding method in Section 2.3, we can learn embeddings of all items in Taobao to capture higher-order similarities in users’ behavior sequences, which are ignored by previous CF-based methods. However, it is still challenging to learn accurate embeddings for “cold-start” items, i.e., those with no interactions of users.

To address the cold-start problem, we propose to enhance BGE using side information attached to cold-start items. In the context of RSs in e-commerce, side information refers to the category, shop, price, etc., of an item, which are widely used as key features in the ranking stage but rarely applied in the matching stage. We can alleviate the cold-start problem by incorporating side information in graph embedding. For example, two hoodies (same category) from UNIQLO (same shop) may look alike, and a person who likes Nikon lens may also has an interest in Canon Camera (similar category and similar brand). It means that items with similar side information should be closer in the embedding space. Based on this assumption, we propose the GES method as illustrated in Figure 3.

For the sake of clarity, we modify slightly the notations. We use $W$ to denote the embedding matrix of items or side information. Specifically,  $W^0_v$ denotes the embedding of item $v$, and $W^s_v$ denotes the embedding of the $s$-th type of side information attached to item $v$. Then, for item $v$ with $n$ types of side information, we have $n + 1$ vectors $W^0_v ,\cdots,W^n_v \in R^d$ , where $d$ is the embedding dimension. Note that the dimensions of the embeddings of items and side information are empirically set to the same value. 

As shown in Figure 3, to incorporate side information, we concatenate the $n + 1$ embedding vectors for item $v$ and add a layer with average-pooling operation to aggregate all of the embeddings related to item $v$, which is 
$$
H_v = \frac{1}{n+1}\sum_{s=0}^{n}{W^s_v} \quad (6)
$$
where $H_v$ is the aggregated embeddings of item $v$. In this way, we incorporate side information in such a way that items with similar side information will be closer in the embedding space. This results in more accurate embeddings of cold-start items and improves the offline and online performance (see Section 3).

> 通过应用2.3节中的 embedding 方法，我们可以学习淘宝中所有 item 的embedding，以捕获 user 行为序列中的高阶相似性，而先前CF-baed的方法则忽略了这些相似性。但是，学习“冷启动”(即那些没有 user 交互的 item )的准确 embedding 仍然具有挑战性。
>
> 为了解决冷启动问题，我们建议使用附加到冷启动 item 上的 side information来增强BGE。在电子商务的RSs领域中，side information是指 item 的品类、商店、价格等，这些信息在 ranking 阶段被广泛使用，但在 matching 阶段则很少使用。我们可以通过在 graph embedding 中加入 side information 来缓解冷启动问题。例如：优衣库(同一家店)的两款帽衫(同品类)可能看起来很像，喜欢尼康镜头的人可能对佳能相机(品类、品牌相似)也很感兴趣。这意味着具有相似 side information 的 item 在 embedding 空间中应该更接近。基于此假设，我们提出如图3所示的GES方法。
>
> 为了清楚起见，我们稍微修改了一下表示法。我们用 $W$ 表示 item 或 side information 的 embedding 矩阵。具体地，$W^0_v$ 表示 item $v$ 的 embedding ，$W^s_v$ 表示 item $v$ 附带的第 $s$ 类 side information 的 embedding。然后，对于 side information 类型为 $n$ 的项 $v$ ，有​ $n + 1$ 向量 $W^0_v，\cdots,W^n_v \in R^d$，其中 $d$ 是 embedding 维数。请注意，item 和 side information 的 embedding 向量维度根据经验设置为相同的值。
>
> 如图3所示，为了聚合 side information ，我们连接了 item $v$ 的 $n + 1$ 个embedding 向量，并添加了一个 average pooling 层来聚合所有与 item $v$ 相关的 embedding ，即：
> $$
> H_v = \frac{1}{n+1}\sum_{s=0}^{n}{W^s_v} \quad (6)
> $$
> 其中 $H_v$ 聚合了 item $v$ 的 embedding。通过这种方式，我们将 side information 合并到一起，这样，具有相似 side information 的 item 将在 embedding 空间中更接近。这可以更加准确地 embedding 冷启动 item，并提高离线和线上效果(参见第3节)。

### 2.5 Enhanced Graph Embedding with Side Information

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig3.png)

**Figure 3: The general framework of GES and EGES. SI denotes the side information, and “SI 0” represents the item itself. In practice, 1) Sparse features tend to be onehot-encoder vectors for items and different SIs. 2) Dense embeddings are the representation of items and the corresponding SI. 3) The hidden representation is the aggregation embedding of an item and its corresponding SI.**

Despite the performance gain of GES, a problem remains when integrating different kinds of side information in the embedding procedure. In Eq. (6), the assumption is that different kinds of side information contribute equally to the final embedding, which does not reflect the reality. For example, a user who has bought an iPhone tends to view Macbook or iPad because of the brand “Apple”, while a user may buy clothes of different brands in the same shop in Taobao for convenience and lower price. Therefore, different kinds of side information contribute differently to the co-occurrence of items in users’ behaviors.

> 尽管 GES 的效果有所提高，但在 embeding 过程中整合不同种类的 side information 时仍然存在一个问题。在Eq.(6)中，假设不同种类的 side information 对最终 embedding 的贡献是相等的，这并不能反映实际情况。例如，购买了 iPhone 的 user 会因为“Apple”这个品牌而倾向于去看Macbook或者iPad，而 user 可能因为方便且便宜而在淘宝的同一家商店中购买了不同品牌的衣服。因此，不同种类的 side information 对 user 行为中 item 共现的贡献不同。

To address this problem, we propose the EGES method to aggregate different types of side information. The framework is the same to GES (see Figure 3). The idea is that different types of side information have different contributions when their embeddings are aggregated. Hence, we propose a weighted average layer to aggregate the embeddings of the side information related to the items. Given an item $v$ , let $A \in R^{|V|×(n+1)}$ be the weight matrix and the entry $A_{ij}$ the weight of the $j$-th type of side information of the $i$-th item. Note that $A_{∗0}$, i.e., the first column of $A$, denotes the weight of item $v$ itself. 

For simplicity, we use $a^s_v$ to denote the weight of the $s$-th type of side information of item $v$ with $a^0_v$ denoting the weight of item $v$ itself. The weighted average layer combining different side information is defined in the following:
$$
H_v = \frac{\sum_{j=0}^ne^{a_v^j}W_v^j}{\sum_{j=0}^{n}e^{a_v^j}} \quad (7)
$$
where we use $e^{a^j_v}$ instead of $a^j_v$ to ensure that the contribution of each side information is greater than $0$ , and $\sum_{j=0}^ne^{a_v^j}$  is used to normalize the weights related to the embeddings of different side information.

For node $v$ and its context node $u$ in the training data, we use $Z_u \in R^d$ to represent its embedding and $y$ to denote the label. Then, the objective function of EGES becomes
$$
\mathcal{L}(v,u,y) = -[ylog(\sigma(H_v^TZ_u)) + (1-y)log(1- \sigma(H_v^TZ_u))] \qquad (8)
$$
To solve it, the gradients are derived in the following:
$$
\frac{\delta\mathcal{L}}{\delta Z_u} = (\sigma(H_v^TZ_u) -y)H_v \qquad (9)
$$
For $s$-th side information
$$
\frac{\delta\mathcal{L}}{\delta{a_v^s}} \\
= \frac{\delta\mathcal{L}}{\delta H_v}\frac{\delta\mathcal{H_v}}{\delta a_v^s}\frac{}{} 
\\
= (\sigma(H_v^TZ_u) - y)Z_u\frac{
(\sum_{j=0}^n e^{a_v^j})e^{a_v^s}W_v^s - e^{a_v^s}\sum_{j=0}^ne^{a_v^j}W_v^j
}
{
(\sum_{j=0}^n e^{a_v^j})^2
}
\\ (10)
$$

$$
\frac{\delta \mathcal{L}}{\delta W_v^s} \\
= \frac{\delta \mathcal{L}}{\delta H_v} \frac{\delta H_v}{\delta W_v^s}
\\
= \frac{e^{a^s_v}}{\sum_{j=0}^{n}e^{a_v^j}}(\sigma(H_v^TZ_u)-y)Z_u
\\
(11)
$$

The pseudo code of EGES is listed in Algorithm 1, and the pseudo code of the weighted Skip-Gram updater is shown in Algorithm 2. The final hidden representation of each item is computed by Eq. (7).

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Alg1.png)

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Alg2.png)

> 为了解决这个问题，我们提出了 EGES 方法来聚合不同类型的 side information 。该框架与GES相同(见图3)。其思想是，不同类型的 side information 在其 embedding 被聚合时具有不同的贡献。因此，我们提出一个加权平均层来聚合与 item 相关的 side information 的 embedding。给定一个 item $v$，设 $A \in R^{|V|×(n+1)}$ 为权重矩阵，$A_{ij}$ 为 $i$-th item 的 $j$-th 类型 side information的权重。注意，$A_{∗0}$，即， $A$ 的第一列，表示 item $v$ 本身的权重。
>
> 为简单起见，我们使用 $a^s_v$ 表示 item $v$ 的第 $s$ 类型 side information的权值， $a^0_v$ 表示 item $v$ 本身的权值。结合不同 side information 的加权平均层定义如下:
> $$
> H_v = \frac{\sum_{j=0}^ne^{a_v^j}W_v^j}{\sum_{j=0}^{n}e^{a_v^j}} \quad (7)
> $$
> 其中，我们用 $e^{a^j_v}$ 代替 $a^j_v$ ，以保证每个 side information 的贡献大于 $0$ ，并使用 $\sum_{j=0}^ne^{a_v^j}$ 来归一化不同的 side information embedding 的权值。
>
> 对于训练数据中的节点 $v$ 和它的上下文节点 $u$ ， 我们使用 $Z_u \in R^d$ 表示它的 embedding ，$y$ 表示标签。然后，EGES的目标函数变为：
> $$
> \mathcal{L}(v,u,y) = -[ylog(\sigma(H_v^TZ_u)) + (1-y)log(1- \sigma(H_v^TZ_u))] \qquad (8)
> $$
> 为了解决这个问题，可以通过以下方式得出梯度：
> $$
> \frac{\delta\mathcal{L}}{\delta Z_u} = (\sigma(H_v^TZ_u) -y)H_v \qquad (9)
> $$
> 对于第 $s$ 个 side information.
> $$
> \frac{\delta\mathcal{L}}{\delta{a_v^s}} \\
> = \frac{\delta\mathcal{L}}{\delta H_v}\frac{\delta\mathcal{H_v}}{\delta a_v^s}\frac{}{} 
> \\
> = (\sigma(H_v^TZ_u) - y)Z_u\frac{
> (\sum_{j=0}^n e^{a_v^j})e^{a_v^s}W_v^s - e^{a_v^s}\sum_{j=0}^ne^{a_v^j}W_v^j
> }
> {
> (\sum_{j=0}^n e^{a_v^j})^2
> }
> \\ (10)
> $$
>
> $$
> \frac{\delta \mathcal{L}}{\delta W_v^s} \\
> = \frac{\delta \mathcal{L}}{\delta H_v} \frac{\delta H_v}{\delta W_v^s}
> \\
> = \frac{e^{a^s_v}}{\sum_{j=0}^{n}e^{a_v^j}}(\sigma(H_v^TZ_u)-y)Z_u
> \\
> (11)
> $$
>
> EGES的伪代码在算法1中列出，加权的 Skip-Gram 更新器的伪代码在算法2中显示。每个项目的最终隐藏表示由等式计算。 (7)。

## 3 EXPERIMENTS 

In this section, we conduct extensive experiments to demonstrate the effectiveness of our proposed methods. First, we evaluate the methods by the link prediction task, and then report the online experimental results on Mobile Taobao App. Finally, we present some real-world cases to give insight into the proposed methods in Taobao.

**Table 1: Statistics of the two datasets. #SI denotes the number of types of side information. Sparsity is computed according to $1 - \frac{\#Edges}{\#Nodes \times (\#Nodes -1)}$.**

![截屏2020-12-25 下午4.28.22](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Table1.png)

> 在本节中，我们将进行大量的实验来证明我们所提出的方法的有效性。首先，我们通过 link prediction 任务对所提出的方法进行评估，然后在淘宝App报告线上实验结果。最后，我们给出了一些真实的案例，已深入了解淘宝提出的方法。

### 3.1 Offline Evaluation 

**Link Prediction.** The link prediction task is used in the offline experiments because it is a fundamental problem in networks. Given a network with some edges removed, the link prediction task is to predict the occurrence of links. Following similar experimental settings in [30], 1/3 of the edges are randomly chosen and removed as ground truth in the test set, and the remaining graph is taken as the training set. The same number of node pairs in the test data with no edges connecting them are randomly chosen as negative samples in the test set. To evaluate the performance of link prediction, the Area Under Curve (AUC) score is adopted as the performance metric.

**Dataset.** We use two datasets for the link prediction task. The first is Amazon Electronics2 provided by [12], denoted as Amazon. The second is extracted from Mobile Taobao App, denoted as Taobao. Both of these two datasets include different types of side information. For the Amazon dataset, the item graph is constructed from “co-purchasing” relations (denoted as also_bought in the provided data), and three types of side information are used, i.e., category, sub-category and brand. For the Taobao dataset, the item graph is constructed according to Section 2.2. Note that, for the sake of efficiency and effectiveness, twelve types of side information are used in Taobao’s live production, including retailer, brand, purchase level, age, gender, style, etc. These types of side information have been demonstrated to be useful according to years of practical experience in Taobao. The statistics of the two datasets are shown in Table 1. We can see that the sparsity of the two datasets are greater than 99%.

**Comparing Methods.** Experiments are conducted to compare four methods: BGE, LINE, GES, and EGES. LINE was proposed in [17], which captures the first-order and second-order proximity in graph embedding. We use the implementation provided by the authors3 , and run it using first-order and second-order proximity, which are denoted, respectively, as LINE(1st) and LINE(2nd). We implement the other three methods. The emdedding dimension of all the methods is set to $160$. For our BGE, GES and EGES, the length of random walk is $10$, the number of walks per node is $20$, and the context window is $5$.

**Results Analysis.** The results are shown in Table 2. We can see that GES and EGES outperform BGE, LINE(1st) and LINE(2st) in terms of AUC on both datasets. This demonstrates the effectiveness of the proposed methods. In other words, the sparsity problem is alleviated by incorporating side information. When comparing the improvements on Amazon and Taobao, we can see that the performance gain is more significant on Taobao dataset. We attribute this to the larger number of types of effective and informative side information used on Taobao dataset. When comparing GES and EGES, we can see that the performance gain on Amazon is lager than that on Taobao. It may be due to the fact that the performance on Taobao is already very good, i.e., 0.97. Thus, the improvement of EGES is not prominent. On Amazon dataset, EGES outperforms GES significantly in terms of AUC. Based on these results, we can observe that incorporating side information can be very useful for graph embedding, and the accuracy can be further improved by weighted aggregation of the embeddings of various side information.

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Table2.png)

**Table 2: AUCs of different methods on the two datasets. Percentages in the brackets are the improvements of AUC comparing to BGE.**

> **Link Prediction** :  link prediction 任务用于离线实验，因为它是网络中的一个基本问题。给定一个删除了一些边的网络，link prediction 任务就是预测链接的发生。遵循[30]中类似的实验设置, 随机选择$\frac{1}{3}$的边删除并作为测试集的正样本 , 图的剩余部分作为训练集。相同数目的没有边连接的节点对（node pairs）会被随机选中作为负样本。采用AUC作为评估 link prediction 的指标。
>
> **数据集** : 我们使用两个数据集来进行 link prediction。第一个是[12]提供的Amazon Electronics2，记作 Amazon。第二个提取自淘宝手机App，记为淘宝。这两个数据集都包含不同类型的 side information。对于Amazon数据集，item graph 是由“共同购买”关系(在提供的数据中表示为 also_bought )构建的，使用了三种类型的 side information，即 类目（category），子类目(sub-category)以及品牌(brand)。对于淘宝数据集，根据2.2节的方式构建 item graph。需要注意的是，为了提高效率和效果，淘宝的真实生产环境中使用了 零售商（retailer）, 品牌（brand）, 购买级别（purchase level）, 年代（age）, 适用性别（gender）, 风格（style）等12种 side information。根据多年在淘宝的实践经验，这些类型的 side information 已经被证明是有用的。两个数据集的统计结果如表1所示。我们可以看到，两个数据集的稀疏性都大于99%。
>
> **比较方法** :  对BGE、LINE、GES、EGES四种方法进行了对照实验。LINE在[17]中被提出，它可以捕获在 graph embedding 中的第一阶和第二阶的近似关系。我们使用作者提供的实现，使用一阶和二阶近似（LINE(1st)和LINE(2nd)）来运行它。我们实现了其他三种方法。所有方法的 emdedding 维度设置为 $160$。对于我们的BGE、GES和EGES，随机游走的长度为 $10$，每个节点的漫步次数为 $20$，上下文窗口为 $5$。
>
> **结果分析** : 结果如表2所示。我们可以看到，GES和EGES在两个数据集上的AUC都优于BGE, LINE(1)和LINE(2st)。这证明了所提方法的有效性。换句话说，通过聚合 side information，稀疏性问题得到了缓解。对比在亚马逊和淘宝上的改进，我们可以看到，在 Taobao 数据集上的效果增益更大。我们将它归功于在 Taobao 数据集上使用了更多类型的有效的、有信息量的 side information。对比 GES 和 EGES ，当比较 GES 和 EGES 时，我们可以看到，在Amazon上的效果收益比在 Taobao 上的要大。这可能归功于 Taobao 的效果已经非常好了，比如：0.97. 因而，EGES的提升不显著。在Amazon数据集上，EGES在AUC方面显著优于GES。基于这些结果，我们可以观察到加入 side information 对于  graph embedding 是非常有用的，并且通过对各种 side information 的 embedding 进行加权聚合可以进一步提高 embedding 的准确性。

### 3.2 Online A/B Test

We conduct online experiments in an A/B testing framework. The experimental goal is Click-Through-Rate (CTR) on the homepage of Mobile Taobao App. We implement the above graph embedding methods and then generate a number of similar items for each item as recommendation candidates. The final recommending results on the homepage in Taobao (see Figure 1) is generated by the ranking engine, which is implemented based on a deep neural network model. We use the same method to rank the candidate items in the experiment. As mentioned above, the quality of the similar items directly affects the recommending results. Therefore, the recommending performance, i.e., CTR, can represent the effectiveness of different methods in the matching stage. We deploy the four methods in an A/B test framework and the results of seven days in November 2017 are shown in Figure 4. Note that “Base” represents an item-based CF method which has been widely used in Taobao before graph embedding methods was deployed. It calculates the similarity between two items according to item co-occurrence and user voting weight. The similarity measurement is well-tuned and suitable for Taobao’s business.

From Figure 4, we can see that EGES and GES outperform BGE and Base consistently in terms of CTR, which demonstrates the effectiveness of the incorporation of side information in graph embedding. Further, the CTR of Base is larger than that of BGE. It means that well-tuned CF based methods can beat simple embedding method because a large number of hand-crafted heuristic strategies have been exploited in practice. On the other hand, EGES outperforms GES consistently, which aligns with the results in the offline experimental results in Section 3.1. It further demonstrates that weighted aggregation of side information is better than average aggregation.

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig4.png)

**Figure 4: Online CTRs of different methods in seven days in November 2017.**

> 我们在A/B测试框架下进行线上实验。实验的目标是淘宝移动端App主页的点击率(CTR)。我们实现了上述的 graph embedding 方法，然后为每个 item 生成多个相似的 item 作为推荐候选 item 。淘宝首页的最终推荐结果(见图1)由 ranking 引擎生成，基于深度神经网络模型实现。在实验中，我们使用相同的方法对候选 item 进行 ranking。如上所述，相似 item 的质量直接影响推荐结果。因而，推荐效果（例如：CTR）可以受 matching 阶段不同的方法而影响。我们在A/B测试框架中部署了这四种方法，2017年11月的7天测试结果如图4所示。需要注意的是，“Base”代表的是一种基于 item 的 CF 方法，在 graph embedding 方法出现之前，该方法在淘宝上得到了广泛的应用。它根据 item共现度和 user 投票权重来计算两个 item 之间的相似度。相似度量经过精心调整，适合淘宝的业务。
>
> 从图4中我们可以看到，EGES 和 GES 在 CTR 上的表现一致优于 BGE 和Base，这说明了在 graph embedding 中整合 side information的有效性。此外，Base的CTR大于BGE。这意味着，经过良好调参的CF-based方法可以战胜简单的 embedding 方法，因为在实际中会大量使用人工经验的策略。另一方面，EGES始终优于GES，这与3.1节的离线实验结果一致。进一步证明了 side information 的加权融合优于平均融合。

### 3.3 Case Study

In this section, we present some real-world cases in Taobao to illustrate the effectiveness of the proposed methods. The cases are examined in three aspects: 1) visualization of the embeddings by EGES, 2) cold start items, and 3) weights in EGES.

> 在这一节中，我们以淘宝上的一些实际案例来说明所提出的方法的有效性。本文从三个方面研究了这些案例:
>
> 1. 通过 EGES 的 embedding 的可视化
> 2. 冷启动 items 
> 3. 在 EGES 权重

#### 3.3.1 Visualization. 

In this part, we visualize the embeddings of items learned by EGES. We use the visualization tool provided by tensorflow . The results are shown in Figure 7. From Figure 7 (a), we can see that shoes of different categories are in separate clusters. Here one color represents one category of shoes, like badminton, table tennis, or football shoes. It demonstrates the effectiveness of the learned embeddings with incorporation of side information, i.e., items with similar side information should be closer in the embedding space. From Figure 7 (b), we further analyze the embeddings of three kinds of shoes: badminton, table tennis, and football. It is very interesting to observe that badminton and table tennis shoes are closer to each other while football shoes are farther in the embedding space. This can be explained by a phenomenon that people in China who like table tennis have much overlapping with those who like badminton. However, those who like football are quite different from those who like indoor sports, i.e., table tennis and badminton. In this sense, recommending badminton shoes to those who have viewed table tennis shoes is much better than recommending football shoes.

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig7.png)

**Figure 7: Visualization of the learned embeddings of a set of randomly chosen shoes. Item embeddings are projected into a 2-D plane via principal component analysis (PCA). Different colors represent different categories. Items in the same category are grouped together.**

> 在这一部分，我们可视化了 EGES学习到的 item embedding。我们使用tensorflow提供的可视化工具。结果如图7所示。从图7 (a)中，我们可以看到不同类别的鞋子在单独的簇中。在这里，一种颜色代表一种类型的鞋子，比如羽毛球鞋、乒乓球鞋或足球鞋。结果表明，聚合 side information的方法是有效的，即拥有相似 side information 的 item 在 embedding 空间中应该更接近。在图7 (b)中，我们进一步分析了羽毛球、乒乓球、足球三种鞋的嵌入情况。很有趣的是，羽毛球鞋和乒乓球鞋在嵌入空间中距离较近，而足球鞋在 embedding 空间中距离较远。这可以用一个现象来解释，在中国，喜欢乒乓球的人与喜欢羽毛球的人有很多重叠。然而，喜欢足球的人与喜欢室内运动如乒乓球和羽毛球的人有很大的不同。从这个意义上说，给看过乒乓球鞋的人推荐羽毛球鞋要比推荐足球鞋好得多。

#### 3.3.2 Cold Start Items. 

In this part, we show the quality of the embeddings of cold start items. For a newly updated item in Taobao, no embedding can be learned from the item graph, and previous CF based methods also fail in handling cold start items. Thus, we represent a cold start item with the average embeddings of its side information. Then, we retrieve the most similar items from the existing items based on the dot product of the embeddings of two items. The results are shown in Figure 5. We can see that despite the missing of users’ behaviors for the two cold start items, different side information can be utilized to learn their embeddings effectively in terms of the quality of the top similar items. In the figure, we annotate for each similar item the types of side information connected to the cold start item. We can see that the shops of the items are very informative for measuring the similarity of two items, which also aligns with the weight of each side information in the following part.

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig5.png)

**Figure 5: Similar items for cold start items. Top 4 similar items are shown. Note that “cat” means category**

> 在这一部分中，我们将展示冷启动 item 的 embedding 质量。对于淘宝上新更新的一个 item ，无法从 item graph 中学习到 embedding，之前 CF-based 的方法也无法处理冷启动 item 。因此，我们用 side information 的来表示冷启动 item 。然后，我们根据两个 item 的 embedding 点积从现有 item 中检索最相似的 item。结果如图5所示。我们可以看到，尽管两种冷启动 item 的 user 行为缺失，但可以利用不同的 side information 有效地学习它们的 embedding，从而了解排名靠前的同类 item 的质量。在图中，我们为每个类似的 item 做了注释 (连接到冷启动 item 上的 side information 的类型)。我们可以看到，items 的所属商店（shops）是用于衡量两个 items 相似度上非常重要的信息，它也会在下面部分使和每个 side information 的权重进行对齐。

#### 3.3.3 Weights in EGES. 

In this part, we visualize the weights of different types of side information for various items. Eight items in different categories are selected and the weights of all side information related to these items are extracted from the learned weight matrix A. The results are shown in Figure 6, where each row records the results of one item. Several observations are worth noting: 

1. The weight distributions of different items are different, which aligns with our assumption that different side information contribute differently to the final representation.
2. Among all the items, the weights of “Item”, representing the embeddings of the item itself, are consistently larger than those of all the other side information. It confirms the intuition that the embedding of an item itself remains to be the primary source of users’ behaviors whereas side information provides additional hints for inferring users’ behaviors.
3. Besides “Item”, the weights of “Shop” are consistently larger than those of the other side information. It aligns with users’ behaviors in Taobao, that is, users tend to purchase items in the same shop for convenience and lower price.

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig6.png)

**Figure 6: Weights for different side information of various items. Here “Item” means the embedding of an item itself.**

> 在这一部分中，我们可视化不同 item 的不同类型的 side information 的权重。选择了 8 个不同类别的 item，并从学习的权重矩阵 A 中提取与这些 item 相关的所有 side information 的权重。结果如图 6 所示，其中每一行记录一个 item 的结果。以下几点值得注意:
>
> 1. 不同 item 的权重分布是不同的，这与我们的假设一致，不同的 side information 对最终表示的贡献是不同的。
> 2. 在所有 items 中，”Item”的权重，表示了 item 自身的 embeddings ，会一直大于其它的 side information 的权重。必须承认的是，一个 item 自身的 embedding 仍然是 user 行为的主要源，其中 side information 提供了额外的 hints 来推断 user 行为。
> 3. 除了 ” Item ” 外，”Shop” 的权重会一直大于其它 side information 的权重。这与淘宝的 user 行为相一致，也就是说，user 可能出于方便或便宜等因素，趋向于购买在相同店内的 items。

## 4 SYSTEM DEPLOYMENT AND OPERATION

In this section, we introduce the implementation and deployment of the proposed graph embedding methods in Taobao. We first give a high-level introduction of the whole recommending platform powering Taobao and then elaborate on the modules relevant to our embedding methods.

In Figure 8, we show the architecture of the recommending platform in Taobao. The platform consists of two subsystems: online and offline. For the online subsystem, the main components are Taobao Personality Platform (TPP) and Ranking Service Platform (RSP). A typical workflow is illustrated in the following:

- When a user launches Mobile Taobao App, TPP extracts the user’s latest information and retrieves a candidate set of items from the offline subsystem, which is then fed to RSP. RSP ranks the candidate set of items with a fine-tuned deep neural net model and returns the ranked results to TPP.
- Users’ behaviors during their visits in Taobao are collected and saved as log data for the offline subsystem.

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/AlibabaRecommendation/Paper/Fig8.png)

**Figure 8: Architecture of the recommending platform in Taobao.**

> 在本节中，我们将介绍所提出的 graph embedding 方法在淘宝上的实现和部署。首先对整个支持淘宝的推荐平台进行了详细的介绍，然后对 embedding 方法相关的模块进行了详细的阐述。
>
> 图 8 展示了淘宝推荐平台的架构。该平台由线上和离线两个子系统组成。线上子系统主要组成部分为淘宝个性平台(TPP)和排序服务平台(RSP)。一个典型的工作流程如下所示:
>
> - 当 user 启动手机淘宝App时，TPP提取 user 的最新信息，并从离线子系统中检索候选 item 集，然后反馈给RSP。RSP使用一个 fine-tued 的深度神经网络模型对候选 item 集合进行排序，并将排序结果返回给TPP。
> - 收集 user 访问淘宝时的行为，并作为日志数据保存，供离线子系统使用。

The workflow of the offline subsystem, where graph embedding methods are implemented and deployed, is described in the following: 

- The logs including users’ behaviors are retrieved. The item graph is constructed based on the users’ behaviors. In practice, we choose the logs in the recent three months. Before generating session-based users’ behavior sequences, anti-spam processing is applied to the data. The remaining logs contains about 600 billion entries. Then, the item graph is constructed according to the method described in Section 2.2. 
- To run our graph embedding methods, two practical solutions are adopted: 1) The whole graph is split into a number of sub-graphs, which can be processed in parallel in Taobao’s Open Data Processing Service (ODPS) distributed platform. There are around 50 million nodes in each subgraph. 2) To generate the random walk sequences in the graph, we use our iteration-based distributed graph framework in ODPS. The total number of generated sequences by random walk is around 150 billion.
- To implement the proposed embedding algorithms, 100 GPUs are used in our XTF platform. On the deployed platform, with 150 billion samples, all modules in the offline subsystem, including log retrieval, anti-spam processing, item graph construction, sequence generation by random walk, embedding, item-to-item similarity computation and map generation, can be executed in less than six hours. Thus, our recommending service can respond to users’ latest behaviors in a very short time.

> 实现和部署 graph embedding 方法的离线子系统的工作流程描述如下:
>
> - 检索 user 行为日志。item graph 是根据 user 的行为构造的。实际上，我们选择的是最近三个月的日志。在生成基于会话的 user 行为序列之前，对数据进行 anit-spam 处理。剩余的日志包含大约 6000 亿个条目。然后，根据2.2节中描述的方法构建 item graph。
> - 为了运行我们的 graph embedding 方法，我们采用了两种实用的解决方案:1)将整个图分解为多个子图，这些子图可以在淘宝开放数据处理服务(ODPS)分布式平台上并行处理。每个子图中大约有5000万个节点。2)为了生成图中的 random walk 序列，我们在ODPS中使用了基于迭代的分布式图框架。random walk 生成的序列总数约为1500亿。
> - 为了实现所提出的 embedding 算法，在XTF平台上使用了100个gpu。在部署平台上，1500亿个样本，离线子系统的所有模块，包括日志检索、aniti-spam、item graph 构建、random walk、embedding、item2item embedding 以及 map生成，都可以在 6 个小时内完成。这样，我们的推荐服务可以在很短的时间内响应 user 的最近行为。

## 5 RELATED WORK 

In this section, we briefly review the related work of graph embedding, graph embedding with side information, and graph embedding for RS.

> 在本节中，我简要回顾 graph embedding 、聚合了 side information 的 graph embedding 和 RS graph embedding 的相关工作。

### 5.1 Graph Embedding

Graph Embedding algorithms have been proposed as a general network representation method. They have been applied to many real-world applications. In the past few years, there has been a lot of research in the field focusing on designing new embedding algorithms. These methods could be categorized into three broad categories: 

1. Factorization methods such as LINE [1] try to approximately factorize the adjacency matrix and preserve both first order and second proximities; 
2. Deep learning methods [3, 20, 21] enhance the model’s ability of capturing non-linearity in graph; 
3. Random walk based techniques [7, 8, 15] use random walks on graphs to obtain node representations which are extraordinary efficient and thus could be used in extremely large-scale networks. In this paper, our embedding framework is based on random walk.

> graph embedding 算法已被提出作为一种通用的网络表示方法。它们已经被应用到许多现实世界的应用中。在过去的几年里，在这一领域有很多关于设计新的 embedding 算法的研究。这些方法可以分为三大类:
>
> 1. 因式分解方法，如Line[1]，试图近似分解邻接矩阵，同时保持一阶和二阶邻近;
> 2. 深度学习方法[3,20,21]增强了模型捕捉 graph 中非线性的能力;
> 3. 基于 random walk 的技术[7,8,15]利用 graph 上的 random walk 来获得非常高效的节点表示，因此可以在超大规模网络中使用。本文的 embedding 框架是基于 random walk 的。

### 5.2 Graph Embedding with Side Information

The above graph embedding methods only use the topological structure of the network, which suffer from the sparsity and cold start problems. In recent years, a lot of works tried to incorporate side information to enhance graph embedding methods. Most works build their tasks based on the assumption that nodes with similar side information should be closer in the embedding space. To achieve this, a joint framework was proposed to optimize the embedding objective function with a classifier function [10, 19]. In [24], Xie et al. further embedded a complicated knowledge graph with the nodes in a hierarchical structure, like sub-categories, etc. Besides, textual information related to the nodes is incorporated into graph embedding [18, 23, 25, 26]. Moreover, in [4], Chang et al. proposed a deep learning framework to simultaneously deal with the text and image features for heterogeneous graph embedding. In this paper, we mainly process discrete side information related to items in Taobao, such as category, brand, price, etc., and design a hidden layer to aggregate different types of side information in the embedding framework.

> 上述 graph embedding 方法仅利用了网络的拓扑结构，存在稀疏性和冷启动问题。近年来，许多研究试图整合 side information 来增强 graph embedding。大多数工作都是基于这样的假设来完成任务的，即具有相似side information 的节点在 embedding 空间中应该更接近。为此，提出了一种联合框架，利用分类器函数优化 embedding 目标函数[10,19]。Xie等人在[24]中进一步嵌入了复杂的知识图谱，节点具有层次结构，如子类别等。并将与节点相关的文本信息纳入 graph embedding 中[18,23,25,26]。Chang等人在[4]中提出了一种深度学习框架，同时处理文本和图像特征，用于异构 graph embedding。本文主要处理与淘宝 item 相关的离散 side information，如品类、品牌、价格等，并在 embedding 框架中设计一个隐含层来聚合不同类型的 side information。

### 5.3 Graph Embedding for RS

RSs have been one of the most popular downstream tasks of graph embedding. With the representation in hand, various prediction models can be used to recommend. In [27, 29], embeddings of users and items are learned under the supervision of meta-path and meta-graphs, respectively, in heterogeneous information networks. Yu et al. [27] proposed a linear model to aggregate the embeddings for recommendation while Zhao et al. [29] proposed to apply factorization machine to the embeddings for recommendation. In [28], Zhang et al. proposed a joint embedding framework to learn the embeddings of graph, text and images, which are used for recommendation. In [30], Zhou et al. proposed graph embedding to capture asymmetric similarities for node recommendation. In this paper, our graph embedding methods are integrated in a two-stage recommending platform. Thus, the performance of the embeddings directly affects the final recommending results.

> RSs已经成为最流行的  graph embedding 下游任务之一。有了表示法，可以使用各种预测模型来推荐。在 [27 , 29] 中，在异构信息网络中，user 和 item 的 embedding 分别在 meta-path 和 meta-graphs 的监督下学习。
>
> Yu等人[27]提出了一种线性模型来聚合 embedding 用于推荐，Zhao等人[29]提出了将因式分解机应用于 embedding 用于推荐。在[28]中，Zhang等人提出了一种联合 embedding 框架，学习用于推荐的、文本和图像的 embedding 。在[30]中，Zhou等人提出了 graph embedding 以获取非对称相似性以进行 node 推荐。在本文中，我们的 graph embedding 方法集成在一个两阶段推荐平台。因此，embedding 的 效果直接影响最终的推荐结果。

## 6 CONCLUSION AND FUTURE WORK

Taobao’s billion-scale data (one billion users and two billion items) is putting tremendous stress on its RS in terms of scalability, sparsity and cold start.

In this paper, we present graph embedding based methods to address these challenges. To cope with the sparsity and cold-start problems, we propose to incorporate side information into graph embedding. 

Offline experiments are conducted to demonstrate the effectiveness of side information in improving recommending accuracy. 

Online CTRs are also reported to demonstrate the effectiveness and feasibility of our proposed methods in Taobao’s live production. 

Real-world cases are analyzed to highlight the strength of our proposed graph embedding methods in clustering related items using users’ behavior history and dealing with cold start items using side information.

Finally, to address the scalability and deployment issues of our proposed solutions in Taobao, we elaborate on the the platforms for training our graph embedding methods and the overall workflow of the recommendation platform in Taobao. For future work, we will pursue two directions. The first is to utilize attention mechanism in our graph embedding methods, which can provide more flexibility to learn the weights of different side information. The second direction is to incorporate textual information into our methods to exploit the large number of reviews attached to the items in Taobao.

> 淘宝十亿级的数据(10亿用户和20亿商品)在可扩展性、稀疏性和冷启动方面给其RS带来了巨大压力。
>
> 在本文中，我们提出了基于 graph embedding 的方法来解决这些挑战。为了解决稀疏性和冷启动问题，我们建议在 graph embedding 中加入 side information。
>
> 离线实验证明了 side information 在提高推荐准确性方面的有效性。
>
> 线上的点击率也证明了我们所提出的方法在淘宝生产环境中的有效性和可行性。
>
> 通过对真实案例的分析，突出我们提出的 graph embedding 方法在使用用户历史行为聚类相关 item 和使用 side information 处理冷启动 item 中的优势。
>
> 最后，针对我们提出的解决方案在淘宝上的可扩展性和部署问题，我们详细阐述了用于训练我们的graph embedding方法的平台以及淘宝推荐平台的整体工作流程。
>
> 在今后的工作中，我们将朝着两个方向努力。一是在graph embedding方法中引入 attention 机制，可以更灵活地学习不同 side information 的权重。第二个方向是将文本信息融入到我们的方法中，利用淘宝商品所附带的大量评论。

## 7 ACKNOWLEDGMENTS 

We would like to thank colleagues of our team - Wei Li, Qiang Liu, Yuchi Xu, Chao Li, Zhiyuan Liu, Jiaming Xu, Wen Chen and Lifeng Wang for useful discussions and supports on this work. We are grateful to our cooperative team - search engineering team. We also thank the anonymous reviewers for their valuable comments and suggestions that help improve the quality of this manuscript.

> 我们要感谢我们团队的同事-李伟，刘强，徐玉池，李超，刘志远，徐佳明，陈雯和王立峰，对这项工作进行了有益的讨论和支持。 我们感谢我们的合作团队-搜索工程团队。 我们也感谢匿名审稿人的宝贵意见和建议，这些意见和建议有助于提高该稿件的质量。