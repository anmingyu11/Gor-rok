Homophily (i.e., "love of the same") is the tendency of individuals to associate and bond with similar others. The presence of homophily has been discovered in a vast array of network studies. More than 100 studies that have observed homophily in some form or another and they establish that similarity breeds connection. These include age, gender, class, and organizational role.Individuals in homophilic relationships share common characteristics (beliefs, values, education, etc.) that make communication and relationship formation easier. Homophily often leads to homogamy—marriage between people with similar characteristics.Homophily is a metric studied in the field of Social Network Analysis in which it is also known as Assortativity.

>Homophily（即“对同一个人的爱”）是个人趋向于与相似的他人联系并建立关系的趋势。
>
>在各种各样的网络研究中都发现了同构的存在。
>
>超过100项研究以某种形式或其他形式观察了同质性，并且他们发现相似性孕育了联系。 这些因素包括年龄，性别，阶级和组织角色。homophily关系中的人们具有共同的特征（信念，价值观，教育程度等），这使得沟通和关系的形成更加容易。
>
>Homophily经常导致同性婚姻，即具有相似特征的人之间的婚姻。同性恋是社交网络分析领域研究的一种度量标准，在其中它也被称为分类。

----------------------------------

node2vec所体现的网络的同质性和结构性在推荐系统中也是可以被很直观的解释的。**结构性相同的物品很可能是同品类、同属性、或者经常被一同购买的物品，而同质性相同的物品则是各品类的爆款、各品类的最佳凑单商品等拥有类似趋势或者结构性属性的物品**。

毫无疑问，二者在推荐系统中都是非常重要的特征表达。由于node2vec的这种灵活性，以及发掘不同特征的能力，甚至可以把不同node2vec生成的embedding融合共同输入后续深度学习网络，以保留物品的不同特征信息。

> 周游了世界才知道中国人之间的同质性，周游了中国才知道中国人之间的结构性

----------------------------------

Network Embedding算法主要有经历了以下三代的发展：

- 第一代：基于矩阵特征向量（谱聚类）
- 第二代：基于Random Walk（Deep Walk & Node2Vec）
- 第三代：基于Deep Learning（SDNE， GCN， GraphSAGE）

Node2Vec是第二代NE算法中比较经典的一种，本篇解读主要聚焦于算法的实现上，算法的深入理解敬请期待专栏的下一篇文章。

## **一、算法思想**

文章主要解决的问题是如何将图的一些拓扑特征进行学习（Representation Learning），那么自然就想到了将Node映射到低维特征空间作向量表示（Embedding）。这些向量可以作为其他任务的输入特征，即将无监督的表征学习任务单独抽出来。

Embedding的思路和DeepWalk是一个套路，即在图中进行游走采样得到多个节点序列，将节点看成word，使用NLP中的word2vec算法对节点进行embedding训练。

-----------------------

## Alias method 采样

[https://zhuanlan.zhihu.com/p/42630740](https://zhuanlan.zhihu.com/p/42630740)

之前做的论文中，有一处关键的计算涉及到对一个高维度的离散分布求数学期望，这个问题本来可以简单地$sum-over$这个概率分布的support解决，然而这个方案被reviewer质疑运行速度太慢，于是又想出一招用alias method做Monte Carlo采样算积分。用了采样以后程序变得更容易并行化，用C写了下效率大增。由于网上已经有相当多介绍alias method采样方法的文章了，其中不少还非常详细，我再写一遍也不过鹦鹉学舌，因此本文只对实现方法作简单记录。