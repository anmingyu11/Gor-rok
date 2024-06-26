# DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

## Abstract

Learning sophisticated feature interactions behind user behaviors is critical in maximizing CTR for recommender systems. Despite great progress, existing methods seem to have a strong bias towards low- or high-order interactions, or require expertise feature engineering. In this paper, we show that it is possible to derive an end-to-end learning model that emphasizes both low- and highorder feature interactions. The proposed model, DeepFM, combines the power of factorization machines for recommendation and deep learning for feature learning in a new neural network architecture. Compared to the latest Wide & Deep model from Google, DeepFM has a shared input to its “wide” and “deep” parts, with no need of feature engineering besides raw features. Comprehensive experiments are conducted to demonstrate the effectiveness and efficiency of DeepFM over the existing models for CTR prediction, on both benchmark data and commercial data.

> 学习用户行为背后复杂的特征交互对于推荐系统最大化点击率（CTR）至关重要。尽管已经取得了很大进展，但现有方法似乎对低阶或高阶交互存在强烈偏向，或者需要专业的特征工程。在本文中，我们表明可以推导出一种端到端的学习模型，该模型同时强调低阶和高阶特征交互。所提出的模型DeepFM在新的神经网络架构中结合了因子分解机在推荐方面的优势和深度学习在特征学习方面的优势。与谷歌最新的Wide & Deep模型相比，DeepFM的“wide”和“deep”部分共享输入，除了原始特征外，无需其他特征工程。我们在基准数据和商业数据上进行了全面的实验，以证明DeepFM在现有CTR预测模型上的有效性和效率。

## 1 Introduction

The prediction of click-through rate (CTR) is critical in recommender system, where the task is to estimate the probability a user will click on a recommended item. In many rec- ommender systems the goal is to maximize the number of clicks, so the items returned to a user should be ranked by estimated CTR; while in other application scenarios such as online advertising it is also important to improve revenue, so the ranking strategy can be adjusted as CTR×bid across all candidates, where “bid” is the benefit the system receives if the item is clicked by a user. In either case, it is clear that the key is in estimating CTR correctly.

It is important for CTR prediction to learn implicit feature interactions behind user click behaviors. By our study in a mainstream apps market, we found that people often download apps for food delivery at meal-time, suggesting that the (order-2) interaction between app category and time-stamp can be used as a signal for CTR. As a second observation, male teenagers like shooting games and RPG games, which means that the (order-3) interaction of app category, user gen- der and age is another signal for CTR. In general, such inter- actions of features behind user click behaviors can be highly sophisticated, where both low- and high-order feature interac- tions should play important roles. According to the insights of the Wide & Deep model [Cheng *et al.*, 2016] from google, considering low- and high-order feature interactions simultaneously brings additional improvement over the cases of con- sidering either alone.

The key challenge is in effectively modeling feature inter- actions. Some feature interactions can be easily understood, thus can be designed by experts (like the instances above). However, most other feature interactions are hidden in data and difficult to identify a priori (for instance, the classic as- sociation rule “diaper and beer” is mined from data, instead of discovering by experts), which can only be captured *auto- matically* by machine learning. Even for easy-to-understand interactions, it seems unlikely for experts to model them ex- haustively, especially when the number of features is large.

Despite their simplicity, generalized linear models, such as *FTRL* [McMahan *et al.*, 2013], have shown decent perfor- mance in practice. However, a linear model lacks the abil- ity to learn feature interactions, and a common practice is to manually include pairwise feature interactions in its fea- ture vector. Such a method is hard to generalize to model high-order feature interactions or those never or rarely appear in the training data [Rendle, 2010]. Factorization Machines *(FM)* [Rendle, 2010] model pairwise feature interactions as inner product of latent vectors between features and show very promising results. While in principle FM can model high-order feature interaction, in practice usually only order- 2 feature interactions are considered due to high complexity.

> 点击率（CTR）预测在推荐系统中至关重要，该任务的目标是估计用户点击推荐项目的概率。在许多推荐系统中，目标是最大化点击次数，因此应根据估计的CTR对用户返回的项目进行排名；而在其他应用场景中，如在线广告，提高收入也很重要，因此可以根据所有候选广告中的CTR×出价来调整排名策略，其中“出价”是指如果用户点击了该广告，系统所获得的收益。在任何情况下，很明显，关键在于正确估计CTR。
>
> 对于CTR预测来说，学习用户点击行为背后的隐式特征交互非常重要。通过我们对主流应用市场的研究，我们发现人们经常在用餐时间下载送餐应用，这表明应用类别和时间戳之间的（二阶）交互可以作为CTR的信号。第二个观察结果是，十几岁的男性喜欢射击游戏和角色扮演游戏，这意味着应用类别、用户性别和年龄之间的（三阶）交互是CTR的另一个信号。总的来说，用户点击行为背后的这种特征交互可能非常复杂，其中低阶和高阶特征交互都应该发挥重要作用。根据谷歌的Wide & Deep模型[Cheng等人，2016]的见解，同时考虑低阶和高阶特征交互比单独考虑任何一种情况都会带来额外的改进。
>
> 主要的挑战在于如何有效地对特征交互进行建模。一些特征交互很容易理解，因此可以由专家设计（如上面的实例）。然而，大多数其他特征交互都隐藏在数据中，很难事先识别（例如，经典的关联规则“尿布和啤酒”是从数据中挖掘出来的，而不是由专家发现的），这只能通过机器学习来自动捕获。即使对于易于理解的交互，专家似乎也不太可能对它们进行详尽的建模，特别是当特征数量很大时。
>
> 尽管广义线性模型（如FTRL[McMahan等人，2013]）很简单，但在实践中已经显示出不错的性能。然而，线性模型缺乏学习特征交互的能力，一种常见的做法是在其特征向量中手动包含成对特征交互。这种方法很难推广到对高阶特征交互或训练数据中从未或很少出现的特征交互进行建模[Rendle，2010]。因子分解机（FM）[Rendle，2010]将成对特征交互建模为特征之间潜在向量的内积，并显示出非常有希望的结果。虽然原则上FM可以对高阶特征交互进行建模，但由于复杂性较高，实践中通常只考虑二阶特征交互。

As a powerful approach to learning feature representa- tion, deep neural networks have the potential to learn so- phisticated feature interactions. Some ideas extend CNN and RNN for CTR predition [Liu *et al.*, 2015; Zhang *et al.*, 2014], but CNN-based models are biased to the in- teractions between neighboring features while RNN-based models are more suitable for click data with sequential de- pendency. [Zhang *et al.*, 2016] studies feature representa- tions and proposes *Factorization-machine supported Neural Network (FNN)*. This model pre-trains FM before applying DNN, thus limited by the capability of FM. Feature interac- tion is studied in [Qu *et al.*, 2016], by introducing a prod- uct layer between embedding layer and fully-connected layer, and proposing the *Product-based Neural Network* (*PNN*). As noted in [Cheng *et al.*, 2016], PNN and FNN, like other deep models, capture little low-order feature interactions, which are also essential for CTR prediction. To model both low- and high-order feature interactions, [Cheng *et al.*, 2016] pro- poses an interesting hybrid network structure (*Wide & Deep*) that combines a linear (“wide”) model and a deep model. In this model, two different inputs are required for the “wide part” and “deep part”, respectively, and the input of “wide part” still relies on expertise feature engineering.

One can see that existing models are biased to low- or high- order feature interaction, or rely on feature engineering. In this paper, we show it is possible to derive a learning model that is able to learn feature interactions of all orders in an end- to-end manner, without any feature engineering besides raw features. Our main contributions are summarized as follows:

- We propose a new neural network model DeepFM (Figure 1) that integrates the architectures of FM and deep neural networks (DNN). It models low-order fea- ture interactions like FM and models high-order fea- ture interactions like DNN. Unlike the wide & deep model [Cheng *et al.*, 2016], DeepFM can be trained end- to-end without any feature engineering.
- DeepFM can be trained efficiently because its wide part and deep part, unlike [Cheng *et al.*, 2016], share the same input and also the embedding vector. In [Cheng *et al.*, 2016], the input vector can be of huge size as it in- cludes manually designed pairwise feature interactions in the input vector of its wide part, which also greatly increases its complexity.
- We evaluate DeepFM on both benchmark data and com- mercial data, which shows consistent improvement over existing models for CTR prediction.

> 深度神经网络作为一种强大的学习特征表示方法，具有学习复杂特征交互的潜力。有些方法扩展了卷积神经网络（CNN）和循环神经网络（RNN）用于CTR预测[Liu等人，2015；Zhang等人，2014]，但基于CNN的模型偏向于邻近特征之间的交互，而基于RNN的模型更适合具有顺序依赖关系的点击数据。[Zhang等人，2016]研究了特征表示，并提出了“因子分解机支持的神经网络（FNN）”。该模型在应用深度神经网络（DNN）之前对因子分解机（FM）进行预训练，因此受到FM能力的限制。[Qu等人，2016]研究了特征交互，通过在嵌入层和全连接层之间引入一个乘积层，提出了“基于乘积的神经网络（PNN）”。如[Cheng等人，2016]所指出的，PNN和FNN与其他深度模型一样，很少捕获低阶特征交互，而这些交互对于CTR预测也至关重要。为了对低阶和高阶特征交互进行建模，[Cheng等人，2016]提出了一种有趣的混合网络结构（“Wide & Deep”），该结构结合了线性（“wide”）模型和深度模型。在这个模型中，“wide部分”和“deep部分”分别需要两个不同的输入，而“wide部分”的输入仍然依赖于专业的特征工程。
>
> 可以看出，现有模型偏向于低阶或高阶特征交互，或者依赖于特征工程。在本文中，我们证明了可以推导出一种学习模型，该模型能够以端到端的方式学习所有阶数的特征交互，除了原始特征外，不需要任何特征工程。我们的主要贡献总结如下：
>
> - 我们提出了一种新的神经网络模型DeepFM（图1），该模型集成了因子分解机（FM）和深度神经网络（DNN）的结构。它像FM一样对低阶特征交互进行建模，像DNN一样对高阶特征交互进行建模。与Wide & Deep模型[Cheng等人，2016]不同，DeepFM可以进行端到端的训练，无需任何特征工程。
>
> - DeepFM可以高效地进行训练，因为其“wide部分”和“deep部分”与[Cheng等人，2016]不同，它们共享相同的输入和嵌入向量。在[Cheng等人，2016]中，输入向量可能非常大，因为它在其“wide部分”的输入向量中包含了手动设计的成对特征交互，这也大大增加了其复杂性。
>
> - 我们在基准数据和商业数据上评估了DeepFM，结果显示与现有的CTR预测模型相比，其性能有了一致的提升。

## 2 Our Approach

Suppose the data set for training consists of n instances $
(\chi, y)
$, where $\chi$ is an $m$-fields data record usually involving a pair of user and item, and $y \in {0, 1}$ is the associated label indicating user click behaviors ($y = 1$ means the user clicked the item, and $y = 0$ otherwise). $\chi$ may include categorical fields (e.g., gender, location) and continuous fields (e.g., age). Each categorical field is represented as a vec- tor of one-hot encoding, and each continuous field is repre- sented as the value itself, or a vector of one-hot encoding af- ter discretization. Then, each instance is converted to $(x, y)$ where $x = [x_{field_1},x_{field_2},...,x_{filed_j},...,x_{field_m}]$ is a $d$- dimensional vector, with $x_{field_j}$ being the vector representa- tion of the $j$-th field of $\chi$. Normally, $x$ is high-dimensional and extremely sparse. The task of CTR prediction is to build a prediction model $\hat{y} = CTR\_model(x)$ to estimate the prob- ability of a user clicking a specific app in a given context.

> 假设训练数据集由 n 个实例 $(\chi, y)$ 组成，其中 $\chi$ 是一个涉及用户和项目对的 $m$ 字段数据记录，而 $y \in {0, 1}$ 是表示用户点击行为的关联标签（$y = 1$表示用户点击了该项目，否则 $y = 0$）。$\chi$ 可能包含分类字段（例如，性别、位置）和连续字段（例如，年龄）。每个分类字段都表示为一个独热编码向量，而每个连续字段则表示为其值本身，或离散化后的独热编码向量。然后，每个实例都转换为 $(x, y)$，其中 $x = [x_{field_1},x_{field_2},...,x_{field_j},...,x_{field_m}]$ 是一个 $d$ 维向量，$x_{field_j}$是$\chi$的第$j$个字段的向量表示。通常，$x$是高维且极其稀疏的。CTR预测的任务是建立预测模型$\hat{y} = CTR\_model(x)$，以估计在给定上下文中用户点击特定应用程序的概率。

#### 2.1 DeepFM

We aim to learn both low- and high-order feature interactions. To this end, we propose a Factorization-Machine based neural network (DeepFM). As depicted in Figure 11, DeepFM consists of two components, *FM component* and *deep component*, that share the same input. For feature $i$, a scalar $w_i$ is used to weigh its order-1 importance, a latent vector $V_i$ is used to measure its impact of interactions with other features. $V_i$ is fed in FM component to model order-2 feature interactions, and fed in deep component to model high-order feature interactions. All parameters, including $w_i$ , $V_i$ , and the net- work parameters ($W^{(l)} , b^{(l)}$ below) are trained jointly for the combined prediction model:
$$
\hat{y}=\operatorname{sigmoid}\left(y_{F M}+y_{D N N}\right),
$$
where $\hat{y} \in (0, 1)$ is the predicted CTR, $y_{FM}$ is the output of FM component, and $y_{DNN}$​ is the output of deep component.

**FM Component**

The FM component is a factorization machine, which is proposed in [Rendle, 2010] to learn feature interactions for recommendation. Besides a linear (order-1) interactions among features, FM models pairwise (order-2) feature interactions as inner product of respective feature latent vectors.

It can capture order-2 feature interactions much more effectively than previous approaches especially when the dataset is sparse. In previous approaches, the parameter of an interaction of features $i$ and $j$ can be trained only when feature $i$ and feature $j$ both appear in the same data record. While in FM, it is measured via the inner product of their latent vectors $V_i$ and $V_j$ . Thanks to this flexible design, FM can train latent vector $V_i$ ($V_j$) whenever $i$ (or $j$) appears in a data record. Therefore, feature interactions, which are never or rarely appeared in the training data, are better learnt by FM.

As Figure 2 shows, the output of FM is the summation of an Addition unit and a number of Inner Product units:
$$
y_{F M}=\langle w, x\rangle+\sum_{i=1}^d \sum_{j=i+1}^d\left\langle V_i, V_j\right\rangle x_i \cdot x_j
$$
where $w \in R^d$ and $V_i \in R^k \text{(k is given)}$. The Addition unit ($⟨w, x⟩$) reflects the importance of order-1 features, and the Inner Product units represent the impact of order-2 feature interactions.

**Deep Component**

The deep component is a feed-forward neural network, which is used to learn high-order feature interactions. As shown in Figure 3, a data record (a vector) is fed into the neu- ral network. Compared to neural networks with image [He *et al.*, 2016] or audio [Boulanger-Lewandowski *et al.*, 2013] data as input, which is purely continuous and dense, the in- put of CTR prediction is quite different, which requires a new network architecture design. Specifically, the raw feature input vector for CTR prediction is usually highly sparse3, super high-dimensional4, categorical-continuous-mixed, and grouped in fields (e.g., gender, location, age). This suggests an embedding layer to compress the input vector to a low- dimensional, dense real-value vector before further feeding into the first hidden layer, otherwise the network can be over- whelming to train.

Figure 4 highlights the sub-network structure from the in- put layer to the embedding layer. We would like to point out the two interesting features of this network structure: 1) while the lengths of different input field vectors can be different, their embeddings are of the same size ($k$); 2) the latent fea- ture vectors ($V$ ) in FM now serve as network weights which are learned and used to compress the input field vectors to the embedding vectors. In [Zhang *et al.*, 2016], $V$ is pre-trained by FM and used as initialization. In this work, rather than using the latent feature vectors of FM to initialize the networks as in [Zhang *et al.*, 2016], we include the FM model as part of our overall learning architecture, in addition to the other DNN model. As such, we eliminate the need of pre-training by FM and instead jointly train the overall network in an end-to-end manner. Denote the output of the embedding layer as:
$$
a^{(0)}=\left[e_1, e_2, \ldots, e_m\right]
$$
where $e_i$ is the embedding of $i$-th field and $m$ is the number of fields. Then, $a(0)$ is fed into the deep neural network, and the forward process is:
$$
a^{(l+1)}=\sigma\left(W^{(l)} a^{(l)}+b^{(l)}\right)
$$
where $l$ is the layer depth and $σ$ is an activation function. $a(l)$, $W(l)$, $b(l)$ are the output, model weight, and bias of the $l$-th layer. After that, a dense real-value feature vector is gener- ated, which is finally fed into the sigmoid function for CTR prediction: $y_{DNN} = W^{|H|+1} · a^{|H|} + b^{|H|}+1$, where $|H|$ is the number of hidden layers.

It is worth pointing out that FM component and deep com- ponent share the same feature embedding, which brings two important benefits: 1) it learns both low- and high-order fea- ture interactions from raw features; 2) there is no need for ex- pertise feature engineering of the input, as required in Wide & Deep [Cheng *et al.*, 2016].

> **论文翻译：**
>
> 我们的目标是学习低阶和高阶特征交互。为此，我们提出了一种基于因子分解机的神经网络（DeepFM）。如图11所示，DeepFM由两部分组成，即*FM组件*和*深度组件*，它们共享相同的输入。对于特征$i$，一个标量$w_i$用于衡量其一阶重要性，一个潜在向量$V_i$用于衡量它与其他特征的交互影响。$V_i$被输入到FM组件中以模拟二阶特征交互，并被输入到深度组件中以模拟高阶特征交互。所有参数，包括$w_i$，$V_i$，以及网络参数（下面的$W^{(l)} , b^{(l)}$）都联合训练用于组合预测模型：
>
> $$
> \hat{y}=\operatorname{sigmoid}\left(y_{F M}+y_{D N N}\right),
> $$
>
> 其中$\hat{y} \in (0, 1)$是预测的CTR，$y_{FM}$是FM组件的输出，$y_{DNN}$是深度组件的输出。
>
> **FM组件**
>
> FM组件是一个因子分解机，由[Rendle, 2010]提出，用于学习推荐的特征交互。除了特征之间的一阶（线性）交互之外，FM还将二阶（成对）特征交互建模为各自特征潜在向量的内积。
>
> FM比以往的方法更有效地捕捉二阶特征交互，特别是在数据集稀疏时。在以前的方法中，只有特征$i$和特征$j$都出现在同一条数据记录中时，才能训练特征$i$和$j$的交互参数。而在FM中，它是通过它们的潜在向量$V_i$和$V_j$的内积来衡量的。由于这种灵活的设计，每当$i$（或$j$）出现在数据记录中时，FM就可以训练潜在向量$V_i$（$V_j$）。因此，FM可以更好地学习在训练数据中从未或很少出现的特征交互。
>
> 如图2所示，FM的输出是一个加法单元和多个内积单元的总和：
>
> $$
> y_{F M}=\langle w, x\rangle+\sum_{i=1}^d \sum_{j=i+1}^d\left\langle V_i, V_j\right\rangle x_i \cdot x_j
> $$
>
> 其中$w \in R^d$和$V_i \in R^k \text{(k 是给定的)}$。加法单元（$⟨w, x⟩$）反映了一阶特征的重要性，而内积单元代表了二阶特征交互的影响。
>
> **深度组件**
>
> 深度组件是一个前馈神经网络，用于学习高阶特征交互。如图3所示，一个数据记录（向量）被输入到神经网络中。与以图像[He *et al.*, 2016]或音频[Boulanger-Lewandowski *et al.*, 2013]数据为输入的神经网络相比，这些输入是纯连续和密集的，CTR预测的输入却截然不同，这要求设计一种新的网络架构。具体来说，CTR预测的原始特征输入向量通常是高度稀疏的、超高维的、类别-连续混合的，并按字段分组（例如，性别、位置、年龄）。这表明在进一步输入到第一个隐藏层之前，需要一个嵌入层将输入向量压缩为低维、密集的实值向量，否则网络可能会难以训练。
>
> 图4突出了从输入层到嵌入层的子网结构。我们想指出这种网络结构的两个有趣特征：1）虽然不同输入字段向量的长度可以不同，但它们的嵌入向量大小相同（$k$）；2）FM中的潜在特征向量（$V$）现在作为网络权重，用于学习和压缩输入字段向量到嵌入向量。在[Zhang *et al.*, 2016]中，$V$通过FM进行预训练并用作初始化。在这项工作中，我们并没有像[Zhang *et al.*, 2016]那样使用FM的潜在特征向量来初始化网络，而是将FM模型作为我们整体学习架构的一部分，除了其他DNN模型之外。因此，我们消除了对FM预训练的需求，而是以端到端的方式联合训练整个网络。表示嵌入层的输出为：
>
> $$
> a^{(0)}=\left[e_1, e_2, \ldots, e_m\right]
> $$
>
> 其中$e_i$是第$i$个字段的嵌入，$m$是字段的数量。然后，$a(0)$被输入到深度神经网络中，前向过程是：
>
> $$
> a^{(l+1)}=\sigma\left(W^{(l)} a^{(l)}+b^{(l)}\right)
> $$
>
> 其中$l$是层深度，$σ$是激活函数。$a(l)$，$W(l)$，$b(l)$是第$l$层的输出、模型权重和偏置。之后，生成一个密集的实值特征向量，最终被输入到sigmoid函数中进行CTR预测：$y_{DNN} = W^{|H|+1} · a^{|H|} + b^{|H|}+1$，其中$|H|$是隐藏层的数量。
>
> 值得注意的是，FM组件和深度组件共享相同的特征嵌入，这带来了两个重要好处：1）它直接从原始特征中学习低阶和高阶特征交互；2）无需进行Wide & Deep [Cheng *et al.*, 2016]中所需的输入专家特征工程。
>

![Figure1](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure1.png)

![Figure2](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure2.png)

![Figure3](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure3.png)

![Figure4](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure4.png)

#### 2.2 Relationship with Other Neural Networks

Inspired by the enormous success of deep learning in var- ious applications, several deep models for CTR prediction are developed recently. This section compares the proposed DeepFM with existing deep models for CTR prediction.

**FNN**

As Figure 5 (left) shows, FNN is a FM-initialized feed- forward neural network [Zhang *et al.*, 2016]. The FM pre- training strategy results in two limitations: 1) the embedding parameters might be over affected by FM; 2) the efficiency is reduced by the overhead introduced by the pre-training stage. In addition, FNN captures only high-order feature interac- tions. In contrast, DeepFM needs no pre-training and learns both high- and low-order feature interactions.

**PNN**

For the purpose of capturing high-order feature interactions, PNN imposes a product layer between the embedding layer and the first hidden layer [Qu *et al.*, 2016]. According todifferent types of product operation, there are three variants: IPNN, OPNN, and PNN∗, where IPNN is based on inner product of vectors, OPNN is based on outer product, and PNN∗ is based on both inner and outer products. Like FNN, all PNNs ignore low-order feature interactions.

**Wide & Deep**

Wide & Deep (Figure 5 (right)) is proposed by Google to model low- and high-order feature interactions simultane- ously. As shown in [Cheng *et al.*, 2016], there is a need for ex- pertise feature engineering on the input to the “wide” part (for instance, cross-product of users’ install apps and impression apps in app recommendation). In contrast, DeepFM needs no such expertise knowledge to handle the input by learning directly from the input raw features.

A straightforward extension to this model is replacing LR by FM (we also evaluate this extension in Section 3). This extension is similar to DeepFM, but DeepFM shares the fea- ture embedding between the FM and deep component. The sharing strategy of feature embedding influences (in back- propagate manner) the feature representation by both low- and high-order feature interactions, which models the repre- sentation more precisely.

**Summarizations**

To summarize, the relationship between DeepFM and the other deep models in four aspects is presented in Table 1. As can be seen, DeepFM is the only model that requires no pretraining and no feature engineering, and captures both low- and high-order feature interactions.

> 受深度学习在各种应用中巨大成功的启发，最近开发了几种用于CTR预测的深度学习模型。本节将提出的DeepFM与现有的用于CTR预测的深度学习模型进行比较。
>
> **FNN**
>
> 如图5（左）所示，FNN是一个FM初始化的前馈神经网络[Zhang等人，2016]。FM预训练策略导致两个限制：1）嵌入参数可能过度受FM影响；2）预训练阶段引入的开销降低了效率。此外，FNN仅捕获高阶特征交互。相比之下，DeepFM无需预训练，并学习高阶和低阶特征交互。
>
> **PNN**
>
> 为了捕获高阶特征交互，PNN在嵌入层和第一个隐藏层之间强加了一个乘积层[Qu等人，2016]。根据不同的乘积操作类型，有三种变体：IPNN、OPNN和PNN∗，其中IPNN基于向量的内积，OPNN基于外积，PNN∗基于内积和外积。与FNN一样，所有PNN都忽略了低阶特征交互。
>
> **Wide & Deep**
>
> Wide & Deep（图5（右））由Google提出，用于同时建模低阶和高阶特征交互。如[Cheng等人，2016]所示，需要对“wide”部分的输入进行专家特征工程（例如，在应用程序推荐中，用户安装的应用程序和印象应用程序的交叉乘积）。相比之下，DeepFM无需此类专业知识来处理输入，而是直接从输入原始特征中学习。
>
> 对该模型的一个直接扩展是用FM替换LR（我们也在第3节中评估了这一扩展）。这个扩展类似于DeepFM，但DeepFM在FM和深度组件之间共享特征嵌入。特征嵌入的共享策略通过低阶和高阶特征交互以反向传播的方式影响特征表示，从而更精确地建模表示。
>
> **总结**
>
> 总结来说，DeepFM与其他深度学习模型在四个方面的关系如表1所示。可以看出，DeepFM是唯一不需要预训练和特征工程，并捕获低阶和高阶特征交互的模型。

![Figure5](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure5.png)

![Table1](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Table1.png)

## 3 Experiments

In this section, we compare our proposed DeepFM and the other state-of-the-art models empirically. The evaluation re- sult indicates that our proposed DeepFM is more effective than any other state-of-the-art model and the efficiency of DeepFM is comparable to the best ones among all the deep models.

> 在本节中，我们通过实验比较了我们提出的DeepFM与其他最先进模型。评估结果表明，我们提出的DeepFM比其他任何最先进模型都更有效，并且DeepFM的效率在所有深度模型中可与最佳模型相媲美。

#### 3.1 Experiment Setup

**Datasets**

We evaluate the effectiveness and efficiency of our proposed DeepFM on the following two datasets.

1) Criteo Dataset: Criteo dataset 5 includes 45 million users’ click records. There are 13 continuous features and 26 cate- gorical ones. We split the dataset into two parts: 90% is for training, while the rest 10% is for testing.
2) Company∗ Dataset: In order to verify the performance of DeepFM in real industrial CTR prediction, we conduct exper- iment on Company∗ dataset. We collect 7 consecutive days of users’ click records from the game center of the Company∗ App Store for training, and the next 1 day for testing. There are around 1 billion records in the whole collected dataset. In this dataset, there are app features (e.g., identification, cat- egory, and etc), user features (e.g., user’s downloaded apps, and etc), and context features (e.g., operation time, and etc).

**Evaluation Metrics**
 We use two evaluation metrics in our experiments: AUC (Area Under ROC) and Logloss (cross entropy).

**Model Comparison**
 We compare 9 models in our experiments: LR, FM, FNN, PNN (three variants), Wide & Deep (two variants), and DeepFM. In the Wide & Deep model, for the purpose of elim- inating feature engineering effort, we also adapt the original Wide & Deep model by replacing LR by FM as the wide part. In order to distinguish these two variants of Wide & Deep, we name them LR & DNN and FM & DNN, respectively.6

**Parameter Settings**

To evaluate the models on Criteo dataset, we follow the pa- rameter settings in [Qu *et al.*, 2016] for FNN and PNN: (1) dropout: 0.5; (2) network structure: 400-400-400; (3) opti- mizer: Adam; (4) activation function: tanh for IPNN, relu for other deep models. To be fair, our proposed DeepFM uses the same setting. The optimizers of LR and FM are FTRL and Adam respectively, and the latent dimension of FM is 10.

To achieve the best performance for each individual model on Company∗ dataset, we conducted carefully parameter study, which is discussed in Section 3.3.

> **数据集**
>
> 我们在以下两个数据集上评估我们提出的DeepFM的有效性和效率。
>
> 1) Criteo数据集：Criteo数据集5包含了4500万用户的点击记录。其中有13个连续特征和26个分类特征。我们将数据集分成两部分：90%用于训练，其余10%用于测试。
>
> 2) Company∗数据集：为了验证DeepFM在真实工业点击率预测中的性能，我们在Company∗数据集上进行了实验。我们收集了Company∗ App Store游戏中心连续7天的用户点击记录用于训练，接下来的1天用于测试。整个收集的数据集中大约有10亿条记录。在这个数据集中，有应用特征（例如，标识、类别等）、用户特征（例如，用户下载的应用等）和上下文特征（例如，操作时间等）。
>
> **评估指标**
>
> 我们在实验中使用两个评估指标：AUC（ROC曲线下面积）和Logloss（交叉熵）。
>
> **模型比较**
>
> 我们在实验中比较了9个模型：LR、FM、FNN、PNN（三个变体）、Wide & Deep（两个变体）和DeepFM。在Wide & Deep模型中，为了消除特征工程的工作量，我们也通过用FM替换LR作为Wide部分来适应原始的Wide & Deep模型。为了区分这两个Wide & Deep变体，我们分别将它们命名为LR & DNN和FM & DNN。6
>
> **参数设置**
>
> 为了评估Criteo数据集上的模型，我们遵循[Qu等人，2016]中的参数设置用于FNN和PNN：(1) dropout：0.5；(2) 网络结构：400-400-400；(3) 优化器：Adam；(4) 激活函数：IPNN使用tanh，其他深度模型使用relu。为了公平起见，我们提出的DeepFM使用相同的设置。LR和FM的优化器分别是FTRL和Adam，FM的潜在维度是10。
>
> 为了在Company∗数据集上实现每个模型的最佳性能，我们进行了仔细的参数研究，这将在第3.3节中讨论。

#### 3.2 Performance Evaluation

In this section, we evaluate the models listed in Section 3.1 on the two datasets to compare their effectiveness and efficiency.

**Efficiency Comparison**

The efficiency of deep learning models is important to real-

world applications. We compare the efficiency of differ-

ent models on Criteo dataset by the following formula:$\frac{|training time of deep CT R model| }{|training time of LR|}$ . The results are shown in Figure 6, including the tests on CPU (left) and GPU (right), where we have the following observations: 1) pre-training of FNN makes it less efficient; 2) Although the speed up of IPNN and PNN∗ on GPU is higher than the other models, they are still computationally expensive because of the in- efficient inner product operations; 3) The DeepFM achieves almost the most efficient in both tests.

**Effectiveness Comparison**

The performance for CTR prediction of different models on Criteo dataset and Company∗ dataset is shown in Table 2 (note that the numbers in the table are averaged by 5 runs of training-testing, and the variances of AUC and Logloss are in the order of 1E-5), where we have the following observations:

- Learning feature interactions improves the performance of CTR prediction model. This observation is from the fact that LR (which is the only model that does not con- sider feature interactions) performs worse than the other models. As the best model, DeepFM outperforms LR by 0.82% and 2.6% in terms of AUC (1.1% and 4.0% in terms of Logloss) on Company∗ and Criteo datasets.
- Learning high- and low-order feature interactions si- multaneously and properly improves the performance of CTR prediction model. DeepFM outperforms the models that learn only low-order feature interactions (namely, FM) or high-order feature interactions (namely, FNN, IPNN, OPNN, PNN∗). Compared to the second best model, DeepFM achieves more than 0.34% and 0.41% in terms of AUC (0.34% and 0.76% in terms of Logloss) on Company∗ and Criteo datasets.
- Learning high- and low-order feature interactions si- multaneously while sharing the same feature embed- ding for high- and low-order feature interactions learn- ing improves the performance of CTR prediction model. DeepFM outperforms the models that learn high- and low-order feature interactions using separate feature em- beddings (namely, LR & DNN and FM & DNN). Com- pared to these two models, DeepFM achieves more than 0.48% and 0.44% in terms of AUC (0.58% and 0.80% in terms of Logloss) on Company∗ and Criteo datasets.

Overall, our proposed DeepFM model beats the competi- tors by more than 0.34% and 0.35% in terms of AUC and Logloss on Company∗ dataset, respectively. In fact, a small improvement in offline AUC evaluation is likely to lead to a significant increase in online CTR. As reported in [Cheng *et al.*, 2016], compared with LR, Wide & Deep improves AUC by 0.275% (offline) and the improvement of online CTR is 3.9%. The daily turnover of Company∗’s App Store is mil- lions of dollars, therefore even several percents lift in CTR brings extra millions of dollars each year. Moreover, we also conduct t-test between our proposed DeepFM and the other compared models. The p-value of DeepFM against FM & DNN under Logloss metric on Company∗ is less than 1.5 × 10−3, and the others’ p-values on both datasets are less than 10−6, which indicates that our improvement over exist- ing models is significant.

> **效率比较**
>
> 深度学习模型的效率对于现实世界的应用很重要。我们通过以下公式比较了Criteo数据集上不同模型的效率：$\frac{|深度学习CTR模型的训练时间|}{|LR的训练时间|}$。结果如图6所示，包括在CPU（左）和GPU（右）上的测试，我们有以下观察：1) FNN的预训练使其效率降低；2) 虽然IPNN和PNN∗在GPU上的加速比其他模型高，但由于低效的内积运算，它们的计算成本仍然很高；3) DeepFM在两次测试中几乎是最有效的。
>
> **有效性比较**
>
> 不同模型在Criteo数据集和Company∗数据集上的CTR预测性能如表2所示（注意，表中的数字是5次训练-测试的平均值，AUC和Logloss的方差在1E-5的数量级），我们有以下观察：
>
> - 学习特征交互可以提高CTR预测模型的性能。这一观察来自于LR（唯一不考虑特征交互的模型）比其他模型表现更差的事实。作为最佳模型，DeepFM在Company∗和Criteo数据集上的AUC分别比LR提高了0.82%和2.6%（在Logloss方面分别提高了1.1%和4.0%）。
>
> - 同时适当地学习高阶和低阶特征交互可以提高CTR预测模型的性能。DeepFM优于仅学习低阶特征交互（即FM）或高阶特征交互（即FNN、IPNN、OPNN、PNN∗）的模型。与第二好的模型相比，DeepFM在Company∗和Criteo数据集上的AUC分别提高了0.34%和0.41%（在Logloss方面分别提高了0.34%和0.76%）。
>
> - 同时学习高阶和低阶特征交互，同时为高阶和低阶特征交互学习共享相同的特征嵌入，可以提高CTR预测模型的性能。DeepFM优于使用单独的特征嵌入学习高阶和低阶特征交互的模型（即LR & DNN和FM & DNN）。与这两个模型相比，DeepFM在Company∗和Criteo数据集上的AUC分别提高了0.48%和0.44%（在Logloss方面分别提高了0.58%和0.80%）。
>
> 总的来说，我们提出的DeepFM模型在Company∗数据集上的AUC和Logloss方面分别比竞争对手提高了0.34%和0.35%以上。事实上，离线AUC评估的微小改进可能会导致在线CTR的显著增加。据[Cheng等人，2016]报道，与LR相比，Wide & Deep的AUC提高了0.275%（离线），而在线CTR的改进为3.9%。Company∗的App Store的日营业额高达数百万美元，因此CTR的百分之几的提升每年都能带来数百万美元的额外收入。此外，我们还对我们提出的DeepFM和其他比较模型进行了t检验。DeepFM与FM & DNN在Company∗上的Logloss指标的p值小于1.5 × 10−3，其他两个数据集上的p值均小于10−6，这表明我们对现有模型的改进是显著的。

![Table2](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Table2.png)

![Figure6](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure6.png)

#### 3.3 Hyper-Parameter Study

We study the impact of different hyper-parameters of differ- ent deep models, on Company∗ dataset. The order is: 1) ac- tivation functions; 2) dropout rate; 3) number of neurons per layer; 4) number of hidden layers; 5) network shape.

**Activation Function**

According to [Qu *et al.*, 2016], *relu* and *tanh* are more suit- able for deep models than *sigmoid*. In this paper, we compare the performance of deep models when applying *relu* and *tanh*. As shown in Figure 7, relu is more appropriate than tanh for all the deep models, except for IPNN. Possible reason is that relu induces sparsity.

**Dropout**

Dropout [Srivastava *et al.*, 2014] refers to the probability that a neuron is kept in the network. Dropout is a regularization technique to compromise the precision and the complexity of the neural network. We set the dropout to be 1.0, 0.9, 0.8, 0.7, 0.6, 0.5. As shown in Figure 8, all the models are able to reach their own best performance when the dropout is properly set (from 0.6 to 0.9). The result shows that adding reasonable randomness to model can strengthen model’s robustness.

**Number of Neurons per Layer**

When other factors remain the same, increasing the number of neurons per layer introduces complexity. As we can ob- serve from Figure 9, increasing the number of neurons does not always bring benefit. For instance, DeepFM performs sta- bly when the number of neurons per layer is increased from 400 to 800; even worse, OPNN performs worse when we in- crease the number of neurons from 400 to 800. This is be- cause an over-complicated model is easy to overfit. In our dataset, 200 or 400 neurons per layer is a good choice.

**Number of Hidden Layers**

As presented in Figure 10, increasing number of hidden lay- ers improves the performance of the models at the beginning, however, their performance is degraded if the number of hid- den layers keeps increasing, because of overfitting.

**Network Shape**

We test four different network shapes: constant, increasing, decreasing, and diamond. When we change the network shape, we fix the number of hidden layers and the total num- ber of neurons. For instance, when the number of hidden lay- ers is 3 and the total number of neurons is 600, then four dif- ferent shapes are: constant (200-200-200), increasing (100- 200-300), decreasing (300-200-100), and diamond (150-300- 150). As we can see from Figure 11, the “constant” network shape is empirically better than the other three options, which is consistent with previous studies [Larochelle *et al.*, 2009].

> 我们研究了不同深度模型的不同超参数对公司数据集的影响。顺序如下：1）激活函数；2）丢弃率；3）每层的神经元数量；4）隐藏层的数量；5）网络形状。
>
> **激活函数**
>
> 根据[Qu等人，2016]，对于深度模型来说，relu和tanh比sigmoid更适合。在本文中，我们比较了应用relu和tanh时深度模型的性能。如图7所示，除了IPNN之外，relu比其他所有深度模型都更合适。可能的原因是relu诱导稀疏性。
>
> **丢弃率**
>
> 丢弃率[Srivastava等人，2014]是指神经元保留在网络中的概率。丢弃是一种权衡神经网络精度和复杂度的正则化技术。我们将丢弃率设置为1.0、0.9、0.8、0.7、0.6、0.5。如图8所示，当适当设置丢弃率时（从0.6到0.9），所有模型都能达到各自的最佳性能。结果表明，在模型中添加合理的随机性可以增强模型的鲁棒性。
>
> **每层的神经元数量**
>
> 当其他因素保持不变时，增加每层的神经元数量会引入复杂性。如图9所示，增加神经元数量并不总是带来好处。例如，当每层的神经元数量从400增加到800时，DeepFM的性能保持稳定；更糟糕的是，当我们将神经元的数量从400增加到800时，OPNN的性能变得更差。这是因为过于复杂的模型容易过拟合。在我们的数据集中，每层200或400个神经元是一个好选择。
>
> **隐藏层的数量**
>
> 如图10所示，增加隐藏层的数量一开始会提高模型的性能，但如果隐藏层的数量继续增加，由于过拟合，它们的性能会下降。
>
> **网络形状**
>
> 我们测试了四种不同的网络形状：常数、递增、递减和菱形。当我们改变网络形状时，我们固定隐藏层的数量和神经元的总数。例如，当隐藏层的数量为3，神经元的总数为600时，四种不同的形状为：常数（200-200-200）、递增（100-200-300）、递减（300-200-100）和菱形（150-300-150）。如图11所示，“常数”网络形状在经验上比其他三个选项更好，这与之前的研究[Larochelle等人，2009]一致。

![Figure7](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure7.png)

![Figure8](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure8.png)

![Figure9](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure9.png)

![Figure10](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure10.png)

![Figure11](/Users/anmingyu/Github/Gor-rok/Papers/rank/DeepFM/Figure11.png)

## 4 Related Work

In this paper, a new deep neural network is proposed for CTR prediction. The most related domains are CTR prediction and deep learning in recommender system.

CTR prediction plays an important role in recommender system [Richardson *et al.*, 2007; Juan *et al.*, 2016]. Besides generalized linear models and FM, a few other models are proposed for CTR prediction, such as tree-based model [He *et al.*, 2014], tensor based model [Rendle and Schmidt-Thieme, 2010], support vector machine [Chang *et al.*, 2010], and bayesian model [Graepel *et al.*, 2010].

The other related domain is deep learning in recommender systems. In Section 1 and Section 2.2, several deep learn- ing models for CTR prediction are already mentioned, thus we do not discuss about them here. Several deep learn- ing models are proposed in recommendation tasks other than CTR prediction (e.g., [Covington *et al.*, 2016; Salakhutdi- nov *et al.*, 2007; van den Oord *et al.*, 2013; Wu *et al.*, 2016; Zheng *et al.*, 2016; Wu *et al.*, 2017; Zheng *et al.*, 2017]). [Salakhutdinov *et al.*, 2007; Sedhain *et al.*, 2015; Wang *et al.*, 2015] propose to improve Collaborative Filter- ing via deep learning. The authors of [Wang and Wang, 2014; van den Oord *et al.*, 2013] extract content feature by deep learning to improve the performance of music recommenda- tion. [Chen *et al.*, 2016] devises a deep learning network to consider both image feature and basic feature of display ad- verting. [Covington *et al.*, 2016] develops a two-stage deep learning framework for YouTube video recommendation.

> 在本文中，提出了一种新的深度神经网络用于点击率（CTR）预测。最相关的领域是推荐系统中的CTR预测和深度学习。
>
> CTR预测在推荐系统中起着重要作用[Richardson等人，2007；Juan等人，2016]。除了广义线性模型和因子分解机（FM）之外，还提出了其他一些用于CTR预测的模型，如基于树的模型[He等人，2014]、基于张量的模型[Rendle和Schmidt-Thieme，2010]、支持向量机[Chang等人，2010]和贝叶斯模型[Graepel等人，2010]。
>
> 另一个相关领域是推荐系统中的深度学习。在第1节和第2.2节中，已经提到了几种用于CTR预测的深度学习模型，因此这里不再讨论。除了CTR预测之外，还有一些深度学习模型被提出用于推荐任务（例如，[Covington等人，2016；Salakhutdinov等人，2007；van den Oord等人，2013；Wu等人，2016；Zheng等人，2016；Wu等人，2017；Zheng等人，2017]）。[Salakhutdinov等人，2007；Sedhain等人，2015；Wang等人，2015]提出通过深度学习改进协同过滤。[Wang和Wang，2014；van den Oord等人，2013]通过深度学习提取内容特征，以提高音乐推荐的性能。[Chen等人，2016]设计了一个深度学习网络，同时考虑显示广告的图像特征和基本特征。[Covington等人，2016]为YouTube视频推荐开发了一个两阶段的深度学习框架。

## 5 Conclusions

In this paper, we proposed DeepFM, a factorization-machine based neural network for CTR prediction, to overcome the shortcomings of the state-of-the-art models. DeepFM trains a deep component and an FM component jointly. It gains performance improvement from these advantages: 1) it does not need any pre-training; 2) it learns both high- and low- order feature interactions; 3) it introduces a sharing strat- egy of feature embedding to avoid feature engineering. The experiments on two real-world datasets demonstrate that 1) DeepFM outperforms the state-of-the-art models in terms of AUC and Logloss on both datasets; 2) The efficiency of DeepFM is comparable to the most efficient deep model in the state-of-the-art.

> 在本文中，我们提出了DeepFM，一种基于因子分解机的神经网络用于点击率（CTR）预测，以克服现有模型的缺点。DeepFM联合训练一个深度组件和一个FM组件。它从这些优势中获得性能提升：1）它不需要任何预训练；2）它学习高阶和低阶特征交互；3）它引入了一种特征嵌入的共享策略，以避免特征工程。在两个真实世界数据集上的实验表明：1）DeepFM在AUC和Logloss方面均优于两个数据集上的现有模型；2）DeepFM的效率与现有技术中最有效的深度模型相当。