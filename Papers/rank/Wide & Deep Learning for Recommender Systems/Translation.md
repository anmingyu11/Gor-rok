# Wide & Deep Learning for Recommender Systems

## ABSTRACT

Generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. Memorization of feature interactions through a wide set of cross-product feature transformations are effective and interpretable, while generalization requires more feature engineering effort. With less feature engineering, deep neural networks can generalize better to unseen feature combinations through low-dimensional dense embeddings learned for the sparse features. However, deep neural networks with embeddings can over-generalize and recommend less relevant items when the user-item interactions are sparse and high-rank. 

In this paper, we present Wide & Deep learning—jointly trained wide linear models and deep neural networks—to combine the benefits of memorization and generalization for recommender systems. We productionized and evaluated the system on Google Play, a commercial mobile app store with over one billion active users and over one million apps. Online experiment results show that Wide & Deep significantly increased app acquisitions compared with wide-only and deep-only models. We have also open-sourced our implementation in TensorFlow.

> 具有非线性特征变换的广义线性模型被广泛应用于具有稀疏输入的大规模回归和分类问题。通过广泛的 cross- product 特征变换来记忆特征交叉是有效和可解释的，而泛化则需要更多的特征工程工作。特征工程较少的情况下，通过学习稀疏特征的 low-dimensional dense embedding ，深度神经网络可以更好地泛化到看不见的特征组合。然而，当 user 与 item 的交互是稀疏且 high-rank 时，有着 embedding 的深度神经网络会过度泛化并且推荐不太相关的产品。
>
> 在本文中，我们提出了 Wide & Deep learning - 联合训练 wide 线性模型 和 deep 神经网络- 将记忆性和泛化性的优点结合起来并应用于推荐系统。我们在 Google Play 上对该系统进行了生产和评估，Google Play 是一家商业移动应用商店，拥有超过10亿活跃用户和100多万个应用。线上实验结果显示，Wide&Deep 与 Wide-Only 和 Deep-Only 两种模式相比，显著增加了APP的采购量。我们还开源了我们在 TensorFlow 中的实现。

## 1. INTRODUCTION

A recommender system can be viewed as a search ranking system, where the input query is a set of user and contextual information, and the output is a ranked list of items. Given a query, the recommendation task is to find the relevant items in a database and then rank the items based on certain objectives, such as clicks or purchases.

One challenge in recommender systems, similar to the general search ranking problem, is to achieve both memorization and generalization. Memorization can be loosely defined as learning the frequent co-occurrence of items or features and exploiting the correlation available in the historical data. Generalization, on the other hand, is based on transitivity of correlation and explores new feature combinations that have never or rarely occurred in the past. 

Recommendations based on memorization are usually more topical and directly relevant to the items on which users have already performed actions. Compared with memorization, generalization tends to improve the diversity of the recommended items. In this paper, we focus on the apps recommendation problem for the Google Play store, but the approach should apply to generic recommender systems.

For massive-scale online recommendation and ranking systems in an industrial setting, generalized linear models such as logistic regression are widely used because they are simple, scalable and interpretable. The models are often trained on binarized sparse features with one-hot encoding. E.g., the binary feature **“user_installed_app=netflix”** has value $1$ if the user installed Netflix. Memorization can be achieved effectively using cross-product transformations over sparse features, such as **AND(user_installed_app=netflix, impression_app=pandora”)**, whose value is $1$ if the user installed Netflix and then is later shown Pandora. This explains how the co-occurrence of a feature pair correlates with the target label. Generalization can be added by using features that are less granular, such as **AND(user_installed_category=video, impression_category=music)**, but manual feature engineering is often required. One limitation of cross-product transformations is that they do not generalize to query-item feature pairs that have not appeared in the training data.

> 推荐系统可以被视为搜索排序系统，其中输入查询是一组 user 和 context 信息，而输出是排序后的 item 列表。在给定查询的情况下，推荐任务是在数据库中查找相关 item，然后根据特定目标(例如 clicks 或 purchases)对 item 进行排名。
>
> 类似于一般搜索排序问题，推荐系统中的一项挑战是同时实现记忆化和泛化。
>
> - 记忆性可以粗略地定义为学习 item 或特征的频繁共现，并利用历史数据中可用的相关性。
> - 泛化性基于相关性的传递性，并探索过去从未或很少出现的新特征组合。
>
> 基于记忆性的推荐通常更具针对性，并与用户已经有过行为的 item相关。
> 与记忆性相比，泛化性更倾向于提高推荐的 item 的多样性。在本文中，我们关注的是 Google Play 商店的应用程序推荐问题，但该方法应该适用于通用推荐系统。
>
> 对于工业背景下的大规模在线推荐和排序系统，广义线性模型(如逻辑回归)被广泛使用，因为它们简单、可扩展和可解释。该模型通常采用 one-hot 编码的二值稀疏特征进行训练。例如，如果用户安装了 netflix ，二值特征 **"user_installed_app=netflix"** 的值为 $1$。通过稀疏特征的 cross-product 转换可以有效地实现记忆性，例如 **AND(user_installed_app=netflix，impression_app=pandora ")**，如果用户安装了 netflix，然后显示 pandora，其值为 $1$。这解释了特征对的共现是如何与目标标签相关联的。泛化可以通过使用粒度更小的特性来添加，例如 **AND(user_installed_category=video，impression_category=music)**，但通常需要手工特征工程。cross-product 转换的一上限是它们不能泛化到没有出现在训练数据中的 query-item 特征对。

Embedding-based models, such as factorization machines [5] or deep neural networks, can generalize to previously unseen query-item feature pairs by learning a low-dimensional dense embedding vector for each query and item feature, with less burden of feature engineering. 

However, it is difficult to learn effective low-dimensional representations for queries and items when the underlying query-item matrix is sparse and high-rank, such as users with specific preferences or niche items with a narrow appeal. In such cases, there should be no interactions between most query-item pairs, but dense embeddings will lead to nonzero predictions for all query-item pairs, and thus can over-generalize and make less relevant recommendations. On the other hand, linear models with cross-product feature transformations can memorize these “exception rules” with much fewer parameters.

In this paper, we present the Wide & Deep learning framework to achieve both memorization and generalization in one model, by jointly training a linear model component and a neural network component as shown in Figure 1.

The main contributions of the paper include:

- The Wide & Deep learning framework for jointly training feed-forward neural networks with embeddings and linear model with feature transformations for generic recommender systems with sparse inputs.
- The implementation and evaluation of the Wide & Deep recommender system productionized on Google Play, a mobile app store with over one billion active users and over one million apps.
- We have open-sourced our implementation along with a high-level API in TensorFlow1.

While the idea is simple, we show that the Wide & Deep framework significantly improves the app acquisition rate on the mobile app store, while satisfying the training and serving speed requirements.

![Figure1](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig1.png)

**Figure 1: The spectrum of Wide & Deep models**

> Embdding-based 的模型，如因子分解机[5]或深度神经网络，可以通过学习每个 query-item 特征的 low-dimensional dense embedding 向量来推广到以前未曾见过的 query-item 特征对，而特征工程的负担更小。
>
> 然而，当 query-item 矩阵是稀疏且高阶的(例如用户具有特定的偏好或小众)时，很难学习 query-item 的有效低维表示。在这种情况下，大多数 query-item 对之间是没有交互的，但是 dense embedding 将使得所有 query-item 对的预测为非零，从而可能过度泛化并给出了不相关的 item。另一方面，使用 cross-product 特征变换的线性模型可以用更少的参数记住这些 “额外规则”。
>
> 在本文中，我们提出了 Wide & Deep 学习框架，通过联合训练线性模型组件和神经网络组件，在一个模型中实现记忆性和泛化性，如图1所示。
>
> 论文的主要贡献包括：
>
> - 针对输入稀疏的通用推荐系统，提出了一种用于联合训练 embedding 前馈神经网络和特征变换线性模型的 Wide&Deep 学习框架。
> - W&D 推荐系统的实践和评估是在 Google Play 上生产的，Google Play是一家移动应用商店，拥有超过10亿活跃用户和超过100万个应用程序。
> - 我们已经将我们的实现开源，其中包括 TensorFlow1 中的高级 API 。
>
> 虽然想法很简单，但我们表明，Wide&Deep 框架在满足训练和服务速度要求的同时，显著提高了移动应用商店上的应用获取率。

## 2. RECOMMENDER SYSTEM OVERVIEW

An overview of the app recommender system is shown in Figure 2. A query, which can include various user and contextual features, is generated when a user visits the app store. The recommender system returns a list of apps (also referred to as impressions) on which users can perform certain actions such as clicks or purchases. These user actions, along with the queries and impressions, are recorded in the logs as the training data for the learner.

Since there are over a million apps in the database, it is intractable to exhaustively score every app for every query within the serving latency requirements (often $O(10)$ milliseconds). Therefore, the first step upon receiving a query is retrieval. The retrieval system returns a short list of items that best match the query using various signals, usually a combination of machine-learned models and human-defined rules. 

After reducing the candidate pool, the ranking system ranks all items by their scores. The scores are usually $P(y|\textbf{x})$, the probability of a user action label $y$ given the features $\mathbf{x}$, including user features (e.g., country, language, demographics), contextual features (e.g., device, hour of the day, day of the week), and impression features (e.g., app age, historical statistics of an app). In this paper, we focus on the ranking model using the Wide & Deep learning framework.

![Figure2](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig2.png)

**Figure 2: Overview of the recommender system.**

> 图2 显示了推荐系统的应用概况。当用户访问应用商店时，会生成一个query，该 query 可以包括各种用户和上下文特征。推荐系统返回用户可以在其上执行诸如点击或购买等特定动作的 app 列表(也称为impressions)。这些用户动作以及 queries 和 impressions 被记录在日志中，作为学习者的训练数据。
>
> 由于数据库中有 100多万个应用程序，因此很难在服务延迟要求(通常为$O(10)$毫秒)内对每个 query 的每个 app 进行详尽的评分。因此，收到查询后的第一步是检索。检索系统使用各种信号(通常是机器学习的模型和人定义的规则的组合)返回与 query 最匹配的 item 的列表。
>
> 减少候选库规模后，排序系统会根据所有 item 的分数对其进行排序。分数通常是 $P(y|\mathbf{x})$、给定特征 $\mathbf{x}$ 的用户 action 标签 $y$ 的概率，所述特征 $\mathbf{x}$ 包括用户特征 (例如，国家、语言、人口统计) 、上下文特征 (例如，设备、每天的小时、星期几) 和 impression 特征 (例如，应用的年龄、应用的历史统计)。在本文中，我们重点研究了基于 W&D 学习框架的排序模型。

## 3. WIDE & DEEP LEARNING

#### 3.1 The Wide Component

The wide component is a generalized linear model of the form $y = \mathbf{w}^T\mathbf{x} + b$, as illustrated in Figure 1 (left). $y$ is the prediction, $\mathbf{x} = [x_1, x_2, \cdots, x_d]$ is a vector of $d$ features, $\mathbf{w} = [w_1, w_2, \cdots, w_d]$ are the model parameters and $b$ is the bias. The feature set includes raw input features and transformed features. One of the most important transformations is the cross-product transformation, which is defined as: 
$$
\phi_{k}(\mathbf{x})=\prod_{i=1}^{d} x_{i}^{c_{k i}} 
\quad c_{k i} \in\{0,1\}
\qquad(1)
$$
where $c_{ki}$ is a boolean variable that is $1$ if the $i$-th feature is part of the $k$-th transformation $\phi_k$, and $0$ otherwise. For binary features, a cross-product transformation (e.g., **“AND(gender=female, language=en)”**) is $1$ if and only if the constituent features (**“gender=female” and “language=en”**) are all $1$, and $0$ otherwise. This captures the interactions between the binary features, and adds nonlinearity to the generalized linear model.

> Wide 组件是形式为 $y=\mathbf{w}^T\mathbf{x}+b$ 的广义线性模型，如图1(左)所示。$y$ 是预测值，$\mathbf{x}=[x_1，x_2，\cdots，x_d]$ 是 $d$ 特征的向量，$\mathbf{w}=[w_1，w_2，\cdots，w_d]$ 是模型参数，$b$ 是偏差。特征集合包括原始输入特征和变换后的特征。最重要的变换之一是 cross-product 变换，其定义为：
> $$
> \phi_{k}(\mathbf{x})=\prod_{i=1}^{d} x_{i}^{c_{k i}} 
> \quad c_{k i} \in\{0,1\}
> \qquad(1)
> $$
> 其中，$c_{ki}$ 是布尔变量，如果第  i 个特征是第 k 个转换 $\phi_k$ 的一部分，则为 $1$ ，否则为 $0$ 。对于二元特征，当且仅当构成特征 (**“AND(gender=female, language=en)”**) 都为 $1$ 时，cross-product 变换为 $1$，否则为 $0$ (例如，**“AND(gender=female，language=en)”**)。这捕获了二值特征之间的交叉，并将非线性添加到广义线性模型中。

#### 3.2 The Deep Component

The deep component is a feed-forward neural network, as shown in Figure 1 (right). For categorical features, the original inputs are feature strings (e.g., **“language=en”**). Each of these sparse, high-dimensional categorical features are first converted into a low-dimensional and dense real-valued vector, often referred to as an embedding vector. The dimensionality of the embeddings are usually on the order of $O(10)$ to $O(100)$. The embedding vectors are initialized randomly and then the values are trained to minimize the final loss function during model training. These low-dimensional dense embedding vectors are then fed into the hidden layers of a neural network in the forward pass. Specifically, each hidden layer performs the following computation:
$$
a^{(l+1)}=f\left(W^{(l)} a^{(l)}+b^{(l)}\right) \qquad(12)
$$
where $l$ is the layer number and $f$ is the activation function, often rectified linear units (ReLUs). $a^{(l)}$ , $b^{(l)}$ , and $W^{(l)}$ are the activations, bias, and model weights at $l$-th layer.

> deep 组件是前馈神经网络，如图1(右)所示。对于类别特征，原始输入是特征字符串(例如，**“language=en”**)。这些稀疏的、high-dimensional categorical features 中的每一个首先被转换成 low-dimensional and dense 实值向量，通常被称为 embedding 向量。embedding 的维度通常在 $O(10)$ 到 $O(100)$ 的数量级上。在模型训练过程中，随机初始化 embedding 向量，然后训练这些值以最小化最终的损失函数。然后，这些低维的密集 embedding 向量在前向传播中被送到神经网络的隐藏层。具体地说，每个隐藏层执行以下计算：
> $$
> a^{(l+1)}=f\left(W^{(l)} a^{(l)}+b^{(l)}\right) \qquad(12)
> $$
> 其中 $l$ 是层号，$f$ 是激活函数，通常是整流线性单元(RELU)。$a^{(L)}$、$b^{(L)}$ 和 $W^{(L)}$ 是第 $l$ 层的激活函数、偏置单元和模型权重。

#### 3.3 Joint Training of Wide & Deep Model

The wide component and deep component are combined using a weighted sum of their output log odds as the prediction, which is then fed to one common logistic loss function for joint training. Note that there is a distinction between joint training and ensemble. In an ensemble, individual models are trained separately without knowing each other, and their predictions are combined only at inference time but not at training time. 

In contrast, joint training optimizes all parameters simultaneously by taking both the wide and deep part as well as the weights of their sum into account at training time. There are implications on model size too: For an ensemble, since the training is disjoint, each individual model size usually needs to be larger (e.g., with more features and transformations) to achieve reasonable accuracy for an ensemble to work. In comparison, for joint training the wide part only needs to complement the weaknesses of the deep part with a small number of cross-product feature transformations, rather than a full-size wide model.

Joint training of a Wide & Deep Model is done by backpropagating the gradients from the output to both the wide and deep part of the model simultaneously using mini-batch stochastic optimization. 

In the experiments, we used Follow-the-regularized-leader (FTRL) algorithm [3] with L1 regularization as the optimizer for the wide part of the model, and AdaGrad [1] for the deep part.

The combined model is illustrated in Figure 1 (center). For a logistic regression problem, the model’s prediction is:
$$
P(Y=1 \mid \mathbf{x})=\sigma\left(\mathbf{w}_{w i d e}^{T}[\mathbf{x}, \phi(\mathbf{x})]+\mathbf{w}_{d e e p}^{T} a^{\left(l_{f}\right)}+b\right)
\qquad (3)
$$
where $Y$ is the binary class label, $\sigma(·)$ is the sigmoid function, $\phi(x)$ are the cross product transformations of the original features $\mathbf{x}$, and $b$ is the bias term. $\mathbf{w}_{wide}$ is the vector of all wide model weights, and $\mathbf{w}_{deep}$ are the weights applied on the final activations $a^{(l_f)}$ .

> 将 wide component 和 deep component 以其输出对数概率的加权和作为预测值进行组合，然后将其输入一个通用的 logistic 损失函数进行联合训练。注意联合训练和 ensemble 训练是有区别的。在 ensemble中，单个模型在互相独立的情况下独立训练，并且它们的预测只在推理时组合，而不是在训练时。而联合训练则通过在训练时同时考虑 Deep 和 Wide 部分以及它们之和的权重来同时优化所有参数。
>
> 对模型大小也有影响：对于一个 ensemble 模型，由于训练是不连贯的，每个独立的模型大小通常需要更大(例如，有更多的特征和特征变换)，以达到一个 ensemble 工作的合理精度。相比之下，对于联合训练，Wide 部分只需要通过少量的 cross-product 特征变换来弥补 Deep 部分的不足，而不需要一个全尺寸的 Wide 模型。
>
> W&D 模型的联合训练是通过使用小批量随机优化将梯度从输出同时反向传播到模型的 Wide 和 Deep 两部分来完成的。
>
> 在实验中，我们使用带 L1 正则化的 Followthe-regular-leader(FTRL)算法[3]作为模型的 wide 部分的优化器，而 AdaGrad[1] 作为模型的 deep 部分的优化器。
>
> 组合模型如图1(中间)所示。对于逻辑回归问题，模型的预测为：
> $$
> P(Y=1 \mid \mathbf{x})=\sigma\left(\mathbf{w}_{w i d e}^{T}[\mathbf{x}, \phi(\mathbf{x})]+\mathbf{w}_{d e e p}^{T} a^{\left(l_{f}\right)}+b\right)
> \qquad (3)
> $$
> 其中 $Y$ 是二分类标签，$\sigma(·)$ 是 sigmoid 函数，$\phi(X)$ 是原始特征 $\mathbf{x}$ 的 cross-product 变换，$b$ 是偏置项。$\mathbf{w}_{wide}$ 是所有 wide 模型权重的向量，$\mathbf{w}_{deep}$ 是应用于最终激活 $a^{(l_f)}$ 的权重。

## 4. SYSTEM IMPLEMENTATION

The implementation of the apps recommendation pipeline consists of three stages: data generation, model training, and model serving as shown in Figure 3.

> APPS 推荐流程的实现包括三个阶段：数据生成、模型训练和模型服务，如图3所示。

![Figure3](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig3.png)

**Figure 3: Apps recommendation pipeline overview.**

#### 4.1 Data Generation

In this stage, user and app impression data within a period of time are used to generate training data. Each example corresponds to one impression. The label is app acquisition: $1$ if the impressed app was installed, and $0$ otherwise.

Vocabularies, which are tables mapping categorical feature strings to integer IDs, are also generated in this stage. The system computes the ID space for all the string features that occurred more than a minimum number of times. Continuous real-valued features are normalized to $[0, 1]$ by mapping a feature value $\mathbf{x}$ to its cumulative distribution function $P(X ≤ x)$, divided into $n_q$ quantiles. The normalized value is $\frac{i−1}{n_q - 1}$ for values in the $i$-th quantiles. Quantile boundaries are computed during data generation.

> 在这个阶段，我们使用一段时间内的 user 和 app impression 数据来生成训练数据。每个例子对应一个 impression。标签是应用是否被获取：如果应用被安装，则为 1，否则为 0。
>
> 词汇表是将类别特征映射到 id 的表，也在这个阶段生成。系统为所有出现次数超过最小次数的特征计算 ID 空间。通过将特征值 $\mathbf{x}$ 映射到其累积分布函数 $P(x≤x)$，将连续实值特征归一化为 $[0,1]$ ，并将其分为 $n_q$ 分位数。对于 $i$ -th分位数的值，归一化值为 $\frac{i−1}{n_q - 1}$。分位数边界在数据生成期间计算。

#### 4.2 Model Training

The model structure we used in the experiment is shown in Figure 4. During training, our input layer takes in training data and vocabularies and generate sparse and dense features together with a label. The wide component consists of the cross-product transformation of user installed apps and impression apps.For the deep part of the model, A $32$- dimensional embedding vector is learned for each categorical feature. We concatenate all the embeddings together with the dense features, resulting in a dense vector of approximately $1200$ dimensions. The concatenated vector is then fed into $3$ ReLU layers, and finally the logistic output unit.

The Wide & Deep models are trained on over 500 billion examples. Every time a new set of training data arrives, the model needs to be re-trained. However, retraining from scratch every time is computationally expensive and delays the time from data arrival to serving an updated model. To tackle this challenge, we implemented a warm-starting system which initializes a new model with the embeddings and the linear model weights from the previous model.

Before loading the models into the model servers, a dry run of the model is done to make sure that it does not cause problems in serving live traffic. We empirically validate the model quality against the previous model as a sanity check.

![Figure4](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Fig4.png)

**Figure 4: Wide & Deep model structure for apps recommendation.**

> 我们在实验中使用的模型结构如图4所示。在训练过程中，我们的输入层接收训练数据和词汇表，并生成稀疏和密集的特征以及标签。wide 的组成部分包括用户安装的应用程序和 impression 应用程序的 cross-product 变换。对于模型的 deep 部分，为每个类别特征学习一个32维的 embedding 向量。
> 我们将所有的 embedding 和 dense 特征连接在一起，产生大约 1200 维的 dense 向量。然后，将连接后的向量送入 3 个 RELU层，最后送入逻辑输出单元(logistic output)。
>
> Wide&Deep 模型对超过 5000亿 个例子进行了训练。每当一组新的训练数据到达时，模型都需要重新训练。然而，每次从头开始训练的计算成本很高，并且会延长从数据到达到提供更新好后的模型所需的时间。为了应对这一挑战，我们实现了一个热启动系统，该系统使用来自前一个模型的 embedding 和线性模型权重来初始化一个新模型。
>
> 在将模型加载到模型服务器之前，会对模型进行模拟运行，以确保在服务实时流量时不会出现问题。我们根据先前的模型对模型质量进行经验性的验证，以此作为一种合理性的检查。

#### 4.3 Model Serving

Once the model is trained and verified, we load it into the model servers. For each request, the servers receive a set of app candidates from the app retrieval system and user features to score each app. Then, the apps are ranked from the highest scores to the lowest, and we show the apps to the users in this order. The scores are calculated by running a forward inference pass over the Wide & Deep model.

In order to serve each request on the order of 10 ms, we optimized the performance using multithreading parallelism by running smaller batches in parallel, instead of scoring all candidate apps in a single batch inference step.

> 一旦模型经过训练和验证，我们就将其加载到模型服务器中。对于每个请求，服务器从应用程序检索系统接收一组候选应用程序和用户特征，以对每个应用程序进行打分。然后，应用程序从最高分到最低分进行排序，我们按照这个顺序向用户展示这些应用程序。分数是通过在 Wide&Deep 模型上运行正向推理过程来计算的。
>
> 为了在 10ms 量级上为每个请求提供服务，我们通过并行运行较小的批处理来使用多线程并行来优化性能，而不是在单个批处理推理步骤中对所有候选应用程序进行评分。

## 5. EXPERIMENT RESULTS

To evaluate the effectiveness of Wide & Deep learning in a real-world recommender system, we ran live experiments and evaluated the system in a couple of aspects: app acquisitions and serving performance.

> 为了评估 Wide&Deep 学习在推荐系统中的实践有效性，我们进行了线上实验，并从应用程序获取和服务性能两个方面对系统进行了评估。

#### 5.1 App Acquisitions

We conducted live online experiments in an A/B testing framework for 3 weeks. For the control group, 1% of users were randomly selected and presented with recommendations generated by the previous version of ranking model, which is a highly-optimized wide-only logistic regression model with rich cross-product feature transformations. For the experiment group, 1% of users were presented with recommendations generated by the Wide & Deep model, trained with the same set of features. As shown in Table 1, Wide & Deep model improved the app acquisition rate on the main landing page of the app store by +3.9% relative to the control group (statistically significant). The results were also compared with another 1% group using only the deep part of the model with the same features and neural network structure, and the Wide & Deep mode had +1% gain on top of the deep-only model (statistically significant).

Besides online experiments, we also show the Area Under Receiver Operator Characteristic Curve (AUC) on a holdout set offline. While Wide & Deep has a slightly higher offline AUC, the impact is more significant on online traffic. One possible reason is that the impressions and labels in offline data sets are fixed, whereas the online system can generate new exploratory recommendations by blending generalization with memorization, and learn from new user responses.

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Table1.png)

**Table 1: Offline & online metrics of different models. Online Acquisition Gain is relative to the control.**

> 我们在 A/B testing 框架中进行了为期 3 周的线上实验。对于控制组，随机选择 1% 的用户，并向他们展示由前一版本的排序模型生成的推荐列表，该模型是一种高度优化的 广义 Logistic regression 模型，具有丰富的 cross-product 特征变换。对于实验组，1% 的用户被呈现给由 Wide & Deep 模型生成的推荐，并使用相同的特征集进行训练。
>
> 如 表1 所示，Wide & Deep model 与对照组相比，应用商店主登陆页面的 APP 获取率提升了 +3.9%(统计显著)。结果还与仅使用具有相同特征和神经网络结构的模型的 deep 的另一个 1% 组进行了比较，Wide&Deep 模式在仅使用 deep 模型的基础上有 +1% 的增益(统计学意义)。
>
> 除了线上实验外，我们还给出了离线情况下 ROC 曲线下的面积(AUC)。虽然 Wide&Deep 的线下 AUC 略高，但对线上流量的影响更大。一个可能的原因是离线数据集中的 impression 和 label 是固定的，而 online 系统可以通过混合泛化性和记忆性来生成新的探索性推荐，并从新用户的 responses 中学习。

#### 5.2 Serving Performance

Serving with high throughput and low latency is challenging with the high level of traffic faced by our commercial mobile app store. At peak traffic, our recommender servers score over 10 million apps per second. With single threading, scoring all candidates in a single batch takes 31 ms. We implemented multithreading and split each batch into smaller sizes, which significantly reduced the client-side latency to 14 ms (including serving overhead) as shown in Table 2.

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/CTR/Wide & Deep Learning for Recommender Systems/Table2.png)

**Table 2: Serving latency vs. batch size and threads.**

> 我们的移动应用商店面临着高流量的情况下，提供高吞吐量和低延迟的服务是具有挑战性的。在流量高峰期，我们的推荐服务器每秒可获得超过 1000万 个应用程序的评分。
>
> 使用单线程，对一个 batch 中的候选集进行评分需要 31 毫秒。我们实现了多线程，并将每个批处理拆分为更小的大小，这显著地将客户端延迟降低到了 14ms (包括服务开销)，如表2所示。

## 6. RELATED WORK

The idea of combining wide linear models with crossproduct feature transformations and deep neural networks with dense embeddings is inspired by previous work, such as factorization machines [5] which add generalization to linear models by factorizing the interactions between two variables as a dot product between two low-dimensional embedding vectors. In this paper, we expanded the model capacity by learning highly nonlinear interactions between embeddings via neural networks instead of dot products.

In language models, joint training of recurrent neural networks (RNNs) and maximum entropy models with n-gram features has been proposed to significantly reduce the RNN complexity (e.g., hidden layer sizes) by learning direct weights between inputs and outputs [4]. In computer vision, deep residual learning [2] has been used to reduce the difficulty of training deeper models and improve accuracy with shortcut connections which skip one or more layers. Joint training of neural networks with graphical models has also been applied to human pose estimation from images [6]. In this work we explored the joint training of feed-forward neural networks and linear models, with direct connections between sparse features and the output unit, for generic recommendation and ranking problems with sparse input data.

In the recommender systems literature, collaborative deep learning has been explored by coupling deep learning for content information and collaborative filtering (CF) for the ratings matrix [7]. There has also been previous work on mobile app recommender systems, such as AppJoy which used CF on users’ app usage records [8]. Different from the CF-based or content-based approaches in the previous work, we jointly train Wide & Deep models on user and impression data for app recommender systems.

> 将 wide 线性模型与 cross-product 变换和深度神经网络与 dense embedding相结合的想法受到了以前工作的启发，例如因子分解机[5]，它通过将两个变量之间的交互分解为两个 low-dimensional embedding 之间的点积来增加线性模型的泛化能力。在本文中，我们通过神经网络而不是点积来学习 embedding 之间的高度非线性相交互，从而加强了模型的能力。
>
> 在语言模型中，循环神经网络(RNN) 和具有 n-gram 特征的最大熵模型的联合训练被提出通过学习输入和输出之间的直接权重来显著降低 RNN 的复杂度(例如，隐藏层大小)[4]。在计算机视觉中，深度残差学习[2]已被用来降低训练更深层模型的难度，并通过跳过一层或多层的快捷连接来提高精度。神经网络和图形模型的联合训练也被应用于从图像中估计人体姿势[6]。在这项工作中，我们探索了前馈神经网络和线性模型的联合训练，稀疏特征和输出单元之间有直接的联系，用于稀疏输入数据的通用推荐和排序问题。
>
> 在推荐系统文献中，通过将内容信息的深度学习和评分矩阵的协同过滤(CF)结合起来，已经探索了 collaborativate deep learning[7]。之前也有关于移动应用推荐系统的工作，比如 AppJoy 在用户的应用使用记录中使用 CF[8]。
> 与以往基于 CF 或基于内容的方法不同，我们针对APP推荐系统的用户和 impression 数据联合训练 Wide&Deep 模型。

## 7. CONCLUSION

Memorization and generalization are both important for recommender systems. Wide linear models can effectively memorize sparse feature interactions using cross-product feature transformations, while deep neural networks can generalize to previously unseen feature interactions through lowdimensional embeddings. We presented the Wide & Deep learning framework to combine the strengths of both types of model. We productionized and evaluated the framework on the recommender system of Google Play, a massive-scale commercial app store. Online experiment results showed that the Wide & Deep model led to significant improvement on app acquisitions over wide-only and deep-only models.

> 对于推荐系统来说，记忆性和泛化性都是非常重要的。wide 线性模型可以通过 cross-product 特征变换有效地记忆稀疏特征交互，而深度神经网络可以通过 low dimensional embeddings 将稀疏特征交互推广到以前未曾见过的特征交互作用。为了结合这两种模型的优点，我们提出了 Wide&Deep 学习框架。我们在大型应用商店 Google Play 的推荐系统上对该框架进行了生产和评估。线上实验结果表明，Wide&Deep 模式比仅 Wide-Only 模式和 Depth-Only 模式在 APP 获取方面有显著改善。

## 8. REFERENCES

[1] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 12:2121–2159, July 2011. 

[2] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2016. 

[3] H. B. McMahan. Follow-the-regularized-leader and mirror descent: Equivalence theorems and l1 regularization. In Proc. AISTATS, 2011. 

[4] T. Mikolov, A. Deoras, D. Povey, L. Burget, and J. H. Cernocky. Strategies for training large scale neural network language models. In IEEE Automatic Speech Recognition & Understanding Workshop, 2011. 

[5] S. Rendle. Factorization machines with libFM. ACM Trans. Intell. Syst. Technol., 3(3):57:1–57:22, May 2012. 

[6] J. J. Tompson, A. Jain, Y. LeCun, and C. Bregler. Joint training of a convolutional network and a graphical model for human pose estimation. In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, NIPS, pages 1799–1807. 2014. 

[7] H. Wang, N. Wang, and D.-Y. Yeung. Collaborative deep learning for recommender systems. In Proc. KDD, pages 1235–1244, 2015. 

[8] B. Yan and G. Chen. AppJoy: Personalized mobile application discovery. In MobiSys, pages 113–126, 2011.