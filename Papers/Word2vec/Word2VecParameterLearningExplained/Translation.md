> 本文作者容新博士在2017年的飞机事故中丧生。

# word2vec Parameter Learning Explained

## Abstract

The word2vec model and application by Mikolov et al. have attracted a great amount of attention in recent two years. The vector representations of words learned by word2vec models have been shown to carry semantic meanings and are useful in various NLP tasks. As an increasing number of researchers would like to experiment with word2vec or similar techniques, I notice that there lacks a material that comprehensively explains the parameter learning process of word embedding models in details, thus preventing researchers that are non-experts in neural networks from understanding the working mechanism of such models.

This note provides detailed derivations and explanations of the parameter update equations of the word2vec models, including the original continuous bag-of-word (CBOW) and skip-gram (SG) models, as well as advanced optimization techniques, including hierarchical softmax and negative sampling. Intuitive interpretations of the gradient equations are also provided alongside mathematical derivations.

In the appendix, a review on the basics of neuron networks and backpropagation is provided. I also created an interactive demo, wevi, to facilitate the intuitive understanding of the model.1

> Mikolov等人的 word2vec 模型及其应用。近两年来引起了极大的关注。通过word2vec模型学习的单词的向量表征已经被证明具有语义，并且在各种 NLP 任务中发挥作用。随着越来越多的研究人员愿意尝试 word2vec 或类似的技术，我注意到目前还缺乏全面详细地解释 word embedding 模型的参数学习过程的材料，从而阻碍了非神经网络专家的研究人员理解这些模型的工作机制。
>
> 本问详细推导和解释了 word2vec模型的参数更新方程，包括原始的 continuous bag of word (CBOW)和 skip-gram(SG)模型，以及包括 HS 和 NS在内的高级优化技术。此外，还提供了梯度方程的直观解释以及数学推导。
>
> 在附录中，提供了有关神经元网络和反向传播基础知识的综述。 我还创建了一个交互式演示[wevi](https://ronxin.github.io/wevi/)，以促进对模型的直观理解。1

## 1 Continuous Bag-of-Word Model

#### 1.1 One-word context

We start from the simplest version of the continuous bag-of-word model (CBOW) introduced in Mikolov et al. (2013a). We assume that there is only one word considered per context, which means the model will predict one target word given one context word, which is like a bigram model. For readers who are new to neural networks, it is recommended that one go through Appendix A for a quick review of the important concepts and terminologies before proceeding further.

Figure 1 shows the network model under the simplified context definition2 . In our setting, the vocabulary size is $V$ , and the hidden layer size is $N$. The units on adjacent layers are fully connected. The input is a one-hot encoded vector, which means for a given input context word, only one out of $V$ units  ${x_1, \cdots  , x_V }$ will be $1$, and all other units are $0$.

The weights between the input layer and the output layer can be represented by a $V \times N$ matrix $W$. Each row of $W$ is the $N$-dimension vector representation $v_w$ of the associated word of the input layer. Formally, row $i$ of $W$ is $v^T_w$. Given a context (a word), assuming $x_k = 1$ and $x_{k^{'}} = 0$  for $k^{'} \ne k$, we have
$$
h = W^Tx = W^T_{(k,\cdot)} :=v_{w_I}^{T}, \qquad (1)
$$
which is essentially copying the $k$-th row of $W$ to $h$. $v_{w_I}$ is the vector representation of the input word $w_I$ . This implies that the link (activation) function of the hidden layer units is simply linear (i.e., directly passing its weighted sum of inputs to the next layer).

From the hidden layer to the output layer, there is a different weight matrix $W^{'} = {w^{'}_{ij}}$, which is an $N × V$ matrix. Using these weights, we can compute a score $u_j$ for each word in the vocabulary,
$$
u_j = {v_{w_j}^{'}}^{T}h, \qquad (2)
$$
where $v^{'}_{w_j}$ is the $j$-th column of the matrix $W^{'}$ . Then we can use softmax, a log-linear classification model, to obtain the posterior distribution of words, which is a multinomial distribution.
$$
p(w_j | w_I) = y_j = \frac{exp(u_j)}{\sum_{j^{'}=1}^{V} exp(u_{j^{'}})} \qquad (3)
$$
where $y_j$ is the output of the $j$-the unit in the output layer. Substituting (1) and (2) into (3), we obtain
$$
p(w_j|w_I) = \frac{exp(v_{w_j}^{'})}{\sum_{}^{} exp({v_{w_j^{'}}^{'}}^T v_{w_I})} \qquad (4)
$$
Note that $v_w$ and $v^{'}_w$ are two representations of the word $w$.  $v_w$ comes from rows of $W$, which is the input→hidden weight matrix, and $v^{'}_w$ comes from columns of $W^{'}$ , which is the hidden→output matrix. In subsequent analysis, we call $v_w$ as the “input vector”, and $v^{'}_w$ as the “output vector” of the word $w$.

![Fig1](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/Word2VecParameterLearningExplained/Fig1.png)

**Figure 1: A simple CBOW model with only one word in the context**

> 我们从Mikolov等人介绍的 bag-of-word model  (CBOW)的最简单版本开始。(2013A)。我们假设每个上下文只考虑一个单词，这意味着模型将在给定一个上下文单词的情况下预测一个目标单词，这类似于二元语法模型。对于刚接触神经网络的读者，建议您先阅读附录A，快速复习一下重要的概念和术语，然后再继续学习。
>
> 图1显示了简化上下文定义2下的网络模型。在我们的设置中，词汇表大小为$V$，隐藏层大小为 $N$。相邻层上的单元完全连接在一起。输入是一个one-hot向量，这意味着对于给定的输入上下文词，$V$单元 ${x_1，\cdots，x_V}$ 中只有一个将是 $1$，所有其他单元都是 $0$。
>
> 输入层和输出层之间的权重可以用 $V \times N$ 矩阵 $W$ 来表示。$W$ 的每行是输入层的关联词的 $N$ 维向量表示 $v_w$。形式上，$W$ 的第 $i$ 行是 $v^T_w$。给定一个上下文(一个单词)，假设 $x_k = 1$ 且 $x_{k^{'}} = 0$ 对于 $k^{'} \ne k$, 我们有
> $$
> h = W^Tx = W^T_{(k,\cdot)} :=v_{w_I}^{T}, \qquad (1)
> $$
> 这实质上是将 $W$ 的第 $k$ 行复制到 $h$ 。$v_{w_i}$是输入词 $w_i$ 的向量表征。这意味着隐藏层单元的链接(激活)函数简单地是线性的(即，将其输入的加权和直接传递到下一层)。
>
> 从隐藏层到输出层，有一个不同的权重矩阵$W^{‘}={w^{’}_{ij}}$，它是一个$N×V$矩阵。使用这些权重，我们可以为词典的每个单词计算分数$u_j$，
> $$
> u_j = {v_{w_j}^{'}}^{T}h, \qquad (2)
> $$
> 其中 $v^{'}_{w_j}$ 是矩阵 $W^{'}$ 的 第 $j$ 列。然后，我们可以使用 log-linear classification model Softmax来得到单词的后验分布，它是一个多项式分布。
> $$
> p(w_j | w_I) = y_j = \frac{exp(u_j)}{\sum_{j^{'}=1}^{V} exp(u_{j^{'}})} \qquad (3)
> $$
> 其中，$y_j$ 是输出层中单元 $j$ 的输出。将(1) (2)代入 (3) ，有
> $$
> p(w_j|w_I) = \frac{exp(v_{w_j}^{'})}{\sum_{}^{} exp({v_{w_j^{'}}^{'}}^T v_{w_I})} \qquad (4)
> $$
> 请注意，$v_w$ 和 $v^{'}_w$ 是单词 $w$ 的两个表征形式。$v_w$ 来自 $W$ 的行，$W$ 是  input layer → hidden layer 的权重矩阵，$v^{'}_w$ 来自 $W^{'}$ 的列，$W^{'}$ 是 hidden layer → output layer。在随后的分析中，我们将 $v_w$ 称为“输入向量”，将 $v^{'}_w$ 称为单词 $w$ 的“输出向量”。

#### Update equation for hidden→output weights

Let us now derive the weight update equation for this model. Although the actual computation is impractical (explained below), we are doing the derivation to gain insights on this original model with no tricks applied. For a review of basics of backpropagation, see Appendix A.

The training objective (for one training sample) is to maximize (4), the conditional probability of observing the actual output word $w_O$ (denote its index in the output layer as $j^{∗}$ ) given the input context word $w_I$ with regard to the weights.
$$
\begin{align*}
max\ p(w_O|w_I) &= max \ y_j^*
\qquad (5)
\\
&= max \ log \ y_{j^*} \qquad(6)
\\
&= u_{j^*} - log \ \sum_{j^{'}=1}^{V}exp(u_{j^{'}}) := -E, \qquad (7)

\end{align*}
$$
where $E = − log \ p(w_O|w_I)$ is our loss function (we want to minimize $E$ ), and $j^∗$ is the index of the actual output word in the output layer. Note that this loss function can be understood as a special case of the cross-entropy measurement between two probabilistic distributions.

Let us now derive the update equation of the weights between hidden and output layers. Take the derivative of $E$ with regard to $j$-th unit’s net input $u_j$ , we obtain
$$
\frac{\partial E}{\partial u_j} = y_j − t_j := e_j
\qquad (8)
$$
where $t_j = \mathop{1}(j = j^∗)$, i.e., $t_j$ will only be 1 when the j-th unit is the actual output word, otherwise $t_j = 0$. Note that this derivative is simply the prediction error $e_j$ of the output layer.

> 现在让我们推导出该模型的权重更新方程。虽然实际的计算是不切实际的(下面解释)，但我们进行推导是为了深入了解这个没有应用任何技巧的原始模型。有关反向传播的基础知识的回顾，请参见附录A。
>
> 训练目标(对于一个训练样本)是最大化(4)，给定输入上下文单词 $w_I$ 的权值，观察实际输出单词 $w_O$ (表示其在输出层的索引为 $j^{∗}$ )的条件概率。
> $$
> \begin{align*}
> max\ p(w_O|w_I) &= max \ y_j^*
> \qquad (5)
> \\
> &= max \ log \ y_{j^*} \qquad(6)
> \\
> &= u_{j^*} - log \ \sum_{j^{'}=1}^{V}exp(u_{j^{'}}) := -E, \qquad (7)
> 
> \end{align*}
> $$
> 其中 $E = −log \ p(w_O|w_i)$ 是我们的损失函数(我们希望最小化 $E$ )，$j^∗$ 是输出层中实际输出词的索引。注意，该损失函数可以理解为两个概率分布之间的交叉熵测量的特例。
>
> 现在让我们推导隐藏层和输出层之间权值的更新方程。求 $E$ 对 $j$ 个 单位网络输入 $u_j$ 的导数，得到
> $$
> \frac{\partial E}{\partial u_j} = y_j − t_j := e_j
> \qquad (8)
> $$
> 其中 $t_j=\mathop{1}(j=j^∗)$ ，即当第 j 个单位为实际输出词时，$t_j$ 将仅为 1，否则$t_j=0$。请注意，该导数只是输出层的预测误差$e_j$。

----------------------

## Acknowledgement

The author would like to thank Eytan Adar, Qiaozhu Mei, Jian Tang, Dragomir Radev, Daniel Pressel, Thomas Dean, Sudeep Gandhe, Peter Lau, Luheng He, Tomas Mikolov, Hao Jiang, and Oded Shmueli for discussions on the topic and/or improving the writing of the note.

## References

Goldberg, Y. and Levy, O. (2014). word2vec explained: deriving mikolov et al.’s negativesampling word-embedding method. arXiv:1402.3722 [cs, stat]. arXiv: 1402.3722.

Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013a). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.

Mnih, A. and Hinton, G. E. (2009). A scalable hierarchical distributed language model. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L., editors, Advances in Neural Information Processing Systems 21, pages 1081–1088. Curran Associates, Inc.

Morin, F. and Bengio, Y. (2005). Hierarchical probabilistic neural network language model. In AISTATS, volume 5, pages 246–252. Citeseer.

---------------------------

## A Back Propagation Basics

#### A.1 Learning Algorithms for a Single Unit

Figure 5 shows an artificial neuron (unit). ${x_1, \cdots , x_K}$ are input values; ${w_1,\cdots, w_K}$ are weights; $y$ is a scalar output; and $f$ is the link function (also called activation/decision/transfer function).

![Figure5](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/Word2VecParameterLearningExplained/Fig5.png)

**Figure 5: An artificial neuron**

The unit works in the following way:
$$
y = f(u) \qquad (62)
$$
where $u$ is a scalar number, which is the net input (or “new input”) of the neuron. $u$ is defined as
$$
u = \sum_{i=0}^{K} w_ix_i \qquad(63)
$$
Using vector notation, we can write
$$
u = w^Tx \qquad {(64)}
$$
Note that here we ignore the bias term in $u$. To include a bias term, one can simply add an input dimension (e.g., $x_0$) that is constant $1$.

Apparently, different link functions result in distinct behaviors of the neuron. We discuss two example choices of link functions here. 

The first example choice of $f(u)$ is the unit step function (aka Heaviside step function):
$$
f(u) = \begin{cases}
1 \quad if \ u > 0 \\
0 \quad otherwise
\end{cases}
\qquad (65)
$$
A neuron with this link function is called a perceptron. The learning algorithm for a perceptron is the perceptron algorithm. Its update equation is defined as:
$$
w^{(new)} = w^{(old)} - \eta \cdot (y - t) \cdot x \qquad (66)
$$
where $t$ is the label (gold standard) and $η$ is the learning rate $(η > 0)$. Note that a perceptron is a linear classifier, which means its description capacity can be very limited. If we want to fit more complex functions, we need to use a non-linear model.

The second example choice of $f(u)$ is the logistic function (a most common kind of sigmoid function), defined as
$$
σ(u) = \frac{1}{1+e^{-u}}
 \qquad (67)
$$
The logistic function has two primary good properties: (1) the output $y$ is always between $0$ and $1$, and (2) unlike a unit step function, σ(u) is smooth and differentiable, making the derivation of update equation very easy

Note that $σ(u)$ also has the following two properties that can be very convenient and will be used in our subsequent derivations:
$$
\sigma(−u) = 1 − \sigma (u) \qquad (68)
$$

$$
\frac{\partial \sigma(u)}{du} = \sigma(u)\sigma(-u)
\qquad (69)
$$

We use stochastic gradient descent as the learning algorithm of this model. In order to derive the update equation, we need to define the error function, i.e., the training objective. The following objective function seems to be convenient:
$$
E = \frac{1}{2}(t − y)^2
\qquad (70)
$$
We take the derivative of $E$ with regard to $w_i$
$$
\begin{align*}
\frac{\partial E}{\partial w_i} &= \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial w_i} \qquad (71)
\\
&= (y-t)\cdot y(1-y) \cdot x_i \qquad(72)
\end{align*}
$$
where $\frac{\partial y}{\partial u} = y(1 − y)$ because $y = f(u) = \sigma(u)$, and (68) and (69). Once we have the derivative, we can apply stochastic gradient descent:
$$
w^{(new)} = w^{(old)} − \eta · (y − t) \cdot y(1 − y) \cdot x
\qquad (73)
$$

> 图5 显示了一个人工神经元(单元)。${x_1，\cdots，x_K}$是输入值；${w_1，\cdots，w_K}$ 是权重；$y$ 是标量输出；$f$是链接函数(也称为激活/决策/传递函数)。
>
> 该单元的工作方式如下：
> $$
> y = f(u), \qquad (62)
> $$
> 其中 $u$ 是标量，它是神经元的输入(或“新输入”)。$u$ 定义为
> $$
> u = \sum_{i=0}^{K} w_ix_i \qquad(63)
> $$
> 使用向量表示法，我们可以
> $$
> u = w^Tx \qquad {(64)}
> $$
> 请注意，这里我们忽略了以 $u$ 为单位的偏移项。要包括偏置项，只需添加一个为常量 $1$ 的输入维度(例如 $x_0$)。
>
> 显然，不同的链接函数会导致神经元行为的不同。 我们在这里讨论链接函数的两个示例选择。
>
> $f(u)$ 的第一个示例选择是单位阶跃函数(也称为Heaviside阶跃函数)：
> $$
> f(u) = \begin{cases}
> 1 \quad if \ u > 0 \\
> 0 \quad otherwise
> \end{cases}
> \qquad (65)
> $$
> 具有这种连接函数的神经元被称为感知机。感知机的学习算法是感知机算法。其更新方程式定义为：
> $$
> w^{(new)} = w^{(old)} - \eta \cdot (y - t) \cdot x \qquad (66)
> $$
> 其中 $t$ 是标签(gold standard跟ground truth是一个意思)，$η$ 是学习率 $(η>0)$。请注意，感知机是线性分类器，这意味着它的描述能力可能非常有限。如果我们想要拟合更复杂的函数，我们需要使用非线性模型。
>
> 第二个选择 $f(u)$ 的示例是Logistic函数(一种最常见的Sigmoid函数)，定义为
> $$
> σ(u) = \frac{1}{1+e^{-u}}
>  \qquad (67)
> $$
> Logistic函数有两个主要的优良性质：(1)输出 $y$ 总是在 $0$ 和 $1$ 之间；(2)与单位阶跃函数不同，$σ(u)$ 是光滑和可微的，使得更新方程的推导非常容易。
>
> 请注意，$\sigma(u)$还具有以下两个属性，这两个属性非常方便，将在后续推导中使用：
> $$
> \sigma(−u) = 1 − \sigma (u) \qquad (68)
> $$
>
> $$
> \frac{\partial \sigma(u)}{du} = \sigma(u)\sigma(-u)
> \qquad (69)
> $$
>
> 我们使用随机梯度下降作为该模型的学习算法。为了推导更新方程，我们需要定义误差函数，即训练目标。下面的目标函数似乎很方便：
> $$
> E = \frac{1}{2}(t − y)^2
> \qquad (70)
> $$
> 我们取 $E$ 关于 $w_i$ 的导数
> $$
> \begin{align*}
> \frac{\partial E}{\partial w_i} &= \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial w_i} \qquad (71)
> \\
> &= (y-t)\cdot y(1-y) \cdot x_i \qquad(72)
> \end{align*}
> $$
> 其中 $\frac{\partial y}{\partial u}=y(1−y)$ ，因为 $y=f(u)=\sigma(u)$，并且(68)和(69)。一旦我们有了导数，我们就可以应用随机梯度下降：
> $$
> w^{(new)} = w^{(old)} − \eta · (y − t) \cdot y(1 − y) \cdot x
> \qquad (73)
> $$

#### A.2 Back-propagation with Multi-Layer Network

Figure 6 shows a multi-layer neural network with an input layer ${x_k} = {x_1, \cdots , x_K}$, a hidden layer ${h_i} = {h_1, \cdots , h_N}$, and an output layer ${y_j} = {y_1,\cdots, y_M}$. For clarity we use $k$, $i$, $j$ as the subscript for input, hidden, and output layer units respectively. We use $u_i$ and $u_{j}^{'}$ to denote the net input of hidden layer units and output layer units respectively.

We want to derive the update equation for learning the weights $w_{ki}$ between the input and hidden layers, and $w_{ij}^{'}$ between the hidden and output layers. We assume that all the computation units (i.e., units in the hidden layer and the output layer) use the logistic function $\sigma(u)$ as the link function. 

Therefore, for a unit $h_i$ in the hidden layer, its output is defined as
$$
h_i = \sigma(u_i) = \sigma (\sum_{k=1}^{K}w_{ki}x_k)
\qquad (74)
$$
Similarly, for a unit $y_j$ in the output layer, its output is defined as
$$
y_j = \sigma(u_j^{'}) = \sigma(\sum_{i=1}^{N}w_{ij}^{'}h_i) 
\qquad (75)
$$
We use the squared sum error function given by
$$
E(x, t, W, W^{'}) = \frac{1}{2} \sum_{j=1}^{M} (y_j − t_j)^2
, \qquad(76)
$$
where $W = \{w_{ki}\}$,  a $K \times N$ weight matrix (input-hidden), and $W^{'} = \{w_{ij}^{'}\}$, a $N \times M$ weight matrix (hidden-output). $t = {t_1,\cdots, t_M}$, a $M$-dimension vector, which is the gold-standard labels of output.

To obtain the update equations for $w_{ki}$ and $w^{'}_{ij}$ , we simply need to take the derivative of the error function $E$ with regard to the weights respectively. To make the derivation straightforward, we do start computing the derivative for the right-most layer (i.e., the output layer), and then move left. For each layer, we split the computation into three steps, computing the derivative of the error with regard to the output, net input, and weight respectively. This process is shown below.

![Fig6](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/Word2VecParameterLearningExplained/Fig6.png)

**Figure 6: A multi-layer neural network with one hidden layer**

>  图 6 显示了具有输入层 ${x_k}={x_1，\cdots，x_K}$，隐藏层 ${h_i}={h_1，\cdots，h_N}$，以及输出层 ${y_j}={y_1，\cdots，y_M}$的多层神经网络。为清楚起见，我们分别使用 $k$、 $i$、$j$ 作为输入层、隐藏层和输出层单元的下标。我们用 $u_i$和 $u_{j}^{'}$分别表示隐含层单元和输出层单元的输入。
>
> 我们想要推导出 学习输入层和隐藏层之间的权重$w_{ki}$以及隐藏层和输出层之间的权重$w_{ij}^{‘}$的更新公式。我们假设所有计算单元(即隐藏层和输出层中的单元)都使用logistic函数$\sigma(u)$作为链接函数。
>
> 因此，对于隐藏层中的一个  $h_i$ 单元，其输出定义为
> $$
> h_i = \sigma(u_i) = \sigma (\sum_{k=1}^{K}w_{ki}x_k)
> \qquad (74)
> $$
> 同样，对于输出层中的单元 $y_j$，其输出定义为
> $$
> y_j = \sigma(u_j^{'}) = \sigma(\sum_{i=1}^{N}w_{ij}^{'}h_i) 
> \qquad (75)
> $$
> 我们使用的平方和误差函数为
> $$
> E(x, t, W, W^{'}) = \frac{1}{2} \sum_{j=1}^{M} (y_j − t_j)^2
> , \qquad(76)
> $$
> 其中 $W=\{w_{ki}\}$ 是 $K \times N$ 权重矩阵(input -> hidden)，$W^{'}=\{w_{ij}^{'}\}$，$N \times M$ 的权重矩阵(hidden -> output)。$t={t_1，\cdots，t_M}$，一个 $M$  维向量，它是输出层的 gold-standard 标签。
>
> 要获得 $w_{ki}$ 和 $w_{ij}^{‘}$ 的更新方程，我们只需分别取误差函数 $E$ 关于权重的导数。为了使推导简洁明了，我们开始计算最右侧层(即输出层)的导数，然后向左移动。对于每一层，我们将计算分成三个步骤，分别计算误差关于输出、网络 输入和权重的导数。此过程如下所示。

We start with the output layer. The first step is to compute the derivative of the error w.r.t. the output:
$$
\frac{\partial E}{\partial y_j} = y_j - t_j \qquad (77)
$$
The second step is to compute the derivative of the error with regard to the net input of the output layer. Note that when taking derivatives with regard to something, we need to keep everything else fixed. Also note that this value is very important because it will be reused multiple times in subsequent computations. We denote it as $EI^{'}_j$ for simplicity.
$$
\frac{\partial E}{ \partial u_j^{'}} = \frac{\partial E}{\partial y_j} \cdot \frac{\partial y_j}{\partial u_{j}^{'}} = (y_j - t_j) \cdot y_j(1-y_j) := EI_j^{'} \qquad (78)
$$
The third step is to compute the derivative of the error with regard to the weight between the hidden layer and the output layer.
$$
\frac{\partial E}{\partial w_{ij}^{'}} = \frac{\partial E}{\partial u_{j}^{'}} \cdot \frac{\partial u_j^{'}}{\partial w_{ij}^{'}} = EI_{j}^{'} \cdot h_i \qquad (79)
$$
So far, we have obtained the update equation for weights between the hidden layer and the output layer.
$$
\begin{align*}
w_{ij}^{'(new)} &= w_{ij}^{'(old)} - \eta \cdot \frac{\partial E}{\partial w_{ij}^{'}}
\qquad (80)
\\
&= w_{ij}^{'(old)} -\eta \cdot EI_{j}^{'} \cdot h_i
\qquad (81)
\end{align*}
$$
where $\eta > 0$ is the learning rate.

We can repeat the same three steps to obtain the update equation for weights of the previous layer, which is essentially the idea of back propagation.

We repeat the first step and compute the derivative of the error with regard to the output of the hidden layer. Note that the output of the hidden layer is related to all units in the output layer.
$$
\frac{\partial E}{\partial h_i} = \sum_{j=1}^{M} \frac{\partial E}{\partial u_j^{'}} \frac{\partial u_j^{'}}{\partial h_i} = \sum_{j=1}^{M}EI_{j}^{'} \cdot w_{ij}^{'}

\qquad (82)
$$
Then we repeat the second step above to compute the derivative of the error with regard to the net input of the hidden layer. This value is again very important, and we denote it as $EI_i$.
$$
\frac{\partial E}{\partial u_i} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial u_i}= \sum_{j=1}^{M} EI^{'}_{j} \cdot w_{ij}^{'} \cdot h_i(1-h_i) := EI_i 
\qquad (83)
$$
Next we repeat the third step above to compute the derivative of the error with regard to the weights between the input layer and the hidden layer.
$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial u_i} \cdot \frac{\partial u_i}{\partial w_{ki}} = EI_i \cdot x_{k},
\qquad (84)
$$
Finally, we can obtain the update equation for weights between the input layer and the hidden layer.
$$
w_{ki}^{(new)} = w_{ki}^{(old)} − \eta \cdot EI_i \cdot x_k
\qquad(85)
$$
From the above example, we can see that the intermediate results ($EI^{'}_j$ ) when computing the derivatives for one layer can be reused for the previous layer. Imagine there were another layer prior to the input layer, then $EI_i$ can also be reused to continue computing the chain of derivatives efficiently. Compare Equations (78) and (83), we may find that in (83), the factor $\sum_{j=1}^{M}EI_{j}^{'}w_{ij}^{'}$ is just like the “error” of the hidden layer unit $h_i$ . We may interpret this term as the error “back-propagated” from the next layer, and this propagation may go back further if the network has more hidden layers.

> 我们从输出层开始。第一步是计算误差的导数关于输出：
> $$
> \frac{\partial E}{\partial y_j} = y_j - t_j \qquad (77)
> $$
> 第二步是计算误差相对于 output layer 输入的导数。请注意，当对某个变量进行导数计算时，我们需要保持其他变量都是固定的。另请注意，该值非常重要，因为它将在后续计算中多次重复使用。为简单起见，我们将其表示为$EI^{'}_j$。
> $$
> \frac{\partial E}{ \partial u_j^{'}} = \frac{\partial E}{\partial y_j} \cdot \frac{\partial y_j}{\partial u_{j}^{'}} = (y_j - t_j) \cdot y_j(1-y_j) := EI_j^{'} \qquad (78)
> $$
> 第三步是计算误差相对于隐藏层和输出层之间的权重的导数。
> $$
> \frac{\partial E}{\partial w_{ij}^{'}} = \frac{\partial E}{\partial u_{j}^{'}} \cdot \frac{\partial u_j^{'}}{\partial w_{ij}^{'}} = EI_{j}^{'} \cdot h_i \qquad (79)
> $$
> 到目前为止，我们已经得到了隐藏层和输出层之间的权重更新方程。
> $$
> \begin{align*}
> w_{ij}^{'(new)} &= w_{ij}^{'(old)} - \eta \cdot \frac{\partial E}{\partial w_{ij}^{'}}
> \qquad (80)
> \\
> &= w_{ij}^{'(old)} -\eta \cdot EI_{j}^{'} \cdot h_i
> \qquad (81)
> \end{align*}
> $$
> 其中 $\eta>0$ 是学习速率。
>
> 我们可以重复相同的三个步骤来获得上一层权重的更新方程，这本质上是反向传播的思想。
>
> 我们重复第一步，并计算误差相对于隐藏层输出的导数。请注意，隐藏层的输出与输出层中的所有单位相关。
> $$
> \frac{\partial E}{\partial h_i} = \sum_{j=1}^{M} \frac{\partial E}{\partial u_j^{'}} \frac{\partial u_j^{'}}{\partial h_i} = \sum_{j=1}^{M}EI_{j}^{'} \cdot w_{ij}^{'}
> 
> \qquad (82)
> $$
> 然后，我们重复上述第二步，以计算误差相对于隐藏层的净输入的导数。这个值同样非常重要，我们将其表示为$Ei_i$。
> $$
> \frac{\partial E}{\partial u_i} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial u_i}= \sum_{j=1}^{M} EI^{'}_{j} \cdot w_{ij}^{'} \cdot h_i(1-h_i) := EI_i 
> \qquad (83)
> $$
> 接下来，我们重复上面的第三步，以计算关于输入层和隐藏层之间的权重的误差的导数。
> $$
> \frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial u_i} \cdot \frac{\partial u_i}{\partial w_{ki}} = EI_i \cdot x_{k},
> \qquad (84)
> $$
> 最后，我们可以得到输入层和隐层之间的权重更新方程。
> $$
> w_{ki}^{(new)} = w_{ki}^{(old)} − \eta \cdot EI_i \cdot x_k
> \qquad(85)
> $$
> 从上面的例子可以看出，计算一层导数时的中间结果($EI^{'}_{j}$)可以重用于上一层。假设在输入层之前还有另一层，那么$EI_i$ 也可以被重用来继续高效地计算导数链。比较公式(78)和(83)，我们可以发现，因子 $\sum_{j=1}^{M}EI_{j}^{'}w_{ij}^{'}$ 就像隐藏层单元 $h_i$ 的“误差”。我们可以将此术语解释为来自下一层的 “反向传播” 误差，如果网络具有更多的隐藏层，则此传播可能会进一步追溯。

## B wevi: Word Embedding Visual Inspector

An interactive visual interface, wevi (word embedding visual inspector), is available online to demonstrate the working mechanism of the models described in this paper. See Figure 7 for a screenshot of wevi.

The demo allows the user to visually examine the movement of input vectors and output vectors as each training instance is consumed. The training process can be also run in batch mode (e.g., consuming 500 training instances in a row), which can reveal the emergence of patterns in the weight matrices and the corresponding word vectors. Principal component analysis (PCA) is employed to visualize the “high”-dimensional vectors in a 2D scatter plot. The demo supports both CBOW and skip-gram models.

After training the model, the user can manually activate one or multiple input-layer units, and inspect which hidden-layer units and output-layer units become active. The user can also customize training data, hidden layer size, and learning rate. Several preset training datasets are provided, which can generate different results that seem interesting, such as using a toy vocabulary to reproduce the famous word analogy: king - queen = man - woman.

It is hoped that by interacting with this demo one can quickly gain insights of the working mechanism of the model. The system is available at http://bit.ly/wevi-online. The source code is available at http://github.com/ronxin/wevi.

>  在线提供了一个交互式可视化界面 WEVI (Word Embedding Visual Inspector)来演示本文所描述的模型的工作机制。有关wevi的屏幕截图，请参见图7。
>
> 该 demo 允许用户在使用每个训练实例时直观地检查输入向量和输出向量的移动。训练过程还可以以批处理模式运行(例如，连续消耗500个训练实例)，这可以揭示权重矩阵和对应的词向量中出现的模式。主成分分析(PCA)被用来可视化二维散点图中的“高”维向量。该演示同时支持 CBOW 和 Skip-gram模型。
>
> 训练模型后，用户可以手动激活一个或多个输入层单元，并检查哪些隐藏层单元和输出层单元变为活动状态。用户还可以定制训练数据、隐藏层大小和学习率。提供了几个预设的训练数据集，它们可以生成看起来很有趣的不同结果，例如使用玩具词汇重现著名的单词类比：King-  Queen = Man - Women。
>
> 希望通过与此 demo 的互动，可以快速了解该模型的工作机制。

![Fig7](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/Word2VecParameterLearningExplained/Fig7.png)

**Figure 7: wevi screenshot (http://bit.ly/wevi-online)**

