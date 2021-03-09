> 本文作者容新博士在2017年的飞机事故中丧生。

# word2vec Parameter Learning Explained

## Abstract

The word2vec model and application by Mikolov et al. have attracted a great amount of attention in recent two years. The vector representations of words learned by word2vec models have been shown to carry semantic meanings and are useful in various NLP tasks. As an increasing number of researchers would like to experiment with word2vec or similar techniques, I notice that there lacks a material that comprehensively explains the parameter learning process of word embedding models in details, thus preventing researchers that are non-experts in neural networks from understanding the working mechanism of such models.

This note provides detailed derivations and explanations of the parameter update equations of the word2vec models, including the original continuous bag-of-word (CBOW) and skip-gram (SG) models, as well as advanced optimization techniques, including hierarchical softmax and negative sampling. Intuitive interpretations of the gradient equations are also provided alongside mathematical derivations.

In the appendix, a review on the basics of neuron networks and backpropagation is provided. I also created an interactive demo, wevi, to facilitate the intuitive understanding of the model.1

> Mikolov等人的 word2vec 模型及其应用。近两年来引起了极大的关注。通过word2vec模型学习的单词的向量表示已经被证明具有语义特性，并且在各种 NLP 任务中发挥作用。随着越来越多的研究人员愿意尝试 word2vec 或类似的技术，我注意到目前还缺乏全面详细地解释 word embedding 模型的参数学习过程的材料，从而阻碍了非神经网络专家的研究人员理解这些模型的工作机制。
>
> 本问详细推导和解释了 word2vec 模型的参数更新方程，包括原始的 continuous-bag-of-word (CBOW) 和 skip-gram (SG)模型，以及包括 hierarchical softmax 和  negative sampling 在内的高级优化技术。此外，还提供了梯度方程的直观解释以及数学推导。
>
> 在附录中，提供了有关神经元网络和反向传播基础知识的综述。 我还创建了一个交互式演示[wevi](https://ronxin.github.io/wevi/)，以促进对模型的直观理解。1

## 1 Continuous Bag-of-Word Model

#### 1.1 One-word context

We start from the simplest version of the continuous bag-of-word model (CBOW) introduced in Mikolov et al. (2013a). We assume that there is only one word considered per context, which means the model will predict one target word given one context word, which is like a bigram model. For readers who are new to neural networks, it is recommended that one go through Appendix A for a quick review of the important concepts and terminologies before proceeding further.

Figure 1 shows the network model under the simplified context definition(In Figures 1, 2, 3, and the rest of this note, $\textbf{W}^{'}$ is not the transpose of $\textbf{W}$, but a different matrix instead) . In our setting, the vocabulary size is $V$ , and the hidden layer size is $N$. The units on adjacent layers are fully connected. The input is a one-hot encoded vector, which means for a given input context word, only one out of $V$ units  $\{x_1, \cdots  , x_V \}$ will be $1$, and all other units are $0$.

The weights between the input layer and the output layer can be represented by a $V \times N$ matrix $\textbf{W}$. Each row of $\textbf{W}$ is the $N$-dimension vector representation $\textbf{v}_w$ of the associated word of the input layer. Formally, row $i$ of $\textbf{W}$ is $\textbf{v}^T_w$. Given a context (a word), assuming $x_k = 1$ and $x_{k^{'}} = 0$  for $k^{'} \ne k$, we have
$$
\textbf{h} = \textbf{W}^T\textbf{x} = \textbf{W}^T_{(k,\cdot)} :=\textbf{v}_{w_I}^{T}, \qquad (1)
$$
which is essentially copying the $k$-th row of $\textbf{W}$ to $\textbf{h}$. $\textbf{v}_{w_I}$ is the vector representation of the input word $w_I$ . This implies that the link (activation) function of the hidden layer units is simply linear (i.e., directly passing its weighted sum of inputs to the next layer).

From the hidden layer to the output layer, there is a different weight matrix $\textbf{W}^{'} = \{w^{'}_{ij}\}$, which is an $N × V$ matrix. Using these weights, we can compute a score $u_j$ for each word in the vocabulary,
$$
u_j = {\textbf{v}_{w_j}^{'}}^{T}\textbf{h}, \qquad (2)
$$
where $\textbf{v}^{'}_{w_j}$ is the $j$-th column of the matrix $\textbf{W}^{'}$ . Then we can use softmax, a log-linear classification model, to obtain the posterior distribution of words, which is a multinomial distribution.
$$
p(w_j | w_I) = y_j = \frac{exp(u_j)}{\sum_{j^{'}=1}^{V} exp(u_{j^{'}})} \qquad (3)
$$
where $y_j$ is the output of the $j$-the unit in the output layer. Substituting (1) and (2) into (3), we obtain
$$
p(w_j|w_I) = \frac{exp({\textbf{v}_{w_j}^{'}}^T \textbf{v}_{w_I})}{\sum_{j^{'}=1}^{V}exp({\textbf{v}_{w_{j^{'}}}^{'}}^T \textbf{v}_{w_I})} \qquad (4)
$$
Note that $\textbf{v}_w$ and $\textbf{v}^{'}_w$ are two representations of the word $w$.  $\textbf{v}_w$ comes from rows of $\textbf{W}$, which is the input→hidden weight matrix, and $\textbf{v}^{'}_w$ comes from columns of $\textbf{W}^{'}$ , which is the hidden→output matrix. In subsequent analysis, we call $\textbf{v}_w$ as the “input vector”, and $\textbf{v}^{'}_w$ as the **“output vector”** of the word $w$.

![Figure1](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig1.png)

**Figure 1: A simple CBOW model with only one word in the context**

> 我们从Mikolov等人介绍的 bag-of-word model  (CBOW)的最简单版本开始。(2013A)。我们假设每个上下文只考虑一个单词，这意味着模型将在给定一个上下文单词的情况下预测一个目标单词，这类似于 bigram 语法模型。对于刚接触神经网络的读者，建议您先阅读附录A，快速复习一下重要的概念和术语，然后再继续学习。
>
> 图1显示了简化上下文定义2下的网络模型。在我们的设置中，词汇表大小为$V$，隐藏层大小为 $N$。相邻层上的单元完全连接在一起。输入是一个 one-hot 向量，这意味着对于给定的输入上下文词，$V$ 单元 ${x_1，\cdots，x_V}$ 中只有一个将是 $1$，所有其他单元都是 $0$。
>
> 输入层和输出层之间的权重可以用 $V \times N$ 矩阵 $\textbf{W}$ 来表示。$\textbf{W}$ 的每行是输入层的关联词的 $N$ 维向量表示 $\textbf{v}_w$。形式上，$\textbf{W}$ 的第 $i$ 行是 $\textbf{v}^T_w$。给定一个上下文(一个单词)，假设 $x_k = 1$ 且 $x_{k^{'}} = 0$ , $k^{'} \ne k$, 我们有
> $$
> \textbf{h} = \textbf{W}^T\textbf{x} = \textbf{W}^T_{(k,\cdot)} :=\textbf{v}_{w_I}^{T}, \qquad (1)
> $$
> 这实质上是将 $\textbf{W}$ 的第 $k$ 行复制到 $\textbf{h}$ 。$\textbf{v}_{w_i}$是输入词 $w_i$ 的向量表表示。这意味着隐藏层单元的链接(激活)函数是简单地线性关系(即，将其输入的加权和直接传递到下一层)。
>
> 从隐藏层到输出层，有一个不同的权重矩阵$\textbf{W}^{'}={w^{'}_{ij}}$，它是一个$N×V$矩阵。使用这些权重，我们可以为词典的每个词计算分数$u_j$，
> $$
> u_j = {\textbf{v}_{w_j}^{'}}^{T}\textbf{h}, \qquad (2)
> $$
> 其中 $\textbf{v}^{'}_{w_j}$ 是矩阵 $\textbf{W}^{'}$ 的 第 $j$ 列。然后，我们可以使用 log-linear classification model-softmax来得到单词的后验分布，它是一个多项式分布。
> $$
> p(w_j | w_I) = y_j = \frac{exp(u_j)}{\sum_{j^{'}=1}^{V} exp(u_{j^{'}})} \qquad (3)
> $$
> 其中，$y_j$ 是输出层中单元 $j$ 的输出。将(1) (2)代入 (3) ，有
> $$
> p(w_j|w_I) = \frac{exp({\textbf{v}_{w_j}^{'}}^T \textbf{v}_{w_I})}{\sum_{j^{'}=1}^{V}exp({\textbf{v}_{w_{j^{'}}}^{'}}^T \textbf{v}_{w_I})} \qquad (4)
> $$
> 请注意，$\textbf{v}_w$ 和 $\textbf{v}^{'}_w$ 是词 $w$ 的两个向量表示形式。$\textbf{v}_w$ 来自 $\textbf{W}$ 的行，$\textbf{W}$ 是  input → hidden  的权重矩阵，$\textbf{v}^{'}_w$ 来自 $\textbf{W}^{'}$ 的列，$\textbf{W}^{'}$ 是 hidden → output 的权重矩阵。在随后的分析中，我们将 $\textbf{v}_w$ 称为**“输入向量”**，将 $\textbf{v}^{'}_w$ 称为词 $w$ 的**“输出向量”**。

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
where $t_j = \mathbb{1}(j = j^∗)$, i.e., $t_j$ will only be $1$ when the $j$-th unit is the actual output word, otherwise $t_j = 0$. Note that this derivative is simply the prediction error $e_j$ of the output layer.

Next we take the derivative on $w_{ij}^{'}$ to obtain the gradient on the hidden→output weights.
$$
\frac{\partial E}{\partial w_{ij}^{'}} = \frac{\partial E}{\partial u_j} \cdot\frac{\partial u_j}{\partial w_{ij}^{'}} = e_j \cdot h_i \qquad (9)
$$
Therefore, using stochastic gradient descent, we obtain the weight updating equation for hidden → output weights:
$$
{w_{ij}^{'}}^{(new)} = {w_{ij}^{'}}^{(old)} - \eta \cdot e_j \cdot h_i. 
\qquad (10)
$$
or
$$
{\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} - \eta \cdot e_j \cdot \textbf{h} \qquad for \ j=1,2,\cdots,V. \qquad (11)
$$
where $\eta > 0$ is the learning rate, $e_j = y_j − t_j$ , and $h_i$ is the $i$-th unit in the hidden layer; $\textbf{v}^{'}_{w_j}$ is the output vector of $w_j$ . Note that this update equation implies that we have to go through every possible word in the vocabulary,  check its output probability $y_j$ , and compare $y_j$ with its expected output $t_j$ (either $0$ or $1$). 

- If  $y_j > t_j$  (“overestimating”), then we subtract a proportion of the hidden vector $\textbf{h}$ (i.e., $v_{w_I}$ ) from $\textbf{v}^{'}_{w_j}$ , thus making $\textbf{v}^{'}_{w_j}$ farther away from $\textbf{v}_{w_I}$ ;
- If $y_j < t_j$ (“underestimating”, which is true only if $t_j = 1$, i.e., $w_j = w_O$), we add some $\textbf{h}$ to $\textbf{v}^{'}_{w_O}$ , thus making $\textbf{v}^{'}_{w_O}$ closer3(Here when I say “closer” or “farther”, I meant using the inner product instead of Euclidean as the distance measurement.) to $\textbf{v}_{w_I}$ ;
- If $y_j$ is very close to $t_j$ , then according to the update equation, very little change will be made to the weights. Note, again, that $\textbf{v}_w$ (input vector) and $\textbf{v}^{'}_{w}$ (output vector) are two different vector representations of the word $w$ ;

> 现在让我们推导出该模型的权重更新方程。虽然真的去计算的话是不切实际的(下面解释)，但我们进行推导是为了深入了解这个没有应用任何优化技巧的原始模型。有关反向传播的基础知识的回顾，请参见附录A。
>
> 训练目标(对于一个训练样本)是最大化(4)，给定输入上下文单词 $w_I$ 的权值，观察实际输出单词 $w_O$ (其在输出层的索引为 $j^{∗}$ )的条件概率。
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
> 其中 $t_j=\mathop{1} (j=j^∗)$ ，即当第 $j$ 个单位为实际输出词时，$t_j$ 将仅为 $1$，否则$t_j=0$。请注意，该导数只是输出层的预测误差 $e_j$ 。
>
> 接下来，我们对 $w_{ij}^{'}$ 求导，以获得 hidden → output 权重的梯度。
> $$
> \frac{\partial E}{\partial w_{ij}^{'}} = \frac{\partial E}{\partial u_j} \cdot\frac{\partial u_j}{\partial w_{ij}^{'}} = e_j \cdot h_i \qquad (9)
> $$
> 因此，利用随机梯度下降，我们得到了 hidden →output 权重的权重更新方程：
> $$
> {w_{ij}^{'}}^{(new)} = {w_{ij}^{'}}^{(old)} - \eta \cdot e_j \cdot h_i. 
> \qquad (10)
> $$
> 或者
> $$
> {\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} - \eta \cdot e_j \cdot \textbf{h} \qquad for \ j=1,2,\cdots,V. \qquad (11)
> $$
> 其中 $\eta>0$ 是学习率，$e_j=y_j−t_j$，$h_i$ 是隐藏层中的 $i$ 个单位；$\textbf{v}^{'}_{w_j}$ 是 $w_j$ 的输出层向量。注意，这个更新公式意味着我们必须遍历词汇表中的每个可能的单词，检查其输出概率 $y_j$ ，并将 $y_j$ 与其预期输出 $t_j$ ( $0$ 或 $1$ )进行比较。
>
> - 如果 $y_j > t_j$ (“overestimating”)，则从 $\textbf{v}^{'}_{w_j}$ 中减去一定比例的隐藏层向量 $\textbf{h}$ (即 $\textbf{v}_{w_i}$ )，从而使 $\textbf{v}^{'}_{w_j}$ 远离 $\textbf{v}_{w_i}$ ；
> - 如果 $y_j<t_j$ (“underestimating”，只有当 $t_j=1$，即 $w_j=w_O$ 时才是正确的)，我们将一定比例的 $\textbf{h}$ 加到 $\textbf{v}^{'}_{w_O}$ 上，从而使 $\textbf{v}^{’}_{w_O}$ 接近 $\textbf{v}_{w_i}$ 。
> - 如果 $y_j$ 与 $t_j$ 非常接近，则根据更新公式，权重几乎不会发生变化。再次注意，$\textbf{v}_w$ (输入向量)和 $\textbf{v}^{'}_{w}$ (输出向量)是单词 $w$ 的两个不同的向量表示。
>
> 这里，当我说“更近”或“更远”时，我的意思是用内积而不是欧几里得作为距离的度量。

## Update equation for input→hidden weights

Having obtained the update equations for $\textbf{W}^{'}$ , we can now move on to $\textbf{W}$. We take the derivative of $E$ on the output of the hidden layer, obtaining
$$
\frac{\partial{E}}{\partial{h_i}} = \sum^{V}_{j=1} \frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial h_i}= \sum_{j=1}^{V}e_j \cdot w_{ij}^{'} :=EH_i
\qquad(12)
$$
where $h_i$ is the output of the $i$-th unit of the hidden layer; $u_j$ is defined in (2), the net input of the $j$-th unit in the output layer; and $e_j = y_j − t_j$ is the prediction error of the $j$-th word in the output layer. $EH$, an $N$-dim vector, is the sum of the output vectors of all words in the vocabulary, weighted by their prediction error.

Next we should take the derivative of $E$ on $\textbf{W}$. First, recall that the hidden layer performs a linear computation on the values from the input layer. Expanding the vector notation in (1) we get
$$
h_i = \sum_{k=1}^{V} x_k \cdot w_{ki} \qquad (13)
$$
Now we can take the derivative of $E$ with regard to each element of $\textbf{W}$, obtaining
$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial w_{ki}} = EH_i \cdot x_k
\qquad (14)
$$
This is equivalent to the tensor product of $\textbf{x}$ and $EH$, i.e.,
$$
\frac{\partial E}{\partial W} = \textbf{x} \bigotimes EH = \textbf{x}EH^{T} 
\qquad(15)
$$
from which we obtain a $V × N$ matrix. Since only one component of $\textbf{x}$ is non-zero, only one row of $\frac{\partial E}{\partial \textbf{W}}$ is non-zero, and the value of that row is $EH^T$ , an $N$-dim vector. We obtain the update equation of $\textbf{W}$ as
$$
\textbf{v}_{w_I}^{(new)} = \textbf{v}_{w_I}^{(old)} - \eta EH^T \qquad (16)
$$
where $\textbf{v}_{w_I}$ is a row of $\textbf{W}$, the “input vector” of the only context word, and is the only row of $\textbf{W}$ whose derivative is non-zero. All the other rows of $\textbf{W}$ will remain unchanged after this iteration, because their derivatives are zero.

Intuitively, since vector $EH$ is the sum of output vectors of all words in vocabulary weighted by their prediction error $e_j = y_j −t_j$ , we can understand (16) as adding a portion of every output vector in vocabulary to the input vector of the context word. If, in the output layer, the probability of a word $w_j$ being the output word is overestimated ($y_j > t_j$), then the input vector of the context word $w_I$ will tend to move farther away from the output vector of $w_j$ ; conversely if the probability of $w_j$ being the output word is underestimated ($y_j < t_j$), then the input vector $w_I$ will tend to move closer to the output vector of $w_j$ ; if the probability of $w_j$ is fairly accurately predicted, then it will have little effect on the movement of the input vector of $w_I$ . The movement of the input vector of $w_I$ is determined by the prediction error of all vectors in the vocabulary; the larger the prediction error, the more significant effects a word will exert on the movement on the input vector of the context word.

As we iteratively update the model parameters by going through context-target word pairs generated from a training corpus, the effects on the vectors will accumulate. We can imagine that the output vector of a word $w$ is “dragged” back-and-forth by the input vectors of $w$’s co-occurring neighbors, as if there are physical strings between the vector of $w$ and the vectors of its neighbors. Similarly, an input vector can also be considered as being dragged by many output vectors. This interpretation can remind us of gravity, or force-directed graph layout. The equilibrium length of each imaginary string is related to the strength of cooccurrence between the associated pair of words, as well as the learning rate. After many iterations, the relative positions of the input and output vectors will eventually stabilize.

> 获得 $\textbf{W}^{'}$ 的更新方程后，我们现在开始转到 $\textbf{W}$。我们对 $E$ 求对隐藏层的输出的导数，得到
> $$
> \frac{\partial{E}}{\partial{h_i}} = \sum^{V}_{j=1} \frac{\partial E}{\partial u_j} \cdot \frac{\partial u_j}{\partial h_i}= \sum_{j=1}^{V}e_j \cdot w_{ij}^{'} :=EH_i
> \qquad(12)
> $$
> 其中 $h_i$ 是隐藏层的 $i$ 个单元的输出；$u_j$ 在 (2) 中定义，是输出层中 $j$ 个单元的输入；$e_j=y_j−t_j$ 是输出层中 $j$ 个词的预测误差。**$EH$ 是一个 $N$ 维向量，是词典中所有词的输出向量按它们的预测误差加权之和。**
>
> 接下来，我们应该求 $E$ 对 $\textbf{W}$ 的导数。首先，回想一下，隐藏层是输入层的线性计算。展开(1)中的向量，我们得到
> $$
> h_i = \sum_{k=1}^{V} x_k \cdot w_{ki} \qquad (13)
> $$
> 现在我们可以求 $E$ 对 $\textbf{W}$ 的每个元素的导数，得到
> $$
> \frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial w_{ki}} = EH_i \cdot x_k
> \qquad (14)
> $$
> 这相当于 $\textbf{x}$ 和 $EH$ 的张量积，即,
> $$
> \frac{\partial E}{\partial \textbf{W}} = \textbf{x} \bigotimes EH = \textbf{x}EH^{T} 
> \qquad(15)
> $$
> 由此我们得到一个 $V×N$ 矩阵。由于 $\textbf{x}$ 只有一个分量是非零的，所以只有 $\frac{\partial E}{\partial \textbf{W}}$ 的一行非零，该行的值是 $EH^T$ ，即 $N$ - dim向量。我们得到了 $\textbf{W}$ 的更新方程
> $$
> \textbf{v}_{w_I}^{(new)} = \textbf{v}_{w_I}^{(old)} - \eta EH^T \qquad (16)
> $$
> 其中 $\textbf{v}_{w_i}$ 是 $\textbf{W}$ 的行，即唯一 context 词的“输入向量”，并且是导数不为零的 $\textbf{W}$ 的唯一行。在此迭代之后， $\textbf{W}$ 的所有其他行将保持不变，因为它们的导数为零。
>
> 直观地，由于**向量 $EH$ 是词典中所有词的输出向量经其预测误差 $e_j = y_j−t_j$ 加权后的总和**，我们可以理解为 (16) 将词典中每个输出向量的一部分加到上下文单词的输入向量上。如果在输出层中，一个词 $w_j$ 作为输出词的概率被 overestimating $(y_j > t_j)$，那么上下文单词 $w_I$ 的输入向量将倾向于远离 $w_j$ 的输出向量; 反之，如果 underestimating 了 $w_j$ 作为输出词的概率 $(y_j < t_j)$ ，则输入向量 $w_I$ 将趋向于接近 $w_j$ 的输出向量; 如果对 $w_j$ 的概率预测得比较准确，则对 $w_I$ 的输入向量的移动影响不大。**$w_I$ 的输入向量的移动由词汇表中所有向量的预测误差决定; 预测误差越大，词对上下文词输入向量的运动影响就越大。**
>
> 由于在训练过程中，我们是通过迭代训练语料生成的 context-target词对来更新模型参数，每次迭代更新对向量的影响也是累积的。我们可以想象成词 $w$ 的输出向量被 $w$ 的共现邻居的输入向量的来回往复的拖拽。就好比有真实的弦在词 $w$ 和其邻居词之间。同样的，输入向量也可以被想象成被很多输出向量拖拽。这种解释可以提醒我们想象成一个重力，或者其他力导向的图的布局。每个假想的弦的平衡长度与相关单词对之间同现的强度以及学习率有关。经过多次迭代，输入和输出向量的相对位置最终将稳定下来。

#### 1.2 Multi-word context

![Figure2](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig2.png)

**Figure 2: Continuous bag-of-word model**

Figure 2 shows the CBOW model with a multi-word context setting. When computing the hidden layer output, instead of directly copying the input vector of the input context word, the CBOW model takes the average of the vectors of the input context words, and use the product of the input→hidden weight matrix and the average vector as the output.
$$
\begin{align*}
\textbf{h} 
&= \frac{1}{C}\textbf{W}^T(\textbf{x}_1 + \textbf{x}_2 + \cdots + \textbf{x}_C)
\qquad(17)
\\
&= \frac{1}{C}(\textbf{v}_{w_1} + \textbf{v}_{w_2} + \cdots + \textbf{v}_{w_C})^T
\qquad(18)
\end{align*}
$$
where $C$ is the number of words in the context, $w_1, \cdots , w_C$ are the words the in the context, and $\textbf{v}_w$ is the input vector of a word $w$. 

The loss function is
$$
\begin{align*}
E 
&= -log \ p(w_O |w_{I,1},\cdots,w_{I,C})
\qquad (19)
\\
&= -u_{j^*} \ + \ log \ \sum_{j^{'} = 1}^{V}exp(u_{j^{'}})
\qquad (20)
\\
&= {\textbf{-v}_{w_O}^{'}}^T \cdot \textbf{h} \ + \ log \ \sum_{j^{'} = 1}^{V}exp({\textbf{v}_{w_j}^{'}}^T \cdot \textbf{h})
\qquad (21)
\\
\end{align*}
$$
which is the same as (7), the objective of the one-word-context model, except that $\textbf{h}$ is different, as defined in (18) instead of (1).

The update equation for the hidden → output weights stay the same as that for the one-word-context model (11). We copy it here:
$$
{\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} - \eta \cdot e_j \cdot \textbf{h} \qquad for \ j=1,2,\cdots,V. 
\qquad (22)
$$
Note that we need to apply this to every element of the hidden→output weight matrix for each training instance.

The update equation for input→hidden weights is similar to (16), except that now we need to apply the following equation for every word $w_{I,c}$ in the context:
$$
\textbf{v}_{w_{I,c}}^{(new)} = \textbf{v}_{w_{I,c}}^{(old)} - \frac{1}{C} \cdot \eta \cdot EH^T 
\qquad 
for \ c = 1,2,\cdots,C. 
\qquad (23)
$$
where $\textbf{v}_{w_{I,c}}$ is the input vector of the $c$-th word in the input context; $\eta$ is a positive learning rate; and $EH = \frac{\partial E}{\partial h_i}$ is given by (12). The intuitive understanding of this update equation is the same as that for (16).

> Figure2 显示了具有 multi-word-context 设置的 CBOW 模型。在计算隐藏层输出时，CBOW 模型不是直接复制输入上下文词的输入向量，而是取输入上下文词向量的平均值，并用 input →hidden 的权值矩阵与平均向量的乘积作为输出。
> $$
> \begin{align*}
> \textbf{h} 
> &= \frac{1}{C}\textbf{W}^T(\textbf{x}_1 + \textbf{x}_2 + \cdots + \textbf{x}_C)
> \qquad(17)
> \\
> &= \frac{1}{C}(\textbf{v}_{w_1} + \textbf{v}_{w_2} + \cdots + \textbf{v}_{w_C})^T
> \qquad(18)
> \end{align*}
> $$
> 其中 $C$ 是上下文中的单词数，$w_1,\cdots,w_C$ 是上下文中的单词， $\textbf{v}_w$ 是单词 $w$ 的输入向量。损失函数为
> $$
> \begin{align*}
> E 
> &= -log \ p(w_O |w_{I,1},\cdots,w_{I,C})
> \qquad (19)
> \\
> &= -u_{j^*} \ + \ log \ \sum_{j^{'} = 1}^{V}exp(u_{j^{'}})
> \qquad (20)
> \\
> &= {\textbf{-v}_{w_O}^{'}}^T \cdot \textbf{h} \ + \ log \ \sum_{j^{'} = 1}^{V}exp({\textbf{v}_{w_j}^{'}}^T \cdot \textbf{h})
> \qquad (21)
> \\
> \end{align*}
> $$
> 它与 one-word-context 的目标函数(7)相同，只是 $\textbf{h}$ 不同，$\textbf{h}$ 如(18)中定义的，而不是(1)中定义的。
>
> hidden → output 的权重的更新公式与 one-word-context 模型 (11) 相同。我们将其复制到此处：
> $$
> {\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} - \eta \cdot e_j \cdot \textbf{h} \qquad for \ j=1,2,\cdots,V. 
> \qquad (22)
> $$
> 请注意，我们需要将其应用于每个训练实例的 hidden→output 权重矩阵的每个元素。
>
> input → hidden 权重的更新公式类似于(16)，不同之处在于现在我们需要对上下文中的每个单词 $w_{I,c}$ 应用以下公式
> $$
> \textbf{v}_{w_{I,c}}^{(new)} = \textbf{v}_{w_{I,c}}^{(old)} - \frac{1}{C} \cdot \eta \cdot EH^T 
> \qquad 
> for \ c = 1,2,\cdots,C. 
> \qquad (23)
> $$
> 其中 $\textbf{v}_{w_{I,c}}$ 是输入上下文中 的第 $c$ 个单词输入向量; $\eta$ 为正学习率; $EH = \frac{\partial E}{\partial h_i}$ 由(12)给出。这个更新方程的直观理解与(16)相同。

## 2 Skip-Gram Model

![Figure3](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig3.png)

**Figure 3: The skip-gram model.**

The skip-gram model is introduced in Mikolov et al. (2013a,b). Figure 3 shows the skipgram model. It is the opposite of the CBOW model. The target word is now at the input layer, and the context words are on the output layer.

We still use $\textbf{v}_{w_I}$ to denote the input vector of the only word on the input layer, and thus we have the same definition of the hidden-layer outputs $\textbf{h}$ as in (1), which means $\textbf{h}$ is simply copying (and transposing) a row of the input→hidden weight matrix, $\textbf{W}$, associated with the input word $w_I$ . We copy the definition of $\textbf{h}$ below:
$$
\textbf{h} = \textbf{W}^T_{(k,\cdot)}:= \textbf{v}_{w_I}^{T}, 
\qquad (24)
$$
On the output layer, instead of outputing one multinomial distribution, we are outputing $C$ multinomial distributions. Each output is computed using the same hidden→output matrix:
$$
p(w_{c,j}=w_{O,c}|w_I) = y_{c,j} = \frac{exp(u_{c,j})}{\sum_{j^{'}=1}^{V}exp(u_{j^{'}})}
\qquad (25) 
$$
where $w_{c,j}$ is the $j$-th word on the $c$-th panel of the output layer; $w_{O,c}$ is the actual $c$-th word in the output context words; $w_I$ is the only input word; $y_{c,j}$ is the output of the $j$-th unit on the $c$-th panel of the output layer; $u_{c,j}$ is the net input of the $j$-th unit on the $c$-th panel of the output layer. Because the output layer panels share the same weights, thus
$$
u_{c,j}=u_j={\textbf{v}_{w_j}^{'}}^T \cdot \textbf{h}, \quad for \ c=1,2,\cdots,C
\qquad(26)
$$
where $\textbf{v}^{'}_{w_j}$ is the output vector of the $j$-th word in the vocabulary, $w_j$ , and also $\textbf{v}^{'}_{w_j}$ is taken from a column of the hidden→output weight matrix,  $\textbf{W}^{'}$.

The derivation of parameter update equations is not so different from the one-word-context model. The loss function is changed to
$$
\begin{align*}
E
&= -log \ p(w_{O,1},w_{O,2}, \cdots , w_{O,C} | w_I)
\qquad (27)
\\
&= -log \prod_{c=1}^{C} \frac{exp(u_{c,j_{c}^{*}})}{\sum_{j^{'}}^{V}exp(u_{j^{'}})}
\qquad (28)
\\
&= -\sum_{c=1}^{C}u_{j_c^*} + C \cdot log \sum_{j^{'} = 1}^{V}exp(u_{j^{'}})
\qquad (29)
\end{align*}
$$
where $j^∗_c$ is the index of the actual $c$-th output context word in the vocabulary.

We take the derivative of $E$ with regard to the net input of every unit on every panel of the output layer, $u_{c,j}$ and obtain
$$
\frac{\partial E}{\partial u_{c,j}} = y_{c,j} - t_{c,j} :=e_{c,j} 
\qquad (30)
$$
which is the prediction error on the unit, the same as in (8). For notation simplicity, we define a $V$ -dimensional vector $EI = {EI_1, \cdots ,EI_V }$ as the sum of prediction errors over all context words:
$$
EI_j = \sum_{c=1}^{C}e_{c,j} \qquad(31)
$$
Next, we take the derivative of $E$ with regard to the hidden→output matrix $\textbf{W}^{'}$ , and obtain
$$
\frac{\partial E}{\partial w_{ij}^{'}} = \sum_{c=1}^{C} \frac{\partial E}{\partial u_{c,j}} \cdot \frac{\partial u_{c,j}}{\partial w_{ij}^{'}} = EI_j \cdot h_i 
\qquad(32)
$$
Thus we obtain the update equation for the hidden→output matrix $\textbf{W}^{'}$，
$$
{w_{ij}^{'}}^{(new)} = {w_{ij}^{'}}^{(old)} - \eta \cdot EI_j \cdot h_i \qquad (33)
$$
or
$$
{\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} -  \eta \cdot EI_j \cdot \textbf{h} \qquad for \ j =1,2, \cdots , V. \qquad(34)
$$

The intuitive understanding of this update equation is the same as that for (11), except that the prediction error is summed across all context words in the output layer. Note that we need to apply this update equation for every element of the hidden→output matrix for each training instance.

The derivation of the update equation for the input→hidden matrix is identical to (12) to (16), except taking into account that the prediction error $e_j$ is replaced with $EI_j$ . We directly give the update equation:
$$
\textbf{v}_{w_I}^{(new)} = \textbf{v}_{w_I}^{(old)} - \eta \cdot EH^T \qquad (35)
$$
where $EH$ is an $N$-dim vector, each component of which is defined as
$$
EH_i = \sum_{j=1}^{V}EI_j \cdot w_{ij}^{'} \qquad (36)
$$

The intuitive understanding of (35) is the same as that for (16).

> Mikolov等人提出了 skip-gram 模型。(2013a,b)。Figure 3 显示了 skip-gram模型。与 CBOW 模型相反。目标词(即中心词)现在位于输入层，上下文词位于输出层。
>
> 我们仍然使用 $\textbf{v}_{w_I}$ 来表示输入层上唯一单词的输入向量，因此我们具有与(1)中相同的隐藏层输出 $\textbf{h}$ 的定义，这意味着 $\textbf{h}$ 只是复制(并转置)与输入单词 $w_I$ 相关联的 input→hidden 权重矩阵 $\textbf{W}$ 的行。我们将 $\textbf{h}$ 的定义复制到下面：
> $$
> \textbf{h} = \textbf{W}^T_{(k,\cdot)}:= \textbf{v}_{w_I}^{T}, 
> \qquad (24)
> $$
> 在输出层，我们不是输出一个多项分布，而是输出 $C$ 个多项分布。每个输出都使用相同的 hidden→output 矩阵来计算:
> $$
> p(w_{c,j}=w_{O,c}|w_I) = y_{c,j} = \frac{exp(u_{c,j})}{\sum_{j^{'}=1}^{V}exp(u_{j^{'}})}
> \qquad (25)
> $$
> 其中，$w_{c,j}$ 是相对于第 $c$ 个输出层的第 $j$ 个单词；$w_{O,c}$ 是相对于输出上下文词中的第 $c$ 个单词；$w_I$ 是唯一的输入词；$y_{c,j}$ 是第 $c$ 个输出层上的第 $j$ 个单元的输出；$u_{c,j}$ 是第 $c$ 个 输出层的第 $j$ 个单元的输入。
>
> (注：原文用的是 panel , 对应的应该是图里的 C 个 panel, 我觉得这里意思应该是在正负样本的角度会有 $C$ 组，即 context 的长度为 $C$ , 其中 $j$ 对应的是相对于正负样本集合中的第 $j$ 个词，这里的正负样本集合指的是词典中的所有词)
>
> 由于所有输出层共享相同的权重矩阵，因此
> $$
> u_{c,j}=u_j={\textbf{v}_{w_j}^{'}}^T \cdot \textbf{h}, \quad for \ c=1,2,\cdots,C
> \qquad(26)
> $$
> 其中 $\textbf{v}^{’}_{w_j}$ 是词典中第 $j$ 个词的输出向量 $w_j$ ，并且 $\textbf{v}^{’}_{w_j}$ 取自 hidden → output 权重矩阵 $\textbf{W}^{'}$ 的列。
>
> 参数更新方程的推导与 one-word-context 模型没有太大不同。损失函数更改为
> $$
> \begin{align*}
> E
> &= -log \ p(w_{O,1},w_{O,2}, \cdots , w_{O,C} | w_I)
> \qquad (27)
> \\
> &= -log \prod_{c=1}^{C} \frac{exp(u_{c,j_{c}^{*}})}{\sum_{j^{'}}^{V}exp(u_{j^{'}})}
> \qquad (28)
> \\
> &= -\sum_{c=1}^{C}u_{j_c^*} + C \cdot log \sum_{j^{'} = 1}^{V}exp(u_{j^{'}})
> \qquad (29)
> \end{align*}
> $$
> 其中 $j^∗_c$ 上下文中第 $c$ 个词(注：正样本)在词典中的索引。
>
> 我们求 $E$ 对 $u_{c,j}$ 的偏导(注：是$u_{c,j}$，不是 $u_{j_{c}^{*}}$),得到：
> $$
> \frac{\partial E}{\partial u_{c,j}} = y_{c,j} - t_{c,j} :=e_{c,j} 
> \qquad (30)
> $$
> 跟公式(8)一致，这是一个单元上的预测误差。为方便表示，我们将 $V$ 维向量 $EI={EI_1，\cdots，EI_V}$ , 向量是预测了$C$ 个预测词的误差总和：
> $$
> EI_j = \sum_{c=1}^{C}e_{c,j} \qquad(31)
> $$
> 接下来，我们求 $E$ 对 hidden -> output 的权重矩阵 $\textbf{W}^{'}$ 的偏导，可得：
> $$
> \frac{\partial E}{\partial w_{ij}^{'}} = \sum_{c=1}^{C} \frac{\partial E}{\partial u_{c,j}} \cdot \frac{\partial u_{c,j}}{\partial w_{ij}^{'}} = EI_j \cdot h_i 
> \qquad(32)
> $$
> 因此，我们得到了 hidden → ouput 矩阵 $\textbf{W}^{'}$ 的更新方程
> $$
> {w_{ij}^{'}}^{(new)} = {w_{ij}^{'}}^{(old)} - \eta \cdot EI_j \cdot h_i \qquad (33)
> $$
> 或者
> $$
> {\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} -  \eta \cdot EI_j \cdot \textbf{h} \qquad for \ j =1,2, \cdots , V. \qquad(34)
> $$
>
> 对这个更新方程的直观理解与 (11) 相同，除了预测误差是 $C$ 个输出层的总和。注意，我们需要对每个训练实例的 hidden → output 矩阵的每个元素应用这个更新方程。
>
> 除了预测误差 $e_j$ 被 $EI_j$ 替换之外，input → hidden 矩阵的更新方程的推导与(12)至(16)相同。我们直接给出更新公式：
> $$
> \textbf{v}_{w_I}^{(new)} = \textbf{v}_{w_I}^{(old)} - \eta \cdot EH^T \qquad (35)
> $$
> 其中 $EH$ 是 $N$ 维向量，其每个分量定义为
> $$
> EH_i = \sum_{j=1}^{V}EI_j \cdot w_{ij}^{'} \qquad (36)
> $$
> (35) 的直观理解与 (16) 相同。

## 3 Optimizing Computational Efficiency

So far the models we have discussed (“bigram” model, CBOW and skip-gram) are both in their original forms, without any efficiency optimization tricks being applied.

For all these models, there exist two vector representations for each word in the vocabulary: the input vector $\textbf{v}_w$, and the output vector $\textbf{v}^{'}_w$. Learning the input vectors is cheap; but learning the output vectors is very expensive. From the update equations (22) and (33), we can find that, in order to update $\textbf{v}^{'}_w$, for each training instance, we have to iterate through every word $w_j$ in the vocabulary, compute their net input $u_j$ , probability prediction $y_j$ (or $y_{c,j}$ for skip-gram), their prediction error $e_j$ (or $EI_j$ for skip-gram), and finally use their prediction error to update their output vector $\textbf{v}^{'}_j$ .

Doing such computations for all words for every training instance is very expensive, making it impractical to scale up to large vocabularies or large training corpora. To solve this problem, an intuition is to limit the number of output vectors that must be updated per training instance. One elegant approach to achieving this is hierarchical softmax; another approach is through sampling, which will be discussed in the next section.

Both tricks optimize only the computation of the updates for output vectors. In our derivations, we care about three values: 

1. $E$, the new objective function; 
2. $\frac{\partial E}{\partial \textbf{v}_w^{'}}$ , the new update equation for the output vectors;
3. $\frac{\partial E}{\partial \textbf{h}}$, the weighted sum of predictions errors to be backpropagated for updating input vectors.

> 到目前为止，我们讨论的模型(bigram、CBOW、skip-gram)都是它们的原始形式，没有应用任何优化技巧。
>
> 对于所有这些模型，词典中的每个单词都有两个向量表示：输入向量 $\textbf{v}_w$ 和输出向量 $\textbf{v}^{'}_w$ 。学习输入向量的代价较小，但是学习输出向量是非常昂贵的。从更新公式(22)和(33)中我们可以发现，为了更新 $\textbf{v}^{'}_w$ ，对于每个训练实例，我们必须迭代词典中的每个单词 $w_j$，计算它们的输入 $u_j$ 、预测概率 $y_j$ (或对于 skip-gram 为 $y_{c，j}$)、它们的预测误差 $e_j$ (或对于 skip-gram 为 $EI_j$)，并最终使用它们的预测误差去更新输出向量 $\textbf{v}_{j}^{'}$
>
> 对每个训练实例的所有单词进行这样的计算非常昂贵，这使得扩展到大型词典或大型训练语料库是不切实际的。要解决这个问题，直觉上是限制每个训练实例必须更新的输出向量的数量。实现这一目标的一种优雅方法是 hierarchical softmax; 另一种方法是通过抽样，这将在下一节中讨论。
>
> (注：其实从直观理解上来看，用整个词典的单词做输出层的话还应该容易导致梯度爆炸的情况出现，词典越大越容易。)
>
> 这两种技巧都只优化了输出向量更新的计算。在推导过程中，我们关心三个值:
>
> 1. $E$ ，新的目标函数;
> 2. $\frac{\partial E}{\partial \textbf{v}_w^{'}}$，新的输出向量更新方程;
> 3. $\frac{\partial E}{\partial \textbf{h}}$，为更新输入向量而反向传播的预测误差的加权和。

#### 3.1 Hierarchical Softmax

![Figure4](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig4.png)

**Figure 4: An example binary tree for the hierarchical softmax model. The white units are words in the vocabulary, and the dark units are inner units. An example path from root to $w_2$ is highlighted. In the example shown, the length of the path $L(w_2) = 4$. $n(w, j)$ means the j-th unit on the path from root to the word $w$.**

Hierarchical softmax is an efficient way of computing softmax (Morin and Bengio, 2005; Mnih and Hinton, 2009). The model uses a binary tree to represent all words in the vocabulary. The $V$ words must be leaf units of the tree. It can be proved that there are $V−1$ inner units. For each leaf unit, there exists a unique path from the root to the unit; and this path is used to estimate the probability of the word represented by the leaf unit. See Figure 4 for an example tree.

In the hierarchical softmax model, there is no output vector representation for words. Instead, each of the $V − 1$ inner units has an output vector $\textbf{v}_{n(w,j)}^{'}$ . And the probability of a word being the output word is defined as
$$
p(w = w_O) = \prod_{j=1}^{L(w)-1} \ \sigma([[n(w,j+1) = ch(n(w,j))]] \cdot {\textbf{v}_{n(w,j)}^{'}}^T \textbf{h})
\qquad (37)
$$

- where $ch(n)$ is the left child of unit $n$;

- $\textbf{v}^{'}_{n(w,j)}$ is the vector representation (“output vector”) of the inner unit $n(w, j)$; 

- $\textbf{h}$ is the output value of the hidden layer (in the skip-gram model $\textbf{h} = \textbf{v}_{w_I}$ ; and in CBOW, $\textbf{h} = \frac{1}{C}\sum_{c=1}^{C}\textbf{v}_{w_c}$ );

   $[[x]]$ is a special function defined as
  $$
  [[x]] = \begin{cases}
  1 \qquad if \ x \ is \ true;
  \\
  -1 \qquad otherwise.
  \end{cases}
  \qquad (38)
  $$

Let us intuitively understand the equation by going through an example. Looking at Figure 4, suppose we want to compute the probability that $w_2$ being the output word. We define this probability as the probability of a random walk starting from the root ending at the leaf unit in question. At each inner unit (including the root unit), we need to assign the probabilities of going left and going right.4(While an inner unit of a binary tree may not always have both children, a binary Huffman tree’s inner units always do. Although theoretically one can use many different types of trees for hierarchical softmax, word2vec uses a binary Huffman tree for fast training.) We define the probability of going left at an inner unit $n$ to be
$$
p(n,left) = \sigma({\textbf{v}_{n}^{'}}^T \cdot \textbf{h}) \qquad (39)
$$
which is determined by both the vector representation of the inner unit, and the hidden layer’s output value (which is then determined by the vector representation of the input word(s)). Apparently the probability of going right at unit $n$ is
$$
p(n,right) = 1 - \sigma({\textbf{v}_{n}^{'}}^T \cdot \textbf{h}) = \sigma(-{\textbf{v}_{n}^{'}}^T \cdot \textbf{h})
\qquad (40)
$$
Following the path from the root to $w_2$ in Figure 4, we can compute the probability of $w_2$ being the output word as
$$
p(w_2 = w_O)
\\
\begin{align*}
&= p(n(w_2,1),left) \cdot p(n(w_2,2),left) \cdot p(n(w_2,3),right)
\qquad (41)
\\
&= \sigma({\textbf{v}_{n(w_2,1)}^{'}}^T\textbf{h}) \cdot \sigma({\textbf{v}_{n(w_2,2)}^{'}}^T\textbf{h}) \cdot \sigma({- \textbf{v}_{n(w_2,3)}^{'}}^T\textbf{h})
\qquad (42)
\end{align*}
$$
which is exactly what is given by (37). It should not be hard to verify that
$$
\sum_{i=1}^{V} \ p(w_i=w_O) = 1 \qquad (43)
$$
making the hierarchical softmax a well defined multinomial distribution among all words.

Now let us derive the parameter update equation for the vector representations of the inner units. For simplicity, we look at the one-word-context model first. Extending the update equations to CBOW and skip-gram models is easy.

For the simplicity of notation, we define the following shortenizations without introducing ambiguity:
$$
[[·]] := [[n(w, j + 1) = ch(n(w, j))]]
\qquad (44)
$$

$$
\textbf{v}_{j}^{'} := \textbf{v}_{n_{w,j}}^{'}
\qquad (45)
$$

For a training instance, the error function is defined as
$$
E = -log \ p(w=w_O|w_I) = - \sum_{j=1}^{L(w)-1} log(\sigma([[\cdot]]{\textbf{v}_{j}^{'}}^T\textbf{h})
\qquad (46)
$$
We take the derivative of $E$  with regard to $\textbf{v}^{'}_j\textbf{h}$, obtaining
$$
\begin{align*}
\frac{\partial E}{\partial \textbf{v}_{j}^{'}\textbf{h}}
&=(\sigma([[\cdot]]{\textbf{v}_{j}^{'}}^T \textbf{h}) - 1)[[\cdot]]

\qquad (47)
\\
&=
\begin{cases}
\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - 1 \quad ([[\cdot]] = 1)
\\
\sigma({\textbf{v}_{j}^{'}}^T \textbf{h})
\qquad ([[\cdot]] = -1)
\end{cases}
\qquad (48)
\\
&= \sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - t_j
\qquad (49)
\end{align*}
$$
where $t_j = 1$ if $[[·]] = 1$ and $t_j = 0$ otherwise.

Next we take the derivative of $E$ with regard to the vector representation of the inner unit $n(w, j)$ and obtain
$$
\frac{\partial E}{\partial \textbf{v}_{j}^{'}} 
= \frac{\partial E}{\partial \textbf{v}_{j}^{'}\textbf{h}} 
\cdot
\frac{\partial \textbf{v}_{j}^{'}\textbf{h}}{\partial \textbf{v}_{j}^{'}} = (\sigma({\textbf{v}_{j}^{'}}^T\textbf{h} - t_j)) \cdot \textbf{h}
\qquad (50)
$$
which results in the following update equation:
$$
{\textbf{v}_{j}^{'}}^{(new)} = {\textbf{v}_{j}^{'}}^{(old)} - \eta(\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) -t_j)\cdot\textbf{h}
\qquad (50)
$$
which should be applied to $j = 1, 2, \cdots , L(w) − 1$. We can understand $\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - t_j$ as the prediction error for the inner unit $n(w, j)$. The “task” for each inner unit is to predict whether it should follow the left child or the right child in the random walk. $t_j = 1$ means the ground truth is to follow the left child; $t_j = 0$ means it should follow the right child. $\sigma({\textbf{v}_{j}^{'}}^T\textbf{h})$ is the prediction result. For a training instance, if the prediction of the inner unit is very close to the ground truth, then its vector representation $\textbf{v}_{j}^{'}$ will move very little; otherwise $\textbf{v}^{'}_j$ will move in an appropriate direction by moving (either closer or farther away5 from $\textbf{h}$) so as to reduce the prediction error for this instance. This update equation can be used for both CBOW and the skip-gram model. When used for the skip-gram model, we need to repeat this update procedure for each of the $C$ words in the output context.

In order to backpropagate the error to learn input→hidden weights, we take the derivative of $E$ with regard to the output of the hidden layer and obtain
$$
\begin{align*}
\frac{\partial E}{\partial \textbf{h}}
&= \sum_{j=1}^{L(w)-1} \frac{\partial E}{\partial \textbf{v}_{j}^{'}\textbf{h}} \cdot \frac{\partial \textbf{v}_{j}^{'} \textbf{h}}{\partial \textbf{h}}
\qquad(52)
\\
&= \sum_{j=1}^{L(w)-1} (\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - t_j) \cdot \textbf{v}_{j}^{'}
\qquad(53)
\\
&:= EH
\qquad(54)
\end{align*}
$$
which can be directly substituted into (23) to obtain the update equation for the input vectors of CBOW. For the skip-gram model, we need to calculate a $EH$ value for each word in the skip-gram context, and plug the sum of the $EH$ values into (35) to obtain the update equation for the input vector.

From the update equations, we can see that the computational complexity per training instance per context word is reduced from $O(V)$ to $O(log(V ))$, which is a big improvement in speed. We still have roughly the same number of parameters ($V−1$ vectors for inner-units compared to originally $V$ output vectors for words).

> Hierarchical softmax是一种有效的计算 softmax 的方法(Morin 和 Bengio, 2005; Mnih 和 Hinton, 2009)。该模型使用二叉树来表示词典中的所有单词。$V$ 个单词必须存储于二叉树的叶子节点。可以被证明一共有 $V-1$ 个内部节点(非叶子节点)。对于每个叶子节点，有一条唯一的路径可以从根节点到达该叶子节点；该路径被用来计算该叶子节点代表的词的概率。参考 Figure 4。
>
> **Hierarchical softmax 模型没有词的输出向量**，取而代之的是， $V-1$ 个内部节点(非叶子节点)都有一个输出向量 ${\textbf{v}}_{n(w,j)}^{'}$ 。一个词作为输出词的概率被定义为：
> $$
> p(w = w_O) = \prod_{j=1}^{L(w)-1} \ \sigma([[n(w,j+1) = ch(n(w,j))]] \cdot {\textbf{v}_{n(w,j)}^{'}}^T \textbf{h})
> \qquad (37)
> $$
>
> - 其中 $ch(n)$ 是 $n$ 的左子节点。
>
> - $\textbf{v}^{'}_{n(w,j)}$ 是非叶子节点 $n(w, j)$ 的向量表示，即输出向量; 
>
> - $\textbf{h}$ 是隐藏层的输出值 (在 Skip-gram 模型中 $\textbf{h} = \textbf{v}_{w_I}$ ; 在 CBOW 中, $ \textbf{h} = \frac{1}{C}\sum_{c=1}^{C}\textbf{v}_{w_c}$); 
>
>   $[[x]]$ 是一个特殊函数，定义如下（注：类似指示函数）：
>   $$
>   [[x]] = \begin{cases}
>   1 \qquad if \ x \ is \ true;
>   \\
>   -1 \qquad otherwise.
>   \end{cases}
>   \qquad (38)
>   $$
>
> 让我们通过一个例子直观地理解这个方程。看图4，假设我们想计算 $w_2$ 作为输出单词的概率。我们把这个概率定义为从根节点到叶子节点的随机游走的概率。在每个非叶子结点(包括根节点)，我们需要分配向左和向右的概率。(4虽然二叉树的内部节点不一定都有两个子节点，但二叉Huffman树的内部节点总是有两个子节点。虽然理论上可以使用许多不同类型的树来进行 hierarchical softmax，但word2vec使用二叉Huffman树来进行快速训练。)
>
> 我们定义在一个内部节点 $n$ 处向左的概率为
> $$
> p(n,left) = \sigma({\textbf{v}_{n}^{'}}^T \cdot \textbf{h}) \qquad (39)
> $$
> 它由内部节点结点的向量表示和隐藏层的输出值(由输入层的向量表示决定)共同决定。显然，在非叶子结点 $n$ 向右的概率是
> $$
> p(n,right) = 1 - \sigma({\textbf{v}_{n}^{'}}^T \cdot \textbf{h}) = \sigma(-{\textbf{v}_{n}^{'}}^T \cdot \textbf{h})
> \qquad (40)
> $$
> 根据 Figure 4 中从根节点到 $w_2$节点的路径，我们可以计算出 $w_2$ 作为输出词的概率
> $$
> p(w_2 = w_O)
> \\
> \begin{align*}
> &= p(n(w_2,1),left) \cdot p(n(w_2,2),left) \cdot p(n(w_2,3),right)
> \qquad (41)
> \\
> &= \sigma({\textbf{v}_{n(w_2,1)}^{'}}^T\textbf{h}) \cdot \sigma({\textbf{v}_{n(w_2,2)}^{'}}^T\textbf{h}) \cdot \sigma({- \textbf{v}_{n(w_2,3)}^{'}}^T\textbf{h})
> \qquad (42)
> \end{align*}
> $$
> 这正是由 (37) 给出的。不难看出(注：？？？哪里看出的？请高人指点这里怎么证明)：
> $$
> \sum_{i=1}^{V} \ p(w_i=w_O) = 1 \qquad (43)
> $$
> 这使得 hierarchical softmax 模型是一个的关于所有单词的良好定义的多项分布。
>
> 现在我们推导出内部节点的向量表示的参数更新方程。为了简单起见，我们先看看一个 one-word-context 模型。然后很容易再将更新公式扩展到 CBOW 和 skip-gram 模型。
>
> 为简化表示，我们定义了下列不会引起歧义缩写:
> $$
> [[·]] := [[n(w, j + 1) = ch(n(w, j))]]
> \qquad (44)
> $$
>
> $$
> \textbf{v}_{j}^{'} := \textbf{v}_{n_{w,j}}^{'}
> \qquad (45)
> $$
>
> 对于一个训练实例，误差函数定义为
> $$
> E = -log \ p(w=w_O|w_I) = - \sum_{j=1}^{L(w)-1} log(\sigma([[\cdot]]{\textbf{v}_{j}^{'}}^T\textbf{h})
> \qquad (46)
> $$
> 求 $E$ 对 $\textbf{v}^{'}_j\textbf{h}$ ,的导数，得
> $$
> \begin{align*}
> \frac{\partial E}{\partial \textbf{v}_{j}^{'}\textbf{h}}
> &=(\sigma([[\cdot]]{\textbf{v}_{j}^{'}}^T \textbf{h}) - 1)[[\cdot]]
> 
> \qquad (47)
> \\
> &=
> \begin{cases}
> \sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - 1 \quad ([[\cdot]] = 1)
> \\
> \sigma({\textbf{v}_{j}^{'}}^T \textbf{h})
> \qquad ([[\cdot]] = -1)
> \end{cases}
> \qquad (48)
> \\
> &= \sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - t_j
> \qquad (49)
> \end{align*}
> $$
> 其中，如果$[[·]]=1$，则 $t_j=1$；否则，$t_j=0$。
>
> 接下来求 $E$ 对内部节点 $n(w,j)$ 的偏导数，可得：
> $$
> \frac{\partial E}{\partial \textbf{v}_{j}^{'}} 
> = \frac{\partial E}{\partial \textbf{v}_{j}^{'}\textbf{h}} 
> \cdot
> \frac{\partial \textbf{v}_{j}^{'}\textbf{h}}{\partial \textbf{v}_{j}^{'}} = (\sigma({\textbf{v}_{j}^{'}}^T\textbf{h} - t_j)) \cdot \textbf{h}
> \qquad (50)
> $$
> 最终的更新公式为：
> $$
> {\textbf{v}_{j}^{'}}^{(new)} = {\textbf{v}_{j}^{'}}^{(old)} - \eta(\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) -t_j)\cdot\textbf{h}
> \qquad (50)
> $$
> 公式会从 $j = 1, 2, \cdots , L(w) − 1$ 依次迭代. 我们可以将 $\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - t_j$ 理解为  内部结点 $n(w, j)$ 的预测误差. 每个非叶子节点的“任务”是预测在随机游走中是向左还是向右。$t_j = 1$ 表示向左; $t_j = 0$ 表示向右.
>
> **$\sigma({\textbf{v}_{j}^{'}}^T\textbf{h})$ 表示预测结果. 对于训练实例来说, 如果节点的预测结果和真实路径非常相似, 那么 $\textbf{v}_{j}^{'}$ 的向量表示只需要微小的改动; 否则 $\textbf{v}^{'}_j$ 就会按适当的方向进行调整（要么靠近，要么远离  $\textbf{h}$ ) 来减小预测误差。**
>
> 这个更新公式既可以用于 CBOW 模型，也可以用于 skip-gram 模型。当用于 skip-gram 模型时，需要对输出的 $C$ 个单词中的每个词重复这个更新过程。
>
> 为了反向传播误差去学习  input -> hidden 之间的权重矩阵，我们对 $E$ 求隐藏层的输出的导数，得：
> $$
> \begin{align*}
> \frac{\partial E}{\partial \textbf{h}}
> &= \sum_{j=1}^{L(w)-1} \frac{\partial E}{\partial \textbf{v}_{j}^{'}\textbf{h}} \cdot \frac{\partial \textbf{v}_{j}^{'} \textbf{h}}{\partial \textbf{h}}
> \qquad(52)
> \\
> &= \sum_{j=1}^{L(w)-1} (\sigma({\textbf{v}_{j}^{'}}^T\textbf{h}) - t_j) \cdot \textbf{v}_{j}^{'}
> \qquad(53)
> \\
> &:= EH
> \qquad(54)
> \end{align*}
> $$
> 可直接代入(23)得到 CBOW 输入向量的更新方程。对于 skip-gram 模型，我们需要为 skip-gram 上下文中的每个单词计算一个 $EH$ 值，并将 $EH$ 值的总和代入(35)，得到输入向量的更新方程。
>
> 从更新方程中可以看出，每个训练实例的每个上下文词的计算复杂度从 $O(V)$ 降低到 $O(log(V))$ ，在速度上有很大的提高。我们仍然有大致相同数量的参数(非叶子结点向量 $V−1$个，原始 $V$ 个输出向量)。

#### 3.2 Negative Sampling

The idea of negative sampling is more straightforward than hierarchical softmax: in order to deal with the difficulty of having too many output vectors that need to be updated per iteration, we only update a sample of them.

Apparently the output word (i.e., the ground truth, or positive sample) should be kept in our sample and gets updated , and we need to sample a few words as negative samples (hence “negative sampling”). A probabilistic distribution is needed for the sampling process, and it can be arbitrarily chosen. We call this distribution the noise distribution, and denote it as $P_n(w)$. One can determine a good distribution empirically. 6(As described in (Mikolov et al., 2013b), word2vec uses a unigram distribution raised to the $\frac{3}{4}$ th power for the best quality of results.)

In word2vec, instead of using a form of negative sampling that produces a well-defined posterior multinomial distribution, the authors argue that the following simplified training objective is capable of producing high-quality word embeddings: 7(Goldberg and Levy (2014) provide a theoretical analysis on the reason of using this objective function)
$$
E = -log \ \sigma({\textbf{v}_{w_O}^{'}}^T\textbf{h})
- \sum_{w_j \in \mathcal{W_{neg}}} log \ \sigma(-{\textbf{v}_{w_j}^{'}}^T\textbf{h})
\qquad(55)
$$
where $w_O$ is the output word (i.e., the positive sample), and $\textbf{v}_{w_O}^{'}$ is its output vector; $\textbf{h}$ is the output value of the hidden layer: $\textbf{h}=\frac{1}{C} \sum_{c=1}^{C}\textbf{v}_{w_c}$ in the CBOW model and $\textbf{h} = \textbf{v}_{w_I}$ in the skip-gram model; $\mathcal{W}_{neg} = \{w_j |j = 1,\cdots, K\}$ is the set of words that are sampled based on $P_n(w)$, i.e., negative samples.

To obtain the update equations of the word vectors under negative sampling, we first take the derivative of $E$ with regard to the net input of the output unit $w_j$ :
$$
\begin{align*}
\frac{\partial E}{\partial {\textbf{v}_{w_j}^{'}}^T \textbf{h}} &=
\begin{cases}
\sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) - 1 \qquad if \ w_j = w_O
\\
\sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) \qquad if \ w_j \in \mathcal{W}_{neg}
\end{cases}
\qquad (56)
\\
&= \sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) -t_j
\qquad(57)
\end{align*}
$$
where $t_j$ is the “label” of word $w_j$ . $t = 1$ when $w_j$ is a positive sample; $t = 0$ otherwise. Next we take the derivative of $E$ with regard to the output vector of the word $w_j$ ,
$$
\frac{\partial E}{\partial \textbf{v}_{w_j}^{'}} 
= \frac{\partial E}{\partial {\textbf{v}_{w_j}^{'}}^T \textbf{h}} 
\cdot
\frac{\partial {\textbf{v}_{w_j}^{'}}^T\textbf{h}}{\partial \textbf{v}_{w_j}^{'}} = (\sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h} - t_j)) \cdot \textbf{h}
\qquad (58)
$$
which results in the following update equation for its output vector:
$$
{\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} - \eta(\sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) -t_j)\cdot\textbf{h}
\qquad (59)
$$
which only needs to be applied to $w_j \in \{w_O\}∪\mathcal{W}_{neg}$ instead of every word in the vocabulary. This shows why we may save a significant amount of computational effort per iteration.

The intuitive understanding of the above update equation should be the same as that of (11). This equation can be used for both CBOW and the skip-gram model. For the skip-gram model, we apply this equation for one context word at a time.

To backpropagate the error to the hidden layer and thus update the input vectors of words, we need to take the derivative of $E$ with regard to the hidden layer’s output, obtaining
$$
\begin{align*}
\frac{\partial E}{\partial \textbf{h}}
&= \sum_{w_j \in \{w_O\} \bigcup \mathcal{W}_{neg}}
\frac{\partial E}{\partial {\textbf{v}_{w_j}^{'}}^T \textbf{h}} \cdot \frac{\partial {\textbf{v}_{w_j}^{'}}^T\textbf{h}}{\partial \textbf{h}}
\qquad(60)
\\
&= \sum_{w_j \in \{w_O\} \bigcup \mathcal{W}_{neg}}
(\sigma({v_{w_j}^{'}}^T\textbf{h}) - t_j)\textbf{v}_{w_j}^{'}
:= EH
\qquad(61)
\end{align*}
$$
By plugging $EH$ into (23) we obtain the update equation for the input vectors of the CBOW model. For the skip-gram model, we need to calculate a $EH$ value for each word in the skip-gram context, and plug the sum of the $EH$ values into (35) to obtain the update equation for the input vector.

> Negative sampling 的理念比 Hierarchical softmax更简单 : 为了解决每次迭代都需要更新太多输出向量的问题，我们只更新其中的一个。
>
> 显然，输出词  (即ground truth，或正样本) 应该保存在我们的样本中并得到更新，我们需要将一些单词作为负样本(即“负样本”)进行采样。抽样过程需要一个概率分布，可以任意选择。我们称这个分布为噪声分布，表示为$P_n(w)$。我们可以通过经验来确定一个好的分布(如(Mikolov et al., 2013b)所述，word2vec使用 unigram 分布的 $\frac{3}{4}$ 次方以获得最佳质量的词向量)。
>
> 在 word2vec 中，作者认为用以下简化的训练目标取代用一个定义好的后验多项分布的负采样形式，也能够产生高质量的词嵌入(Goldberg和Levy(2014)对使用该目标函数的原因进行了理论分析。)：
> $$
> E = -log \ \sigma({\textbf{v}_{w_O}^{'}}^T\textbf{h})
> - \sum_{w_j \in \mathcal{W_{neg}}} log \ \sigma(-{\textbf{v}_{w_j}^{'}}^T\textbf{h})
> \qquad(55)
> $$
> 其中 $w_O$ 是输出词(即正样本)， $\textbf{v}_{w_O}^{'}$ 是其输出向量； $\textbf{h}$ 是隐藏层的输出值： CBOW : $\textbf{h}=\frac{1}{C}\sum_{c=1}^{C}\textbf{v}_{w_c}$, skip-gram :  $\textbf{h}=\textbf{v}_{w_c}$。$\mathcal{W}_{neg}=\{w_j|j=1，\cdots，K\}$ 是基于 $P_n(w)$ 采样的单词集合，即负样本。
>
> 为了获取负采样词向量的更新方程，我们首先求 $E$ 对 输出单元的输入 $w_j$的导数:
> $$
> \begin{align*}
> \frac{\partial E}{\partial {\textbf{v}_{w_j}^{'}}^T \textbf{h}} &=
> \begin{cases}
> \sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) - 1 \qquad if \ w_j = w_O
> \\
> \sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) \qquad if \ w_j \in \mathcal{W}_{neg}
> \end{cases}
> \qquad (56)
> \\
> &= \sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) -t_j
> \qquad(57)
> \end{align*}
> $$
> 其中 $t_j$ 是单词 $w_j$ 的“标签”。当 $w_j$ 为正样本时，$t=1$；否则，$t=0$。接下来，我们取 $E$ 对词 $w_j$ 的输出向量的导数，
> $$
> \frac{\partial E}{\partial \textbf{v}_{w_j}^{'}} 
> = \frac{\partial E}{\partial {\textbf{v}_{w_j}^{'}}^T \textbf{h}} 
> \cdot
> \frac{\partial {\textbf{v}_{w_j}^{'}}^T\textbf{h}}{\partial \textbf{v}_{w_j}^{'}} = (\sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h} - t_j)) \cdot \textbf{h}
> \qquad (58)
> $$
> 以下的输出向量更新公式：
> $$
> {\textbf{v}_{w_j}^{'}}^{(new)} = {\textbf{v}_{w_j}^{'}}^{(old)} - \eta(\sigma({\textbf{v}_{w_j}^{'}}^T\textbf{h}) -t_j)\cdot\textbf{h}
> \qquad (59)
> $$
> 其中只需要更新 $w_j \in \{w_O\}∪\mathcal{W}_{neg} $，而不是词典中的每个词。这也解释了为什们我们可以在一次迭代中节省巨大的计算量。
>
> 直觉上对该更新公式的理解和公式(11)一致。该公式可以通用于 CBOW 模型和 skip-gram 模型。对于 skip-gram 模型，我们一次作用于一个上下文单词。
>
> 为了使误差反向传播到隐藏层来更新词的输入向量，我们求 $E$ 对隐藏层输出 $\textbf{h}$ 的偏导：
> $$
> \begin{align*}
> \frac{\partial E}{\partial \textbf{h}}
> &= \sum_{w_j \in \{w_O\} \bigcup \mathcal{W}_{neg}}
> \frac{\partial E}{\partial {\textbf{v}_{w_j}^{'}}^T \textbf{h}} \cdot \frac{\partial {\textbf{v}_{w_j}^{'}}^T\textbf{h}}{\partial \textbf{h}}
> \qquad(60)
> \\
> &= \sum_{w_j \in \{w_O\} \bigcup \mathcal{W}_{neg}}
> (\sigma({v_{w_j}^{'}}^T\textbf{h}) - t_j)\textbf{v}_{w_j}^{'}
> := EH
> \qquad(61)
> \end{align*}
> $$
> 把 $EH$ 带入公式(23)可得 CBOW 模型的输入向量的更新公式。对于 skip-gram 模型，我们计算每个单词的 $EH$ 值并加和再带入公式(35)就可得到输入向量的更新公式。

## Acknowledgement

The author would like to thank Eytan Adar, Qiaozhu Mei, Jian Tang, Dragomir Radev, Daniel Pressel, Thomas Dean, Sudeep Gandhe, Peter Lau, Luheng He, Tomas Mikolov, Hao Jiang, and Oded Shmueli for discussions on the topic and/or improving the writing of the note.

## References

Goldberg, Y. and Levy, O. (2014). word2vec explained: deriving mikolov et al.’s negativesampling word-embedding method. arXiv:1402.3722 [cs, stat]. arXiv: 1402.3722.

Mikolov, T., Chen, K., Corrado, G., and Dean, J. (2013a). Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781.

Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., and Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, pages 3111–3119.

Mnih, A. and Hinton, G. E. (2009). A scalable hierarchical distributed language model. In Koller, D., Schuurmans, D., Bengio, Y., and Bottou, L., editors, Advances in Neural Information Processing Systems 21, pages 1081–1088. Curran Associates, Inc.

Morin, F. and Bengio, Y. (2005). Hierarchical probabilistic neural network language model. In AISTATS, volume 5, pages 246–252. Citeseer.

---------------------------



