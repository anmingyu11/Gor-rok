# Distributed Representations of Words and Phrases and their Compositionality

## Abstract

The recently introduced continuous Skip-gram model is an efficient method for learning high-quality distributed vector representations that capture a large number of precise syntactic and semantic word relationships. In this paper we present several extensions that improve both the quality of the vectors and the training speed. By subsampling of the frequent words we obtain significant speedup and also learn more regular word representations. We also describe a simple alternative to the hierarchical softmax called negative sampling. An inherent limitation of word representations is their indifference to word order and their inability to represent idiomatic phrases. For example, the meanings of “Canada” and “Air” cannot be easily combined to obtain “Air Canada”. Motivated by this example, we present a simple method for finding phrases in text, and show that learning good vector representations for millions of phrases is possible.

> 最近推出的 skip-gram 模型是一种学习高质量分布式向量表示的有效方法，能够捕捉大量精确的语法和语义词关系。在本文中，我们提出了几个扩展，既提高了向量的质量和训练速度。通过对频繁词的下采样，我们获得了显著的加速，也学习了更多常规单词的表示形式。
>
> 我们还描述了一种简单的 hierarchical softmax 替代方案，称为 negative sampling。词表示的一个硬伤是它们不关心词序和它们不能代表习惯短语。例如，“Canada”和“Air”的意思不能轻易组合成“Air Canada”。在这个例子的启发下，我们提出了一个简单的方法来在文本中寻找短语，并表明学习数百万短语的向量表示是可能的。

## 1 Introduction

Distributed representations of words in a vector space help learning algorithms to achieve better performance in natural language processing tasks by grouping similar words. One of the earliest use of word representations dates back to 1986 due to Rumelhart, Hinton, and Williams [13]. This idea has since been applied to statistical language modeling with considerable success [1]. The follow up work includes applications to automatic speech recognition and machine translation [14, 7], and a wide range of NLP tasks [2, 20, 15, 3, 18, 19, 9].

Recently, Mikolov et al. [8] introduced the Skip-gram model, an efficient method for learning highquality vector representations of words from large amounts of unstructured text data. Unlike most of the previously used neural network architectures for learning word vectors, training of the Skip-gram model (see Figure 1) does not involve dense matrix multiplications. This makes the training extremely efficient: an optimized single-machine implementation can train on more than 100 billion words in one day.

The word representations computed using neural networks are very interesting because the learned vectors explicitly encode many linguistic regularities and patterns. Somewhat surprisingly, many of these patterns can be represented as linear translations. For example, the result of a vector calculation vec(“Madrid”) - vec(“Spain”) + vec(“France”) is closer to vec(“Paris”) than to any other word vector [9, 8].

> 词在向量空间中的分布式表示通过对相似的词汇进行分组，帮助学习算法在自然语言处理任务中获得更好的表现。单词表示法的最早使用可以追溯到1986年，由Rumelhart, Hinton和Williams[13]提出。这一思想已经被应用到统计语言模型中，并取得了相当大的成功。后续工作包括自动语音识别和机器翻译的应用[14,7]，以及广泛的自然语言处理任务[2, 20, 15, 3, 18, 19, 9]。
>
> 最近，Mikolov等人[8]介绍了 Skip-gram 模型，这是一种从大量非结构化文本数据中学习单词的高质量向量表示的有效方法。与以前用于学习词向量的大多数神经网络结构不同，Skip-gram模型的训练(见图1)不涉及密集的矩阵乘法。这使得训练非常高效 : 一个优化的单机实现可以在一天内训练超过1000亿个单词。
>
> 使用神经网络计算的词向量表示非常有趣，因为所学习的向量显式地编码了许多语言规律和模式。 令人惊讶的是，许多这些模式都可以表示为线性平移。 例如，向量计算的结果
> $$
> vec(“马德里”) - vec(“西班牙”) + vec(“法国”)
> $$
> 比任何其他单词向量都更接近 $vec(“巴黎”) $[9，8]。

![Fig1](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Fig1.png)

Figure 1: The Skip-gram model architecture. The training objective is to learn word vector representations that are good at predicting the nearby words.

In this paper we present several extensions of the original Skip-gram model. We show that subsampling of frequent words during training results in a significant speedup (around 2x - 10x), and improves accuracy of the representations of less frequent words. In addition, we present a simplified variant of Noise Contrastive Estimation (NCE) [4] for training the Skip-gram model that results in faster training and better vector representations for frequent words, compared to more complex hierarchical softmax that was used in the prior work [8].

Word representations are limited by their inability to represent idiomatic phrases that are not compositions of the individual words. For example, “Boston Globe” is a newspaper, and so it is not a natural combination of the meanings of “Boston” and “Globe”. Therefore, using vectors to represent the whole phrases makes the Skip-gram model considerably more expressive. Other techniques that aim to represent meaning of sentences by composing the word vectors, such as the recursive autoencoders [15], would also benefit from using phrase vectors instead of the word vectors.

The extension from word based to phrase based models is relatively simple. First we identify a large number of phrases using a data-driven approach, and then we treat the phrases as individual tokens during the training. To evaluate the quality of the phrase vectors, we developed a test set of analogical reasoning tasks that contains both words and phrases. A typical analogy pair from our test set is “Montreal”:“Montreal Canadiens”::“Toronto”:“Toronto Maple Leafs”. It is considered to have been answered correctly if the nearest representation to vec(“Montreal Canadiens”) - vec(“Montreal”) + vec(“Toronto”) is vec(“Toronto Maple Leafs”).

Finally, we describe another interesting property of the Skip-gram model. We found that simple vector addition can often produce meaningful results. For example, vec(“Russia”) + vec(“river”) is close to vec(“Volga River”), and vec(“Germany”) + vec(“capital”) is close to vec(“Berlin”). This compositionality suggests that a non-obvious degree of language understanding can be obtained by using basic mathematical operations on the word vector representations.

> 在本文中，我们介绍了基于原始的 Skip-gram 模型的几个扩展。 我们表明，在训练过程中对频繁出现的单词进行下采样会显着提高训练速度（约2倍-10倍），并提高了非频繁词表示的准确性。 另外，我们提出了一种噪声对比估计（NCE）的简化变体[4]，用于训练Skip-gram模型，与之前的工作[8]中使用的更复杂的 hierarchical softmax 相比，该模型可以更快地训练频繁单词，并获得更好的向量表示。
>
> 词的表达方式受到了限制，因为它们无法表达不是由单个单词组成的习语短语。 例如，“Boston Globe”是报纸，因此它不是 “Boston” 和 “Globe” 含义的自然组合。 因此，使用向量表示整个短语可以使 Skip-gram 模型的表达能力大大提高。 旨在通过组合单词向量来表示句子含义的其他技术（例如递归自动编码器[15]）也将从使用短语向量而不是单词向量中受益。
>
> 从基于单词的模型扩展到基于短语的模型相对简单。首先，我们使用数据驱动的方法识别大量的短语，然后在训练期间将短语作为单个标记处理。为了评估短语向量的质量，我们开发了一个包含单词和短语的类比推理任务测试集。我们测试集中的一个典型类比是 “Montreal”:“Montreal Canadiens”::“Toronto”:“Toronto Maple Leafs”。
> $$
> vec(“Montreal Canadiens”)- vec(“Montreal”)+ vec(“Toronto”)
> $$
> 如果与向量间运算最接近的表示是 $vec(“Toronto Maple Leafs”)$，则认为回答正确。

## 2 The Skip-gram Model

The training objective of the Skip-gram model is to find word representations that are useful for predicting the surrounding words in a sentence or a document. More formally, given a sequence of training words $w_1, w_2, w_3 \ ,..., \ w_T$ , the objective of the Skip-gram model is to maximize the average log probability
$$
\frac{1}{T}\sum_{t=1}^{T} \sum_{-c \le j \le c , j \ne 0}log \ p(w_{t+j}|w_t) \qquad (1)
$$
where $c$ is the size of the training context (which can be a function of the center word $w_t$). Larger $c$ results in more training examples and thus can lead to a higher accuracy, at the expense of the training time. 

The basic Skip-gram formulation defines $p(w_{t+j} | w_t)$ using the softmax function :
$$
p(w_O|w_I) = \frac{exp({v^{'}_{w_O}}^T v_{w_I})}{\sum_{w=1}^{W}exp({v^{'}_{w}}^T v_{w_I})} \qquad (2)
$$
where $v_w$ and $v^{'}_w$ are the “input” and “output” vector representations of $w$, and $W$ is the number of words in the vocabulary. This formulation is impractical because the cost of computing $∇log\ p(w_O|w_I)$ is proportional to $W$, which is often large ($10^5$–$10^7$ terms).

> Skip-gram模型的训练目标是找到可用于预测句子或文档中周围单词的单词表示形式。(找到对于预测词在句子或文档中周边词有用的词向量)更正式地说，给定一个训练词序列 $w_1, w_2, w_3 \ ,..., \ w_T$ Skip-gram模型的目标是使平均对数概率最大化。
>
> 其中 $c$ 是训练上下文的大小(可以是中心词 $w_t$ 的函数(邻域函数))。更大的 $c$ 会有更多的训练示例，从而导致更高的准确性，但会牺牲训练时间。
>
> 基本的 skip-gram 使用 softmax 函数定义 $p(w_{t+j} | w_t)$:
> $$
> p(w_O|w_I) = \frac{exp({v^{'}_{w_O}}^T v_{w_I})}{\sum_{w=1}^{W}exp({v^{'}_{w}}^T v_{w_I})} \qquad (2)
> $$
> 其中 $ v_w $ 和 $ v ^ {'} _ w $ 是 $ w $ 的“输入”和“输出”向量表示，而 $ W $ 是词典中的词数。 这种公式是不切实际的，因为计算 $ log \ p（w_O | w_I）$ 的成本与 $ W $ 成正比，而后者通常很大（$ 10 ^ 5 $ – $ 10 ^ 7 $）。

#### 2.1 Hierarchical Softmax

A computationally efficient approximation of the full softmax is the hierarchical softmax. In the context of neural network language models, it was first introduced by Morin and Bengio [12]. The main advantage is that instead of evaluating $W$ output nodes in the neural network to obtain the probability distribution, it is needed to evaluate only about $log_2(W)$ nodes.

The hierarchical softmax uses a binary tree representation of the output layer with the $W$ words as its leaves and, for each node, explicitly represents the relative probabilities of its child nodes. These define a random walk that assigns probabilities to words.

More precisely, each word $w$ can be reached by an appropriate path from the root of the tree. Let $n(w, j)$ be the $j$-th node on the path from the root to $w$, and let $L(w)$ be the length of this path, so $n(w, 1) = root$ and $n(w, L(w)) = w$. In addition, for any inner node $n$, let $ch(n)$ be an arbitrary fixed child of $n$ and let $[[x]]$ be $1$ if $x$ is $true$ and $-1$ otherwise. 

Then the hierarchical softmax defines $p(w_O|w_I)$ as follows:
$$
p(w|w_I) = \prod^{L(w)−1}_{j=1}\sigma{([[n(w, j + 1) = ch(n(w, j))]] \cdot \ {v_{n(w,j)}^{'}}^T v_{w_I})}  \qquad (3)
$$
where $\sigma (x) = 1/(1 + exp(−x))$. It can be verified that $\sum_{w=1}^{W} p(w|w_I) = 1$. This implies that the cost of computing $log \ p(w_O|w_I)$ and $∇log \ p(w_O|w_I)$ is proportional to $L(w_O)$, which on average is no greater than $log \ W$. Also, unlike the standard softmax formulation of the Skip-gram which assigns two representations $v_w$ and $v^{'}_{w}$ to each word $w$, the hierarchical softmax formulation has one representation $v_w$ for each word $w$ and one representation $v^{'}_{n}$ for every inner node $n$ of the binary tree.

The structure of the tree used by the hierarchical softmax has a considerable effect on the performance. Mnih and Hinton explored a number of methods for constructing the tree structure and the effect on both the training time and the resulting model accuracy [10]. In our work we use a binary Huffman tree, as it assigns short codes to the frequent words which results in fast training. It has been observed before that grouping words together by their frequency works well as a very simple speedup technique for the neural network based language models [5, 8].

> 一个近似计算全部 softmax 的方法是 hierarchical softmax。在神经网络语言模型的背景下，它首先由 Morin 和 Bengio [12] 提出。其主要优点是无需计算神经网络中 $W$个输出节点来获得概率分布，只需计算约 $log_2(W) $节点即可。
>
> Hierarchical Softmax 使用二叉树的叶子节点表示 $W$ 个词，对于每个节点，显示地表示每个子节点的概率值。这里定义了一个随机游走，将概率赋给单词。
>
> 更准确地说，每个词 $w$ 都可以从树的根通过适当的路径到达。设 $n(w, j)$ 为 $root$ 到 $w$ 路径上的 $j$-th 节点，设 $L(w)$ 为这条路径的长度，则 $n(w, 1) = root$ ， $n(w, L(w)) = w$ 。另外，对于任意内部节点 $n$，设 $ch(n)$ 为 $n$ 的任意固定子节点(?? 这里应该意思是左右分支的意思)，如果 $x$ 为 $true$，则设 $[[x]]$ 为 $1$，否则为 $-1$ (指示器函数) 。
>
>
> $$
> p(w|w_I) = \prod^{L(w)−1}_{j=1}\sigma{([[n(w, j + 1) = ch(n(w, j))]] \cdot \ {v_{n(w,j)}^{'}}^T v_{w_I})}  \qquad (3)
> $$
>
> 其中 $\sigma (x) = 1/(1 + exp(−x))$. 可以证明 $\sum_{w=1}^{W} p(w|w_I) = 1$。这意味着计算 $log \ p(w_O|w_I)$ 和 $∇log \ p(w_O|w_I)$ 的成本与 $L(w_O)$ 成正比。平均不大于 $log \ W$。
>
> 此外， 与 Skip-gram 的标准 Softmax 公式不同，后者为每个单词 $w$ 分配了两种表示形式 $v_w$ 和 $v^{'}_w$。而 Hierarchical Softmax 公式针对每个单词 $w$ 有表示 $v_w$，对于二叉树的内部节点 $n$ ，对于每个单词 $w$ 的每个内部节点 $n$ 具有一个表示 $v^{'}_n$ 。
>
> Hierarchical Softmax 使用的树结构对性能有相当大的影响。 Mnih 和 Hinton 探索了许多构建树的方法，以及对训练时间和得到的模型精度[10]的影响。在我们的工作中，使用二叉哈夫曼树，因为给频繁词分配短码，从而快速训练。之前已经观察到，根据单词的频率将单词分组在一起，对于基于神经网络的语言模型来说，这是一种非常简单的加速技术[5,8]。

#### 2.2 Negative Sampling

An alternative to the hierarchical softmax is  Noise Contrastive Estimation (NCE), which was introduced by Gutmann and Hyvarinen [4] and applied to language modeling by Mnih and Teh [11]. NCE posits that a good model should be able to differentiate data from noise by means of logistic regression. This is similar to hinge loss used by Collobert and Weston [2] who trained the models by ranking the data above noise.

While NCE can be shown to approximately maximize the log probability of the softmax, the Skipgram model is only concerned with learning high-quality vector representations, so we are free to simplify NCE as long as the vector representations retain their quality. 

We define Negative sampling (NEG) by the objective
$$
log \ \sigma({v^{'}_{w_O}}^{T}v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \~{} P_n(w)}[log \ \sigma({-v^{'}_{w_i}}^{T}v_{w_I})] \qquad (4)
$$
which is used to replace every $log \ P(w_O|w_I)$ term in the Skip-gram objective. Thus the task is to distinguish the target word $w_O$ from draws from the noise distribution $P_n(w)$ using logistic regression, where there are $k$ negative samples for each data sample. Our experiments indicate that values of $k$ in the range $5 \ – \ 20$ are useful for small training datasets, while for large datasets the $k$ can be as small as $2 \  – \ 5$. The main difference between the Negative sampling and NCE is that NCE needs both samples and the numerical probabilities of the noise distribution, while Negative sampling uses only samples. And while NCE approximately maximizes the log probability of the softmax, this property is not important for our application.

Both NCE and NEG have the noise distribution $P_n(w)$ as a free parameter. We investigated a number of choices for $P_n(w)$ and found that the unigram distribution $U(w)$ raised to the $3/4$ rd power (i.e., $\frac{U(w)^{3/4}}{Z}$) outperformed significantly the unigram and the uniform distributions, for both NCE and NEG on every task we tried including language modeling (not reported here).

![Fig2](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Fig2.png)

**Figure 2: Two-dimensional PCA projection of the 1000-dimensional Skip-gram vectors of countries and their capital cities. The figure illustrates ability of the model to automatically organize concepts and learn implicitly the relationships between them, as during the training we did not provide any supervised information about what a capital city means.**

> 与 Hierarchical Softmax相比，另一种选择是 Noise Contrastive Estimation (NCE)，它由 Gutmann 和 Hyvarinen [4]引入，并由 Mnih 和 Teh [11]应用于语言建模。NCE认为，好的模型应该能够通过逻辑回归将数据与噪声区分开。这类似于 Collobert 和 Weston [2]所使用的 Hinge Loss，他们通过将数据排在噪声之上来训练模型。
>
> 虽然 NCE 可以近似地最大化 softmax 的对数概率，但 Skipgram 模型只关心学习高质量的向量表示，所以我们可以自由地简化 NCE ，只要向量表示保持其质量。
>
> 我们定义负采样的目标函数：
> $$
> log \ \sigma({v^{'}_{w_O}}^{T}v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \~{} P_n(w)}[log \ \sigma({-v^{'}_{w_i}}^{T}v_{w_I})] \qquad (4)
> $$
> 它用于替换 Skip-gram目标函数中的每个 $log \ P(w_O | w_I)$项。 因此，任务是使用逻辑回归从噪声分布 $P_n(w)$ 的抽样中区分出目标词 $w_O$，其中每个数据样本有 $k$ 个负样本。 我们的实验表明，$k$ 值在 $5 – 20$ 的范围内对于小型训练数据集很有用，而对于大型数据集，k可以小至 $2 \ – \ 5$。 负采样与 NCE 之间的主要区别在于 NCE 需要采样和噪声分布的数值概率，而负采样仅使用采样。 虽然 NCE 近似地最大化了 Softmax 的对数概率，但是此属性对于我们的应用程序并不重要。
>
> NCE 和 NEG 都将噪声分布 $P_n(w)$ 作为自由参数。 我们研究了 $P_n(w)$ 的多种选择，发现提高到 $3/4$ 次幂的 unigram 分布 $U(w)$（即 $\frac{U(w)^{3/4}}{Z}$）明显优于 unigram 和均匀分布， 对于 NCE 和 NEG，我们尝试了包括语言建模在内的每项任务（本文里没有）。

#### 2.3 Subsampling of Frequent Words

In very large corpora, the most frequent words can easily occur hundreds of millions of times (e.g., “in”, “the”, and “a”). Such words usually provide less information value than the rare words. For example, while the Skip-gram model benefits from observing the co-occurrences of “France” and “Paris”, it benefits much less from observing the frequent co-occurrences of “France” and “the”, as nearly every word co-occurs frequently within a sentence with “the”. This idea can also be applied in the opposite direction; the vector representations of frequent words do not change significantly after training on several million examples.

To counter the imbalance between the rare and frequent words, we used a simple subsampling approach: each word $w_i$ in the training set is discarded with probability computed by the formula
$$
P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} \qquad (5)
$$
where $f(w_i)$ is the frequency of word $w_i$ and $t$ is a chosen threshold, typically around $10 \ −\ 5$ . We chose this subsampling formula because it aggressively subsamples words whose frequency is greater than $t$ while preserving the ranking of the frequencies. Although this subsampling formula was chosen heuristically, we found it to work well in practice. It accelerates learning and even significantly improves the accuracy of the learned vectors of the rare words, as will be shown in the following sections.

![Table1](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Table1.png)

**Table 1: Accuracy of various Skip-gram 300-dimensional models on the analogical reasoning task as defined in [8]. NEG-k stands for Negative Sampling with k negative samples for each positive sample; NCE stands for Noise Contrastive Estimation and HS-Huffman stands for the Hierarchical Softmax with the frequency-based Huffman codes.**

> 在非常大的语料库中，最常见的单词很容易出现数亿次(如“In”、“the”、“a”)。这类词提供的信息价值通常低于罕见词。例如，虽然 Skip-gram 模型从观察 “France” 和 “Paris” 的共现中提升，但它从观察“France”和“the”的频繁同时出现中获益要少得多，因为几乎每个单词都在一个句子中与“the”同时出现。这个想法也可以用在相反的方向上; 经过几百万个 example 的训练，频繁词的向量表示没有明显变化。
>
> 为了解决罕见词和频繁词之间的不平衡问题，我们采用了一种简单的下采样 : 丢弃训练集中的每个单词 $w_i$，其丢弃概率由公式计算：
> $$
> P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} \qquad (5)
> $$
> 其中 $f(w_i)$是单词 $w_i$ 的频率，$t$ 是一个选定的阈值，通常在 $10\ − \ 5$左右。我们选择这个下采样公式是因为它积极地对频率大于 $t$ 的单词进行重采样，同时保留频率的排序。虽然这一抽样公式的选择是启发式的，但我们发现它在实践中是有效的。它加速了学习，甚至显著提高了罕见词的向量的准确性，这将在以下几节中展示。

## 3 Empirical Results

In this section we evaluate the Hierarchical Softmax (HS), Noise Contrastive Estimation, Negative Sampling, and subsampling of the training words. We used the analogical reasoning task1 introduced by Mikolov et al. [8]. The task consists of analogies such as “Germany” : “Berlin” :: “France” : ?, which are solved by finding a vector $x$ such that $vec(x)$ is closest to $vec(“Berlin”) - vec(“Germany”) + vec(“France”)$ according to the cosine distance (we discard the input words from the search). This specific example is considered to have been answered correctly if $x$ is “Paris”. The task has two broad categories: the syntactic analogies (such as “quick” : “quickly” :: “slow” : “slowly”) and the semantic analogies, such as the country to capital city relationship.

For training the Skip-gram models, we have used a large dataset consisting of various news articles (an internal Google dataset with one billion words). We discarded from the vocabulary all words that occurred less than 5 times in the training data, which resulted in a vocabulary of size 692K. The performance of various Skip-gram models on the word analogy test set is reported in Table 1. The table shows that Negative Sampling outperforms the Hierarchical Softmax on the analogical reasoning task, and has even slightly better performance than the Noise Contrastive Estimation. The subsampling of the frequent words improves the training speed several times and makes the word representations significantly more accurate.

It can be argued that the linearity of the skip-gram model makes its vectors more suitable for such linear analogical reasoning, but the results of Mikolov et al. [8] also show that the vectors learned by the standard sigmoidal recurrent neural networks (which are highly non-linear) improve on this task significantly as the amount of the training data increases, suggesting that non-linear models also have a preference for a linear structure of the word representations.

> 在本节中，我们评估 Hierarchical Softmax (HS)、Noise Contrastive Estimation、Negative Sampling和训练词的下采样。我们使用 Mikolov 等人介绍的类比推理任务1。该任务由类似于 “德国”:“柏林”::“法国”:? 这样的类比组成，通过根据余弦距离找到向量 $x$，使得 $vec(x)$ 最接近 $vec(“柏林”)- vec(“德国”)+ vec(“法国”)$  (我们从搜索中丢弃输入单词)来解决。如果 $x$ 是 “Paris”，这个特定的例子被认为是正确的。这项任务有两大类：句法类比(如“快”:“快”::“慢”:“慢”)和语义类比，如国家与首都城市的关系。
>
> 为了训练 Skip-Gram模型，我们使用了一个由各种新闻文章组成的大型数据集(一个内部的谷歌数据集，有10亿个单词)。我们将所有在训练数据中出现次数小于5次的单词从词汇表中剔除，得到的词汇表大小为692K。Table1列出了各种 skip-gram 模型在单词类比测试集上的表现。从表中可以看出，NS 在类比推理任务上优于HS，甚至比 NCE 的表现稍好。频繁词的下采样提高了训练速度数倍，并使词的表征显著提高了准确性。
>
> 可以说，Skip-Gram 模型的线性特性使它的向量更适合于这种线性类比推理，但是 Mikolov 等人的结果 [8] 还表明，standard sigmoidal recurrent neural networks （高度非线性）所学习的向量随着训练数据量的增加而在此任务上得到了显着改善，这表明非线性模型也偏向于单词表示的线性结构。

## 4 Learning Phrases

As discussed earlier, many phrases have a meaning that is not a simple composition of the meanings of its individual words. To learn vector representation for phrases, we first find words that appear frequently together, and infrequently in other contexts. For example, “New York Times” and “Toronto Maple Leafs” are replaced by unique tokens in the training data, while a bigram “this is” will remain unchanged.

This way, we can form many reasonable phrases without greatly increasing the size of the vocabulary; in theory, we can train the Skip-gram model using all n-grams, but that would be too memory intensive. Many techniques have been previously developed to identify phrases in the text; however, it is out of scope of our work to compare them. We decided to use a simple data-driven approach, where phrases are formed based on the unigram and bigram counts, using：
$$
score(w_i, w_j ) = \frac{count(w_iw_j) − δ}{count(w_i) \times count(w_j)} \qquad (6)
$$
The $\delta$ is used as a discounting coefficient and prevents too many phrases consisting of very infrequent words to be formed. The bigrams with score above the chosen threshold are then used as phrases. Typically, we run $2-4$ passes over the training data with decreasing threshold value, allowing longer phrases that consists of several words to be formed. We evaluate the quality of the phrase representations using a new analogical reasoning task that involves phrases. Table 2 shows examples of the five categories of analogies used in this task. This dataset is publicly available on the web2 .

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Table2.png)

**Table 2: Examples of the analogical reasoning task for phrases (the full test set has 3218 examples). The goal is to compute the fourth phrase using the first three. Our best model achieved an accuracy of 72% on this dataset.**

> 如前所述，许多短语的含义不是其各个单词含义的简单组合。 为了学习短语的向量表示，我们首先找到频繁出现的单词，而在其他情况下很少出现。例如， “New Yourk Times” 和 “Toronto Maple Leafs” 在训练数据中被替换为唯一标记，而二元组 “this is” 将保持不变。
>
> 这样，我们就可以在不大大增加词汇量的情况下形成许多合理的短语;理论上讲，我们可以使用所有的 n-gram 来训练 Skip-gram 模型，但那样会占用太多的内存。以前已经开发了许多技术来识别文本中的短语；然而，比较它们超出了我们的工作范围。我们决定使用一种简单的数据驱动方法，其中短语是基于unigram和bigram计数形成的，使用：
> $$
> score(w_i, w_j ) = \frac{count(w_iw_j) − δ}{count(w_i) \times count(w_j)} \qquad (6)
> $$
> $\delta$ 用作 discounting 系数，可防止形成太多由非常少见的单词组成的短语。然后将分数高于所选阈值的二元词用作短语。通常，我们以递减的阈值对训练数据运行 $2-4$，从而允许形成由多个单词组成的较长短语。 我们使用涉及短语的新的类比推理任务来评估短语表征的质量。 表 2 显示了此任务中使用的五种类别 类比的示例。 该数据集可在web2上公开获得。

#### 4.1 Phrase Skip-Gram Results

Starting with the same news data as in the previous experiments, we first constructed the phrase based training corpus and then we trained several Skip-gram models using different hyperparameters. As before, we used vector dimensionality 300 and context size 5. This setting already achieves good performance on the phrase dataset, and allowed us to quickly compare the Negative Sampling and the Hierarchical Softmax, both with and without subsampling of the frequent tokens. The results are summarized in Table 3.

The results show that while Negative Sampling achieves a respectable accuracy even with $k = 5$, using $k = 15$ achieves considerably better performance. Surprisingly, while we found the Hierarchical Softmax to achieve lower performance when trained without subsampling, it became the best performing method when we downsampled the frequent words. This shows that the subsampling can result in faster training and can also improve accuracy, at least in some cases.

To maximize the accuracy on the phrase analogy task, we increased the amount of the training data by using a dataset with about 33 billion words. We used the hierarchical softmax, dimensionality of 1000, and the entire sentence for the context. This resulted in a model that reached an accuracy of 72%. We achieved lower accuracy 66% when we reduced the size of the training dataset to 6B words, which suggests that the large amount of the training data is crucial.

To gain further insight into how different the representations learned by different models are, we did inspect manually the nearest neighbours of infrequent phrases using various models. In Table 4, we show a sample of such comparison. Consistently with the previous results, it seems that the best representations of phrases are learned by a model with the hierarchical softmax and subsampling.

![Table3](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Table3.png)

**Table 3: Accuracies of the Skip-gram models on the phrase analogy dataset. The models were trained on approximately one billion words from the news dataset.**

![Table4](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Table4.png)

**Table 4: Examples of the closest entities to the given short phrases, using two different models.**

![Table5](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Table5.png)

**Table 5: Vector compositionality using element-wise addition. Four closest tokens to the sum of two vectors are shown, using the best Skip-gram model.**

> 从与之前实验相同的新闻数据开始，我们首先构建基于短语的训练语料库，然后使用不同的超参数训练几个跳跃模型。
>
> 与前面一样，我们使用向量维度 $300$ 和 $context=5$。这一设置已经在短语数据集上获得了良好的表现，并允许我们快速比较 NS 和 HS，包括对频繁词进行下采样和不进行下采样。结果如表3所示。
>
> 结果表明，即使在 $ k = 5 $的情况下，NS 也能达到可观的精度，而使用 $ k = 15 $ 则可以获得更好的表现。令人惊讶的是，虽然我们发现在不进行  subsampling 的情况下进行训练时，HS 的表现会不佳，但是当我们对频繁词进行下采样时，它成为性能最佳的方法。 这表明，至少在某些情况下，subsampling 可以提高训练速度，还可以提高准确性。
>
> 为了最大限度地提高短语类比任务的准确性，我们使用一个约330亿个单词的数据集来增加训练数据的数量。我们使用分层softmax，维度1000，以及整个句子作为上下文。这使得模型的精确度达到了72%。当我们将训练数据集的大小减少到6B个单词时，我们获得了较低的66%的准确率，这表明大量的训练数据是至关重要的。
>
> 为了进一步深入了解不同模型学习到的表示有多大的不同，我们使用不同的模型手工检查了不常用短语的最近邻。在表4中，我们展示了这样的比较示例。与先前的结果一致，似乎短语的最佳表征是通过具有 subsampling 的 HS 模型学习的。

## 5 Additive Compositionality

We demonstrated that the word and phrase representations learned by the Skip-gram model exhibit a linear structure that makes it possible to perform precise analogical reasoning using simple vector arithmetics. Interestingly, we found that the Skip-gram representations exhibit another kind of linear structure that makes it possible to meaningfully combine words by an element-wise addition of their vector representations. This phenomenon is illustrated in Table 5.

The additive property of the vectors can be explained by inspecting the training objective. The word vectors are in a linear relationship with the inputs to the softmax nonlinearity. As the word vectors are trained to predict the surrounding words in the sentence, the vectors can be seen as representing the distribution of the context in which a word appears. These values are related logarithmically to the probabilities computed by the output layer, so the sum of two word vectors is related to the product of the two context distributions. The product works here as the AND function: words that are assigned high probabilities by both word vectors will have high probability, and the other words will have low probability. Thus, if “Volga River” appears frequently in the same sentence together with the words “Russian” and “river”, the sum of these two word vectors will result in such a feature vector that is close to the vector of “Volga River”.

> 我们证明了通过Skip-gram模型学习的单词和短语表示具有线性结构，这使得可以使用简单的向量运算去执行精确的类比推理。 有趣的是，我们发现，Skip-gram表征还表现出另一种线性结构，可以通过逐元素相加其向量表征来有意义地组合单词。 表5说明了这种现象。
>
> **通过对训练目标的考察，可解释向量的可加性。词向量与 Softmax 的非线性的输入呈线性关系。当词向量被训练来预测句子中的周围词时，这些向量可以被看作是一个词出现在上下文中的分布。这些值与输出层计算的概率以对数形式相关**($log \ \sigma (w_i \cdot w_I) $)**，因此两个词向量的和与两个上下文分布的乘积相关。这个乘积在这里作为 AND 函数：被两个词向量分配高概率的词将有高概率，而其他的词将有低概率。因此，如果 “Volga River” 与 “Russian” 和 “river” 这两个词频繁出现在同一个句子中，这两个词向量之和就会产生一个与“伏尔加河”向量相近的特征向量。**

## 6 Comparison to Published Word Representations

Many authors who previously worked on the neural network based representations of words have published their resulting models for further use and comparison: amongst the most well known authors are Collobert and Weston [2], Turian et al. [17], and Mnih and Hinton [10]. We downloaded their word vectors from the web3 . Mikolov et al. [8] have already evaluated these word representations on the word analogy task, where the Skip-gram models achieved the best performance with a huge margin.

To give more insight into the difference of the quality of the learned vectors, we provide empirical comparison by showing the nearest neighbours of infrequent words in Table 6. These examples show that the big Skip-gram model trained on a large corpus visibly outperforms all the other models in the quality of the learned representations. This can be attributed in part to the fact that this model has been trained on about 30 billion words, which is about two to three orders of magnitude more data than the typical size used in the prior work. Interestingly, although the training set is much larger, the training time of the Skip-gram model is just a fraction of the time complexity required by the previous model architectures.

![Table6](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/DistributedRepresentationsofWordsandPhrasesandtheirCompositionality/Table6.png)

**Table 6: Examples of the closest tokens given various well known models and the Skip-gram model trained on phrases using over 30 billion training words. An empty cell means that the word was not in the vocabulary.**

> 许多以前研究基于神经网络的单词表示的作者已经发表了他们的结果模型，以供进一步使用和比较：其中最著名的作者有 Collobert 和 Weston [2]，Turian等人。[17]，以及 Mnih 和 Hinton [10]。我们从web3下载了他们的词库。Mikolov[8]我们已经在词语类比任务中对这些词语表征进行了评估，其中 Skip-gram 模型的表现最好，与其他模型差距很大。
>
> 为了更深入地了解学习向量的质量差异，我们在 表6 中给出了不常用单词的最近邻，从而提供了经验比较。这些例子表明，在大型语料库上训练的 Skip-gram 模型在学习表征的质量上明显优于所有其他模型。
>
> 这在一定程度上可以归因于这样一个事实，即这个模型已经训练了大约300亿个单词，这比之前工作中使用的典型数据大小大约多了 2 到 3 个数量级。有趣的是，虽然训练集要大得多，但 Skip-gram模型的训练时间只是以前模型体系结构所需时间复杂度的一小部分。有趣的是，尽管训练集要大得多，但是Skip-gram模型的训练时间只是以前模型体系结构所需时间复杂度的一小部分。（时间复杂度降低了，导致虽然数据规模增大，但是所需时间依然降低了。）

## 7 Conclusion

This work has several key contributions. We show how to train distributed representations of words and phrases with the Skip-gram model and demonstrate that these representations exhibit linear structure that makes precise analogical reasoning possible. The techniques introduced in this paper can be used also for training the continuous bag-of-words model introduced in [8].

We successfully trained models on several orders of magnitude more data than the previously published models, thanks to the computationally efficient model architecture. This results in a great improvement in the quality of the learned word and phrase representations, especially for the rare entities. We also found that the subsampling of the frequent words results in both faster training and significantly better representations of uncommon words. Another contribution of our paper is the Negative sampling algorithm, which is an extremely simple training method that learns accurate representations especially for frequent words.

The choice of the training algorithm and the hyper-parameter selection is a task specific decision, as we found that different problems have different optimal hyperparameter configurations. In our experiments, the most crucial decisions that affect the performance are the choice of the model architecture, the size of the vectors, the subsampling rate, and the size of the training window.

A very interesting result of this work is that the word vectors can be somewhat meaningfully combined using just simple vector addition. Another approach for learning representations of phrases presented in this paper is to simply represent the phrases with a single token. Combination of these two approaches gives a powerful yet simple way how to represent longer pieces of text, while having minimal computational complexity. Our work can thus be seen as complementary to the existing approach that attempts to represent phrases using recursive matrix-vector operations [16].

We made the code for training the word and phrase vectors based on the techniques described in this paper available as an open-source project4 .

> 这项工作有几个关键贡献。我们展示了如何用 Skip-gram 模型训练单词和短语的分布式表示，并证明这些表示具有线性结构，这使得精确的类比推理成为可能。本文介绍的技术也可用于训练文献[8]中介绍的连续词袋模型。
>
> 由于计算效率高的模型架构，我们成功地对模型进行了比以前发布的模型多几个数量级的数据训练。 这极大地提高了所学单词和短语表示的质量，特别是对于稀有实体。 我们还发现，频繁词的下采样不仅可以加快训练速度，而且可以显着改善不常见单词的表示方式。 我们论文的另一个贡献是负采样算法，它是一种非常简单的训练方法，可以学习准确的表征，尤其是对于频繁单词。
>
> 训练算法的选择和超参数的选择是一个任务特定的决策，因为我们发现不同的问题有不同的最优超参数配置。在我们的实验中，影响性能的最关键的决定是模型结构的选择、向量的大小、子采样率和训练窗口的大小。
>
> 这项工作的一个非常有趣的结果是，只需使用简单的向量加法，就可以在某种程度上有意义地组合词向量。本文提出的学习短语表征的另一种方法是用单个 token 简单地表示短语。这两种方法的结合给出了一种强大而简单的方式来表示较长的文本，同时具有最小的计算复杂度。因此，我们的工作可以被视为对现有方法的补充，该方法试图使用 recursive matrix  向量运算来表征短语[16]。
>
> 我们根据本文中提到的的技术开发了用于训练词和短语向量的代码，并作为一个开源项目4提供。

## References

[1] Yoshua Bengio, R´ejean Ducharme, Pascal Vincent, and Christian Janvin. A neural probabilistic language model. The Journal of Machine Learning Research, 3:1137–1155, 2003. 

[2] Ronan Collobert and Jason Weston. A unified architecture for natural language processing: deep neural networks with multitask learning. In Proceedings of the 25th international conference on Machine learning, pages 160–167. ACM, 2008. 

[3] Xavier Glorot, Antoine Bordes, and Yoshua Bengio. Domain adaptation for large-scale sentiment classification: A deep learning approach. In ICML, 513–520, 2011. 

[4] Michael U Gutmann and Aapo Hyv¨arinen. Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics. The Journal of Machine Learning Research, 13:307–361, 2012. 

[5] Tomas Mikolov, Stefan Kombrink, Lukas Burget, Jan Cernocky, and Sanjeev Khudanpur. Extensions of recurrent neural network language model. In Acoustics, Speech and Signal Processing (ICASSP), 2011 IEEE International Conference on, pages 5528–5531. IEEE, 2011. 

[6] Tomas Mikolov, Anoop Deoras, Daniel Povey, Lukas Burget and Jan Cernocky. Strategies for Training Large Scale Neural Network Language Models. In Proc. Automatic Speech Recognition and Understanding, 2011. 

[7] Tomas Mikolov. Statistical Language Models Based on Neural Networks. PhD thesis, PhD Thesis, Brno University of Technology, 2012. 

[8] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR Workshop, 2013. 

[9] Tomas Mikolov, Wen-tau Yih and Geoffrey Zweig. Linguistic Regularities in Continuous Space Word Representations. In Proceedings of NAACL HLT, 2013. 

[10] Andriy Mnih and Geoffrey E Hinton. A scalable hierarchical distributed language model. Advances in neural information processing systems, 21:1081–1088, 2009. 

[11] Andriy Mnih and Yee Whye Teh. A fast and simple algorithm for training neural probabilistic language models. arXiv preprint arXiv:1206.6426, 2012. 

[12] Frederic Morin and Yoshua Bengio. Hierarchical probabilistic neural network language model. In Proceedings of the international workshop on artificial intelligence and statistics, pages 246–252, 2005. 

[13] David E Rumelhart, Geoffrey E Hintont, and Ronald J Williams. Learning representations by backpropagating errors. Nature, 323(6088):533–536, 1986. 

[14] Holger Schwenk. Continuous space language models. Computer Speech and Language, vol. 21, 2007. 

[15] Richard Socher, Cliff C. Lin, Andrew Y. Ng, and Christopher D. Manning. Parsing natural scenes and natural language with recursive neural networks. In Proceedings of the 26th International Conference on Machine Learning (ICML), volume 2, 2011. 

[16] Richard Socher, Brody Huval, Christopher D. Manning, and Andrew Y. Ng. Semantic Compositionality Through Recursive Matrix-Vector Spaces. In Proceedings of the 2012 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2012.

[17] Joseph Turian, Lev Ratinov, and Yoshua Bengio. Word representations: a simple and general method for semi-supervised learning. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, pages 384–394. Association for Computational Linguistics, 2010. 

[18] Peter D. Turney and Patrick Pantel. From frequency to meaning: Vector space models of semantics. In Journal of Artificial Intelligence Research, 37:141-188, 2010. 

[19] Peter D. Turney. Distributional semantics beyond words: Supervised learning of analogy and paraphrase. In Transactions of the Association for Computational Linguistics (TACL), 353–366, 2013. 

[20] Jason Weston, Samy Bengio, and Nicolas Usunier. Wsabie: Scaling up to large vocabulary image annotation. In Proceedings of the Twenty-Second international joint conference on Artificial Intelligence-Volume Volume Three, pages 2764–2770. AAAI Press, 2011.