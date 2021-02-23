> 整篇文章读起来让人感觉很困惑，在不影响理解原文的基础上在有些段落的括号里加上了自己的理解。

# Efficient Estimation of Word Representations in Vector Space

## Abstract

We propose two novel model architectures for computing continuous vector representations of words from very large data sets. The quality of these representations is measured in a word similarity task, and the results are compared to the previously best performing techniques based on different types of neural networks. We observe large improvements in accuracy at much lower computational cost, i.e. it takes less than a day to learn high quality word vectors from a 1.6 billion words data set. Furthermore, we show that these vectors provide state-of-the-art performance on our test set for measuring syntactic and semantic word similarities.

> 我们提出了两种新的模型结构来计算来自大数据的词的连续向量表示。这些表示的质量在一个单词相似度任务中进行评估，并将结果与先前基于不同类型神经网络的最先进的技术进行比较。我们观察到不但准确性的大幅提高，而且计算成本低得多，即，从16亿个单词的数据集中学习高质量的单词向量只需不到一天的时间。 此外，我们证明了这些向量在评估语法和语义的相似性的测试集上提供了 state-of-the-art 的表现。
>

## 1 Introduction

Many current NLP systems and techniques treat words as atomic units - there is no notion of similarity between words, as these are represented as indices in a vocabulary. This choice has several good reasons - simplicity, robustness and the observation that simple models trained on huge amounts of data outperform complex systems trained on less data. An example is the popular N-gram model used for statistical language modeling - today, it is possible to train N-grams on virtually all available data (trillions of words [3]).

However, the simple techniques are at their limits in many tasks. For example, the amount of relevant in-domain data for automatic speech recognition is limited - the performance is usually dominated by the size of high quality transcribed speech data (often just millions of words). In machine translation, the existing corpora for many languages contain only a few billions of words or less. Thus, there are situations where simple scaling up of the basic techniques will not result in any significant progress, and we have to focus on more advanced techniques.

With progress of machine learning techniques in recent years, it has become possible to train more complex models on much larger data set, and they typically outperform the simple models. Probably the most successful concept is to use distributed representations of words [10]. For example, neural network based language models significantly outperform N-gram models [1, 27, 17].

> 当前的许多NLP系统和技术都将单词视为原子单位，词之间没有相似性的概念，因为它们在词典中表示为索引。这种选择有几个很好的理由：simplicity，robustness 以及观察到用大量数据训练的简单模型优于用较少数据训练的复杂系统。一个例子是用于统计语言建模的较为流行的 N-gram 模 型，如今，可以在几乎所有可用数据（万亿个单词[3]）上训练 N-gram。
>
> 但是，简单的技术在许多任务中都处于极限。 例如，用于自动语音识别的相关领域内数据的数量是有限的——性能通常取决于高质量转录语音数据的大小(通常只有数百万词)。 在机器翻译中，许多语言的现有语料库仅包含数十亿个单词或更少。因此，在某些情况下，简单扩展基本技术不会有任何重大进展，我们必须专注于更高级的技术。
>
> 随着近年来机器学习技术的进步，在更大的数据集上训练更复杂的模型成为可能，它们通常比简单的模型表现更好。可能最成功的概念是使用单词[10]的分布式表示。例如，基于神经网络的语言模型明显优于 N-gram 模型[1,27,17]。

#### 1.1 Goals of the Paper

The main goal of this paper is to introduce techniques that can be used for learning high-quality word vectors from huge data sets with billions of words, and with millions of words in the vocabulary. As far as we know, none of the previously proposed architectures has been successfully trained on more than a few hundred of millions of words, with a modest dimensionality of the word vectors between 50 - 100.

We use recently proposed techniques for measuring the quality of the resulting vector representations, with the expectation that not only will similar words tend to be close to each other, but that words can have multiple degrees of similarity [20]. This has been observed earlier in the context of inflectional languages - for example, nouns can have multiple word endings, and if we search for similar words in a subspace of the original vector space, it is possible to find words that have similar endings [13, 14].

Somewhat surprisingly, it was found that similarity of word representations goes beyond simple syntactic regularities. Using a word offset technique where simple algebraic operations are performed on the word vectors, it was shown for example that $vector("King") - vector("Man") + vector("Woman")$ results in a vector that is closest to the vector representation of the word Queen [20].

In this paper, we try to maximize accuracy of these vector operations by developing new model architectures that preserve the linear regularities among words. We design a new comprehensive test set for measuring both syntactic and semantic regularities1(The test set is available at [here](www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt)), and show that many such regularities can be learned with high accuracy. Moreover, we discuss how training time and accuracy depends on the dimensionality of the word vectors and on the amount of the training data.

> 本文的主要目标是介绍一些可用于从拥有数十亿词，且词典大小达百万的庞大语料中学习高质量的词向量的技术。据我们所知，之前提出的模型结构都没有成功地训练过超过数亿个单词(词向量的维度在50 - 100之间)。
>
> 我们使用最近提出的技术来衡量产生的向量表示的质量，期望不仅相似的单词会趋向于彼此接近，而且单词可以有多个相似程度[20]。
>
> 早在屈折(inflectional )语言的上下文中已经观察到这一点——例如，名词可以有多个结尾，并且如果我们在原始向量空间的子空间中搜索相似的单词，就有可能找到具有相似结尾的单词[13,14]。
>
> 令人惊讶地发现，词表示的相似性超出了简单的语法规律性。 使用对词向量执行简单代数运算的 word offset 技术，例如， 
> $$
> vector("King") - vector("Man") + vector("Woman")
> $$
> 产生的向量最接近 Queen[20]的词向量表示。
>
> **在本文中，我们试图通过开发新的模型架构来最大限度地提高这些向量运算的准确性，以保持词之间的线性规律。** 
>
> 我们设计了一种新的综合测试集来测量语法和语义规则1，并且表明许多这样的规则可以被高精度地学习。此外，我们讨论词向量的维数和训练数据的数量对词向量表示的精度和训练时间的影响。

#### 1.2 Previous Work

Representation of words as continuous vectors has a long history [10, 26, 8]. A very popular model architecture for estimating neural network language model (NNLM) was proposed in [1], where a feedforward neural network with a linear projection layer and a non-linear hidden layer was used to learn jointly the word vector representation and a statistical language model. This work has been followed by many others.

Another interesting architecture of NNLM was presented in [13, 14], where the word vectors are first learned using neural network with a single hidden layer. The word vectors are then used to train the NNLM. Thus, the word vectors are learned even without constructing the full NNLM. In this work, we directly extend this architecture, and focus just on the first step where the word vectors are learned using a simple model.

It was later shown that the word vectors can be used to significantly improve and simplify many NLP applications [4, 5, 29]. Estimation of the word vectors itself was performed using different model architectures and trained on various corpora [4, 29, 23, 19, 9], and some of the resulting word vectors were made available for future research and comparison2 . However, as far as we know, these architectures were significantly more computationally expensive for training than the one proposed in [13], with the exception of certain version of log-bilinear model where diagonal weight matrices are used [23].

> 将词表示为连续向量已经有很长的历史了[10,26,8]。 在[1]中提出了一种非常流行的用于 estimating 神经网络语言模型（NNLM）的模型架构，其中使用具有 线性投影层 和 非线性隐藏层 的 前馈神经网络 来共同学习单词向量表示和统计语言模型。这项工作已被其他许多人关注。
>
> NNLM的另一种有趣的架构在[13，14]中提出，其中首先使用具有单个隐藏层的神经网络来学习词向量。 然后，将词向量用于训练 NNLM。 因此，即使不构建完整的 NNLM，也可以学习词向量。在这项工作中，我们直接扩展了这个模型架构，并只关注第一步，即使用一个简单的模型学习词向量。
>
> 后来的研究表明，词向量可以用来显著改善和简化许多自然语言处理应用[4,5,29]。使用不同的模型架构对词向量本身进行评估，并在各种语料库上进行训练[4,29,23,19,9]，得到的部分词向量可供未来研究和比较2使用。然而，据我们所知，除了使用 diagonal weight matrices [23] 的 log-bilinear 模型的某些版本外，这些架构在训练方面的计算成本明显高于 [13] 中提出的架构。

## 2 Model Architectures

Many different types of models were proposed for estimating continuous representations of words, including the well-known Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA). In this paper, we focus on distributed representations of words learned by neural networks, as it was previously shown that they perform significantly better than LSA for preserving linear regularities among words [20, 31]; LDA moreover becomes computationally very expensive on large data sets.

Similar to [18], to compare different model architectures we define first the computational complexity of a model as the number of parameters that need to be accessed to fully train the model. Next, we will try to maximize the accuracy, while minimizing the computational complexity.

For all the following models, the training complexity is proportional to
$$
O = E \times T \times Q \ , \qquad (1)
$$
where $E$ is number of the training epochs, $T$ is the number of the words in the training set and $Q$ is defined further for each model architecture. Common choice is $E = 3 − 50$ and $T$ up to one billion. All models are trained using stochastic gradient descent and backpropagation [26].

> 许多不同类型的模型来评估词的连续表示，包括众所周知的 潜在语义分析(LSA)(注：SVD实现) 和 潜在狄利克雷分配(LDA)()。在本文中，我们关注的是通过神经网络学习的词的分布式表示，因为之前的研究表明，在保持单词之间的线性规律方面，神经网络的表现明显优于 LSA [20,31];此外，LDA在大型数据集上的计算成本非常高。
>
> 与[18]类似，为了比较不同的模型架构，我们首先将模型的计算复杂度定义为完全训练模型需要访问的参数的数量。接下来，我们将尝试使准确性最大化，同时使计算复杂度最小化。
>
> 对于以下所有模型，训练复杂度都成正比：
> $$
> O = E \times T \times Q \ , \qquad (1)
> $$
> 其中$E$为训练 epoch 数，$T$为训练集中单词数，$Q$ 为每种模型体系结构的进一步定义(译者注：这应该是每个模型训练一个词需要的参数量？)。一般情况下 $E \in [3,50]$  T 多达十亿 。所有模型都使用 随机梯度下降 和反向传播[26]进行训练。

#### 2.1 Feedforward Neural Net Language Model (NNLM)

The probabilistic feedforward neural network language model has been proposed in [1]. It consists of input, projection, hidden and output layers. At the input layer, $N$ previous words are encoded using 1-of-V coding, where $V$ is size of the vocabulary. The input layer is then projected to a projection layer $P$ that has dimensionality $N \times D$, using a shared projection matrix. As only $N$ inputs are active at any given time, composition of the projection layer is a relatively cheap operation.

The NNLM architecture becomes complex for computation between the projection and the hidden layer, as values in the projection layer are dense. For a common choice of $N = 10$, the size of the projection layer ($P$) might be 500 to 2000, while the hidden layer size $H$ is typically 500 to 1000 units. Moreover, the hidden layer is used to compute probability distribution over all the words in the vocabulary, resulting in an output layer with dimensionality $V$ . Thus, the computational complexity per each training example is
$$
Q = N × D + N × D × H + H × V, \qquad (2)
$$
where the dominating term is $H × V$ . However, several practical solutions were proposed for avoiding it; either using hierarchical versions of the softmax [25, 23, 18], or avoiding normalized models completely by using models that are not normalized during training [4, 9]. With binary tree representations of the vocabulary, the number of output units that need to be evaluated can go down to around $log2(V)$. Thus, most of the complexity is caused by the term $N × D × H$.

In our models, we use hierarchical softmax where the vocabulary is represented as a Huffman binary tree. This follows previous observations that the frequency of words works well for obtaining classes in neural net language models [16]. Huffman trees assign short binary codes to frequent words, and this further reduces the number of output units that need to be evaluated: while balanced binary tree would require $log_2(V)$ outputs to be evaluated, the Huffman tree based hierarchical softmax requires only about $log_2$(Unigram perplexity$(V)$). For example when the vocabulary size is one million words, this results in about two times speedup in evaluation. While this is not crucial speedup for neural network LMs as the computational bottleneck is in the $N×D×H$ term, we will later propose architectures that do not have hidden layers and thus depend heavily on the efficiency of the softmax normalization.

> 在[1]中提出了概率前馈神经网络语言模型。它由输入层、投影层、隐藏层和输出层组成。在输入层，前面的 $N$ 个单词使用1-of-$V$编码(one-hot)，其中 $V$ 是词汇表的大小。然后使用共享的投影矩阵(将词投影成词向量)，将输入层投影到维度为 $N × D$ 的投影层 $P$ 上。由于在任何给定时间内只有 $N$ 个输入是活跃的，所以投影层的合成是一个代价相对少的操作。
>
> NNLM结构在投影层和隐藏层之间的计算变得很复杂，因为投影层中的值很密集。 对于一个常见的 $ N = 10 $ ，投影层的大小（$ P $）可能为 500 到 2000，而隐藏层的大小 $H$ 通常为 500 到 1000 个单位。 隐藏层用来计算词汇表中所有单词的概率分布，得到一个维数为 $V$ 的输出层。因此，每个训练实例的计算复杂度为：
> $$
> Q = N × D + N × D × H + H × V, \qquad (2)
> $$
> 其中主导项是 $H × V$。 但是，为避免这种情况提出了一些实际的解决方案。 要么使用 hierarchical softmax[25，23，18]，要么使用在训练过程中未 归一化的模型来完全避免归一化模型[4，9]。 用词汇表的二叉树表示，需要评估的输出单位数可以降低到  $log_2(V)$ 左右。 因此，大多数复杂性是由$N×D×H$ 导致的。
>
> 在我们的模型中，我们使用 Hierarchical softmax, 其中将词典表示为哈夫曼树。这与之前的观察结果一致，即在神经网络语言模型[16]中，单词的频率对于获取类很有效(？？可能意思是哈夫曼树通过词频来获取词的类)。哈夫曼树将短二进制码分配给频繁的单词，这进一步减少了需要评估的输出单元数量：平衡二叉树需要评估 $log2(V)$ 输出，而基于哈夫曼树的 hierarchical softmax, 只需要 $log_2$(Unigram perplexity($V$))。例如，当词汇量为 100万 个单词时，这会使评估速度加快两倍左右(？？从n缩减到logn只加快了两倍左右？)。虽然这对于神经网络语言模型来说不是关键的加速，因为计算瓶颈在 $N×D×H$ 项中，我们稍后将提出没有隐藏层的架构，因此严重依赖于 softmax 归一化的效率。(？？这里应该是去掉统治项 $H \times V$后计算瓶颈在$N \times D \times H$。)

#### 2.2 Recurrent Neural Net Language Model (RNNLM)

Recurrent neural network based language model has been proposed to overcome certain limitations of the feedforward NNLM, such as the need to specify the context length (the order of the model N), and because theoretically RNNs can efficiently represent more complex patterns than the shallow neural networks [15, 2]. The RNN model does not have a projection layer; only input, hidden and output layer. What is special for this type of model is the recurrent matrix that connects hidden layer to itself, using time-delayed connections. This allows the recurrent model to form some kind of short term memory, as information from the past can be represented by the hidden layer state that gets updated based on the current input and the state of the hidden layer in the previous time step. 

The complexity per training example of the RNN model is
$$
Q = H × H + H × V, \qquad (3)
$$
where the word representations $D$ have the same dimensionality as the hidden layer $H$. Again, the term $H × V$ can be efficiently reduced to $H × log2(V)$ by using hierarchical softmax. Most of the complexity then comes from $H × H$.

> 基于递归神经网络的语言模型已被提出，以克服前馈 NNLM 的某些局限性，如需要指定上下文长度(模型 N 的 阶数)，以及因为理论上 RNNs 可以比浅神经网络有效地表示更复杂的模式[15,2]。RNN模型没有投影层;只有输入、隐藏和输出层。这类模型的特别之处在于递归矩阵，它使用延时连接将隐藏层与自身连接起来。这使得循环模型可以形成某种短期记忆，因为来自过去的信息可以用隐藏层状态来表示，该隐藏层状态会根据当前输入和前一个时间阶段的隐藏层状态进行更新。
>
> RNN模型每个训练实例的复杂度为
> $$
> Q = H × H + H × V, \qquad (3)
> $$
> 其中词的表示 $D$ 与隐藏层 $H$ 的维数相同。再一次的，$H × V$ 项通过使用 hierarchical softmax 可以有效地减少到 $H × log2(V)$ 。大多数复杂性来自于$H × H$。

#### 2.3 Parallel Training of Neural Networks

To train models on huge data sets, we have implemented several models on top of a large-scale distributed framework called DistBelief [6], including the feedforward NNLM and the new models proposed in this paper. The framework allows us to run multiple replicas of the same model in parallel, and each replica synchronizes its gradient updates through a centralized server that keeps all the parameters. For this parallel training, we use mini-batch asynchronous gradient descent with an adaptive learning rate procedure called Adagrad [7]. Under this framework, it is common to use one hundred or more model replicas, each using many CPU cores at different machines in a data center

> 为了在大数据集上训练模型，我们在一个名为 DistBelief[6]的大规模分布式框架上实现了几个模型，包括 前馈NNLM和本文提出的新模型。该框架允许我们并行运行同一模型的多个副本，并且每个副本都通过保留所有参数的集中式服务器同步其梯度更新。在这个并行训练中，我们使用了小批量异步梯度下降和一个称为Adagrad[7]的自适应学习率程序。在这个框架下，通常使用100个或更多的模型副本，每个副本在一个数据中心的不同机器上使用多个CPU核。

## 3 New Log-linear Models

In this section, we propose two new model architectures for learning distributed representations of words that try to minimize computational complexity. The main observation from the previous section was that most of the complexity is caused by the non-linear hidden layer in the model. While this is what makes neural networks so attractive, we decided to explore simpler models that might not be able to represent the data as precisely as neural networks, but can possibly be trained on much more data efficiently.

The new architectures directly follow those proposed in our earlier work [13, 14], where it was found that neural network language model can be successfully trained in two steps: first, continuous word vectors are learned using simple model, and then the N-gram NNLM is trained on top of these distributed representations of words. While there has been later substantial amount of work that focuses on learning word vectors, we consider the approach proposed in [13] to be the simplest one. Note that related models have been proposed also much earlier [26, 8].

> 在本节中，我们提出了两种新的模型结构来学习词的分布式表示，以尽量减少计算复杂度。上一节的主要观察结果是，大部分的复杂度是由模型中的非线性隐藏层造成的。虽然这正是神经网络如此吸引人的原因，但我们决定探索更简单的模型，这些模型可能不能像神经网络那样精确地表示数据，但可能可以更有效地对数据进行训练。
>
> 新的结构直接遵循我们之前提出的结构(13、14)，发现NNLM模型可以通过两个步骤成功地进行训练:
>
> - 首先, 使用简单模型学习连续的词向量 ,然后使用N- gram NNLM在这些单词的分布式表示之上训练。虽然后来有大量的工作集中在学习词向量上，但我们认为在[13]中提出的方法是最简单的。请注意，相关模型也已经提早提出[26，8]。

#### 3.1 Continuous Bag-of-Words Model

The first proposed architecture is similar to the feedforward NNLM, where the non-linear hidden layer is removed and the projection layer is shared for all words (not just the projection matrix); thus, all words get projected into the same position (their vectors are averaged). We call this architecture a bag-of-words model as the order of words in the history does not influence the projection. Furthermore, we also use words from the future; we have obtained the best performance on the task introduced in the next section by building a log-linear classifier with four future and four history words at the input, where the training criterion is to correctly classify the current (middle) word. Training complexity is then
$$
Q = N \times D + D \times log_2(V) \qquad (4)
$$
We denote this model further as CBOW, as unlike standard bag-of-words model, it uses continuous distributed representation of the context. The model architecture is shown at Figure 1. Note that the weight matrix between the input and the projection layer is shared for all word positions in the same way as in the NNLM.

> 第一个提出的架构类似于前馈的NNLM，其中去掉了非线性隐含层，投影层为所有单词共享(不仅仅是投影矩阵); 因此，所有词都被映射到到相同的位置(它们的向量被平均)。我们称这种结构为“词袋模型”，因为历史上的单词顺序并不影响映射。此外，我们也会使用未来的词 ; 通过构建一个对数线性分类器，输入中包含四个未来和四个历史词（可理解为$context(w) \ /\ w$），我们的训练准则是正确分类当前（中间）单词，从而在下一节介绍的任务上获得了最佳表现。那么训练的复杂度是
> $$
> Q = N \times D + D \times log_2(V) \qquad (4)
> $$
> 我们进一步表示这个模型为 CBOW，与标准的词袋模型不同，它使用了上下文的连续分布式表示(意思应该是使用了一个词的上下文的词向量？)。模型结构如图1所示。请注意，输入层和投影层之间的权值矩阵与 NNLM 中相同，对于所有词位置都是共享的。

#### 3.2 Continuous Skip-gram Model

The second architecture is similar to CBOW, but instead of predicting the current word based on the context, it tries to maximize classification of a word based on another word in the same sentence. More precisely, we use each current word as an input to a log-linear classifier with continuous projection layer, and predict words within a certain range before and after the current word. We found that increasing the range improves quality of the resulting word vectors, but it also increases the computational complexity. Since the more distant words are usually less related to the current word than those close to it, we give less weight to the distant words by sampling less from those words in our training examples.

The training complexity of this architecture is proportional to
$$
Q = C × (D + D × log2(V )) \qquad (5)
$$
where $C$ is the maximum distance of the words. Thus, if we choose $C = 5$, for each training word we will select randomly a number $R$ in range $< 1 ; C >$, and then use $R$ words from history and $R$ words from the future of the current word as correct labels. This will require us to do $R × 2$ word classifications, with the current word as input, and each of the $R + R$ words as output. In the following experiments, we use $C = 10$.

![](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Fig1.png)

**Figure 1: New model architectures. The CBOW architecture predicts the current word based on the context, and the Skip-gram predicts surrounding words given the current word.**

> 第二种体系结构与CBOW类似，但它不是根据上下文预测当前单词，而是根据同一句子中的另一个单词最大限度地分类一个单词。
>
> 我们将每个当前词映射为连续投影层用作对数线性分类器的输入，并预测当前词前后的特定范围内的词。
>
> 我们发现，增加范围可以提高所得词向量的质量，但同时也会增加计算复杂度。 由于距离较远的词通常与当前词的相关性比与距离最近的词的关联性小，因此我们在训练示例中通过从这些词中进行降采样来给予距离较远的单词较少的权重。
>
> 这种结构的训练复杂度是：
> $$
> Q = C × (D + D × log2(V )) \qquad (5)
> $$
> 其中$ C $是单词的最大距离。 因此，如果我们选择$ C = 5 $，则对于每个训练词，我们将随机选择在 $< 1 ; C >$ 范围里的数字 $R$ , 然后使用当前词历史的 $ R $ 个单词和未来的 $ R $ 个单词作为正样本（$context(w) \ /\ w$）。 这将需要我们对 $ R×2 $ 词进行分类，将当前词作为输入，并将每个 $ R + R $ 单词作为输出。 在以下实验中，我们使用 $ C = 10 $。

## 4 Results

To compare the quality of different versions of word vectors, previous papers typically use a table showing example words and their most similar words, and understand them intuitively. Although it is easy to show that word France is similar to Italy and perhaps some other countries, it is much more challenging when subjecting those vectors in a more complex similarity task, as follows. We follow previous observation that there can be many different types of similarities between words, for example, word big is similar to bigger in the same sense that small is similar to smaller. Example of another type of relationship can be word pairs big - biggest and small - smallest [20]. We further denote two pairs of words with the same relationship as a question, as we can ask: ”What is the word that is similar to small in the same sense as biggest is similar to big?”

Somewhat surprisingly, these questions can be answered by performing simple algebraic operations with the vector representation of words. To find a word that is similar to small in the same sense as biggest is similar to big, we can simply compute vector X = vector(”biggest”)−vector(”big”) + vector(”small”). Then, we search in the vector space for the word closest to X measured by cosine distance, and use it as the answer to the question (we discard the input question words during this search). When the word vectors are well trained, it is possible to find the correct answer (word smallest) using this method.

Finally, we found that when we train high dimensional word vectors on a large amount of data, the resulting vectors can be used to answer very subtle semantic relationships between words, such as a city and the country it belongs to, e.g. France is to Paris as Germany is to Berlin. Word vectors with such semantic relationships could be used to improve many existing NLP applications, such as machine translation, information retrieval and question answering systems, and may enable other future applications yet to be invented.

![](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table1.png)

**Table 1: Examples of five types of semantic and nine types of syntactic questions in the SemanticSyntactic Word Relationship test set**

> 为了比较不同版本的词向量的质量，以前的论文通常使用一个表格来显示示例词及其最相似的词，并直观地理解它们。尽管很容易证明“法国”一词与"意大利"以及也许其他一些国家相似，但是在对这些向量进行更复杂的相似性任务时，挑战性却要大得多，如下所示。
>
> 我们根据前面的观察发现，词之间可以有许多不同类型的相似之处，例如，单词 big 和 bigger 的相似之处就像单词 small 和 smaller 的相似之处一样。另一种类型的关系可以是单词对 big - biggest和 small -smallest[20]。我们还可以问两对与一个问题具有相同关系的单词，就像我们可以问：“与small相似且和 biggest 和 big 具有相同意义的单词是什么?”
>
> 出乎意料的是，这些问题可以通过对单词的向量表示进行简单的代数运算来回答。要找到一个与 small相似的单词，就像 biggest 与 big 相似一样，我们可以简单地计算向量 $X = vector("biggest") − vector("big") + vector(“small”)$。然后，我们在向量空间中搜索由余弦距离度量的最接近 $X$ 的单词，并使用它作为问题的答案(在搜索过程中我们丢弃输入的问题单词)。当词向量训练充分时，使用这种方法可以找到正确的答案(单词 smallest)。
>
> 最后，我们发现当我们在大量数据上训练高维词向量时，所得向量可用于回答单词之间的非常微妙的语义关系，例如城市和它所属的国家/地区，例如 法国去巴黎，德国去柏林。 具有这种语义关系的词向量可以用于改善许多现有的NLP应用程序，例如机器翻译，信息检索和问答系统，并且可以使尚未发明的其他未来应用程序成为可能。

#### 4.1 Task Description 

To measure quality of the word vectors, we define a comprehensive test set that contains five types of semantic questions, and nine types of syntactic questions. Two examples from each category are shown in Table 1. Overall, there are 8869 semantic and 10675 syntactic questions. The questions in each category were created in two steps: first, a list of similar word pairs was created manually. Then, a large list of questions is formed by connecting two word pairs. For example, we made a list of 68 large American cities and the states they belong to, and formed about 2.5K questions by picking two word pairs at random. We have included in our test set only single token words, thus multi-word entities are not present (such as New York).

We evaluate the overall accuracy for all question types, and for each question type separately (semantic, syntactic). Question is assumed to be correctly answered only if the closest word to the vector computed using the above method is exactly the same as the correct word in the question; synonyms are thus counted as mistakes. This also means that reaching 100% accuracy is likely to be impossible, as the current models do not have any input information about word morphology. However, we believe that usefulness of the word vectors for certain applications should be positively correlated with this accuracy metric. Further progress can be achieved by incorporating information about structure of words, especially for the syntactic questions.

> 为了衡量词向量的质量，我们定义了一个包含 5 类语义题和 9 类语法题的综合测试集。表1显示了每个类别中的两个示例。
> 总共有 8869 个语义问题和 10675 个语法问题。每个类别的问题是通过两个步骤创建的:首先，手动创建一个相似的词对列表。然后，将两个词对连接起来，形成一个大的问题列表。例如，我们做了一个包含 68 个美国大城市及其所属州的列表，并通过随机选择两个词对形成了大约 2500 个问题。我们在测试集中只包含单个标记词，因此不存在多词实体(如New York)。
>
> 我们评估所有问题类型以及每种问题类型（语义，句法）的总体准确性。 仅当与使用上述方法计算出的向量最接近的词与问题中的正确词完全相同时，才认为问题得到了正确答案。 因此，同义词被视为错误。 这也意味着达到100％的准确性很可能是不可能的，因为当前模型没有任何有关词法的输入信息。 但是，我们认为，单词向量在某些应用中的有用性应与此精度指标正相关。 通过合并有关单词结构的信息，尤其是针对句法问题的信息，可以取得进一步的进步。

#### 4.2 Maximization of Accuracy

We have used a Google News corpus for training the word vectors. This corpus contains about 6B tokens. We have restricted the vocabulary size to 1 million most frequent words. Clearly, we are facing time constrained optimization problem, as it can be expected that both using more data and higher dimensional word vectors will improve the accuracy. To estimate the best choice of model architecture for obtaining as good as possible results quickly, we have first evaluated models trained on subsets of the training data, with vocabulary restricted to the most frequent 30k words. The results using the CBOW architecture with different choice of word vector dimensionality and increasing amount of the training data are shown in Table 2.

It can be seen that after some point, adding more dimensions or adding more training data provides diminishing improvements. So, we have to increase both vector dimensionality and the amount of the training data together. While this observation might seem trivial, it must be noted that it is currently popular to train word vectors on relatively large amounts of data, but with insufficient size (such as 50 - 100). Given Equation 4, increasing amount of training data twice results in about the same increase of computational complexity as increasing vector size twice. For the experiments reported in Tables 2 and 4, we used three training epochs with stochastic gradient descent and backpropagation. We chose starting learning rate 0.025 and decreased it linearly, so that it approaches zero at the end of the last training epoch.

![Table2](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table2.png)

**Table 2: Accuracy on subset of the Semantic-Syntactic Word Relationship test set, using word vectors from the CBOW architecture with limited vocabulary. Only questions containing words from the most frequent 30k words are used.**

![Table3](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table3.png)

**Table 3: Comparison of architectures using models trained on the same data, with 640-dimensional word vectors. The accuracies are reported on our Semantic-Syntactic Word Relationship test set, and on the syntactic relationship test set of [20]**

![Table4](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table4.png)

**Table 4: Comparison of publicly available word vectors on the Semantic-Syntactic Word Relationship test set, and word vectors from our models. Full vocabularies are used.**

> 我们已经使用Google新闻语料库来训练单词向量。 该语料库包含约 6B tokens。 我们已将词汇量限制为一百万个最常用词。 显然，我们面临时间受限的优化问题，可预见到的，**使用更多的数据和更高维的词向量都将提高准确性。** 为了估算模型架构的最佳选择，以便快速获得尽可能好的结果，我们首先评估了在训练数据子集上训练的模型，并将词汇限制为词频最高的30k单词。 表2 显示了使用 CBOW 架构并选择不同的单词向量维数和增加训练数据量的结果。
>
> 可以看出，经过一段时间之后，添加更多的维度或添加更多的训练数据将减少改进。 因此，我们必须同时增加向量维数和训练数据量。 尽管这种观察看似微不足道，但必须指出的是，目前流行的是在相对大量的数据上训练单词向量，但其大小不足（例如50-100）。 给定公式4，**训练数据量增加两倍会导致计算复杂度的增加与向量尺寸增加两次所导致的增加大致相同。** 对于 表2 和 表4 中报道的实验，我们用 SGD 和 BP 训练了 3 个 epoch。 我们选择开始学习率 0.025 并线性降低它，以使其在最后一个训练 epoch 结束时接近零。

#### 4.3 Comparison of Model Architectures

First we compare different model architectures for deriving the word vectors using the same training data and using the same dimensionality of 640 of the word vectors. In the further experiments, we use full set of questions in the new Semantic-Syntactic Word Relationship test set, i.e. unrestricted to the 30k vocabulary. We also include results on a test set introduced in [20] that focuses on syntactic similarity between words3 .

The training data consists of several LDC corpora and is described in detail in [18] (320M words, 82K vocabulary). We used these data to provide a comparison to a previously trained recurrent neural network language model that took about 8 weeks to train on a single CPU. We trained a feedforward NNLM with the same number of 640 hidden units using the DistBelief parallel training [6], using a history of 8 previous words (thus, the NNLM has more parameters than the RNNLM, as the projection layer has size 640 × 8).

In Table 3, it can be seen that the word vectors from the RNN (as used in [20]) perform well mostly on the syntactic questions. The NNLM vectors perform significantly better than the RNN - this is not surprising, as the word vectors in the RNNLM are directly connected to a non-linear hidden layer. The CBOW architecture works better than the NNLM on the syntactic tasks, and about the same on the semantic one. Finally, the Skip-gram architecture works slightly worse on the syntactic task than the CBOW model (but still better than the NNLM), and much better on the semantic part of the test than all the other models.

Next, we evaluated our models trained using one CPU only and compared the results against publicly available word vectors. The comparison is given in Table 4. The CBOW model was trained on subset of the Google News data in about a day, while training time for the Skip-gram model was about three days.

For experiments reported further, we used just one training epoch (again, we decrease the learning rate linearly so that it approaches zero at the end of training). Training a model on twice as much data using one epoch gives comparable or better results than iterating over the same data for three epochs, as is shown in Table 5, and provides additional small speedup.

![Table5](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table5.png)

**Table 5: Comparison of models trained for three epochs on the same data and models trained for one epoch. Accuracy is reported on the full Semantic-Syntactic data set.**

> 首先，我们比较了不同模型结构在使用相同训练数据和相同维度(640)的词向量导出词向量的。在进一步的实验中，我们在新的语义-句法词汇关系测试集中使用全组问题，即不受30k词汇的限制。我们还包括[20]中引入的测试集的结果，该测试集关注单词3之间的语法相似性。
>
> 训练数据由几个LDC语料库组成，在[18] (3.2亿单词，82K单词) 中进行了详细描述。我们使用这些数据来与之前在单个CPU上训练约 8周的 RNNLM 模型进行比较。我们使用 DistBelief 并行训练 [6]，使用 $8$ 个先前单词的历史训练了 前向 NNLM，其隐藏单元数为 $640$个（因此，NNLM比RNNLM具有更多的参数，因为投影层的大小为 $640 × 8$ ）。
>
> 在 表3 中，可以看到 RNN 中的词向量(如[20]中所使用的)主要在语法问题上表现良好。 NNLM 向量的性能明显优于 RNN ——这并不奇怪，因为 NNLM 中的词向量直接与非线性隐含层相连。在语法任务上， CBOW 结构比 NNLM 工作得更好，在语义任务上也差不多。最后， Skip-gram 结构在语法任务上比 CBOW 模型稍差一些(但仍然比 NNLM 好)，在测试的语义部分也比所有其他模型好得多。
>
> 接下来，我们评估了仅使用一个 CPU 训练的模型，并将结果与可公开获得的词向量进行了比较。 表4中给出了比较。 CBOW 模型在大约一天的时间里对 Google 新闻数据的子集进行了训练，而 Skip-gram 模型的训练时间大约为三天。
>
> 对于进一步的实验报告，我们只使用了一个训练 epoch (再次，我们线性降低学习率，使它在训练结束时接近零)。**使用一个 epoch 在两倍数量的数据上训练模型，得到的结果与在三个 epoch 中迭代相同数据相比是相当的或更好的，如表5所示，并提供额外的小加速。**

#### 4.4 Large Scale Parallel Training of Models

As mentioned earlier, we have implemented various models in a distributed framework called DistBelief. Below we report the results of several models trained on the Google News 6B data set, with mini-batch asynchronous gradient descent and the adaptive learning rate procedure called Adagrad [7]. We used 50 to 100 model replicas during the training. The number of CPU cores is an estimate since the data center machines are shared with other production tasks, and the usage can fluctuate quite a bit. Note that due to the overhead of the distributed framework, the CPU usage of the CBOW model and the Skip-gram model are much closer to each other than their single-machine implementations. The result are reported in Table 6.

![Table6](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table6.png)

**Table 6: Comparison of models trained using the DistBelief distributed framework. Note that training of NNLM with 1000-dimensional vectors would take too long to complete.**

![Table7](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table7.png)

**Table 7: Comparison and combination of models on the Microsoft Sentence Completion Challenge.**

> 如前所述，我们已经在称为 DistBelief 的分布式框架中实现了各种模型。 下面我们报告在 Google News 6B 数据集上训练的几种模型的结果，这些模型具有小批量异步梯度下降和称为 Adagrad [7] 的自适应学习速率过程。 在训练期间，我们使用了 50 到 100 个模型副本。 由于数据中心计算机是与其他生产任务共享的，因此 CPU 内核的数量是一个估计值，并且使用情况可能会有很大的波动。 请注意，由于分布式框架的开销，CBOW模型和 Skip-gram 模型的 CPU 使用率比单机实现更接近。 结果记录在 表6 中。

#### 4.5 Microsoft Research Sentence Completion Challenge

The Microsoft Sentence Completion Challenge has been recently introduced as a task for advancing language modeling and other NLP techniques [32]. This task consists of 1040 sentences, where one word is missing in each sentence and the goal is to select word that is the most coherent with the rest of the sentence, given a list of five reasonable choices. Performance of several techniques has been already reported on this set, including N-gram models, LSA-based model [32], log-bilinear model [24] and a combination of recurrent neural networks that currently holds the state of the art performance of 55.4% accuracy on this benchmark [19].

We have explored the performance of Skip-gram architecture on this task. First, we train the 640- dimensional model on 50M words provided in [32]. Then, we compute score of each sentence in the test set by using the unknown word at the input, and predict all surrounding words in a sentence. The final sentence score is then the sum of these individual predictions. Using the sentence scores, we choose the most likely sentence.

A short summary of some previous results together with the new results is presented in Table 7. While the Skip-gram model itself does not perform on this task better than LSA similarity, the scores from this model are complementary to scores obtained with RNNLMs, and a weighted combination leads to a new state of the art result 58.9% accuracy (59.2% on the development part of the set and 58.7% on the test part of the set).

>最近引入了Microsoft完形填空挑战，作为推进语言建模和其他 NLP 技术的一项任务[32]。 该任务由 1040 个句子组成，其中每个句子中缺少一个单词，目标是在给出五个合理选项的情况下，选择与该句子其余部分最相关的单词。 几个技术已经在这个数据集上进行了实验，包括 N-gram 模型， LSA-based 的模型[32]，log-bilinear 模型[24] 以及目前保持最佳表现的的递归神经网络的组合。 在此基准上的准确度为55.4％[19]。
>
>我们已经探索了此任务上 skip-gram 结构的性能。 首先，我们在[32]中提供的 5,000万 个单词上训练 640维 模型。 然后，我们通过使用输入处的未知词来计算测试集中每个句子的分数，并预测句子中所有周围的词。 那么最终的句子分数就是这些单个预测的总和。 使用句子分数，我们选择最可能的句子。（这里指的是通过skip-gram做完形填空的过程）
>
>表7 给出了一些 先前结果 以及 新结果 的简短摘要。尽管 Skip-gram 模型本身在执行此任务方面的表现并不比LSA相似性好，但是该模型的得分与RNNLM获得的得分是互补的，并且 加权组合可带来一个state of art的结果 58.9％ 的准确度 (59.2% on the development part of the set and 58.7% on the test part of the set)。

## 5 Examples of the Learned Relationships

Table 8 shows words that follow various relationships. We follow the approach described above: the relationship is defined by subtracting two word vectors, and the result is added to another word. Thus for example, Paris - France + Italy = Rome. As it can be seen, accuracy is quite good, although there is clearly a lot of room for further improvements (note that using our accuracy metric that assumes exact match, the results in Table 8 would score only about 60%). 

We believe that word vectors trained on even larger data sets with larger dimensionality will perform significantly better, and will enable the development of new innovative applications. Another way to improve accuracy is to provide more than one example of the relationship. By using ten examples instead of one to form the relationship vector (we average the individual vectors together), we have observed improvement of accuracy of our best models by about 10% absolutely on the semantic-syntactic test.

It is also possible to apply the vector operations to solve different tasks. For example, we have observed good accuracy for selecting out-of-the-list words, by computing average vector for a list of words, and finding the most distant word vector. This is a popular type of problems in certain human intelligence tests. Clearly, there is still a lot of discoveries to be made using these techniques.

![Table8](/Users/helloword/Anmingyu/Gor-rok/Papers/Word2vec/EfficientEstimationofWordRepresentationsinVectorSpace/Table8.png)

**Table 8: Examples of the word pair relationships, using the best word vectors from Table 4 (Skipgram model trained on 783M words with 300 dimensionality).**

> 表8显示了遵循各种关系的单词。我们采用上面描述的方法:通过减去两个词向量来定义关系，然后将结果添加到另一个词上。例如，$巴黎 - 法国 + 意大利 = 罗马$。可以看出，精确度相当好，尽管还有很多进一步改进的空间 (请注意，使用假设精确匹配的准确性指标，表8中的结果仅得分约60%)。
>
> 我们相信，在更大的数据集上训练维度更高的词向量将表现得更好，并使开发新的创新应用程序成为可能。另一种提高准确性的方法是提供多个关系示例。通过使用10个例子而不是一个例子来形成关系向量(我们将单个向量平均在一起)，我们观察到我们的最佳模型在语义-句法测试上的准确性提高了大约10%。
>
> 也可以应用向量运算来解决不同的任务。例如，我们已经观察到，通过计算一系列词的平均向量和找到距离最远的词向量，可以很好地选择 out-of-the-list 的词。在某些人类智力测试中，这是一种常见的问题。显然，使用这些技术仍有许多发现有待发现。

## 6 Conclusion

In this paper we studied the quality of vector representations of words derived by various models on a collection of syntactic and semantic language tasks. We observed that it is possible to train high quality word vectors using very simple model architectures, compared to the popular neural network models (both feedforward and recurrent). Because of the much lower computational complexity, it is possible to compute very accurate high dimensional word vectors from a much larger data set. Using the DistBelief distributed framework, it should be possible to train the CBOW and Skip-gram models even on corpora with one trillion words, for basically unlimited size of the vocabulary. That is several orders of magnitude larger than the best previously published results for similar models.

An interesting task where the word vectors have recently been shown to significantly outperform the previous state of the art is the SemEval-2012 Task 2 [11]. The publicly available RNN vectors were used together with other techniques to achieve over 50% increase in Spearman’s rank correlation over the previous best result [31]. The neural network based word vectors were previously applied to many other NLP tasks, for example sentiment analysis [12] and paraphrase detection [28]. It can be expected that these applications can benefit from the model architectures described in this paper.

Our ongoing work shows that the word vectors can be successfully applied to automatic extension of facts in Knowledge Bases, and also for verification of correctness of existing facts. Results from machine translation experiments also look very promising. In the future, it would be also interesting to compare our techniques to Latent Relational Analysis [30] and others. We believe that our comprehensive test set will help the research community to improve the existing techniques for estimating the word vectors. We also expect that high quality word vectors will become an important building block for future NLP applications.

> 本文在一组句法和语义语言任务的基础上，研究了由不同模型得到的词向量表示的质量。我们观察到，与流行的神经网络模型(前馈和递归)相比，使用非常简单的模型架构可以训练高质量的词向量。 由于计算复杂度低得多，因此可以从更大的数据集中计算出非常准确的高维词向量。使用DistBelief分布式框架，即使在具有一万亿个单词的语料库上，也可以训练 CBOW 和 Skip-gram 模型，而词典基本上是无限的。 这比以前发布的同类最佳结果最好的数量级大几个数量级。
>
> 一个有趣的任务是 SemEval-2012 Task 2[11]，在这个任务中，词向量的表现明显优于之前的技术。公开可用的 RNN 向量与其他技术一起使用，使Spearman's rank correlation 比以前的最佳结果提高了50％以上[31]。基于神经网络的词向量已被应用于许多其他的自然语言处理任务，如情感分析[12]和复述检测[28]。可以预期，这些应用程序可以从本文描述的模型架构中受益。
>
> 我们正在进行的工作表明，词向量可以成功地应用于知识库中 facts 的自动扩展，也可以用于验证现有 facts 的正确性。机器翻译实验的结果看起来也很有希望。将来，将我们的技术与潜在关系分析[30]和其他方法进行比较也将很有趣。 我们相信，我们全面的测试集将帮助研究社区改进现有的评估词向量的技术。 我们还期望高质量的词向量将成为未来NLP应用程序的重要组成部分。

## 7 Follow-Up Work

After the initial version of this paper was written, we published single-machine multi-threaded C++ code for computing the word vectors, using both the continuous bag-of-words and skip-gram architectures4 . The training speed is significantly higher than reported earlier in this paper, i.e. it is in the order of billions of words per hour for typical hyperparameter choices. We also published more than 1.4 million vectors that represent named entities, trained on more than 100 billion words. Some of our follow-up work will be published in an upcoming NIPS 2013 paper [21].

> 撰写本文的初始版本后，我们发布了使用连续词袋和skip-gram架构4的单机多线程C ++代码，用于计算单词向量。 训练速度明显高于本文早先报道的速度，即对于典型的超参数选择而言，训练速度约为每小时数十亿个单词。 我们还发布了140万个代表命名实体的向量，并接受了超过1000亿个单词的训练。 我们的一些后续工作将在 NIPS 2013 即将发表的论文中发表[21]。

## References

[1] Y. Bengio, R. Ducharme, P. Vincent. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137-1155, 2003. 

[2] Y. Bengio, Y. LeCun. Scaling learning algorithms towards AI. In: Large-Scale Kernel Machines, MIT Press, 2007. 

[3] T. Brants, A. C. Popat, P. Xu, F. J. Och, and J. Dean. Large language models in machine translation. In Proceedings of the Joint Conference on Empirical Methods in Natural Language Processing and Computational Language Learning, 2007. 

[4] R. Collobert and J. Weston. A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning. In International Conference on Machine Learning, ICML, 2008. 

[5] R. Collobert, J. Weston, L. Bottou, M. Karlen, K. Kavukcuoglu and P. Kuksa. Natural Language Processing (Almost) from Scratch. Journal of Machine Learning Research, 12:2493- 2537, 2011. 

[6] J. Dean, G.S. Corrado, R. Monga, K. Chen, M. Devin, Q.V. Le, M.Z. Mao, M.A. Ranzato, A. Senior, P. Tucker, K. Yang, A. Y. Ng., Large Scale Distributed Deep Networks, NIPS, 2012. 

[7] J.C. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research, 2011. 

[8] J. Elman. Finding Structure in Time. Cognitive Science, 14, 179-211, 1990. 

[9] Eric H. Huang, R. Socher, C. D. Manning and Andrew Y. Ng. Improving Word Representations via Global Context and Multiple Word Prototypes. In: Proc. Association for Computational Linguistics, 2012. 

[10] G.E. Hinton, J.L. McClelland, D.E. Rumelhart. Distributed representations. In: Parallel distributed processing: Explorations in the microstructure of cognition. Volume 1: Foundations, MIT Press, 1986. 

[11] D.A. Jurgens, S.M. Mohammad, P.D. Turney, K.J. Holyoak. Semeval-2012 task 2: Measuring degrees of relational similarity. In: Proceedings of the 6th International Workshop on Semantic Evaluation (SemEval 2012), 2012. 

[12] A.L. Maas, R.E. Daly, P.T. Pham, D. Huang, A.Y. Ng, and C. Potts. Learning word vectors for sentiment analysis. In Proceedings of ACL, 2011. 

[13] T. Mikolov. Language Modeling for Speech Recognition in Czech, Masters thesis, Brno University of Technology, 2007. 

[14] T. Mikolov, J. Kopecky, L. Burget, O. Glembek and J. ´ Cernock ˇ y. Neural network based lan- ´ guage models for higly inflective languages, In: Proc. ICASSP 2009. 

[15] T. Mikolov, M. Karafiat, L. Burget, J. ´ Cernock ˇ y, S. Khudanpur. Recurrent neural network ´ based language model, In: Proceedings of Interspeech, 2010. 

[16] T. Mikolov, S. Kombrink, L. Burget, J. Cernock ˇ y, S. Khudanpur. Extensions of recurrent neural ´ network language model, In: Proceedings of ICASSP 2011. 

[17] T. Mikolov, A. Deoras, S. Kombrink, L. Burget, J. Cernock ˇ y. Empirical Evaluation and Com- ´ bination of Advanced Language Modeling Techniques, In: Proceedings of Interspeech, 2011.

[18] T. Mikolov, A. Deoras, D. Povey, L. Burget, J. Cernock ˇ y. Strategies for Training Large Scale ´ Neural Network Language Models, In: Proc. Automatic Speech Recognition and Understanding, 2011. 

[19] T. Mikolov. Statistical Language Models based on Neural Networks. PhD thesis, Brno University of Technology, 2012. 

[20] T. Mikolov, W.T. Yih, G. Zweig. Linguistic Regularities in Continuous Space Word Representations. NAACL HLT 2013.

[21] T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean. Distributed Representations of Words and Phrases and their Compositionality. Accepted to NIPS 2013. 

[22] A. Mnih, G. Hinton. Three new graphical models for statistical language modelling. ICML, 2007. 

[23] A. Mnih, G. Hinton. A Scalable Hierarchical Distributed Language Model. Advances in Neural Information Processing Systems 21, MIT Press, 2009. 

[24] A. Mnih, Y.W. Teh. A fast and simple algorithm for training neural probabilistic language models. ICML, 2012. 

[25] F. Morin, Y. Bengio. Hierarchical Probabilistic Neural Network Language Model. AISTATS, 2005. 

[26] D. E. Rumelhart, G. E. Hinton, R. J. Williams. Learning internal representations by backpropagating errors. Nature, 323:533.536, 1986. 

[27] H. Schwenk. Continuous space language models. Computer Speech and Language, vol. 21, 2007. 

[28] R. Socher, E.H. Huang, J. Pennington, A.Y. Ng, and C.D. Manning. Dynamic Pooling and Unfolding Recursive Autoencoders for Paraphrase Detection. In NIPS, 2011. 

[29] J. Turian, L. Ratinov, Y. Bengio. Word Representations: A Simple and General Method for Semi-Supervised Learning. In: Proc. Association for Computational Linguistics, 2010. 

[30] P. D. Turney. Measuring Semantic Similarity by Latent Relational Analysis. In: Proc. International Joint Conference on Artificial Intelligence, 2005. 

[31] A. Zhila, W.T. Yih, C. Meek, G. Zweig, T. Mikolov. Combining Heterogeneous Models for Measuring Relational Similarity. NAACL HLT 2013. 

[32] G. Zweig, C.J.C. Burges. The Microsoft Research Sentence Completion Challenge, Microsoft Research Technical Report MSR-TR-2011-129, 2011.