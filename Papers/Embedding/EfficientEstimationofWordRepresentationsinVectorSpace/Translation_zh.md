# Efficient Estimation of Word Representations in Vector Space

## Abstract

我们提出了两种新的模型结构来计算来自大数据的词的连续向量表示。这些表示的质量在一个单词相似度任务中进行评估，并将结果与先前基于不同类型神经网络的最先进的技术进行比较。我们观察到不但准确性的大幅提高，而且计算成本低得多，即，从16亿个单词的数据集中学习高质量的单词向量只需不到一天的时间。 此外，我们证明了这些向量在评估语法和语义的相似性的测试集上提供了 state-of-the-art 的表现。

## 1 Introduction

当前的许多NLP系统和技术都将单词视为原子单位-单词之间没有相似性的概念，因为它们在词汇表中表示为索引。这种选择有几个很好的理由：简单性，鲁棒性以及观察到用大量数据训练的简单模型优于用较少数据训练的复杂系统。一个例子是用于统计语言建模的流行 N-gram 模型-如今，可以在几乎所有可用数据（万亿个单词[3]）上训练 N-gram。

但是，简单的技术在许多任务中都有其天花板。 例如，用于自动语音识别的**相关领域内**数据量是有限的，其性能通常取决于高质量转录语音数据的大小(通常只有数百万词)。 在机器翻译中，许多语言的现有语料库仅包含数十亿个单词或更少。因此，在某些情况下，简单扩展基础技术不会有任何重大进展，我们必须专注于更高级的技术。

随着近年来机器学习技术的进步，在更大的数据集上训练更复杂的模型成为可能，它们通常比简单的模型表现更好。可能最成功的概念是使用词[10]的分布式表示(distributed representation)。例如，基于神经网络的语言模型明显优于 N-gram 模型[1,27,17]。

#### 1.1 Goals of the Paper

本文的主要目标是介绍一些可用于从拥有数十亿词，且词典大小达百万的庞大语料中学习高质量的词向量的技术。据我们所知，之前提出的模型结构都没有成功地训练过超过数亿个单词(词向量的维度在50 - 100之间)。

我们使用最近提出的技术来衡量产生的向量表示的质量，期望不仅相似的单词会趋向于彼此接近，而且单词可以有多个相似程度[20]。

早在屈折(inflectional)语言的上下文中已经观察到这一点——例如，名词可以有多个结尾，并且如果我们在原始向量空间的子空间中搜索相似的单词，就有可能找到具有相似结尾的单词[13,14]。

令人惊讶地发现，词表示的相似性超出了简单的语法规律性。 使用对词向量执行简单代数运算的 word offset 技术，例如， 
$$
vector("King") - vector("Man") + vector("Woman")
$$
产生的向量最接近 Queen[20]的词向量表示。

**在本文中，我们试图通过开发新的模型架构来最大限度地提高这些向量运算的准确性，以保持词之间的线性规律。** 

我们设计了一种新的综合测试集来测量语法和语义规则1，并且表明许多这样的规则可以被高精度地学习。此外，我们讨论词向量的维数和训练数据的数量对词向量表示的精度和训练时间的影响。

#### 1.2 Previous Work

将词表示为连续向量已经有很长的历史了[10,26,8]。 在[1]中提出了一种非常流行的用于训练(注：原文为estimating) 神经网络语言模型（NNLM）的模型架构，其中使用具有 线性投影层 和 非线性隐藏层 的 前馈神经网络 来共同学习单词向量表示和统计语言模型。这项工作已被其他许多人关注。

NNLM 的另一种有趣的架构在[13，14]中提出，其中首先使用具有单个隐藏层的神经网络来学习词向量。 然后，将词向量用于训练 NNLM。 因此，即使不构建完整的 NNLM，也可以学习词向量。在这项工作中，我们直接扩展了这个模型架构，并只关注第一步，即使用一个简单的模型学习词向量。

后来的研究表明，词向量可以用来显著改善和简化许多自然语言处理应用[4,5,29]。使用不同的模型架构对词向量本身进行近似，并在各种语料库上进行训练[4,29,23,19,9]，得到的部分词向量可供未来研究和比较2使用。然而，据我们所知，除了使用 diagonal weight matrices [23] 的 log-bilinear 模型的某些版本外，这些架构在训练方面的计算成本明显高于 [13] 中提出的架构。

## 2 Model Architectures

许多不同类型的模型来估计词的连续表示，包括众所周知的 潜在语义分析(LSA) 和潜在 狄利克雷分配(LDA)。在本文中，我们关注的是通过神经网络学习的词的分布式表示，因为之前的研究表明，在保持单词之间的线性规律方面，神经网络的表现明显优于 LSA [20,31]; 此外，LDA 在大型数据集上的计算成本非常高。

与[18]类似，为了比较不同的模型架构，我们首先将模型的计算复杂度定义为完全训练模型需要访问的参数的数量。接下来，我们将尝试使准确性最大化，同时使计算复杂度最小化。

对于以下所有模型，训练复杂度都与下面公式成正比：
$$
O = E \times T \times Q \ , \qquad (1)
$$
其中$E$ 为训练 epoch 数，$T$ 为训练集中总词数(注：非词典长度)，$Q$ 为每种模型体系结构的进一步定义(注：这应该是每个模型训练一个词需要的参数量)。一般情况下 $E = [3,50]$  $T$ 多达十亿 。所有模型都使用 随机梯度下降 和 反向传播[26] 进行训练。

#### 2.1 Feedforward Neural Net Language Model (NNLM)

在[1]中提出了概率前馈神经网络语言模型。它由输入层、投影层、隐藏层和输出层组成。在输入层，前面的 $N$ 个单词使用1-of-$V$编码(one-hot)，其中 $V$ 是词汇表的大小。然后使用共享的投影矩阵(注：将词投影成词向量)，将输入层投影到维度为 $N × D$ 的投影层 $P$ 上。由于从头到尾只有 $N$ 个输入是被激活，所以投影层的合成是一个代价相对少的操作。

NNLM 架构在投影层和隐藏层之间的计算变得很复杂，因为投影层中的值很密集。 对于一个常见的 $ N = 10 $ ，投影层的大小($ P $)可能为 500 到 2000 (注：词向量维度一般为 50 - 200)，而隐藏层的大小 $H$ 通常为 500 到 1000 个单位。 隐藏层用来计算词典中所有单词的概率分布，得到一个维数为 $V$ 的输出层。因此，每次训练的计算复杂度为：
$$
Q = N × D + N × D × H + H × V, \qquad (2)
$$
其中主导项是 $H × V$。 虽然，为避免这种情况提出了一些可行的解决方案。 要么使用 hierarchical softmax[25，23，18]，或者通过使用在训练时没有归一化的模型 [4,9] 来完全避免归一化模型。 将词典用二叉树表示，需要计算的输出单元数量可以降低到  $log_2(V)$ 左右。 于是，余下部分大多数的复杂性是由 $N×D×H$ 导致的。

在我们的模型中，我们使用 Hierarchical softmax, 其中将词典表示为哈夫曼树。这与之前的观察结果一致，即在神经网络语言模型[16]中，词频对于获取类很有效。(注：这里没太理解，可能意思是哈夫曼树是通过词频建立的，所以获取词很有效？)。哈夫曼树将短的二进制码分给频繁词，这进一步减少了需要计算的输出单元数量：平衡二叉树需要计算 $log2(V)$ 个输出单元，而基于哈夫曼树的 hierarchical softmax, 只需要 $log_2(Unigram\_perplexity(V))$ (注：根据上下文，这的 unigram-perplexity 应该是个开根号之类的函数)。

例如，当词典长度为 100万 时，这会使计算速度加快两倍左右(注：提高两倍应该是相比于平衡二叉树)。虽然这对于神经网络语言模型来说不是关键的加速，因为计算瓶颈在 $N×D×H$ 项中，我们稍后将提出没有隐藏层的架构，因此严重依赖于 softmax 归一化的效率。

#### 2.2 Recurrent Neural Net Language Model (RNNLM)

RNN-based 的语言模型已被提出，以克服前馈 NNLM 的某些局限性，如需要指定上下文长度(模型的阶数 $N$ )，以及因为理论上 RNNs 可以比浅神经网络有效地表示更复杂的模式[15,2]。RNN模型没有投影层; 只有输入、隐藏和输出层。这类模型的特别之处在于递归矩阵，它使用延时连接将隐藏层与自身连接起来。这使得循环模型可以形成某种短期记忆，因为来自过去的信息可以用隐藏层状态来表示，该隐藏层状态会根据当前输入和前一个时间阶段的隐藏层状态进行更新。

RNN模型每次训练的复杂度为
$$
Q = H × H + H × V, \qquad (3)
$$
其中词向量的维度 $D$ 与隐藏层 $H$ 的维数相同。再一次的，$H × V$ 项通过使用 hierarchical softmax 可以有效地减少到 $H × log_2(V)$ 。大多数复杂性来自于$H × H$

#### 2.3 Parallel Training of Neural Networks

为了在大数据集上训练模型，我们在一个名为 DistBelief[6] 的大规模分布式框架上实现了几个模型，包括 feedforward NNLM 和本文提出的新模型。该框架允许我们并行运行同一模型的多个副本，并且每个副本都通过保留所有参数的集中式服务器同步其梯度更新。在这个并行训练中，我们使用了小批量异步梯度下降和一个称为Adagrad[7]的自适应学习率程序。在这个框架下，通常使用100个或更多的模型副本，每个副本在一个数据中心的不同机器上使用多个 CPU 核。

## 3 New Log-linear Models

在本节中，我们提出了两种新的模型结构来学习词的分布式表示，以尽量减少计算复杂度。上一节的主要观察结果是，大部分的复杂度是由模型中的非线性隐藏层造成的。虽然这正是神经网络如此吸引人的原因，但我们决定探索更简单的模型，这些模型可能不能像神经网络那样精确地表示数据，但可能可以更有效地对数据进行训练。

新的结构直接遵循我们之前提出的结构(13、14)，发现 NNLM 模型可以通过两个步骤成功地进行训练:

- 使用简单模型学习连续的词向量 ,
- 使用N-gram NNLM 在这些词向量之上训练。

虽然后来有大量的工作集中在学习词向量上，但我们认为在[13]中提出的方法是最简单的。请注意，相关模型也已经在很早提出[26，8]。

#### 3.1 Continuous Bag-of-Words Model

第一个提出的架构类似于 前馈 NNLM，去掉了非线性隐含层，投影层为所有单词共享(不仅仅是投影矩阵); 因此，所有词都被映射到到相同的位置(它们的向量被平均)。我们称这种架构为“词袋模型”，因为输入词的顺序(注：context words)不影响映射层结果。 通过构建一个 log-linear 分类器，输入中包含四个未来和四个历史词（可理解为$context(w) \ /\ w$），我们的训练准则是正确分类当前（中间）单词，从而在下一节介绍的任务上获得了最佳表现。

训练的复杂度是：
$$
Q = N \times D + D \times log_2(V) \qquad (4)
$$
我们进一步表示这个模型为 CBOW，与标准的词袋模型不同，它使用了上下文的所有词的词向量。模型结构如图1所示。请注意，输入层和投影层之间的权值矩阵与 NNLM 中相同，对于所有词位置都是共享的。

#### 3.2 Continuous Skip-gram Model

第二种架构与 CBOW 类似，但它不是根据上下文预测当前单词，而是根据同一句子中的另一个单词最大限度地分类一个单词。

我们将每个当前词经过投影层映射作为 log-linear 分类器的输入，并预测当前词前后的特定范围内的词。(注：context window)

我们发现，增加范围可以提高所得词向量的质量，但同时也会增加计算复杂度。 由于距离较远的词通常与当前词的相关性比与距离最近的词的关联性小，因此我们在训练中通过从这些词中进行降采样(注：下文里有解释)来给予距离较远的单词较少的权重。

这种架构的训练复杂度是：
$$
Q = C × (D + D × log_2(V )) \qquad (5)
$$
其中 $ C $ 是单词的最大距离。 因此，如果我们选择 $ C = 5 $，则对于每个训练词，我们将随机选择在 $< 1 ; C >$ 范围里的数字 $R$ , 然后使用当前词历史的 $ R $ 个单词和未来的 $ R $ 个单词作为正样本。 这将需要我们对 $ R×2 $ 词进行分类，将当前词作为输入，并将每个 $ R + R $ 单词作为输出。 在以下实验中，我们使用 $ C = 10 $。

![Figure1](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Fig1.png)

**Figure 1: 新模型架构。CBOW 架构根据上下文预测中心词，而 skip-gram 预测中心词的周围单词。**

## 4 Results

为了比较不同版本的词向量的质量，以前的论文通常使用一个表格来显示示例词及其最相似的词，并直观地理解它们。尽管很容易证明 “法国” 一词与 “意大利” 以及也许其他一些国家相似，但是这些向量进行更复杂的相似性任务时，挑战性却要大得多，如下所示。

我们根据前面的观察发现，词与词之间可以有许多不同类型的相似之处，例如，单词 big 和 bigger 的相似之处就像单词 small 和 smaller 的相似之处一样。另一种类型的关系可以是单词对 big - biggest和 small -smallest[20]。我们还可以问两对与一个问题具有相同关系的单词，就像我们可以问：“与small相似且和 biggest 和 big 的相似具有相同意义的单词是什么?”

出乎意料的是，这些问题可以通过对词向量表示进行简单的代数运算来回答。要找到一个与 small 相似的词，就像 biggest 与 big 相似一样，我们可以简单地计算向量
$$
X = vector("biggest") − vector("big") + vector(“small”)
$$
然后，我们在向量空间中搜索由余弦距离度量的离 $X$ 最近的单词，并使用它作为问题的答案(在搜索过程中我们丢弃输入的问题单词)。当词向量训练充分时，使用这种方法可以找到正确的答案(单词 smallest)。

最后，我们发现当我们在大量数据上训练高维词向量时，所得向量可用于回答单词之间的非常微妙的语义关系，例如城市和它所属的国家/地区，例如 法国 - 巴黎，德国 - 柏林。具有这种语义关系的词向量可以用于改善许多现有的 NLP 应用程序，例如机器翻译，信息检索和问答系统，并且可以使尚未发明的其他未来应用程序成为可能。

![Table1](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table1.png)

**Table 1 : 语义语法词关系测试集中的五种语义和九种语法问题的示例**

#### 4.1 Task Description 

为了衡量词向量的质量，我们定义了一个包含 5 类语义题和 9 类语法题的综合测试集。表1显示了每个类别中的两个示例。

总共有 8869 个语义问题和 10675 个语法问题。每个类别的问题是通过两个步骤创建的：首先，手动创建一个相似的词对列表。然后，将两个词对连接起来，形成一个大的问题列表。例如，我们做了一个包含 68 个美国大城市及其所属州的列表，并通过随机选择两个词对形成了大约 2500 个问题。我们在测试集中只包含单个标记词，因此不存在多词实体 (如New York)。

我们评估所有问题类型以及每种问题类型（语义，句法）的总体准确性。 仅当与使用上述方法计算出的向量最接近的词与问题中的正确词完全相同时，才认为问题得到了正确答案。因此，同义词被视为错误。这也意味着达到 100％ 的准确性很可能是不可能的，因为当前模型没有任何有关语法的输入信息。但是，我们认为，词向量在某些应用中的性能与此精度正相关。通过合并有关单词结构的信息，尤其是针对语法问题的信息，可以取得进一步的进步。

#### 4.2 Maximization of Accuracy

我们已经使用Google新闻语料库来训练词向量。该语料库包含约 6B tokens。 我们已将词汇量限制为一百万个频繁词(注：词频取 top 100万)。 显然，我们面临时间受限的优化问题，可预见到的，**使用更多的数据和更高维的词向量都将提高准确性。** 为了估算模型架构的最佳选择，以便快速获得尽可能好的结果，我们首先评估在训练数据子集上训练的模型，并将词典限制为词频最高的 30k 单词。 表2 显示了使用 CBOW 架构并选择不同的单词向量维数和增加训练数据量的结果。

可以看出，在一定程度后，单独添加更多的维度或添加更多的训练数据的效果会减弱。 因此，我们必须同时增加向量维数和训练数据量。 尽管这种观察看似微不足道，但必须指出的是，目前流行的是在相对大量的数据上训练词向量，但其大小不足（例如50-100）。 给定公式4，**训练数据量增加两倍与向量维度增加两倍所导致的计算复杂度的增加大致相同。** 对于 表2 和 表4 中报道的实验，我们用 SGD 和 BP 训练了 3 个 epoch。 我们选择开始学习率 0.025 并线性降低它，以使其在最后一个训练 epoch 结束时接近零。

![Table2](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table2.png)

**表 2 ：使用来自 CBOW 架构且词汇量有限的词向量，语义 - 语法词关系测试集的子集的准确性。只使用包含最频繁的 30k 单词中的问题。**

![Table3](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table3.png)

**表 3：使用 640 维词向量在相同数据上训练的模型架构比较。我们的语义- 语法单词关系测试集和[20]的语法关系测试集都报告了准确率。**

![Table4](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table4.png)

**表 4 : 语义 - 语法词关系测试集中公开可用的词向量和我们模型中的词向量的比较。使用完整的词典。**

#### 4.3 Comparison of Model Architectures

首先，我们比较了不同模型架构在使用相同训练数据和相同维度(640)输出的词向量。在进一步的实验中，我们在新的 语义 - 语法 词汇关系测试集中使用全量问题，即不受 30k 大小的词典的限制。我们还包括[20]中引入的测试集的结果，该测试集关注单词3之间的语法相似性。

训练数据由几个 LDC 语料库组成，在[18] (3.2亿单词，82K单词) 中进行了详细描述。我们使用这些数据来与之前在单个 CPU 上训练约 8 周的 RNNLM 模型进行比较。我们使用 DistBelief 并行训练 [6]，使用 $8$ 个 history 词训练了 feedforward NNLM，其隐藏单元数为 $640$个（因此，NNLM 比 RNNLM 具有更多的参数，因为投影层的大小为 $640 × 8$ ）。

在 表3 中，可以看到 RNN 中的词向量(如[20]中所使用的)主要在语法问题上表现良好。 NNLM 向量的性能明显优于 RNN ——这并不奇怪，因为 NNLM 中的词向量直接与非线性隐藏层链接。在语法任务上， CBOW 结构比 NNLM 工作得更好，在语义任务上也差不多。最后， Skip-gram 架构在语法任务上比 CBOW 模型稍差一些(但仍然比 NNLM 好)，在测试的语义部分也比所有其他模型好得多。

接下来，我们评估了仅使用一个 CPU 训练的模型，并将结果与可公开获得的词向量进行了比较。 表4 中给出了比较。 CBOW 模型在大约一天的时间里对 Google 新闻数据的子集进行了训练，而 Skip-gram 模型的训练时间大约为三天。

对于进一步的实验报告，我们只使用了一个训练 epoch (再次，我们线性地降低学习率，使它在训练结束时接近零)。**如 表5 所示，使用一个 epoch 在两倍的数据上训练模型可以获得与在三个 epoch 的相同量级数据上迭代相媲美或更好的结果，并提供额外的小加速比。**

![Table5](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table5.png)

**Table 5 : 同一数据上三个 epoch 上训练的模型与一个 epoch 训练的模型的比较。在完整的语义-语法数据集上报告准确性。**

#### 4.4 Large Scale Parallel Training of Models

如前所述，我们已经在称为 DistBelief 的分布式框架中实现了各种模型。 下面我们报告在 Google News 6B 数据集上训练的几种模型的结果，这些模型具有小批量异步梯度下降和称为 Adagrad [7] 的自适应学习率过程。 在训练期间，我们使用了 50 到 100 个模型副本。 由于数据中心计算机是与其他生产任务共享的，因此 CPU 核的数量是一个估计值，并且使用情况可能会有很大的波动。 请注意，由于分布式框架的开销，CBOW 模型和 Skip-gram 模型的 CPU 使用率与单机实现更接近。 结果记录在 表6 中。

![Table6](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table6.png)

**Table 6 : 使用 DistBelief 分布式框架训练模型的效率比较。需要注意的是，使用 1000 维向量训练 NNLM 的时间太长，无法完成。**

#### 4.5 Microsoft Research Sentence Completion Challenge

最近引入了 Microsoft 句子补全挑战作为推进语言模型和其他 NLP 技术的一项任务[32]。 该任务由 1040 个句子组成，其中每个句子中缺少一个单词，目标是在给出五个合理选项的情况下，选择与该句子其余部分最相关的单词。 使用了几个技术在这个数据集上进行了实验，包括 N-gram 模型， LSA-based 模型[32]，log-bilinear 模型[24] 以及目前保持最佳表现的的 RNNLMs 的组合。 在此基准上的准确度为55.4％[19]。

我们已经探索了此任务上 skip-gram 结构的性能。 首先，我们在[32]中提供的 5,000万 个单词上训练 640维 模型。 然后，我们通过使用输入处的未知词来计算测试集中每个句子的分数，并预测句子中所有周围的词。 那么最终的句子分数就是这些单个预测的总和。 使用句子分数，我们选择最可能的句子。

表7 给出了一些 先前结果 以及 新结果 的简短摘要。尽管 skip-gram 模型本身在执行此任务方面的表现并不比 LSA相似性好，但是该模型的得分与 RNNLMs 获得的得分是互补的，并且加权组合可带来一个state of art的结果 58.9％ 的 accuracy (开发集 59.2% ，测试集 58.7%)。

![Table7](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table7.png)

**Table 7 : 微软句子补全挑战赛各模型的比较和组合。**

## 5 Examples of the Learned Relationships

表8 显示了遵循各种关系的单词。我们采用上面描述的方法:通过减去两个词向量来定义关系，然后将结果添加到另一个词上。例如，
$$
巴黎 - 法国 + 意大利 = 罗马
$$
可以看出，精确度相当好，尽管还有很多进一步改进的空间 (请注意，使用假设精确匹配的准确性指标，表 8 中的结果仅得分约 60%)。

我们相信，在更大的数据集上训练维度更高的词向量将表现得更好，并使开发新的创新应用程序成为可能。另一种提高准确性的方法是提供多个关系示例。通过使用10个例子而不是一个例子来形成关系向量(我们将单个向量平均在一起)，我们观察到我们的最佳模型在语义-语法测试上的准确性提高了大约10%。

也可以应用向量运算来解决不同的任务。例如，我们已经观察到，通过计算一系列词的平均向量和找到距离最远的词向量，可以很好地选择 out-of-the-list 的词。在某些人类智力测试中，这是一种常见的问题。显然，使用这些技术有很大的开发空间。

![Table8](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/EfficientEstimationofWordRepresentationsinVectorSpace/Table8.png)

**Table 8: 词对关系的示例，使用 表4 中的最佳词向量(Skip-gram模型，300维， 7.83亿 个词)。**

## 6 Conclusion

本文在一组语义和语法语言任务的基础上，研究了由不同模型得到的词向量表示的质量。我们观察到，与流行的神经网络模型(feedforward 和 recurrent)相比，使用非常简单的模型架构可以训练高质量的词向量。 由于计算复杂度低得多，因此可以从更大的数据集中计算出非常准确的高维词向量。使用DistBelief分布式框架，即使在具有一万亿个单词的语料库上，也可以训练 CBOW 和 Skip-gram 模型，而词典基本上是无限的。 这比以前发布的同类最佳结果最好的数量级还要大几个数量级。

一个有趣的任务是 SemEval-2012 Task 2[11]，在这个任务中，词向量的表现明显优于之前的技术。公开可用的 RNN 向量与其他技术一起使用，使 Spearman's rank correlation 比以前的最佳结果提高了50％以上[31]。基于神经网络的词向量已被应用于许多其他的自然语言处理任务，如情感分析[12]和复述检测[28]。可以预期，这些应用程序可以从本文描述的模型架构中受益。

我们正在进行的工作表明，词向量可以成功地应用于知识库中 facts 的自动扩展，也可以用于验证现有 facts 的正确性。机器翻译实验的结果看起来也很有希望。将来，将我们的技术与潜在关系分析[30]和其他方法进行比较也将很有趣。 我们相信，我们全面的测试集将帮助研究社区改进现有的评估词向量的技术。 我们还期望高质量的词向量将成为未来NLP应用程序的重要组成部分。

## 7 Follow-Up Work

撰写本文的初始版本后，我们发布了使用 CBOW 和 Skip-gram 架构4的单机多线程C ++代码，用于计算单词向量。 训练速度明显高于本文之前的速度，即对于典型的超参数选择而言，训练速度约为每小时数十亿个单词。 我们还发布了 140万 个代表命名实体的向量，并接受了超过 1000亿 个单词的训练。 我们的一些后续工作将在 NIPS 2013 即将发表的论文中发表[21]。

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

------------------

## Q&A

- 深度学习中模型复杂度是如何定义的？

  $E$ 代表 epoch，$Q$ 代表模型参数数量，$T$ 代表训练实体个数。
  $$
  O = E \times Q \times T
  $$

- w2v的优势在于？

  - 大量数据训练的简单模型优于少量数据训练的复杂模型。
  - w2v 最大限度地提高了向量线性运算的准确性，保持了词之间的线性规律。

- NNLM，RNNLMs，Skip-gram，CBOW的：计算复杂度？

  - NNLM: $Q = N × D + N × D × H + H × V$
  - RNNLMs: $Q = H × H + H × V$
  - CBOW HS : $Q = N \times D + D \times log_2(V)$
  - Skip-gram : $Q = C × (D + D × log_2(V ))$

- NNLM，Skip-gram，CBOW的模型结构。

- LSA，PLSA，LDA 与 w2v 的区别？

  - LSA 词-主题模型，使用矩阵分解。
  - LDA，PLSA是一类。

- skip-gram , CBOW 的训练原理？

  - context window
  - CBOW：用周围词预测中心词。
  - skip-gram：用中心词预测周围词。

- skip-gram 在训练中的降采样细节？

  - 取在 $C$ 以内的 $<1,R>$ 选择一个 $R$ 作为训练窗口长度。

- w2v 中向量的线性运算怎么理解，代表了什么？

  - 向量之间的线性关系的一种表达。
  - 法国-巴黎+意大利=罗马，感性认知上来讲，其至少包含了法国作为国家的属性的信息，和首都的信息。

- Embedding 质量的评估方式？

  - w2v中通过语法和语义任务的精准度与其他主流语言模型训练得到的词向量。
  - 定性查看：线性运算。
  - 完形填空。

- 如何通过调整参数提高 vec 质量？

  1. 数据量
  2. 词向量维数
  3. 迭代次数
  
- 如何用word2vec去做句子补全任务？

