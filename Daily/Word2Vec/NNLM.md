# A NEURAL PROBABILISTIC LANGUAGE MODEL

为什么之前的模型不引入NN到LM ：

 已有的标准统计方法用于解决语言模型训练问题已经很成熟，使用已有的标准统计方法顺理成章合情合理无容置疑；

 如果使用NN到LM训练的问题中，神经网络的size会非常巨大，训练到神经网络收敛将会在时间、空间上有巨大的开销。

为什么NNLM ：

 2000年徐伟：通过将NN引入LM进行实验，（意外地？）发现NNLM和已有的标准统计方法比起来表现更优。

 2001年Bengio：learn a distributed representation for words 学习分布式的表征（embedding）to fight the curse of dimensionality 为了克服维度灾难；能够更好地利用更长的上下文。

------------------------------

# 如何实现NNLM ：

神经概率语言模型可以分为四层，即输入层，embedding层，隐藏层，输出层。也可以分为三层，即把input层和embedding层合在一起当做输入层，然后隐藏层、输出层，其实在实际操作中实际的输入就是embedding层（词向量）

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/NNLM_2_1.png)

----------------

## NNLM学习笔记

为了更好的解决问题, Bengio 等人于2003年提出了第一篇运用神经网络搭建语言模型的文章: **A Neural Probabilistic Language Model,** 也被称作**NNLM** (Neural Network Language Model). 这也是第一次提出了词向量的概念, 即**将文本用稠密, 低维, 连续的向量**表达.

NNLM的主要任务是利用前 n - 1 个词汇，预测第 n 个词汇。

NNLM整体的框架非常的简单, 所以直接上框架图:

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/NNLM_2.png)

可以看到， NNLM主要由三层网络构成：输入层，隐含层，输出层。

输入层一共有n - 1个词汇的输入，每个词汇将由one-hot vector的形式编码。对于每个one-hot vector ( $1 \times |V|$)， 它们将会与Embedding size 为 m 的矩阵 C ( $|V| \times m$ )相乘， 得到一个distribution vector ($1 \times m$).

其中，$|V|$ 为词表的大小（即语料库中出现过的所有唯一词汇数量），$m$为embedding size， 通常比$|V|$小很多，这样也就达到了降维的目的。$C$ 这个参数矩阵其实相当于一本字典，每一行都储存了相应位置词汇的词向量，每当有词汇输入的时候，根据词汇的 one-hot-vector， $C$提取出相应行的向量，即为该词汇的词向量。$C$ 由神经网络的 back propagation 训练，不断优化，从而得到更好的词向量，即更优秀的表达能力。

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/NNLM_3.png)

### 隐含层

得到所有词的词向量后（即一个$(n-1) \times m$ 的矩阵），为了利用所有词汇信息，将它们 concatenate 到一起，得到一个$(n-1)m \times 1$ 的向量 $x$

隐含层为一个简单的$tanh$激活层，其公式为：
$$
hidden = tanh(Hx + d)
$$
其维度信息为:
$$
H : h \times (n-1)m
$$

$$
d : h \times 1
$$

$$
x : (n-1)m \times 1
$$

### 输出层

输出层利用了隐含层的输出以及原始合并词向量，最后套上 softmax , 其公式为：
$$
y = b + Wx + Utanh(Hx + d)
$$

$$
p(w_t | w_{t-1},...,w_{t-n+1}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}
$$

其维度信息为：
$$
U : |V| \times h
$$

$$
W : |V| \times (n-1)m
$$

$$
b : |V| \times 1
$$

如果不想利用原始合并词向量的信息的话，可以将$ W $设为零矩阵。

其损失函数为：
$$
L = \frac{1}{T} \sum_t log f(w_t,w_{t-1},...,w_{t-n+1};\theta)+ R(\theta)
$$
论文原文比较简单，建议阅读，当然，文中所提的训练技巧放在当下其实没有太多的意义，没必要去深究，重要的是感受词向量的提出。

```
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

dtype = torch.FloatTensor

sentences = ['I like milk', 'He love apple', 'She hate banana']
vocab_list = set(sorted([i for x in sentences for i in x.split(' ')]))
word2index = {w:i for i, w in enumerate(vocab_list)}
index2word = {i:w for i, w in enumerate(vocab_list)}

V = len(vocab_list) # vocab size
n = len(sentences[0].split(' ')) - 1 # window size
h = 3 # hidden size
m = 3 # feature size

class NNLM(nn.Module):
    
    def __init__(self, V, n, h, m):
        super().__init__()
        self.C = nn.Embedding(V, m)
        self.H = nn.Parameter(torch.zeros(m*n, h))
        self.d = nn.Parameter(torch.zeros(h))
        self.U = nn.Parameter(torch.zeros(h, V))
        self.W = nn.Parameter(torch.zeros(n*m, V))
        self.b = nn.Parameter(torch.zeros(V))
        
    def forward(self, input):
        x = self.C(input)
        x = x.view(-1, n*m)
        hidden = torch.tanh(torch.mm(x, self.H) + self.d)
        y = self.b + torch.mm(x, self.W) + torch.mm(hidden, self.U)
        output = torch.softmax(y, 1)
        return output

input_batch = []
output_batch = []

for s in sentences:
    words = s.split(' ')
    input_batch.append([word2index[i] for i in words[:-1]])
    output_batch.append(word2index[words[-1]])

input_batch = torch.LongTensor(input_batch)
output_batch = torch.LongTensor(output_batch)

model = NNLM(V, n, h, m)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

loss_list = []
for epoch in range(1000):
    output = model(input_batch)
    loss = criterion(output, output_batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data.numpy())
    
    if epoch % 200 == 0:
        print('Epoch{}: {}'.format(epoch, loss))

for index, input in enumerate(input_batch):
    predict = model(input)
    input_word = [index2word[int(index)] for index in input]
    print('-'*40)
    print('Predict:',' '.join(input_word),'->', index2word[int(torch.argmax(predict, dim=1))], '\nTruth:', sentences[index])
```



----------------------

## nn.embedding

This module is often used to store word embeddings and retrieve them using indices.
The input to the module is a list of indices, and the output is the corresponding
word embeddings.

对于这个，我的理解是这样的`torch.nn.Embedding` 是一个矩阵类，当我传入参数之后，我可以得到一个矩阵对象，比如上面代码中的
`embeds = torch.nn.Embedding(2,5)` 通过这个代码，我就获得了一个两行三列的矩阵对象`embeds`。这个时候，矩阵对象embeds的输入就是一个索引列表（当然这个列表
应该是longtensor格式，得到的结果就是对应索引的词向量）

我们这里有一点需要格外注意，在上面的结果中，有个这个东西 `requires_grad = True`

我在开始接触pytorch的时候，对embedding的一个疑惑就是它是如何定义自动更新的。因为现在我们得到的这个词向量是随机初始化的结果，
在后续神经网络反向传递过程中，这个参数是需要更新的。

这里我想要点出一点来，就是词向量在这里是使用标准正态分布进行的初始化。我们可以通过查看源代码来进行验证。

在源代码中

```python
if _weight is None:
  self.weight = Parameter(torch.Tensor(num_embeddings, embedding_dim)) ##定义一个Parameter对象
  self.reset_parameters() #随后对这个对象进行初始化


def reset_parameters(self): #标准正态进行初始化
  init.normal_(self.weight)
  if self.padding_idx is not None:
    with torch.no_grad():
      self.weight[self.padding_idx].fill_(0)
```

-----------------



统计语言模型建模的目标是学习语言中单词序列的联合概率函数。由于维数上的灾难，这本质上是困难的：基于n-gram的传统但非常成功的方法是通过连接在训练集中看到的非常短的重叠序列来获得泛化。

我们建议通过学习单词的分布式表示来对抗维数的灾难。该模型同时学习：

1. 每个单词的分布式表示
2. 用这些表示的单词序列的概率函数

获得泛化是因为如果以前从未见过的单词序列与形成已经看过的句子的单词（在具有附近表示的意义上）的单词构成，则该单词序列获得很高的概率。

在合理的时间内训练如此大的模型（具有数百万个参数）本身就是一个重大的挑战。
$$
\theta\leftarrow\theta\frac{\partial log\hat{P}(w_t|w_{t-1},...,w_{t-n 1})}{\partial\theta}\tag{3.5}
$$


本论文报告了使用神经网络进行概率函数的实验，在两个文本语料库上表明，所提出的方法显著改进了最先进n元语法模型，并且所提出的方法允许利用更长的上下文。

```
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# 准备词表与相关字典
vocab = set(sentence)
print(vocab)
word2index = {w:i for i, w in enumerate(vocab)}
print(word2index)
index2word = {i:w for i, w in enumerate(vocab)}
print(index2word)

# 准备N-gram训练数据 each tuple is ([word_i-2, word_i-1], target word)
trigrams = [([sentence[i], sentence[i+1]], sentence[i+2]) for i in range(len(sentence)-2)]
print(trigrams[0])


# 模型所需参数
CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

# 创建模型
class NGramLanguageModler(nn.Module):

    def __init__(self, vocab_size, context_size, embedding_dim, hidden_dim):
        super(NGramLanguageModler, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(context_size * embedding_dim, vocab_size)
        self.linear3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embedding(inputs).view(1, -1)
        out = torch.tanh(self.linear1(embeds))
        out = self.linear3(out) + self.linear2(embeds)
        return out

losses = []
loss_function = nn.CrossEntropyLoss()
model = NGramLanguageModler(len(vocab), CONTEXT_SIZE, EMBEDDING_DIM, 128)
optimizer = optim.SGD(model.parameters(), lr = 0.001)

for epoch in range(10):
    total_loss = 0

    for context, target in trigrams:
        # Step 1. Prepare the inputs to be passed to the model
        context_idx = torch.tensor([[word2index[w]] for w in context], dtype=torch.long)

        # Step 2. Before passing in a new instance, you need to zero out the gradients from the old instance
        model.zero_grad()

        # Step 3. Run forward pass
        out = model(context_idx)

        # Step 4. Compute your loss function.
        loss = loss_function(out, torch.tensor([word2index[target]], dtype=torch.long))

        # Step 5. Do the backword pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    
    losses.append(total_loss)

print(losses) # The loss decreased every iteration over the training data!

# 结果
{'and', 'the', 'answer', 'gazed', 'besiege', 'To', "'This", 'mine', 'old', 'Thy', 'own', 'blood', 'now,', 'thy', 'say,', "youth's", 'worth', 'thriftless', 'of', 'Will', 'a', 'use,', 'thine', 'where', 'count,', 'Shall', 'Where', 'sum', 'much', "deserv'd", 'succession', 'new', 'held:', 'to', 'And', 'praise.', 'When', 'livery', 'all-eating', "beauty's", 'within', 'be', 'treasure', 'weed', 'How', 'deep', 'all', 'trenches', 'more', 'eyes,', "feel'st", 'beauty', 'sunken', 'forty', 'winters', 'This', 'shall', 'my', 'thou', 'proud', 'Proving', 'when', 'warm', 'dig', 'shame,', 'lusty', 'in', 'small', 'field,', 'an', 'it', 'couldst', 'make', 'thine!', "excuse,'", 'being', 'Then', 'art', 'brow,', 'see', 'cold.', 'fair', 'were', 'his', 'so', 'lies,', 'made', 'days;', 'child', 'If', 'on', 'praise', 'by', 'asked,', 'old,', "totter'd", 'Were'}
{'and': 0, 'the': 1, 'answer': 2, 'gazed': 3, 'besiege': 4, 'To': 5, "'This": 6, 'mine': 7, 'old': 8, 'Thy': 9, 'own': 10, 'blood': 11, 'now,': 12, 'thy': 13, 'say,': 14, "youth's": 15, 'worth': 16, 'thriftless': 17, 'of': 18, 'Will': 19, 'a': 20, 'use,': 21, 'thine': 22, 'where': 23, 'count,': 24, 'Shall': 25, 'Where': 26, 'sum': 27, 'much': 28, "deserv'd": 29, 'succession': 30, 'new': 31, 'held:': 32, 'to': 33, 'And': 34, 'praise.': 35, 'When': 36, 'livery': 37, 'all-eating': 38, "beauty's": 39, 'within': 40, 'be': 41, 'treasure': 42, 'weed': 43, 'How': 44, 'deep': 45, 'all': 46, 'trenches': 47, 'more': 48, 'eyes,': 49, "feel'st": 50, 'beauty': 51, 'sunken': 52, 'forty': 53, 'winters': 54, 'This': 55, 'shall': 56, 'my': 57, 'thou': 58, 'proud': 59, 'Proving': 60, 'when': 61, 'warm': 62, 'dig': 63, 'shame,': 64, 'lusty': 65, 'in': 66, 'small': 67, 'field,': 68, 'an': 69, 'it': 70, 'couldst': 71, 'make': 72, 'thine!': 73, "excuse,'": 74, 'being': 75, 'Then': 76, 'art': 77, 'brow,': 78, 'see': 79, 'cold.': 80, 'fair': 81, 'were': 82, 'his': 83, 'so': 84, 'lies,': 85, 'made': 86, 'days;': 87, 'child': 88, 'If': 89, 'on': 90, 'praise': 91, 'by': 92, 'asked,': 93, 'old,': 94, "totter'd": 95, 'Were': 96}
{0: 'and', 1: 'the', 2: 'answer', 3: 'gazed', 4: 'besiege', 5: 'To', 6: "'This", 7: 'mine', 8: 'old', 9: 'Thy', 10: 'own', 11: 'blood', 12: 'now,', 13: 'thy', 14: 'say,', 15: "youth's", 16: 'worth', 17: 'thriftless', 18: 'of', 19: 'Will', 20: 'a', 21: 'use,', 22: 'thine', 23: 'where', 24: 'count,', 25: 'Shall', 26: 'Where', 27: 'sum', 28: 'much', 29: "deserv'd", 30: 'succession', 31: 'new', 32: 'held:', 33: 'to', 34: 'And', 35: 'praise.', 36: 'When', 37: 'livery', 38: 'all-eating', 39: "beauty's", 40: 'within', 41: 'be', 42: 'treasure', 43: 'weed', 44: 'How', 45: 'deep', 46: 'all', 47: 'trenches', 48: 'more', 49: 'eyes,', 50: "feel'st", 51: 'beauty', 52: 'sunken', 53: 'forty', 54: 'winters', 55: 'This', 56: 'shall', 57: 'my', 58: 'thou', 59: 'proud', 60: 'Proving', 61: 'when', 62: 'warm', 63: 'dig', 64: 'shame,', 65: 'lusty', 66: 'in', 67: 'small', 68: 'field,', 69: 'an', 70: 'it', 71: 'couldst', 72: 'make', 73: 'thine!', 74: "excuse,'", 75: 'being', 76: 'Then', 77: 'art', 78: 'brow,', 79: 'see', 80: 'cold.', 81: 'fair', 82: 'were', 83: 'his', 84: 'so', 85: 'lies,', 86: 'made', 87: 'days;', 88: 'child', 89: 'If', 90: 'on', 91: 'praise', 92: 'by', 93: 'asked,', 94: 'old,', 95: "totter'd", 96: 'Were'}
(['When', 'forty'], 'winters')
[542.6012270450592, 536.4575519561768, 530.3622291088104, 524.314457654953, 518.3134853839874, 512.3586511611938, 506.44934606552124, 500.58502769470215, 494.7652368545532, 488.98955368995667]

```



作者：魏鹏飞
链接：https://www.jianshu.com/p/38c19944fb7e
来源：简书
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

--------------------------

# 神经网络语言模型（NNLM）

## **1. 模型原理**

用神经网络来训练语言模型的思想最早由百度 IDL （深度学习研究院）的徐伟提出[1]，其中这方面的一个经典模   是NNLM（Nerual Network Language Model），具体内容可参考 Bengio 2003年发表在 JMLR 上的论文[2]。

模型的训练数据是一组词序列$(w_{1},...,w_{T}) ,w_{t} \in V$。其中 $V$ 是所有单词的集合（即词典），$V_i$表示字典中的第 $i$ 个单词。NNLM的目标是训练如下模型：
$$
f(w_t,w_{t-1},...,w_{t-n+2},w_{t-n+1}) = p(w_t|w_1^{t-1})
$$
其中$w_t$表示词序列中第$t$个单词，$w_1^{t-1}$表示从第$1$个词到第$t$个词组成的子序列。模型需要满足的约束条件是：

- $f(w_t,w_{t-1},...,w_{t-n+2},w_{t-n+1}) > 0$
- $\sum_{i=1}^{V}f(i,w_{t-1},...,w_{t-n+2},w_{t-n+1}) = 1$

下图展示了模型的总体架构：

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/NNLM_2.png)

该模型可分为**特征映射**和**计算条件概率分布**两部分：

1. 特征映射：通过映射矩阵 $C \in R^{|V| \times m}$将输入的每个词映射为一个特征向量，$C(i)  \in R^m$表示词典中第 $i$ 个词对应的特征向量，其中$m$ 表示特征向量的维度。该过程将通过特征映射得到的$C(w_{t-n+1},...,C(w_{t-1}))$合并成一个$(n-1)m$维的向量:$(C(w_t-n+1),...,C(w_{t-1}))$
2. 计算条件概率分布：通过一个函数$g$（是$g$前馈或递归神经网络）将输入的词向量序列$(C(w_{t-n+1}),...,C(w_{t-1}))$转化为一个概率分布 $y \in R ^{|V|}$ , $y$中第$i$位表示词序列中第$ t $ 个词是的$V_i$概率，即:
3. $f(i,w_{t-1},...,w_{t-n+2},w_{t-n+1}) = g(i,C(w_{t-n+1},...,C(w_{t-1})))$

下面重点介绍神经网络的结构，网络输出层采用的是 softmax 函数，如下式所示：

- $p(w_t|w_{t-1},...,w_{t-n+2},w_{t-n+1}) = \frac{e^{y_{w_t}}}{\sum_{i}^{e^{y_i}}}$

其中$y = b + Wx + Utanh(d+Hx)$，模型的参数$ \theta = (b，d，W，U，H，C)$。$ x=(C(w_{t-n+1}),...,C(w_{t-1}))$是神经网络的输入。

$W \in R^{|V|×(n-1)m}$是可选参数，如果输入层与输出层没有直接相连（如图中绿色虚线所示），则可令 $W=0$。 $H \in R^{h×(n-1)m}$是输入层到隐含层的权重矩阵，其中$h$表示隐含层神经元的数目。$U \in R^{|V|×h}$是隐含层到输出层的权重矩阵。$d\in R^{h}$和 $b \in R^{|V|}$分别是隐含层和输出层的偏置参数。

**需要注意的是：**一般的神经网络模型不需要对输入进行训练，而该模型中的输入$x=(C(w_{t-n+1}),...,C(w_{t-1}))$是词向量，也是需要训练的参数。由此可见模型的权重参数与词向量是同时进行训练，

**模型训练完成后同时得到网络的权重参数和词向量。**

## 训练过程

模型的训练目标是最大化以下似然函数：

- $L=\frac{1}{T} \sum_{t}^{ } logf(w_{t},w_{t-1},...,w_{t-n+2}, w_{t-n+1}; \theta) + R(\theta)$其中$ \theta$为模型的所有参数，$R(\theta)$为正则化项

使用梯度下降算法更新参数的过程如下：

-  $\theta \leftarrow \theta +\epsilon \frac{\partial logp(w_{t}|w_{t-1},...,w_{t-n+2}, w_{t-n+1}) }{\partial \theta}$ ,其中 $\epsilon $为步长。

----------------

# 语言模型：从n元模型到NNLM

## 前言

虽然目前来看，基于深度学习的语言模型技术已经完全超越了传统语言模型，但是我个人认为从整个发展的脉络的角度来看能够加深我们对整个领域的理解。这篇文章是语言模型的一个初学者版本，如果想了解最新的语言模型， 需要关注paper前沿。

## 什么是语言模型

**一个语言模型可以简单理解为一个句子 s 在所有句子中出现的概率分布 P(s)。**举个简单的例子：

> 如果一个人所说的话语中每100个句子里大约有一句是Okay，则可以认为p(Okay) ≈ 0.01。而对于句子“An apple ate the chicken”我们可以认为其概率为0，因为几乎没有人会说这样的句子。

我们再举个简单的例子（引用自《数学之美》）：

> 美联储主席本 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccdot) 伯南克昨天告诉媒体7000亿美元的救助资金将借给上百家银行，保险公司和汽车公司。

这句话很容易理解，因为它的句子很通顺，读起来就像是人说的。我们来更改部分词的顺序：

> 本 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccdot) 伯南克美联储主席昨天7000亿美元的救助资金告诉媒体将借给银行，保险公司和汽车公司上百家。

这句话是不是就比较费劲了，读起来我们也能理解，但是感觉就不太像是人说的。我们再改一改，如下：

> 联主美储席本 ![[公式]](https://www.zhihu.com/equation?tex=%5Ccdot) 伯诉南将借天的救克告媒咋助资金70元亿00美给上百败百家银保行，汽车险公司公司和。

ok，这句话你还能看懂吗，我觉得你很难看懂，这句话看起来就不像是人说的。

那么我们就引出了语言模型这个概念：

> 语言模型就是一个句子 s 在所有句子中出现的概率分布，假设第一个句子出现在该语言模型中的概率为 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B-20%7D) ，第二个句子出现的概率为 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B-25%7D) ，第三个句子出现的概率是 ![[公式]](https://www.zhihu.com/equation?tex=10%5E%7B-70%7D) 。那么第一句话是人话的概率要比第三句话的可能性大得多。

**注意一点：**语言模型与句子是否合乎语法是没有关系的，即使一个句子完全合乎语法逻辑，我们仍然可以认为它出现的概率接近为 0。

## 语言模型中的概率是怎么产生的？

我们先假设 $S$ 表示某一个有意义的句子，$S$ 是由一连串特定顺序排列的词$w_1,w_2,...,w_n$  组成，这里$n$是句子的长度，那么我们怎么获取句子 $S$ 在一个语言模型中的概率 $P(S)$ ?

最简单的方法是建立一个无比庞大的语料库，该语料库中包含了人类成百上千年间可能讲过的所有的话，那么我们不就可以算出这句话出现的概率了吗，可惜，傻子都知道这种方法不可能行得通。

这里就体现出数学的精妙之处了，我们要计算的是$P(S)$， 而 $S$ 是一个序列$w_1,w_2,...,w_n$ ，那么根据概率论的相关知识我们可以得出：$P(S) = P(w_1,w_2,...,w_n)$ 再展开我们可以得到：
$$
P(w_1,w_2,...,w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) ... P(w_n|w_1,w_2,...,w_{n-1})
$$
我们观察上式，会发现，$P(w_1)$ 比较好算，$P(w_2|w_1)$ 也还ok，$P(w_3,w_1,w_2)$ 就比较有难度了，随着$n$的增大，计算会越来越难，$P(w_n|w_1,w_2,...,w_{n-1})$ 几乎根本不可能估算出来。怎么办？

## 马尔可夫假设

数学家马尔可夫针对无法计算 $P(w_n|w_1,w_2,...,w_{n-1})$  这种情况，提出了一种偷懒且高效的方法：

> 每当遇到这种情况时，就假设任意一个词$w_i$ 出现的概率只同它前面的词$w_{i-1}$ 有关，这就是很有名的马尔可夫假设。

基于此思想，n-gram model诞生了。

## n-gram model（n元模型）

n元模型的思想就是：

> 出现在第$i$位上的词$w_i$仅与它前面的$(n-1)$个历史词有关。

通常情况下，$n$的取值不能太大，实际中，$n=3$是最常见的情况。$n$过小，产生的概率不够准确，$n$过大，计算量太大。

### 1. 一元模型

当$n=1$时,即出现在第$i$位上的词 $w_i$独立，一元文法被记作unigram,或uni-gram,或monogram。
$$
P(S) = \prod_{i=1}^{l} P(w_i) 
$$
其余比如二元模型， 三元模型就不举例了， 比较简单，大同小异。

## n 元模型的缺陷

1. 无法建模更远的关系，语料的不足使得无法训练更高阶的语言模型。
2. 无法建模出词之间的相似度。
3. 训练语料里面有些 $n$ 元组没有出现过,其对应的条件概率就是 $0$,导致计算一整句话的概率为 $0$。解决这个问题有两种常用方法： 平滑法和回退法。



## 神经网络语言模型（NNLM）

第一篇提出神经网络语言模型的论文是Bengio大神在2003年发表的《A Neural Probabilistic Language Model》。我个人建议这篇文章是必读的， 因为模型最简单， 便于理解， 下面我来分析一下这个模型， 建议参照论文饮用更佳：

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/NNLM_2.png)

观察上图，假设有一组词序列：$w_1,w_2,...,w_t$ ，其中$w_i \in V$  ， $V$是所有单词的集合。我们的输入是一个词序列，而我们的输出是一个概率值，表示根据context预测出下一个词是 $i$ 的概率。用数学来表示，我们最终是要训练一个模型：
$$
f(w_t, w_{t-1}, \cdots, w_{t-n+1}) = P(w_t = i | context) = P(w_t | w_1^{t-1})
$$
其中有：

- $w_t$ 表示这个词序列中的第 $t$ 个单词，  表$w_{t-n+1}$示输入长度为$n$的词序列中的第一个单词
- $w_1^{t-1}$ 表示从第$1$个单词到第$t-1$ 个单词组成的子序列

我们发现， 这个过程与上面提到的其实是一样的， 其实就是求：
$$
p(w_n | w_1,...,w_{n-1})
$$
该模型需要满足两个约束条件：
$$
f(w_t,w_{t-1},\cdots,w_{t-n+2},w_{t-n+1}) >0
$$

$$
\sum_{i=1}^{|V|} f(i, w_{t-1}, \cdots, w_{t-n+2}, w_{t-n+1}) = 1
$$

其中 $|V|$ 表示词表的大小

## 模型的前向传播

该模型结构分为三层，分别是输入层，一层隐层，输出层。下面我分别来介绍。

#### 输入层： 从One-hot到distribution representation

对于输入层，我们先将输入词序列 $w_1, \cdots, w_n$ 映射到词表中，如词$w_i$ 是词表中是第$i$个元素，编码为one-hot embedding；然后我们再将这个元素映射到实向量$C(i)$ 中，其中$C(i) \in R^m$ ，它表示的是**词表中第$i$个词**的**distributed representation**。 $C$实际上就是一个$|V| \times m$  的自由参数矩阵， $|V|$ 表示词表的大小，$m$ 表示每个词的维度。
$$
x(k) \leftarrow C(w_{t-k}) \qquad x(k) 是m维的向量 \\ x = (x(1), x(2), \cdots, x(n-1)) \qquad x 是 (n-1) × m 维的矩阵
$$
其中，词的映射过程用图表示如下：

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Word2Vec/NNLM_3.png)

## 隐层：

$$
o \leftarrow d + Hx; \qquad o是长度为h的向量 \\ a \leftarrow tanh(o) \qquad a是长度为h的向量
$$

## 输出层

在输出层中，同时受到隐层输出 $a$ 和 输入层输出$ x$ 的影响，公式如下：
$$
y = b + Wx + Utanh(d + Hx) \\ p(w_t|w_{t-1}, \cdots, w_{t-n+1}) = \frac{e^{y_{w_t}}}{\sum_i e^{y_i}}
$$
最后，一目了然，$x$ 是输入， $P(w_t| w_{t-1, \cdots, w_{t-n+1}})$是输出，需要更新的参数包括：  $\theta=(b,d,W,U,H,C)$。也就是说这里我们已经求出在$w_{t-n+1}$ 到$w_{t-1}$ 出现的概率， 接下来训练的时候， 我们就能够通过计算优化出$P(s)$ ，来完成整个语言模型的训练。

## 模型的训练

损失函数为：
$$
L = \dfrac{1}{T}\sum_t \log \hat{P}(w_t|w_{t-1},\dots,w_{t-n+1})+R(\theta)
$$
具体的训练过程我就不赘述了，除非复现模型，否则这些训练细节你是没有必要细究的。

## 最后

最后，总结一下， 从$n$元模型到神经网络语言模型， 明显神经网络语言模型更胜一筹。 从深度学习方法在机器翻译领域获得突破后， 深度学习技术在NLP上已经几乎全面超过了传统的方法，可以预见的是，未来几年， 传统的方法或将陆续退出舞台， 或者与深度学习方法结合， 总之，好好学习，多写代码，你我共勉。

