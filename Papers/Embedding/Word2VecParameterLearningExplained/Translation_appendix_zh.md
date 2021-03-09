## A Back Propagation Basics

#### A.1 Learning Algorithms for a Single Unit

![Figure5](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig5.png)

**Figure 5: An artificial neuron**

图5 显示了一个人工神经元(单元)。$\{x_1，\cdots，x_K\}$ 是输入值；$\{w_1，\cdots，w_K\}$ 是权重；$y$ 是标量输出；$f$ 是链接函数(也称为激活/决策/传递函数)。

该单元的工作方式如下：
$$
y = f(u), \qquad (62)
$$
其中 $u$ 是标量，它是神经元的输入(或“新输入”)。$u$ 定义为
$$
u = \sum_{i=0}^{K} w_ix_i \qquad(63)
$$
使用向量表示法，我们可以
$$
u = \textbf{w}^T\textbf{x} \qquad {(64)}
$$
请注意，这里我们忽略了以 $u$ 为单位的偏移项。要包括偏置项，只需添加一个为常量 $1$ 的输入维度(例如 $x_0$)。

显然，不同的链接函数会导致神经元行为的不同。 我们在这里讨论链接函数的两个示例选择。

$f(u)$ 的第一个示例选择是单位阶跃函数(也称为Heaviside阶跃函数)：
$$
f(u) = \begin{cases}
1 \quad if \ u > 0 \\
0 \quad otherwise
\end{cases}
\qquad (65)
$$
具有这种链接函数的神经元被称为感知机。感知机的学习算法是感知机算法。其更新方程式定义为：
$$
\textbf{w}^{(new)} = \textbf{w}^{(old)} - \eta \cdot (y - t) \cdot \textbf{x} \qquad (66)
$$
其中 $t$ 是标签( gold standard 跟 ground truth是一个意思)，$η$ 是学习率 $(η>0)$。请注意，感知机是线性分类器，这意味着它的描述能力可能非常有限。如果我们想要拟合更复杂的函数，我们需要使用非线性模型。

第二个选择 $f(u)$ 的示例是 Logistic 函数(一种最常见的 Sigmoid 函数)，定义为
$$
σ(u) = \frac{1}{1+e^{-u}}
\qquad (67)
$$
Logistic函数有两个主要的优良性质：(1)输出 $y$ 总是在 $0$ 和 $1$ 之间；(2)与单位阶跃函数不同，$σ(u)$ 是光滑和可微的，使得更新方程的推导非常容易。

请注意，$\sigma(u)$还具有以下两个属性，这两个属性非常方便，将在后续推导中使用：
$$
\sigma(−u) = 1 − \sigma (u) \qquad (68)
$$

$$
\frac{\partial \sigma(u)}{du} = \sigma(u)\sigma(-u)
\qquad (69)
$$

我们使用随机梯度下降作为该模型的学习算法。为了推导更新方程，我们需要定义误差函数，即训练目标。下面的目标函数似乎很方便：
$$
E = \frac{1}{2}(t − y)^2
\qquad (70)
$$
我们取 $E$ 关于 $w_i$ 的导数
$$
\begin{align*}
\frac{\partial E}{\partial w_i} &= \frac{\partial E}{\partial y} \cdot \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial w_i} \qquad (71)
\\
&= (y-t)\cdot y(1-y) \cdot x_i \qquad(72)
\end{align*}
$$
其中 $\frac{\partial y}{\partial u}=y(1−y)$ ，因为 $y=f(u)=\sigma(u)$，并且(68)和(69)。一旦我们有了导数，我们就可以应用随机梯度下降：
$$
\textbf{w}^{(new)} = \textbf{w}^{(old)} − \eta · (y − t) \cdot y(1 − y) \cdot \textbf{x}
\qquad (73)
$$

#### A.2 Back-propagation with Multi-Layer Network

![Fig6](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig6.png)

**Figure 6: A multi-layer neural network with one hidden layer**

图 6 显示了具有输入层 $\{x_k\}= \{x_1，\cdots，x_K\}$，隐藏层 $\{h_i\}=\{h_1，\cdots，h_N\}$，以及输出层 $\{y_j\}=\{y_1，\cdots，y_M\}$的多层神经网络。为清楚起见，我们分别使用 $k,i,j$ 作为输入层、隐藏层和输出层单元的下标。我们用 $u_i$ 和 $u_{j}^{'}$分别表示隐含层单元和输出层单元的输入。

我们想要推导出 学习输入层和隐藏层之间的权重 $w_{ki}$ 以及隐藏层和输出层之间的权重 $w_{ij}^{'}$ 的更新公式。我们假设所有计算单元(即隐藏层和输出层中的单元)都使用 logistic 函数 $\sigma(u)$ 作为链接函数。

因此，对于隐藏层中的一个  $h_i$ 单元，其输出定义为
$$
h_i = \sigma(u_i) = \sigma (\sum_{k=1}^{K}w_{ki}x_k)
\qquad (74)
$$
同样，对于输出层中的单元 $y_j$，其输出定义为
$$
y_j = \sigma(u_j^{'}) = \sigma(\sum_{i=1}^{N}w_{ij}^{'}h_i) 
\qquad (75)
$$
我们使用的平方和误差函数为
$$
E(\textbf{x}, \textbf{t}, \textbf{W}, \textbf{W}^{'}) = \frac{1}{2} \sum_{j=1}^{M} (y_j − t_j)^2
, \qquad(76)
$$
其中 $\textbf{W}=\{w_{ki}\}$ 是 $K \times N$ 权重矩阵(input -> hidden)，$\textbf{W}^{'}=\{w_{ij}^{'}\}$，$N \times M$ 的权重矩阵(hidden -> output)。$\textbf{t}={t_1，\cdots，t_M}$，一个 $M$  维向量，它是输出层的 gold-standard 标签。

要获得 $w_{ki}$ 和 $w_{ij}^{'}$ 的更新方程，我们只需分别取误差函数 $E$ 关于权重的导数。为了使推导简洁明了，我们开始计算最右侧层(即输出层)的导数，然后向左移动。对于每一层，我们将计算分成三个步骤，分别计算误差关于输出、网络 输入和权重的导数。此过程如下所示。

我们从输出层开始。第一步是计算误差 $E$ 对输出层的导数：
$$
\frac{\partial E}{\partial y_j} = y_j - t_j \qquad (77)
$$
第二步是计算误差相对于 output layer 输入的导数。请注意，当对某个变量进行导数计算时，我们需要保持其他变量都是固定的。另请注意，该值非常重要，因为它将在后续计算中多次重复使用。为简单起见，我们将其表示为 $\textbf{EI}^{'}_j$。
$$
\frac{\partial E}{ \partial u_j^{'}} = \frac{\partial E}{\partial y_j} \cdot \frac{\partial y_j}{\partial u_{j}^{'}} = (y_j - t_j) \cdot y_j(1-y_j) := \textbf{EI}_j^{'} \qquad (78)
$$
第三步是计算误差相对于隐藏层和输出层之间的权重的导数。
$$
\frac{\partial E}{\partial w_{ij}^{'}} = \frac{\partial E}{\partial u_{j}^{'}} \cdot \frac{\partial u_j^{'}}{\partial w_{ij}^{'}} = \textbf{EI}_{j}^{'} \cdot h_i \qquad (79)
$$
到目前为止，我们已经得到了隐藏层和输出层之间的权重更新方程。
$$
\begin{align*}
w_{ij}^{'(new)} &= w_{ij}^{'(old)} - \eta \cdot \frac{\partial E}{\partial w_{ij}^{'}}
\qquad (80)
\\
&= w_{ij}^{'(old)} -\eta \cdot \textbf{EI}_{j}^{'} \cdot h_i
\qquad (81)
\end{align*}
$$
其中 $\eta>0$ 是学习速率。

我们可以重复相同的三个步骤来获得上一层权重的更新方程，这本质上是反向传播的思想。

我们重复第一步，并计算误差相对于隐藏层输出的导数。请注意，隐藏层的输出与输出层中的所有3单位相关。
$$
\frac{\partial E}{\partial h_i} = \sum_{j=1}^{M} \frac{\partial E}{\partial u_j^{'}} \frac{\partial u_j^{'}}{\partial h_i} = \sum_{j=1}^{M} \textbf{EI}_{j}^{'} \cdot w_{ij}^{'}

\qquad (82)
$$
然后，我们重复上述第二步，以计算误差相对于隐藏层的输入的导数。这个值同样非常重要，我们将其表示为 $\textbf{EI}_i$。
$$
\frac{\partial E}{\partial u_i} = \frac{\partial E}{\partial h_i} \cdot \frac{\partial h_i}{\partial u_i}= \sum_{j=1}^{M} \textbf{EI}^{'}_{j} \cdot w_{ij}^{'} \cdot h_i(1-h_i) := \textbf{EI}_i 
\qquad (83)
$$
接下来，我们重复上面的第三步，以计算关于输入层和隐藏层之间的权重的误差的导数。
$$
\frac{\partial E}{\partial w_{ki}} = \frac{\partial E}{\partial u_i} \cdot \frac{\partial u_i}{\partial w_{ki}} = \textbf{EI}_i \cdot x_{k},
\qquad (84)
$$
最后，我们可以得到输入层和隐藏层之间的权重更新方程。
$$
w_{ki}^{(new)} = w_{ki}^{(old)} − \eta \cdot \textbf{EI}_i \cdot x_k
\qquad(85)
$$
从上面的例子可以看出，计算一层导数时的中间结果 ($\textbf{EI}^{'}_{j}$) 可以重用于上一层。假设在输入层之前还有另一层，那么 $\textbf{EI}_i$  也可以被重用来继续高效地计算导数链。比较公式(78)和(83)，我们可以发现，因式  $\sum_{j=1}^{M}\textbf{EI}_{j}^{'}w_{ij}^{'}$ 就像隐藏层单元 $h_i$ 的“误差”。我们可以将此术语解释为来自下一层的 “反向传播” 误差，如果网络具有更多的隐藏层，则此传播可能会进一步追溯。

## B wevi: Word Embedding Visual Inspector

在线提供了一个交互式可视化界面 WEVI (Word Embedding Visual Inspector)来演示本文所描述的模型的工作机制。有关wevi的屏幕截图，请参见图7。

该 demo 允许用户在使用每个训练实例时直观地检查输入向量和输出向量的移动。训练过程还可以以批处理模式运行(例如，连续消耗500个训练实例)，这可以揭示权重矩阵和对应的词向量中出现的模式。主成分分析(PCA)被用来可视化二维散点图中的“高”维向量。该演示同时支持 CBOW 和 skip-gram模型。

训练模型后，用户可以手动激活一个或多个输入层单元，并检查哪些隐藏层单元和输出层单元变为活动状态。用户还可以定制训练数据、隐藏层大小和学习率。提供了几个预设的训练数据集，它们可以生成看起来很有趣的不同结果，例如使用玩具词汇重现著名的单词类比：King-  Queen = Man - Women。

希望通过与此 demo 的互动，可以快速了解该模型的工作机制。

![Fig7](https://raw.githubusercontent.com/anmingyu11/Gor-rok/master/Papers/Embedding/Word2VecParameterLearningExplained/Fig7.png)

**Figure 7: wevi screenshot (http://bit.ly/wevi-online)**

------------------

## Q&A

- 神经元的结构

  > Figure 5

- 向量的行列表示

  - $\textbf{x}^T$ 为行向量 $\textbf{x}$为列向量

- 一个简单神经元(Figure5)的前向传播与反向传播过程。

- 简单的多层神经网络结构

  > Figure 6

- 多层神经网络的前向传播与反向传播过程。

