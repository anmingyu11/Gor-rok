> https://zhuanlan.zhihu.com/p/91652813

# 深入理解GBDT多分类算法

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/fig1.png)

> 目录：
> \1. GBDT多分类算法
> 1.1 Softmax回归的对数损失函数
> 1.2 GBDT多分类原理
> \2. GBDT多分类算法实例
> \3. 手撕GBDT多分类算法
> 3.1 用Python3实现GBDT多分类算法
> 3.2 用sklearn实现GBDT多分类算法
> \4. 总结
> \5. Reference

## **1. GBDT 多分类算法**

## **1.1 Softmax 回归的对数损失函数**

当使用逻辑回归处理多标签的分类问题时，如果一个样本只对应于一个标签，我们可以假设每个样本属于不同标签的概率服从于几何分布，使用多项逻辑回归（Softmax Regression）来进行分类：
$$
P\left(Y=y_{i} \mid x\right)=h_{\theta}(x)\left

[\begin{array}{c}
P(Y=1 \mid x ; \theta) \\
P(Y=2 \mid x ; \theta) \\
\cdot \\
\cdot \\
\cdot \\
P(Y=k \mid x ; \theta)
\end{array}\right]
$$

$$
=\frac{1}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x}}\left[\begin{array}{c}
e^{\theta_{1}^{T} x} \\
e^{\theta_{2}^{T} x} \\
\cdot \\
\cdot \\
\cdot \\
e^{\theta_{k}^{T} x}
\end{array}\right]
$$

其中， $\theta_{1}, \theta_{2}, \ldots, \theta_{k} \in \R^{n}$ 为模型的参数，而 $\frac{1}{\sum_{j=1}^{k} e^{\theta_{j}^{T} x}}$ 可以看作是对概率的归一化。一般来说，多项逻辑回归具有参数冗余的特点，即将 $\theta_{1}, \theta_{2}, \ldots, \theta_{k}$ 同时加减一个向量后预测结果不变，因为 $P(Y=1 \mid x)+P(Y=2 \mid x)+\ldots+P(Y=k \mid x)=1$ ，所以 $P(Y=1 \mid x)=1-P(Y=2 \mid x)-\ldots-P(Y=k \mid x)$ 。

假设从参数向量 $\theta_{j}^{T}$ 中减去向量  $\psi$ ，这时每一个 $\theta_j^T$ 都变成了 $\theta_{j}^{T}-\psi(j=1,2, \ldots, k)$ 。此时假设函数变成了以下公式：
$$
\begin{aligned}
P\left(Y=y_{j} \mid x ; \theta\right) &=\frac{e^{\theta_{j}^{T} x}}{\sum_{i=1}^{k} e^{\theta_{i}^{T} x}} \\
&=\frac{e^{\left(\theta_{j}^{T}-\psi\right) x}}{\sum_{i=1}^{k} e^{\left(\theta_{i}^{T}-\psi\right) x}} \\
&=\frac{e^{\theta_{j}^{T} x} \times e^{-\psi x}}{\sum_{i=1}^{k} e^{\theta_{i}^{T} x} \times e^{-\psi x}} \\
&=\frac{e^{\theta_{j}^{T} x}}{\sum_{i=1}^{k} e^{\theta_{i}^{T} x}}
\end{aligned}
$$
从上式可以看出，从 $\theta_j^T$  中减去 $\psi$ 完全不影响假设函数的预测结果，这表明前面的Softmax 回归模型中存在冗余的参数。特别地，当类别数为 2 时，
$$
h_{\theta}(x)=\frac{1}{e^{\theta_{1}^{T} x}+e^{\theta_{2}^{T} x}}\left[\begin{array}{l}
e^{\theta_{1}^{T} x} \\
e^{\theta_{2}^{T} x}
\end{array}\right]
$$
利用参数冗余的特点，我们将所有的参数减去 $\theta_1$ ，上式变为：
$$
\begin{aligned}
h_{\theta}(x) &=\frac{1}{e^{0 \cdot x}+e^{\left(\theta_{2}^{T}-\theta_{1}^{T}\right) x}}\left[\begin{array}{c}
e^{0 \cdot x} \\
e^{\left(\theta_{2}^{T}-\theta_{1}^{T}\right) x}
\end{array}\right] \\
&=\left[\begin{array}{cc}
\frac{1}{1+e^{\theta^{T} x}} \\
1-\frac{1}{1+e^{\theta} x}
\end{array}\right]
\end{aligned}
$$
其中 $\theta = \theta_2 - \theta_1$ 。而整理后的式子与逻辑回归一致。因此，多项逻辑回归实际上是二分类逻辑回归在多标签分类下的一种拓展。

当存在样本可能属于多个标签的情况时，我们可以训练 $k$ 个二分类的逻辑回归分类器。第 $i$ 个分类器用以区分每个样本是否可以归为第 $i$ 类，训练该分类器时，需要把标签重新整理为“第 $i$ 类标签”与“非第 $i$ 类标签”两类。通过这样的办法，我们就解决了每个样本可能拥有多个标签的情况。

在二分类的逻辑回归中，对输入样本 $x$ 分类结果为类别 $1$ 和 $0$ 的概率可以写成下列形式：
$$
P(Y=y \mid x ; \theta)=\left(h_{\theta}(x)\right)^{y}\left(1-h_{\theta}(x)\right)^{1-y}
$$
其中，$h_{\theta}(x)=\frac{1}{1+e^{-\theta^{T} x}}$ 是模型预测的概率值， $y$  是样本对应的类标签。

将问题泛化为更一般的多分类情况：
$$
P\left(Y=y_{i} \mid x ; \theta\right)=\prod_{i=1}^{K} P\left(y_{i} \mid x\right)^{y_{i}}=\prod_{i=1}^{K} h_{\theta}(x)^{y_{i}}
$$
由于连乘可能导致最终结果接近 0 的问题，一般对似然函数取对数的负数，变成最小化对数似然函数。
$$
-\log P\left(Y=y_{i} \mid x ; \theta\right)=-\log \prod_{i=1}^{K} P\left(y_{i} \mid x\right)^{y_{i}}=-\sum_{i=1}^{K} y_{i} \log \left(h_{\theta}(x)\right)
$$

> **补充：交叉熵**
>
> 假设 $p$ 和 $q$ 是关于样本集的两个分布，其中 $p$ 是样本集的真实分布， $q$ 是样本集的估计分布，那么按照真实分布 $p$ 来衡量识别一个样本所需要编码长度的期望（即，平均编码长度）：
> $$
> H(p)=\sum_{i}^{n} p_{i} \log \frac{1}{p_{i}}=\sum_{i}^{n}-p_{i} \log p_{i}
> $$
> 如果用估计分布 $q$ 来表示真实分布 $p$ 的平均编码长度，应为：
> $$
> H(p, q)=\sum_{i=1}^{n} p_{i} \log \frac{1}{q_{i}}=\sum_{i=1}^{n}-p_{i} \log q_{i}
> $$
> 这是因为用 $q$ 来编码的样本来自于真实分布 $p$ ，所以期望值 $H(p,q)$ 中的概率是 $p_i$ 。而 $H(p,q)$ 就是交叉熵。
>
> 可以看出，在多分类问题中，通过最大似然估计得到的对数似然损失函数与通过交叉熵得到的交叉熵损失函数在形式上相同。

## **1.2 GBDT 多分类原理**

将 GBDT 应用于二分类问题需要考虑逻辑回归模型，同理，对于 GBDT 多分类问题则需要考虑以下 Softmax 模型：
$$
\begin{aligned}
P(y=1 \mid x) &=\frac{e^{F_{1}(x)}}{\sum_{i=1}^{k} e^{F_{i}(x)}} \\
P(y=2 \mid x) &=\frac{e^{F_{2}(x)}}{\sum_{i=1}^{k} e^{F_{i}(x)}} \\
\cdots & \cdots \\
P(y=k \mid x) &=\frac{e^{F_{k}(x)}}{\sum_{i=1}^{k} e^{F_{i}(x)}}
\end{aligned}
$$
其中 $F_{1}, \ldots ,F_{k}$ 是 $k$ 个不同的 CART 回归树集成。每一轮的训练实际上是训练了   $k$ 棵树去拟合 softmax 的每一个分支模型的负梯度。softmax 模型的单样本损失函数为：
$$
\text { loss }=-\sum_{i=1}^{k} y_{i} \log P\left(y_{i} \mid x\right)=-\sum_{i=1}^{k} y_{i} \log \frac{e^{F_{i}(x)}}{\sum_{j=1}^{k} e^{F_{j}(x)}}
$$
这里的 $y_i(i=1...k)$ 是样本 label 在 $k$ 个类别上作 one-hot 编码之后的取值，只有一维为 1 ，其余都是 0 。由以上表达式不难推导：
$$
-\frac{\partial l o s s}{\partial F_{i}}=y_{i}-\frac{e^{F_{i}(x)}}{\sum_{j=1}^{k} e^{F_{j}(x)}}=y_{i}-p\left(y_{i} \mid x\right)
$$
可见，这 $k$ 棵树同样是拟合了样本的真实标签与预测概率之差，与 GBDT 二分类的过程非常类似。下图是 Friedman 在论文中对 GBDT 多分类给出的伪代码：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig2.png)

根据上面的伪代码具体到多分类这个任务上面来，我们假设总体样本共有 $K$ 类。来了一个样本 $x$ ，我们需要使用 GBDT 来判断 $x$ 属于样本的哪一类。

**第一步我们在训练的时候，是针对样本 $x$ 每个可能的类都训练一个分类回归树。** 举例说明，目前样本有三类，也就是 $K=3$ ，样本 $x$ 属于第二类。那么针对该样本的分类标签，其实可以用一个三维向量 $[0,1,0]$ 来表示。$0$  表示样本不属于该类，   $1$ 表示样本属于该类。由于样本已经属于第二类了，所以第二类对应的向量维度为 $1$ ，其它位置为 $0$ 。

**针对样本有三类的情况，我们实质上在每轮训练的时候是同时训练三颗树。**第一颗树针对样本 $x$ 的第一类，输入为 $(x,0)$ 。第二颗树输入针对样本 $x$ 的第二类，输入为 $(x,1)$ 。第三颗树针对样本 $x$ 的第三类，输入为 $(x,0)$ 。这里每颗树的训练过程其实就 CART 树的生成过程。在此我们参照 CART 生成树的步骤即可解出三颗树，以及三颗树对 $x$ 类别的预测值 $F_1(x),F_2(x),F_3(x)$ , 那么在此类训练中，我们仿照多分类的逻辑回归 ，使用 Softmax 来产生概率，则属于类别 $1$ 的概率为：
$$
p_{1}(x)=\frac{\exp \left(F_{1}(x)\right)}{\sum_{k=1}^{3} \exp \left(F_{k}(x)\right)}
$$
可以针对类别 $1$ 求出残差 $\tilde{y}_{1}=0-p_{1}(x)$ ；类别 $2$ 求出残差 $\tilde{y}_{2}=0-p_{2}(x)$ ；类别   $3$ 求出残差 $\tilde{y}_{3}=0-p_{3}(x)$ 。

然后开始第二轮训练，针对第一类输入为 $\left(x, \tilde{y}_{1}\right)$ , 针对第二类输入为 $\left(x, \tilde{y}_{2}\right)$ ，针对第三类输入为 $\left(x, \tilde{y}_{3}\right)$ 。继续训练出三颗树。一直迭代 M 轮。每轮构建 3 颗树。
$$
\begin{array}{l}
F_{1 M}(x)=\sum_{m=1}^{M} c_{1 m} I\left(x \epsilon R_{1 m}\right) \\
F_{2 M}(x)=\sum_{m=1}^{M} c_{2 m} I\left(x \epsilon R_{2 m}\right) \\
F_{3 M}(x)=\sum_{m=1}^{M} c_{3 m} I\left(x \epsilon R_{3 m}\right)
\end{array}
$$
当训练完以后，新来一个样本 $x_1$  ，我们要预测该样本类别的时候，便可以有这三个式子产生三个值 $F_{1M},F_{2M},F_{3M}$ 。样本属于某个类别的概率为：
$$
p_{i}(x)=\frac{\exp \left(F_{i M}(x)\right)}{\sum_{k=1}^{3} \exp \left(F_{k M}(x)\right)}
$$

## **2. GBDT 多分类算法实例**

**（1）数据集**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig3.png)

**（2）模型训练阶段**

首先，由于我们需要转化 3 个二分类的问题，所以需要先做一步 one-hot ：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig4.png)

> 参数设置：
> 学习率：learning_rate = 1
> 树的深度：max_depth = 2
> 迭代次数：n_trees = 5

首先对所有的样本，进行初始化 $F_{k 0}\left(x_{i}\right)=\frac{\operatorname{count}(k)}{\operatorname{count}(n)}$ ，就是各类别在总样本集中的占比，结果如下表。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig5.png)

**注意：**在 Friedman 论文里全部初始化为 0 ，但在 sklearn 里是初始化先验概率（就是各类别的占比），这里我们用 sklearn 中的方法进行初始化。

**1）对第一个类别 $(y_i = 0)$ 拟合第一颗树 $(m=1)$ 。**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig6.png)

**首先，**利用公式 
$$
p_{k, m}(x)=\frac{e^{F_{k, m}(x)}}{\sum_{l=1}^{K} e^{F_{l, m}(x)}}
$$
计算概率。

**其次，**计算负梯度值，以 $x_1$ 为例 $(k=0,i=1)$ ：
$$
\begin{aligned}
\tilde{y}_{i k} &=y_{i, k}-p_{k, m-1} \\
\tilde{y}_{10} &=y_{1,0}-p_{0,0}=1-0.3412=0.6588
\end{aligned}
$$
同样地，计算其它样本可以有下表：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig7.png)

**接着，** 寻找回归树的最佳划分节点。在 GBDT 的建树中，可以采用如 MSE、MAE 等作为分裂准则来确定分裂点。本文采用的分裂准则是 MSE，具体计算过程如下。遍历所有特征的取值，将每个特征值依次作为分裂点，然后计算左子结点与右子结点上的 MSE ，寻找两者加和最小的一个。

比如，选择 $x_8=1$  作为分裂点时 $(x<1)$  。

左子结点上的集合的 MSE 为：
$$
M S E_{l e f t}=(-0.3412-(-0.3412))^{2}=0
$$
右子节点上的集合的 MSE 为：
$$
\begin{aligned}
M S E_{\text {right }} &=(0.6588-0.04342)^{2}+\ldots+(-0.3412-0.04342)^{2} \\
&=3.2142
\end{aligned}
$$
比如选择 $x_{9}=2$ 作为分裂点时 $(x<2)$ 。
$$
M S E_{l e f t}=0, M S E_{r i g h t}=3.07692, M S E=3.07692
$$
对所有特征计算完后可以发现，当选择 $x_{6}=31$ 做为分裂点时，可以得到最小的MSE， $M S E=1.42857$ 。

下图展示以 $31$ 为分裂点的 $\tilde{y}_{i0}$  拟合一颗回归树的示意图：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig8.png)

**然后，**我们的树满足了设置，还需要做一件事情，给这棵树的每个叶子节点分别赋一个参数 $\gamma_{jkm}$ （也就是我们文章提到的 $c$ ），来拟合残差。
$$
\begin{array}{l}
\gamma_{101}=1.1066 \\
\gamma_{201}=-1.0119
\end{array}
$$
**最后，**更新 $F_{k m}\left(x_{i}\right)$ 可得下表：
$$
F_{k m}\left(x_{i}\right)=F_{k, m-1}\left(x_{i}\right)+\eta * \sum_{x_{i} \in R_{j k m}} \gamma_{j k m} * I\left(x_{i} \in R_{j k m}\right)
$$
![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig9.png)

至此第一个类别（类别 $0$ ）的第一颗树拟合完毕，下面开始拟合第二个类别（类别$1$ ）的第一颗树。

**2）对第二个类别 $(y_i=1)$ 拟合第一颗树 $(m-1)$ 。**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig10.png)

**首先，**利用
$$
p_{k, m}(x)=\frac{e^{F_{k, m}(x)}}{\sum_{l=1}^{K} e^{F_{l, m}(x)}}
$$
计算概率。

**其次，**计算负梯度值，以 $x_1$ 为例 $(k=1,i=1)$ ：
$$
\begin{aligned}
\tilde{y}_{i k} &=y_{i, k}-p_{k, m-1} \\
\tilde{y}_{11} &=y_{1,1}-p_{1,0}=0-0.3412=-0.3412
\end{aligned}
$$
同样地，计算其它样本可以有下表：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig11.png)

**然后，**以 $x_0 = 6$ 为分裂点的 $\tilde{y}_{i 1}$ 拟合一颗回归树，可计算得到叶子节点：
$$
\gamma_{111}=1.9540, \quad \gamma_{211}=-0.2704
$$
**最后，**更新 $F_{k m}\left(x_{i}\right)$ 可得下表：
$$
F_{k m}\left(x_{i}\right)=F_{k, m-1}\left(x_{i}\right)+\eta * \sum_{x_{i} \in R_{j k m}} \gamma_{j k m} * I\left(x_{i} \in R_{j k m}\right)
$$
![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/06_GBDT_multi/Fig12.png)

至此第二个类别（类别1）的第一颗树拟合完毕。然后再拟合第三个类别（类别2）的第一颗树，过程也是重复上述步骤，所以这里就不再重复了。在拟合完所有类别的第一颗树后就开始拟合第二颗树。反复进行，直到训练了 M 轮。

## **3. 手撕GBDT多分类算法**

略

## **4. 总结**

在本文中，我们首先从 Softmax 回归引出 GBDT 的多分类算法原理；其次用实例来讲解 GBDT 的多分类算法；然后不仅用 Python3 实现 GBDT 多分类算法，还用sklearn 实现 GBDT 多分类算法；最后简单的对本文做了一个总结。至此，GBDT 用于解决回归任务、二分类任务和多分类任务就完整的深入理解了一遍。

-------------

## QA

1.  推导Softmax回归的对数损失函数

   > 见 1.1
   
2.  GBDT 多分类原理

    > 见 1.2

3.  GBDT多分类算法实例

    > 见 2