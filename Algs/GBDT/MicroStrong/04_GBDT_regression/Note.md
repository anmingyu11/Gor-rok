# 深入理解GBDT回归算法

> \1. GBDT简介
> \2. GBDT回归算法
> 2.1 GBDT回归算法推导
> 2.2 GBDT回归算法实例
> \3. 手撕GBDT回归算法
> 3.1 用Python3实现GBDT回归算法
> 3.2 用sklearn实现GBDT回归算法
> \4. GBDT回归任务常见的损失函数
> \5. GBDT的正则化
> \6. 关于GBDT若干问题的思考
> \7. 总结
> \8. Reference

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig1.png)

## **1. GBDT 简介**

Boosting、Bagging 和 Stacking 是集成学习(Ensemble Learning)的三种主要方法。Boosting 是一族可将弱学习器提升为强学习器的算法，不同于 Bagging 、 Stacking 方法，Boosting 训练过程为串联方式，弱学习器的训练是有顺序的，每个弱学习器都会在前一个学习器的基础上进行学习，最终综合所有学习器的预测值产生最终的预测结果。

梯度提升（Gradient boosting）算法是一种用于回归、分类和排序任务的机器学习技术，属于Boosting算法族的一部分。之前我们介绍过 [Gradient Boosting算法](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Ods1PHhYyjkRA8bS16OfCg)在迭代的每一步构建一个能够沿着梯度最陡的方向降低损失的学习器来弥补已有模型的不足。经典的 AdaBoost 算法只能处理采用指数损失函数的二分类学习任务，而梯度提升方法通过设置不同的可微损失函数可以处理各类学习任务（多分类、回归、Ranking 等），应用范围大大扩展。梯度提升算法利用损失函数的负梯度作为残差拟合的方式，如果其中的基函数采用决策树的话，就得到了梯度提升决策树 (Gradient Boosting Decision Tree ,  GBDT)。

基于梯度提升算法的学习器叫做 GBM(Gradient Boosting Machine)。理论上， GBM 可以选择各种不同的学习算法作为基学习器。现实中，用得最多的基学习器是决策树。

**决策树有以下优点：**

- 决策树可以认为是 if-then 规则的集合，易于理解，可解释性强，预测速度快。
- 决策树算法相比于其他的算法需要更少的特征工程，比如可以不用做特征标准化。
- 决策树可以很好的处理字段缺失的数据。
- 决策树能够自动组合多个特征，也有特征选择的作用。
- 对异常点鲁棒。
- 可扩展性强，容易并行。

**决策树有以下缺点：**

- 缺乏平滑性（回归预测时输出值只能输出有限的若干种数值）。
- 不适合处理高维稀疏数据。
- 单独使用决策树算法时容易过拟合。

我们可以通过抑制决策树的复杂性，降低单棵决策树的拟合能力，再通过梯度提升的方法集成多个决策树，最终能够很好的解决过拟合的问题。由此可见，梯度提升方法和决策树学习算法可以互相取长补短，是一对完美的搭档。

## **2. GBDT回归算法**

## 2.1 GBDT回归算法推导

当我们采用的基学习器是决策树时，那么梯度提升算法就具体到了梯度提升决策树。GBDT 算法又叫 MART（Multiple Additive Regression），是一种迭代的决策树算法。GBDT 算法可以看成是 $M$ 棵树组成的加法模型，其对应的公式如下：
$$
F(x, w)=\sum_{m=0}^{M} \alpha_{m} h_{m}\left(x, w_{m}\right)=\sum_{m=0}^{M} f_{m}\left(x, w_{m}\right)
$$
其中， $x$ 为输入样本； $w$ 为模型参数； $h$ 为分类回归树； $\alpha$ 为每棵树的权重。GBDT 算法的实现过程如下：

给定训练数据集：$T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{N}, y_{N}\right)\right\}$ 其中， $x_{i} \in \chi \subseteq R^{n}$ ，  为 $\chi$ 输入空间， $y_{i} \in Y \subseteq R$ ，$Y$ 为输出空间，损失函数为 $L(y, f(x))$ ，我们的目标是得到最终的回归树 $F_M$ 。

**1）初始化第一个弱学习器 $F_0(x)$ ：**
$$
F_{0}(x)=\underset{c}{\arg \min } \sum_{i=1}^{N} L\left(y_{i}, c\right)
$$
**2）对于建立 M 棵分类回归树** $m=1,2, \ldots, M$ **：**

**a）**对 $i=1,2, \ldots, N$ ，计算第 $m$ 棵树对应的响应值（损失函数的负梯度，即伪残差）：
$$
r_{m, i}=-\left[\frac{\partial L\left(y_{i}, F\left(x_{i}\right)\right)}{\partial F(x)}\right]_{F(x)=F_{m-1}(x)}
$$
**b）**对于 $i=1,2, \ldots, N$ ，利用 CART 回归树拟合数据 $\left(x_{i}, r_{m, i}\right)$ ，得到第 $m$ 棵回归树，其对应的叶子节点区域为 $R_{m, j}$ ，其中 $j=1,2, \ldots, J_{m}$ ，且 $J_{m}$ 为第 $m$ 棵回归树叶子节点的个数。

**c）**对于 $J_m$ 个叶子节点区域 $j=1,2, \ldots, J_{m}$，计算出最佳拟合值：
$$
c_{m, j}=\underset{c}{\arg \min } \sum_{x_{i} \in R_{m, j}} L\left(y_{i}, F_{m-1}\left(x_{i}\right)+c\right)
$$
**d）**更新强学习器 $F_m(x)$ ：
$$
F_{m}(x)=F_{m-1}(x)+\sum_{j=1}^{J_{m}} c_{m, j} I\left(x \in R_{m, j}\right)
$$
**3）得到强学习器 $F_M(x)$ 的表达式：**
$$
F_{M}(x)=F_{0}(x)+\sum_{m=1}^{M} \sum_{j=1}^{J_{m}} c_{m, j} I\left(x \in R_{m, j}\right)
$$

## 2.2 GBDT回归算法实例

**（1）数据集介绍**

训练集如下表所示，一组数据的特征有年龄和体重，身高为标签值，共有 4 组数据。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig2.png)

测试数据如下表所示，只有一组数据，年龄为 25、体重为 65，我们用在训练集训练好的 GBDT 模型预测该组数据的身高值为多少。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig3.png)

**（2）模型训练阶段**

**参数设置：**

- 学习率：learning_rate = 0.1
- 迭代次数：n_trees = 5
- 树的深度：max_depth = 3

**1）初始化弱学习器：**
$$
F_{0}(x)=\underset{c}{\arg \min } \sum_{i=1}^{N} L\left(y_{i}, c\right)
$$
损失函数为平方损失，因为平方损失函数是一个凸函数，直接求导，导数等于零，得到 $c$ 。
$$
\sum_{i=1}^{N} \frac{\partial L\left(y_{i}, c\right)}{\partial c}=\sum_{i=1}^{N} \frac{\partial\left(\frac{1}{2}\left(y_{i}-c\right)^{2}\right)}{\partial c}=\sum_{i=1}^{N} c-y_{i}
$$
令导数等于 $0$：
$$
\sum_{i=1}^{N} c-y_{i}=0 \Rightarrow c=\frac{\sum_{i=1}^{N} y_{i}}{N}
$$
所以初始化时，$c$  取值为所有训练样本标签值的均值。 $c=(1.1+1.3+1.7+1.8) / 4=1.475$ ，此时得到的初始化学习器为 $F_{0}(x)=c=1.475$ 。

**2）对于建立 $M$ 棵分类回归树** $m=1,2, \ldots, M$ **：**

由于我们设置了迭代次数：n_trees = 5，且设置了 M = 5。

**首先计算负梯度**，根据上文损失函数为平方损失时，负梯度就是残差，也就是 $y$ 与上一轮得到的学习器  $F_{m-1}$ 的差值：

现将残差的计算结果列表如下：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig4.png)

此时将残差作为样本的真实值来训练弱学习器 $F_1(x)$ ，即下表数据：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig5.png)

接着，寻找回归树的最佳划分节点，遍历每个特征的每个可能取值。从年龄特征值为 5 开始，到体重特征为 70 结束，分别计算分裂后两组数据的平方损失（Square Error），$SE_l$ 为左节点的平方损失， $SE_r$ 为右节点的平方损失，找到使平方损失和 $SE_{sum} = SE_l + SE_r$ 最小的那个划分节点，即为最佳划分节点。

例如：以年龄 7 为划分节点，将小于 7 的样本划分为到左节点，大于等于 7 的样本划分为右节点。左节点包括 $x_0$，右节点包括样本 $x_1,x_2,x_3$， $SE_l=0$，$SE_r=0.140$ ，$S E_{s u m}=0.140$ ，所有可能的划分情况如下表所示：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig6.png)

以上划分点的总平方损失最小为 **0.025** 有两个划分点：年龄 21 和体重 60，所以随机选一个作为划分点，这里我们选**年龄 21。**现在我们的第一棵树长这个样子：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig7.png)

我们设置的参数中树的深度 max_depth = 3 ，现在树的深度只有 2 ，需要再进行一次划分，这次划分要对左右两个节点分别进行划分：

对于**左节点**，只含有 0 , 1 两个样本，根据下表结果我们选择**年龄 7 为**划分点（也可以选**体重 30 **）。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig8.png)

对于**右节点**，只含有 2 , 3 两个样本，根据下表结果我们选择**年龄 30**为划分点（也可以选**体重 70**）。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig9.png)

现在我们的第一棵回归树长下面这个样子：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig10.png)



此时我们的树深度满足了设置，还需要做一件事情，给这每个叶子节点分别赋一个参数 $c$ ，来拟合残差。
$$
c_{1, j}=\underset{c}{\arg \min } \sum_{x_{i} \in R_{1, j}} L\left(y_{i}, F_{0}\left(x_{i}\right)+c\right)
$$
这里其实和上面初始化弱学习器是一样的，对平方损失函数求导，令导数等于零，化简之后得到每个叶子节点的参数 $c$ ，其实就是标签值的均值。这个地方的标签值不是原始的  $y$，而是本轮要拟合的标残差 $y - f_0(x)$ 。

根据上述划分结果，为了方便表示，规定从左到右为第 1,2,3,4 个叶子结点，其计算值过程如下：
$$
\begin{array}{ll}
\left(x_{0} \in R_{1,1}\right), & c_{1,1}=1.1-1.475=-0.375 \\
\left(x_{1} \in R_{1,2}\right), & c_{1,2}=1.3-1.475=-0.175 \\
\left(x_{2} \in R_{1,3}\right), & c_{1,3}=1.7-1.475=0.225 \\
\left(x_{3} \in R_{1,4}\right), & c_{1,4}=1.8-1.475=0.325
\end{array}
$$
此时的树长这下面这个样子：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig11.png)

此时可更新强学习器，需要用到参数学习率：learning_rate=0.1，用 $lr$ 表示。
$$
F_{1}(x)=F_{0}(x)+l r * \sum_{j=1}^{4} c_{1, j} I\left(x \in R_{1, j}\right)
$$
为什么要用学习率呢？这是 **Shrinkage** 的思想，如果每次都全部加上拟合值 $c$ ，即学习率为 1，很容易一步学到位导致 GBDT 过拟合。

**重复此步骤，直到 $m>5$ 结束，最后生成 5 棵树。**

下面将展示每棵树最终的结构，这些图都是我GitHub上的代码生成的，感兴趣的同学可以去运行一下代码。[https://github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning/GBDT_Regression](https://link.zhihu.com/?target=https%3A//github.com/Microstrong0305/WeChat-zhihu-csdnblog-code/tree/master/Ensemble%20Learning/GBDT_Regression)

**第一棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig12.png)

**第二棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig13.png)

**第三棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig14.png)

**第四棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig15.png)

**第五棵树：**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/04_GBDT_Regression/Fig16.png)

**3）得到最后的强学习器：**
$$
F_{5}(x)=F_{0}(x)+\sum_{m=1}^{5} \sum_{j=1}^{4} c_{m, j} I\left(x \in R_{m, j}\right)
$$
**（3）模型预测阶段**

- $F_0(x)=1.475$
- 在 $F_1(x)$ 中，测试样本的年龄为 25，大于划分节点 21 岁，又小于 30 岁，所以被预测为 **0.2250**。
- 在 $F_2(x)$ 中，测试样本的年龄为 25，大于划分节点 21 岁，又小于 30 岁，所以被预测为 **0.2025**。
- 在 $F_3(x)$ 中，测试样本的年龄为 25，大于划分节点 21 岁，又小于 30 岁，所以被预测为 **0.1823**。
- 在 $F_4(x)$ 中，测试样本的年龄为 25，大于划分节点 21 岁，又小于 30 岁，所以被预测为 **0.1640**。
- 在 $F_5(x)$ 中，测试样本的年龄为 25，大于划分节点21岁，又小于30岁，所以被预测为 **0.1476**。

**最终预测结果为：**
$$
F(x)=1.475+0.1 *(0.225+0.2025+0.1823+0.164+0.1476)=1.56714
$$

## **3. 手撕GBDT回归算法**

略

## **4. GBDT 回归任务常见的损失函数**

对于 GBDT 回归模型，sklearn 中实现了四种损失函数，有均方差 'ls' , 绝对损失 'lad' , Huber 损失 'huber' 和分位数损失 'quantile' 。默认是均方差 'ls' 。一般来说，如果数据的噪音点不多，用默认的均方差 'ls' 比较好。如果是噪音点较多，则推荐用抗噪音的损失函数 'huber' 。而如果我们需要对训练集进行分段预测的时候，则采用 'quantile' 。下面我们具体来了解一下这四种损失函数。

**（1）均方差**，这个是最常见的回归损失函数了，公式如下：
$$
L(y, f(x))=(y-f(x))^{2}
$$
对应的负梯度误差为：
$$
y_{i}-f\left(x_{i}\right)
$$
**（2）绝对损失**，这个损失函数也很常见，公式如下：
$$
L(y, f(x))=|y-f(x)|
$$
对应的负梯度误差为：
$$
\operatorname{sign}\left(y_{i}-f\left(x_{i}\right)\right)
$$
**（3）Huber损失**，它是均方差和绝对损失的折衷产物，对于远离中心的异常点，采用绝对损失，而中心附近的点采用均方差。这个界限一般用分位数点度量。损失函数如下：
$$
L(y, f(x))=\left\{\begin{aligned}
\frac{1}{2}(y-f(x))^{2} &|y-f(x) \leq \delta| \\
\delta\left(|y-f(x)|-\frac{\delta}{2}\right) &|y-f(x)>\delta|
\end{aligned}\right.
$$
对应的负梯度误差为：
$$
r\left(y_{i}, f\left(x_{i}\right)\right)=\left\{\begin{array}{rr}
y_{i}-f\left(x_{i}\right) & \left|y_{i}-f\left(x_{i}\right) \leq \delta\right| \\
\delta \cdot \operatorname{sign}\left(y_{i}-f\left(x_{i}\right)\right) & \left|y_{i}-f\left(x_{i}\right)>\delta\right|
\end{array}\right.
$$
**（4）分位数损失**，它对应的是分位数回归的损失函数，表达式为：
$$
L(y, f(x))=\sum_{y \geq f(x)} \theta|y-f(x)|+\sum_{y<f(x)}(1-\theta)|y-f(x)|
$$
其中， $\theta$ 为分位数，需要我们在回归前指定。对应的负梯度误差为：
$$
r\left(y_{i}, f\left(x_{i}\right)\right)=\left\{\begin{array}{rl}
\theta & y_{i} \geq f\left(x_{i}\right) \\
\theta-1 & y_{i}<f\left(x_{i}\right)
\end{array}\right.
$$
对于 Huber 损失和分位数损失，主要用于健壮回归，也就是减少异常点对损失函数的影响。

## **5. GBDT的正则化**

为了防止过拟合，GBDT主要有五种正则化的方式。

**（1）“Shrinkage”：** 这是一种正则化（regularization）方法，为了防止过拟合，在每次对残差估计进行迭代时，不直接加上当前步所拟合的残差，而是乘以一个系数 $\alpha$ 。系数 $\alpha$ 也被称为学习率（learning rate），因为它可以对梯度提升的步长进行调整，也就是它可以影响我们设置的回归树个数。对于前面的弱学习器的迭代：
$$
F_{m}(x)=F_{m-1}(x)+h_{m}(x)
$$
如果我们加上了正则化项，则有：
$$
F_{m}(x)=F_{m-1}(x)+\alpha h_{m}(x)
$$
$\alpha$ 的取值范围为  $0<\alpha \leq 1$ 。对于同样的训练集学习效果，较小的  意$\alpha$味着我们需要更多的弱学习器的迭代次数。通常我们用学习率和迭代最大次数一起来决定算法的拟合效果。即参数 learning_rate 会强烈影响到参数 n_estimators（即弱学习器个数）。learning_rate 的值越小，就需要越多的弱学习器数来维持一个恒定的训练误差(training error)常量。经验上，推荐小一点的 learning_rate 会对测试误差 (test error) 更好。在实际调参中推荐将 learning_rate 设置为一个小的常数（e.g. learning_rate <= 0.1），并通过 early stopping 机制来选 n_estimators。

**（2）“Subsample”：** 第二种正则化的方式是通过子采样比例（subsample），取值为 (0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为 1，则全部样本都使用，等于没有使用子采样。如果取值小于 1，则只有一部分样本会去做 GBDT 的决策树拟合。选择小于 1 的比例可以减少方差，即防止过拟合，但会增加样本拟合的偏差，因此取值不能太低。推荐在 [0.5, 0.8] 之间。

使用了子采样的 GBDT 有时也称作随机梯度提升树 (Stochastic Gradient Boosting Tree, SGBT)。由于使用了子采样，程序可以通过采样分发到不同的任务去做Boosting 的迭代过程，最后形成新树，从而减少弱学习器难以并行学习的弱点。

**（3）对于弱学习器即 CART 回归树进行正则化剪枝。**这一部分在学习决策树原理时应该掌握的，这里就不重复了。

**（4）“Early Stopping”：**Early Stopping 是机器学习迭代式训练模型中很常见的防止过拟合技巧，具体的做法是选择一部分样本作为验证集，在迭代拟合训练集的过程中，如果模型在验证集里错误率不再下降，就停止训练，也就是说控制迭代的轮数（树的个数）。在 sklearn 的 GBDT 中可以设置参数 n_iter_no_change 实现early stopping。

**（5）“Dropout”：** Dropout 是 deep learning 里很常用的正则化技巧，很自然的我们会想能不能把 Dropout 用到 GBDT 模型上呢？AISTATS2015有篇文章《DART: Dropouts meet Multiple Additive Regression Trees》进行了一些尝试。文中提到GBDT 里会出现 over-specialization 的问题：前面迭代的树对预测值的贡献比较大，后面的树会集中预测一小部分样本的偏差。Shrinkage 可以减轻 over-specialization 的问题，但不是很好。作者想通过 Dropout 来平衡所有树对预测的贡献。

具体的做法是：每次新加一棵树，这棵树要拟合的并不是之前全部树 ensemble 后的残差，而是随机抽取的一些树 ensemble ；同时新加的树结果要规范化一下。对这一部分感兴趣的同学可以阅读一下原论文。

## **6. 关于 GBDT 若干问题的思考**

**（1）GBDT 与 AdaBoost的区别与联系？**

AdaBoost 和 GBDT 都是重复选择一个表现一般的模型并且每次基于先前模型的表现进行调整。不同的是，AdaBoost 是通过调整错分数据点的权重来改进模型，GBDT 是通过计算负梯度来改进模型。因此，相比 AdaBoost , GBDT 可以使用更多种类的目标函数，而当目标函数是均方误差时，计算损失函数的负梯度值在当前模型的值即为残差。

**（2）GBDT 与随机森林（Random Forest，RF）的区别与联系？**

**相同点：**都是由多棵树组成，最终的结果都是由多棵树一起决定。

**不同点：**1）集成的方式：随机森林属于 Bagging 思想，而 GBDT 是 Boosting 思想。2）偏差-方差权衡：RF 不断的降低模型的方差，而 GBDT 不断的降低模型的偏差。3）训练样本方式：RF 每次迭代的样本是从全部训练集中有放回抽样形成的，而 GBDT 每次使用全部样本。4）并行性：RF 的树可以并行生成，而 GBDT 只能顺序生成(需要等上一棵树完全生成)。5）最终结果：RF 最终是多棵树进行多数表决（回归问题是取平均），而 GBDT 是加权融合。6）数据敏感性：RF 对异常值不敏感，而 GBDT 对异常值比较敏感。7）泛化能力：RF 不易过拟合，而 GBDT 容易过拟合。

**（3）我们知道残差 = 真实值 - 预测值，明明可以很方便的计算出来，为什么 GBDT 的残差要用用负梯度来代替？为什么要引入麻烦的梯度？有什么用呢？**

**回答第一小问：**在 GBDT 中，无论损失函数是什么形式，每个决策树拟合的都是负梯度。准确的说，不是用负梯度代替残差，而是当损失函数是均方损失时，负梯度刚好是残差，残差只是特例。

**回答二三小问：**GBDT 的求解过程就是梯度下降在函数空间中的优化过程。在函数空间中优化，每次得到增量函数，这个函数就是 GBDT 中一个个决策树，负梯度会拟合这个函数。要得到最终的 GBDT 模型，只需要把初始值或者初始的函数加上每次的增量即可。我这里高度概括的回答了这个问题，详细推理过程可以参考：

[梯度提升（Gradient Boosting）算法，地址：https://mp.weixin.qq.com/s/Ods1PHhYyjkRA8bS16OfCg](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s/Ods1PHhYyjkRA8bS16OfCg)

## **7. 总结**

在本文中，我们首先引出回归树与梯度提升算法结合的优势；然后详细推导了 GBDT 回归算法的原理，并用实际案例解释 GBDT 回归算法；其次不仅用 Python3实现 GBDT 回归算法，还用 sklearn 实现 GBDT 回归算法；最后，介绍了 GBDT 回归任务常见的损失函数、GBDT 的正则化 和我对 GBDT 回归算法的若干问题的思考。GBDT 中的树是回归树（不是分类树），GBDT 可以用来做回归预测，这也是我们本文讲的 GBDT 回归算法，但是 GBDT 调整后也可以用于分类任务。让我们期待一下 GBDT 分类算法，在分类任务中的表现吧！

## **8. Reference**

由于参考的文献较多，我把每一部分都重点参考了哪些文章详细标注一下。

