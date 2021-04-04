> https://zhuanlan.zhihu.com/p/84139957

# 深入理解提升树（Boosting tree）算法

>目录：
>\1. Boosting基本概念
>\2. 前向分步加法模型
>2.1 加法模型
>2.2 前向分步算法
>\3. 提升树
>3.1 提升树模型
>3.2 提升树算法
>3.2.1 二叉分类提升树
>3.2.1 二叉回归提升树
>\4. 回归提升树示例
>\5. 完整的示例代码
>\6. 关于提升树的若干问题思考
>\7. 总结
>\8. Reference

## **1. Boosting基本概念**

提升（Boosting）方法是一种常用的统计学习方法，应用广泛且有效。在分类问题中，它通过改变训练样本的权重，学习多个分类器，并将这些分类器进行线性组合，提高分类的性能。

提升方法基于这样一种思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合所得出的判断，要比其中任何一个专家单独的判断好。实际上，就是“三个臭皮匠顶个诸葛亮”的道理。

历史上，Kearns 和 Valiant 首先提出了“强可学习（strongly learnable）”和“弱可学习（weakly learnable）”的概念。指出：在概率近似正确（probably approximately correct，PAC）学习的框架中，一个概念（一个类），如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么就称这个概念是 **强可学习** 的；一个概念，如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么就称这个概念是**弱可学习**的。非常有趣的是Schapire后来证明强可学习与弱可学习是等价的，也就是说，在 PAC 学习的框架下，一个概念是强可学习的充分必要条件是这个概念是弱可学习的。

这样一来，问题便成为，在学习中，如果已经发现了“弱学习算法”，那么能否将它提升（boost）为“强学习算法”。大家知道，发现弱学习算法通常要比发现强学习算法容易得多。那么如何具体实施提升，便成为开发提升方法时所要解决的问题。关于提升方法的研究很多，有很多算法被提出。最具代表性的是 AdaBoost 算法（AdaBoost algorithm）。

**Boosting算法的两个核心问题：**

（1）在每一轮如何改变训练数据的权值或概率分布？

AdaBoost 的做法是，提高那些被前一轮弱分类器错误分类样本的权值，而降低那些被正确分类样本的权值。这样一来，那些没有得到正确分类的数据，由于其权值的加大而受到后一轮的弱分类器的更大关注。于是，分类问题被一系列的弱分类器“分而治之”。

（2）如何将弱分类器组合成一个强分类器？

弱分类器的组合，AdaBoost 采取加权多数表决的方法。具体地，加大分类误差率小的弱分类器的权值，使其在表决中起较大的作用，减小分类误差率大的弱分类器的权值，使其在表决中起较小的作用。

提升树是以分类树或回归树为基本分类器的提升方法。提升树被认为是统计学习中性能最好的方法之一。提升方法实际采用加法模型（即基函数的线性组合）与前向分步算法。以决策树为基函数的提升方法称为提升树（boosting tree）。对分类问题决策树是二叉分类树，对回归问题决策树是二叉回归树。下面让我们深入理解提升树的具体算法吧！

## **2. 前向分步加法模型**

**2.1 加法模型**

考虑加法模型（Additive Model）如下：
$$
f(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)
$$
其中，$b\left(x ; \gamma_{m}\right)$  为基函数，$\gamma_{m}$ 为基函数的参数， $\beta_{m}$ 为基函数的系数。显然上式是一个加法模型。

**2.2 前向分布算法**

在给定训练数据及损失函数 $L(Y, f(x))$ 的条件下，学习加法模型 $f(x)$ 成为经验风险极小化，即损失函数极小化的问题：
$$
\min _{\left(\beta_{m}, \gamma_{m}\right)} \sum_{i=1}^{N} L\left(y_{i}, \sum_{m=1}^{M} \beta_{m} b\left(x_{i} ; \gamma_{m}\right)\right)
$$
通常这是一个复杂的优化问题。前向分布算法（forward stagewise algorithm）求解这一优化问题的想法是：因为学习的是加法模型，如果能够从前向后，每一步只学习一个基函数及其系数，逐步逼近上面要优化的目标函数，那么就可以简化优化的复杂度。

具体地，每步只需优化如下损失函数：
$$
\min _{(\beta, \gamma)} \sum_{i=1}^{N} L\left(y_{i}, \beta b\left(x_{i} ; \gamma\right)\right)
$$
给定训练数据集
$$
T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{N}, y_{N}\right)\right\}, x_{i} \in X \subseteq R^{n}, y_{i} \in Y=\{-1,+1\}
$$
损失函数 $L(Y, f(x))$ 和基函数的集合  $\{b(X ; \gamma)\}$，学习加法模型 $f(x)$ 的前向分步算法如下：

**前向分步算法步骤如下：**

**输入：**训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{N}, y_{N}\right)\right\}$ ；损失函数 $L(Y,f(x))$；基函数集 $\{b(X ; \gamma)\}$；

**输出：**加法模型 $f(x)$。

（1）初始化 $f_{0}(x)=0$

（2）对 $m=1,2, \ldots, M$

（a）极小化损失函数：
$$
\left(\beta_{m}, \gamma_{m}\right)=\operatorname{argmin}_{\beta, \gamma} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+\beta b\left(x_{i} ; \gamma\right)\right)
$$
得到参数 $\beta_{m}, \gamma_{m}$

（b）更新：
$$
f_{m}(x)=f_{m-1}(x)+\beta_{m} b\left(x ; \gamma_{m}\right)
$$
（3）得到加法模型：
$$
f(x)=f_{M}(x)=\sum_{m=1}^{M} \beta_{m} b\left(x ; \gamma_{m}\right)
$$
这样，前向分步算法将同时求解从 $m=1$ 到 $M$ 的所有参数 $\beta_{m}, \quad \gamma_{m}$ 的优化问题简化为逐次求解各个 $\beta_{m}, \quad \gamma_{m}$ 的优化问题。

## **3. 提升树**

提升树是以分类树或回归树为基本分类器的提升方法。提升树被认为是统计学习中性能最好的方法之一。

**3.1 提升树模型**

提升方法实际采用加法模型（即基函数的线性组合）与前向分步算法。以决策树为基函数的提升方法称为提升树（boosting tree）。对分类问题决策树是二叉分类树，对回归问题决策树是二叉回归树。提升树模型可以表示为决策树的加法模型：
$$
f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)
$$
其中，$T(x;\Theta_m)$  表示决策树； $\Theta_m$ 为决策树的参数；$M$ 为树的个数。

**3.2 提升树算法**

提升树算法采用前向分步算法。首先确定初始提升树 $f_0(x)=0$ ，第 $m$ 步的模型是：
$$
f_{m}(x)=f_{m-1}(x)+T\left(x ; \Theta_{m}\right)
$$
其中， $f_{m-1}(x)$ 为当前模型，通过经验风险极小化确定下一棵决策树的参数  $\Theta_{m}$:
$$
\hat{\Theta}_{m}=\operatorname{argmin}_{\left(\Theta_{m}\right)} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+T\left(x_{i} ; \Theta_{m}\right)\right)
$$
由于树的线性组合可以很好地拟合训练数据，即使数据中的输入与输出之间的关系很复杂也是如此，所以提升树是一个高性能的学习算法。

下面讨论针对不同问题的提升树学习算法，其主要区别在于使用的损失函数不同。包括用平方误差损失函数的回归问题，用指数损失函数的分类问题，以及用一般损失函数的一般决策问题。

**3.2.1 二叉分类提升树**

对于二分类问题，提升树算法只需将 AdaBoost 算法中的基本分类器限制为二类分类树即可，可以说这时的提升树算法是 AdaBoost 算法的特殊情况，这里不再细述。下面叙述回归问题的提升树。

**3.2.2 二叉回归提升树**

已知一个训练数据集 $T=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{N}, y_{N}\right)\right\}, \quad x_{i} \in X \subseteq R^{n}$ ，$x$ 为输入空间, $y_{i} \in Y \subseteq R$ , $y$ 为输出空间。如果将输入空间 $x$ 划分为 $J$ 个互不相交的区域 $R_{1}, R_{2}, \ldots, R_{J}$ , 并且在每个区域上确定输出的常量 $m_j$ ，那么树可表示为：
$$
T(x ; \Theta)=\sum_{j=1}^{J} c_{j} I\left(x \in R_{j}\right)
$$
其中，参数 $\Theta=\left\{\left(R_{1}, c_{1}\right),\left(R_{2}, c_{2}\right), \ldots,\left(R_{J}, c_{J}\right)\right\}$ 表示树的区域划分和各区域上的常数。$J$ 是回归树的复杂度即叶结点个数。

回归问题提升树使用以下前向分步算法：
$$
\begin{array}{l}
f_{0}(x)=0 \\
f_{m}(x)=f_{m-1}(x)+T\left(x ; \Theta_{m}\right), m=1,2, \ldots, M \\
f_{M}(x)=\sum_{m=1}^{M} T\left(x ; \Theta_{m}\right)
\end{array}
$$
在前向分步算法的第 $m$ 步，给定当前模型 $f_{m-1}(x)$ ，需求解：
$$
\hat{\Theta}_{m}=\operatorname{argmin}_{\left(\Theta_{m}\right)} \sum_{i=1}^{N} L\left(y_{i}, f_{m-1}\left(x_{i}\right)+T\left(x_{i} ; \Theta_{m}\right)\right)
$$
得到 $\hat{\Theta}_{m}$ ，即第 $m$ 棵树的参数。

当采用平方误差损失函数时，$L(y, f(x))=(y-f(x))^{2}$  ，其损失变为：
$$
L\left(y, f_{m-1}(x)+T\left(x ; \Theta_{m}\right)\right)=\left[y-f_{m-1}(x)-T\left(x ; \Theta_{m}\right)\right]^{2}=\left[r-T\left(x ; \Theta_{m}\right)\right]^{2}
$$
这里，$r=y-f_{m-1}(x)$ ，是当前模型拟合数据的残差（residual）。所以，对回归问题的提升树算法来说，只需简单地拟合当前模型的残差。这样，算法是相当简单的。现在将回归问题的提升树算法叙述如下：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Alg1.png)

## **4. 回归提升树示例**

本示例来源于李航著的《统计学习方法》第 8 章提升方法中的例 8.2。已知如表1所示的训练数据，$x$ 的取值范围为区间 $[0.5, 10.5]$，$y$ 的取值范围为区间 $[5.0, 10.0]$，学习这个回归问题的提升树模型，考虑只用树桩作为基函数。

**说明：树桩是由一个根节点直接连接两个叶结点的简单决策树。**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Table1.png)

- **按照算法8.3，第1步求 $f_{1}(x)$ 即回归树 $T_{1}(x)$ 。**

> **样本输入空间划分的基本步骤如下：**
>
> 首先通过以下优化问题：
> $$
> \min _{(s)}\left[\min _{\left(c_{1}\right)} \sum_{x_{i} \in R_{1}}\left(y_{i}-c_{1}\right)^{2}+\min _{\left(c_{2}\right)} \sum_{x_{i} \in R_{2}}\left(y_{i}-c_{2}\right)^{2}\right]
> $$
> 求解训练数据的切分点 $s$：
> $$
> R_{1}=\{x \mid x \leq s\}, R_{2}=\{x \mid x>s\}
> $$
> 容易求得在 $R_{1}, R_{2}$ 内部使平方损失误差达到最小的 $c_{1}, c_{2}$ 为：
> $$
> c_{1}=\frac{1}{N_{1}} \sum_{x_{i} \in R_{1}} y_{i}, c_{2}=\frac{1}{N_{2}} \sum_{x_{i} \in R_{2}} y_{i}
> $$
> 这里 $N_1,N_2$ 是  $R_1,R_2$ 的样本点数。

**（1）求训练数据的切分点**

这里的切分点指的是将 $x$ 值划分界限，数据中 $x$ 的范围是 $[1,10]$，假设我们考虑如下切分点：

> 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5

对各切分点，不难求出相应的 $R_{1}, R_{2}, c_{1}, c_{2}$ 及 $m(s)$，$m(s)$ 计算公式如下： 
$$
m(s)=\min _{\left(c_{1}\right)} \sum_{x_{i} \in R_{1}}\left(y_{i}-c_{1}\right)^{2}+\min _{\left(c_{2}\right)} \sum_{x_{i} \in R_{2}}\left(y_{i}-c_{2}\right)^{2}
$$
例如：当 $s=1.5$ 时， $R_{1}=\{1\}, R_{2}=\{2,3, \ldots, 10\}$，那么：
$$
\begin{array}{l}
c_{1}=5.56 \\
c_{2}=\frac{1}{9}(5.70+5.91+6.40+6.80+7.05+8.90+8.70+9.00+9.05)=7.50 \\
m(1.5)=\min _{\left(c_{1}\right)} \sum_{x_{i} \in R_{1}}\left(y_{i}-c_{1}\right)^{2}+\min _{\left(c_{2}\right)} \sum_{x_{i} \in R_{2}}\left(y_{i}-c_{2}\right)^{2}=0+15.72=15.72
\end{array}
$$
现将 $s$ 及 $m(s)$ 的计算结果列表如下：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig2.png)

**（2）求回归树**

由表2可知，当 $s=6.5$ 时，$m(s)$ 达到最小值，此时 $R_{1}=\{1,2, \ldots, 6\}, R_{2}=\{7,8,9,10\}$ ，且：
$$
\begin{array}{l}
c_{1}=\frac{1}{6}(5.56+5.70+5.91+6.40+6.80+7.05)=6.24 \\
c_{2}=\frac{1}{4}(8.90+8.70+9.00+9.05)=8.91
\end{array}
$$
因此，回归树 $T_1(x)$ 为：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig3.png)

**（3）求当前加法模型 $f_1(x)$**

当前的加法模型为：
$$
f_{1}(x)=T_{1}(x)
$$
**（4）求当前加法模型的残差**

用 $f_{1}(x)$ 拟合训练数据的残差如表3，表中 $r_{2 i}=y_{i}-f_{1}\left(x_{i}\right), i=1,2, \ldots, 10$ 。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Table4.png)

用$f_1(x)$ 拟合训练数据的平方损失误差为：
$$
L\left(y, f_{1}(x)\right)=\sum_{i=1}^{10}\left(y_{i}-f_{1}\left(x_{i}\right)\right)^{2}=1.93
$$
这里的误差为 $1.93$，如果我们定义终止时候的误差比这个误差要小，那么算法继续执行以上步骤，直到满足误差为止。

**第2步，求回归树 $T_2(x)$ 。方法与求 $T_1(x)$ 一样，只是拟合的数据是表3的残差。** 

**（1）求解数据的切分点**

仍然对区域 $R=\{1,2, \ldots, 10\}$ 求解数据的切分点。当 $s=1.5$ 时, $R_{1}{ }^{\prime}=\{1\}, R_{2}{ }^{\prime}=\{2,3, \ldots, 10\}$ 那么：
$$
\begin{array}{l}
c_{1^{\prime}}=-0.68 \\
c_{2^{\prime}}=\frac{1}{9}(-0.54-0.33+0.16+0.56+0.81-0.01-0.21+0.09+0.14)=0.07
\end{array}
$$

$$
m(1.5)=\min _{\left(c_{1}\right)} \sum_{x_{i} \in R_{1}}\left(r_{2 i}-c_{1}\right)^{2}+\min _{\left(c_{2}\right)} \sum_{x_{i} \in R_{2}}\left(r_{2 i}-c_{2}\right)^{2}=0+1.42=1.42
$$

现将 $s$ 及 $m(s)$ 的计算结果列表如下（见表4）：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig4.png)

**（2）求回归树**

由 表4 可知，当 $s=3.5$ 时 $m(s)$ 达到最小值，此时  $R_{1^{\prime}}=\{1,2,3\}, R_{2^{\prime}}=\{4,5, \ldots, 10\}, c_{1}=-0.52, c_{2}=0.22$，所以回归树 $T_{2}(x)$ 为：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig5.png)

**（3）求当前加法模型 $f_2(x)$**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig6.png)

**（4）求当前加法模型的残差**

用 $f_2(x)$ 拟合训练数据的残差如 表5 ，表中 $r_{3 i}=y_{i}-f_{2}\left(x_{i}\right), i=1,2, \ldots, 10$ 。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig7.png)

用 $f_2(x)$ 拟合训练数据的平方损失误差是：

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig8.png)

**之后的过程同步骤2一样，我就不在这里赘述啦！最后，给出完整的回归提升树模型。**

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/02_BoostingTree/Fig9.png)

## **5. 完整的示例代码**

略

## **6. 关于提升树的若干问题思考**

**（1）提升树与回归树之间的关系？**

以决策树为基函数的提升方法称为提升树，对分类问题决策树为二叉分类树，对回归问题决策树是二叉回归树。

**（2）提升树与梯度提升的区别？**

李航老师《统计学习方法》中提到了在使用平方误差损失函数和指数损失函数时，提升树的残差求解比较简单，但是在使用一般的损失误差函数时，残差求解起来不是那么容易。针对这一问题，Freidman 提出了梯度提升（Gradient Boosting）算法，就是利用最速下降法的近似方法，关键是利用损失函数的负梯度在当前模型的值作为回归问题提升树算法中的残差的近似值，拟合一个回归树。

**（3）提升树与 GBDT 之间的关系？**

提升树模型每一次的提升都是靠上次的预测结果与训练数据中 label 值的差值作为新的训练数据进行重新训练，**由于原始的回归树指定了平方损失函数所以可以直接计算残差，而梯度提升决策树（Gradient Boosting Decision Tree, GDBT）针对的是一般损失函数，所以采用负梯度来近似求解残差，**将残差计算替换成了损失函数的梯度方向，将上一次的预测结果带入梯度中求出本轮的训练数据**。**这两种模型就是在生成新的训练数据时采用了不同的方法。

**思考：讲到这里我又有一个问题，李航老师的《统计学习方法》中提到的梯度提升与 GBDT 又有什么区别和联系呢？**这个问题我还没有想明白，暂且留在这里吧！

## **7. 总结**

本文讨论了针对不同问题的提升树学习算法，它们的主要区别在于使用的损失函数不同。包括用平方误差损失函数的回归问题，例如，本文讲解的回归问题的提升树算法；用指数损失函数的分类问题，例如，基本分类器是二分类树的 AdaBoost 算法；以及用一般损失函数的一般决策问题，例如梯度提升算法。

Boosting 族代表性算法包括：GBDT、XGBoost（eXtreme Gradient Boosting）、LightGBM （Light Gradient Boosting Machine）和 CatBoost（Categorical Boosting）等，提升树算法是这些 Boosting 族高级算法的基础。因此，深入理解提升树算法对于我们后续学习 Boosting 族高级算法很重要。

