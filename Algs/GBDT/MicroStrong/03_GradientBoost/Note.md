> https://zhuanlan.zhihu.com/p/86354141

# 梯度提升（Gradient Boosting）算法

> 目录：
> \1. 引言
> \2. 梯度下降法
> \3. 梯度提升算法
> \4. 梯度提升原理推导
> \5. 对梯度提升算法的若干思考
> \6. 总结
> \7. Reference

## **1. 引言**

提升树利用加法模型与前向分歩算法实现学习的优化过程。当损失函数是平方误差损失函数和指数损失函数时，每一步优化是很简单的。但对一般损失函数而言，往往每一步优化并不那么容易。针对这一问题，Freidman 提出了梯度提升（gradient boosting）算法。Gradient Boosting 是 Boosting 中的一大类算法，它的思想借鉴于梯度下降法，其基本原理是根据当前模型损失函数的负梯度信息来训练新加入的弱分类器，然后将训练好的弱分类器以累加的形式结合到现有模型中。采用决策树作为弱分类器的 Gradient Boosting 算法被称为 GBDT ，有时又被称为MART（Multiple Additive Regression Tree）。GBDT 中使用的决策树通常为CART。

## **2. 梯度下降法**

在机器学习任务中，需要最小化损失函数 $L(\theta)$ ，其中 $\theta$ 是要求解的模型参数。梯度下降法通常用来求解这种无约束最优化问题，它是一种迭代方法：选取初值 $\theta_0$ ，不断迭代，更新 $\theta$ 的值，进行损失函数的极小化。这里我们还需要初始化算法终止距离 $\varepsilon$ 以及步长 $\alpha$ 。

**使用梯度下降法求解的基本步骤为：**

> （1）确定当前位置的损失函数的梯度，对于 $\theta_t$ ，其梯度表达式如下： $\frac{\delta L(\theta)}{\delta \theta_{t}}$
>
> （2）用步长 $\alpha$ 乘以损失函数的梯度，得到当前位置下降的距离，即： $\alpha * \frac{\delta L(\theta)}{\delta \theta_{t}}$
>
> （3）确定是否 $\theta_t$ 梯度下降的距离小于 $\varepsilon$ ，如果小于 $\varepsilon$ 则算法终止，当前的 $\theta_t$ 即为最终结果。否则进入步骤（4）。
>
> （4）更新 $\theta_{t}$ ，其更新表达式如下。更新完毕后继续转入步骤（1）。
> $$
> \theta_{t}=\theta_{t}-\alpha * \frac{L(\theta)}{\delta \theta_{t}}
> $$

我们也可以用泰勒公式表示损失函数，用更数学的方式解释梯度下降法：

- 迭代公式： $\theta_{t}=\theta_{t-1}+\Delta \theta$

- 将 $L(\theta_t)$ 在 $\theta_{t-1}$ 处进行一阶泰勒展开：
  $$
  \begin{aligned}
  L\left(\theta_{t}\right) &=L\left(\theta_{t-1}+\Delta \theta\right) \\
  & \approx L\left(\theta_{t-1}\right)+L^{\prime}\left(\theta_{t-1}\right) \Delta \theta
  \end{aligned}
  $$

> 解释：要使得 $L(\theta_t) < L(\theta_{t-1})$ ，可取：$\Delta \theta=-\alpha L^{\prime}\left(\theta_{t-1}\right)$ 如何推导？
> 用到概念：梯度的负方向是函数值局部下降最快的方向
> ,则 $L\left(\theta_{t-1}+\Delta \theta\right)-L\left(\theta_{t-1}\right) \approx L^{\prime}\left(\theta_{t-1}\right) \Delta \theta$ ，
>
> 则我们可以得出：$L'(\theta_{t-1})$ 为函数值的变化量，我们要注意的是 $L^{\prime}\left(\theta_{t-1}\right)$ 和 $\Delta\theta$ 均为向量，$L^{\prime}\left(\theta_{t-1}\right) \Delta \theta$ 也就是两个向量进行点积，而向量进行点积的最大值，也就是两者共线的时候，也就是说 $\Delta \theta$ 的方向和 $L^{\prime}\left(\theta_{t-1}\right)$ 方向相同的时候，点积值最大，这个点积值也代表了从 $L(\theta_{t-1})$ 点到 $L\left(\theta_{t-1}+\Delta \theta\right)$ 点的上升量。
> 而 $L'(\theta_{t-1})$ 正是代表函数值在处的 $\theta_{t-1}$ 梯度。前面又说明了 $\Delta\theta$ 的方向和 $L'(\theta_{t-1})$ 方向相同的时候，点积值（变化值）最大，所以说明了梯度方向是函数局部上升最快的方向。也就证明了梯度的负方向是局部下降最快的方向。所以 $\Delta \theta=-\alpha L^{\prime}\left(\theta_{t-1}\right)$ ，这里的 $\alpha$ 是步长，可通过 line search 确定，但一般直接赋一个小的数。

这里多说一点，我为什么要用泰勒公式推导梯度下降法，是因为我们在面试中经常会被问到 GBDT 与 XGBoost 的区别和联系？其中一个重要的回答就是：GBDT 在模型训练时只使用了代价函数的一阶导数信息，XGBoost 对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。当然，GBDT 和 XGBoost 还有许多其它的区别与联系，感兴趣的同学可以自己查阅一些相关的资料。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/03_GradientBoost/Fig1.png)

## **3. 梯度提升算法**

在梯度下降法中，我们可以看出，对于最终的最优解 $\theta^*$ ，是由初始值 $\theta_0$ 经过 $T$ 次迭代之后得到的，这里设 $\theta_{0}=-\frac{\delta L(\theta)}{\delta \theta_{0}}$ ，则 $\theta^*$ 为：
$$
\theta^{*}=\sum_{t=0}^{T} \alpha_{t} *\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{t-1}}
$$
其中，$\left[-\frac{\delta L(\theta)}{\delta \theta}\right]_{\theta=\theta_{t-1}}$ 表示 $\theta$ 在 $\theta_{t-1}$ 处泰勒展开式的一阶导数。

**在函数空间中**，我们也可以借鉴梯度下降的思想，进行最优函数的搜索。对于模型的损失函数 $L(y,F(x))$ ，为了能够求解出最优的函数 $F^*(x)$ ，首先设置初始值为 $ F_{0}(x)=f_{0}(x)$：

以函数 $F(x)$ 作为一个整体，与梯度下降法的更新过程一致，假设经过 $T$ 次迭代得到最优的函数为 $F^*(x)$：
$$
F^{*}(x)=\sum_{t=0}^{T} f_{t}(x)
$$
其中， $f_t(x)$ 为：
$$
f_{t}(x)=-\alpha_{t} g_{t}(x)=-\alpha_{t} *\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}
$$
可以看到，这里的梯度变量是一个函数，是在函数空间上求解，而我们以前梯度下降算法是在多维参数空间中的负梯度方向，变量是参数。为什么是多维参数，因为一个机器学习模型中可以存在多个参数。而这里的变量是函数，更新函数通过当前函数的负梯度方向来修正模型，使模型更优，最后累加的模型为近似最优函数。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/03_GradientBoost/Fig2.png)

**总结：Gradient Boosting 算法在每一轮迭代中，首先计算出当前模型在所有样本上的负梯度，然后以该值为目标训练一个新的弱分类器进行拟合并计算出该弱分类器的权重，最终实现对模型的更新。**

## **4. 梯度提升原理推导**

在梯度提升的 $0 \leq t \leq T$ 步中，假设已经有一些不完美的模型 $F_{t-1}$ (最初可以使用非常弱的模型，它只是预测输出训练集的平均值)。梯度提升算法不改变 $F_{t-1}$ ，而是通过增加估计器 $f$ 构建新的模型 $F_{t}(x)=F_{t-1}(x)+f(x)$ 来提高整体模型的效果。那么问题来了，如何寻找 $f$ 函数呢？梯度提升方法的解决办法是认为最好的  $f$ 应该使得：
$$
F_{t}(x)=F_{t-1}(x)+f(x)=y
$$
或者等价于：
$$
f(x)=y-F_{t-1}(x)
$$
因此，梯度提升算法将 $f$ 与残差 $y-F_{t-1}(x)$ 拟合。与其他 boosting 算法的变体一样，  $F_t$ 修正它的前身 $F_{t-1}$ 。我们观察到残差 $y-F_{t-1}(x)$ 是损失函数 $\frac{1}{2}(y-F(x))^{2}$ 的负梯度方向，因此可以将其推广到其他不是平方误差(分类或是排序问题)的损失函数。也就是说，梯度提升算法是一种梯度下降算法，只需要更改损失函数和梯度就能将其推广。

当采用平方误差损失函数时， $L(y, f(x))=(y-f(x))^{2}$ ，其损失函数变为：
$$
\begin{array}{l}
L\left(y, f_{t}(x)\right) \\
=L\left(y, f_{t-1}(x)+T\left(x ; \theta_{t}\right)\right) \\
=\left(y-f_{t-1}(x)-T\left(x ; \theta_{t}\right)\right)^{2} \\
=\left(r-T\left(x ; \theta_{t}\right)\right)^{2}
\end{array}
$$
其中， $r=y-f_{t-1}(x)$ 是当前模型拟合数据的**残差（residual）**。在使用更一般的损失函数时，我们使用损失函数的负梯度在当前模型的值 
$$
-\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}
$$
作为提升树算法中残差的近似值，拟合一个梯度提升模型。当使用一般的损失函数时，为什么会出现上式的结果呢？下面我们就来详细阐述。

我们知道，对函数 $f(x)$ 在 $x = x_{t-1}$ 处的泰勒展示式为：
$$
f(x) \approx f\left(x_{t-1}\right)+f^{\prime}\left(x_{t-1}\right)\left(x-x_{t-1}\right)
$$
因此，损失函数 $L(y,F(x))$  在 $F(x) = F_{t-1}(x)$ 处的泰勒展开式就是：
$$
L(y, F(x)) \approx L\left(y, F_{t-1}(x)\right)+\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}\left(F(x)-F_{t-1}(x)\right)
$$
将 $F(x) = F_{t-1}(x)$ 带入上式，可得：
$$
L\left(y, F_{t}(x)\right) \approx L\left(y, F_{t-1}(x)\right)+\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}\left(F_{t}(x)-F_{t-1}(x)\right)
$$
因此，
$$
-\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}
$$
应该对应于平方误差损失函数中的 $y -f_{t-1}(x)$ ，这也是我们为什么说**对于平方损失函数拟合的是残差；对于一般损失函数，拟合的就是残差的近似值。**

## **5. 对梯度提升算法的若干思考**

**（1）梯度提升与梯度下降的区别和联系是什么？**

GBDT 使用梯度提升算法作为训练方法，而在逻辑回归或者神经网络的训练过程中往往采用梯度下降作为训练方法，二者之间有什么联系和区别呢？

下表是梯度提升算法和梯度下降算法的对比情况。可以发现，两者都是在每一轮迭代中，利用损失函数相对于模型的负梯度方向的信息来对当前模型进行更新，只不过在梯度下降中，模型是以参数化形式表示，从而模型的更新等价于参数的更新。而在梯度提升中，模型并不需要进行参数化表示，而是直接定义在函数空间中，从而大大扩展了可以使用的模型种类。

![](/Users/helloword/Anmingyu/Gor-rok/Algs/GBDT/MicroStrong/03_GradientBoost/Fig3.png)

**（2）梯度提升和提升树算法的区别和联系？**

提升树利用加法模型与前向分歩算法实现学习的优化过程。当损失函数是平方误差损失函数和指数损失函数时，每一步优化是很简单的。但对一般损失函数而言，往往每一步优化并不那么容易。

针对这一问题，Freidman提出了梯度提升（gradient boosting）算法。这是利用损失函数的负梯度在当前模型的值 
$$
-\left[\frac{\delta L(y, F(x))}{\delta F(x)}\right]_{F(x)=F_{t-1}(x)}
$$
作为提升树算法中残差的近似值，拟合一个梯度提升模型。

**（3）梯度提升和 GBDT 的区别和联系？**

- 采用决策树作为弱分类器的 Gradient Boosting 算法被称为 GBDT，有时又被称为 MART（Multiple Additive Regression Tree）。GBDT 中使用的决策树通常为 CART。
- GBDT 使用梯度提升（Gradient Boosting）作为训练方法。 

**（4）梯度提升算法包含哪些算法？**

Gradient Boosting 是 Boosting 中的一大类算法，其中包括：GBDT（Gradient Boosting Decision Tree）、XGBoost（eXtreme Gradient Boosting）、LightGBM （Light Gradient Boosting Machine）和 CatBoost（Categorical Boosting）等。

**（5）对于一般损失函数而言，为什么可以利用损失函数的负梯度在当前模型的值作为梯度提升算法中残差的近似值呢？**

我们观察到在提升树算法中，残差 $y-F_{t-1}(x)$ 是损失函数 $\frac{1}{2}(y-F(x))^{2}$  的负梯度方向，因此可以将其推广到其他不是平方误差(分类或是排序问题)的损失函数。也就是说，梯度提升算法是一种梯度下降算法，不同之处在于更改损失函数和求其负梯度就能将其推广。即，可以将结论推广为对于一般损失函数也可以利用损失函数的负梯度近似拟合残差。

## **6. 总结**

我之前已经写过关于的文章。回归树可以利用集成学习中的 Boosting 框架改良升级得到提升树，提升树再经过梯度提升算法改造就可以得到 GBDT 算法，GBDT 再进一步可以升级为 XGBoost、LightGBM 或者 CatBoost 。在学习这些模型的时候，我们把它们前后连接起来，就能更加系统的理解这些模型的区别与联系。

