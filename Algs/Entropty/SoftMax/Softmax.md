> [https://www.zhihu.com/question/23765351](https://www.zhihu.com/question/23765351)

## Softmax

Softmax将多个输出限定在之间，并且满足概率之和为1，通过softmax可以将信号呈指数的加大或者减少，突出想要增强的信息，

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Entropty/softmax2.png)

因此在多分类输出层经常加入softmax，完成多分类的目的，在损失函数上采样交叉熵损失去评估真实标签与预测值之间分布的差异，如图2所示，$y_1,y_2,y_3$ 是预测值，是 $\hat{y_1},\hat{y_2},\hat{y_3}$ 真实值，在多分类中其取值往往为 onehot 的编码方式，而如果直接将 softmax 应用在多标签分类中(即取值 $\hat{y}$ 不是onehot, 是multihot),有可能存在问题，后面将通过softmax的推导来说明这个问题。

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Entropty/cross_entropy.png)

为了与论文公式符号保持一致(避免与上述截图中的符号搞混)，本文将应用到的符号先提前定义声明,

- $C$ : 表示类别，或者标签的总数

- $y_i$ : 表示样本的真实的标签(在本文的语境中是bounding box的标签)

- 交叉熵损失函数表示为 : 
  $$
  L = - \sum_{i=1}^{C}y_ilog(\sigma_i)
  $$
  其中 
  $$
  \sigma_i=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
  $$

- 对于 $z_1,z_2,\cdots,z_C$ 相关的全连接参数定义为 $w_1,w_2,\cdots,w_C$ , bias项为 $b_1,b_2,\cdots,b_C$

由于模型的参数为 $w,b$ , 根据链式法则，可以得到如下公式:
$$
\frac{\partial{L}}{\partial{w_i}} = \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial w_i}
\\
\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial z_i} \frac{\partial z_i}{\partial b_i}
$$
显然，根据 softmax 的计算过程，以及交叉熵损失函数 $L$，
$$
\begin{matrix}
z_1 &= w_1^T x + b_1
\\
z_2 &= w_2^T x + b_2
\\
\vdots
\\
z_i &= w_i^T x + b_i
\\
z_C &= w_C^T x + b_C
\\
\end{matrix}
$$
可以快速得到
$$
\frac{\partial z_i}{\partial w_i}=x,\frac{\partial z_i}{\partial b_i} = 1
$$
 , 因为我们的主要目标是计算得到 
$$
\frac{\partial L}{\partial z_i}
$$
由于 $L$ 计算公式中，包含了 $\sigma_1,\sigma_2,\cdots,\sigma_i,\sigma_C$ ，而各个 $\sigma$ 中又包含了 $z_1,z_2,\cdots,z_i,z_C$ ， 根据链式法则，可以画出求
$$
\frac{\partial L}{\partial z_i}
$$
变量之间的依赖关系图如下，以及求导过程：

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Entropty/softmax3.png)
$$
\begin{aligned} 
\frac{\partial L}{ \partial z_i} 
&= 
\frac{\partial L}{ \partial \sigma_1}
\frac{\partial \sigma_1}{\partial z_i} +  
\frac{\partial L}{ \partial \sigma_2} 
\frac{\partial \sigma_2}{\partial z_i} + 
\cdots + 
\frac{\partial L}{ \partial \sigma_C} 
\frac{\partial \sigma_C}{\partial z_i} 
\\  
&= \sum_{k=1}^C \frac{\partial L}{ \partial \sigma_k} \frac{\partial \sigma_k}{ \partial z_i} \end{aligned} 
$$
对于上述求导公式，首先根据交叉熵损失函数的公式：
$$
L = -\sum_{i=1}^{C}y_i \ log(\sigma_i)
$$
, 将其看做是关于 $\sigma_i$ 的函数，因此可以得到:
$$
\frac{\partial L}{\partial \sigma_k} = - \frac{y_k}{\sigma_k}
$$
对于 
$$
\frac{\partial \sigma_k}{\partial z_i}
$$
, 需要分两种情况来考虑

**当** $k \ne i$ **时**, (将看 $e^{z_k}$ 做是常数)， 求导过程如下
$$
\begin{aligned}
\sigma_k &= \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}
\\ 
\frac{\partial \sigma_k}{ \partial z_i} 
&= 
\frac{- e^{z_k} e^{z_i}}{ (\sum_{j=1}^C e^{z_j})^2 } \\ 
&= - \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}} 
\frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} 
\\ 
&= - \sigma_k \sigma_i 
\end{aligned}
$$
**当** $k=i$ **时**(不能将 $e^{z_k}$ 看做是常数)， 求导过程如下
$$
\begin{aligned} 
\sigma_k 
&= \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}} 
\\ 
\frac{\partial \sigma_k}{ \partial z_i} 
&= \frac{e^{z_i}(\sum_{j=1}^C e^{z_j}) - e^{z_i}  e^{z_i}}{(\sum_{j=1}^C e^{z_j})^2} 
\\  
&= \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} - \frac{e^{z_i}  e^{z_i}}{(\sum_{j=1}^C e^{z_j})^2} 
\\ 
&= \sigma_i - \sigma_i^2 
\\ 
&= \sigma_i(1 - \sigma_i)
\end{aligned}
$$
综合上述的求导过程，可以得到:
$$
\begin{aligned}     
\frac{\partial L}{ \partial z_i} 
&= \sum_{k=1}^C \frac{\partial L}{ \partial \sigma_k} \frac{\partial \sigma_k}{ \partial z_i} 
\\     
&= \sum_{k=1}^C -\frac{y_k}{\sigma_k} \frac{\partial \sigma_k}{ \partial z_i} 
\\     
&= -\frac{y_k}{\sigma_k} \frac{\partial \sigma_k}{ \partial z_i} + \sum_{k=1,k\neq i}^C -\frac{y_k}{\sigma_k} \frac{\partial \sigma_k}{ \partial z_i} 
\\      
&= -\frac{y_i}{\sigma_i} \sigma_i(1 - \sigma_i) + \sum_{k=1,k\neq i}^C -\frac{y_k}{\sigma_k} (- \sigma_k \sigma_i) 
\\      
&= y_i(\sigma_i - 1) + \sum_{k=1,k\neq i}^C y_k \sigma_i 
\\      
&= -y_i + y_i \sigma_i + y_1 \sigma_i + y_2 \sigma_i + \cdots + y_{i-1} \sigma_i + y_{i+1} \sigma_i + \cdots  + y_k \sigma_i 
\\
&= -y_i + \sigma_i \sum_{k=1}^C y_k 
\end{aligned}
$$
在多分类问题中, $y_k$ 表示的是样本标签属于 $k$ 类别，$y_k = [0,0,0,\cdots,1,0,0,\cdots]$ ，整个向量为 onehot 编码，因此上式中 $\sum_{k=1}^{C} y_k = 1$ , 因此，
$$
\begin{aligned}     
\frac{\partial L}{ \partial z_i} 
&=  \sigma_i - y_i 
\\
\frac{\partial L }{\partial w_i} 
&= \frac{\partial L}{\partial z_i}
\frac{\partial z_i}{ \partial w_i} = ( \sigma_i - y_i) x 
\\     
\frac{\partial L }{\partial b_i} 
&= \frac{\partial L}{\partial z_i} \frac{\partial z_i}{ \partial b_i} =  \sigma_i - y_i 
\end{aligned} 
\\
$$

------------------

- Hardmax
- Softmax

-----------------------

**Softmax 是 LogSumExp 的偏微分。LogSumExp 是 max 函数的 smooth 版本。**

------------------------------

为什么要取指数，第一个原因是要模拟max的行为，所以要让大的更大。第二个原因是需要一个可导的函数。

------------------------------------------

就是如果某一个 $z_j$ 大过其他 $z$ ,那这个映射的分量就逼近于1,其他就逼近于0，主要应用就是多分类，sigmoid函数只能分两类，而softmax能分多类，softmax是sigmoid的扩展。

----------------------------

归一化过程：向量的元素（单位化）标准化，使得在对 output 层每个节点概率求和值为1，方便分类（classification）

K个不同的线性函数得到的结果：
$$
P(y=j|x) = \frac{e^{x^{'}w_j}}{\sum_{k=1}^{K}e^{x^Tw_k}}
$$

> That's much closer to the argmax! Because we use the natural exponential, we hugely increase the probability of the biggest score and decrease the probability of the lower scores when compared with standard normalization. Hence the "max" in softmax.

> So, you want to pick a constant big enough to approximate argmax well, and also small enough to express these big and small numbers in calculations.
>
> And of course, ![[公式]](https://www.zhihu.com/equation?tex=e) also has pretty nice derivative.

> Hence, this shows us that the softmax function is the function that is maximizing the entropy in the distribution of images.

--------------

![](/Users/helloword/Anmingyu/Gor-rok/Daily/Entropty/logits_softmax_prob.png)