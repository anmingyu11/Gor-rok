> Jerome H, Friedman* IMS 1999 Reitz Lecture

# Greedy Function Approximation: A Gradient Boosting Machine

## Abstract

Function estimation/approximation is viewed from the perspective of numerical optimization in function space, rather than parameter space. A connection is made between stagewise additive expansions and steepest-descent minimization. A general gradientdescent “boosting” paradigm is developed for additive expansions based on any fitting criterion. Specific algorithms are presented for least-squares, least-absolute-deviation, and Huber-M loss functions for regression, and multi-class logistic likelihood for classification. 

Special enhancements are derived for the particular case where the individual additive components are regression trees, and tools for interpreting such “TreeBoost” models are presented. Gradient boosting of regression trees produces competitive, highly robust, interpretable procedures for both regression and classification, especially appropriate for mining less than clean data. Connections between this approach and the boosting methods of Freund and Shapire 1996, and Friedman, Hastie, and Tibshirani 2000 are discussed.

> 函数 估计/逼近 从函数空间的数值优化的角度来看待，而不是从参数空间来看待。在 additive expansions 和 steepest-descent 最小化之间建立了联系。提出了一种基于任意拟合准则的加性展开的通用梯度下降“boosting”范式。具体的算法提出了 least-squares，least-absolute-deviation，Huber-M 损失函数的回归，和多分类 (logistic 似然)。
>
> 针对特定情况（其中各个加法成分是回归树）进行了特殊增强，并提供了用于解释此类“ TreeBoost”模型的工具。回归树的梯度提升为回归和分类提供了具有竞争力的、高度健壮的、可解释的程序，特别适合挖掘不干净的数据。 本文讨论了这一方法与 Freund and Shapire 1996 和 Friedman, Hastie和Tibshirani 2000 的 boosting 方法之间的联系。

## 1 Function estimation

In the function estimation or “predictive learning” problem, one has a system consisting of a random “output” or “response” variable y and a set of random “input” or “explanatory” variables x = {xi, • • -,xn}. Using a “training” sample of known (y, x)-values, the goal is to obtain an estimate or approximation F(x), of the function F*(x) mapping x to y, that minimizes the expected value of some specified loss function L(y, F(x)) over the joint distribution of all (y, x)-values