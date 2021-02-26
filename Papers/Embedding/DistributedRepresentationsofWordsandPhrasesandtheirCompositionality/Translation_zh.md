# Distributed Representations of Words and Phrases and their Compositionality

## Abstract

最近提出的 skip-gram 模型是一种学习高质量分布式向量表示的有效方法，能够捕捉大量精确的词的语法和语义关系。在本文中，我们提出了几种扩展，既提高了向量的质量，又提高了训练速度。通过对频繁词的下采样，我们获得了显著的加速，也学习了更多常规单词的表示。

我们还描述了一种简单的 hierarchical softmax 替代方案，称为 negative sampling。词表示的一个硬伤是它们不关心词序，和它们不能代表习惯短语。例如，“Canada”和“Air”的意思不能轻易组合成“Air Canada”。在这个例子的启发下，我们提出了一个简单的方法来在文本中寻找短语，并表明学习数百万短语的向量表示是可能的。

## 1 Introduction

