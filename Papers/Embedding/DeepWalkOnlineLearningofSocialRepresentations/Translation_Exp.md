## 5. EXPERIMENTAL DESIGN

In this section we provide an overview of the datasets and methods which we will use in our experiments. Code and data to reproduce our results will be available at the first author’s website.1

> 在本节中，我们概述了我们将在实验中使用的数据集和方法。复制我们结果的代码和数据将在第一作者的网站上提供。

## 5.1 Datasets

An overview of the graphs we consider in our experiments is given in Figure 1. 

- BlogCatalog [39] is a network of social relationships provided by blogger authors. The labels represent the topic categories provided by the authors.
- Flickr [39] is a network of the contacts between users of the photo sharing website. The labels represent the interest groups of the users such as ‘black and white photos’.
- YouTube [40] is a social network between users of the popular video sharing website. The labels here represent groups of viewers that enjoy common video genres (e.g. anime and wrestling).

> 图1 给出了我们在实验中考虑的图的概述。
>
> - BlogCatalog[39]是一个由博客作者提供的社会关系网络。标签代表作者提供的主题类别。
> - Flickr[39]是照片共享网站用户之间的联系人网络。标签代表了用户的兴趣组，比如“黑白照片”。
> - YouTube[40]是流行的视频分享网站用户之间的社交网络。这里的标签代表了喜欢常见视频类型(例如 动漫 和 摔跤 )的观众群体。

## 5.2 Baseline Methods

To validate the performance of our approach we compare it against a number of baselines: 

- SpectralClustering [41]: This method generates a representation in R d from the d-smallest eigenvectors of Le, the normalized graph Laplacian of G. Utilizing the eigenvectors of Le implicitly assumes that graph cuts will be useful for classification.
- Modularity [39]: This method generates a representation in R d from the top-d eigenvectors of B, the Modularity matrix of G. The eigenvectors of B encode information about modular graph partitions of G [35]. Using them as features assumes that modular graph partitions will be useful for classification.
- EdgeCluster [40]: This method uses k-means clustering to cluster the adjacency matrix of G. Its has been shown to perform comparably to the Modularity method, with the added advantage of scaling to graphs which are too large for spectral decomposition.
- wvRN [25]: The weighted-vote Relational Neighbor is a relational classifier. Given the neighborhood Ni of vertex vi, wvRN estimates Pr(yi|Ni) with the (appropriately normalized) weighted mean of its neighbors (i.e Pr(yi|Ni) = 1 Z P vj∈Ni wij Pr(yj | Nj )). It has shown surprisingly good performance in real networks, and has been advocated as a sensible relational classification baseline [26].
- Majority: This naive method simply chooses the most frequent labels in the training set.

> 