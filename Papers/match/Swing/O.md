# Large Scale Product Graph Construction for Recommendation in E-commerce

## ABSTRACT

Building a recommendation system that serves billions of users on daily basis is a challenging problem, as the system needs to make astronomical number of predictions per second based on real-time user behaviors with O(1) time complexity. Such kind of large scale recommendation systems usually rely heavily on pre-built index of products to speedup the recommendation service so that the waiting time of online user is un-noticeable. One important indexing structure is the product-product index, where one can retrieval a list of ranked products given a seed product. The index can be viewed as a weighted product-product graph.

In this paper, we present our novel technologies to efficiently build such kind of indexed product graphs. In particular, we propose the Swing algorithm to capture the substitute relationships between products, which can utilize the substructures of user-item click bi-partitive graph. Then we propose the Surprise algorithm for the modeling of complementary product relationships, which utilizes product category information and solves the sparsity problem of user co-purchasing graph via clustering technique. Base on these two approaches, we can build the basis product graph for recommendation in Taobao. The approaches are evaluated comprehensively with both offline and online experiments, and the results demonstrate the effectiveness and efficiency of the work.

> 构建一个每天为数十亿用户服务的推荐系统是一个具有挑战性的问题，因为该系统需要基于实时用户行为，在O(1)的时间复杂度内每秒做出海量的预测。这种大规模推荐系统通常严重依赖于预先构建的产品索引来加速推荐服务，以使在线用户的等待时间变得微乎其微。一个重要的索引结构是产品-产品索引，其中可以根据种子产品检索排名产品的列表。该索引可以看作是一个加权的产品-产品图。
>
> 在本文中，我们展示了高效构建此类索引产品图的新技术。特别是，我们提出了Swing算法来捕捉产品之间的替代关系，该算法可以利用用户-项目点击二分图的子结构。然后我们提出了Surprise算法来建模互补产品关系，该算法利用产品信息类别，并通过聚类技术解决了用户共同购买图的稀疏性问题。基于这两种方法，我们可以为淘宝的推荐构建基础产品图。这些方法通过离线和在线实验进行了综合评估，结果证明了工作的有效性和效率。