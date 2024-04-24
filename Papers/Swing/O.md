# Large Scale Product Graph Construction for Recommendation in E-commerce

## ABSTRACT

Building a recommendation system that serves billions of users on daily basis is a challenging problem, as the system needs to make astronomical number of predictions per second based on real-time user behaviors with O(1) time complexity. Such kind of large scale recommendation systems usually rely heavily on pre-built index of products to speedup the recommendation service so that the waiting time of online user is un-noticeable. One important indexing structure is the product-product index, where one can retrieval a list of ranked products given a seed product. The index can be viewed as a weighted product-product graph.

> 构建一个每天为数十亿用户提供服务的推荐系统是一个具有挑战性的问题，因为该系统需要以O(1)时间复杂度每秒进行天文数字的实时用户行为预测。这类大规模的推荐系统通常依赖于预先构建的 product 索引，以加速推荐服务，以确保在线用户的等待时间几乎不可察觉。其中一个重要的索引结构是 product-product 索引，它允许根据一个种子product 来检索一组排名靠前的产品。这个索引可以被视为一个带权重的product-product图。

In this paper, we present our novel technologies to efficiently build such kind of indexed product graphs. In particular, we propose the Swing algorithm to capture the substitute relationships between products, which can utilize the substructures of user-item click bi-partitive graph. Then we propose the Surprise algorithm for the modeling of complementary product relationships, which utilizes product category information and solves the sparsity problem of user co-purchasing graph via clustering technique. Base on these two approaches, we can build the basis product graph for recommendation in Taobao. The approaches are evaluated comprehensively with both offline and online experiments, and the results demonstrate the effectiveness and efficiency of the work.

> 在本篇论文中，我们展示了用于高效构建此类产品索引图的新技术。特别是，我们提出了Swing算法来捕捉产品之间的替代关系，该算法可以利用用户-项目点击二分图的子结构。然后，我们提出了Surprise算法，用于建立互补产品关系模型，该算法利用产品类别信息并通过聚类技术解决用户共同购买图的稀疏问题。基于这两种方法，我们可以构建淘宝推荐的基本产品图。本篇论文全面评估了这两种方法的有效性和效率，通过离线实验和在线实验的结果表明了其有效性。