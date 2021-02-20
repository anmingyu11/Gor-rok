今天会解读NE（Network Embedding）领域的一篇经典论文《node2vec: Scalable Feature Learning for Networks》。

Network Embedding算法主要有经历了以下三代的发展：

- 第一代：基于矩阵特征向量（谱聚类）
- 第二代：基于Random Walk（Deep Walk & Node2Vec）
- 第三代：基于Deep Learning（SDNE， GCN， GraphSAGE）

Node2Vec是第二代NE算法中比较经典的一种，本篇解读主要聚焦于算法的实现上，算法的深入理解敬请期待专栏的下一篇文章。

## **一、算法思想**

文章主要解决的问题是如何将图的一些拓扑特征进行学习（Representation Learning），那么自然就想到了将Node映射到低维特征空间作向量表示（Embedding）。这些向量可以作为其他任务的输入特征，即将无监督的表征学习任务单独抽出来。

Embedding的思路和DeepWalk是一个套路，即在图中进行游走采样得到多个节点序列，将节点看成word，使用NLP中的word2vec算法对节点进行embedding训练。

## **二、算法实现**

本文结合代码解析该算法的实现过程，代码来源于[aditya-grover/node2vec](https://link.zhihu.com/?target=https%3A//github.com/aditya-grover/node2vec)中的Spark实现，在仓库代码的基础上修复了一些bug（有向图代码报错）。

### 1. **定义Node和Edge属性类**

#### 1.1 定义Node的属性类NodeAttr

```scala
case class NodeAttr(var neighbors: Array[(Long, Double)] = Array.empty[(Long, Double)], 
                    var path: Array[Long] = Array.empty[Long]) extends Serializable {
  def neighborNumbers: Int = neighbors.length

  def setupAlias: (Array[Int], Array[Double]) = {
    val K = neighborNumbers
    val J = Array.fill(K)(0)
    val q = Array.fill(K)(0.0)

    val smaller = new ArrayBuffer[Int]()
    val larger = new ArrayBuffer[Int]()

    val sum = neighbors.map(_._2).sum
    neighbors.zipWithIndex.foreach { case ((nodeId, weight), i) =>
      q(i) = K * weight / sum
      if (q(i) < 1.0) {
        smaller.append(i)
      } else {
        larger.append(i)
      }
    }

    while (smaller.nonEmpty && larger.nonEmpty) {
      val small = smaller.remove(smaller.length - 1)
      val large = larger.remove(larger.length - 1)
      J(small) = large
      q(large) = q(large) + q(small) - 1.0
      if (q(large) < 1.0) smaller.append(large)
      else larger.append(large)
    }

    (J, q)
  }

}
```

NodeAttr包含两个属性：

1）`neighbors`：节点的邻居集合Array，每个邻居是一个二元组`（nodeId, weight）`；

2）`path`：采样节点序列id。

`setupAlias`是生成`Alias`表的方法，返回(`Alias`数组，`Prob`数组)。

在这里，Node2vec采用了别名采样法（Alias Method for Sampling）进行采样。

> 别名采样法：一种高效处理离散分布随机变量的取样问题
>
> Example:

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/Node2Vec/经典NetworkEmbedding算法Node2Vec/1.png)

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/Node2Vec/经典NetworkEmbedding算法Node2Vec/2.png)

> 算法复杂度：构造Alias Table复杂度O(n)，采样复杂度为O(1)
> 参考资料：
> [Darts, Dice, and Coins](https://link.zhihu.com/?target=http%3A//www.keithschwarz.com/darts-dice-coins/)
> [https://blog.csdn.net/haolexiao/article/details/65157026](https://link.zhihu.com/?target=https%3A//blog.csdn.net/haolexiao/article/details/65157026)

图中节点对其邻居节点进行概率采样，$PDF=weight/sum(weight)$，采用`setupAlias`方法生成Alias数组$J$和Prob数组$q$、`drawAlias`方法进行概率采样：

```scala
def drawAlias(J: Array[Int], q: Array[Double]): Int = {
    val K = J.length
    val kk = math.floor(math.random * K).toInt
    if (math.random < q(kk)) kk
    else J(kk)
}
```

### **1.2 定义Edge的属性类**

```scala
case class EdgeAttr(var dstNeighbors: Array[Long] = Array.empty[Long], var J: Array[Int] = Array.empty[Int],
 var q: Array[Double] = Array.empty[Double]) extends Serializable
```

在`EdgeAttr`中，`dstNeighbors`是`dst`节点的邻居，$J$和$q$是dst节点作概率采样的Alias Table。

## 2. **定义算法类**

```scala
class Node2Vec (
  var iter: Int = 10,                  // Iteration times to train w2v
  var lr: Double = 0.025,              // Learning rate
  var dim: Int = 128,                  // Embedding dimension
  var window: Int = 10,                // Context size
  var walkLength: Int = 80,            // Random work length
  var numWalks: Int = 10,              // Walks per node
  var p: Double = 1.0,                 // Return parameter
  var q: Double = 1.0,                 // In-out parameter
  var degree: Int = 30,                // Maximum degree
  var isBidirectional: Boolean = true, // Whether bi-direction graph
  var partitionNum: Int = 1000         // Partition number
) extends Serializable{

  var n2vGraph: Graph[NodeAttr, EdgeAttr] = _
  var randomWalkPaths: RDD[(Long, ArrayBuffer[Long])] = _     //Initialize walks to Empty
  var sc: SparkContext = _
}
```

算法的超参数、Node2Vec图属性信息以及随机游走路径的初始化。

## **3.原始数据Transform**（convertToNode2VecGraph）

原始数据输入的格式是`Graph[VD, Double]`，通过这一步将其变换成node2vec算法对应的图存储格式`Graph[NodeAttr, EdgeAttr]`。

```scala
def convertToNode2VecGraph[VD: ClassTag](graph: Graph[VD, Double]): this.type = {
  val bcMaxDegree = sc.broadcast(degree)
  val bcIsBidirectional = sc.broadcast(isBidirectional)
  val indexedNodes = graph.triplets.flatMap { triplet  =>
    if(bcIsBidirectional.value){
      Array(
        (triplet.srcId, Array((triplet.dstId, triplet.attr))),
        (triplet.dstId, Array((triplet.srcId, triplet.attr)))
      )
    } else {
      Array((triplet.srcId, Array((triplet.dstId, triplet.attr))))
    }
  }.reduceByKey(_ ++ _).map { case (nodeId, neighbors: Array[(VertexId, Double)]) =>
    var nearestNeighbors = neighbors
    if (neighbors.length > bcMaxDegree.value) {
      nearestNeighbors = neighbors.sortWith {case (left, right) => left._2 > right._2}.slice(0, bcMaxDegree.value)
    }
    (nodeId, NodeAttr(neighbors = nearestNeighbors.distinct))
  }.repartition(partitionNum).cache

  val indexedEdges = indexedNodes.flatMap { case (srcId, clickNode) =>
    clickNode.neighbors.map { case (dstId, weight) =>
      Edge(srcId, dstId, EdgeAttr())
    }
  }.repartition(partitionNum).cache

  n2vGraph = Graph(indexedNodes, indexedEdges)
  this
}
```

1. 生成`Node`的属性，将`maxDegree`个权重最大的邻居存入`NodeAttr.neighbors`；
2. 仅保留经过邻居筛选后的边，初始化边的`EdgeAttr`。

## 4. **初始化转移概率**（initTransitionProb）

```scala
def initTransitionProb(): this.type = {
  val bcP = sc.broadcast(p)
  val bcQ = sc.broadcast(q)

  n2vGraph = n2vGraph.mapVertices{case (vertexId, clickNode) =>
    if (clickNode != null) {
      val (j, q) = clickNode.setupAlias
      val nextNodeIndex = Node2Vec.drawAlias(j, q)
      clickNode.path = Array(vertexId, clickNode.neighbors(nextNodeIndex)._1)
      clickNode
    } else {
      NodeAttr()
    }
  }.mapTriplets { edgeTriplet: EdgeTriplet[NodeAttr, EdgeAttr] =>
    val (j, q) = Node2Vec.setupEdgeAlias(bcP.value, bcQ.value)(edgeTriplet)
    edgeTriplet.attr.J = j
    edgeTriplet.attr.q = q
    edgeTriplet.attr.dstNeighbors = edgeTriplet.dstAttr.neighbors.map(_._1)

    edgeTriplet.attr
  }.cache

  this
}
```

初始化阶段有下面两个步骤：

1. 每个Node，`path`属性是本身的节点ID和对邻居进行概率采样一次的邻居节点ID
2. 每条Edge，存储dst节点的邻居以及Alias Table

计算转移概率$P(next=x|dst=v,src=t)$：

![](/Users/helloword/Anmingyu/Gor-rok/Daily/GraphEmbedding/Node2Vec/经典NetworkEmbedding算法Node2Vec/3.png)

```scala
def setupEdgeAlias(p: Double = 1.0, q: Double = 1.0)(edgeTriplet: EdgeTriplet[NodeAttr, EdgeAttr]): (Array[Int], Array[Double]) = {
  val srcId = edgeTriplet.srcId
  val srcNeighbors = edgeTriplet.srcAttr.neighbors
  val dstNeighbors = edgeTriplet.dstAttr.neighbors
  val neighbors = dstNeighbors.map { case (dstNeighborId, weight) =>
    var unnormProb = weight / q
    if (srcId == dstNeighborId) unnormProb = weight / p
    else if (srcNeighbors.exists(_._1 == dstNeighborId)) unnormProb = weight
    (dstNeighborId, unnormProb)
  }
  NodeAttr(neighbors).setupAlias
}
```

该方法返回Edge中dst节点下一步游走的Alias Table，这也是Node2Vec相对于RandomWalk改进的地方。

## **5. 随机游走(randomWalk)**

```scala
def randomWalk(): this.type = {
  val edge2attr = n2vGraph.triplets.map { edgeTriplet =>
    (s"${edgeTriplet.srcId}\t${edgeTriplet.dstId}", edgeTriplet.attr)
  }.repartition(partitionNum).cache
  edge2attr.first

  for (i <- 0 until numWalks) {
    var prevWalk: RDD[(Long, ArrayBuffer[Long])] = null
    var randomWalk = n2vGraph.vertices.map { case (nodeId, clickNode) =>
      val pathBuffer = new ArrayBuffer[Long]()
      pathBuffer.append(clickNode.path : _*)
      (nodeId, pathBuffer)
    }
    .filter{case (_, pathBuffer) => pathBuffer.nonEmpty}.cache
    var activeWalks = randomWalk.first
    n2vGraph.unpersist(blocking = false)
    n2vGraph.edges.unpersist(blocking = false)

    // node2vecWalk(G0; u; l)
    for (walkCount <- 0 until walkLength) {
      prevWalk = randomWalk
      randomWalk = randomWalk.map { case (srcNodeId, pathBuffer) =>
        val prevNodeId = pathBuffer(pathBuffer.length - 2)
        val currentNodeId = pathBuffer.last
        (s"$prevNodeId\t$currentNodeId", (srcNodeId, pathBuffer))
      }.join(edge2attr).map { case (edge, ((srcNodeId, pathBuffer), attr)) =>
        try {
          if (pathBuffer != null && pathBuffer.nonEmpty && attr.dstNeighbors != null && attr.dstNeighbors.nonEmpty){
            val nextNodeIndex = Node2Vec.drawAlias(attr.J, attr.q)
            val nextNodeId = attr.dstNeighbors(nextNodeIndex)
            pathBuffer.append(nextNodeId)
          }
          (srcNodeId, pathBuffer)
        } catch {
          case e: Exception => throw new RuntimeException(e.getMessage)
        }
      }.cache
      activeWalks = randomWalk.first()
      prevWalk.unpersist(blocking = false)
    }

    //Append walk to walks
    if (randomWalkPaths != null) {
      val prevRandomWalkPaths = randomWalkPaths
      randomWalkPaths = randomWalkPaths.union(randomWalk).cache()
      randomWalkPaths.first
      prevRandomWalkPaths.unpersist(blocking = false)
    } else {
      randomWalkPaths = randomWalk
    }
    randomWalk.unpersist(blocking = false)
  }

  this
}
```

## **6. Embedding**

```scala
def embedding(): Node2VecModel  = {
  val randomPaths = randomWalkPaths.map { case (vertexId, pathBuffer) =>
    Try(pathBuffer.map(_.toString)).getOrElse(null)
  }.filter(_ != null)
  val w2vModel = new Word2Vec()
  .setLearningRate(lr)
  .setNumIterations(iter)
  .setMinCount(0)
  .setVectorSize(dim)
  .setWindowSize(window)
  .fit(randomPaths)
  new Node2VecModel(sc, randomWalkPaths, w2vModel)
}
```

这里定义了一个Node2Vec的模型类，按照word2vec算法进行训练。

```scala
class Node2VecModel(val sc: SparkContext,
                    val randomWalkPaths: RDD[(Long, ArrayBuffer[Long])],
                    val w2vModel: Word2VecModel) extends Serializable {

  def getVectors: Map[String, Array[Float]] = w2vModel.getVectors

  def saveModel(path: String): Unit = {
    w2vModel.save(sc, path)
  }

}
```

## **7. 训练接口**

```scala
def fit[VD: ClassTag](sc: SparkContext, graph: Graph[VD, Double]): Node2VecModel = {
  setupContext(sc)
  convertToNode2VecGraph(graph)
  initTransitionProb()
  randomWalk()
  embedding()
}
```

提供一个训练接口方便外部调用。

## 三、算法讨论

Node2vec算法以现在的算法观来看从思想上还是比较简单的，但是背后所包含的问题一点也不少，我列举几个供大家思考和讨论：

1. 作者说BFS探索的是结构性，DFS探索的是同质性，这个怎么理解？你自己又是怎么理解的
2. 如何评估Node2Vec作表征学习的效果？
3. Node2Vec可以作增量训练吗？什么样的情况下适合增量训练？
4. 图中节点非常多（千万甚至亿级），word2vec出现计算瓶颈时工程上应该如何解决？

