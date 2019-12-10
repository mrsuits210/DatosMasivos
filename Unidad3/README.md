**K-Mens**

K-Means clustering is one of the simplest and popular unsupervised machine learning algorithms. The goal of this algorithm is to find groups in the data, with the number of groups/clusters represented by the variable K. K-Means algorithm iteratively allocates every data point to the nearest cluster based on the features. In every iteration of the algorithm, each data point is assigned to its nearest cluster based on some distance metric, which is usually Euclidean distance. The outputs of the K-means clustering algorithm are the centroids of K clusters and the labels of training data. Once the algorithm runs and identified the groups from a data set, any new data can be easily assigned to a group.

K-Means algorithm can be used to identifies unknown groups in complex and unlabeled data sets. Following are some business use cases of K-Means clustering.

- Customer segmentation based on purchase history
- Customer segmentation based on interest
- Insurance fraud detection
- Transaction fraud detection
- Detect unauthorized IoT devices based on network traffic
- Identity crime locality
- Group inventory by sales
[Code](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/K-means_convergence.gif/512px-K-means_convergence.gif)
```
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

import org.apache.spark.ml.clustering.KMeans

//Cargamos nuestro dataset
val data = spark.read.format("csv").option("inferSchema","true").option("header","true").csv("Wholesale_customers_data.csv")
//hacemos un printschema para ver la estructura de la data
data.printSchema()

val feature_data = data.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")

// Generamos un vector donde se almacenara las caracteristicas del dataset a evaluar
// y se mG mediante la columna features   
val assembler = new VectorAssembler().setInputCols(Array("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")).setOutputCol("features")
// Transformamos los datos utlizando nuestro dataset
val featureSet = assembler.transform(feature_data)


// Trains a k-means model.
val kmeans = new KMeans().setK(3).setSeed(1L)
val model = kmeans.fit(featureSet)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(featureSet)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

//Transformamos los valores categoricos a datos numeros para poder procesarlo

//val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
//val dataindex = labelIndexer.fit(featureSet).transform(featureSet)
```
