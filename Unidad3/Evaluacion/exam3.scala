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