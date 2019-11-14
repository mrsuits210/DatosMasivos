
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import spark.implicits._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer

val data = spark.read.format("csv").option("inferSchema","true").option("header","true").csv("iris.csv")

data.printSchema()
// identify the feature colunms
val assembler = new VectorAssembler().setInputCols(Array("sepal_length","sepal_width","petal_length","petal_width")).setOutputCol("features")
val featureSet = assembler.transform(data)


val labelIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
val dataindex = labelIndexer.fit(featureSet).transform(featureSet)




//Los valores que se indentifican son indexados y se añade maxCategories para que los valores mayores a 4 sean tratados como continuos
///val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(featureSet)

// split data random in trainingset (70%) and testset (30%)



val splits = dataindex.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

val layers = Array[Int](4, 5, 4, 3)



val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

val model = trainer.fit(train)

//  Creamos nuestro modelo de computacion para determinar nuestro valor de predccion mas acertado 
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


