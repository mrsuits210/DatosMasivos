
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import spark.implicits._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.IndexToString
import org.apache.spark.sql.Dataset._

val Spark= SparkSession.builder().getOrCreate()

val data = spark.read.format("csv")
.option("sep",";")
.option("inferSchema","true")
.option("header","true")
.csv("iris.csv")

// identify the feature colunms
val inputColumns = ("SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm")
val VectorAssembler assembler = new VectorAssembler().setInputCols(inputColumns).setOutputCol("features")
val featureSet = assembler.transform(data);

// split data random in trainingset (70%) and testset (30%)


val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

val layers = Array[Int](3, 5, 4, 3)



val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

  val model = trainer.fit(train)

//  Creamos nuestro modelo de computacion para determinar nuestro valor de predccion mas acertado 
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")



