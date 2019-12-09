import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.log4j._
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler

Logger.getLogger("org").setLevel(Level.ERROR)

//Importamos nuestro DataSet
val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

data.printSchema()
data.show(1)
//Creamos nuestro laber indexer para comprarar 
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)

//Inicializamos el vector asembler por datos de tipo numericos y agregamos la columna features como output 
val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
val features = assembler.transform(data)



// Identifica categoricamente nuestro dataset en vector 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

val Array(trainingData, testData) = features.randomSplit(Array(0.7, 0.3))
//Creamos un objeto DecisionTree
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
//Rama de prediccion
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
//Juntamos los datos en un pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
//Create a model of the entraining
val model = pipeline.fit(trainingData)
//Transformacion de datos en el modelo
val predictions = model.transform(testData)
//Desplegamos predicciones
predictions.select("predictedLabel", "y", "features").show(5)
//Evaluamos la exactitud
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

val t1 = System.nanoTime
val duration = (System.nanoTime - t1) / 1e9d

