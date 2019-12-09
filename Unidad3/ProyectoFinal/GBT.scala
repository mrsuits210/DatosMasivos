import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.log4j._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DateType
import org.apache.spark.sql.{SparkSession, SQLContext}
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler


Logger.getLogger("org").setLevel(Level.ERROR)
val spark = SparkSession.builder().getOrCreate()
// Cargamos nuestro dataset 

val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

//Inicializamos nuestro StringIndexer 
val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)

val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
val features = assembler.transform(data)



// Indetificamos nuestros valores categoricos y los indexamos
// 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

// Dividimos nuestra data en datos de entramiento y datos de prueba 70% para entrenamiento y 30% para prueba
val Array(trainingData, testData) = features.randomSplit(Array(0.7, 0.3))

// Entrenamos nuestro modelo GBT 
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// Convertimos nuestro labels indexados en labels originales 
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// Chain indexers and GBT in a Pipeline.
// Unimos nuestros index con nuestro modelo GBT y los compactamos en un pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Entrenamos nuestro modelo con datos indexados 
val model = pipeline.fit(trainingData)

// Hacemos las predicciones
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "y", "features").show(5)

// Seleccionamos (prediction, true label) y el error de computo 
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")

val t1 = System.nanoTime

val duration = (System.nanoTime - t1) / 1e9d