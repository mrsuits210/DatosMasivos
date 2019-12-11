import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.log4j._


Logger.getLogger("org").setLevel(Level.ERROR)

//Sesion Spark
val spark = SparkSession.builder().getOrCreate()
//Cargamos nuestro Dataset 
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)

val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
val features = assembler.transform(data)


// Indetificamos nuestros valores categoricos y los indexamos
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(features)

// Dividimos la data en datos de entrenamiento y datos de prueba 70% para entramiento y 30% para prueba
val Array(trainingData, testData) = features.randomSplit(Array(0.7, 0.3))

// Entrenamos nuestro modelo de RanfomForest
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

//Convertimos labels indexados a los labels originales 
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// unimos nuestros labels indexados con nuestro modelo RF y lo incluimos en un pipeline
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Entrenamos nuestro modelo 
val model = pipeline.fit(trainingData)

// Hacemos las predicciones
val predictions = model.transform(testData)

// Seleccionamos ejemplo de columnas para mostrar 
predictions.select("predictedLabel", "y", "features").show(5)

// Seleccionamos (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")


// Imprimir el tiempo de ejecucion 
val t1 = System.nanoTime

val duration = (System.nanoTime - t1) / 1e9d