//Hacemos las importaciones de nuestras liberias a utilizar de ml 
// pipeline para tratar los datos y unirlos 
// GBT para la calsificacion y el modelaje 
// Feature para indexar nuestra data 

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Cargamos nuestro dataset desde mlib 
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Indexamos nuestras etiquetas para trabajarlas
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

//Indetifica automaticamente las categorias y las indexa 
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// Dividimos nuestro data set en datos de entrenamiento y datos de prueba o testing 
// 70% para entrenamiento y %30 para prueba. 
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Entrenamos nuestro modelo GBT. 
val gbt = new GBTClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setMaxIter(10)
  .setFeatureSubsetStrategy("auto")

// Convertimos nuestras etiquetas indexadas en etiquetas originales 
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// unimos nuestro modelo gbt nuestros etiquetas indexadas mediante Pipeline 
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Entrenamos nuestro pipeline mediante los datos de entrenamiento y la funcion fit 
val model = pipeline.fit(trainingData)

// Creamos nuestras preoducciones transformando nuestra data 
val predictions = model.transform(testData)

// seleccionamos algunas columnas de prueba y las mostramos.
predictions.select("predictedLabel", "label", "features").show(5)

// Seleccionamos nuestras predicciones y etiquetas verdaderas asi como el margen de error obtenido. 
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")

  //Evaluaciones y optemos el procentaje de las prediccion 
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")