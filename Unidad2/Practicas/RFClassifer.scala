//Importamos nuestra liberias de ml 
// Pipeline para compatar los datos 
// Clasificacion con random forest 
// y nuestro feature para indexar nuestra data y poder trabajar con los arboles. 

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Cargamos nuestro dataset desde la mlib 
val data = spark.read.format("libsvm").load("Data/sample_libsvm_data.txt")

// Creamos indices de las etiquetas 
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)

  //Identifica categoricamente las caracteristicas y las indexa 
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4)
  .fit(data)

// dividimos los datos en conjunto de datos de entrenamiento y de testing 
// 70% para entramiento 30% para testing 
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Entrenamos nuestro modelo de randomForest 
val rf = new RandomForestClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")
  .setNumTrees(10)

// Convierte las etiquetas indexadas a etiquetas originales
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// une los index con nuestro modelo randomforest mediante pipeline
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Entrenamos nuestro modelo pipeline con los datos de entrenamiento
val model = pipeline.fit(trainingData)

// Creamos las predicciones mediante la testdata 
val predictions = model.transform(testData)

// Seleccionamos columnas de ejemplo para mostrar nuestras predicciones. 
predictions.select("predictedLabel", "label", "features").show(5)

// Selecciona la predicciones verdaderas y calcula el error de prueba 
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
  //Imprimimos el valor de la prediccion y su margen de error. 
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")