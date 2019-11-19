//Importamos la liberia pipeline para juntar los datos
// y las librerias deceisiontree clasification model para realizar nuestros arboles 
// asi como para clasificarlo y evaluarlo. 

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Carmaos nuestro data set predefinido por spark
val data = spark.read.format("libsvm").load("Data/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.

//Iniciamos nuestra etiqueta y nuestro feature
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("indexedLabel")
  .fit(data)
// Identifica categoricamente nuestro dataset en vector 
val featureIndexer = new VectorIndexer()
  .setInputCol("features")
  .setOutputCol("indexedFeatures")
  .setMaxCategories(4) 
  .fit(data)

// Creamos nuestro arrglo con nuestros datos de entrenamiento y de testing 
//70% para entrenamiento y 30%datos para testing 
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Entrenamos nuestro Arbol medianto dt 
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("indexedFeatures")

// Convertimos nuestro arbol indexado
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// une nuestros datos indexados a un pipeline es decir compacta todos los datos .
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Entrenamos nuestra data con los datos de entramiento
val model = pipeline.fit(trainingData)

// Creamos las predicciones 
val predictions = model.transform(testData)

//Seleccionamos algunos predicciones para mostrar en pantalla
predictions.select("predictedLabel", "label", "features").show(5)

//Seleccionamos las prediccion y los valores verdaderos a si como los datos de error despues del modelado 
 val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
//Imprimimos nuestra prediccion junto con su porcentaje de aceptacion o predicccion. 
println(s"Test Error = ${(1.0 - accuracy)}")

//Modelamos nuestro arbol y lo imprimimos 
val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

