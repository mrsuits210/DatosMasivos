
//Importamos la liberias de clasificacion NaiveBayes 
// importamos un evualudaro multiclase. 

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Cargammos nuestro dataset desde mllib 
val data = spark.read.format("libsvm").load("Data/sample_libsvm_data.txt")

//Dividimos nuestro dataset en datos de entrenamiento y datos de prueba. 
// 70% para datos de entrenamiento y 30% para datos de prueba
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

// Entrenamos nuestro modelo  NaiveBayes
val model = new NaiveBayes()
  .fit(trainingData)

// Seleccionamos un ejemplo para mostrar la data 
val predictions = model.transform(testData)
predictions.show()

// Seleccionamos la prediccion y las etiquetas verdaderas asi como el margen de error obtenido 
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")