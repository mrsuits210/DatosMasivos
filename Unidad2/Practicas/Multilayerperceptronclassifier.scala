//Importamos las liberiras de ml para utilizar multuplayer perceptron con su clasifcador y evaluador. 

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Obtenemos nuestro dataset desde mlib 
val data = spark.read.format("libsvm")
  .load("Data/sample_multiclass_classification_data.txt")

// Dividimos nuestro data set entre datos de entramiento y datos de testing o prueba. 
// 60% Entrenamiento y 40% datos de prueba. 
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// Especificamos nuestras capas de redes neuronales 
// 3 capas de entrada (features), dos intermedias  tamaño 5 y 4
// y una salida de  3 (classes)
val layers = Array[Int](4, 5, 4, 3)

//creamos nuestro entrenador y seteamos los parametros
val trainer = new MultilayerPerceptronClassifier()
  .setLayers(layers)
  .setBlockSize(128)
  .setSeed(1234L)
  .setMaxIter(100)

// Entrenamos nuestro modelo. 
val model = trainer.fit(train)

//  Creamos nuestro modelo de computacion para determinar nuestro valor de predccion mas acertado 
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
