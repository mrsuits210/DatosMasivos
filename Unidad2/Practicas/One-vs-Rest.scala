// Importamos nuestra liberiras OnevsRest 
// y logisticregresion que esun metodo de clasifcacion binaria para combinarlo con one vs rest. 

import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Cargamos nuestra data desde mllib 
val inputData = spark.read.format("libsvm")
  .load("Data/sample_multiclass_classification_data.txt")

// Generamos nuestro para dividir entre datos de entrenamiento y datos de testing 
// 80% para entrenamiento , 20% para datos de testing. 
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// Instanciamos nuestro clasifcador binarios 
val classifier = new LogisticRegression()
  .setMaxIter(10)
  .setTol(1E-6)
  .setFitIntercept(true)

// Instanciamos onevsrest con nuestro calsificador binarios antes instanciado. 
val ovr = new OneVsRest().setClassifier(classifier)

// entrenamos nuestro sistema multiclase. .
val ovrModel = ovr.fit(train)

// Puntua el modelo de datos de las pruebas 
val predictions = ovrModel.transform(test)

// oobtenemos el evaluador 
val evaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")

  // obtenemos nuestro porcentaje de exito y de error de nuestro modelo de datos de prueba 
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")