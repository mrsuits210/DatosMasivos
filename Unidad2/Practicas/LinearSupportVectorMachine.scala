
//Importamos nuestra liberia de linear super vector machine 
import org.apache.spark.ml.classification.LinearSVC

// Cargamos nuestro dataset desde mllib 
val training = spark.read.format("libsvm").load("Data/sample_libsvm_data.txt")

//Seteamos nuestro modelo 
val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)

// Dotamos nuestro modelo con los datos de entrenamiento 
val lsvcModel = lsvc.fit(training)

//Imprime los coeficiones y los intercepta para realizar nuestro linear super vector. 
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
