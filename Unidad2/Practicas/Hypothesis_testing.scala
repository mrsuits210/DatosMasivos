
// Importamos nuestra liberia para trabajar con Vectores
import org.apache.spark.ml.linalg.{Vector, Vectors}
// Importamos ChiSquareTest para utilizarlo en nuestra prueba estadistica 
import org.apache.spark.ml.stat.ChiSquareTest
//Inicializamos un dataset en vectores para poder trabajar con nuestro analisis estadistico. 
val data = Seq(
  (0.0, Vectors.dense(0.5, 10.0)),
  (0.0, Vectors.dense(1.5, 20.0)),
  (1.0, Vectors.dense(1.5, 30.0)),
  (0.0, Vectors.dense(3.5, 30.0)),
  (0.0, Vectors.dense(3.5, 40.0)),
  (1.0, Vectors.dense(3.5, 40.0))
)

//Creamos nuestro data frames con la data label y features que es como se vera nuestro data set ya tratado 
val df = data.toDF("label", "features")
//Aplicamos la funcion ChiSquareTest para compar que tan acertada es nuestra hipotesis 
//Aplicando el metodo estadistico para definir a nuestro DF 
val chi = ChiSquareTest.test(df, "features", "label").head

//Imprimimos los valores 
println(s"pValues = ${chi.getAs[Vector](0)}")
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
println(s"statistics ${chi.getAs[Vector](2)}")