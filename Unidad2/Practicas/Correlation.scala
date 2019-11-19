//Se importan de la liberia Ml de psark matrix y vector para haer la funcion de vector
import org.apache.spark.ml.linalg.{Matrix, Vectors}
//se importa la liberia ml correlation para hacer el metodo de correlacion. 
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.Row
//Creamos nuestro data set mandando datos atravez de vectores 
val data = Seq(
  Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
  Vectors.dense(4.0, 5.0, 0.0, 3.0),
  Vectors.dense(6.0, 7.0, 0.0, 8.0),
  Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)
//creamos un data frame aplicando el metodo matrix junto con correlacion 
val df = data.map(Tuple1.apply).toDF("features")

// Imprimimos los resultados. 
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
println(s"Pearson correlation matrix:\n $coeff1")

val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
println(s"Spearman correlation matrix:\n $coeff2")


