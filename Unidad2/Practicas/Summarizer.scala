
//importamos la libreria ml para vectores 
// importamos summarizer para utilizar nuestro algoritmo. 
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.stat.Summarizer

//Creamos nuestro data set con 2 vectores y su densidad
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0)
)
//Creamos nuestro data frame con features y weight 
val df = data.toDF("features", "weight")
// Creamos una variable la cual obtendria mediante sumarizer la varianza de nuestro vector
val (meanVal, varianceVal) = df.select(metrics("mean", "variance")
  .summary($"features", $"weight").as("summary"))
  .select("summary.mean", "summary.variance")
  .as[(Vector, Vector)].first()
// Imprimimos los valores resultantes 
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features"))
  .as[(Vector, Vector)].first()

println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
