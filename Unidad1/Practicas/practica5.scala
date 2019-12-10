import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Sales.csv")

df.printSchema()

df.show()


df.select(countDistinct("Sales")).show()

df.select(sumDistinct("Sales")).show()

df.select(variance("Sales")).show()

df.select(stddev("Sales")).show()

df.select(collect_set("Sales")).show()


df.select(last("Sales")).show(); // Muestra el ultimo valor de la columna


df.select(variance("Sales")).show();  // Regresa la varianza del grupo de valores

df.select(corr("Sales","Person")).show(); // Devuelve un conjunto de objetos con elementos eliminados o duplicdos, d

df.select(skewness("Sales")).show();  // Devuelve el sesgo de  los valores de la columna,

df.select(stddev_pop("Sales")).show(); //Devuelve la desviacion estandar de la poblacion.

df.select(kurtosis("Sales")).show();  // Devuelve la curtosis de los valores

df.select(avg("Sales")).show();  // Devuelve el promedio de los valores del grupo.

df.select(collect_list("Sales")).show(); // Devuelve una lista de objetos con sus duplicados.

df.select(covar_pop("Sales","Person")).show();  // Devuelve la covarancia de la poblacion de  2 columnas

df.select(covar_samp("Sales","Person")).show();  // Devuelve la covarianza de muestra para dos columnas.

df.select(first("Sales")).show(); // Devuelve el primer valor de una columna en un grupo.

df.select(approx_count_distinct("Sales")).show(); // devuelve el número aproximado de elementos distintos en un grupo
