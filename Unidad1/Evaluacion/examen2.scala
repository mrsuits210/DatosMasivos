import org.apache.spark.sql.SparkSession //
 
val spark = SparkSession.builder().getOrCreate() // Pregunta 1

val df = spark.read.option("header", "true").option("inferSchema","true")csv("Netflix_2011_2016.csv") // Pregunta 2 


df.columns // Pregunta 3 

df.printSchema()//Pregunta 4 

df.show(5)  // Pregunta 5 

df.describe().show() // Pregunta 6 

val df2 = df.withColumn("HV Ratio", df("High")+df("Volume")) //Pregunta 7 

df2.show()


//Pregunta  9
// Es como el precio Final despues de los ajustes antes del cierre del mercado. 



df.select(min("Volume"),max("Volume")).show(); // Pregunta 10 


  // Pregunta 11 (A) 


df.filter($"Close"<600).count()


// Pregunta 11(B)

(df.filter($"High" > 500).count() * 1.0/ df.count())*100



df.select(corr($"High", $"Volume")).show() // Pregunta 11 (C)


//Pregunta 11 (d)

val df2 = df.withColumn("Year", year(df("Date")))
val dfmins = df2.groupBy("Year").max()
dfmins.select($"Year", $"max(High)").show()



// Pregunta 11 (e) 

val df3 = df.withColumn("Month", month(df("Date")))
val dfmes = df3.groupBy("Month").mean() 
dfmes.select($"Month", $"avg(Close)").show(); 
