
```
//  In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
//  This data set contains the following features:
//    'Daily Time Spent on Site': consumer time on site in minutes
//    'Age': cutomer age in years
//    'Area Income': Avg. Income of geographical area of consumer
//    'Daily Internet Usage': Avg. minutes a day consumer is on the internet
//    'Ad Topic Line': Headline of the advertisement
//    'City': City of consumer
//    'Male': Whether or not consumer was male
//    'Country': Country of consumer
//    'Timestamp': Time at which consumer clicked on Ad or closed window
//    'Clicked on Ad': 0 or 1 indicated clicking on Ad
```

```
// Se importan las librerias a utilizar

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.types.DateType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.log4j._


Logger.getLogger("org").setLevel(Level.ERROR)

//inicializamos nuestra sesion de spark
val spark = SparkSession.builder().getOrCreate()

// Cargamos nuestro dataset
val data  = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("advertising.csv")

// imprimmos nuestro esquema del dataset
data.printSchema()

//Imprimimos la primera linea del dataset
data.head(1)
data.select("Clicked on Ad").show()
val timedata = data.withColumn("Hour",hour(data("Timestamp")))

//Traemos los datos mas reelevantes del dataset
val logregdataall = timedata.select(data("Clicked on Ad").as("label"),$"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Hour",$"Male")
val feature_data = data.select($"Daily Time Spent on Site",$"Age",$"Area Income",$"Daily Internet Usage",$"Timestamp",$"Male")
val logregdataal = (data.withColumn("Hour",hour(data("Timestamp")))
val logregdataal = logregdataall.na.drop()

//Creamos nuestro vector assembler con los labels para trabajar y ponemos de salida features
val assembler = new VectorAssembler().setInputCols(Array("Daily Time Spent on Site","Age","Area Income","Daily Internet Usage","Hour","Male")).setOutputCol("features")

// dividimos nuestro dataset en datos de entrenamiento y datos de prueba 70 % para entrenamiento y 30% para prueba
val Array(training, test) = logregdataall.randomSplit(Array(0.7, 0.3), seed = 12345)
val lr = new LogisticRegression()
val pipeline = new Pipeline().setStages(Array(assembler,lr))

//Creamos nuestro modelo
val model = pipeline.fit(training)

//transformamos nuestros valores de test y hacemos las prediccioness
val results = model.transform(test)
val predictionAndLabels = results.select($"prediction",$"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

//se imprimen las metricas y la exactitud de la prediccion
println("Confusion matrix:")
println(metrics.confusionMatrix)
metrics.accuracy
```
