# Introduction
It is increasingly common to see machine learning algorithms with the ability to predict, classify, recommend elements that help large companies position themselves and users to have better recommendations, based on information previously learned by the algorithm, an example of these would be, the recommendations of companies like netflix or spotify that through your tastes make you a selection of series or music which could be to your liking.
There are different ways to develop an algorithm focused on machine learning, one of them is the one explained in this work,
for this we will be using apache spark with the Scala language, thanks to its MLIb liberia we can train algorithms using large amounts of data.
In this work, three classification algorithms were included, which will be explained and finally compared to choose the best option when performing one task or another.

# theoretical framework
In general, Machine Learning algorithms can be classified into two types:

- Supervised: these algorithms are trained with a set of data whose result is already known. That is, the model is trained with data from which the input and output are already known.
- Unsupervised: the model is trained with data that the result is not yet known. That is, the input is known but not the output of the data.

Spark MLlib provides both supervised and unsupervised learning algorithms that offer solutions to the three most used techniques in the world of Machine Learning:


[Imagen1 ](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/1.png)


In this document we will be seeing only algorithms for classification using the spark mlib library, the algorithms that we will play in this work will be the following:
- Decision tree classifier
- Random forest classifier
- Gradient-boosted tree classifier

## Decision Tree Classifier

Decision trees are popular methods for classification and regression tasks of machine learning.
Decision trees are widely used because they are easy to interpret, handle categorical characteristics, extend to the multi-class classification configuration, do not require scale of characteristics and are capable of capturing nonlinearities and characteristic interactions.
A decision tree is a simple representation to classify examples. It is a supervised machine learning where data is continuously divided according to a certain parameter.

A decision tree consists of:
- Nodes: test the value of a certain attribute.
- Borders / branch: corresponds to the result of a test and connects to the next node or sheet.
- Sheet nodes: terminal nodes that predict the result (represent class labels or class distribution).

[Imagen2 ](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/2.png)

*To understand the concept of Decision Tree, consider the previous example. Let's say you want to predict whether a person is fit or not, given their information such as age, eating habits, physical activity, etc. Decision nodes are questions like 'What is the age?', 'Do you exercise?', 'Do you eat a lot of pizzas? And the sheets represent results as 'fit' or 'not fit'*

There are two main types of decision trees:
- Classification trees.
What we have seen before is an example of a classification tree, where the result was a variable like 'fit' or 'not fit'. Here the decision variable is categorical / discrete.
Such a tree is constructed through a process known as binary recursive partition. This is an iterative process of dividing data into partitions and then dividing them further into each of the branches.
[Imagen3](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/3.png)
- Regression trees.
Decision trees where the objective variable can take continuous values ​​(typically real numbers) are called regression trees. (for example, the price of a house or the length of a patient's stay in a hospital)
[Imagen4](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/4.png)

### Creating a decision tree
Decision trees are created using a heuristic called recursive partition. This approach is also commonly known as divide and conquer because it divides the data into subsets, which are then repeatedly divided into even smaller subsets, and so on until the process stops when the algorithm determines that the data within the subsets are sufficiently homogeneous. , or other detention criteria have been met.

Advantages of classification with decision trees:
- Cheap to build.
- Extremely fast in the classification of unknown records.
- Easy to interpret for small trees.
- Accuracy comparable to other classification techniques for many simple data sets.
- Excludes unimportant features.
Disadvantages of classification with decision trees:
- Overfitting
- The decision limit is limited to being parallel to the axes of attributes.
- Decision tree models are often skewed towards divisions in characteristics that have a large number of levels.
- Small changes in training data can lead to major changes in decision logic.
- Large trees can be difficult to interpret and the decisions they make may seem contradictory.

## Random Forest
'Random Forest' as the name implies is a forest and the forest consists of trees. Here the trees mentioned are decision trees. Then, the full definition will be "Random Forest is a random collection of decision trees". Therefore, this algorithm is basically an extension of the Decision Tree algorithm.

In this algorithm, we create multiple decision trees in their entirety. Yes, we don't need to prune our decision trees here. There is no such limitation for trees in Random Forest. The problem here is that we do not provide all the data for each decision tree to consume. We provide a random subset of our training data to each decision tree. This process is called Bagging or Bootstrap Aggregating.
[Imagen5](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/5.png)

Bagging is a general procedure that can be used to reduce the variance of those algorithms that have high variance. In this process, subsamples are created for the data set and a subset of attributes, which we use to train our decision models and then consider each model and choose the decision by voting - (classification) or taking the average (regression) For the forest randomly, we usually take two thirds of the data with replacement (the data can be repeated for any other decision tree, it does not need to be unique data). And the subset of the attributes m

In Random Forest each decision tree predicts a response for an instance. And the final answer is decided based on the vote. That means (in classification) that the response that most decision trees receive becomes the final answer. (In the regression, the average of all responses becomes the final answer).

[Imagen6](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/6.png)

## Gradient-boosted tree classifier
It is a machine learning technique used for regression analysis and for statistical classification problems, which produces a predictive model in the form of a set of weak prediction models, typically decision trees.
GBT builds trees one at a time, where each new tree helps correct the mistakes made by a previously trained tree.

[Imagen7](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/7.png)

### objective
It is to repeatedly take advantage of the patterns in the residuals and strengthen a model with weak predictions and improve it. Once we reach a stage where the residues do not have any pattern that can be modeled, we can stop modeling the residues (otherwise, it could cause overfitting). Algorithmically, we are minimizing our loss function, so that the loss of proof reaches its minimums.

Advantage
- Since enhanced trees are derived from the optimization of an objective function, basically GBM can be used to solve almost all objective functions that we can write in gradient.
- Performs the optimization in the function space (instead of in the parameter space), which greatly facilitates the use of custom loss functions.
- Predictive power too high.
Disadvantages
- GBMs are more sensitive to overfitting if the data is noisy.
- Training usually takes longer due to the fact that trees are built sequentially.
- GBMs are more difficult to tune than RF. There are generally three parameters: number of trees, tree depth and learning rate, and each tree built is generally shallow.


# Implementation
For the comparison of these three algorithms we use apache spark, with its extension to work with the scala language.
Scala is a modern multi-paradigm programming language designed to express common programming patterns in a concise, elegant, and secure typing way. Easily integrate features of object-oriented and functional languages.
Thanks to the implementation of this language we can program an algorithm in a simpler way with a performance that exceeds other languages, because we work through a base with spark.

[Imagen8](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/8.png)

# Results
The following dataset was used to make the comparison.
Bank Marketing Data Set
for all algorithms in order to have a more accurate prediction result. In the same way it was treated the same in all three algorithms.

For all algorithms the following instruction was used to load the dataset
```
val spark = SparkSession.builder().getOrCreate()
val data = spark.read.option("header","true").option("inferSchema","true").option("delimiter",";").format("csv").load("bank-full.csv")

val labelIndexer = new StringIndexer().setInputCol("y").setOutputCol("indexedLabel").fit(data)

val assembler = new VectorAssembler().setInputCols(Array("age","balance","day","duration","campaign","pdays","previous")).setOutputCol("features")
val features = assembler.transform(data)
```

After loading the data and transforming it, its due method was applied to classify each algorithm, both random forest and gradient boosted were applied to perform 10 iterations or 10 trees to make the comparison more effective.

We train our models with the previously loaded dataset.
- Decision tree
`val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")`

- Random Forest.

`val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)`

- GBT
`val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")`


The various unions between the labels and the pipelines are made to work with the indexed information and finally the predictions, the true values ​​and the error test are selected
- Decision Tree
```
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

```

- Random Forest
```
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")
val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
```
- GBT
```
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
```
# Results table
The System.nanoTime function was used to see the execution time of each algorithm.

[Imagen9](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad4/Unidad4/images/9.png)

If we realize almost all three have a similar accuracy starting with the highest which is Decision Tree, then Gradient-Boosted and finally Random Forest.
In this example, the winner would be Decision Tree because it had a slightly better prediction and better execution time compared to the other two algorithms, but that is because we train the models with a certain amount of iterations which in the end point of view the Winner would be random forest because it is almost as accurate as gradient boosted, but due to its bagging technique it does not tend to overfitting as gradient boosted or Decision tree would do that depending on the iterations and the type of data you put are very prone to overfitting.
In decision trees, overfitting occurs when the tree is designed to perfectly fit all samples in the training data set. Therefore, it ends with branches with strict rules of scarce data. Therefore, this affects accuracy when predicting samples that are not part of the training set.

# Conclusions

Due to the large amounts of machine learning algorithms for qualifying learning, there is no specific answer to see which one is better or worse, this because depending on what type of information or project you are going to implement, it could work better for you one or the other, because the algorithms can work better with different types of dataset or worse, in the comparison we have just made, we observe that the differences between the algorithms in terms of predictions and execution times are not very relevant due to We do not handle millions of data but in actual practice these fractions could mean thousands of information or loss prediction that could be useful.
In conclusion you could say that the best algorithm for the classification is the one that works best for you with what you want to do, it may be that gradient-boosted is useful for you since you will train it with the trees predefined by you, or also choose to use Random forest because the bagging technique is less prone to overfitting, or handle a somewhat simple dataset with which decision tree would be more feasible to use.

# References
[1] Anuj Saxena, A. S. (2017, 29 agosto). Machine Learning with Random Forests. Recuperado 9 diciembre, 2019, de https://blog.knoldus.com/machine-learning-with-random-forests/
[2] Afroz Chakure, A. C. (2018, July 5). Decision Tree Classification. Retrieved December 9, 2019, from  https://towardsdatascience.com/decision-tree-classification-de64fc4d5aac
[3] Ying Hu. (2015). Faster Gradient-Boosting Decision Trees. 2016, de Intel Sitio web:
https://techdecoded.intel.io/resources/faster-gradient-boosting-decision-trees/#gs.avgd9k
[4] Bradley Boehmke y Brandon Greenwell. (2019). Hands-On Machine Learning con R. 2019, de bradley boehmke Sitio web
:https://bradleyboehmke.github.io/HOML/gbm.html#basic-gbm
[5] sefiks. (2018). A Step by Step Gradient Boosting Decision Tree Example. 04/2018, de sefiks.com Sitio web
:https://sefiks.com/2018/10/04/a-step-by-step-gradient-boosting-decision-tree-example/
[6] Global Software Support, G. S. (2019, 9 octubre). Random Forest Classifier - Machine Learning | Global Software Support. Recuperado 9 diciembre, 2019, de https://www.globalsoftwaresupport.com/random-forest-classifier/
