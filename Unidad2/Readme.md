**Gradient Boosted Tree Classifer**

#Definition
It is a machine learning technique used for regression analysis and for statistical classification problems, which produces a predictive model in the form of a set of weak prediction models, typically decision trees.
GBT builds trees one at a time, where each new tree helps correct the mistakes made by a previously trained tree.

![Comparativa ramdom forest y GBT](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen1.png)
![Comparativa ramdom forest y GBTs](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen2.png)

Compared to random forest, GBT works with trees sequentially, so the first tree always depends on the previous one.
As in the following picture.
![Comparativa ramdom forest y GBT](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen3.png)

*How Gradient Boosted Trees Classifer works*
Gradient reinforcement models have a very low interpretation capacity because the second tree in the model no longer predicts the same objective as the original model, the later trees in the model seek to predict how far the original predictions of the truth were from use the residues of the previous trees In this way, each subsequent tree of the gradient increase model slowly reduces the general error of the previous trees.
This allows gradient augmentation models to have a very high predictive power but a low capacity for interpretation.

##Advantages##
* Since boosted trees are derived from the optimization of an objective function, basically GBM can be used to solve almost all objective functions that we can write in gradient.
* Performs the optimization in the function space (instead of in the parameter space), which greatly facilitates the use of custom loss functions.
* Predictive power too high.

##Disadvantages##
* GBMs are more sensitive to overfitting if the data is noisy.
* Training usually takes longer due to the fact that trees are built sequentially.
* GBMs are more difficult to tune than RF. There are generally three parameters: number of trees, tree depth and      learning rate, and each tree built is generally shallow.
