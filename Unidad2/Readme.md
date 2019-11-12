**Gradient Boosted Tree Classifer**

## Definition
It is a machine learning technique used for regression analysis and for statistical classification problems, which produces a predictive model in the form of a set of weak prediction models, typically decision trees.
GBT builds trees one at a time, where each new tree helps correct the mistakes made by a previously trained tree.

![Comparativa ramdom forest y GBT](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen1.png)
![Comparativa ramdom forest y GBTs](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen2.png)

Compared to random forest, GBT works with trees sequentially, so the first tree always depends on the previous one.
As in the following picture.
![Comparativa ramdom forest y GBT](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen3.png)

## How Gradient Boosted Trees Classifer works
Gradient reinforcement models have a very low interpretation capacity because the second tree in the model no longer predicts the same objective as the original model, the later trees in the model seek to predict how far the original predictions of the truth were from use the residues of the previous trees In this way, each subsequent tree of the gradient increase model slowly reduces the general error of the previous trees.
This allows gradient augmentation models to have a very high predictive power but a low capacity for interpretation.

## Advantages
* Since boosted trees are derived from the optimization of an objective function, basically GBM can be used to solve almost all objective functions that we can write in gradient.
* Performs the optimization in the function space (instead of in the parameter space), which greatly facilitates the use of custom loss functions.
* Predictive power too high.

## Disadvantages
* GBMs are more sensitive to overfitting if the data is noisy.
* Training usually takes longer due to the fact that trees are built sequentially.
* GBMs are more difficult to tune than RF. There are generally three parameters: number of trees, tree depth and      learning rate, and each tree built is generally shallow.

**Example**
- The blue dot plots (left) are input (x) versus output (y)
- The red line (left) shows the values ​​predicted by the decision tree
- The green dots (right) show the residuals in front of the input (x) for the ith iteration
- The iteration represents sequential order of adjustment of the gradient increase tree
![Example](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen5.png)

*The residual versus versus graphs resemble what we see in the twentieth iteration. But the model is becoming more complex and the predictions fit the training data too much and are trying to learn each training data. Therefore, it would have been better to stop at the twentieth iteration.*

![Example](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen6.png)
![Example](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen7.png)

**Hyperparameter**
To solve overfitting or overfitting we can use a Learning Rate, which controls how the following trees are added.
![Example](https://raw.githubusercontent.com/mrsuits210/DatosMasivos/Unidad2/Unidad2/Images/Imagen9.png)

**where we can use gradient boosted Classifer**
Gradient enhancement can be used in the classification learning field. Yahoo and Yandex commercial web search engines use gradient boosting variants in their search engines.
*YAHOO
Our algorithms personalize and classify the content in our search and media products, boost ad selection, detect spam and prevent abuse. These algorithms generate value for advertisers, performance for publishers and productivity for consumers.*

## References
[1]. Ying Hu. (2015). Faster Gradient-Boosting Decision Trees. 2016, de Intel Sitio web:
https://techdecoded.intel.io/resources/faster-gradient-boosting-decision-trees/#gs.avgd9k
[2] Bradley Boehmke y Brandon Greenwell. (2019). Hands-On Machine Learning con R. 2019, de bradleyboehmke Sitio web:
https://bradleyboehmke.github.io/HOML/gbm.html#basic-gbm
[3] sefiks. (2018). A Step by Step Gradient Boosting Decision Tree Example. 04/2018, de sefiks.com Sitio web:
https://sefiks.com/2018/10/04/a-step-by-step-gradient-boosting-decision-tree-example/
[4] Abolfazl Ravanshad. (2018). Gradient Boosting vs Random Forest. 04/18, de medium Sitio web:
https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80
[5] Javii. (2019). Como funciona Gradient Boosting. 04/2019, de spainml Sitio web:
https://spainml.com/blog/como-funciona-gradient-boosting/
