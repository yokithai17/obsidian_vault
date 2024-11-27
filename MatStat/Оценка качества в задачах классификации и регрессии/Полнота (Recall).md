
Полнота — это доля истинно положительных классификаций. Полнота показывает, какую долю объектов, реально относящихся к положительному классу, мы предсказали верно.

$$Recall = \dfrac{TP}{TP+FN}$$

Полнота (_recall_) демонстрирует способность алгоритма обнаруживать данный класс вообще.

Имея матрицу ошибок, очень просто можно вычислить точность и полноту для каждого класса. Точность (_precision_) равняется отношению соответствующего диагонального элемента матрицы и суммы всей строки класса. Полнота (_recall_) — отношению диагонального элемента матрицы и суммы всего столбца класса. Формально:

$$Precision_{c} = \dfrac{A_{c,c}}{\sum_{i=1}^{n}A_{c,i}}$$
$$Recall_{c} = \dfrac{A_{c,c}}{\sum_{i=1}^{n}A_{i,c}}$$
Результирующая точность классификатора рассчитывается как арифметическое среднее его точности по всем классам. То же самое с полнотой. Технически этот подход называется **macro-averaging**.

```python
 # код для для подсчета точности и полноты:
 # Пример классификатора, способного проводить различие между всего лишь двумя**
 # классами, "пятерка" и "не пятерка" из набора рукописных цифр MNIST**
 import numpy as np
 from sklearn.datasets import fetch_openml
 from sklearn.model_selection import cross_val_predict
 from sklearn.metrics import precision_score, recall_score
 from sklearn.linear_model import SGDClassifier
 mnist = fetch_openml('mnist_784', version=1)
 X, y = mnist["data"], mnist["target"]
 y = y.astype(np.uint8)
 X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
 y_train_5 = (y_train == 5) # True для всех пятерок, False для в сех остальных цифр. Задача опознать пятерки
 y_test_5 = (y_test == 5)
 sgd_clf = SGDClassifier(random_state=42) # классификатор на основе метода стохастического градиентного спуска (Stochastic Gradient Descent SGD)
 sgd_clf.fit(X_train, y_train_5) # обучаем классификатор распозновать пятерки на целом обучающем наборе
 y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
 # print(confusion_matrix(y_train_5, y_train_pred))
 # array([[53892, 687]
 #        [ 1891, 3530]])
 print(precision_score(y_train_5, y_train_pred)) # == 3530 / (3530 + 687)
 print(recall_score(y_train_5, y_train_pred)) # == 3530 / (3530 + 1891)
   
 # 0.8370879772350012
 # 0.6511713705958311
```