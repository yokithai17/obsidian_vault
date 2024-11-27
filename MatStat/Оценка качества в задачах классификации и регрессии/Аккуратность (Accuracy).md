
Интуитивно понятной, очевидной и почти неиспользуемой метрикой является _accuracy_ — доля правильных ответов алгоритма

$$accuracy = \dfrac{TP+TN}{TP+TN+FP+FN}$$

Допустим, мы хотим оценить работу спам-фильтра почты. У нас есть 100 не-спам писем, 90 из которых наш классификатор определил верно (True Negative = 90, False Positive = 10), и 10 спам-писем, 5 из которых классификатор также определил верно (True Positive = 5, False Negative = 5). Тогда accuracy:

$$accuracy = \dfrac{5+90}{5+90+10+5}=86,4$$

Однако если мы просто будем предсказывать все письма как не-спам, то получим более высокую _аккуратность_:

$$accuracy = \dfrac{0+100}{0+100+0+10}=90,9$$
При этом, наша модель совершенно не обладает никакой предсказательной силой, так как изначально мы хотели определять письма со спамом. Преодолеть это нам поможет переход с общей для всех классов метрики к отдельным показателям качества классов.

```python
 # код для для подсчета аккуратности:
 # Пример классификатора, способного проводить различие между всего лишь двумя**
 # классами, "пятерка" и "не пятерка" из набора рукописных цифр MNIST**
 import numpy as np
 from sklearn.datasets import fetch_openml
 from sklearn.model_selection import cross_val_predict
 from sklearn.metrics import accuracy_score
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
 print(accuracy_score(y_train_5, y_train_pred)) # == (53892 + 3530) / (53892 + 3530  + 1891 +687)
 
 # 0.9570333333333333
```
