# Scikit-Learn

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

iris = sns.load_dataset("iris")
print(iris.head())

# print(type(iris))

# print(type(iris.values))

# print(iris.values.shape)

# print(iris.columns)

# print(iris.index)

# sns.pairplot(iris, hue="species")

# plt.show()

# Строки - образцы - отдельный объект (sample)
# Столбцы - признаки (feature)
# Метки - принадлежность к классу


# 5.1. Обучение с учителем - predict()

from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(1)
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)

model = LinearRegression(fit_intercept=True)
reg = model.fit(x[:, np.newaxis], y)

xfit = np.linspace(x.min(), x.max(), 1000)
# yfit = model.predict(xfit[:, None])

# plt.plot(xfit, yfit, "r")

# plt.plot(xfit, xfit * reg.coef_ + reg.intercept_, "k")

# y = kx + b

# plt.show()


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

model = make_pipeline(PolynomialFeatures(7), LinearRegression())
reg = model.fit(x[:, np.newaxis], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict(xfit[:, None])

plt.scatter(x, y)
plt.plot(xfit, yfit, "r")

plt.show()

# exit()


# Классификация. Логистическая регрессия

from sklearn.linear_model import LogisticRegression

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()

x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

x_00 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
x_11 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()

x = iris[iris["species"] != "virginica"].iloc[:, 0].to_numpy()

y1 = np.full(50, 1)
y2 = np.full(50, 2)

y = np.ravel([y1, y2])
print(y)

model = LogisticRegression()
model.fit(x[:, None], y)

xfit = np.linspace(x.min(), x.max(), 1000)
yfit = model.predict_proba(xfit[:, None])

# print(yfit)

plt.plot(xfit, 1 + 4 * yfit[:, 1], "green")
plt.plot(xfit, 1 + 4 * yfit[:, 0], "red")

plt.show()

# exit()


# Деревья решений

from sklearn.tree import DecisionTreeClassifier

x = iris[iris["species"] != "virginica"].iloc[:, 0:2].to_numpy()
y = iris[iris["species"] != "virginica"].iloc[:, 4]

y1 = np.full(50, 1)
y2 = np.full(50, 2)

y = np.ravel([y1, y2])

model = DecisionTreeClassifier()
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()
x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

plt.show()


# Метод опорных векторов. Классификация

from sklearn.svm import SVC

x = iris[iris["species"] != "virginica"].iloc[:, 0:2].to_numpy()
y = iris[iris["species"] != "virginica"].iloc[:, 4]

# print(x)
# print(y)

y1 = np.full(50, 1)
y2 = np.full(50, 2)
print(y1)
print(type(y1))

y = np.ravel([y1, y2])
# print(x)
# print(y)

model = SVC(kernel="linear")
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

x_0 = iris[iris["species"] == "setosa"].iloc[:, 0].to_numpy()
y_0 = iris[iris["species"] == "setosa"].iloc[:, 1].to_numpy()
x_1 = iris[iris["species"] == "versicolor"].iloc[:, 0].to_numpy()
y_1 = iris[iris["species"] == "versicolor"].iloc[:, 1].to_numpy()

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

plt.show()


# Наивная байесовская классификация

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

x_m = model.theta_[0]
x_var = model.var_[0]
y_m = model.theta_[1]
y_var = model.var_[1]

z1 = 1 / (2 * np.pi * (x_var[0] * x_var[1]) ** 0.5) * np.exp(
    - ((xx - x_m[0])**2) / (2 * x_var[0])
    - ((yy - x_m[1])**2) / (2 * x_var[1])
)

ax.contour(xx, yy, z1)

z2 = 1 / (2 * np.pi * (y_var[0] * y_var[1]) ** 0.5) * np.exp(
    - ((xx - y_m[0])**2) / (2 * y_var[0])
    - ((yy - y_m[1])**2) / (2 * y_var[1])
)

ax.contour(xx, yy, z2)

plt.show()

ax = plt.axes(projection="3d")
ax.contour3D(xx, yy, z1, 50)

ax.contour3D(xx, yy, z2, 50)

plt.show()


# k - ближ. соседей

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x, y)

xx, yy = np.meshgrid(
    np.linspace(x[:, 0].min(), x[:, 0].max(), 100),
    np.linspace(x[:, 1].min(), x[:, 1].max(), 100),
)

plt.scatter(x_0, y_0, color="red", alpha=0.5)
plt.scatter(x_1, y_1, color="green", alpha=0.5)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax = plt.gca()

ax.contourf(xx, yy, Z, alpha=0.3, levels=[0, 1.5, 3])

plt.show()
