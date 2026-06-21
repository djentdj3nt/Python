# =============================================================================
# Часть 1. Дерево решений на датасете Iris
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

iris = sns.load_dataset("iris")

print(iris.head())

species_int = []

for row in iris.values:
    match row[4]:
        case "setosa":
            species_int.append(1)

        case "versicolor":
            species_int.append(2)

        case "virginica":
            species_int.append(3)

# species_int_df = pd.DataFrame(species_int)
# print(species_int_df.head())

data = iris[["sepal_length", "petal_length"]]
data["species"] = species_int

print(data.head())

data_df = data[
    (data["species"] == 1) |
    (data["species"] == 2)
]

print(data_df.shape)

data_of_setosa = data[data["species"] == 1]
data_of_versicolor = data[data["species"] == 2]

plt.scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
plt.scatter(data_of_versicolor["sepal_length"], data_of_versicolor["petal_length"])

X = data_df[["sepal_length", "petal_length"]]
y = data_df["species"]

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X, y)

x1_p = np.linspace(
    min(data_df["sepal_length"]),
    max(data_df["sepal_length"]),
    100
)

x2_p = np.linspace(
    min(data_df["petal_length"]),
    max(data_df["petal_length"]),
    100
)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

print(X1_p.shape)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns=["sepal_length", "petal_length"]
)

print(X_p.head())


# =============================================================================
# Часть 2. Влияние глубины дерева решений
# =============================================================================

max_depth = [[1, 2, 3, 4], [5, 6, 7, 8]]

fig, ax = plt.subplots(2, 4, sharex="col", sharey="row")

for i in range(2):
    j = 0

    for md in max_depth[i]:
        model = DecisionTreeClassifier(max_depth=md)
        model.fit(X, y)

        y_p = model.predict(X_p)

        ax[i, j].scatter(
            data_of_setosa["sepal_length"],
            data_of_setosa["petal_length"]
        )

        ax[i, j].scatter(
            data_of_versicolor["sepal_length"],
            data_of_versicolor["petal_length"]
        )

        ax[i, j].contourf(
            X1_p,
            X2_p,
            y_p.reshape(X1_p.shape),
            alpha=0.3,
            levels=[0, 1.5, 2.5, 3.5]
        )

        ax[i, j].set_title(f"max_depth={md}")

        j += 1

plt.show()


# =============================================================================
# Часть 3. Bagging и Random Forest
# =============================================================================

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier

fig, ax = plt.subplots(1, 3, sharex="col", sharey="row")

model1 = DecisionTreeClassifier(max_depth=3)
model1.fit(X, y)

y1_p = model1.predict(X_p)

ax[0].scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
ax[0].scatter(data_of_versicolor["sepal_length"], data_of_versicolor["petal_length"])
ax[0].contourf(
    X1_p,
    X2_p,
    y1_p.reshape(X1_p.shape),
    alpha=0.3,
    levels=[0, 1.5, 2.5, 3.5]
)
ax[0].set_title("DecisionTree")

model2 = DecisionTreeClassifier(max_depth=3)
bagging = BaggingClassifier(
    model2,
    n_estimators=10,
    max_samples=0.6,
    random_state=1
)
bagging.fit(X, y)

y2_p = bagging.predict(X_p)

ax[1].scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
ax[1].scatter(data_of_versicolor["sepal_length"], data_of_versicolor["petal_length"])
ax[1].contourf(
    X1_p,
    X2_p,
    y2_p.reshape(X1_p.shape),
    alpha=0.3,
    levels=[0, 1.5, 2.5, 3.5]
)
ax[1].set_title("Bagging")

model3 = RandomForestClassifier(
    max_depth=3,
    n_estimators=10,
    max_samples=0.6,
    random_state=1
)
model3.fit(X, y)

y3_p = model3.predict(X_p)

ax[2].scatter(data_of_setosa["sepal_length"], data_of_setosa["petal_length"])
ax[2].scatter(data_of_versicolor["sepal_length"], data_of_versicolor["petal_length"])
ax[2].contourf(
    X1_p,
    X2_p,
    y3_p.reshape(X1_p.shape),
    alpha=0.3,
    levels=[0, 1.5, 2.5, 3.5]
)
ax[2].set_title("RandomForest")

plt.show()

# + простые модели + быстро решаются + параллелизм
# + голосование
# + непараметрические - эффективная работа с данные


# =============================================================================
# Часть 4. PCA на Iris Setosa
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

iris = sns.load_dataset("iris")

data = iris[["sepal_length", "petal_length", "species"]]
data_setosa = data[data["species"] == "setosa"]

print(data_setosa.head())

X = data_setosa["sepal_length"]
Y = data_setosa["petal_length"]

data_setosa = data_setosa.drop(columns=["species"])

plt.scatter(X, Y)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data_setosa)

print("1")
print(pca.components_)
print(pca.mean_)
print(pca.explained_variance_)

plt.scatter(X, Y)
plt.scatter(pca.mean_[0], pca.mean_[1])

# 1-я главная компонента
plt.plot(
    [
        pca.mean_[0],
        pca.mean_[0] + pca.components_[0][0] * np.sqrt(pca.explained_variance_[0])
    ],
    [
        pca.mean_[1],
        pca.mean_[1] + pca.components_[0][1] * np.sqrt(pca.explained_variance_[0])
    ]
)

# 2-я главная компонента
plt.plot(
    [
        pca.mean_[0],
        pca.mean_[0] + pca.components_[1][0] * np.sqrt(pca.explained_variance_[1])
    ],
    [
        pca.mean_[1],
        pca.mean_[1] + pca.components_[1][1] * np.sqrt(pca.explained_variance_[1])
    ]
)

pca1 = PCA(n_components=1)
pca1.fit(data_setosa)

X_pca1 = pca1.transform(data_setosa)

print(data_setosa.shape)
print(X_pca1.shape)

X_new = pca1.inverse_transform(X_pca1)

plt.scatter(X_new[:, 0], X_new[:, 1])

plt.show()
