import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def f(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-10, 10, 50)

print(x.shape)
print(y.shape)

X, Y = np.meshgrid(x, y)

print(X.shape)
print(Y.shape)

print(X)
print(Y)

Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface(X, Y, Z, cmap="viridis")

plt.show()

fig = plt.figure()
ax = plt.axes(projection="3d")

angle = np.linspace(0, 4 * np.pi, 50)
r = np.linspace(0, 6, 30)

R, Angle = np.meshgrid(r, angle)

X = R * np.sin(Angle)
Y = R * np.cos(Angle)
Z = f(X, Y)

ax.plot_surface(X, Y, Z, cmap="viridis")

plt.show()

fig = plt.figure()
ax = plt.axes(projection="3d")

angle = np.linspace(0, 1.5 * np.pi, 50)
r = np.linspace(0, 6, 30)

R, Angle = np.meshgrid(r, angle)

X = R * np.sin(Angle)
Y = R * np.cos(Angle)
Z = f(X, Y)

# ax.plot_surface(X, Y, Z, cmap="viridis")
ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), cmap="viridis")

plt.show()

import seaborn as sns

sns.set_style("darkgrid")

cars = pd.read_csv("./digital_python-25-26/data/cars.csv")

print(cars.head())


# Число
# sns.histplot(cars["selling_price"])
# sns.displot(cars["selling_price"])
# sns.kdeplot(cars["selling_price"])


# Парная диаграмма

# sns.pairplot(cars)
# sns.pairplot(data=cars, hue="transmission")


# Тепловая карта

cars_corr = cars[["year", "selling_price", "seats", "mileage"]]
sns.heatmap(cars_corr.corr(), cmap="viridis", annot=True)

plt.show()

# sns.scatterplot(x="seats", y="mileage", data=cars)
# sns.scatterplot(x="seats", y="mileage", data=cars, hue="fuel")
# sns.scatterplot(x="year", y="selling_price", data=cars)

plt.show()

# sns.regplot(x="seats", y="mileage", data=cars)
# sns.lmplot(x="seats", y="mileage", data=cars, col="transmission",
#            col_wrap=2, hue="fuel")


# relplot

# sns.relplot(x="seats", y="mileage", data=cars, kind="scatter")
# sns.relplot(x="seats", y="mileage", data=cars, kind="scatter", hue="fuel")
# sns.relplot(
#     x="seats",
#     y="mileage",
#     data=cars,
#     kind="scatter",
#     col="transmission",
#     col_wrap=2,
#     hue="fuel",
# )

# sns.relplot(
#     x="seats",
#     y="mileage",
#     data=cars,
#     kind="line",
#     col="transmission",
#     col_wrap=2,
#     hue="fuel",
# )

sns.relplot(
    x="seats",
    y="mileage",
    data=cars,
    kind="scatter",
    col="transmission",
    col_wrap=2,
    hue="fuel",
)

plt.show()

# sns.lineplot(x="seats", y="mileage", data=cars, hue="fuel")

# sns.jointplot(x="year", y="selling_price", data=cars)
# sns.jointplot(x="year", y="selling_price", data=cars, kind="kde")
# sns.jointplot(x="year", y="selling_price", data=cars, kind="hex")
# sns.jointplot(x="year", y="selling_price", data=cars, hue="transmission")

# sns.barplot(x="fuel", y="selling_price", data=cars, estimator=np.mean)
# sns.barplot(
#     x="fuel",
#     y="selling_price",
#     data=cars,
#     estimator=np.mean,
#     hue="transmission",
# )

# sns.pointplot(
#     x="fuel",
#     y="selling_price",
#     data=cars,
#     estimator=np.mean,
#     hue="transmission",
# )

# sns.boxplot(x="fuel", y="selling_price", data=cars, hue="transmission")
# sns.violinplot(x="fuel", y="selling_price", data=cars, hue="transmission")
# sns.stripplot(x="fuel", y="selling_price", data=cars, hue="transmission")

g = sns.violinplot(
    x="fuel",
    y="selling_price",
    data=cars,
    hue="transmission",
)

sns.stripplot(
    x="fuel",
    y="selling_price",
    data=cars,
    hue="transmission",
)

plt.show()
