import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Простые графики
x = np.linspace(0, 10, 1000)
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.3))

lines = plt.plot(x, y)

# fig, ax = plt.subplots()
# ax.plot(x, np.sin(x), "-k", label="Синус")
# ax.plot(x, np.cos(x), "--r", label="Косинус")
# ax.axis("equal")
# ax.legend(frameon=True, shadow=True, borderpad=1, loc="lower center", ncol=2)

plt.show()

# ----------------------------------------
# Несколько линий + легенда
# ----------------------------------------

y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0, 2, 0.5))

# lines = plt.plot(x, y)
# plt.legend(lines, ["первая", "вторая", "третья", "четвертая"])

# plt.plot(x, y[:, 0], label="первый")
# plt.plot(x, y[:, 1], label="второй")
# plt.plot(x, y[:, 2:])
# plt.legend()

# ----------------------------------------
# Работа с csv и scatter plot
# ----------------------------------------

cities = pd.read_csv("./digital_python-25-26/data/california_cities.csv")

latd = cities["latd"]
longd = cities["longd"]
population_total = cities["population_total"]
area_total_km2 = cities["area_total_km2"]

print(cities.head())

plt.scatter(
    longd,
    latd,
    c=np.log10(population_total),
    s=area_total_km2,
    alpha=0.5
)

plt.colorbar()

plt.scatter([], [], s=100, c="k", alpha=0.5, label="100 $km^2$")
plt.scatter([], [], s=300, c="k", alpha=0.5, label="300 $km^2$")
plt.scatter([], [], s=500, c="k", alpha=0.5, label="500 $km^2$")

plt.legend(frameon=False, labelspacing=2, title="Площадь")

plt.axis("equal")
plt.show()

# ----------------------------------------
# Несколько легенд
# ----------------------------------------

fig, ax = plt.subplots()

lines = ax.plot(
    x,
    np.sin(x[:, np.newaxis] - np.pi / 2 * np.arange(0, 4))
)

ax.axis("equal")

ax.legend(lines[:2], ["line A", "line B"], loc="lower right")

leg = mpl.legend.Legend(
    ax,
    lines[2:],
    ["line C", "line D"],
    loc="upper right"
)

ax.add_artist(leg)

leg2 = mpl.legend.Legend(
    ax,
    lines[2:],
    ["line C", "line D"],
    loc="upper left"
)

# ax.add_artist(leg2)

plt.show()

# ----------------------------------------
# imshow / цветовые карты
# ----------------------------------------

y = np.sin(x) * np.cos(x[:, np.newaxis])

# plt.imshow(y, cmap="jet")
# plt.imshow(y, cmap="viridis")
plt.imshow(y, cmap="RdBu")
plt.colorbar()

plt.show()

# ----------------------------------------
# digits dataset
# ----------------------------------------

from sklearn.datasets import load_digits

digits = load_digits(n_class=6)

fig, ax = plt.subplots(8, 8)

for i, ax_ in enumerate(ax.flat):
    ax_.imshow(digits.images[i], cmap="binary")
    ax_.set(xticks=[], yticks=[])

plt.show()

# ----------------------------------------
# Isomap
# ----------------------------------------

from sklearn.manifold import Isomap

iso = Isomap(n_components=2, n_neighbors=10)
prj = iso.fit_transform(digits.data)

plt.scatter(
    prj[:, 0],
    prj[:, 1],
    c=digits.target,
    cmap=plt.cm.get_cmap("jet", 6)
)

plt.colorbar(ticks=range(6))
plt.clim(-0.5, 5.5)

plt.show()
