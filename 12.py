import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime

x = np.linspace(0, 1, 20)

ax1 = plt.axes()
ax1.plot(np.sin(x))

# нижний, левый, ширина, высота
ax2 = plt.axes([0.4, 0.3, 0.2, 0.1])
ax2.plot(np.cos(x))

plt.show()

fig = plt.figure()

ax1 = fig.add_axes([0.1, 0.5, 0.8, 0.4])
ax2 = fig.add_axes([0.1, 0.3, 0.2, 0.1])

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.show()

fig, ax = plt.subplots(2, 3)

x1 = np.linspace(0, 10, 50)
x2 = np.linspace(0, 20, 50)

for i in range(2):
    for j in range(3):
        ax[i, j].plot(np.sin(x1 + np.pi / 4 * (i * 2 + j)))

plt.show()

grid = plt.GridSpec(2, 3)

# 0 1 2
# 0 X Y Y
# 1 Z Z K

plt.subplot(grid[0, 0])
plt.plot(np.sin(x1))

plt.subplot(grid[0, 1:])
plt.plot(np.cos(x1))

plt.subplot(grid[1, :2])
plt.plot(np.tan(x))

plt.subplot(grid[1, 2])
plt.plot(x)

plt.show()

grid = plt.GridSpec(2, 3)

# 0 1 2
# 0 X Y K
# 1 Z Z K

plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1])
plt.subplot(grid[:, 2])
plt.subplot(grid[1, :2])

plt.show()

grid = plt.GridSpec(4, 4, wspace=0.2, hspace=0.2)

# Z X X X
# Z X X X
# Z X X X
#   Y Y Y

rng = np.random.default_rng(1)

x, y = rng.multivariate_normal(
    [0, 0],
    [[1, 2], [3, 4]],
    1000
).T

main_axes = plt.subplot(grid[:-1, 1:], yticklabels=[])
y_axes = plt.subplot(grid[:-1, 0])
x_axes = plt.subplot(grid[-1, 1:])

main_axes.plot(x, y, "ok", alpha=0.2)

y_axes.hist(y, 40, orientation="horizontal", color="grey")
y_axes.invert_xaxis()

x_axes.hist(x, 40, color="grey")
x_axes.invert_yaxis()

plt.show()

births = pd.read_csv("./digital_python-25-26/data/births-1969.csv")

births["day"] = births["day"].astype(int)

births.index = pd.to_datetime(
    births["year"] * 10000 +
    births["month"] * 100 +
    births["day"],
    format="%Y%m%d"
)

print(births)

births_dom = births.pivot_table(
    "births",
    index=[births.month, births.day]
)

births_dom.index = [
    datetime(1969, month, day)
    for (month, day) in births_dom.index
]

fig, ax = plt.subplots()

births_dom.plot(ax=ax)

ax.text("1969-1-1", 5500, "Новый год")

ax.xaxis.set_major_locator(
    mpl.dates.MonthLocator(bymonthday=15)
)

ax.xaxis.set_major_formatter(
    mpl.dates.DateFormatter("%h")
)

ax.annotate(
    "Текст аннотации",
    xy=("1969-1-1", 5500),
    xytext=("1969-12-1", 4500),
    arrowprops=dict(facecolor="black")
)

plt.show()
