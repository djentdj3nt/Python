import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

grid = plt.GridSpec(1, 2)

ax1 = plt.subplot(grid[0, 0])
ax1.set_xscale("log")
ax1.set_xlim(1, 1000)
ax1.grid(True, which="major")

ax2 = plt.subplot(grid[0, 1])
ax2.set_yscale("log")
ax2.set(ylim=(1, 1000))
ax2.grid(True, which="minor", axis="y")

print(ax1.xaxis.get_major_locator())
print(ax1.xaxis.get_major_formatter())
print(ax1.xaxis.get_minor_locator())
print(ax1.xaxis.get_minor_formatter())

print(ax1.yaxis.get_major_locator())
print(ax1.yaxis.get_major_formatter())
print(ax1.yaxis.get_minor_locator())
print(ax1.yaxis.get_minor_formatter())

ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax2.xaxis.set_major_locator(plt.NullLocator())

plt.show()

from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces().images

fig, ax = plt.subplots(7, 7)
fig.subplots_adjust(hspace=0, wspace=0)

for i in range(7):
    for j in range(7):
        ax[i, j].xaxis.set_major_locator(plt.NullLocator())
        ax[i, j].yaxis.set_major_locator(plt.NullLocator())
        ax[i, j].imshow(faces[7 * i + j], cmap="gray")

plt.show()

x = np.linspace(0, 4 * np.pi, 1000)

fig, ax = plt.subplots()

ax.plot(x, np.sin(x), label="Sinus")
ax.plot(x, np.cos(x), label="Cosinus")

ax.legend()
plt.show()

def ff(value, tick_number):
    N = int(np.round(2 * value / np.pi))

    if N == 0:
        return 0
    elif N == 1:
        return r"$\frac{\pi}{2}$"
    elif N == 2:
        return r"$\pi$"
    elif N % 2 == 0:
        t = int(N / 2)
        return f"{t}" + r"$\pi$"
    else:
        return f"{N}" + r"$\frac{\pi}{2}$"

    return value


x = np.linspace(0, 4 * np.pi, 1000)

fig, ax = plt.subplots()

ax.plot(x, np.sin(x), label="Sinus")
ax.plot(x, np.cos(x), label="Cosinus")

ax.grid(True)
ax.legend()
ax.set_xlim(0, 4 * np.pi)

ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))

ax.xaxis.set_major_formatter(plt.FuncFormatter(ff))

plt.show()

fig, ax = plt.subplots(5, 1, figsize=(8, 8))

x = np.linspace(0, 10, 10)

for i in ax.flat:
    i.plot(x, x * 0 + 2)

ax[0].xaxis.set_major_locator(plt.NullLocator())
ax[1].xaxis.set_major_locator(plt.MultipleLocator(0.8))
ax[2].xaxis.set_major_locator(plt.FixedLocator([1, 3, 8, 9]))
ax[3].xaxis.set_major_locator(plt.LinearLocator(numticks=4))
ax[4].xaxis.set_major_locator(plt.IndexLocator(base=1.5, offset=0.3))

plt.show()

x = np.random.randn(1000)

plt.axes(facecolor="#adadad")
plt.hist(x)

plt.show()

from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection="3d")

plt.show()

x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)

print(x.shape)
print(y.shape)

X, Y = np.meshgrid(x, y)

print(X.shape)
print(Y.shape)

print(X)
print(Y)

Z = np.sin(np.sqrt(X ** 2 + Y ** 2))

print(Z.shape)
print(Z)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.contour3D(X, Y, Z, 100)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.view_init(30)

plt.show()
