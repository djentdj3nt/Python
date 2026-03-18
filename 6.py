import numpy as np
import pandas as pd

# Отсутствующие данные
a = np.nan
a = -999999

# Pandas: 1) NaN, None 2) pd.NA
a = None
print(type(a))

a = np.array([1, 2, 3])
b = np.array([1, None, 3])
print(a.sum)

c = np.array([1, np.nan, 3])
print(c)
print(c.sum())

print(1 + np.nan)

x = pd.Series([1, 2, 3])
print(x)
x[0] = None
x[1] = np.nan
print(x)

x = pd.Series([1, np, nan, None, 5])
print(x)

x = pd.Series([1, np.nan, None, 5, pd.NA], dtype="Int32")
print(x)

x = pd.DataFrame([[1, np.nan, None], [1, 2, 3], [2, np.nan, 3]])
print(x)

print(x.dropna(axis=0))
print(x.dropna(axis=1))

x = pd.DataFrame([[None, np.nan, None], [1, 2, 3], [2, np.nan, 3]])
print(x)

print(x.dropna(axis=0, thresh=0))

x = pd.DataFrame([[None, np.nan, None], [1, 2, 3], [2, np.nan, 3]])
print(x)

print(x.dropna(axis=0, thresh=1))

# Иерархическая индексация
index = [
    ("A", 2025),
    ("B", 2026),
    ("C", 2025),
    ("D", 2026),
    ("E", 2025),
]

data = [1, 2, 3, 4, 5]

s = pd.Series(data, index=index)
print(s)

print(s[[i for i in index if i[1] == 2025]])

mi = pd.MultiIndex.from_product([["A", "B"], [2025, 2026]])
print(mi)
